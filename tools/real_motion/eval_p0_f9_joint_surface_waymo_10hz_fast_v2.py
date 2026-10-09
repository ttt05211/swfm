#!/usr/bin/env python3
"""Second exact 10Hz execution, V1/V2 paired checks and read-only continuation."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contextlib import nullcontext
import json
import os
import signal
import threading
import time

import numpy as np
import torch

from real_motion.waymo_i2world import SHAPE, UPSTREAM_URL, file_sha256, fingerprint
from real_motion.waymo_i2world_10hz import PROTOCOL, REPORT_KEYS, REPORT_SECONDS, WaymoI2World10HzSource, format_10hz_reports
from real_motion.waymo_geometry_execution import GeometryPrefetchSource
from real_motion.waymo_geometry_execution_v2 import PROTOCOL as EXECUTION_PROTOCOL
from real_motion.waymo_native_execution import prepare_waymo_native
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution
from tools.real_motion import eval_p0_f9_joint_surface_waymo_10hz as original
from tools.real_motion.joint_surface_checkpoint_selection import AVERAGE_EPOCHS, load_evaluation_model
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_surface_long_rollout_common import verify_first_block
from tools.real_motion.waymo_zero_shot_common import evaluate_windows, write_json
from tools.real_motion.waymo_fast_execution import read_continuation, migrate_state
from tools.real_motion.waymo_fast_execution_v2 import (FastV2WaymoProvider as FastWaymoSurfaceProvider,
    FastV2SurfaceExecution as FastSurfaceBlockExecution, paired_speed_v2 as paired_speed)
from tools.real_motion.eval_p0_f9_joint_surface_waymo_10hz_fast import IMPLEMENTATION_FILES as V1_FILES

IMPLEMENTATION_FILES = (*V1_FILES, 'real_motion/native/waymo_execution.cpp',
    'real_motion/waymo_native_execution.py', 'real_motion/waymo_geometry_execution_v2.py',
    'tools/real_motion/waymo_fast_execution_v2.py', 'tools/real_motion/eval_p0_f9_joint_surface_waymo_10hz_fast_v2.py')


def parser():
    p = original.parser(); p.description = __doc__; p.set_defaults(cpu_workers=4)
    p.add_argument('--geometry-cache-mib', type=int, default=1024)
    p.add_argument('--surface-chunk', type=int, default=4096)
    p.add_argument('--no-history-prefetch', action='store_true')
    p.add_argument('--continue-from-dir', help='read-only stopped original or fast-v1 10Hz output; new --out-dir required')
    p.add_argument('--speed-windows', type=int, default=16, help='paired correctness/timing before new evaluation; 0 skips timing only')
    p.add_argument('--speed-only', action='store_true', help='no GT/metrics/saved evaluation counts; requires speed-windows>0')
    return p


def main(stop_event=None, argv=None):
    p = parser(); a = p.parse_args(argv)
    if (not 1 <= a.cpu_workers <= 8 or not 0 <= a.frame_cache_mib <= 4096
            or not 0 <= a.geometry_cache_mib <= 4096 or not 256 <= a.surface_chunk <= 65536
            or a.checkpoint_every < 1 or a.max_windows < 0 or a.expected_scenes < 0
            or not 0 <= a.speed_windows <= 128 or a.execution != 'native_parallel'):
        p.error('invalid bounded settings; fast execution requires native_parallel')
    if (a.resume and (a.audit_only or a.continue_from_dir or a.speed_only)) or (a.speed_only and not a.speed_windows):
        p.error('resume/audit/continuation/speed-only combination is invalid')
    if a.audit_only and (a.continue_from_dir or a.speed_only): p.error('audit-only cannot continue/evaluate')
    out, data_root = Path(a.out_dir).resolve(), Path(a.waymo_root).resolve()
    if out.is_relative_to(data_root) or any((d/'training.json').is_file() for d in (out,*out.parents)):
        p.error('output cannot be inside data/original training directories')
    if a.resume:
        if not (out/'contract.json').is_file() or not (out/'state.json').is_file(): p.error('resume needs SAME fast output')
    elif out.exists(): p.error('new output required; never overwrite an existing experiment')
    previous = json.loads((out/'contract.json').read_text(encoding='utf-8')) if a.resume else None
    old_contract = old_state = receipt = None
    if a.continue_from_dir:
        old_dir = Path(a.continue_from_dir).resolve()
        if out.is_relative_to(old_dir) or old_dir.is_relative_to(out): p.error('distinct nonnested continuation output required')
        old_contract, old_state, receipt = read_continuation(old_dir)
        if old_contract.get('protocol') != PROTOCOL: p.error('source must be SAME native-index 10Hz, not 2Hz')
    source = WaymoI2World10HzSource.from_files(data_root, info_file=a.info_file, pose_file=a.pose_file,
        raw_free_label=a.raw_free_label, cache_mib=a.frame_cache_mib)
    if a.expected_scenes and source.metadata['scenes'] != a.expected_scenes: p.error('official scene count changed')
    if a.max_windows > len(source.windows): p.error('max-windows exceeds anchors')
    selected = source.windows[:a.max_windows or len(source.windows)]; inventory = source.preflight(selected)
    if a.audit_only:
        source.prediction_inputs(selected[0]); out.mkdir(parents=True)
        write_json(out/'audit.json', dict(status='audit_only', data=source.metadata, inventory=inventory,
            windows=len(selected), future_GT_loaded=False, model_loaded=False))
        return 0
    if not torch.cuda.is_available(): p.error('actual CUDA required for model evaluation; audit-only does not require CUDA')
    pcfg = make_prepare_config(load_runtime_config(a.config))
    if (tuple(pcfg.grid.shape_hwd) != SHAPE or not np.allclose(pcfg.grid.voxel_size, (.4,)*3, rtol=0, atol=1e-12)
            or not np.allclose((pcfg.grid.x_min,pcfg.grid.y_min,pcfg.grid.z_min), (-40,-40,-1), rtol=0, atol=1e-12)
            or pcfg.future_frames != 6 or pcfg.free_label != 17 or pcfg.frame_dt_s != .5):
        p.error('unchanged trained six-slot 0.5s / 0.4m predictor required; never retime')
    recorded_contract = old_contract or previous
    checkpoint = (Path(recorded_contract['checkpoint']).resolve() if recorded_contract and not a.checkpoint
                  else original.resolve_checkpoint(a))
    digest = file_sha256(checkpoint)
    if out.is_relative_to(checkpoint.parent): p.error('output cannot be inside frozen mean directory')
    saved_model, joint = load_evaluation_model(checkpoint, device='cuda', z_bins=16)
    if (saved_model.get('source_epochs') != list(AVERAGE_EPOCHS) or not saved_model.get('averaging')
            or joint.transport.config.history_frames != 4 or file_sha256(checkpoint) != digest):
        raise RuntimeError('unchanged FOUR-history frozen mean required')
    native = prepare_waymo_native()
    print('WAYMO10_V2_NATIVE '+json.dumps(native.manifest),flush=True)
    root = Path(__file__).resolve().parents[2]
    contract = dict(protocol=PROTOCOL, upstream=UPSTREAM_URL, windows=len(selected),
        population='all_official_10hz_native_anchors' if not a.max_windows else 'DIAGNOSTIC_PREFIX_NOT_FULL',
        checkpoint=str(checkpoint), checkpoint_sha256=digest, source_epochs=list(AVERAGE_EPOCHS),
        data=source.metadata, data_root=str(data_root), inventory=inventory, manifest_fingerprint=source.manifest_fingerprint,
        thresholds=[.5,None], frame_cache_mib=a.frame_cache_mib, cpu_workers=a.cpu_workers,
        execution=a.execution, graphs=not a.no_graphs, parallel_majority=a.parallel_majority,
        runtime_environment={k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')},
        torch_version=str(torch.__version__), config_sha256=file_sha256(a.config),
        implementation={f:file_sha256(root/f) for f in IMPLEMENTATION_FILES},
        input_budget_note='ours FOUR total incl t0, not identical to upstream temporal tokenizer previous/current budget',
        model_slot_clock='unchanged trained 0.5s slots; no interpolation/scaling/retraining; dataset native 0.1s steps',
        reporting=dict(upstream_eval_metric='miou',eval_time=[1,3,5],nominal_seconds=list(REPORT_SECONDS),
                       keys=list(REPORT_KEYS),single_pass=True),
        future_ego_poses='metadata conditioning, not predicted', Waymo_adaptation='label mapping only; no learned adaptation',
        fast_execution=dict(protocol=EXECUTION_PROTOCOL, geometry_cache_mib=a.geometry_cache_mib,
            surface_chunk=a.surface_chunk, next_history_prefetch=not a.no_history_prefetch,
            persistent_geometry_writes=False, cached_learned_state=False, tube_forward_winners='stable_exact_native',
            surface_fit='same_16_neighbours_native_fp64_reductions', registration_reuse='immutable_source_sort_and_ephemeral_current_tree',
            native_fingerprint=native.manifest['fingerprint']))
    if receipt: contract['execution_migration'] = receipt
    elif previous and 'execution_migration' in previous: contract['execution_migration'] = previous['execution_migration']
    resume = None
    if a.resume:
        if fingerprint(previous) != fingerprint(contract): raise RuntimeError('fast resume contract changed')
    elif old_state is not None:
        resume = migrate_state(old_contract, old_state, contract, shape=source.shape)
        if resume['completed_windows'] == len(selected): p.error('source evaluation is already complete')
    if not a.resume: out.mkdir(parents=True)
    with evaluation_lock(out):
        if a.resume:
            if (out/'waymo_validation.json').is_file():
                done = json.loads((out/'waymo_validation.json').read_text(encoding='utf-8'))
                if done.get('status') == 'complete': print('Already complete: '+str(out/'summary.txt'),flush=True); return 0
            resume = json.loads((out/'state.json').read_text(encoding='utf-8'))
        else:
            write_json(out/'contract.json', contract)
            if old_state is not None:
                write_json(out/'source_contract_snapshot.json', old_contract)
                write_json(out/'source_state_snapshot.json', old_state)
                write_json(out/'state.json', resume)
        torch.set_num_threads(1); started = time.perf_counter()
        cursor = resume['completed_windows'] if resume else 0
        if not a.resume and a.speed_windows:
            speed = paired_speed(source, selected[cursor:cursor+a.speed_windows], joint, pcfg, 'cuda',
                workers=a.cpu_workers, graphs=not a.no_graphs, parallel_majority=a.parallel_majority,
                surface_chunk=a.surface_chunk, geometry_mib=a.geometry_cache_mib, stop_event=stop_event)
            if file_sha256(checkpoint) != digest: raise RuntimeError('frozen source weights changed during speed check')
            write_json(out/'speed.json', speed)
            print(f"WAYMO10_V2_PAIRED_SPEED v1={speed['seconds_per_window']['fast_v1']:.4f}s/window "
                  f"v2={speed['seconds_per_window']['fast_v2']:.4f}s/window speedup={speed['speedup']:.3f} "
                  f"SIX/probability_bytes=PASS windows={speed['windows']}",flush=True)
            if a.speed_only: return 0
            if speed['speedup'] <= 1.0:
                print('V2 not faster on paired population; full evaluation NOT started. Original prefix preserved.',flush=True)
                return 0
        provider = FastWaymoSurfaceProvider(joint, pcfg, 'cuda', a.cpu_workers, geometry_mib=a.geometry_cache_mib)
        execution = FastSurfaceBlockExecution(provider, mode=a.execution, workers=a.cpu_workers,
            query_workers=a.cpu_workers, graphs=not a.no_graphs, surface_chunk=a.surface_chunk)
        streaming = GeometryPrefetchSource(source, selected, provider.geometry, enabled=not a.no_history_prefetch)
        majority = ParallelNativeMajority(min(4,a.cpu_workers)) if a.parallel_majority else None
        @torch.no_grad()
        def predict(record, raw, *, verify):
            tick = time.perf_counter()
            with strong_majority_execution(majority) if majority is not None else nullcontext():
                prep = provider.prepare_columns(None, record, include_gt=False, raw_window=raw)
            stages = dict(history_and_transport_prepare=time.perf_counter()-tick)
            stages.update({'prepare.'+k:v for k,v in provider.fast_prepare_stages.items()})
            dense, edits, timing, scores = execution.predict(prep); stages.update(timing)
            if verify: verify_first_block(provider,prep.state['rec'],prep,dense,scores,execution)
            return prep.baseline,dense,edits,stages
        print(f'WAYMO10_V2 start={cursor}/{len(selected)}; single-frame RAM geometry only; '
              'same weights/indices/all SIX outputs; no disk cache',flush=True)
        try:
            with (out/'progress.jsonl').open('a',encoding='utf-8') as handle:
                def progress(row):
                    row['geometry_cache'] = provider.geometry.stats()
                    row['history_geometry_ready_wait_seconds'] = streaming.last_wait_seconds
                    row['next_history_submit_seconds'] = streaming.last_submit_seconds
                    row['timing_note'] = 'history_io includes frame-geometry ready wait; worker seconds overlap; NOT pure disk I/O'
                    handle.write(json.dumps(row,allow_nan=False)+'\n'); handle.flush()
                    if row['window']%32 == 0 or row['window'] == len(selected):
                        print(f"WAYMO10_V2 {row['window']}/{row['windows']} seconds={row['seconds']:.3f} "
                              f"geometry_hits={row['geometry_cache']['hits']}",flush=True)
                result = evaluate_windows(streaming,selected,predict,contract,saved=resume,
                    save=lambda state:write_json(out/'state.json',state),progress=progress,
                    stop_event=stop_event,checkpoint_every=a.checkpoint_every)
        finally:
            streaming.close(); execution.close(); provider.close()
            if majority is not None: majority.close()
        if file_sha256(checkpoint) != digest: raise RuntimeError('frozen source weights changed')
        result.update(reports=format_10hz_reports(result['reports']),contract=contract,
            elapsed_seconds_this_invocation=time.perf_counter()-started,
            reused_prefix_windows=cursor, new_windows_this_invocation=result['completed_windows']-cursor,
            frame_io=dict(reads=source.io_reads,hits=source.cache_hits,RAM_bytes=source.cache_bytes),
            geometry_cache=provider.geometry.stats(),
            timing_note='saved cumulative timing may include original prefix; new fast window times in progress.jsonl; NOT formal FPS')
        write_json(out/'waymo_validation.json',result)
        summary = original.text_report(result)+f"FAST: reused_prefix_windows={cursor}; geometry={provider.geometry.stats()}\n"
        (out/'summary.txt').write_text(summary,encoding='utf-8'); print(summary,flush=True)
        print('RESULT: '+str(out/'summary.txt'),flush=True)
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM): signal.signal(signum,lambda *args:event.set())
    sys.exit(main(event))
