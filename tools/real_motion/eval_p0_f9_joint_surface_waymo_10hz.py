#!/usr/bin/env python3
"""Frozen mean, literal I2-World load_interval=1 / eval_time=1,3,5 (NOT 1/2/3s)."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution
from tools.real_motion.eval_p0_f9_joint_surface_waymo import (IMPLEMENTATION_FILES as SHARED_FILES,
    WaymoSurfaceProvider, parser as shared_parser, resolve_checkpoint)
from tools.real_motion.joint_surface_checkpoint_selection import AVERAGE_EPOCHS, load_evaluation_model
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_surface_long_rollout_common import SurfaceBlockExecution, verify_first_block
from tools.real_motion.waymo_zero_shot_common import evaluate_windows, write_json

IMPLEMENTATION_FILES = (*SHARED_FILES, 'real_motion/waymo_i2world_10hz.py',
                       'tools/real_motion/eval_p0_f9_joint_surface_waymo_10hz.py')


def parser():
    p = shared_parser(); p.description = __doc__
    p.set_defaults(cpu_workers=2)  # conservative concurrent same-card default
    p._option_string_actions['--parallel-majority'].help = (
        'existing exact Strong majority, min(4, cpu-workers) threads; no forecast cache')
    return p


def text_report(result):
    lines = ['===== SURFACE CCR / I2-WORLD WAYMO 10HZ INDEX ZERO-SHOT =====',
        'status='+result['status'], f"windows={result['completed_windows']}/{result['contract']['windows']}",
        'load_interval=1; official eval_metric=miou; eval_time=1/3/5 (zero-based future indices).',
        'Native +2/+4/+6 frames = nominal 0.2/0.4/0.6s, NOT physical 1/2/3s.',
        'ONE six-frame probability pass; three independent horizon counts, instead of three model reruns.',
        'Frozen FOUR-total-history mean; trained slot embeddings/clock retained, NO temporal rescaling/training/tuning.',
        'branch/horizon                IoU   I2_mIoU standard_mIoU']
    for branch, report in result['reports'].items():
        for key, row in [('average', report['average']), *report['horizons'].items()]:
            scores = ['NA' if row[k] is None else f'{row[k]:.6f}'
                      for k in ('IoU', 'i2world_mIoU', 'standard_mIoU')]
            lines.append(f'{branch+"/"+key:28s} '+ ' '.join(f'{s:>10}' for s in scores))
    lines += ['Actual target timestamps: '+json.dumps(result['contract']['data']['actual_report_dt_s_including_padded_targets']),
        'Timestamp gap audit: '+json.dumps(result['contract']['data']['timestamp_gap_audit']),
        'Scene boundaries repeat nearest valid history/future frames, poses and targets; all anchors retained.',
        'I2_mIoU excludes exactly-zero class IoU; standard includes valid zero classes; no visibility mask.',
        'Future poses are supplied conditioning; future GT occupancy read only AFTER SIX predictions.',
        'This is literal public index-protocol reproduction, NOT a physically matched 1/2/3s comparison.',
        'Wall time includes history I/O/GT/metrics; NOT formal FPS. Concurrent jobs may contend for CPU/GPU.',
        'Original 2Hz code/contracts/results, weights and nuScenes caches unchanged; no automatic promotion.',
        'details=waymo_validation.json; integer recovery=state.json']
    return '\n'.join(lines)+'\n'


def main(stop_event=None, argv=None):
    p = parser(); a = p.parse_args(argv)
    if (not 1 <= a.cpu_workers <= 8 or not 0 <= a.frame_cache_mib <= 4096
            or a.checkpoint_every < 1 or a.max_windows < 0 or a.expected_scenes < 0):
        p.error('invalid bounded execution/population settings')
    if a.audit_only and a.resume:
        p.error('audit-only is not an evaluation resume')
    out, data_root = Path(a.out_dir).resolve(), Path(a.waymo_root).resolve()
    if out.is_relative_to(data_root) or any((d/'training.json').is_file() for d in (out, *out.parents)):
        p.error('output cannot be inside data/original training directories')
    if a.resume:
        if not (out/'contract.json').is_file() or not (out/'state.json').is_file():
            p.error('resume requires the SAME existing 10Hz evaluation output')
    elif out.exists():
        p.error('new output required; never overwrite an existing experiment')
    source = WaymoI2World10HzSource.from_files(data_root, info_file=a.info_file, pose_file=a.pose_file,
        raw_free_label=a.raw_free_label, cache_mib=a.frame_cache_mib)
    if a.expected_scenes and source.metadata['scenes'] != a.expected_scenes:
        p.error(f"Waymo scene count {source.metadata['scenes']} != {a.expected_scenes}; check official metadata")
    if a.max_windows > len(source.windows):
        p.error('--max-windows exceeds available anchors')
    selected = source.windows[:a.max_windows or len(source.windows)]
    inventory = source.preflight(selected)
    print(f'WAYMO 10HZ INDEX: anchors={len(selected)}; eval_time=1/3/5 -> native +2/+4/+6, '
          'nominal 0.2/0.4/0.6s; all original clocks/weights retained.', flush=True)
    if a.audit_only:
        source.prediction_inputs(selected[0])
        out.mkdir(parents=True)
        audit = dict(status='audit_only', data=source.metadata, inventory=inventory,
            manifest_fingerprint=source.manifest_fingerprint, windows=len(selected),
            future_GT_loaded=False, model_loaded=False, upstream=UPSTREAM_URL)
        write_json(out/'audit.json', audit); print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        return 0
    if not torch.cuda.is_available():
        p.error('real CUDA required for evaluation; --audit-only does not require CUDA')
    pcfg = make_prepare_config(load_runtime_config(a.config))
    if (tuple(pcfg.grid.shape_hwd) != SHAPE or not np.allclose(pcfg.grid.voxel_size, (.4,)*3, rtol=0, atol=1e-12)
            or not np.allclose((pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min), (-40, -40, -1), rtol=0, atol=1e-12)
            or pcfg.future_frames != 6 or pcfg.free_label != 17 or pcfg.frame_dt_s != .5):
        p.error('literal slot protocol requires unchanged 0.4m / trained six-slot 0.5s predictor; do not silently retime')
    checkpoint = resolve_checkpoint(a); digest = file_sha256(checkpoint)
    if out.is_relative_to(checkpoint.parent):
        p.error('output cannot be inside the source mean directory')
    saved, joint = load_evaluation_model(checkpoint, device='cuda', z_bins=16)
    if (saved.get('source_epochs') != list(AVERAGE_EPOCHS) or not saved.get('averaging')
            or joint.transport.config.history_frames != 4 or file_sha256(checkpoint) != digest):
        raise RuntimeError('frozen 5/6/8/12/14 FOUR-history mean required')
    root = Path(__file__).resolve().parents[2]
    contract = dict(protocol=PROTOCOL, upstream=UPSTREAM_URL, windows=len(selected),
        population='all_official_10hz_native_anchors' if not a.max_windows else 'DIAGNOSTIC_PREFIX_NOT_FULL',
        checkpoint=str(checkpoint), checkpoint_sha256=digest, source_epochs=list(AVERAGE_EPOCHS),
        data=source.metadata, data_root=str(data_root), inventory=inventory, manifest_fingerprint=source.manifest_fingerprint,
        thresholds=[.5, None], frame_cache_mib=a.frame_cache_mib, cpu_workers=a.cpu_workers,
        execution=a.execution, graphs=not a.no_graphs, parallel_majority=a.parallel_majority,
        runtime_environment={k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')},
        torch_version=str(torch.__version__), config_sha256=file_sha256(a.config),
        implementation={f:file_sha256(root/f) for f in IMPLEMENTATION_FILES},
        input_budget_note='ours FOUR total incl t0, not identical to upstream temporal tokenizer previous/current budget',
        model_slot_clock='unchanged trained 0.5s slots; no interpolation/scaling/retraining; dataset native 0.1s steps',
        reporting=dict(upstream_eval_metric='miou', eval_time=[1,3,5], nominal_seconds=list(REPORT_SECONDS),
                       keys=list(REPORT_KEYS), single_pass=True),
        future_ego_poses='metadata conditioning, not predicted', Waymo_adaptation='label mapping only; no learned adaptation')
    if not a.resume:
        out.mkdir(parents=True)
    with evaluation_lock(out):
        if a.resume:
            previous = json.loads((out/'contract.json').read_text(encoding='utf-8'))
            if fingerprint(previous) != fingerprint(contract):
                raise RuntimeError('10Hz resume population/weights/data/config/execution/implementation changed; never mix 2Hz')
            if (out/'waymo_validation.json').is_file():
                completed = json.loads((out/'waymo_validation.json').read_text(encoding='utf-8'))
                if completed.get('status') == 'complete':
                    print('10Hz already complete: '+str(out/'summary.txt'), flush=True); return 0
            resume = json.loads((out/'state.json').read_text(encoding='utf-8'))
        else:
            write_json(out/'contract.json', contract); resume = None
        torch.set_num_threads(1)
        provider = WaymoSurfaceProvider(joint, pcfg, 'cuda', a.cpu_workers)
        execution = SurfaceBlockExecution(provider, mode=a.execution, workers=a.cpu_workers,
                                         query_workers=a.cpu_workers, graphs=not a.no_graphs)
        majority = ParallelNativeMajority(workers=min(4, a.cpu_workers)) if a.parallel_majority else None
        started = time.perf_counter()
        @torch.no_grad()
        def predict(record, raw, *, verify):
            tick = time.perf_counter()
            scope = strong_majority_execution(majority) if majority is not None else nullcontext()
            with scope:
                prep = provider.prepare_columns(None, record, include_gt=False, raw_window=raw)
            stage = dict(history_and_transport_prepare=time.perf_counter()-tick)
            dense, edits, readout, probability = execution.predict(prep); stage.update(readout)
            if verify:
                verify_first_block(provider, prep.state['rec'], prep, dense, probability, execution)
            return prep.baseline, dense, edits, stage
        with (out/'progress.jsonl').open('a', encoding='utf-8') as handle:
            def progress(row):
                handle.write(json.dumps(row, allow_nan=False)+'\n'); handle.flush()
                if row['window'] % 32 == 0 or row['window'] == len(selected):
                    print(f"WAYMO_10HZ_ZERO_SHOT {row['window']}/{row['windows']} seconds={row['seconds']:.3f}", flush=True)
            try:
                result = evaluate_windows(source, selected, predict, contract, saved=resume,
                    save=lambda state: write_json(out/'state.json', state), progress=progress,
                    stop_event=stop_event, checkpoint_every=a.checkpoint_every)
            finally:
                execution.close()
                if majority is not None:
                    majority.close()
        if file_sha256(checkpoint) != digest:
            raise RuntimeError('source frozen mean changed during 10Hz evaluation')
        result.update(reports=format_10hz_reports(result['reports']), contract=contract,
            elapsed_seconds_this_invocation=time.perf_counter()-started,
            frame_io=dict(reads=source.io_reads, hits=source.cache_hits, RAM_bytes=source.cache_bytes))
        write_json(out/'waymo_validation.json', result)
        (out/'summary.txt').write_text(text_report(result), encoding='utf-8')
        print(text_report(result), flush=True); print('RESULT: '+str(out/'summary.txt'), flush=True)
        return 0


if __name__ == '__main__':
    event = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *args: event.set())
    sys.exit(main(event))
