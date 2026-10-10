#!/usr/bin/env python3
"""Opt-in lossless STC history sharing; explicit read-only prefix migration."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contextlib import ExitStack
import json
import os
import signal
import threading
import time

import numpy as np
import torch

from tools.real_motion import eval_p0_f9_joint_surface_stc as original
from tools.real_motion.stc_shared_execution import Predictor, check_and_time, migrate_state
from tools.real_motion.stc_camera_evaluation import evaluate, restore
from tools.real_motion.waymo_zero_shot_common import write_json
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from real_motion.waymo_i2world import file_sha256, fingerprint

EXTRA_FILES = (
    'tools/real_motion/stc_shared_execution.py', 'tools/real_motion/eval_p0_f9_joint_surface_stc_shared.py',
    'tools/real_motion/run_p0_f9_joint_surface_stc_shared.sh',
    'tools/real_motion/waymo_fast_execution.py', 'tools/real_motion/waymo_fast_execution_v2.py',
    'real_motion/waymo_geometry_execution.py', 'real_motion/waymo_geometry_execution_v2.py',
    'real_motion/waymo_native_execution.py', 'real_motion/native/waymo_execution.cpp',
    'real_motion/source_evidence_audit.py',
)


def read_receipt(directory):
    directory = Path(directory).resolve()
    paths = [directory/'contract.json', directory/'state.json']
    sha = [file_sha256(p) for p in paths]
    values = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    if sha != [file_sha256(p) for p in paths]: raise RuntimeError('source state changed during read')
    receipt = dict(directory=str(directory), contract_sha256=sha[0], state_sha256=sha[1],
                   completed_windows=values[1]['completed_windows'])
    return *values, receipt


def assert_receipt(receipt):
    for name in ('contract', 'state'):
        if file_sha256(Path(receipt['directory'])/(name+'.json')) != receipt[name+'_sha256']:
            raise RuntimeError('original STC source '+name+' changed during shared execution')


def main(stop_event=None, argv=None):
    parser = original.parser()
    parser.description = __doc__
    parser.add_argument('--continue-from-dir', help='stopped original output; never write its files')
    parser.add_argument('--geometry-cache-mib', type=int, default=512)
    parser.add_argument('--speed-windows', type=int, default=8)
    parser.add_argument('--speed-only', action='store_true')
    a = parser.parse_args(argv)
    if a.audit_only or a.execution != 'native_parallel': parser.error('use original entry for input-only/numpy audit')
    if (not 1 <= a.cpu_workers <= 8 or not 0 <= a.geometry_cache_mib <= 1024
            or not 0 <= a.frame_cache_mib <= 4096 or a.checkpoint_every < 1 or a.speed_windows < 1):
        parser.error('invalid bounded CPU/cache/checkpoint/speed arguments')
    if a.resume and (a.continue_from_dir or a.speed_only): parser.error('resume SAME new output, no repeated migration/speed-only')
    out = Path(a.out_dir).resolve()
    roots = [Path(x).resolve() for x in (a.dataroot, a.stc_root, a.plan_cache)]
    if any(out.is_relative_to(p) for p in roots) or any((p/'training.json').is_file() for p in (out,*out.parents)):
        parser.error('output cannot be inside data/cache/training directories')
    if a.resume:
        if not (out/'contract.json').is_file() or not (out/'state.json').is_file(): parser.error('resume requires SAME shared output')
    elif out.exists(): parser.error('new nonexisting output required')
    # Keep BOTH original and new kernel leases until this invocation is finished.
    # A source process that is still evaluating cannot be snapshotted/migrated.
    with ExitStack() as stack:
        previous = json.loads((out/'contract.json').read_text(encoding='utf-8')) if a.resume else None
        old_contract = old_state = receipt = None
        old_dir = (Path(a.continue_from_dir).resolve() if a.continue_from_dir else
                   Path(previous['execution_migration']['directory']).resolve() if previous and previous.get('execution_migration') else None)
        if old_dir:
            if out.is_relative_to(old_dir) or old_dir.is_relative_to(out): parser.error('distinct nonnested migration output required')
            stack.enter_context(evaluation_lock(old_dir))
            old_contract, old_state, receipt = read_receipt(old_dir)
            if previous and previous['execution_migration'] != receipt: raise RuntimeError('original migrated prefix changed')
        recorded = previous or old_contract
        source = original.STCFourSettingSource.from_files(a.dataroot, a.stc_root, a.plan_cache, cache_mib=a.frame_cache_mib)
        parent = pop_sha = None
        if a.population != 'all':
            if not a.population_manifest: parser.error('dev population requires frozen manifest')
            manifest, _, _ = original.load_manifest(a.population_manifest)
            parent = manifest['parent_keys']; pop_sha = file_sha256(a.population_manifest)
        selected, population = source.select(a.population, parent); inventory = source.preflight(selected)
        print('STC_SHARED_POPULATION '+json.dumps(population | {'selected_keys':'saved in contract.json'}), flush=True)
        if not torch.cuda.is_available(): parser.error('actual CUDA required')
        cfg = original.load_runtime_config(a.config); pcfg = original.make_prepare_config(cfg)
        if (tuple(pcfg.grid.shape_hwd) != original.SHAPE or pcfg.future_frames != 6 or pcfg.frame_dt_s != .5 or pcfg.free_label != 17
                or not np.allclose(pcfg.grid.voxel_size, (.4,)*3, rtol=0, atol=1e-12)
                or not np.allclose((pcfg.grid.x_min,pcfg.grid.y_min,pcfg.grid.z_min), (-40,-40,-1), rtol=0, atol=1e-12)):
            parser.error('unchanged Occ3D trained grid/4 history/6 future required')
        if a.checkpoint: checkpoint = Path(a.checkpoint).resolve()
        elif recorded: checkpoint = Path(recorded['checkpoint']).resolve()
        else:
            if not a.runs_root or not a.run_dir: parser.error('checkpoint or frozen mean discovery required')
            bundle = original.find_frozen_bundle(a.runs_root, a.run_dir, a.source_bundle_dir)
            checkpoint = Path(bundle['candidates'][original.AVERAGE_NAME]['path']).resolve()
        if out.is_relative_to(checkpoint.parent): parser.error('output cannot be inside source mean directory')
        digest = file_sha256(checkpoint)
        root = Path(__file__).resolve().parents[2]
        contract = dict(protocol=original.PROTOCOL, upstream=original.UPSTREAM, windows=len(selected), population=population,
            source=source.metadata, inventory=inventory, population_manifest_sha256=pop_sha,
            dataroot=str(source.root), stc_root=str(source.stc_root), plan_cache=str(source.plan_cache),
            checkpoint=str(checkpoint), checkpoint_sha256=digest, source_epochs=list(original.AVERAGE_EPOCHS),
            thresholds=[.5,None], config_sha256=file_sha256(a.config),
            implementation={f:file_sha256(root/f) for f in dict.fromkeys((*original.IMPLEMENTATION_FILES,*EXTRA_FILES))},
            runtime_environment={k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')},
            torch_version=str(torch.__version__), cpu_workers=a.cpu_workers, frame_cache_mib=a.frame_cache_mib,
            execution=a.execution, graphs=not a.no_graphs, parallel_majority=a.parallel_majority,
            fast_execution=dict(protocol='stc_history_pair_native_exact_v1', geometry_mib=a.geometry_cache_mib,
                speed_windows=a.speed_windows, pair_lifetime='one immutable frozen history/modality',
                future_Strong_projection_output_shared=False), execution_migration=receipt)
        if a.resume:
            if fingerprint(previous) != fingerprint(contract): raise RuntimeError('shared resume contract changed')
            state = json.loads((out/'state.json').read_text(encoding='utf-8'))
            restore(state, contract, int(np.prod(source.shape)))
        elif old_contract:
            state = migrate_state(old_contract, old_state, contract, shape=source.shape)
        else: state = None
        saved_model, joint = original.load_evaluation_model(checkpoint, device='cuda', z_bins=16)
        if (saved_model.get('source_epochs') != list(original.AVERAGE_EPOCHS) or not saved_model.get('averaging')
                or joint.transport.config.history_frames != 4 or file_sha256(checkpoint) != digest):
            raise RuntimeError('unchanged frozen mean required')
        if not a.resume: out.mkdir(parents=True)
        stack.enter_context(evaluation_lock(out))
        if a.resume and (out/'evaluation.json').is_file():
            result = json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
            if result.get('status') == 'complete': print('Already complete: '+str(out/'summary.txt')); return 0
        torch.set_num_threads(1)
        shared = Predictor(joint, pcfg, 'cuda', workers=a.cpu_workers, graphs=not a.no_graphs,
            parallel_majority=a.parallel_majority, geometry_mib=a.geometry_cache_mib)
        stack.callback(shared.close)
        if not a.resume:
            reference = Predictor(joint, pcfg, 'cuda', workers=a.cpu_workers, graphs=not a.no_graphs,
                                   parallel_majority=a.parallel_majority, shared=False)
            try:
                cursor = state['completed_windows'] if state else 0
                if cursor == len(selected): parser.error('source evaluation already complete; nothing left to accelerate')
                windows = selected[cursor:cursor+a.speed_windows]
                print(f'STC_SHARED_CHECK: {len(windows)} same windows; all FOUR settings, no target reads; original prefix={cursor}', flush=True)
                speed = check_and_time(source, windows, reference, shared, stop_event=stop_event)
            finally: reference.close()
            write_json(out/'speed_comparison.json', speed)
            print('STC_SHARED_SPEED '+json.dumps(speed), flush=True)
            if receipt: assert_receipt(receipt)
            if file_sha256(checkpoint) != digest: raise RuntimeError('source mean changed during speed check')
            if a.speed_only: return 0
            if speed['speedup'] <= 1.01: raise RuntimeError('no measured >1% speedup; original evaluation untouched; resume original instead')
            write_json(out/'contract.json', contract)
            if state: write_json(out/'state.json', state)
            shared.provider.clear_pair()
        start = time.perf_counter()
        with (out/'progress.jsonl').open('a', encoding='utf-8') as handle:
            def progress(row):
                handle.write(json.dumps(row, allow_nan=False)+'\n'); handle.flush()
                if row['window'] % 16 == 0 or row['window'] == row['windows']:
                    print(f'STC_SHARED_FOUR {row["window"]}/{row["windows"]} seconds={row["seconds"]:.3f}', flush=True)
            result = evaluate(source, selected, shared, contract, saved=state,
                save=lambda value: write_json(out/'state.json', value), progress=progress,
                stop_event=stop_event, checkpoint_every=a.checkpoint_every)
        if receipt: assert_receipt(receipt)
        if file_sha256(checkpoint) != digest: raise RuntimeError('source mean changed during evaluation')
        result.update(contract=contract, elapsed_seconds_this_invocation=time.perf_counter()-start,
                      history_pairs_reused=shared.provider.reused_histories)
        write_json(out/'evaluation.json', result)
        report = original.summary(result)+'Shared frozen history/native surface execution; old prefix intact; no new learning or score protocol.\n'
        (out/'summary.txt').write_text(report, encoding='utf-8'); print(report, flush=True)
        print('RESULT: '+str(out/'summary.txt'), flush=True)
    return 0


if __name__ == '__main__':
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stopped.set())
    try: sys.exit(main(stopped))
    except InterruptedError:
        print('Stopped during read-only speed check; original evaluation state unchanged.', flush=True)
        sys.exit(130)
