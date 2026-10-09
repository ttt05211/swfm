#!/usr/bin/env python3
"""Read-only stopped 10Hz prefix; same-v2 2x2/4x1 speed probe, never full eval."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.waymo_i2world import SHAPE, file_sha256, fingerprint
from real_motion.waymo_i2world_10hz import PROTOCOL, WaymoI2World10HzSource
from real_motion.native_column_cpu import prepare_native
from real_motion.waymo_native_execution import prepare_waymo_native
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_surface_checkpoint_selection import AVERAGE_EPOCHS, load_evaluation_model
from tools.real_motion.waymo_parallel_execution import PROTOCOL as PARALLEL_PROTOCOL
from tools.real_motion.waymo_worker_layout_benchmark import paired_worker_layout_speed, LAYOUTS
from tools.real_motion.waymo_zero_shot_common import restore, write_json

ROOT = Path(__file__).resolve().parents[2]


def load_spec(contract, config):
    """Require exact saved scientific contract, including all original code."""
    if (contract.get('protocol') != PROTOCOL or contract.get('execution') != 'native_parallel'
            or contract.get('fast_execution', {}).get('protocol') != PARALLEL_PROTOCOL
            or contract.get('thresholds') != [.5, None]):
        raise RuntimeError('stopped unchanged native-index 10Hz parallel prefix required')
    if (str(torch.__version__) != contract['torch_version']
            or file_sha256(config) != contract['config_sha256']
            or {k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')}
            != contract['runtime_environment']):
        raise RuntimeError('probe config/Torch/runtime environment changed')
    for name, digest in contract['implementation'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or file_sha256(path) != digest:
            raise RuntimeError('original parallel implementation changed: ' + name)
    fast = contract['fast_execution']
    if (not 1 <= fast['parallel_chunk'] <= 16 or not 0 <= fast['geometry_cache_mib'] <= 4096
            or not 0 <= contract['frame_cache_mib'] <= 4096):
        raise RuntimeError('invalid saved bounded cache/chunk configuration')
    files = contract['data']['source_files']
    if len(files) != 2:
        raise RuntimeError('two trusted metadata files required')
    source = WaymoI2World10HzSource.from_files(contract['data_root'],
        info_file=files[0]['path'], pose_file=files[1]['path'],
        raw_free_label=contract['data']['raw_free_label'], cache_mib=contract['frame_cache_mib'])
    selected = source.windows[:contract['windows']]
    inventory = source.preflight(selected)
    if (len(selected) != contract['windows'] or source.metadata != contract['data']
            or source.manifest_fingerprint != contract['manifest_fingerprint']
            or inventory != contract['inventory']):
        raise RuntimeError('probe data/population/metadata changed')
    pcfg = make_prepare_config(load_runtime_config(config))
    if (tuple(pcfg.grid.shape_hwd) != SHAPE or not np.allclose(pcfg.grid.voxel_size, (.4,) * 3, rtol=0, atol=1e-12)
            or not np.allclose((pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min), (-40,-40,-1), rtol=0, atol=1e-12)
            or pcfg.future_frames != 6 or pcfg.free_label != 17 or pcfg.frame_dt_s != .5):
        raise RuntimeError('unchanged trained six-slot/grid configuration required')
    checkpoint = Path(contract['checkpoint'])
    if file_sha256(checkpoint) != contract['checkpoint_sha256']:
        raise RuntimeError('probe frozen checkpoint changed')
    saved, joint = load_evaluation_model(checkpoint, device='cpu', z_bins=16)
    if (saved.get('source_epochs') != list(AVERAGE_EPOCHS) or not saved.get('averaging')
            or joint.transport.config.history_frames != 4):
        raise RuntimeError('unchanged FOUR-history frozen mean required')
    del joint
    prepare_native()
    native = prepare_waymo_native()
    if native.manifest['fingerprint'] != fast['native_fingerprint']:
        raise RuntimeError('probe native binary configuration changed')
    return dict(waymo_root=contract['data_root'], info_file=files[0]['path'], pose_file=files[1]['path'],
        raw_free_label=contract['data']['raw_free_label'], frame_cache_mib=contract['frame_cache_mib'],
        shape=source.shape, windows=len(selected), manifest_fingerprint=source.manifest_fingerprint,
        data=source.metadata, inventory=inventory, checkpoint=str(checkpoint),
        checkpoint_sha256=contract['checkpoint_sha256'], pcfg=pcfg, device='cuda', workers=2, backend='v2',
        graphs=contract['graphs'], geometry_mib=fast['geometry_cache_mib'], surface_chunk=fast['surface_chunk'],
        parallel_majority=contract['parallel_majority'], history_prefetch=fast['next_history_prefetch'],
        implementation=contract['implementation'])


def main(stop_event=None, argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--continue-from-dir', required=True, help='STOPPED source; contract/state remain read only')
    p.add_argument('--out-dir', required=True, help='NEW diagnostic directory; no resume state written')
    p.add_argument('--config', default=str(ROOT / 'configs/real_motion_occfm.yaml'))
    p.add_argument('--speed-windows', type=int, default=64)
    p.add_argument('--repeats', type=int, choices=(2, 4), default=2)
    a = p.parse_args(argv)
    if not 32 <= a.speed_windows <= 128:
        p.error('use 32..128 windows; repeats2/4 have balanced order')
    old, out = Path(a.continue_from_dir).resolve(), Path(a.out_dir).resolve()
    if not old.is_dir() or out.exists() or out.is_relative_to(old) or old.is_relative_to(out):
        p.error('existing stopped source and distinct NEW nonnested output required')
    if not torch.cuda.is_available():
        p.error('actual CUDA required; do not report CPU results as server GPU throughput')
    # Hold the source lease for the ENTIRE probe: cannot benchmark while its
    # original evaluator runs or restarts and consumes the same resources.
    with evaluation_lock(old):
        paths = [old / name for name in ('contract.json', 'state.json')]
        original_bytes = [path.read_bytes() for path in paths]
        contract, saved = [json.loads(value) for value in original_bytes]
        state = restore(saved, contract, voxel_count=int(np.prod(SHAPE)))
        cursor = state['completed_windows']
        if cursor >= contract['windows']:
            p.error('source evaluation already complete; no remaining prefix to probe')
        if min(a.speed_windows, contract['windows'] - cursor) < 32:
            p.error('fewer than32 windows remain; continue existing eval instead of paying probe overhead')
        data_root, checkpoint = Path(contract['data_root']).resolve(), Path(contract['checkpoint']).resolve()
        if out.is_relative_to(data_root) or out.is_relative_to(checkpoint.parent):
            p.error('diagnostic output cannot be inside data or frozen checkpoint directory')
        if any((d / 'training.json').is_file() for d in (out, *out.parents)):
            p.error('diagnostic output cannot be inside training output')
        spec = load_spec(contract, a.config)
        indices = list(range(cursor, min(contract['windows'], cursor + a.speed_windows)))
        out.mkdir(parents=True)
        receipt = dict(directory=str(old), completed_windows=cursor,
            contract_sha256=file_sha256(paths[0]), state_sha256=file_sha256(paths[1]))
        write_json(out / 'source_receipt.json', receipt)
        def progress(row):
            print(f"LAYOUT_SPEED {row['arm']} repeat={row['repeat']}/{row['repeats']} "
                  f"windows={row['windows']} seconds/window={row['seconds_per_window']:.4f}", flush=True)
        try:
            result = paired_worker_layout_speed(spec, indices, chunk=contract['fast_execution']['parallel_chunk'],
                repeats=a.repeats, stop_event=stop_event, progress=progress)
            if ([path.read_bytes() for path in paths] != original_bytes
                    or file_sha256(spec['checkpoint']) != spec['checkpoint_sha256']):
                raise RuntimeError('source prefix/checkpoint changed during diagnostic')
        except BaseException as error:
            write_json(out / 'failure.json', dict(status='stopped' if isinstance(error, InterruptedError) else 'failed',
                error_type=type(error).__name__, message=str(error), source=receipt,
                no_metric_cursor_updates=True, no_automatic_resume=True))
            raise
        result.update(source=receipt, probe_implementation={
            str(Path(__file__).relative_to(ROOT)): file_sha256(__file__),
            'tools/real_motion/waymo_worker_layout_benchmark.py': file_sha256(ROOT / 'tools/real_motion/waymo_worker_layout_benchmark.py')})
        write_json(out / 'speed.json', result)
        lines = ['===== WAYMO10 SAME-V2 WORKER LAYOUT / SPEED ONLY =====',
            f"saved_cursor={cursor} probe_windows={len(indices)} repeats={a.repeats}; source prefix unchanged"]
        for name in LAYOUTS:
            lines.append(f"{name}: seconds/window={result['seconds_per_window'][name]:.6f}")
        lines += [f"4x1_vs_2x2_speedup={result['speedup_4x1_vs_2x2']:.4f}",
            'recommended_layout=' + result['recommended_layout'],
            'SIX/probability/motion/integer_counts_exact=PASS',
            'Actual eval throughput incl history/GT/metrics/IPC; NOT single-window Dense Forecast FPS.',
            'Warmup/hashes/checks excluded; history LRUs reset each pass; only first window/worker warmed.',
            'No training/threshold change/metric cursor update/automatic evaluation restart.']
        summary = '\n'.join(lines) + '\n'
        (out / 'summary.txt').write_text(summary, encoding='utf-8')
        print(summary, flush=True)
        print('RESULT: ' + str(out / 'summary.txt'), flush=True)
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *args: event.set())
    sys.exit(main(event))
