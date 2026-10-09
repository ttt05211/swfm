#!/usr/bin/env python3
"""Real exported history warp microbenchmark; NOT model FPS or L40S forecast."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import time
import json
import numpy as np
import torch
from real_motion.geometry import OccupancyGrid, relative_transform
from real_motion.local_replay_bundle import ReplayBundle
from real_motion.runtime_fastpath import inverse_warp_sequence_cuda_exact
from real_motion.strong_warp_execution import strong_warp_execution
from real_motion.strong_w2det import inverse_warp
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--windows', type=int, default=6); p.add_argument('--repeats', type=int, default=5)
    a = p.parse_args(); out = Path(a.out_dir)
    if out.exists() or a.windows < 1 or a.repeats < 1: p.error('new output/positive workload required')
    if not torch.cuda.is_available(): p.error('actual CUDA required')
    torch.set_num_threads(1); bundle = ReplayBundle(a.replay)
    grid = OccupancyGrid(); trials = []; matched = 0
    try:
        for wi in range(min(a.windows, len(bundle.manifest['windows']))):
            _, raw, _ = bundle.window(wi, labels=False)
            sem = raw['history_occ'][-1].copy(); sem[np.isin(sem, DYNAMIC_CLASS_IDS)] = 17
            poses = [relative_transform(raw['history_poses'][-1], pose) for pose in raw['future_poses']]
            def run(backend):
                with strong_warp_execution(backend):
                    return inverse_warp_sequence_cuda_exact(sem, poses, grid=grid, free_label=17, device='cuda')
            expected = run('reference'); actual = run('buffered')
            for pose, old, new in zip(poses, expected, actual):
                ref = inverse_warp(sem, pose, grid, 17)
                for cpu, previous, optimized in zip(ref, old, new):
                    np.testing.assert_array_equal(previous, cpu); np.testing.assert_array_equal(optimized, cpu)
            matched += 1
            for repeat in range(a.repeats):
                order = ('reference', 'buffered') if (wi+repeat) % 2 == 0 else ('buffered', 'reference')
                for backend in order:
                    torch.cuda.synchronize(); tick = time.perf_counter(); result = run(backend)
                    torch.cuda.synchronize(); seconds = time.perf_counter()-tick
                    for old, new in zip(expected, result):
                        for x, y in zip(old, new): np.testing.assert_array_equal(x, y)
                    trials.append(dict(window=wi, repeat=repeat, backend=backend, seconds=seconds))
            print(f'REAL_WARP {wi+1} full200x200x16 x SIX byte_exact=PASS', flush=True)
        means = {mode: float(np.mean([r['seconds'] for r in trials if r['backend'] == mode]))
                 for mode in ('reference', 'buffered')}
        result = dict(windows=matched, repeats=a.repeats, means_ms={k: 1000*v for k, v in means.items()},
            warp_only_speedup=means['reference']/means['buffered'], trials=trials,
            scope='actual local CUDA + real exported histories; SIX inverse warps ONLY; NOT FPS/L40S prediction',
            device=torch.cuda.get_device_name(0), exact_float64_reference_windows=matched)
        out.mkdir(parents=True); (out/'speed.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k != 'trials'}, indent=2), flush=True)
    finally: bundle.close()


if __name__ == '__main__': main()
