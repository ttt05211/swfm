#!/usr/bin/env python3
"""Real history replay, majority ONLY; not model FPS or a server speed claim."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import time
from contextlib import ExitStack
import numpy as np
import torch
from real_motion.geometry import OccupancyGrid, relative_transform
from real_motion.local_replay_bundle import ReplayBundle
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.native_column_cpu import prepare_native, get_prepared_native
from real_motion.runtime_fastpath import inverse_warp_sequence_cuda_exact
from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution
from real_motion.v18_execution_trial import majority_fill_native_exact, _scipy_edges
from real_motion.strong_w2det import majority_fill


def reference_profile(sem, unknown):
    start = time.perf_counter()
    out, ambiguous = get_prepared_native().v18_majority(sem, unknown)
    native_seconds = time.perf_counter()-start
    start = time.perf_counter(); coordinates = np.argwhere(ambiguous)
    coordinates_seconds = time.perf_counter()-start
    classes_seconds = replay_seconds = 0.
    if len(coordinates):
        start = time.perf_counter()
        known = ~unknown; classes = np.unique(sem[known]).astype(np.int64)
        classes_seconds = time.perf_counter()-start
        start = time.perf_counter()
        _scipy_edges(out, sem, known, coordinates, classes, .3)
        replay_seconds = time.perf_counter()-start
    return out, dict(native_seconds=native_seconds, coordinates_seconds=coordinates_seconds,
                     classes_seconds=classes_seconds, scipy_replay_seconds=replay_seconds,
                     ambiguous_cells=len(coordinates), unknown_cells=int(unknown.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--windows', type=int, default=6); p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 6])
    a = p.parse_args(); out = Path(a.out_dir)
    if out.exists() or a.windows < 1 or a.repeats < 1 or not a.workers or len(set(a.workers)) != len(a.workers):
        p.error('new output/positive workload/distinct worker settings required')
    if not torch.cuda.is_available(): p.error('actual CUDA replay warp required')
    torch.set_num_threads(1); prepare_native(); grid = OccupancyGrid()
    bundle = ReplayBundle(a.replay); trials, profiles, matched = [], [], 0
    try:
        with ExitStack() as stack:
            executors = {f'parallel{n}': stack.enter_context(ParallelNativeMajority(n)) for n in a.workers}
            modes = ['reference', *executors]
            for wi in range(min(a.windows, len(bundle.manifest['windows']))):
                _, raw, _ = bundle.window(wi, labels=False)
                sem = raw['history_occ'][-1].copy(); sem[np.isin(sem, DYNAMIC_CLASS_IDS)] = 17
                poses = [relative_transform(raw['history_poses'][-1], pose) for pose in raw['future_poses']]
                # The same six FULL grids feed every arm. Fresh warp outside
                # this majority-only timing; no synthetic unknown distribution.
                warped = inverse_warp_sequence_cuda_exact(sem, poses, grid=grid, free_label=17, device='cuda')
                inputs = [(labels, ~known) for labels, known in warped]
                def run(mode):
                    with strong_majority_execution(executors.get(mode)):
                        return [majority_fill_native_exact(labels, unknown) for labels, unknown in inputs]
                expected = run('reference')
                for (labels, unknown), previous in zip(inputs, expected):
                    scipy_ref = majority_fill(labels, unknown)
                    np.testing.assert_array_equal(previous, scipy_ref)
                    detailed, profile = reference_profile(labels, unknown)
                    np.testing.assert_array_equal(detailed, previous); profiles.append(dict(window=wi, **profile))
                for mode in modes:
                    for previous, actual in zip(expected, run(mode)):
                        np.testing.assert_array_equal(actual, previous)
                matched += 1
                for repeat in range(a.repeats):
                    offset = (wi+repeat) % len(modes)
                    for mode in modes[offset:]+modes[:offset]:
                        start = time.perf_counter(); actual = run(mode); elapsed = time.perf_counter()-start
                        for previous, value in zip(expected, actual): np.testing.assert_array_equal(value, previous)
                        trials.append(dict(window=wi, repeat=repeat, mode=mode, seconds=elapsed))
                print(f'REAL_MAJORITY {wi+1} full200x200x16 x SIX SciPy/native/parallel bytes=PASS', flush=True)
            means = {mode: float(np.mean([row['seconds'] for row in trials if row['mode'] == mode])) for mode in modes}
            result = dict(windows=matched, repeats=a.repeats, means_ms={k: v*1000 for k, v in means.items()},
                majority_only_speedups={k: means['reference']/v for k, v in means.items()},
                reference_profile=profiles, executor_stats={k: v.stats() for k, v in executors.items()}, trials=trials,
                scope='actual exported history grids; SIX majority fills ONLY; NOT model FPS/L40S prediction',
                cuda_device=torch.cuda.get_device_name(0), native=get_prepared_native().info())
            out.mkdir(parents=True); (out/'speed.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in result.items() if k not in ('trials', 'reference_profile', 'native')}, indent=2), flush=True)
    finally:
        bundle.close()


if __name__ == '__main__': main()
