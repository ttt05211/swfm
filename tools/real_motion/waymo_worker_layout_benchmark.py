"""Same-v2 2x2 / 4x1 worker-layout probe; never updates scientific counts."""
from collections import defaultdict
import time

import numpy as np

from real_motion.waymo_i2world import fingerprint
from tools.real_motion.waymo_parallel_execution import SpawnPool

LAYOUTS = {'parallel_2x2': (2, 2), 'parallel_4x1': (4, 1)}


def paired_worker_layout_speed(spec, indices, *, chunk=8, repeats=2, stop_event=None,
                              progress=None):
    """Actual eval incl SIX outputs/GT/counts/IPC; fixed four CPU work slots.

    Pools are resident but run sequentially, not concurrently. Startup, first
    window warmup, correctness hashing and result checks are outside timing.
    Each timed pass clears history LRUs and warms ONLY one window per worker.
    No metric cursor is admitted or saved by this diagnostic.
    """
    indices = list(indices)
    if (len(indices) < 4 or len(indices) > 128 or len(set(indices)) != len(indices)
            or indices != list(range(indices[0], indices[0] + len(indices)))
            or not 1 <= chunk <= 16 or repeats not in (2, 4)):
        raise ValueError('4..128 consecutive windows, chunk1..16, balanced repeats2/4 required')
    pools, startup, signatures, worker_pids = {}, {}, {}, {}
    durations = defaultdict(list)
    details = defaultdict(lambda: defaultdict(float))

    def run(pool, *, hashes=False):
        rows = [r for batch in pool.batches(indices, chunk, hashes=hashes, stop_event=stop_event)
                for r in batch]
        if len(rows) != len(indices):
            raise InterruptedError('worker-layout probe stopped; scientific prefix unchanged')
        return rows

    def validate(rows, *, hashes=False):
        if [r['index'] for r in rows] != indices or not all(r['exactness_passed'] for r in rows):
            raise RuntimeError('worker-layout unverified/wrong population')
        if any(r['edits'].get('removed', 0) for r in rows):
            raise RuntimeError('worker-layout ADD-only rule violated')
        if hashes and any(set(r['signatures'] or {}) != {'transport', 'dense', 'probability', 'motion'}
                          for r in rows):
            raise RuntimeError('worker-layout incomplete SIX/probability/motion hashes')
        return [(r['index'], r['counts'], r['edits']) for r in rows]

    try:
        for name, (processes, threads) in LAYOUTS.items():
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError('worker-layout probe stopped before startup')
            tick = time.perf_counter()
            pool = SpawnPool(dict(spec, backend='v2', workers=threads), processes)
            pools[name] = pool
            pool.reset(indices, chunk)
            startup[name] = time.perf_counter() - tick
        for name, pool in pools.items():
            rows = run(pool, hashes=True)
            validate(rows, hashes=True)
            signatures[name] = [(r['index'], r['signatures'], r['counts'], r['edits']) for r in rows]
            worker_pids[name] = sorted({r['pid'] for r in rows})
        if signatures['parallel_2x2'] != signatures['parallel_4x1']:
            raise RuntimeError('2x2/4x1 SIX/probability/motion/integer-count byte gate failed')
        reference_counts = [(r[0], r[2], r[3]) for r in signatures['parallel_2x2']]
        for repeat in range(repeats):
            order = tuple(LAYOUTS) if repeat % 2 == 0 else tuple(reversed(LAYOUTS))
            for name in order:
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError('worker-layout probe stopped before timed pass')
                pool = pools[name]
                pool.reset(indices, chunk)
                tick = time.perf_counter()
                rows = run(pool)
                elapsed = time.perf_counter() - tick
                if validate(rows) != reference_counts:
                    raise RuntimeError('worker-layout timed integer counts/edits differ from gate')
                seconds = elapsed / len(indices)
                if not np.isfinite(seconds) or seconds <= 0:
                    raise RuntimeError('worker-layout invalid elapsed time')
                durations[name].append(seconds)
                for row in rows:
                    for k, v in row['model_stages'].items():
                        details[name][k] += v
                if progress:
                    progress(dict(arm=name, repeat=repeat + 1, repeats=repeats,
                                  windows=len(indices), seconds_per_window=seconds))
        mean = {k: float(np.mean(v)) for k, v in durations.items()}
        ratio = mean['parallel_2x2'] / mean['parallel_4x1']
        # Both balanced orders must beat baseline. A marginal average gain is
        # not sufficient to recommend paying startup/migration overhead.
        repeat_ratios = [a / b for a, b in zip(durations['parallel_2x2'], durations['parallel_4x1'])]
        selected = 'parallel_4x1' if ratio >= 1.10 and min(repeat_ratios) > 1 else 'parallel_2x2'
        return dict(status='complete', windows=len(indices), indices=indices,
            window_fingerprint=fingerprint(indices), repeats=repeats, chunk=chunk,
            layouts={k: dict(processes=p, threads_per_process=t) for k, (p, t) in LAYOUTS.items()},
            seconds_per_window=mean, speedup_4x1_vs_2x2=ratio,
            repeat_speedups_4x1_vs_2x2=repeat_ratios, repeat_seconds_per_window=dict(durations),
            recommended_layout=selected, recommendation_rule='>=10% mean gain AND faster in every repeat',
            startup_and_first_window_warmup_seconds=startup, correctness_worker_pids=worker_pids,
            counts_exact=True, probability_and_six_dense_bytes_exact=True, motion_bytes_exact=True,
            overlapping_worker_model_seconds_per_window={
                name: {k: v / (len(indices) * repeats) for k, v in stages.items()}
                for name, stages in details.items()},
            no_metric_cursor_updates=True, no_automatic_resume_or_backend_promotion=True,
            scope='same-v2 same-window eval including history/SIX dense/GT/metrics/IPC; NOT formal FPS',
            timing_policy='AB/BA balanced; reset history LRUs each pass; ONLY first window/worker warmed',
            worker_stage_seconds_are_overlapping=True,
            resident_idle_pools_note='six total model workers resident; only ONE layout computes at a time')
    finally:
        for pool in pools.values():
            pool.close()
