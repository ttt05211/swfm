"""Opt-in parallel scheduling of the unchanged native Strong majority vote.

Disjoint X interiors retain a two-row source halo for the frozen 5x5x1
neighbourhood. Integer results are joined in original order; ambiguous votes
still use the original full-grid SciPy edge replay, once on the caller.
No geometry, learned outputs, labels or previous-window results are cached.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
import time

import numpy as np

_EXECUTION = ContextVar('swfm_strong_majority_execution', default=None)


def selected_execution():
    return _EXECUTION.get()


@contextmanager
def strong_majority_execution(execution=None):
    if execution is not None and not isinstance(execution, ParallelNativeMajority):
        raise TypeError('explicit ParallelNativeMajority or reference None required')
    token = _EXECUTION.set(execution)
    try:
        yield
    finally:
        _EXECUTION.reset(token)


class ParallelNativeMajority:
    """Bounded persistent worker pool, owning no per-window prediction storage."""
    def __init__(self, workers=4, *, profile=False):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError('majority workers must be an integer in 1..8')
        self.workers, self.profile = workers, bool(profile)
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix='strong-majority')
        self.lock = Lock()
        self.closed = False
        self.totals = dict(calls=0, blocks=0, ambiguous_cells=0,
                           native_worker_seconds_sum=0., join_seconds=0., scipy_edge_seconds=0.)

    def __call__(self, semantics, unknown_mask, *, kernel=(5, 5, 1), min_fraction=.3, device=None):
        from .native_column_cpu import get_prepared_native
        from .v18_execution_trial import _scipy_edges
        if self.closed:
            raise RuntimeError('Strong majority execution has been closed')
        if tuple(kernel) != (5, 5, 1) or abs(float(min_fraction)-.3) > 1e-12:
            raise ValueError('native majority is frozen to kernel=(5,5,1), threshold=.3')
        native = get_prepared_native()
        sem, unknown = np.asarray(semantics), np.asarray(unknown_mask, bool)
        if sem.ndim != 3 or sem.shape != unknown.shape:
            raise ValueError('semantic/unknown 3D shape mismatch')
        if sem.dtype != np.uint8:
            raise TypeError('native array dtype mismatch: '+str(sem.dtype)+' != uint8')
        # Validate dimensions with the same native contract before dispatch.
        native._grid(sem.shape)
        boundaries = np.linspace(0, sem.shape[0], min(self.workers, sem.shape[0])+1, dtype=int)

        def block(begin, end):
            lo, hi = max(0, begin-2), min(sem.shape[0], end+2)
            tick = time.perf_counter() if self.profile else 0.
            filled, ambiguous = native.v18_majority(sem[lo:hi], unknown[lo:hi])
            seconds = time.perf_counter()-tick if self.profile else 0.
            interior = slice(begin-lo, end-lo)
            bits = ambiguous[interior]
            flat = np.flatnonzero(bits)
            coordinates = np.stack(np.unravel_index(flat, bits.shape), axis=1)
            coordinates[:, 0] += begin
            return begin, end, filled[interior], coordinates, seconds

        if len(boundaries) == 2:
            rows = [block(0, sem.shape[0])]
        else:
            futures = [self.pool.submit(block, int(begin), int(end))
                       for begin, end in zip(boundaries[:-1], boundaries[1:])]
            rows = [future.result() for future in futures]
        # Consume in X order. Every voxel is included exactly once, and the
        # concatenated ambiguous coordinates match full-grid argwhere order.
        tick = time.perf_counter() if self.profile else 0.
        output = np.empty(sem.shape, np.uint8)
        for begin, end, values, _, _ in rows:
            output[begin:end] = values
        coordinates = np.concatenate([row[3] for row in rows], axis=0)
        join_seconds = time.perf_counter()-tick if self.profile else 0.
        tick = time.perf_counter() if self.profile else 0.
        if len(coordinates):
            known = ~unknown
            # Do not sort/scan the entire known volume to replay a few points.
            # The original float32 filter needs only classes present in these
            # patches: all others have zero scores and cannot win at .3.
            _scipy_edges(output, sem, known, coordinates, None, min_fraction)
        scipy_seconds = time.perf_counter()-tick if self.profile else 0.
        with self.lock:
            self.totals['calls'] += 1
            self.totals['blocks'] += len(rows)
            self.totals['ambiguous_cells'] += len(coordinates)
            self.totals['native_worker_seconds_sum'] += sum(row[4] for row in rows)
            self.totals['join_seconds'] += join_seconds
            self.totals['scipy_edge_seconds'] += scipy_seconds
        return output

    def stats(self):
        with self.lock:
            return dict(workers=self.workers, profile_enabled=self.profile, **self.totals,
                        timing_scope='native worker sums OVERLAP; NOT additive wall time or FPS')

    def close(self):
        self.pool.shutdown(wait=True)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
