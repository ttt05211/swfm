"""Bounded CPU-only horizon pipeline; no model, sampling RNG or cache changes."""
from concurrent.futures import ThreadPoolExecutor
import os

from .native_column_cpu import bundle_enabled


def horizon_pipeline_enabled():
    value = os.environ.get('SWFM_COLUMN_CPU_HORIZONS', '1')
    if value not in ('0', '1'):
        raise ValueError('SWFM_COLUMN_CPU_HORIZONS must be 0 or 1')
    return bundle_enabled() and value == '1'


def sampling_worker_budget(cpu_workers=4, requested=0, *, horizons=False):
    if cpu_workers < 1 or requested < 0:
        raise ValueError('positive CPU workers / nonnegative sampling workers required')
    if horizons:
        # Leave room for the two I/O workers and autograd-owning caller on the
        # server's ten-core quota. This bounds ACTIVE scratch maps, not merely
        # the number of Python Futures (at most one window batch is scheduled).
        return min(requested or min(cpu_workers, 6), 8)
    return requested or min(cpu_workers, 4)


class HorizonCpuPool:
    """Separate small queues prevent feature jobs sitting behind all candidates.

    The COMBINED worker count is bounded, not this many workers per executor.
    Every history-index job is enqueued before its dependent feature jobs.
    No worker submits children; a one-worker FIFO pool is also deadlock-free.
    """
    def __init__(self, workers):
        if not 1 <= workers <= 8:
            raise ValueError('horizon pipeline requires 1..8 combined CPU workers')
        self.workers = workers
        self.candidate_workers = max(1, workers // 2)
        self.feature_workers = workers - self.candidate_workers if workers > 1 else 1
        self.candidates = ThreadPoolExecutor(max_workers=self.candidate_workers,
                                             thread_name_prefix='column-candidate')
        self.features = (ThreadPoolExecutor(max_workers=self.feature_workers,
                                            thread_name_prefix='column-feature')
                         if workers > 1 else self.candidates)

    def submit(self, fn, *args, **kwargs):
        return self.candidates.submit(fn, *args, **kwargs)

    def submit_features(self, fn, *args, **kwargs):
        return self.features.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.candidates.shutdown(wait=wait, cancel_futures=cancel_futures)
        if self.features is not self.candidates:
            self.features.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(wait=True, cancel_futures=exc_type is not None)


def cpu_sampling_pool(workers, *, horizons=False):
    return HorizonCpuPool(workers) if horizons else ThreadPoolExecutor(max_workers=workers)
