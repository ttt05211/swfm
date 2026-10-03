"""Bounded CPU-only horizon pipeline; no model, sampling RNG or cache changes."""
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import count
from queue import Empty, PriorityQueue
from threading import Lock
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
    """Work-conserving shared capacity, with ready feature jobs prioritized.

    Candidate-only stages can use ALL workers, rather than half being idle in
    a fixed 3+3 split. Queue priority never preempts an already-running job.
    Parent history-index jobs share feature priority and precede dependents in
    FIFO sequence. No worker submits children; even one worker is safe.
    """
    def __init__(self, workers):
        if not 1 <= workers <= 8:
            raise ValueError('horizon pipeline requires 1..8 combined CPU workers')
        self.workers = workers
        self.candidate_workers = self.feature_workers = workers  # same SHARED capacity, not summed
        self.shared_worker_pool = True
        self._queue = PriorityQueue(); self._sequence = count(); self._lock = Lock(); self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='column-shared')
        try:
            self._runners = [self._executor.submit(self._worker) for _ in range(workers)]
        except BaseException:
            self.shutdown(wait=True, cancel_futures=True)
            raise

    def _worker(self):
        while True:
            _, _, item = self._queue.get()
            try:
                if item is None: return
                future, fn, args, kwargs = item
                if not future.set_running_or_notify_cancel(): continue
                try: value = fn(*args, **kwargs)
                except BaseException as error: future.set_exception(error)
                else: future.set_result(value)
            finally:
                self._queue.task_done()

    def _submit(self, priority, fn, args, kwargs):
        with self._lock:
            if self._closed: raise RuntimeError('cannot schedule new futures after shutdown')
            future = Future()
            self._queue.put((priority, next(self._sequence), (future, fn, args, kwargs)))
            return future

    def submit(self, fn, *args, **kwargs):
        return self._submit(1, fn, args, kwargs)

    def submit_features(self, fn, *args, **kwargs):
        return self._submit(0, fn, args, kwargs)

    def shutdown(self, wait=True, *, cancel_futures=False):
        canceled = []
        with self._lock:
            if not self._closed:
                self._closed = True
                if cancel_futures:
                    while True:
                        try: _, _, item = self._queue.get_nowait()
                        except Empty: break
                        self._queue.task_done()
                        if item is not None: canceled.append(item[0])
                # Sentinels run AFTER all non-canceled CPU work. No worker is
                # forcibly terminated, and running dependents see canceled
                # parent Futures before joining the executor below.
                for _ in range(self.workers): self._queue.put((2, next(self._sequence), None))
        for future in canceled: future.cancel()
        self._executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(wait=True, cancel_futures=exc_type is not None)


def cpu_sampling_pool(workers, *, horizons=False):
    return HorizonCpuPool(workers) if horizons else ThreadPoolExecutor(max_workers=workers)
