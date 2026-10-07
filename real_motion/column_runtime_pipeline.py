"""Bounded RAM-only immutable frame reuse and CPU I/O look-ahead; no CUDA worker."""
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
import numpy as np


class CachedColumnSource:
    def __init__(self, source, max_mib=256, *, copy_on_insert=True):
        if max_mib < 0: raise ValueError('negative frame cache budget')
        self.source, self.limit = source, int(max_mib*2**20)
        self.copy_on_insert=bool(copy_on_insert)
        self.rows, self.pending = OrderedDict(), {}
        self.lock = Lock(); self.bytes = 0; self.hits = 0; self.misses = 0

    def __getattr__(self, name): return getattr(self.source, name)

    def _get(self, key, loader):
        with self.lock:
            if key in self.rows:
                self.hits += 1; self.rows.move_to_end(key); return self.rows[key][0]
            future = self.pending.get(key)
            owner = future is None
            if owner:
                self.misses += 1; future = Future(); self.pending[key] = future
        if not owner: return future.result()
        try:
            value = loader()
            arrays = value if isinstance(value, tuple) else (value,)
            # NuScenes NPZ loaders already return fresh owning arrays. Training
            # can opt out of a second full 3-D copy while still freezing the
            # cached views read-only. Generic callers retain copy-on-insert.
            arrays = tuple(
                np.asarray(a).copy() if self.copy_on_insert else np.asarray(a)
                for a in arrays)
            for a in arrays: a.setflags(write=False)
            value = arrays if isinstance(value, tuple) else arrays[0]
            size = sum(a.nbytes for a in arrays)
            with self.lock:
                if size <= self.limit:
                    while self.rows and self.bytes+size > self.limit:
                        _, (_, old_size) = self.rows.popitem(last=False); self.bytes -= old_size
                    self.rows[key] = (value, size); self.bytes += size
                self.pending.pop(key)
                future.set_result(value)
            return value
        except BaseException as exc:
            with self.lock:
                self.pending.pop(key, None); future.set_exception(exc)
            raise

    def load_occ3d(self, scene, token, require_lidar_mask=True):
        return self._get(('occ', scene, token, bool(require_lidar_mask)),
            lambda: self.source.load_occ3d(scene, token, require_lidar_mask=require_lidar_mask))

    def load_semantics(self, scene, token):
        with self.lock:
            key = ('occ', scene, token, True)
            if key in self.rows:
                self.hits += 1; self.rows.move_to_end(key); return self.rows[key][0][0]
        return self._get(('sem', scene, token), lambda: self.source.load_semantics(scene, token))

    def load_lidar_observation(self, scene, token): return self.load_occ3d(scene, token)[1]


def prefetch_raw_columns(provider, source, records, *, include_gt=True):
    """Bounded NEXT raw windows; ordered consumption, never worker-owned CUDA.

    Default remains one next window. Independent evaluation can opt into
    parallel FIXED CPU geometry via provider.raw_prefetch_workers/depth.
    """
    if not hasattr(provider, 'load_raw_columns'):
        for record in records: yield record, None
        return
    iterator = iter(records)
    workers = getattr(provider, 'raw_prefetch_workers', 1)
    depth = getattr(provider, 'raw_prefetch_depth', workers)
    if type(workers) is not int or type(depth) is not int or not 1 <= workers <= depth <= 32:
        raise ValueError('raw prefetch requires 1 <= workers <= depth <= 32')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        def submit_next():
            record = next(iterator, None)
            if record is None: return False
            pending.append((record, pool.submit(provider.load_raw_columns, source, record, include_gt=include_gt)))
            return True
        try:
            for _ in range(depth):
                if not submit_next(): break
            while pending:
                record, future = pending.popleft()
                raw = future.result()
                submit_next()
                yield record, raw
        finally:
            for _, future in pending: future.cancel()
