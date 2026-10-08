"""Bounded, integrity-checked compressed CPU geometry, never learned state or GT.

This is an implementation cache of local trusted artifacts. Like Torch's local
checkpoints its pickle payload must not be supplied from an untrusted party.
Raw causal inputs are hashed on every lookup; corrupt artifacts fail closed.
The cache never evicts/deletes existing entries: when full it stops admitting.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import pickle
import shutil
from threading import Lock, BoundedSemaphore
import time
import uuid
import warnings
import zlib
import numpy as np

PROTOCOL = 'causal_geometry_cpu_strong_static_frontier_v1'
MAGIC = b'SWFM_CG1'


def causal_input_digest(raw):
    digest = hashlib.sha256()
    for key in ('history_occ', 'history_observed', 'history_poses', 'future_poses'):
        value = np.ascontiguousarray(raw[key])
        digest.update(key.encode()); digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode()); digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def array_bytes(value):
    seen = set()
    def size(v):
        if id(v) in seen: return 0
        seen.add(id(v))
        if isinstance(v, np.ndarray):
            # A view can retain an entire six-frame history allocation. Count
            # the backing allocation once, not just current/previous slices.
            if isinstance(v.base, np.ndarray): return size(v.base)
            return v.nbytes
        if isinstance(v, dict): return sum(size(a) for a in v.values())
        if isinstance(v, (list, tuple)): return sum(size(a) for a in v)
        if hasattr(v, '__dict__'): return size(vars(v))
        return 0
    return size(value)


class CausalGeometryCache:
    _roots_lock = Lock()
    _root_usage = {}

    def __init__(self, root, namespace, *, max_bytes=4*2**30, ram_bytes=128*2**20, reserve_bytes=2**30,
                 compression_level=1):
        if min(max_bytes, ram_bytes, reserve_bytes) < 0: raise ValueError('negative geometry cache budget')
        if not isinstance(compression_level,int) or not 0 <= compression_level <= 9:
            raise ValueError('zlib compression_level must be an integer in [0,9]')
        self.namespace = hashlib.sha256((PROTOCOL+namespace).encode()).hexdigest()
        self.cache_root = Path(root).resolve()
        self.root = self.cache_root/self.namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit, self.ram_limit, self.reserve = int(max_bytes), int(ram_bytes), int(reserve_bytes)
        self.compression_level=int(compression_level)
        self.lock = Lock(); self.rows = OrderedDict(); self.ram_used = 0
        # One quota across ALL provenance namespaces, not 16 GiB per config.
        # Shared admission accounting covers concurrent workers/cache instances
        # in this training process. Do not run two writers against the same root.
        with self._roots_lock:
            if self.cache_root not in self._root_usage:
                self._root_usage[self.cache_root] = sum(p.stat().st_size for p in self.cache_root.glob('*/*.cgc')
                    if len(p.stem) == 64 and all(c in '0123456789abcdef' for c in p.stem))
        self.hits = self.misses = self.writes = self.skipped_writes = 0
        self.writer = None; self.pending = set(); self.writer_slots = BoundedSemaphore(8)
        self.writer_error = None; self.closed = False; self.write_seconds = 0.

    def _remember(self, key, value):
        size = array_bytes(value)
        if size > self.ram_limit: return
        if key in self.rows:
            _, old = self.rows.pop(key); self.ram_used -= old
        while self.rows and self.ram_used+size > self.ram_limit:
            _, (_, old) = self.rows.popitem(last=False); self.ram_used -= old
        self.rows[key] = (value, size); self.ram_used += size

    def _address(self, key, raw):
        causal_sha = causal_input_digest(raw)
        name = hashlib.sha256((repr(tuple(key))+causal_sha).encode()).hexdigest()
        return name, causal_sha

    def _check_error(self):
        if self.writer_error is not None:
            raise RuntimeError('causal geometry background writer failed') from self.writer_error

    def require(self, key, raw):
        """Read an existing verified artifact; never build or write on a miss."""
        name, causal_sha = self._address(key, raw)
        path = self.root/(name+'.cgc')
        with self.lock:
            self._check_error()
            if name in self.rows:
                self.rows.move_to_end(name); self.hits += 1
                return self.rows[name][0]
        if not path.is_file():
            with self.lock: self.misses += 1
            raise FileNotFoundError(
                f'required causal geometry cache miss: key={tuple(key)} path={path}')
        blob = path.read_bytes()
        if (len(blob) < len(MAGIC)+32 or not blob.startswith(MAGIC)
                or hashlib.sha256(blob[len(MAGIC)+32:]).digest() != blob[len(MAGIC):len(MAGIC)+32]):
            raise RuntimeError(f'corrupt causal geometry cache: {path}')
        try:
            payload = pickle.loads(zlib.decompress(blob[len(MAGIC)+32:]))
        except Exception as exc:
            raise RuntimeError(f'invalid causal geometry cache: {path}') from exc
        if (payload.get('namespace') != self.namespace or payload.get('causal_sha') != causal_sha
                or tuple(payload.get('key', ())) != tuple(key)):
            raise RuntimeError(f'causal geometry cache provenance mismatch: {path}')
        value = payload['geometry']
        with self.lock:
            self.hits += 1; self._remember(name, value)
        return value

    def get_or_build(self, key, raw, builder, *, defer_write=False):
        name, causal_sha = self._address(key, raw)
        path = self.root/(name+'.cgc')
        with self.lock:
            self._check_error()
            if name in self.rows:
                self.rows.move_to_end(name); self.hits += 1
                return self.rows[name][0], True
        # No shared cache lock during disk read/decompression or compression.
        # Atomic replace means readers only observe a complete artifact.
        if path.is_file():
            blob = path.read_bytes()
            if (len(blob) < len(MAGIC)+32 or not blob.startswith(MAGIC)
                    or hashlib.sha256(blob[len(MAGIC)+32:]).digest() != blob[len(MAGIC):len(MAGIC)+32]):
                raise RuntimeError(f'corrupt causal geometry cache: {path}')
            try: payload = pickle.loads(zlib.decompress(blob[len(MAGIC)+32:]))
            except Exception as exc: raise RuntimeError(f'invalid causal geometry cache: {path}') from exc
            if (payload.get('namespace') != self.namespace or payload.get('causal_sha') != causal_sha
                    or tuple(payload.get('key', ())) != tuple(key)):
                raise RuntimeError(f'causal geometry cache provenance mismatch: {path}')
            with self.lock:
                self.hits += 1; value = payload['geometry']; self._remember(name, value)
            return value, True
        with self.lock:
            self.misses += 1
        value = builder()
        # Cold training uses the ORIGINAL GPU Strong path on the caller. The
        # worker returns causal history only; incomplete geometry is NOT stored.
        if not defer_write: self._store(name, causal_sha, key, value)
        return value, False

    def store(self, key, raw, value, *, asynchronous=False):
        """Complete immutable CPU geometry; no tensors/learned state/GT allowed."""
        name, causal_sha = self._address(key, raw)
        with self.lock:
            self._check_error()
            if self.closed: raise RuntimeError('causal geometry cache is closed')
            self._remember(name, value)
            if not asynchronous: pass
            elif name in self.pending: return False
            elif not self.writer_slots.acquire(blocking=False):
                # Bounded memory: never stall GPU training waiting for a slow
                # NAS writer. The exact result remains usable in this batch.
                self.skipped_writes += 1; return False
            else:
                self.pending.add(name)
                if self.writer is None: self.writer = ThreadPoolExecutor(max_workers=1)
                self.writer.submit(self._background_store, name, causal_sha, tuple(key), value)
                return True
        self._store(name, causal_sha, key, value)
        return True

    def _background_store(self, name, causal_sha, key, value):
        try: self._store(name, causal_sha, key, value)
        except Exception as exc:
            with self.lock: self.writer_error = exc
        finally:
            with self.lock: self.pending.discard(name)
            self.writer_slots.release()

    def _store(self, name, causal_sha, key, value):
        path = self.root/(name+'.cgc')
        with self.lock:
            self._remember(name, value)
            if path.is_file(): return
            if not self.limit or self.disk_used >= self.limit:
                self.skipped_writes += 1; return
        started = time.perf_counter()
        packed = zlib.compress(pickle.dumps({'namespace': self.namespace, 'key': tuple(key),
            'causal_sha': causal_sha, 'geometry': value}, protocol=5), level=self.compression_level)
        blob = MAGIC+hashlib.sha256(packed).digest()+packed
        written = skipped = disabled = False
        with self._roots_lock:
            if path.is_file(): return
            if self._root_usage[self.cache_root]+len(blob) > self.limit or shutil.disk_usage(self.root).free < len(blob)+self.reserve:
                skipped = True
            else:
                temporary = self.root/(name+'.tmp.'+uuid.uuid4().hex)
                try:
                    with temporary.open('xb') as handle: handle.write(blob)
                    os.replace(temporary, path)
                    self._root_usage[self.cache_root] += len(blob); written = True
                except OSError as exc:
                    # No scientific fallback: recomputed geometry is exact.
                    disabled = skipped = True
                    warnings.warn(f'geometry cache admission disabled: {exc}', RuntimeWarning)
                finally:
                    if temporary.is_file(): temporary.unlink()
        with self.lock:
            if disabled: self.limit = 0
            self.writes += int(written); self.skipped_writes += int(skipped)
            self.write_seconds += time.perf_counter()-started

    def close(self):
        with self.lock:
            self.closed = True; writer = self.writer
        if writer is not None: writer.shutdown(wait=True)
        with self.lock: self._check_error()

    def flush(self):
        """Drain preceding asynchronous writes without closing the reusable cache."""
        with self.lock:
            self._check_error(); writer = self.writer; closed = self.closed
        if writer is not None and not closed: writer.submit(lambda: None).result()
        with self.lock: self._check_error()

    def is_persisted(self, key, raw):
        """Admission audit only; get_or_build remains the integrity-checked reader."""
        name, _ = self._address(key, raw)
        return (self.root/(name+'.cgc')).is_file()

    @property
    def disk_used(self):
        with self._roots_lock: return self._root_usage[self.cache_root]

    def stats(self):
        with self.lock:
            self._check_error()
            return dict(hits=self.hits, misses=self.misses, writes=self.writes,
                skipped_writes=self.skipped_writes, disk_mib=self.disk_used/2**20,
                disk_limit_mib=self.limit/2**20, ram_mib=self.ram_used/2**20,
                pending_writes=len(self.pending), background_write_seconds=self.write_seconds,
                directory=str(self.root), namespace=self.namespace,
                compression_level=self.compression_level)
