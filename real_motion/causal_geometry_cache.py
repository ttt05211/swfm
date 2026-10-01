"""Bounded, integrity-checked compressed CPU geometry, never learned state or GT.

This is an implementation cache of local trusted artifacts. Like Torch's local
checkpoints its pickle payload must not be supplied from an untrusted party.
Raw causal inputs are hashed on every lookup; corrupt artifacts fail closed.
The cache never evicts/deletes existing entries: when full it stops admitting.
"""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import pickle
import shutil
from threading import Lock
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
        if isinstance(v, np.ndarray): return v.nbytes
        if isinstance(v, dict): return sum(size(a) for a in v.values())
        if isinstance(v, (list, tuple)): return sum(size(a) for a in v)
        if hasattr(v, '__dict__'): return size(vars(v))
        return 0
    return size(value)


class CausalGeometryCache:
    _roots_lock = Lock()
    _root_usage = {}

    def __init__(self, root, namespace, *, max_bytes=4*2**30, ram_bytes=128*2**20, reserve_bytes=2**30):
        if min(max_bytes, ram_bytes, reserve_bytes) < 0: raise ValueError('negative geometry cache budget')
        self.namespace = hashlib.sha256((PROTOCOL+namespace).encode()).hexdigest()
        self.cache_root = Path(root).resolve()
        self.root = self.cache_root/self.namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit, self.ram_limit, self.reserve = int(max_bytes), int(ram_bytes), int(reserve_bytes)
        self.lock = Lock(); self.rows = OrderedDict(); self.ram_used = 0
        # One quota across ALL provenance namespaces, not 16 GiB per config.
        # Shared admission accounting covers concurrent workers/cache instances
        # in this training process. Do not run two writers against the same root.
        with self._roots_lock:
            if self.cache_root not in self._root_usage:
                self._root_usage[self.cache_root] = sum(p.stat().st_size for p in self.cache_root.glob('*/*.cgc')
                    if len(p.stem) == 64 and all(c in '0123456789abcdef' for c in p.stem))
        self.hits = self.misses = self.writes = self.skipped_writes = 0

    def _remember(self, key, value):
        size = array_bytes(value)
        if size > self.ram_limit: return
        if key in self.rows:
            _, old = self.rows.pop(key); self.ram_used -= old
        while self.rows and self.ram_used+size > self.ram_limit:
            _, (_, old) = self.rows.popitem(last=False); self.ram_used -= old
        self.rows[key] = (value, size); self.ram_used += size

    def get_or_build(self, key, raw, builder):
        causal_sha = causal_input_digest(raw)
        name = hashlib.sha256((repr(tuple(key))+causal_sha).encode()).hexdigest()
        path = self.root/(name+'.cgc')
        with self.lock:
            if name in self.rows:
                self.rows.move_to_end(name); self.hits += 1
                return self.rows[name][0], True
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
                self.hits += 1; value = payload['geometry']; self._remember(name, value)
                return value, True
            self.misses += 1
        value = builder()
        with self.lock:
            self._remember(name, value)
            if path.is_file(): return value, False
            if not self.limit or self.disk_used >= self.limit:
                self.skipped_writes += 1; return value, False
            packed = zlib.compress(pickle.dumps({'namespace': self.namespace, 'key': tuple(key),
                'causal_sha': causal_sha, 'geometry': value}, protocol=5), level=1)
            blob = MAGIC+hashlib.sha256(packed).digest()+packed
            with self._roots_lock:
                if self._root_usage[self.cache_root]+len(blob) > self.limit or shutil.disk_usage(self.root).free < len(blob)+self.reserve:
                    self.skipped_writes += 1; return value, False
                temporary = self.root/(name+'.tmp.'+uuid.uuid4().hex)
                try:
                    with temporary.open('xb') as handle: handle.write(blob)
                    os.replace(temporary, path)
                    self._root_usage[self.cache_root] += len(blob); self.writes += 1
                except OSError as exc:
                    # No scientific fallback: recomputed geometry is exact.
                    self.limit = 0; self.skipped_writes += 1
                    warnings.warn(f'geometry cache admission disabled: {exc}', RuntimeWarning)
                finally:
                    if temporary.is_file(): temporary.unlink()
        return value, False

    @property
    def disk_used(self):
        with self._roots_lock: return self._root_usage[self.cache_root]

    def stats(self):
        with self.lock:
            return dict(hits=self.hits, misses=self.misses, writes=self.writes,
                skipped_writes=self.skipped_writes, disk_mib=self.disk_used/2**20,
                disk_limit_mib=self.limit/2**20, ram_mib=self.ram_used/2**20)
