"""Explicit lifetime of exact canonical CPU execution; no learned-state cache."""
from concurrent.futures import ThreadPoolExecutor
from .canonical_causal_repair import build_canonical_evidence, map_canonical_evidence
from .native_column_cpu import prepare_native, get_prepared_native


class CanonicalCpuExecution:
    def __init__(self,mode='numpy',workers=4,build_dir=None):
        if mode not in ('numpy','native','native_parallel') or not 1<=workers<=32:
            raise ValueError('CCR execution requires numpy/native/native_parallel and 1..32 workers')
        self.mode=mode;self.kernels=None;self.pool=None
        if mode!='numpy':
            prepare_native(build_dir);self.kernels=get_prepared_native()
        if mode=='native_parallel':self.pool=ThreadPoolExecutor(max_workers=workers)

    def build(self,prepared,grid):
        return build_canonical_evidence(prepared,grid,kernels=self.kernels,executor=self.pool)

    def map(self,evidence,prepared,grid):
        return map_canonical_evidence(evidence,prepared,grid,kernels=self.kernels,executor=self.pool)

    def close(self):
        if self.pool is not None:self.pool.shutdown(wait=True);self.pool=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
