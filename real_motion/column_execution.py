"""Opt-in Local inference execution. No training, weights, GT or cross-window features.

Exact *byte-identical* patches may share only history encoding. Padding stays
patch-local. Query/context/source/attention/action heads still run for EVERY row.
Floating-point batch changes are guarded by the benchmark's byte/count gates.
CUDA graphs retain executable buffers, never a prediction cache.
"""
from collections import OrderedDict
from contextlib import contextmanager
import time
import numpy as np
import torch


class PatchMemory:
    def __init__(self, max_bytes=64*2**20):
        if type(max_bytes) is not int or max_bytes < 1: raise ValueError('positive patch cache budget required')
        self.limit = max_bytes; self.rows = OrderedDict(); self.bytes = self.peak = 0
        self.memory = self.invalid = None; self.capacity = 0; self.free = []
        self.key_bytes = self.storage_bytes = 0
        self.hits = self.misses = self.encoded = 0

    def encode(self, model, arrays, batch):
        # Full immutable bytes as dict keys: Python resolves hash collisions by
        # byte equality. Neither geometry IDs nor rounded learned poses suffice.
        history, flags = arrays['history'], arrays['flags']
        if isinstance(history,torch.Tensor) or isinstance(flags,torch.Tensor):
            return model.encode_history(batch['history'],batch['flags'])
        keys = [(history[i].tobytes(),flags[i].tobytes()) for i in range(len(history))]
        unique = {}; inverse = []
        for i,key in enumerate(keys):
            if key not in unique: unique[key] = i
        positions = {k:j for j,k in enumerate(unique)}
        inverse = [positions[k] for k in keys]
        if self.memory is not None and len(unique) > self.capacity:
            # Do not partially update/evict the cache or encode misses TWICE.
            self.hits += len(keys)-len(unique);self.misses += len(unique);self.encoded += len(unique)
            at=torch.as_tensor(list(unique.values()),device=batch['history'].device)
            x,mask=model.encode_history(batch['history'].index_select(0,at),batch['flags'].index_select(0,at))
            take=torch.as_tensor(inverse,device=at.device)
            return x.index_select(0,take),mask.index_select(0,take)
        wanted = set(unique); missing = []
        for key,i in unique.items():
            if key in self.rows:
                self.rows.move_to_end(key)
            else:
                missing.append(i)
        self.hits += len(keys)-len(missing); self.misses += len(missing)
        if missing:
            ids = torch.as_tensor(missing,device=batch['history'].device)
            memory, invalid = model.encode_history(batch['history'].index_select(0,ids),batch['flags'].index_select(0,ids))
            self.encoded += len(missing)
            if self.memory is None:
                size = sum(map(len,keys[0]))+memory[0].numel()*memory.element_size()+invalid[0].numel()*invalid.element_size()
                self.capacity = self.limit//size
                if self.capacity >= len(unique):
                    self.memory = memory.new_empty((self.capacity,*memory.shape[1:]))
                    self.invalid = invalid.new_empty((self.capacity,*invalid.shape[1:]))
                    self.storage_bytes = self.memory.numel()*memory.element_size()+self.invalid.numel()*invalid.element_size()
                    self.free = list(reversed(range(self.capacity)))
            if self.memory is None or len(unique) > self.capacity:
                # Oversize/one-byte cache: dedup within batch only, no query cap.
                at = torch.as_tensor(list(unique.values()),device=batch['history'].device)
                take = torch.as_tensor(inverse,device=batch['history'].device)
                return memory.index_select(0,take),invalid.index_select(0,take)
            slots=[]
            for i in missing:
                if not self.free:
                    while True:
                        old,slot=self.rows.popitem(last=False)
                        if old not in wanted:
                            self.key_bytes -= sum(map(len,old));self.free.append(slot);break
                        self.rows[old]=slot
                slot=self.free.pop();self.rows[keys[i]]=slot;slots.append(slot)
                self.key_bytes += sum(map(len,keys[i]))
            at=torch.as_tensor(slots,device=self.memory.device)
            # Two batched writes, NOT two tiny GPU clones per missed patch.
            self.memory.index_copy_(0,at,memory);self.invalid.index_copy_(0,at,invalid)
            self.bytes=self.storage_bytes+self.key_bytes;self.peak=max(self.peak,self.bytes)
        # Same query order, duplicates included. Decoder batch shape unchanged.
        take=torch.as_tensor([self.rows[k] for k in keys],device=self.memory.device)
        return self.memory.index_select(0,take),self.invalid.index_select(0,take)

    def audit(self):
        return dict(patch_cache_hits=self.hits,patch_cache_misses=self.misses,encoded_patches=self.encoded,
                    patch_cache_peak_bytes=self.peak,patch_cache_limit_bytes=self.limit)


class ColumnExecution:
    def __init__(self, model, *, graphs=False, reuse=False, max_graphs=2, patch_bytes=64*2**20):
        from .causal_column_model import CausalColumnModel
        from .joint_causal_columns import LinkedColumns
        if type(model) not in (CausalColumnModel,LinkedColumns): raise ValueError('execution backend is Local-only')
        self.model = model; self.graphs_enabled = bool(graphs); self.reuse = bool(reuse)
        self.max_graphs = max_graphs; self.patch_bytes = patch_bytes; self.graphs = {}; self.failures = []
        self.graph_build_seconds = 0.; self.replays = self.eager_calls = 0
        self.current_memory = None
        self.versions = self._signature()

    def _signature(self):
        return tuple((id(v),v._version,v.device,v.dtype,tuple(v.shape))
            for v in (*self.model.parameters(),*self.model.buffers()))

    def validate(self):
        if self.model.training or not torch.is_inference_mode_enabled(): raise RuntimeError('execution requires eval + inference_mode')
        now = self._signature()
        if now != self.versions: raise RuntimeError('weights/calibration changed inside inference execution session')

    def function(self, batch, legal, *, memory=None, invalid=None):
        # CUDA capture must not retain casts from an outer autocast weight
        # cache that is freed when the per-chunk context exits. Same BF16 math;
        # create capture-owned casts instead of replaying dangling pointers.
        with torch.autocast(device_type=legal.device.type,dtype=torch.bfloat16,
                enabled=legal.device.type == 'cuda',cache_enabled=False):
            return self._function(batch,legal,memory=memory,invalid=invalid)

    def _function(self, batch, legal, *, memory=None, invalid=None):
        from .joint_causal_columns import LinkedColumns
        source = batch.get('source_features')
        if memory is None:
            b = dict(batch)
            if type(self.model) is LinkedColumns: b['validate_source'] = False
            g,r = self.model(**b)
        else:
            extra = self.model.source_projection(source) if type(self.model) is LinkedColumns else None
            g,r = self.model.logits_from_history(memory,invalid,**{k:batch[k] for k in ('base','fallback','context','kind','classes')},query_extra=extra)
        p = self.model.calibrated_probabilities(g,r,batch['kind'],legal,validate=False)
        finite = (torch.isfinite(g).all() & torch.isfinite(r).all() & torch.isfinite(p).all())
        if source is not None: finite = finite & torch.isfinite(source).all()
        return p,finite

    def run(self, batch, legal, *, memory=None, invalid=None):
        self.validate()
        device = legal.device
        key = (len(legal),memory is not None,tuple((k,v.dtype,tuple(v.shape)) for k,v in batch.items()))
        if not self.graphs_enabled or device.type != 'cuda':
            self.eager_calls += 1; return self.function(batch,legal,memory=memory,invalid=invalid)
        if key not in self.graphs and len(self.graphs) < self.max_graphs:
            tick = time.perf_counter()
            static = sl = sm = si = graph = result = None
            try:
                # Buffer allocation itself can OOM, before capture starts.
                # Keep it under the same eager-fallback guard as capture.
                static = {k:v.clone() for k,v in batch.items()}; sl = legal.clone()
                sm = None if memory is None else memory.clone(); si = None if invalid is None else invalid.clone()
                stream = torch.cuda.Stream(device=device); stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3): self.function(static,sl,memory=sm,invalid=si)
                torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): result = self.function(static,sl,memory=sm,invalid=si)
                self.graphs[key] = (graph,static,sl,sm,si,result)
            except RuntimeError as exc:
                # Never silently call an unbuilt graph. Honest diagnostic with
                # unchanged eager math; OOM/capture-unsupported fall back.
                self.graphs[key] = None; self.failures.append(str(exc)[:240])
                static = sl = sm = si = graph = result = None
            self.graph_build_seconds += time.perf_counter()-tick
        cached = self.graphs.get(key)
        if cached is None:
            self.eager_calls += 1; return self.function(batch,legal,memory=memory,invalid=invalid)
        graph,static,sl,sm,si,result = cached
        for k,v in batch.items(): static[k].copy_(v)
        sl.copy_(legal)
        if sm is not None: sm.copy_(memory); si.copy_(invalid)
        graph.replay(); self.replays += 1
        return result

    def audit(self):
        return dict(cuda_graph_replays=self.replays,execution_eager_calls=self.eager_calls,
                    cuda_graph_build_seconds=self.graph_build_seconds,cuda_graph_failures=list(self.failures))

    @contextmanager
    def patch_window(self):
        if self.current_memory is not None: raise RuntimeError('nested patch window would mix inference populations')
        self.current_memory = PatchMemory(self.patch_bytes) if self.reuse else None
        try: yield self.current_memory
        finally: self.current_memory = None


@contextmanager
def execution_session(model, *, graphs=False, reuse=False):
    old = getattr(model,'column_execution_session',None)
    session = ColumnExecution(model,graphs=graphs,reuse=reuse)
    model.column_execution_session = session
    try: yield session
    finally:
        if old is None: del model.column_execution_session
        else: model.column_execution_session = old
        session.graphs.clear()


class AsyncProbabilityReadback:
    """D2H-only overlap; no packed uploads. One final wait per horizon.

Graphs overwrite their output buffers: clone on the compute stream before
enqueueing a readback. record_stream alone would NOT prevent graph overwrite.
"""
    def __init__(self, shape, device, *, max_bytes=8*2**20):
        self.device = torch.device(device); self.offset = self.transfers = 0; self.stream = None
        self.host = None; self.parts = []
        size = int(np.prod(shape))*4
        if self.device.type == 'cuda' and 0 < size <= max_bytes:
            self.host = torch.empty(shape,dtype=torch.float32,pin_memory=True)
            self.stream = torch.cuda.Stream(device=self.device)

    def append(self, value):
        if self.stream is None: self.parts.append(value.cpu().numpy()); self.transfers += 1; return
        stable = value.clone()  # includes graph output lifetime protection
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            self.host[self.offset:self.offset+len(value)].copy_(stable,non_blocking=True)
            stable.record_stream(self.stream)
        self.offset += len(value); self.transfers += 1

    def result(self, empty_shape):
        if self.stream is None: return np.concatenate(self.parts) if self.parts else np.empty(empty_shape,np.float32)
        self.stream.synchronize()
        if self.offset != len(self.host): raise RuntimeError('incomplete async readback')
        return self.host.numpy()

    def close(self):
        if self.stream is not None: self.stream.synchronize()

    @property
    def peak_bytes(self): return 0 if self.host is None else self.host.numel()*4
