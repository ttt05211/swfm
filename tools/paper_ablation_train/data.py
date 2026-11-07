"""Only TRAIN is opened. Legacy geometry is read-only; new tracks stay in RAM."""
from collections import OrderedDict
from threading import RLock
import hashlib
import json
from pathlib import Path
from real_motion.causal_geometry_cache import CausalGeometryCache,PROTOCOL as CACHE_PROTOCOL,array_bytes,causal_input_digest
from real_motion.ccr_history_geometry import namespace,PROTOCOL as TRAIN_PROTOCOL
from tools.causal_forward_train.data import ForwardProvider,batches


class MemoryTrackCache:
    def __init__(self,ram_mib=128):
        self.limit=int(ram_mib*2**20);self.rows=OrderedDict();self.bytes=0;self.lock=RLock()
        self.hits=self.misses=0

    def get_or_build(self,key,raw,builder):
        address=(tuple(key),causal_input_digest(raw))
        # Association is CPU-only. One rank's bounded prefetch shares this lock;
        # prevent duplicate large builds instead of creating another disk bank.
        with self.lock:
            if address in self.rows:
                self.hits+=1;self.rows.move_to_end(address);return self.rows[address][0],True
            self.misses+=1;value=builder();size=array_bytes(value)
            if size<=self.limit:
                while self.rows and self.bytes+size>self.limit:
                    _,(_,old)=self.rows.popitem(last=False);self.bytes-=old
                self.rows[address]=(value,size);self.bytes+=size
            return value,False

    def close(self):
        with self.lock: self.rows.clear();self.bytes=0


class TrainProvider(ForwardProvider):
    def attach_ram_tracks(self,ram_mib): self.track_cache=MemoryTrackCache(ram_mib)
    def close_tracks(self): self.track_cache.close()


def attach_train_cache(provider,args,root,source):
    identity=namespace(provider,args,root)
    digest=hashlib.sha256((CACHE_PROTOCOL+identity).encode()).hexdigest()
    directory=Path(args.ccr_history_cache).resolve()/digest
    path=directory/'manifest.json'
    if not path.is_file(): raise RuntimeError('existing TRAIN geometry cache required; NO rebuild: '+str(path))
    m=json.loads(path.read_text(encoding='utf-8'))
    expected=dict(complete=True,namespace=digest,windows=20430,artifacts=20430,protocol=TRAIN_PROTOCOL,
        future_GT_cached=False,learned_outputs_cached=False,sampled_ids_cached=False,optimizer_or_rng_cached=False)
    if any(m.get(k)!=v for k,v in expected.items()): raise RuntimeError('TRAIN geometry cache manifest mismatch')
    provider.ccr_history_cache=CausalGeometryCache(args.ccr_history_cache,identity,max_bytes=0,
        ram_bytes=args.geometry_ram_mib*2**20,reserve_bytes=0,compression_level=6)
    provider.ccr_history_cache_source=source;provider.ccr_history_cache_mode='require'
    provider.ccr_val_history_cache=None;provider.ccr_val_history_cache_source=None
    provider.ccr_val_history_cache_mode='require'  # inherited guard, never open VAL.
    provider.attach_ram_tracks(args.track_ram_mib)
    return dict(train_cache_namespace=digest,persistent_writes=False,new_track_cache='bounded_RAM_only',
        track_ram_mib=args.track_ram_mib,VAL_opened=False,STC_opened=False)
