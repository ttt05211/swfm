from pathlib import Path
import math
import torch
from torch.utils.data import Dataset, Sampler
from .cache import load_cache, ShardedCacheDataset


class RealMotionCacheDataset(Dataset):
    """Auto-detect legacy single .pt or sharded cache directory."""
    def __init__(self, path):
        p=Path(path); self.sharded=p.is_dir()
        if self.sharded:
            self.backend=ShardedCacheDataset(p); self.payload=None; self.samples=None
            self.metadata=self.backend.index.get('metadata',{})
        else:
            self.payload=load_cache(p); self.samples=self.payload['samples']; self.backend=None
            self.metadata=self.payload.get('metadata',{})
    def __len__(self): return len(self.backend) if self.sharded else len(self.samples)
    def __getitem__(self,i): return self.backend[i] if self.sharded else self.samples[i]


def collate_real_motion(batch):
    keys=('moving_history_latent','future_dynamic_target_latent','static_future_latent','kta_future_latent','generation_support')
    out={k:torch.stack([x[k] for x in batch]) for k in keys}
    optional=('trajectory','confident_static_mask','gt_moving_support','planning_support','window_origins','window_valid','sample_id')
    for k in optional:
        if all(k in x for x in batch):
            out[k]=torch.stack([x[k] for x in batch]) if torch.is_tensor(batch[0][k]) else [x[k] for x in batch]
    return out


def _shard_groups(dataset: RealMotionCacheDataset):
    if not getattr(dataset,'sharded',False): raise ValueError('shard-aware sampler requires a sharded cache dataset')
    groups={}
    for idx,entry in enumerate(dataset.backend.entries): groups.setdefault(entry['shard'],[]).append(idx)
    return [(name,groups[name]) for name in sorted(groups)]


class ShardShuffleSampler(Sampler):
    """Single-process shard-local shuffle without random shard thrashing."""
    def __init__(self,dataset:RealMotionCacheDataset,seed=0):
        self.dataset=dataset;self.seed=int(seed);self.epoch=0;self.groups=_shard_groups(dataset)
    def __len__(self):return len(self.dataset)
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __iter__(self):
        g=torch.Generator();g.manual_seed(self.seed+self.epoch);self.epoch+=1
        shard_order=torch.randperm(len(self.groups),generator=g).tolist()
        for si in shard_order:
            _,group=self.groups[si];order=torch.randperm(len(group),generator=g).tolist()
            for j in order:yield group[j]


class DistributedShardSampler(Sampler):
    """Assign whole cache shards to DDP ranks, then shuffle locally.

    A normal DistributedSampler (or a strided global index stream) makes every
    rank load the same large `.pt` shard. Here each physical shard has exactly
    one owner rank per epoch. Shards are greedily balanced by sample count; all
    ranks are then padded *from their own local indices* to equal sampler
    lengths so DDP executes the same number of optimization steps.
    """
    def __init__(self,dataset:RealMotionCacheDataset,num_replicas,rank,shuffle=True,seed=0,drop_last=False):
        self.dataset=dataset;self.num_replicas=int(num_replicas);self.rank=int(rank);self.shuffle=bool(shuffle);self.seed=int(seed);self.drop_last=bool(drop_last);self.epoch=0;self.groups=_shard_groups(dataset)
        if self.num_replicas<=0 or not 0<=self.rank<self.num_replicas:raise ValueError('invalid DDP rank/world size')
        self._cached_epoch=None;self._cached_rows=None
    def set_epoch(self,epoch):self.epoch=int(epoch);self._cached_epoch=None;self._cached_rows=None
    def _rank_rows(self):
        if self._cached_epoch==self.epoch and self._cached_rows is not None:return self._cached_rows
        g=torch.Generator();g.manual_seed(self.seed+self.epoch)
        order=list(range(len(self.groups)))
        if self.shuffle:order=torch.randperm(len(order),generator=g).tolist()
        rows=[[] for _ in range(self.num_replicas)];loads=[0]*self.num_replicas
        for si in order:
            _,group=self.groups[si];owner=min(range(self.num_replicas),key=lambda r:(loads[r],r));local=list(group)
            if self.shuffle and len(local)>1:
                perm=torch.randperm(len(local),generator=g).tolist();local=[local[j] for j in perm]
            rows[owner].extend(local);loads[owner]+=len(local)
        target=(min(loads) if self.drop_last else max(loads)) if loads else 0
        for r,row in enumerate(rows):
            if self.drop_last:
                rows[r]=row[:target]
            elif len(row)<target:
                if not row:
                    global_indices=[idx for _,grp in self.groups for idx in grp]
                    if not global_indices:rows[r]=[];continue
                    row=[global_indices[r%len(global_indices)]]
                need=target-len(row);reps=int(math.ceil(need/len(row))) if need else 0;rows[r]=row+(row*reps)[:need]
        self._cached_epoch=self.epoch;self._cached_rows=rows;return rows
    def __len__(self):return len(self._rank_rows()[self.rank])
    def __iter__(self):return iter(list(self._rank_rows()[self.rank]))
