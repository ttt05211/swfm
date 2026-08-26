from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler
from .cache import load_cache, ShardedCacheDataset


class RealMotionCacheDataset(Dataset):
    """Auto-detect legacy single .pt or v2 sharded directory."""
    def __init__(self, path):
        p = Path(path)
        self.sharded = p.is_dir()
        if self.sharded:
            self.backend = ShardedCacheDataset(p)
            self.payload = None
            self.samples = None
            self.metadata = self.backend.index.get("metadata", {})
        else:
            self.payload = load_cache(p)
            self.samples = self.payload["samples"]
            self.backend = None
            self.metadata = self.payload.get("metadata", {})

    def __len__(self):
        return len(self.backend) if self.sharded else len(self.samples)

    def __getitem__(self, i):
        return self.backend[i] if self.sharded else self.samples[i]


def collate_real_motion(batch):
    keys = (
        "moving_history_latent", "future_dynamic_target_latent", "static_future_latent",
        "kta_future_latent", "generation_support",
    )
    out = {k: torch.stack([x[k] for x in batch]) for k in keys}
    optional = (
        "trajectory", "confident_static_mask", "gt_moving_support",
        "planning_support", "sample_id",
    )
    for k in optional:
        if all(k in x for x in batch):
            if torch.is_tensor(batch[0][k]):
                out[k] = torch.stack([x[k] for x in batch])
            else:
                out[k] = [x[k] for x in batch]
    return out


class ShardShuffleSampler(Sampler):
    """Shuffle shards and samples *within* shards without random shard thrashing.

    A globally shuffled index stream makes a one-shard lazy cache reload a large
    `.pt` for almost every sample. This sampler keeps stochastic training order
    while consuming one shard contiguously before moving to another.
    """
    def __init__(self, dataset: RealMotionCacheDataset, seed=0):
        if not getattr(dataset, "sharded", False):
            raise ValueError("ShardShuffleSampler requires a sharded cache dataset")
        self.dataset=dataset
        self.seed=int(seed)
        self.epoch=0
        groups={}
        for idx,entry in enumerate(dataset.backend.entries):
            groups.setdefault(entry["shard"],[]).append(idx)
        self.groups=[v for _,v in sorted(groups.items())]

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        g=torch.Generator(); g.manual_seed(self.seed+self.epoch)
        self.epoch += 1
        shard_order=torch.randperm(len(self.groups),generator=g).tolist()
        for si in shard_order:
            group=self.groups[si]
            order=torch.randperm(len(group),generator=g).tolist()
            for j in order:
                yield group[j]
