from pathlib import Path
import math
import torch
from torch.utils.data import Dataset, Sampler
from .cache import load_cache, ShardedCacheDataset


class RealMotionCacheDataset(Dataset):
    """Auto-detect legacy single .pt or sharded cache directory."""
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
        "planning_support", "window_origins", "window_valid", "sample_id",
    )
    for k in optional:
        if all(k in x for x in batch):
            if torch.is_tensor(batch[0][k]):
                out[k] = torch.stack([x[k] for x in batch])
            else:
                out[k] = [x[k] for x in batch]
    return out


def _shard_groups(dataset: RealMotionCacheDataset):
    if not getattr(dataset, "sharded", False):
        raise ValueError("shard-aware sampler requires a sharded cache dataset")
    groups = {}
    for idx, entry in enumerate(dataset.backend.entries):
        groups.setdefault(entry["shard"], []).append(idx)
    return [v for _, v in sorted(groups.items())]


class ShardShuffleSampler(Sampler):
    """Single-process shard-local shuffle without random shard thrashing."""
    def __init__(self, dataset: RealMotionCacheDataset, seed=0):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.groups = _shard_groups(dataset)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_order = torch.randperm(len(self.groups), generator=g).tolist()
        for si in shard_order:
            group = self.groups[si]
            order = torch.randperm(len(group), generator=g).tolist()
            for j in order:
                yield group[j]


class DistributedShardSampler(Sampler):
    """DDP sampler that preserves shard locality on every rank."""
    def __init__(self, dataset: RealMotionCacheDataset, num_replicas, rank,
                 shuffle=True, seed=0, drop_last=False):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("invalid DDP rank/world size")
        self.groups = _shard_groups(dataset)
        n = len(dataset)
        if self.drop_last:
            self.num_samples = n // self.num_replicas
        else:
            self.num_samples = int(math.ceil(n / self.num_replicas)) if n else 0
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _ordered_indices(self):
        if not self.shuffle:
            return [idx for group in self.groups for idx in group]
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(self.groups), generator=g).tolist()
        out = []
        for si in shard_order:
            group = self.groups[si]
            order = torch.randperm(len(group), generator=g).tolist()
            out.extend(group[j] for j in order)
        return out

    def __iter__(self):
        indices = self._ordered_indices()
        if self.drop_last:
            indices = indices[:self.total_size]
        elif self.total_size > len(indices):
            pad = self.total_size - len(indices)
            if indices:
                repeats = int(math.ceil(pad / len(indices)))
                indices += (indices * repeats)[:pad]
        if len(indices) != self.total_size:
            raise RuntimeError("distributed shard sampler size invariant failed")
        local = indices[self.rank:self.total_size:self.num_replicas]
        if len(local) != self.num_samples:
            raise RuntimeError("distributed shard sampler rank length mismatch")
        return iter(local)
