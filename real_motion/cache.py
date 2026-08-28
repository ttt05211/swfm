"""Versioned, sharded latent-cache contract for Real-Motion OccFM."""
from pathlib import Path
import json
import torch

CACHE_VERSION = "real_motion_v4"
LEGACY_VERSIONS = ("real_motion_v1", "real_motion_v2", "real_motion_v3")
REQUIRED_KEYS = (
    "moving_history_latent",   # [H,C,50,50]
    "future_dynamic_target_latent",  # [F,C,50,50], causal-support training target
    "static_future_latent",    # [F,C,50,50]
    "kta_future_latent",       # [F,C,50,50]
    "generation_support",      # [F,50,50] bool
)


def canonicalize_sample(sample):
    """Upgrade the old ambiguous target key without changing tensor values."""
    if "future_dynamic_target_latent" not in sample and "future_moving_latent" in sample:
        sample = dict(sample)
        sample["future_dynamic_target_latent"] = sample["future_moving_latent"]
    return sample


def validate_sample(sample):
    sample = canonicalize_sample(sample)
    missing = [k for k in REQUIRED_KEYS if k not in sample]
    if missing:
        raise KeyError(f"cache missing keys: {missing}")
    mh = sample["moving_history_latent"]
    fm = sample["future_dynamic_target_latent"]
    st = sample["static_future_latent"]
    kt = sample["kta_future_latent"]
    ms = sample["generation_support"]
    for name, x in [
        ("moving_history_latent", mh), ("future_dynamic_target_latent", fm),
        ("static_future_latent", st), ("kta_future_latent", kt),
    ]:
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError(f"{name} must be tensor [T,C,H,W]")
    if fm.shape != st.shape or fm.shape != kt.shape:
        raise ValueError("future moving/static/KTA latent shapes must match")
    if tuple(ms.shape) != (fm.shape[0], fm.shape[-2], fm.shape[-1]):
        raise ValueError("generation_support must align with future latent")
    if mh.shape[1:] != fm.shape[1:]:
        raise ValueError("history/future latent C,H,W must match")
    if "planning_support" in sample:
        ps = sample["planning_support"]
        if not torch.is_tensor(ps) or ps.ndim != 3:
            raise ValueError("planning_support must be [T,H,W]")
        if tuple(ps.shape[-2:]) != tuple(fm.shape[-2:]):
            raise ValueError("planning_support H,W must match latent map")
    return True


def save_cache(path, samples, metadata=None):
    """Legacy/small-cache helper. Full training should use sharded cache."""
    samples = [canonicalize_sample(s) for s in samples]
    for s in samples:
        validate_sample(s)
    torch.save({"version": CACHE_VERSION, "metadata": metadata or {}, "samples": samples}, path)


def load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") not in (CACHE_VERSION, *LEGACY_VERSIONS):
        raise ValueError(f"unsupported cache version: {payload.get('version')}")
    payload["samples"] = [canonicalize_sample(s) for s in payload["samples"]]
    for s in payload["samples"]:
        validate_sample(s)
    return payload


def save_sharded_cache(output_dir, samples, shard_size=256, metadata=None):
    """Write lazy-loadable shards plus index.json."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    entries, shard = [], []
    shard_id = 0

    def flush(items, sid):
        if not items:
            return
        for s in items:
            validate_sample(s)
        name = f"shard_{sid:05d}.pt"
        torch.save(items, root / name)
        for local_idx, sample in enumerate(items):
            entries.append({
                "shard": name,
                "index": local_idx,
                "sample_id": str(sample.get("sample_id", f"{sid}:{local_idx}")),
            })

    for sample in samples:
        shard.append(canonicalize_sample(sample))
        if len(shard) >= int(shard_size):
            flush(shard, shard_id)
            shard, shard_id = [], shard_id + 1
    flush(shard, shard_id)

    index = {
        "version": CACHE_VERSION,
        "metadata": metadata or {},
        "num_samples": len(entries),
        "entries": entries,
    }
    (root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


class ShardedCacheDataset(torch.utils.data.Dataset):
    """Lazy one-shard-at-a-time dataset; no tens-of-GB torch.load at startup."""
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if self.index.get("version") not in (CACHE_VERSION, *LEGACY_VERSIONS):
            raise ValueError(f"unsupported sharded cache version {self.index.get('version')}")
        self.entries = self.index["entries"]
        self._shard_name = None
        self._shard = None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        if entry["shard"] != self._shard_name:
            self._shard = torch.load(self.root / entry["shard"], map_location="cpu",
                                     weights_only=False)
            self._shard_name = entry["shard"]
        sample = canonicalize_sample(self._shard[entry["index"]])
        validate_sample(sample)
        return sample
