"""Cache contract for Top-K MSP-routed anchor-centered Sparse World Model."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

MSP_WM_CACHE_VERSION = "msp_topk_anchor_wm_v1"
REQUIRED = (
    "sample_id",
    "scene_name",
    "moving_history_latent",
    "anchor_future_latent",
    "gt_future_latent",
    "window_origins",
    "window_valid",
    "trajectory",
)


def validate_msp_wm_sample(sample, *, topk=None, latent_hw=(50, 50), trajectory_length=12):
    missing = [k for k in REQUIRED if k not in sample]
    if missing:
        raise KeyError(f"MSP-WM cache missing keys {missing}")
    hist = sample["moving_history_latent"]
    anc = sample["anchor_future_latent"]
    tgt = sample["gt_future_latent"]
    if not all(torch.is_tensor(x) and x.ndim == 4 for x in (hist, anc, tgt)):
        raise ValueError("history/anchor/target latents must be [T,C,H,W]")
    if anc.shape != tgt.shape:
        raise ValueError("anchor and GT future latent shapes must match")
    if hist.shape[1:] != anc.shape[1:]:
        raise ValueError("history/future latent C,H,W must match")
    if tuple(anc.shape[-2:]) != tuple(latent_hw):
        raise ValueError(f"latent H,W must be {tuple(latent_hw)}")
    origins = sample["window_origins"]
    valid = sample["window_valid"]
    if not torch.is_tensor(origins) or origins.ndim != 2 or origins.shape[1] != 2:
        raise ValueError("window_origins must be [K,2]")
    if not torch.is_tensor(valid) or tuple(valid.shape) != (origins.shape[0],):
        raise ValueError("window_valid must be [K]")
    if topk is not None and int(origins.shape[0]) != int(topk):
        raise ValueError(f"cache expected Top-{topk}, got K={origins.shape[0]}")
    traj = sample["trajectory"]
    if not torch.is_tensor(traj) or tuple(traj.shape) != (int(trajectory_length), 2):
        raise ValueError(f"trajectory must be [{trajectory_length},2]")
    return True


def save_msp_wm_shards(output_dir, samples, *, shard_size=64, metadata=None):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    entries, shard = [], []
    shard_id = 0

    def flush(items, sid):
        if not items:
            return
        name = f"shard_{sid:05d}.pt"
        torch.save(items, root / name)
        for j, s in enumerate(items):
            entries.append({
                "shard": name,
                "index": j,
                "sample_id": str(s["sample_id"]),
                "scene_name": str(s["scene_name"]),
            })

    for sample in samples:
        validate_msp_wm_sample(sample)
        shard.append(sample)
        if len(shard) >= int(shard_size):
            flush(shard, shard_id)
            shard, shard_id = [], shard_id + 1
    flush(shard, shard_id)
    index = {
        "version": MSP_WM_CACHE_VERSION,
        "metadata": metadata or {},
        "num_samples": len(entries),
        "entries": entries,
    }
    (root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


class MSPWorldModelCacheDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if self.index.get("version") != MSP_WM_CACHE_VERSION:
            raise ValueError(f"unsupported MSP-WM cache version {self.index.get('version')}")
        self.metadata = self.index.get("metadata", {})
        self.entries = self.index.get("entries", [])
        if not self.entries:
            raise RuntimeError("empty MSP-WM cache")
        self._shard_name = None
        self._shard = None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[int(idx)]
        if e["shard"] != self._shard_name:
            self._shard = torch.load(self.root / e["shard"], map_location="cpu", weights_only=False)
            self._shard_name = e["shard"]
        s = self._shard[e["index"]]
        validate_msp_wm_sample(
            s,
            topk=self.metadata.get("topk"),
            latent_hw=tuple(self.metadata.get("latent_hw", [50, 50])),
            trajectory_length=int(self.metadata.get("trajectory_length", 12)),
        )
        return s


def collate_msp_wm(batch):
    tensor_keys = (
        "moving_history_latent", "anchor_future_latent", "gt_future_latent",
        "window_origins", "window_valid", "trajectory",
    )
    out = {k: torch.stack([s[k] for s in batch]) for k in tensor_keys}
    out["sample_id"] = [str(s["sample_id"]) for s in batch]
    out["scene_name"] = [str(s["scene_name"]) for s in batch]
    return out
