"""Cache contracts for MSP-routed anchor-centered Sparse World Models."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

MSP_WM_CACHE_VERSION = "msp_topk_anchor_wm_v1"  # legacy P0-F3
MSP_WM_CACHE_VERSION_V2 = "msp_topk_strong_w2det_fullctx_wm_v2"
SUPPORTED_VERSIONS = {MSP_WM_CACHE_VERSION, MSP_WM_CACHE_VERSION_V2}

COMMON_REQUIRED = (
    "sample_id",
    "scene_name",
    "anchor_future_latent",
    "gt_future_latent",
    "window_origins",
    "window_valid",
    "trajectory",
)


def _history_key(sample):
    if "full_history_latent" in sample:
        return "full_history_latent"
    if "moving_history_latent" in sample:
        return "moving_history_latent"
    raise KeyError("MSP-WM cache requires full_history_latent or moving_history_latent")


def validate_msp_wm_sample(
    sample,
    *,
    topk=None,
    latent_hw=(50, 50),
    trajectory_length=12,
    require_full_history=False,
    require_write_support=False,
):
    missing = [k for k in COMMON_REQUIRED if k not in sample]
    if missing:
        raise KeyError(f"MSP-WM cache missing keys {missing}")
    hkey = _history_key(sample)
    if require_full_history and hkey != "full_history_latent":
        raise KeyError("P0-F4 requires full_history_latent")

    hist = sample[hkey]
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

    if require_write_support and "msp_write_support_latent" not in sample:
        raise KeyError("P0-F4 requires msp_write_support_latent")
    if "msp_write_support_latent" in sample:
        ws = sample["msp_write_support_latent"]
        expected = (int(anc.shape[0]), *tuple(latent_hw))
        if not torch.is_tensor(ws) or tuple(ws.shape) != expected:
            raise ValueError(f"msp_write_support_latent must be {expected}")
        if ws.dtype != torch.bool:
            raise ValueError("msp_write_support_latent must be bool")
    return True


def save_msp_wm_shards(output_dir, samples, *, shard_size=64, metadata=None, version=MSP_WM_CACHE_VERSION):
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported MSP-WM version {version}")
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
        "version": version,
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
        self.version = self.index.get("version")
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported MSP-WM cache version {self.version}")
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
        is_v2 = self.version == MSP_WM_CACHE_VERSION_V2
        validate_msp_wm_sample(
            s,
            topk=self.metadata.get("topk"),
            latent_hw=tuple(self.metadata.get("latent_hw", [50, 50])),
            trajectory_length=int(self.metadata.get("trajectory_length", 12)),
            require_full_history=is_v2,
            require_write_support=is_v2,
        )
        return s


def collate_msp_wm(batch):
    if not batch:
        raise ValueError("cannot collate empty MSP-WM batch")
    hkey = _history_key(batch[0])
    if any(_history_key(s) != hkey for s in batch):
        raise ValueError("mixed MSP-WM history contracts in one batch")
    tensor_keys = [
        hkey,
        "anchor_future_latent",
        "gt_future_latent",
        "window_origins",
        "window_valid",
        "trajectory",
    ]
    if "msp_write_support_latent" in batch[0]:
        tensor_keys.append("msp_write_support_latent")
    out = {k: torch.stack([s[k] for s in batch]) for k in tensor_keys}
    out["sample_id"] = [str(s["sample_id"]) for s in batch]
    out["scene_name"] = [str(s["scene_name"]) for s in batch]
    return out
