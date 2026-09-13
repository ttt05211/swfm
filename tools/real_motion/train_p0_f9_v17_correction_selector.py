#!/usr/bin/env python3
"""Train a tiny causal source selector on frozen KTA-vs-V17 utility labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from real_motion.motion_transport import FEATURE_DIM, FEATURE_NAMES
from real_motion.selective_forecast import (
    CorrectionSelector,
    SELECTIVE_LABEL_CACHE_VERSION,
    SELECTOR_INPUT_CONTRACT,
    SELECTOR_PROTOCOL,
    SPEED_FEATURE_INDEX,
    UTILITY_CONTRACT,
    deterministic_scene_split,
    spearman_corr,
)


def load_label_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != SELECTIVE_LABEL_CACHE_VERSION:
        raise RuntimeError(f"selective label cache version mismatch: {path}")
    meta = obj.get("metadata") or {}
    if meta.get("utility_contract") != UTILITY_CONTRACT:
        raise RuntimeError("utility contract mismatch")
    if meta.get("selector_input_contract") != SELECTOR_INPUT_CONTRACT:
        raise RuntimeError("selector input contract mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("empty selective label cache")
    return meta, records


def flatten(records):
    features = []
    targets = []
    scenes = []
    sample_ids = []
    source_indices = []
    for r in records:
        x = r["features"].float()
        y = r["utility_pp"].float()
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
            raise RuntimeError("feature shape mismatch")
        if y.shape != (x.shape[0],):
            raise RuntimeError("utility shape mismatch")
        features.append(x)
        targets.append(y)
        scenes.extend([str(r["scene_name"])] * len(y))
        sample_ids.extend([str(r["sample_id"])] * len(y))
        source_indices.extend(range(len(y)))
    return {
        "features": torch.cat(features, dim=0),
        "target": torch.cat(targets, dim=0),
        "scene": scenes,
        "sample_id": sample_ids,
        "source_index": source_indices,
    }


def corr_report(pred, target, features):
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    speed = np.asarray(features[:, SPEED_FEATURE_INDEX], dtype=np.float64)
    pos = y > 0
    top20 = max(1, int(round(0.20 * len(y))))
    pred_top = np.argsort(-p, kind="mergesort")[:top20]
    util_top = np.argsort(-y, kind="mergesort")[:top20]
    overlap = len(set(pred_top.tolist()) & set(util_top.tolist())) / max(top20, 1)
    return {
        "count": int(len(y)),
        "target_mean_pp": float(y.mean()) if len(y) else float("nan"),
        "target_std_pp": float(y.std()) if len(y) else float("nan"),
        "positive_fraction": float(pos.mean()) if len(y) else float("nan"),
        "selector_spearman": spearman_corr(p, y),
        "speed_spearman": spearman_corr(speed, y),
        "selector_top20_oracle_overlap": float(overlap),
        "selector_top20_mean_utility_pp": float(y[pred_top].mean()) if len(y) else float("nan"),
        "oracle_top20_mean_utility_pp": float(y[util_top].mean()) if len(y) else float("nan"),
        "speed_top20_mean_utility_pp": float(
            y[np.argsort(-speed, kind="mergesort")[:top20]].mean()
        ) if len(y) else float("nan"),
    }


@torch.no_grad()
def predict(model, x, mean, std, device, batch_size):
    model.eval()
    rows = []
    for lo in range(0, len(x), int(batch_size)):
        xb = x[lo:lo + int(batch_size)].to(device)
        xb = (xb - mean) / std
        rows.append(model(xb).float().cpu())
    return torch.cat(rows, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if a.batch_size <= 0 or a.epochs <= 0 or a.patience <= 0:
        raise ValueError("invalid training settings")

    random.seed(int(a.seed))
    np.random.seed(int(a.seed))
    torch.manual_seed(int(a.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(a.seed))

    meta, records = load_label_cache(a.label_cache)
    flat = flatten(records)
    x = flat["features"].float()
    y = flat["target"].float()
    scenes = flat["scene"]

    train_scenes, val_scenes = deterministic_scene_split(
        scenes, val_fraction=float(a.val_fraction), seed=int(a.seed)
    )
    train_mask = torch.tensor([s in train_scenes for s in scenes], dtype=torch.bool)
    val_mask = torch.tensor([s in val_scenes for s in scenes], dtype=torch.bool)
    if not bool(train_mask.any()) or not bool(val_mask.any()):
        raise RuntimeError("scene split produced an empty selector partition")

    x_train = x[train_mask]
    y_train = y[train_mask]
    x_val = x[val_mask]
    y_val = y[val_mask]

    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = y_train.mean()
    target_std = y_train.std(unbiased=False)
    if not torch.isfinite(target_std) or float(target_std) <= 1e-12:
        raise RuntimeError("selector utility target has no usable variance")

    x_train_n = (x_train - feature_mean) / feature_std
    y_train_n = (y_train - target_mean) / target_std

    ds = TensorDataset(x_train_n, y_train_n)
    gen = torch.Generator().manual_seed(int(a.seed))
    loader = DataLoader(
        ds,
        batch_size=int(a.batch_size),
        shuffle=True,
        generator=gen,
        num_workers=0,
        drop_last=False,
    )

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = CorrectionSelector(FEATURE_DIM, int(a.hidden_dim)).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(a.lr), weight_decay=float(a.weight_decay)
    )
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)

    mean_d = feature_mean.to(device)
    std_d = feature_std.to(device)
    best_state = None
    best_epoch = -1
    best_spearman = -float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, int(a.epochs) + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise FloatingPointError(f"non-finite selector loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach().cpu()) * len(xb)
            count += len(xb)

        pred_val_n = predict(
            model, x_val, mean_d, std_d, device, int(a.batch_size)
        )
        pred_val = pred_val_n * target_std + target_mean
        report = corr_report(
            pred_val.numpy(), y_val.numpy(), x_val.numpy()
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
            **report,
        }
        history.append(row)
        print(
            f"selector epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"val_spearman={row['selector_spearman']:+.4f} "
            f"speed={row['speed_spearman']:+.4f} "
            f"top20_overlap={100*row['selector_top20_oracle_overlap']:.2f}%",
            flush=True,
        )

        score = float(row["selector_spearman"])
        if np.isfinite(score) and score > best_spearman + 1e-6:
            best_spearman = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(a.patience):
            break

    if best_state is None:
        raise RuntimeError("selector never produced a finite validation rank score")
    model.load_state_dict(best_state, strict=True)

    pred_train_n = predict(
        model, x_train, mean_d, std_d, device, int(a.batch_size)
    )
    pred_val_n = predict(
        model, x_val, mean_d, std_d, device, int(a.batch_size)
    )
    pred_train = pred_train_n * target_std + target_mean
    pred_val = pred_val_n * target_std + target_mean

    train_report = corr_report(
        pred_train.numpy(), y_train.numpy(), x_train.numpy()
    )
    val_report = corr_report(
        pred_val.numpy(), y_val.numpy(), x_val.numpy()
    )

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "protocol": SELECTOR_PROTOCOL,
        "state_dict": best_state,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "hidden_dim": int(a.hidden_dim),
        "selector_input_contract": SELECTOR_INPUT_CONTRACT,
        "utility_contract": UTILITY_CONTRACT,
        "normalization": {
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
        },
        "label_cache": str(Path(a.label_cache).resolve()),
        "label_cache_metadata": meta,
        "seed": int(a.seed),
        "train_scenes": sorted(train_scenes),
        "val_scenes": sorted(val_scenes),
        "best_epoch": int(best_epoch),
        "best_val_spearman": float(best_spearman),
        "train_report": train_report,
        "val_report": val_report,
    }
    torch.save(checkpoint, out / "best.pt")
    report = {
        "protocol": SELECTOR_PROTOCOL,
        "best_epoch": int(best_epoch),
        "best_val_spearman": float(best_spearman),
        "num_train_sources": int(train_mask.sum()),
        "num_val_sources": int(val_mask.sum()),
        "num_train_scenes": len(train_scenes),
        "num_val_scenes": len(val_scenes),
        "train_report": train_report,
        "val_report": val_report,
        "history": history,
        "checkpoint": str((out / "best.pt").resolve()),
        "causal_input_only": True,
        "input_features": list(FEATURE_NAMES),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== SELECTOR TRAINING COMPLETE ===")
    print(json.dumps({
        "best_epoch": best_epoch,
        "best_val_spearman": best_spearman,
        "train": train_report,
        "val": val_report,
        "checkpoint": str(out / "best.pt"),
    }, indent=2))


if __name__ == "__main__":
    main()
