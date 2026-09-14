#!/usr/bin/env python3
"""Train the deliberately tiny causal KTA/V17 source selector.

Labels come from the GT-assisted utility cache built on TRAIN scenes.  The
validation cache is used only for a final diagnostic after checkpoint selection;
model selection itself uses a deterministic scene-disjoint split inside the
training cache so the 128-scene validation set is not repeatedly tuned against.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from real_motion.kta_v17_selector import (
    FEATURE_CONTRACT,
    SELECTOR_CACHE_VERSION,
    SELECTOR_FEATURE_DIM,
    SELECTOR_PROTOCOL,
    UTILITY_CONTRACT,
    KtaV17Selector,
    top_fraction_mask,
)

PROTOCOL = "p0_f9_kta_v17_selector_train_v1"


def _load(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != SELECTOR_CACHE_VERSION:
        raise RuntimeError(f"selector cache version mismatch: {path}")
    meta = obj.get("metadata") or {}
    if meta.get("feature_contract") != FEATURE_CONTRACT:
        raise RuntimeError("selector feature contract mismatch")
    if meta.get("utility_contract") != UTILITY_CONTRACT:
        raise RuntimeError("selector utility contract mismatch")
    if int(meta.get("feature_dim", -1)) != SELECTOR_FEATURE_DIM:
        raise RuntimeError("selector feature dim mismatch")
    recs = obj.get("records") or []
    if not recs:
        raise RuntimeError(f"empty selector cache: {path}")
    return meta, recs


def _scene_split(records, dev_fraction, seed):
    scenes = sorted({str(r["scene_name"]) for r in records})
    if len(scenes) < 2:
        raise RuntimeError("need at least two train scenes for internal dev split")
    rng = np.random.default_rng(int(seed))
    order = [scenes[i] for i in rng.permutation(len(scenes))]
    ndev = max(1, int(round(float(dev_fraction) * len(order))))
    ndev = min(ndev, len(order) - 1)
    dev = set(order[:ndev])
    train = [r for r in records if str(r["scene_name"]) not in dev]
    devr = [r for r in records if str(r["scene_name"]) in dev]
    return train, devr, sorted(set(scenes) - dev), sorted(dev)


def _flatten(records):
    xs, ys, window_ids = [], [], []
    for wi, r in enumerate(records):
        x = torch.as_tensor(r["features"], dtype=torch.float32)
        y = torch.as_tensor(r["gt_utility_pct"], dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != SELECTOR_FEATURE_DIM or y.shape != (x.shape[0],):
            raise RuntimeError(f"bad selector record shape: {r['sample_id']}")
        xs.append(x)
        ys.append(y)
        window_ids.extend([wi] * x.shape[0])
    return torch.cat(xs, 0), torch.cat(ys, 0), torch.tensor(window_ids, dtype=torch.long)


def _rank_stats(model, records, mean, std, device, budgets=(0.1, 0.2, 0.4)):
    model.eval()
    all_y, all_p = [], []
    capture = {float(q): [0.0, 0.0] for q in budgets}
    positive_hits = {float(q): [0, 0] for q in budgets}
    with torch.no_grad():
        for r in records:
            x = torch.as_tensor(r["features"], dtype=torch.float32, device=device)
            y = torch.as_tensor(r["gt_utility_pct"], dtype=torch.float32).numpy()
            p = model(x).float().cpu().numpy() * float(std) + float(mean)
            all_y.extend(y.tolist()); all_p.extend(p.tolist())
            positive_mass = float(np.maximum(y, 0.0).sum())
            positive_count = int((y > 0).sum())
            for q in budgets:
                m = top_fraction_mask(p, float(q))
                capture[float(q)][0] += float(np.maximum(y[m], 0.0).sum())
                capture[float(q)][1] += positive_mass
                positive_hits[float(q)][0] += int((y[m] > 0).sum())
                positive_hits[float(q)][1] += positive_count
    yy = np.asarray(all_y, dtype=np.float64)
    pp = np.asarray(all_p, dtype=np.float64)
    if len(yy) > 1 and yy.std() > 0 and pp.std() > 0:
        pearson = float(np.corrcoef(yy, pp)[0, 1])
        ry = np.argsort(np.argsort(yy, kind="stable"), kind="stable").astype(np.float64)
        rp = np.argsort(np.argsort(pp, kind="stable"), kind="stable").astype(np.float64)
        spearman = float(np.corrcoef(ry, rp)[0, 1])
    else:
        pearson = spearman = float("nan")
    return {
        "num_sources": int(len(yy)),
        "pearson": pearson,
        "spearman": spearman,
        "positive_fraction": float((yy > 0).mean()) if len(yy) else 0.0,
        "positive_utility_capture": {
            str(q): float(a / max(b, 1e-12)) for q, (a, b) in capture.items()
        },
        "positive_source_recall": {
            str(q): float(a / max(b, 1)) for q, (a, b) in positive_hits.items()
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dev-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.steps <= 0 or a.batch_size <= 0 or not 0.0 < a.dev_fraction < 0.5:
        raise ValueError("invalid training settings")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_meta, records = _load(a.train_cache)
    val_meta, val_records = _load(a.val_cache)
    overlap = sorted(set(train_meta.get("scene_names", [])) & set(val_meta.get("scene_names", [])))
    if overlap:
        raise RuntimeError(f"selector train/val scene overlap: {overlap[:8]}")

    fit_records, dev_records, fit_scenes, dev_scenes = _scene_split(
        records, a.dev_fraction, a.seed
    )
    xfit, yfit, _ = _flatten(fit_records)
    mean = float(yfit.mean())
    std = float(yfit.std().clamp_min(1e-6))
    yz = (yfit - mean) / std

    model = KtaV17Selector(SELECTOR_FEATURE_DIM, a.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    gen = torch.Generator().manual_seed(a.seed)
    best_loss = float("inf")
    best_state = None
    history = []

    def dev_loss():
        model.eval(); total = n = 0
        with torch.no_grad():
            for r in dev_records:
                x = torch.as_tensor(r["features"], dtype=torch.float32, device=device)
                y = (torch.as_tensor(r["gt_utility_pct"], dtype=torch.float32, device=device) - mean) / std
                pred = model(x)
                total += float(F.smooth_l1_loss(pred, y, reduction="sum").cpu())
                n += int(y.numel())
        model.train()
        return total / max(n, 1)

    model.train()
    for step in range(1, a.steps + 1):
        ids = torch.randint(0, xfit.shape[0], (min(a.batch_size, xfit.shape[0]),), generator=gen)
        xb = xfit[ids].to(device)
        yb = yz[ids].to(device)
        pred = model(xb)
        # Slightly emphasize non-zero utility without turning this into a
        # hand-tuned classification objective.
        weight = 1.0 + 0.5 * (yb.abs() > 0.25).float()
        loss = (F.smooth_l1_loss(pred, yb, reduction="none") * weight).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite selector loss at step {step}")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            dl = dev_loss()
            row = {"step": step, "train_loss": float(loss.detach().cpu()), "dev_loss": dl}
            history.append(row); print("SELECTOR_STEP " + json.dumps(row), flush=True)
            if dl < best_loss:
                best_loss = dl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("selector training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    dev_stats = _rank_stats(model, dev_records, mean, std, device)
    val_stats = _rank_stats(model, val_records, mean, std, device)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    ck = {
        "protocol": SELECTOR_PROTOCOL,
        "trainer_protocol": PROTOCOL,
        "state_dict": best_state,
        "input_dim": SELECTOR_FEATURE_DIM,
        "hidden_dim": int(a.hidden_dim),
        "target_mean": mean,
        "target_std": std,
        "train_cache": str(Path(a.train_cache).resolve()),
        "val_cache": str(Path(a.val_cache).resolve()),
        "seed": int(a.seed),
        "fit_scenes": fit_scenes,
        "dev_scenes": dev_scenes,
        "feature_contract": FEATURE_CONTRACT,
        "utility_contract": UTILITY_CONTRACT,
    }
    torch.save(ck, out / "best.pt")
    report = {
        "protocol": PROTOCOL,
        "best_dev_loss": best_loss,
        "target_mean": mean,
        "target_std": std,
        "num_fit_sources": int(xfit.shape[0]),
        "num_fit_scenes": len(fit_scenes),
        "num_dev_scenes": len(dev_scenes),
        "history": history,
        "dev_rank_stats": dev_stats,
        "val_rank_stats_diagnostic": val_stats,
        "note": "Checkpoint selection uses train-scene internal dev only; val-128 is diagnostic, not a selection set.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== SELECTOR TRAIN COMPLETE ===")
    print(json.dumps(report, indent=2))
    print(f"saved {out/'best.pt'}")


if __name__ == "__main__":
    main()
