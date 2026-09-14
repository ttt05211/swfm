#!/usr/bin/env python3
"""Train the final loss-alignment probe for the causal KTA/V17 selector.

This script intentionally keeps the exact same tiny MLP and causal input
contract as the regression selector.  The only substantive change is the
training objective: utility-weighted within-window pairwise ranking, aligned
with the final per-window Top-Q routing decision.

Checkpoint selection is frozen to internal-train-dev Q20 net utility.  Val-128
is reported once as a diagnostic and never used to select a checkpoint.
"""
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

from real_motion.kta_v17_selector import (
    FEATURE_CONTRACT,
    SELECTOR_CACHE_VERSION,
    SELECTOR_FEATURE_DIM,
    SELECTOR_PROTOCOL,
    UTILITY_CONTRACT,
    KtaV17Selector,
    top_fraction_mask,
    utility_pairwise_ranking_loss,
)

PROTOCOL = "p0_f9_kta_v17_selector_pairwise_ranking_v1"
SCORE_SEMANTICS = "pairwise_ranking_logit_not_calibrated_utility"


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
    fit = [r for r in records if str(r["scene_name"]) not in dev]
    devr = [r for r in records if str(r["scene_name"]) in dev]
    return fit, devr, sorted(set(scenes) - dev), sorted(dev)


def _auc_pos_neg(scores: np.ndarray, utility: np.ndarray) -> float:
    p = np.asarray(scores)[np.asarray(utility) > 0]
    n = np.asarray(scores)[np.asarray(utility) < 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    # Dataset is small enough that exact Mann-Whitney-style pair counting is
    # preferable to adding another dependency.
    wins = 0.0
    for a in p:
        wins += float((a > n).sum())
        wins += 0.5 * float((a == n).sum())
    return float(wins / (len(p) * len(n)))


def _rank_stats(model, records, device, q=0.20):
    model.eval()
    all_scores, all_utility = [], []
    pos_selected = neg_selected = 0.0
    total_pos = total_neg = 0.0
    selected = sources = 0
    pos_selected_count = neg_selected_count = zero_selected_count = 0

    with torch.no_grad():
        for r in records:
            x = torch.as_tensor(r["features"], dtype=torch.float32, device=device)
            y = np.asarray(r["gt_utility_pct"], dtype=np.float64)
            score = model(x).float().cpu().numpy()

            all_scores.append(score)
            all_utility.append(y)

            total_pos += float(y[y > 0].sum())
            total_neg += float(y[y < 0].sum())

            m = top_fraction_mask(score, float(q))
            ys = y[m]
            selected += int(m.sum())
            sources += int(len(y))
            pos_selected += float(ys[ys > 0].sum())
            neg_selected += float(ys[ys < 0].sum())
            pos_selected_count += int((ys > 0).sum())
            neg_selected_count += int((ys < 0).sum())
            zero_selected_count += int((ys == 0).sum())

    s = np.concatenate(all_scores) if all_scores else np.zeros(0, dtype=np.float64)
    y = np.concatenate(all_utility) if all_utility else np.zeros(0, dtype=np.float64)
    net = float(pos_selected + neg_selected)

    return {
        "q": float(q),
        "num_sources": int(len(y)),
        "selection_ratio": float(selected / max(sources, 1)),
        "positive_negative_auc": _auc_pos_neg(s, y),
        "positive_mass_capture": float(pos_selected / max(total_pos, 1e-12)),
        "negative_mass_capture": float(neg_selected / min(total_neg, -1e-12)),
        "net_selected_utility": net,
        "selected_positive_fraction": float(pos_selected_count / max(selected, 1)),
        "selected_negative_fraction": float(neg_selected_count / max(selected, 1)),
        "selected_zero_fraction": float(zero_selected_count / max(selected, 1)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--windows-per-step", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--zero-pair-weight", type=float, default=0.25)
    p.add_argument("--selection-q", type=float, default=0.20)
    p.add_argument("--dev-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if a.steps <= 0 or a.windows_per_step <= 0:
        raise ValueError("steps/windows-per-step must be positive")
    if not 0.0 < a.dev_fraction < 0.5:
        raise ValueError("dev-fraction must be in (0,0.5)")
    if not 0.0 < a.selection_q < 1.0:
        raise ValueError("selection-q must be in (0,1)")
    if a.zero_pair_weight < 0:
        raise ValueError("zero-pair-weight must be non-negative")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    train_meta, records = _load(a.train_cache)
    val_meta, val_records = _load(a.val_cache)
    overlap = sorted(
        set(train_meta.get("scene_names", [])) & set(val_meta.get("scene_names", []))
    )
    if overlap:
        raise RuntimeError(f"selector train/val scene overlap: {overlap[:8]}")

    fit_records, dev_records, fit_scenes, dev_scenes = _scene_split(
        records, a.dev_fraction, a.seed
    )

    model = KtaV17Selector(SELECTOR_FEATURE_DIM, a.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    rng = np.random.default_rng(int(a.seed))

    best_net = -float("inf")
    best_step = -1
    best_state = None
    history = []

    model.train()
    for step in range(1, a.steps + 1):
        ids = rng.integers(0, len(fit_records), size=int(a.windows_per_step))
        losses = []
        pair_counts = {"ph": 0, "pz": 0, "zh": 0}

        for rid in ids:
            r = fit_records[int(rid)]
            x = torch.as_tensor(r["features"], dtype=torch.float32, device=device)
            y = torch.as_tensor(r["gt_utility_pct"], dtype=torch.float32, device=device)
            score = model(x)
            loss_i, stat = utility_pairwise_ranking_loss(
                score, y, zero_pair_weight=float(a.zero_pair_weight)
            )
            losses.append(loss_i)
            pair_counts["ph"] += int(stat["positive_negative_pairs"])
            pair_counts["pz"] += int(stat["positive_zero_pairs"])
            pair_counts["zh"] += int(stat["zero_negative_pairs"])

        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite selector ranking loss at step {step}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            dev = _rank_stats(model, dev_records, device, q=float(a.selection_q))
            row = {
                "step": int(step),
                "train_loss": float(loss.detach().cpu()),
                "dev_q20_net_utility": float(dev["net_selected_utility"]),
                "dev_q20_positive_capture": float(dev["positive_mass_capture"]),
                "dev_q20_negative_capture": float(dev["negative_mass_capture"]),
                "dev_pos_neg_auc": float(dev["positive_negative_auc"]),
                "pairs_ph": int(pair_counts["ph"]),
                "pairs_pz": int(pair_counts["pz"]),
                "pairs_zh": int(pair_counts["zh"]),
            }
            history.append(row)
            print("RANK_SELECTOR_STEP " + json.dumps(row), flush=True)

            if dev["net_selected_utility"] > best_net:
                best_net = float(dev["net_selected_utility"])
                best_step = int(step)
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            model.train()

    if best_state is None:
        raise RuntimeError("ranking selector training produced no checkpoint")

    model.load_state_dict(best_state, strict=True)
    dev_stats = _rank_stats(model, dev_records, device, q=float(a.selection_q))
    val_stats = _rank_stats(model, val_records, device, q=float(a.selection_q))

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ck = {
        "protocol": SELECTOR_PROTOCOL,
        "trainer_protocol": PROTOCOL,
        "state_dict": best_state,
        "input_dim": SELECTOR_FEATURE_DIM,
        "hidden_dim": int(a.hidden_dim),
        # Existing evaluator applies this affine transform before Top-Q.
        # Identity preserves the ranking score exactly.
        "target_mean": 0.0,
        "target_std": 1.0,
        "score_semantics": SCORE_SEMANTICS,
        "train_cache": str(Path(a.train_cache).resolve()),
        "val_cache": str(Path(a.val_cache).resolve()),
        "seed": int(a.seed),
        "fit_scenes": fit_scenes,
        "dev_scenes": dev_scenes,
        "feature_contract": FEATURE_CONTRACT,
        "utility_contract": UTILITY_CONTRACT,
        "selection_q": float(a.selection_q),
        "zero_pair_weight": float(a.zero_pair_weight),
        "checkpoint_selection": "maximize_internal_dev_q20_net_selected_utility",
    }
    torch.save(ck, out / "best.pt")

    report = {
        "protocol": PROTOCOL,
        "score_semantics": SCORE_SEMANTICS,
        "best_step": best_step,
        "best_dev_q20_net_utility": best_net,
        "num_fit_scenes": len(fit_scenes),
        "num_dev_scenes": len(dev_scenes),
        "num_fit_windows": len(fit_records),
        "num_dev_windows": len(dev_records),
        "zero_pair_weight": float(a.zero_pair_weight),
        "selection_q": float(a.selection_q),
        "history": history,
        "dev_rank_stats": dev_stats,
        "val_rank_stats_diagnostic": val_stats,
        "note": (
            "Only the loss/selection criterion differs from the regression selector. "
            "Checkpoint selection uses internal train-scene dev Q20 net utility; "
            "val-128 is diagnostic only."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== RANKING SELECTOR TRAIN COMPLETE ===")
    print(json.dumps({
        "best_step": best_step,
        "best_dev_q20_net_utility": best_net,
        "dev": dev_stats,
        "val_diagnostic": val_stats,
    }, indent=2))
    print(f"saved {out/'best.pt'}")


if __name__ == "__main__":
    main()
