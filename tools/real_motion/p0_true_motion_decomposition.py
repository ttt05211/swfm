#!/usr/bin/env python3
"""P0-A: frozen OccFM-fut-196 Full vs causal-static vs moving/uncertain.

Besides the original aggregate diagnostics, this report now includes:
- ordinary semantic mIoU on the full occupancy grid;
- binary occupied-vs-free IoU ("occupancy_IoU");
- ordinary dynamic-class mIoU on the full occupancy grid;
- paired per-class IoU comparisons on the exact same GT moving/static supports.

The paired tables are the preferred separability diagnostic: a class/horizon is
included only when that class is actually present in the shared GT support.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.prepared import PreparedShardDataset
from real_motion.occfm_io import (
    load_official_vae,
    load_official_wm,
    OccFMVAEAdapter,
    run_frozen_occfm_forecast,
)
from real_motion.metrics.moving_miou_v2 import (
    MovingMIoUV2MultiHorizon,
    SemanticIoUAccumulator,
    REPORT_HORIZONS_S,
    DYNAMIC_CLASS_IDS,
    NUSCENES_LABELS,
)
from real_motion.nuscenes_adapter import gt_true_static_mask
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    get_cfg,
    save_resolved_config,
)

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
FREE_LABEL = 17


class OccupancyIoUAccumulator:
    """Binary occupied-vs-free IoU, accumulated dataset-wide."""

    def __init__(self, free_label=FREE_LABEL):
        self.free_label = int(free_label)
        self.inter = 0
        self.union = 0

    def update(self, pred, gt, mask=None):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        if pred.shape != gt.shape:
            raise ValueError("shape mismatch")
        valid = np.ones_like(gt, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if valid.shape != gt.shape:
            raise ValueError("mask shape mismatch")
        p = (pred != self.free_label) & valid
        g = (gt != self.free_label) & valid
        self.inter += int((p & g).sum())
        self.union += int((p | g).sum())

    def compute(self):
        return 100.0 * self.inter / self.union if self.union else float("nan")


def mean_h(acc):
    per = {h: a.compute() for h, a in acc.items()}
    return {
        "mIoU": float(np.nanmean([per[h]["mIoU"] for h in REPORT_HORIZONS_S])),
        "per_horizon": per,
    }


def mean_occ(acc):
    per = {h: float(a.compute()) for h, a in acc.items()}
    vals = [per[h] for h in REPORT_HORIZONS_S if np.isfinite(per[h])]
    return {
        "IoU": float(np.mean(vals)) if vals else float("nan"),
        "per_horizon": {h: {"IoU": per[h]} for h in REPORT_HORIZONS_S},
    }


def _gt_count_template(classes):
    return {h: {int(c): 0 for c in classes} for h in REPORT_HORIZONS_S}


def _update_gt_counts(dst, horizon, gt, mask, classes):
    gt = np.asarray(gt)
    mask = np.asarray(mask, dtype=bool)
    for c in classes:
        dst[horizon][int(c)] += int(((gt == int(c)) & mask).sum())


def paired_class_report(full_acc, branch_acc, gt_counts, classes):
    """Compare two predictions only on GT-present class/horizon pairs."""
    out = {"per_horizon": {}}
    all_deltas = []
    for h in REPORT_HORIZONS_S:
        full = full_acc[h].compute()
        branch = branch_acc[h].compute()
        rows = []
        h_deltas = []
        for c in classes:
            c = int(c)
            n_gt = int(gt_counts[h][c])
            if n_gt <= 0:
                continue
            f = float(full["per_class"][c])
            b = float(branch["per_class"][c])
            delta = b - f
            rows.append({
                "class_id": c,
                "class_name": NUSCENES_LABELS[c],
                "gt_voxels": n_gt,
                "full_iou": f,
                "branch_iou": b,
                "delta_pp": delta,
            })
            if np.isfinite(delta):
                h_deltas.append(delta)
                all_deltas.append(delta)
        out["per_horizon"][str(h)] = {
            "macro_delta_pp": float(np.mean(h_deltas)) if h_deltas else float("nan"),
            "classes": rows,
        }
    out["macro_delta_pp"] = float(np.mean(all_deltas)) if all_deltas else float("nan")
    out["num_gt_present_class_horizon_pairs"] = len(all_deltas)
    return out


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--prepared", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True, help="official occfm_fut epoch=000196.ckpt")
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--vae-mode", choices=["sample", "mean"], default=None)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    mode = a.vae_mode or get_cfg(cfg, "CACHE.VAE_LATENT_MODE", "sample")
    ds = PreparedShardDataset(a.prepared)
    n = len(ds) if a.max_windows is None else min(len(ds), a.max_windows)
    if n == 0:
        raise RuntimeError("no prepared windows")

    vae, _ = load_official_vae(UP, a.vae_ckpt, a.device)
    wm, _ = load_official_wm(
        UP,
        a.wm_ckpt,
        a.device,
        config_rel=str(get_cfg(cfg, "UPSTREAM.WM_CONFIG", "tools/cfgs/occfm_fut.yaml")),
    )
    ad = OccFMVAEAdapter(vae)

    branches = ("full", "static", "moving")
    mm = {b: MovingMIoUV2MultiHorizon() for b in branches}
    overall = {b: {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S} for b in branches}
    dynamic = {
        b: {h: SemanticIoUAccumulator(classes=DYNAMIC_CLASS_IDS) for h in REPORT_HORIZONS_S}
        for b in branches
    }
    occ = {b: {h: OccupancyIoUAccumulator() for h in REPORT_HORIZONS_S} for b in branches}
    ts = {b: {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S} for b in branches}

    moving_gt_counts = _gt_count_template(DYNAMIC_CLASS_IDS)
    static_classes = tuple(range(17))
    static_gt_counts = _gt_count_template(static_classes)

    for i in range(n):
        s = ds[i]
        traj = torch.as_tensor(s["trajectory"], dtype=torch.float32)
        if tuple(traj.shape) != (12, 2):
            raise RuntimeError("P0-A requires OccFM-fut prepared trajectory [12,2]")

        seed = a.seed + i * 101
        hist = {
            "full": ad.encode(torch.from_numpy(s["full_history_occ"]).unsqueeze(0), mode=mode, seed=seed)[0].cpu(),
            "static": ad.encode(torch.from_numpy(s["static_history_occ"]).unsqueeze(0), mode=mode, seed=seed)[0].cpu(),
            "moving": ad.encode(torch.from_numpy(s["moving_history_occ"]).unsqueeze(0), mode=mode, seed=seed)[0].cpu(),
        }
        zf = ad.encode(
            torch.from_numpy(s["future_gt_occ"]).unsqueeze(0), mode=mode, seed=seed + 1
        )[0].cpu()
        pred = {
            b: run_frozen_occfm_forecast(
                wm,
                hist[b],
                zf,
                trajectory=traj,
                seed=a.seed + i * 1009,
                hist_last=4,
            ).numpy()
            for b in branches
        }

        for h in REPORT_HORIZONS_S:
            fi = REPORT[h]
            gt = s["future_gt_occ"][fi]
            sup = s["gt_moving_support"][fi]
            sm = gt_true_static_mask(gt, sup)

            _update_gt_counts(moving_gt_counts, h, gt, sup, DYNAMIC_CLASS_IDS)
            _update_gt_counts(static_gt_counts, h, gt, sm, static_classes)

            for b in branches:
                overall[b][h].update(pred[b][fi], gt)
                dynamic[b][h].update(pred[b][fi], gt)
                occ[b][h].update(pred[b][fi], gt)
                ts[b][h].update(pred[b][fi], gt, sm)
                mm[b].update(h, pred[b][fi], gt, sup)

    report = {
        "protocol": "P0-A_occfm_fut196_causal_real_motion_decomposition_v2",
        "trajectory_protocol": "occfm_fut_12step_v1",
        "dense_reference": "official_occfm_fut_epoch196",
        "num_windows": n,
        "vae_mode": mode,
        "branch_input_definition": {
            "full": "full semantic history",
            "static": "causal confident-static semantic history only",
            "moving": "causal WM-candidate history = moving | uncertain",
        },
        "branches": {},
    }

    for b in branches:
        report["branches"][b] = {
            "overall_mIoU": mean_h(overall[b]),
            "occupancy_IoU": mean_occ(occ[b]),
            "dynamic_class_mIoU": mean_h(dynamic[b]),
            "true_static_mIoU": mean_h(ts[b]),
            "Moving-mIoU_v2": mm[b].compute(),
        }

    report["paired_class_comparison"] = {
        "moving_support_full_vs_moving_branch": paired_class_report(
            mm["full"].acc, mm["moving"].acc, moving_gt_counts, DYNAMIC_CLASS_IDS
        ),
        "true_static_support_full_vs_static_branch": paired_class_report(
            ts["full"], ts["static"], static_gt_counts, static_classes
        ),
        "note": (
            "Preferred separability diagnostic. Each row uses the identical GT support "
            "and is included only when that GT class is present; branch_iou-full_iou is "
            "therefore an apples-to-apples per-class delta."
        ),
    }

    report["legacy_separability"] = {
        "static_only_minus_full_true_static_pp": (
            report["branches"]["static"]["true_static_mIoU"]["mIoU"]
            - report["branches"]["full"]["true_static_mIoU"]["mIoU"]
        ),
        "moving_only_minus_full_moving_pp": (
            report["branches"]["moving"]["Moving-mIoU_v2"]["mIoU"]
            - report["branches"]["full"]["Moving-mIoU_v2"]["mIoU"]
        ),
        "preferred_metric": "paired_class_comparison",
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))

    print(json.dumps({
        "num_windows": n,
        "full_overall_mIoU": report["branches"]["full"]["overall_mIoU"]["mIoU"],
        "full_occupancy_IoU": report["branches"]["full"]["occupancy_IoU"]["IoU"],
        "full_dynamic_class_mIoU": report["branches"]["full"]["dynamic_class_mIoU"]["mIoU"],
        "paired_moving_macro_delta_pp": report["paired_class_comparison"][
            "moving_support_full_vs_moving_branch"
        ]["macro_delta_pp"],
        "paired_static_macro_delta_pp": report["paired_class_comparison"][
            "true_static_support_full_vs_static_branch"
        ]["macro_delta_pp"],
    }, indent=2))


if __name__ == "__main__":
    main()
