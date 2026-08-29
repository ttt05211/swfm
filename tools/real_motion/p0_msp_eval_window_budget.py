#!/usr/bin/env python3
"""P0-F2: evaluate a trained MSP with deployment-aligned Top-K WM windows.

This script does not retrain MSP and does not train a World Model. It reuses the
P0-F1 best checkpoint, converts its 50x50 latent score maps directly into shared
20x20 spatial WM crops, and evaluates two oracle views on the same 128-scene
validation protocol:

1. support-only oracle: comparable to the frozen Hybrid-v6 support oracle;
2. anchor-preserving repair oracle: keep KTA/zero-motion dynamic anchor outside
   selected windows and replace selected windows with GT dynamic occupancy.

The latter matches the intended final contract: unselected regions preserve the
causal anchor rather than disappearing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    REPORT_HORIZONS_S,
    SemanticIoUAccumulator,
)
from real_motion.msp import (
    FEATURE_DIM,
    MSPProbeHead,
    collate_probe_records,
    latent_support_to_bev,
    rasterize_msp_scores,
)
from real_motion.msp_window import (
    plan_topk_score_windows,
    score_capture_ratio,
    window_plan_support,
)
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    WindowTokens,
    causal_dynamic_target_semantics,
    dynamic_only_semantics,
)
from real_motion.prepared import prepare_nuscenes_window
from real_motion.runtime_config import get_cfg, make_prepare_config
from tools.real_motion.p0_msp_train_probe import _load_cache

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}


def _record_window(r):
    return WindowTokens(
        scene_name=str(r["scene_name"]),
        history_tokens=tuple(r["history_tokens"]),
        t0_token=str(r["t0_token"]),
        future_tokens=tuple(r["future_tokens"]),
    )


def _new_metric_state():
    return {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
    }


def _new_window_state():
    d = _new_metric_state()
    d.update({
        "arrival_hit": np.zeros(6, dtype=np.float64),
        "arrival_total": np.zeros(6, dtype=np.float64),
        "active_latent": np.zeros(6, dtype=np.float64),
        "num_windows": [],
        "slot_compute_ratio": [],
        "unique_latent_ratio": [],
        "score_capture_ratio": [],
    })
    return d


def _mean_h(acc):
    per = {h: acc[h].compute() for h in REPORT_HORIZONS_S}
    return {
        "mIoU": float(np.nanmean([per[h]["mIoU"] for h in REPORT_HORIZONS_S])),
        "per_horizon": per,
    }


def _metric_report(state):
    return {
        "oracle_overall": _mean_h(state["overall"]),
        "oracle_Moving-mIoU_v2": state["moving"].compute(),
    }


def _window_report(state, processed):
    out = _metric_report(state)
    denom = np.maximum(state["arrival_total"], 1.0)
    out.update({
        "future_arrival_recall_per_horizon": (state["arrival_hit"] / denom).tolist(),
        "active_latent_per_horizon": (state["active_latent"] / max(processed, 1)).tolist(),
        "window_backend": {
            "mean_num_windows": float(np.mean(state["num_windows"])) if state["num_windows"] else 0.0,
            "mean_slot_compute_ratio": float(np.mean(state["slot_compute_ratio"])) if state["slot_compute_ratio"] else 0.0,
            "mean_unique_latent_ratio": float(np.mean(state["unique_latent_ratio"])) if state["unique_latent_ratio"] else 0.0,
            "mean_score_capture_ratio": float(np.mean(state["score_capture_ratio"])) if state["score_capture_ratio"] else 0.0,
        },
    })
    return out


def _accumulate_arrival(state, arrival_vox, write_bev):
    a = np.asarray(arrival_vox, dtype=bool)
    w = np.asarray(write_bev, dtype=bool)
    if a.ndim != 4 or w.ndim != 3 or tuple(a.shape[:3]) != tuple(w.shape):
        raise ValueError("arrival/write support shape mismatch")
    state["arrival_total"] += a.sum(axis=(1, 2, 3), dtype=np.float64)
    state["arrival_hit"] += (a & w[..., None]).sum(axis=(1, 2, 3), dtype=np.float64)


def _compose(static, dynamic, protected, *, write_support=None):
    return static_protected_compose(
        torch.from_numpy(np.asarray(static)),
        torch.from_numpy(np.asarray(dynamic)),
        torch.from_numpy(np.asarray(protected)),
        DYNAMIC_CLASS_IDS,
        write_support=(
            None if write_support is None
            else torch.from_numpy(np.asarray(write_support, dtype=bool))
        ),
    ).numpy()


def _load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = (
        "state_dict", "feature_dim", "hidden_dim", "num_heads", "num_modes",
        "future_frames", "resolved_config", "val_metadata",
    )
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(f"MSP checkpoint missing keys {missing}")
    if int(ckpt["feature_dim"]) != FEATURE_DIM:
        raise RuntimeError("checkpoint feature contract differs from current MSP code")
    model = MSPProbeHead(
        feature_dim=FEATURE_DIM,
        hidden_dim=int(ckpt["hidden_dim"]),
        num_heads=int(ckpt["num_heads"]),
        num_modes=int(ckpt["num_modes"]),
        future_frames=int(ckpt["future_frames"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return ckpt, model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--val-info-pkl", required=True)
    p.add_argument("--topk", default="1,2,3")
    p.add_argument("--probe-report", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    topks = tuple(sorted(set(int(x) for x in a.topk.split(",") if x.strip())))
    if not topks or any(k <= 0 for k in topks):
        raise ValueError("topk must contain positive integers")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ckpt, model = _load_model(a.checkpoint, device)
    val_meta, records = _load_cache(a.val_cache)
    if val_meta.get("selection") != "scene_disjoint_midpoint_one_window_per_scene_v1":
        raise RuntimeError("P0-F2 requires the frozen scene-disjoint midpoint val cache")
    ckpt_scenes = tuple(sorted(ckpt["val_metadata"].get("scene_names", [])))
    cache_scenes = tuple(sorted(val_meta.get("scene_names", [])))
    if ckpt_scenes != cache_scenes:
        raise RuntimeError("checkpoint and validation cache scene sets differ")
    if int(ckpt["future_frames"]) != 6:
        raise RuntimeError("P0-F2 frozen protocol expects 6 future frames")

    cfg = ckpt["resolved_config"]
    pcfg = make_prepare_config(cfg)
    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = tuple(int(v) for v in get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20]))
    if latent_hw != (50, 50):
        raise RuntimeError(f"P0-F2 frozen probe expects latent 50x50, got {latent_hw}")
    if window_hw != (20, 20):
        raise RuntimeError(f"P0-F2 deployment audit expects current 20x20 WM windows, got {window_hw}")

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.val_info_pkl, verbose=False)
    decomposition = _new_metric_state()
    hybrid_support = _new_metric_state()
    anchor = _new_metric_state()
    window_support = {k: _new_window_state() for k in topks}
    window_repair = {k: _new_metric_state() for k in topks}

    processed = 0
    for i, r in enumerate(records):
        batch = collate_probe_records([r])
        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        pred = model(batch_dev["features"], batch_dev["candidate_mask"])
        score = rasterize_msp_scores(
            pred, batch_dev, latent_hw=latent_hw, grid=pcfg.grid
        ).cpu()

        plans = {}
        latent_support = {}
        bev_support = {}
        for k in topks:
            plan = plan_topk_score_windows(
                score, window_hw=window_hw, max_windows=int(k)
            )
            spatial = window_plan_support(plan)[0].cpu()
            lat = spatial.unsqueeze(0).expand(pcfg.future_frames, -1, -1).clone()
            bev = latent_support_to_bev(lat, pcfg.grid.shape_hwd[:2]).cpu().numpy()
            plans[k] = plan
            latent_support[k] = lat
            bev_support[k] = bev

        w = _record_window(r)
        base = prepare_nuscenes_window(source, w, pcfg, include_gt=True)
        arrival = np.asarray(base["future_moving_occ"]) != pcfg.free_label
        rule_bev = np.asarray(base["generation_support_occ"], dtype=bool)

        for k in topks:
            state = window_support[k]
            lat = latent_support[k]
            plan = plans[k]
            state["active_latent"] += lat.float().mean(dim=(1, 2)).numpy()
            _accumulate_arrival(state, arrival, bev_support[k])
            nw = int(plan.valid.sum().item())
            state["num_windows"].append(nw)
            state["slot_compute_ratio"].append(
                nw * window_hw[0] * window_hw[1] / float(latent_hw[0] * latent_hw[1])
            )
            state["unique_latent_ratio"].append(float(lat[0].float().mean().item()))
            state["score_capture_ratio"].append(
                float(score_capture_ratio(score, plan)[0].item())
            )

        for h, fi in REPORT.items():
            gt = np.asarray(base["future_gt_occ"])[fi]
            static = np.asarray(base["static_future_occ"])[fi]
            protected = np.asarray(base["confident_static_future_mask"])[fi]
            moving_support = np.asarray(base["gt_moving_support"])[fi]
            gt_dyn = dynamic_only_semantics(gt, pcfg.free_label)

            dec = _compose(static, gt_dyn, protected)
            decomposition["overall"][h].update(dec, gt)
            decomposition["moving"].update(h, dec, gt, moving_support)

            rule_dyn = causal_dynamic_target_semantics(
                gt, rule_bev[fi], pcfg.free_label
            )
            rule_pred = _compose(
                static, rule_dyn, protected, write_support=rule_bev[fi]
            )
            hybrid_support["overall"][h].update(rule_pred, gt)
            hybrid_support["moving"].update(h, rule_pred, gt, moving_support)

            anchor_dyn = np.asarray(base["kta_future_occ"])[fi]
            anchor_pred = _compose(static, anchor_dyn, protected)
            anchor["overall"][h].update(anchor_pred, gt)
            anchor["moving"].update(h, anchor_pred, gt, moving_support)

            for k in topks:
                write = np.asarray(bev_support[k][fi], dtype=bool)

                # Support-only view, directly comparable to the P0-F1 learned
                # support oracle and frozen Hybrid-v6 support oracle.
                ldyn = causal_dynamic_target_semantics(gt, write, pcfg.free_label)
                lpred = _compose(static, ldyn, protected, write_support=write)
                window_support[k]["overall"][h].update(lpred, gt)
                window_support[k]["moving"].update(h, lpred, gt, moving_support)

                # Final-method view: unselected dynamic regions keep the
                # KTA/zero-motion anchor; selected windows receive perfect GT
                # dynamic repair for this oracle only.
                repair_dyn = np.where(write[..., None], gt_dyn, anchor_dyn)
                repair_pred = _compose(static, repair_dyn, protected)
                window_repair[k]["overall"][h].update(repair_pred, gt)
                window_repair[k]["moving"].update(h, repair_pred, gt, moving_support)

        processed += 1
        if i % 8 == 0:
            print("window oracle eval", i, r["sample_id"])

    decomp_report = _metric_report(decomposition)
    hybrid_report = _metric_report(hybrid_support)
    anchor_report = _metric_report(anchor)
    window_support_report = {str(k): _window_report(window_support[k], processed) for k in topks}
    window_repair_report = {str(k): _metric_report(window_repair[k]) for k in topks}

    hybrid_moving = float(hybrid_report["oracle_Moving-mIoU_v2"]["mIoU"])
    anchor_moving = float(anchor_report["oracle_Moving-mIoU_v2"]["mIoU"])
    for k in topks:
        window_support_report[str(k)]["delta_Moving_vs_hybrid_support"] = float(
            window_support_report[str(k)]["oracle_Moving-mIoU_v2"]["mIoU"] - hybrid_moving
        )
        window_repair_report[str(k)]["delta_Moving_vs_anchor"] = float(
            window_repair_report[str(k)]["oracle_Moving-mIoU_v2"]["mIoU"] - anchor_moving
        )

    report = {
        "protocol": {
            "name": "p0_f2_window_aligned_msp_oracle_v1",
            "probe_only": True,
            "world_model_trained": False,
            "checkpoint": str(a.checkpoint),
            "val_windows": int(processed),
            "val_unique_scenes": len(cache_scenes),
            "latent_hw": list(latent_hw),
            "window_hw": list(window_hw),
            "topk": list(topks),
            "score_aggregation": "sum_over_6_future_horizons",
            "selection": "greedy_marginal_score_topk_spatial_windows",
            "shared_windows_across_horizons": True,
            "selected_window_is_fully_active": True,
            "anchor_preserving_contract": (
                "outside selected windows keep KTA/zero-motion dynamic anchor; "
                "inside selected windows use GT dynamic occupancy only for oracle evaluation"
            ),
        },
        "decomposition": decomp_report,
        "frozen_hybrid_v6_support_oracle": hybrid_report,
        "causal_kta_zero_anchor": anchor_report,
        "window_support_oracle_by_topk": window_support_report,
        "anchor_preserving_repair_oracle_by_topk": window_repair_report,
    }

    if a.probe_report is not None:
        old = json.loads(Path(a.probe_report).read_text(encoding="utf-8"))
        report["p0_f1_cell_budget_reference"] = old.get("oracle_curve", {})

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
