#!/usr/bin/env python3
"""Evaluate P0-F6 decoder-aware sparse innovation against Strong W2Det."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.context import crop_prediction_and_context
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    REPORT_HORIZONS_S,
    SemanticIoUAccumulator,
)
from real_motion.models.p0_f4 import make_p0_f4_model
from real_motion.msp import latent_support_to_bev
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V3, MSPWorldModelCacheDataset
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.repair_target import apply_dynamic_repair
from real_motion.windows import WindowPlan, crop_windows, scatter_windows

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
PRED_HW = (20, 20)
CONTEXT_HW = (40, 40)
FULL_HW = (50, 50)
FREE = 17
F6_PROTOCOL = "p0_f6_decoder_aware_sparse_innovation_v1"
REPAIR_CONTRACT = "strong_anchor_outside_support_gt_dynamic_inside_support_v1"
EVAL_KEYS = (
    "eval_future_gt_occ",
    "eval_strong_anchor_occ",
    "eval_repair_target_occ",
    "eval_gt_moving_support",
)


def _new_metrics():
    return {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
    }


def _report(state):
    overall_h = {h: state["overall"][h].compute() for h in REPORT_HORIZONS_S}
    return {
        "overall": {
            "mIoU": float(np.nanmean([overall_h[h]["mIoU"] for h in REPORT_HORIZONS_S])),
            "per_horizon": overall_h,
        },
        "moving": state["moving"].compute(),
    }


def _update(state, horizon, pred, gt, moving_support):
    state["overall"][horizon].update(pred, gt)
    state["moving"].update(horizon, pred, gt, moving_support)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--sparse-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = MSPWorldModelCacheDataset(a.cache)
    if ds.version != MSP_WM_CACHE_VERSION_V3:
        raise RuntimeError("P0-F6 evaluator requires the P0-F5/v3 repair-endpoint cache")
    if int(ds.metadata.get("topk", -1)) != 2:
        raise RuntimeError("P0-F6 evaluator is frozen to Top-2")
    if ds.metadata.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("cache does not use the strong W2Det anchor")
    if ds.metadata.get("repair_endpoint_contract") != REPAIR_CONTRACT:
        raise RuntimeError("cache repair endpoint contract mismatch")
    if ds.metadata.get("target") != "occupancy_sparse_repair_endpoint_vae_latent":
        raise RuntimeError("cache target is not the encoded occupancy repair endpoint")
    if not bool(ds.metadata.get("include_eval_payload", False)):
        raise RuntimeError("P0-F6 final evaluation requires P0-F5 validation eval payload")
    expected_vae = ds.metadata.get("vae_checkpoint_sha256")
    if expected_vae and file_sha256(a.vae_ckpt) != expected_vae:
        raise RuntimeError("VAE checkpoint differs from routed cache")

    ck = torch.load(a.sparse_ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("architecture", {})
    if arch.get("protocol") != F6_PROTOCOL:
        raise RuntimeError("Sparse-WM checkpoint is not P0-F6")
    if arch.get("repair_endpoint_contract") != REPAIR_CONTRACT:
        raise RuntimeError("checkpoint repair endpoint contract mismatch")
    semantic_lambda = ck.get("semantic_lambda", arch.get("semantic_lambda"))
    if semantic_lambda is None or float(semantic_lambda) <= 0:
        raise RuntimeError("P0-F6 checkpoint lacks a positive frozen semantic lambda")

    model = make_p0_f4_model(
        20,
        sample_steps=int(arch.get("sample_steps", 10)),
        source_noise_std=float(arch.get("source_noise_std", 0.0)),
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    vae, _ = load_official_vae(UP, a.vae_ckpt, device)
    va = OccFMVAEAdapter(vae)
    anchor_state = _new_metrics()
    model_state = _new_metrics()
    oracle_state = _new_metrics()
    valid_windows = []
    write_ratios = []
    use_amp = bool(a.amp and device.type == "cuda")

    for i in range(len(ds)):
        s = ds[i]
        missing = [k for k in EVAL_KEYS if k not in s]
        if missing:
            raise RuntimeError(f"{s['sample_id']}: compact eval payload missing {missing}")

        origins = s["window_origins"].unsqueeze(0).long()
        valid = s["window_valid"].unsqueeze(0).bool()
        plan = WindowPlan(origins.to(device), valid.to(device), PRED_HW, FULL_HW)

        hist_full = s["full_history_latent"].unsqueeze(0).to(device)
        anchor_full = s["anchor_future_latent"].unsqueeze(0).to(device)
        hist_local, hist_context, _ = crop_prediction_and_context(
            hist_full, plan, context_hw=CONTEXT_HW
        )
        anchor_w = crop_windows(anchor_full, plan)
        B, K = hist_local.shape[:2]
        flat_valid = plan.valid.reshape(-1)
        fused = anchor_full
        if bool(flat_valid.any()):
            def flat(x):
                return x.reshape(B * K, *x.shape[2:])[flat_valid]

            fh = flat(hist_local)
            fc = flat(hist_context)
            fa = flat(anchor_w)
            orig = plan.origins.reshape(B * K, 2)[flat_valid]
            traj = s["trajectory"].to(device).unsqueeze(0)
            traj = traj[:, None].expand(B, K, 12, 2).reshape(B * K, 12, 2)[flat_valid]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                pred = model.sample(
                    fh,
                    fa,
                    history_context=fc,
                    trajectory=traj,
                    window_origins=orig,
                )
            pad = torch.zeros(B * K, *pred.shape[1:], device=device, dtype=pred.dtype)
            pad[flat_valid] = pred
            fused = scatter_windows(
                pad.reshape(B, K, *pred.shape[1:]), plan, base=anchor_full
            )

        decoded = va.decode_labels(fused.float())[0].cpu().numpy()
        write_lat = s["msp_write_support_latent"].bool()
        write_bev = latent_support_to_bev(write_lat, (200, 200)).cpu().numpy().astype(bool)
        valid_windows.append(int(valid.sum().item()))
        write_ratios.append(float(write_lat.float().mean().item()))

        gt_future = s["eval_future_gt_occ"].cpu().numpy()
        anchor_future = s["eval_strong_anchor_occ"].cpu().numpy()
        repair_target_future = s["eval_repair_target_occ"].cpu().numpy()
        moving_support = s["eval_gt_moving_support"].cpu().numpy().astype(bool)
        for h, fi in REPORT.items():
            gt = gt_future[fi]
            anchor = anchor_future[fi]
            _update(anchor_state, h, anchor, gt, moving_support[fi])

            final = apply_dynamic_repair(
                anchor,
                decoded[fi],
                write_bev[fi],
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=FREE,
            )
            _update(model_state, h, final, gt, moving_support[fi])

            oracle = apply_dynamic_repair(
                anchor,
                gt,
                write_bev[fi],
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=FREE,
            )
            if not np.array_equal(oracle, repair_target_future[fi]):
                raise RuntimeError(
                    f"{s['sample_id']} horizon={h}: cached training endpoint does not "
                    "match the reported same-support GT repair oracle"
                )
            _update(oracle_state, h, oracle, gt, moving_support[fi])

        if i % 8 == 0:
            print("eval", i, s["sample_id"])

    anchor_report = _report(anchor_state)
    model_report = _report(model_state)
    oracle_report = _report(oracle_state)
    am = float(anchor_report["moving"]["mIoU"])
    mm = float(model_report["moving"]["mIoU"])
    om = float(oracle_report["moving"]["mIoU"])
    report = {
        "protocol": {
            "name": "p0_f6_decoder_aware_sparse_innovation_eval_v1",
            "num_windows": len(ds),
            "topk": 2,
            "prediction_hw": list(PRED_HW),
            "history_context_hw": list(CONTEXT_HW),
            "slot_compute_ratio": float(np.mean(valid_windows) * 400.0 / 2500.0),
            "mean_write_latent_ratio": float(np.mean(write_ratios)),
            "write_budget_ratio": float(ds.metadata.get("write_budget_ratio", float("nan"))),
            "anchor": "strong occupancy-only W2Det",
            "history": "full occupancy history latent",
            "training_target": "P0-F5 encoded occupancy sparse repair endpoint",
            "training_objective": "FM MSE + gradient-calibrated decoder-aware 9-way dynamic repair CE",
            "semantic_lambda": float(semantic_lambda),
            "semantic_lambda_calibration": ck.get("semantic_lambda_calibration"),
            "fusion": "Strong W2Det exact default; decoded WM dynamic semantics may write only inside causal MSP horizon support",
            "endpoint_oracle_consistency": "bit-exact checked on all reported horizons",
        },
        "strong_w2det_anchor": anchor_report,
        "trained_sparse_wm": model_report,
        "same_support_gt_repair_oracle": oracle_report,
        "delta_Moving_vs_strong_anchor": mm - am,
        "oracle_delta_Moving_vs_strong_anchor": om - am,
        "remaining_Moving_headroom_to_oracle": om - mm,
        "checkpoint": str(Path(a.sparse_ckpt).resolve()),
        "best_val_objective": ck.get("best_val_objective"),
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
