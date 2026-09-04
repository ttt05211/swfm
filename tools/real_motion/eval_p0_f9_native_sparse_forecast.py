#!/usr/bin/env python3
"""Deployment evaluation for audited P0-F9 native sparse forecasting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
from real_motion.models.p0_f9 import P0_F9_PROTOCOL, make_p0_f9_model
from real_motion.msp import latent_support_to_bev
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V2, MSPWorldModelCacheDataset
from real_motion.native_forecast import (
    crop_coherent_source_noise,
    deterministic_sample_seed,
)
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.repair_target import apply_dynamic_repair
from real_motion.windows import WindowPlan, crop_windows, scatter_windows
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
PRED_HW = (20, 20)
CONTEXT_HW = (40, 40)
FULL_HW = (50, 50)
FREE = 17
HIST_LAST = 4
SOURCE_SPATIAL_CONTRACT = "one_global_gaussian_field_cropped_into_top2_windows"


def _new_metrics():
    return {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
    }


def _update(state, horizon, pred, gt, moving_support):
    state["overall"][horizon].update(pred, gt)
    state["moving"].update(horizon, pred, gt, moving_support)


def _report(state):
    by_h = {h: state["overall"][h].compute() for h in REPORT_HORIZONS_S}
    return {
        "overall": {
            "mIoU": float(np.nanmean([by_h[h]["mIoU"] for h in REPORT_HORIZONS_S])),
            "per_horizon": by_h,
        },
        "moving": state["moving"].compute(),
    }


def _noise(shape, device, dtype, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randn(shape, device=device, dtype=dtype, generator=gen)


def _require_checkpoint_cache_match(ck: dict, ds: MSPWorldModelCacheDataset, vae_ckpt: str) -> None:
    actual_index_sha = file_sha256(ds.root / "index.json")
    if ck.get("val_cache_index_sha256") != actual_index_sha:
        raise RuntimeError("P0-F9 checkpoint was not validated against this exact cache index")
    ck_vae_sha = ck.get("vae_checkpoint_sha256")
    actual_vae_sha = file_sha256(vae_ckpt)
    if ck_vae_sha != actual_vae_sha:
        raise RuntimeError("P0-F9 checkpoint VAE provenance differs from evaluator VAE")
    ckmeta = ck.get("val_metadata") or {}
    meta = ds.metadata
    for key in (
        "protocol",
        "source_v3_cache_index_sha256",
        "source_msp_cache_sha256",
        "msp_checkpoint_sha256",
        "vae_checkpoint_sha256",
        "vae_mode",
        "native_backbone_hist_last",
        "anchor_contract",
        "write_budget_ratio",
    ):
        if ckmeta.get(key) != meta.get(key):
            raise RuntimeError(f"checkpoint/eval-cache metadata mismatch for {key}")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--sparse-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = MSPWorldModelCacheDataset(a.cache)
    if ds.version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError("P0-F9 evaluator requires a v2 absolute-future cache")
    meta = ds.metadata
    if meta.get("protocol") != P0_F9_CACHE_PROTOCOL:
        raise RuntimeError("evaluation cache is not audited P0-F9 v2")
    if meta.get("vae_mode") != "sample":
        raise RuntimeError("P0-F9 evaluator requires posterior-sampled native latents")
    if int(meta.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("P0-F9 evaluator requires native HIST_LAST=4 cache provenance")
    if int(meta.get("topk", -1)) != 2 or list(meta.get("window_hw", [])) != [20, 20]:
        raise RuntimeError("P0-F9 evaluator requires frozen Top-2 20x20 routing")
    if meta.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("P0-F9 evaluator cache does not use Strong-W2Det anchor")
    if not bool(meta.get("include_eval_payload", False)):
        raise RuntimeError("P0-F9 final eval requires a validation cache with eval payload")
    if file_sha256(a.vae_ckpt) != meta.get("vae_checkpoint_sha256"):
        raise RuntimeError("VAE checkpoint differs from P0-F9 cache")

    ck = torch.load(a.sparse_ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("architecture", {})
    if arch.get("protocol") != P0_F9_PROTOCOL or int(arch.get("stage", -1)) != 1:
        raise RuntimeError("checkpoint is not audited P0-F9 Stage-1")
    if int(arch.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("checkpoint HIST_LAST contract differs")
    if arch.get("flow_source_spatial_contract") != SOURCE_SPATIAL_CONTRACT:
        raise RuntimeError("checkpoint source-noise spatial contract differs")
    _require_checkpoint_cache_match(ck, ds, a.vae_ckpt)

    model = make_p0_f9_model(
        20,
        sample_steps=int(arch.get("sample_steps", 10)),
        unconditional_probability=float(arch.get("unconditional_probability", 0.0)),
        guidance_scale=float(arch.get("guidance_scale", 1.0)),
        hist_last=HIST_LAST,
    ).to(device)
    if a.use_ema:
        ema = ck.get("ema")
        if not ema or "state_dict" not in ema:
            raise RuntimeError("P0-F9 checkpoint lacks EMA state")
        model.load_state_dict(ema["state_dict"], strict=True)
        weight_source = "ema"
    else:
        model.load_state_dict(ck["state_dict"], strict=True)
        weight_source = "raw"
    model.eval()

    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)
    anchor_state = _new_metrics()
    model_state = _new_metrics()
    oracle_state = _new_metrics()
    valid_windows = []
    write_ratios = []
    use_amp = bool(a.amp and device.type == "cuda")
    oracle_checks = 0

    for i in range(len(ds)):
        s = ds[i]
        required = (
            "eval_future_gt_occ",
            "eval_strong_anchor_occ",
            "eval_repair_target_occ",
            "eval_gt_moving_support",
        )
        missing = [k for k in required if k not in s]
        if missing:
            raise RuntimeError(f"{s['sample_id']}: eval payload missing {missing}")

        origins = s["window_origins"].unsqueeze(0).long().to(device)
        valid = s["window_valid"].unsqueeze(0).bool().to(device)
        plan = WindowPlan(origins, valid, PRED_HW, FULL_HW)
        hist_full = s["full_history_latent"].unsqueeze(0).to(device)
        physics_full = s["anchor_future_latent"].unsqueeze(0).to(device)
        hist_local, hist_context, _ = crop_prediction_and_context(
            hist_full, plan, context_hw=CONTEXT_HW
        )
        physics_w = crop_windows(physics_full, plan)
        B, K = hist_local.shape[:2]
        flat_valid = plan.valid.reshape(-1)
        fused = physics_full
        if bool(flat_valid.any()):
            def flat(x):
                return x.reshape(B * K, *x.shape[2:])[flat_valid]

            fh = flat(hist_local)
            fc = flat(hist_context)
            fp = flat(physics_w)
            orig = plan.origins.reshape(B * K, 2)[flat_valid]
            traj = s["trajectory"].to(device).unsqueeze(0)
            traj = traj[:, None].expand(B, K, 12, 2).reshape(B * K, 12, 2)[flat_valid]

            # Native source is one coherent 50x50 Gaussian field. Crop it with
            # the same Top-2 plan so overlap cells share exactly one z0 value.
            global_noise = _noise(
                physics_full.shape,
                device,
                physics_full.dtype,
                deterministic_sample_seed(str(s["sample_id"]), a.seed, stream="forecast"),
            )
            initial_noise = crop_coherent_source_noise(global_noise, plan, flat_valid)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                pred = model.sample(
                    fh,
                    fp,
                    history_context=fc,
                    trajectory=traj,
                    window_origins=orig,
                    initial_noise=initial_noise,
                )
            padded = torch.zeros(B * K, *pred.shape[1:], device=device, dtype=pred.dtype)
            padded[flat_valid] = pred
            fused = scatter_windows(
                padded.reshape(B, K, *pred.shape[1:]),
                plan,
                base=physics_full,
            )

        decoded = vae.decode_labels(fused.float())[0].cpu().numpy()
        write_lat = s["msp_write_support_latent"].bool()
        write_bev = latent_support_to_bev(write_lat, (200, 200)).cpu().numpy().astype(bool)
        valid_windows.append(int(valid.sum().item()))
        write_ratios.append(float(write_lat.float().mean().item()))

        gt_future = s["eval_future_gt_occ"].cpu().numpy()
        anchor_future = s["eval_strong_anchor_occ"].cpu().numpy()
        repair_target = s["eval_repair_target_occ"].cpu().numpy()
        moving = s["eval_gt_moving_support"].cpu().numpy().astype(bool)
        for horizon, fi in REPORT.items():
            gt = gt_future[fi]
            anchor = anchor_future[fi]
            _update(anchor_state, horizon, anchor, gt, moving[fi])
            final = apply_dynamic_repair(
                anchor,
                decoded[fi],
                write_bev[fi],
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=FREE,
            )
            _update(model_state, horizon, final, gt, moving[fi])
            oracle = apply_dynamic_repair(
                anchor,
                gt,
                write_bev[fi],
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=FREE,
            )
            if not np.array_equal(oracle, repair_target[fi]):
                raise RuntimeError(
                    f"{s['sample_id']} horizon={horizon}: same-support GT oracle no longer matches cached repair target"
                )
            oracle_checks += 1
            _update(oracle_state, horizon, oracle, gt, moving[fi])

        if i % 8 == 0:
            print("eval", i, s["sample_id"])

    anchor_report = _report(anchor_state)
    trained_report = _report(model_state)
    oracle_report = _report(oracle_state)
    anchor_o = float(anchor_report["overall"]["mIoU"])
    trained_o = float(trained_report["overall"]["mIoU"])
    anchor_m = float(anchor_report["moving"]["mIoU"])
    trained_m = float(trained_report["moving"]["mIoU"])
    oracle_m = float(oracle_report["moving"]["mIoU"])
    report = {
        "protocol": {
            "name": "p0_f9_native_sparse_forecast_eval_v2",
            "weights": weight_source,
            "num_windows": len(ds),
            "topk": 2,
            "prediction_hw": [20, 20],
            "history_context_hw": [40, 40],
            "native_backbone_hist_last": HIST_LAST,
            "latent_distribution": "posterior_sample",
            "flow_source_spatial_contract": SOURCE_SPATIAL_CONTRACT,
            "slot_compute_ratio": float(np.mean(valid_windows) * 400.0 / 2500.0),
            "mean_write_latent_ratio": float(np.mean(write_ratios)),
            "write_budget_ratio": float(meta.get("write_budget_ratio", float("nan"))),
            "wm_task": "absolute future native flow forecasting",
            "physics_role": "Strong-W2Det condition plus exact fallback, never flow source",
            "fusion": "outside MSP support exact Strong-W2Det; inside support use only WM dynamic semantics",
            "sample_seed": int(a.seed),
            "sample_seed_contract": "sha256(base,forecast,sample_id)",
            "cache_index_sha256": file_sha256(ds.root / "index.json"),
            "oracle_consistency_checks": int(oracle_checks),
        },
        "strong_w2det_anchor": anchor_report,
        "trained_p0_f9": trained_report,
        "same_support_gt_oracle": oracle_report,
        "delta_overall_vs_strong_anchor": trained_o - anchor_o,
        "delta_Moving_vs_strong_anchor": trained_m - anchor_m,
        "oracle_delta_Moving_vs_strong_anchor": oracle_m - anchor_m,
        "remaining_Moving_headroom_to_oracle": oracle_m - trained_m,
        "checkpoint": str(Path(a.sparse_ckpt).resolve()),
        "checkpoint_sha256": file_sha256(a.sparse_ckpt),
        "checkpoint_step": int(ck.get("step", -1)),
        "best_val_objective": ck.get("best_val_objective"),
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
