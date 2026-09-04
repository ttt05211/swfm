#!/usr/bin/env python3
"""Frozen sparse OccFM diagnostic for P0-F9.

This evaluator answers one narrow question before any new training:

    Does the current Top-2 20x20 sparse adaptation already destroy the released
    OccFM-Fut forecasting function, or does the damage mainly appear after
    P0-F9 finetuning?

Protocol:
- load the released OccFM-Fut epoch=000196 transition weights;
- construct the current P0-F9 20x20 sparse transition geometry;
- do NOT load any trained P0-F9 checkpoint;
- disable the new full-context and physics-conditioning paths;
- use one global 50x50 Gaussian source field and crop it with the frozen Top-2
  WindowPlan, preserving overlap coherence;
- run two frozen samplers on the exact same source noise:
    1) official OccFM CFG scale = 2;
    2) P0-F9 Stage-1 deployment CFG scale = 1;
- scatter sparse future latents into the exact Strong-W2Det fallback latent;
- decode with the frozen VAE and apply the same dynamic-only deployment fusion;
- report Overall / Moving and the same-support GT oracle.

The two CFG settings separate sparse-geometry loss from the inference-policy
change made by P0-F9. No optimizer, EMA, semantic loss, or finetuning is used.
"""
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

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    REPORT_HORIZONS_S,
    SemanticIoUAccumulator,
)
from real_motion.models.p0_f9 import make_p0_f9_model
from real_motion.msp import latent_support_to_bev
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V2, MSPWorldModelCacheDataset
from real_motion.native_forecast import crop_coherent_source_noise, deterministic_sample_seed
from real_motion.occfm_io import (
    OccFMVAEAdapter,
    file_sha256,
    load_occfm_config,
    load_official_vae,
)
from real_motion.repair_target import apply_dynamic_repair
from real_motion.windows import WindowPlan, crop_windows, scatter_windows
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
PRED_HW = (20, 20)
FULL_HW = (50, 50)
FREE = 17
HIST_LAST = 4
OFFICIAL_CFG = 2.0
P0_F9_CFG = 1.0
PROTOCOL = "p0_f9_frozen_sparse_occfm_diagnostic_v1"


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


def _metric_pair(report):
    return float(report["overall"]["mIoU"]), float(report["moving"]["mIoU"])


def _official_seeded_noise_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Match the first torch.randn_like draw used by official OccFM cfm_eval."""
    if x.device.type != "cuda":
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
    cuda_devices = [x.device.index or 0]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        return torch.randn_like(x)


def _validate_cache(ds: MSPWorldModelCacheDataset, vae_ckpt: str) -> None:
    if ds.version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError("frozen sparse diagnostic requires a v2 absolute-future cache")
    meta = ds.metadata
    if meta.get("protocol") != P0_F9_CACHE_PROTOCOL:
        raise RuntimeError("cache is not audited P0-F9 v2")
    if meta.get("vae_mode") != "sample":
        raise RuntimeError("diagnostic requires posterior-sampled native latents")
    if int(meta.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("diagnostic requires native HIST_LAST=4 provenance")
    if int(meta.get("topk", -1)) != 2 or list(meta.get("window_hw", [])) != [20, 20]:
        raise RuntimeError("diagnostic requires frozen Top-2 20x20 routing")
    if meta.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("diagnostic requires Strong-W2Det anchor cache")
    if not bool(meta.get("include_eval_payload", False)):
        raise RuntimeError("diagnostic requires validation eval payload")
    if file_sha256(vae_ckpt) != meta.get("vae_checkpoint_sha256"):
        raise RuntimeError("VAE checkpoint differs from cache provenance")


def _assert_new_conditioning_is_noop(model) -> None:
    tr = model.transition
    checks = {
        "token_prior_proj": tr.prior_proj.weight,
        "context_proj_weight": tr.context_proj.weight,
        "context_proj_bias": tr.context_proj.bias,
        "physics_gate": tr.physics_fusion.gate,
    }
    bad = []
    for name, tensor in checks.items():
        if tensor is None:
            continue
        if bool((tensor.detach() != 0).any()):
            bad.append(name)
    if bad:
        raise RuntimeError(
            "frozen sparse diagnostic requires exact zero-impact new conditioning; "
            f"nonzero tensors: {bad}"
        )


def _load_dense_reference(path: str | None, ds: MSPWorldModelCacheDataset, n_eval: int):
    if not path:
        return None
    if n_eval != len(ds):
        return {
            "status": "skipped_for_partial_run",
            "reason": "dense baseline comparison is only valid for the full validation set",
        }
    p = Path(path)
    data = json.loads(p.read_text())
    if data.get("protocol") != "p0_f9_official_occfm_native_baseline_v2":
        raise RuntimeError("dense reference is not the audited official OccFM baseline")
    if int(data.get("num_windows", -1)) != len(ds):
        raise RuntimeError("dense reference window count differs from diagnostic cache")
    expected_sha = file_sha256(ds.root / "index.json")
    if data.get("cache_index_sha256") != expected_sha:
        raise RuntimeError("dense reference was evaluated on a different cache index")
    return data


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--occfm-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dense-baseline-json", default=None)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-windows", type=int, default=0,
                   help="0 means full validation set; use a small positive value only for smoke")
    a = p.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("frozen sparse OccFM diagnostic requires CUDA")

    ds = MSPWorldModelCacheDataset(a.cache)
    _validate_cache(ds, a.vae_ckpt)

    cfg = load_occfm_config(UP, "tools/cfgs/occfm_fut.yaml")
    if int(cfg.DATA_CONFIG.HIST_LAST) != HIST_LAST:
        raise RuntimeError("pinned official OccFM HIST_LAST changed")
    if int(cfg.LOSS.SAMPLE_STEP) != 10 or float(cfg.LOSS.ALPHA_STEP) != 3.0:
        raise RuntimeError("pinned official OccFM sampler schedule changed")
    if float(cfg.LOSS.UNCOND_SCALE) != OFFICIAL_CFG:
        raise RuntimeError("pinned official OccFM guidance scale changed")

    model = make_p0_f9_model(
        20,
        sample_steps=int(cfg.LOSS.SAMPLE_STEP),
        unconditional_probability=0.0,
        guidance_scale=OFFICIAL_CFG,
        hist_last=HIST_LAST,
    ).to(device)
    reuse = load_shape_safe(model.transition, a.occfm_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("diagnostic requires the official OccFM-Fut epoch=000196 checkpoint")
    reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)
    _assert_new_conditioning_is_noop(model)
    model.eval().requires_grad_(False)

    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)

    states = {
        "strong_anchor": _new_metrics(),
        "frozen_sparse_official_cfg": _new_metrics(),
        "frozen_sparse_p0f9_cfg": _new_metrics(),
        "same_support_gt_oracle": _new_metrics(),
    }
    oracle_checks = 0
    valid_windows = []
    write_ratios = []
    use_amp = bool(a.amp and device.type == "cuda")
    n_eval = len(ds) if int(a.max_windows) <= 0 else min(len(ds), int(a.max_windows))

    for i in range(n_eval):
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
        hist_w = crop_windows(hist_full, plan)
        physics_w = crop_windows(physics_full, plan)
        B, K = hist_w.shape[:2]
        flat_valid = plan.valid.reshape(-1)
        valid_windows.append(int(valid.sum().item()))

        predictions = {}
        if bool(flat_valid.any()):
            def flat(x):
                return x.reshape(B * K, *x.shape[2:])[flat_valid]

            fh = flat(hist_w)
            fp_zero = torch.zeros_like(flat(physics_w))
            orig = plan.origins.reshape(B * K, 2)[flat_valid]
            traj = s["trajectory"].to(device).unsqueeze(0)
            traj = traj[:, None].expand(B, K, 12, 2).reshape(B * K, 12, 2)[flat_valid]

            sample_seed = deterministic_sample_seed(
                str(s["sample_id"]), a.seed, stream="forecast"
            )
            global_noise = _official_seeded_noise_like(physics_full, sample_seed)
            initial_noise = crop_coherent_source_noise(global_noise, plan, flat_valid)

            for name, scale in (
                ("frozen_sparse_official_cfg", OFFICIAL_CFG),
                ("frozen_sparse_p0f9_cfg", P0_F9_CFG),
            ):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    pred = model.sample(
                        fh,
                        fp_zero,
                        history_context=None,
                        trajectory=traj,
                        window_origins=orig,
                        initial_noise=initial_noise,
                        guidance_scale=scale,
                    )
                padded = torch.zeros(B * K, *pred.shape[1:], device=device, dtype=pred.dtype)
                padded[flat_valid] = pred
                fused = scatter_windows(
                    padded.reshape(B, K, *pred.shape[1:]),
                    plan,
                    base=physics_full,
                )
                predictions[name] = vae.decode_labels(fused.float())[0].cpu().numpy()
        else:
            anchor_labels = s["eval_strong_anchor_occ"].cpu().numpy()
            predictions["frozen_sparse_official_cfg"] = anchor_labels
            predictions["frozen_sparse_p0f9_cfg"] = anchor_labels

        write_lat = s["msp_write_support_latent"].bool()
        write_bev = latent_support_to_bev(write_lat, (200, 200)).cpu().numpy().astype(bool)
        write_ratios.append(float(write_lat.float().mean().item()))

        gt_future = s["eval_future_gt_occ"].cpu().numpy()
        anchor_future = s["eval_strong_anchor_occ"].cpu().numpy()
        repair_target = s["eval_repair_target_occ"].cpu().numpy()
        moving = s["eval_gt_moving_support"].cpu().numpy().astype(bool)

        for horizon, fi in REPORT.items():
            gt = gt_future[fi]
            anchor = anchor_future[fi]
            _update(states["strong_anchor"], horizon, anchor, gt, moving[fi])

            for name in ("frozen_sparse_official_cfg", "frozen_sparse_p0f9_cfg"):
                final = apply_dynamic_repair(
                    anchor,
                    predictions[name][fi],
                    write_bev[fi],
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=FREE,
                )
                _update(states[name], horizon, final, gt, moving[fi])

            oracle = apply_dynamic_repair(
                anchor,
                gt,
                write_bev[fi],
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=FREE,
            )
            if not np.array_equal(oracle, repair_target[fi]):
                raise RuntimeError(
                    f"{s['sample_id']} horizon={horizon}: same-support GT oracle differs "
                    "from cached repair target"
                )
            oracle_checks += 1
            _update(states["same_support_gt_oracle"], horizon, oracle, gt, moving[fi])

        if i % 8 == 0:
            print("frozen_sparse_occfm_eval", i, s["sample_id"])

    metrics = {name: _report(state) for name, state in states.items()}
    anchor_o, anchor_m = _metric_pair(metrics["strong_anchor"])
    off_o, off_m = _metric_pair(metrics["frozen_sparse_official_cfg"])
    p9_o, p9_m = _metric_pair(metrics["frozen_sparse_p0f9_cfg"])
    oracle_o, oracle_m = _metric_pair(metrics["same_support_gt_oracle"])

    dense_reference = _load_dense_reference(a.dense_baseline_json, ds, n_eval)
    dense_comparison = None
    if dense_reference and dense_reference.get("status") != "skipped_for_partial_run":
        dense_metrics = dense_reference["metrics"]
        dense_o = float(dense_metrics["overall"]["mIoU"])
        dense_m = float(dense_metrics["moving"]["mIoU"])
        dense_comparison = {
            "dense_official_overall": dense_o,
            "dense_official_moving": dense_m,
            "frozen_sparse_official_cfg_minus_dense_overall": off_o - dense_o,
            "frozen_sparse_official_cfg_minus_dense_moving": off_m - dense_m,
            "frozen_sparse_p0f9_cfg_minus_dense_overall": p9_o - dense_o,
            "frozen_sparse_p0f9_cfg_minus_dense_moving": p9_m - dense_m,
            "cfg1_minus_cfg2_overall": p9_o - off_o,
            "cfg1_minus_cfg2_moving": p9_m - off_m,
        }

    report = {
        "protocol": PROTOCOL,
        "num_windows": n_eval,
        "cache_index_sha256": file_sha256(ds.root / "index.json"),
        "official_occfm_checkpoint": str(Path(a.occfm_ckpt).resolve()),
        "official_occfm_checkpoint_sha256": file_sha256(a.occfm_ckpt),
        "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
        "latent_distribution": "posterior_sample",
        "hist_last": HIST_LAST,
        "sample_steps": int(cfg.LOSS.SAMPLE_STEP),
        "alpha_step": float(cfg.LOSS.ALPHA_STEP),
        "seed": int(a.seed),
        "sample_seed_contract": "sha256(base,forecast,sample_id)",
        "source_noise_contract": "official-first-global-randn-50x50_then_crop_same_top2_plan",
        "window_contract": "top2_20x20_absolute_50x50_position_coordinates",
        "conditioning_contract": "no_full_context_no_physics_condition_no_finetuning",
        "official_guidance_scale": OFFICIAL_CFG,
        "p0_f9_guidance_scale": P0_F9_CFG,
        "official_transition_reuse_fraction": float(reuse_fraction),
        "official_transition_loaded_tensors": int(reuse.get("loaded", 0)),
        "official_transition_target_tensors": int(reuse.get("target_total", 0)),
        "oracle_bit_exact_checks": int(oracle_checks),
        "mean_valid_top2_windows": float(np.mean(valid_windows)) if valid_windows else 0.0,
        "mean_write_support_ratio": float(np.mean(write_ratios)) if write_ratios else 0.0,
        "metrics": metrics,
        "deltas_vs_strong_anchor": {
            "frozen_sparse_official_cfg_overall": off_o - anchor_o,
            "frozen_sparse_official_cfg_moving": off_m - anchor_m,
            "frozen_sparse_p0f9_cfg_overall": p9_o - anchor_o,
            "frozen_sparse_p0f9_cfg_moving": p9_m - anchor_m,
            "same_support_gt_oracle_overall": oracle_o - anchor_o,
            "same_support_gt_oracle_moving": oracle_m - anchor_m,
        },
        "dense_reference": dense_reference,
        "dense_comparison": dense_comparison,
        "interpretation_contract": {
            "large_negative_frozen_sparse_official_cfg_vs_dense": (
                "sparse geometry/adaptation already loses released OccFM capability before finetuning"
            ),
            "official_cfg_near_dense_but_p0f9_cfg_worse": (
                "guidance-policy change contributes materially to the loss"
            ),
            "both_frozen_sparse_variants_near_dense_but_trained_p0f9_worse": (
                "Stage-1 finetuning/objective is the primary failure source"
            ),
        },
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
