#!/usr/bin/env python3
"""Deployment-controlled frozen sparse OccFM diagnostic for P0-F9.

The goal is to locate the failure source without any new training. A direct
comparison between a sparse deployment result and the raw dense OccFM baseline
is confounded because the former uses Strong-W2Det fallback plus dynamic-only
fusion. This evaluator therefore measures six states on the exact same split:

1. Strong-W2Det anchor;
2. released dense OccFM raw prediction;
3. released dense OccFM used only as the proposal under the exact P0-F9
   same-support dynamic fusion;
4. frozen 20x20 sparse OccFM with official CFG=2 under the same fusion;
5. frozen 20x20 sparse OccFM with P0-F9 CFG=1 under the same fusion;
6. same-support GT oracle.

This isolates:
- fusion effect: dense-same-support minus Strong;
- sparse-geometry/adaptation effect: sparse-CFG2 minus dense-same-support;
- guidance-policy effect: sparse-CFG1 minus sparse-CFG2;
- finetuning effect: compare trained P0-F9 externally against sparse-CFG1.

No P0-F9 checkpoint, optimizer, EMA, semantic loss, physics condition, or new
full-context condition is used in the frozen sparse branches.
"""
from __future__ import annotations

import argparse
import gc
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
    load_official_wm,
    run_frozen_occfm_forecast,
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
PROTOCOL = "p0_f9_frozen_sparse_occfm_diagnostic_v2"


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


def _delta(a, b):
    ao, am = _metric_pair(a)
    bo, bm = _metric_pair(b)
    return {"overall": ao - bo, "moving": am - bm}


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
            "reason": "baseline reproduction check is only valid for the full validation set",
        }
    data = json.loads(Path(path).read_text())
    if data.get("protocol") != "p0_f9_official_occfm_native_baseline_v2":
        raise RuntimeError("dense reference is not the audited official OccFM baseline")
    if int(data.get("num_windows", -1)) != len(ds):
        raise RuntimeError("dense reference window count differs from diagnostic cache")
    expected_sha = file_sha256(ds.root / "index.json")
    if data.get("cache_index_sha256") != expected_sha:
        raise RuntimeError("dense reference was evaluated on a different cache index")
    return data


def _assert_dense_replay_matches(reference, dense_report) -> dict | None:
    if not reference or reference.get("status") == "skipped_for_partial_run":
        return None
    ref_o = float(reference["metrics"]["overall"]["mIoU"])
    ref_m = float(reference["metrics"]["moving"]["mIoU"])
    got_o, got_m = _metric_pair(dense_report)
    diff_o = got_o - ref_o
    diff_m = got_m - ref_m
    if abs(diff_o) > 1e-9 or abs(diff_m) > 1e-9:
        raise RuntimeError(
            "official dense replay does not reproduce the existing baseline: "
            f"dOverall={diff_o:.12g} dMoving={diff_m:.12g}"
        )
    return {
        "status": "bit_metric_exact",
        "overall_difference": diff_o,
        "moving_difference": diff_m,
    }


def _sample_payload(s, device):
    required = (
        "eval_future_gt_occ",
        "eval_strong_anchor_occ",
        "eval_repair_target_occ",
        "eval_gt_moving_support",
    )
    missing = [k for k in required if k not in s]
    if missing:
        raise RuntimeError(f"{s['sample_id']}: eval payload missing {missing}")
    write_lat = s["msp_write_support_latent"].bool()
    write_bev = latent_support_to_bev(write_lat, (200, 200)).cpu().numpy().astype(bool)
    return {
        "gt": s["eval_future_gt_occ"].cpu().numpy(),
        "anchor": s["eval_strong_anchor_occ"].cpu().numpy(),
        "repair_target": s["eval_repair_target_occ"].cpu().numpy(),
        "moving": s["eval_gt_moving_support"].cpu().numpy().astype(bool),
        "write_bev": write_bev,
        "write_ratio": float(write_lat.float().mean().item()),
        "trajectory": s["trajectory"].to(device),
    }


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
    n_eval = len(ds) if int(a.max_windows) <= 0 else min(len(ds), int(a.max_windows))
    dense_reference = _load_dense_reference(a.dense_baseline_json, ds, n_eval)

    cfg = load_occfm_config(UP, "tools/cfgs/occfm_fut.yaml")
    if int(cfg.DATA_CONFIG.HIST_LAST) != HIST_LAST:
        raise RuntimeError("pinned official OccFM HIST_LAST changed")
    if int(cfg.LOSS.SAMPLE_STEP) != 10 or float(cfg.LOSS.ALPHA_STEP) != 3.0:
        raise RuntimeError("pinned official OccFM sampler schedule changed")
    if float(cfg.LOSS.UNCOND_SCALE) != OFFICIAL_CFG:
        raise RuntimeError("pinned official OccFM guidance scale changed")

    # ------------------------------------------------------------------
    # Pass A: replay the exact released dense OccFM and apply the *same*
    # P0-F9 dynamic fusion. This is the proper control for sparse geometry.
    # ------------------------------------------------------------------
    dense_wm, dense_cfg = load_official_wm(UP, a.occfm_ckpt, device)
    if int(dense_cfg.DATA_CONFIG.HIST_LAST) != HIST_LAST:
        raise RuntimeError("loaded official OccFM config HIST_LAST mismatch")
    dense_states = {
        "strong_anchor": _new_metrics(),
        "dense_official_raw": _new_metrics(),
        "dense_official_same_support_fusion": _new_metrics(),
    }

    for i in range(n_eval):
        s = ds[i]
        payload = _sample_payload(s, device)
        sample_seed = deterministic_sample_seed(
            str(s["sample_id"]), a.seed, stream="forecast"
        )
        dense_pred = run_frozen_occfm_forecast(
            dense_wm,
            s["full_history_latent"],
            s["gt_future_latent"],
            trajectory=s["trajectory"],
            seed=sample_seed,
            hist_last=HIST_LAST,
        ).numpy()

        for horizon, fi in REPORT.items():
            gt = payload["gt"][fi]
            anchor = payload["anchor"][fi]
            moving = payload["moving"][fi]
            _update(dense_states["strong_anchor"], horizon, anchor, gt, moving)
            _update(dense_states["dense_official_raw"], horizon, dense_pred[fi], gt, moving)
            dense_fused = apply_dynamic_repair(
                anchor,
                dense_pred[fi],
                payload["write_bev"][fi],
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=FREE,
            )
            _update(
                dense_states["dense_official_same_support_fusion"],
                horizon,
                dense_fused,
                gt,
                moving,
            )
        if i % 8 == 0:
            print("dense_control_eval", i, s["sample_id"])

    dense_reports = {name: _report(state) for name, state in dense_states.items()}
    dense_reproduction = _assert_dense_replay_matches(
        dense_reference, dense_reports["dense_official_raw"]
    )

    del dense_wm
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Pass B: frozen 20x20 sparse adaptation. Only released transition
    # weights are loaded. All new P0-F9 condition paths remain exact no-ops.
    # ------------------------------------------------------------------
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

    sparse_states = {
        "frozen_sparse_official_cfg": _new_metrics(),
        "frozen_sparse_p0f9_cfg": _new_metrics(),
        "same_support_gt_oracle": _new_metrics(),
    }
    oracle_checks = 0
    valid_windows = []
    write_ratios = []
    use_amp = bool(a.amp and device.type == "cuda")

    for i in range(n_eval):
        s = ds[i]
        payload = _sample_payload(s, device)
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
        write_ratios.append(payload["write_ratio"])

        predictions = {}
        if bool(flat_valid.any()):
            def flat(x):
                return x.reshape(B * K, *x.shape[2:])[flat_valid]

            fh = flat(hist_w)
            fp_zero = torch.zeros_like(flat(physics_w))
            orig = plan.origins.reshape(B * K, 2)[flat_valid]
            traj = payload["trajectory"].unsqueeze(0)
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
                fused_latent = scatter_windows(
                    padded.reshape(B, K, *pred.shape[1:]),
                    plan,
                    base=physics_full,
                )
                predictions[name] = vae.decode_labels(fused_latent.float())[0].cpu().numpy()
        else:
            predictions["frozen_sparse_official_cfg"] = payload["anchor"]
            predictions["frozen_sparse_p0f9_cfg"] = payload["anchor"]

        for horizon, fi in REPORT.items():
            gt = payload["gt"][fi]
            anchor = payload["anchor"][fi]
            moving = payload["moving"][fi]
            for name in ("frozen_sparse_official_cfg", "frozen_sparse_p0f9_cfg"):
                final = apply_dynamic_repair(
                    anchor,
                    predictions[name][fi],
                    payload["write_bev"][fi],
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=FREE,
                )
                _update(sparse_states[name], horizon, final, gt, moving)

            oracle = apply_dynamic_repair(
                anchor,
                gt,
                payload["write_bev"][fi],
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=FREE,
            )
            if not np.array_equal(oracle, payload["repair_target"][fi]):
                raise RuntimeError(
                    f"{s['sample_id']} horizon={horizon}: same-support GT oracle differs "
                    "from cached repair target"
                )
            oracle_checks += 1
            _update(sparse_states["same_support_gt_oracle"], horizon, oracle, gt, moving)

        if i % 8 == 0:
            print("frozen_sparse_occfm_eval", i, s["sample_id"])

    sparse_reports = {name: _report(state) for name, state in sparse_states.items()}
    metrics = {**dense_reports, **sparse_reports}

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
        "dense_baseline_reference": dense_reference,
        "dense_baseline_reproduction": dense_reproduction,
        "metrics": metrics,
        "controlled_deltas": {
            "fusion_effect_dense_same_support_minus_strong": _delta(
                metrics["dense_official_same_support_fusion"],
                metrics["strong_anchor"],
            ),
            "sparse_geometry_effect_cfg2_minus_dense_same_support": _delta(
                metrics["frozen_sparse_official_cfg"],
                metrics["dense_official_same_support_fusion"],
            ),
            "guidance_effect_cfg1_minus_cfg2": _delta(
                metrics["frozen_sparse_p0f9_cfg"],
                metrics["frozen_sparse_official_cfg"],
            ),
            "frozen_sparse_cfg1_minus_strong": _delta(
                metrics["frozen_sparse_p0f9_cfg"],
                metrics["strong_anchor"],
            ),
            "oracle_headroom_minus_strong": _delta(
                metrics["same_support_gt_oracle"],
                metrics["strong_anchor"],
            ),
        },
        "interpretation_contract": {
            "dense_same_support_below_strong": (
                "the dynamic takeover fusion is unsafe even when the proposal comes from full dense OccFM"
            ),
            "sparse_cfg2_below_dense_same_support": (
                "20x20 sparse geometry/adaptation loses additional native OccFM capability before finetuning"
            ),
            "sparse_cfg1_below_sparse_cfg2": (
                "changing released OccFM guidance scale 2 -> 1 adds further degradation"
            ),
            "trained_p0f9_below_frozen_sparse_cfg1": (
                "Stage-1 finetuning/objective adds degradation beyond the frozen sparse adaptation"
            ),
        },
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== P0-F9 FROZEN SPARSE CONTROLLED DIAGNOSTIC ===")
    print(f"{'state':42s} {'Overall':>10s} {'Moving':>10s}")
    for name in (
        "strong_anchor",
        "dense_official_raw",
        "dense_official_same_support_fusion",
        "frozen_sparse_official_cfg",
        "frozen_sparse_p0f9_cfg",
        "same_support_gt_oracle",
    ):
        o, m = _metric_pair(metrics[name])
        print(f"{name:42s} {o:10.4f} {m:10.4f}")
    print("\ncontrolled_deltas")
    for name, d in report["controlled_deltas"].items():
        print(f"{name:52s} Overall={d['overall']:+.4f} Moving={d['moving']:+.4f}")
    print("saved", out)


if __name__ == "__main__":
    main()
