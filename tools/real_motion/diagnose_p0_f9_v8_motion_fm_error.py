#!/usr/bin/env python3
"""P0-F9 v8 diagnostic: is native FM error concentrated near true motion?

No training, decoder, rollout, or backward pass is used.  The diagnostic compares
released/frozen sparse OccFM with the v7 native-FM-only checkpoint on the exact
same 128-window validation cache, coherent Gaussian z0, Top-2 crops, and fixed
FM times.  The Moving-v2 GT support is mapped from [T,200,200,Z] occupancy space
to [T,50,50] latent cells by exact 4x4 any pooling with no extra dilation.

The support is deliberately called *motion-associated support*: it contains the
dual-box old/future moving-object region used by Moving-v2, including departure
and arrival neighborhoods.  It is not an instance mask and is not fed to the
model.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import torch

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.fm_group_diagnostics import (
    finalize_grouped_velocity_accumulator,
    motion_support_occ_to_latent,
    new_grouped_velocity_accumulator,
    update_grouped_velocity_accumulator,
)
from real_motion.models.p0_f9 import P0_F9_PROTOCOL, make_p0_f9_model
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V2, MSPWorldModelCacheDataset
from real_motion.native_forecast import crop_coherent_source_noise, deterministic_sample_seed
from real_motion.occfm_io import file_sha256
from real_motion.windows import crop_windows
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL

PROTOCOL = "p0_f9_v8_motion_associated_fm_error_v1"
V7_TRAINING_PROTOCOL = "p0_f9_v7_native_fm_only_v1"
REPORT_FRAMES = {"1.0": 1, "2.0": 3, "3.0": 5}
SOURCE_STREAM = "p0_f9_v8_motion_fm_diag"


def _validate_cache(ds: MSPWorldModelCacheDataset) -> None:
    if ds.version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError("v8 FM diagnostic requires the P0-F9 v2 absolute-future cache")
    meta = ds.metadata
    checks = {
        "protocol": P0_F9_CACHE_PROTOCOL,
        "target": "absolute_gt_future_vae_latent",
        "flow_source": "gaussian_noise_not_anchor",
        "history_contract": "full_native_occ_history_6f",
        "native_backbone_hist_last": safe.HIST_LAST,
        "anchor_contract": "strong_w2det_occ_only_v1",
        "topk": 2,
        "window_hw": [20, 20],
        "context_hw": [40, 40],
        "vae_mode": "sample",
    }
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                f"validation cache mismatch for {key}: {meta.get(key)!r} != {expected!r}"
            )
    if not bool(meta.get("include_eval_payload", False)):
        raise RuntimeError("v8 diagnostic requires eval_gt_moving_support in the validation cache")
    first = ds[0]
    if "eval_gt_moving_support" not in first:
        raise RuntimeError("validation cache metadata claims eval payload but moving support is absent")


def _checkpoint_state(ck: dict, *, use_ema: bool) -> tuple[dict, str]:
    if use_ema:
        ema = ck.get("ema")
        if not isinstance(ema, dict) or "state_dict" not in ema:
            raise RuntimeError("v7 checkpoint has no EMA state")
        return ema["state_dict"], "ema"
    if "state_dict" not in ck:
        raise RuntimeError("v7 checkpoint has no raw model state")
    return ck["state_dict"], "raw"


def _validate_checkpoint(ck: dict, ds: MSPWorldModelCacheDataset, occfm_ckpt: str) -> dict:
    arch = ck.get("architecture", {})
    if arch.get("protocol") != P0_F9_PROTOCOL or int(arch.get("stage", -1)) != 1:
        raise RuntimeError("trained checkpoint is not audited P0-F9 Stage-1")
    if arch.get("training_protocol") != V7_TRAINING_PROTOCOL:
        raise RuntimeError("trained checkpoint is not the v7 native-FM-only control")
    if arch.get("training_objective") != "native_flow_matching_velocity_mse_only":
        raise RuntimeError("trained checkpoint did not use the v7 native FM-only objective")
    if bool(arch.get("semantic_auxiliary", True)):
        raise RuntimeError("trained checkpoint unexpectedly used semantic auxiliary loss")
    if int(arch.get("native_backbone_hist_last", -1)) != safe.HIST_LAST:
        raise RuntimeError("trained checkpoint HIST_LAST differs from the cache contract")
    if ck.get("val_cache_index_sha256") != file_sha256(ds.root / "index.json"):
        raise RuntimeError("trained checkpoint was validated against a different cache index")
    if ck.get("upstream_checkpoint_sha256") != file_sha256(occfm_ckpt):
        raise RuntimeError("--occfm-ckpt differs from the checkpoint parent OccFM weights")
    if ck.get("vae_checkpoint_sha256") != ds.metadata.get("vae_checkpoint_sha256"):
        raise RuntimeError("trained checkpoint/cache VAE latent provenance differs")
    return arch


def _make_model(arch: dict, device: torch.device):
    return make_p0_f9_model(
        20,
        sample_steps=int(arch.get("sample_steps", 10)),
        unconditional_probability=float(arch.get("unconditional_probability", 0.0)),
        guidance_scale=float(arch.get("guidance_scale", 1.0)),
        hist_last=safe.HIST_LAST,
    ).to(device)


def _prepare_sample(sample: dict, device: torch.device) -> dict:
    p = safe._prepare_sparse_sample(sample, device, with_context=True)
    target_full = sample["gt_future_latent"].unsqueeze(0).to(device)
    target_w = crop_windows(target_full, p["plan"])
    p["target"] = target_w.reshape(p["B"] * p["K"], *target_w.shape[2:])[p["flat_valid"]]
    return p


def _motion_mask_windows(sample: dict, prepared: dict, device: torch.device) -> torch.Tensor:
    raw = sample.get("eval_gt_moving_support")
    if raw is None:
        raise RuntimeError(f"{sample['sample_id']}: eval_gt_moving_support missing")
    latent = motion_support_occ_to_latent(raw, latent_hw=safe.FULL_HW)
    if int(latent.shape[0]) != int(prepared["target"].shape[1]):
        raise RuntimeError(
            f"{sample['sample_id']}: moving support future frames differ from latent target"
        )
    full = latent[:, None].unsqueeze(0).to(device=device, dtype=torch.float32)
    windows = crop_windows(full, prepared["plan"])
    flat = windows.reshape(
        prepared["B"] * prepared["K"], *windows.shape[2:]
    )[prepared["flat_valid"]]
    mask = flat[:, :, 0].bool()
    if tuple(mask.shape) != (
        int(prepared["target"].shape[0]),
        int(prepared["target"].shape[1]),
        int(prepared["target"].shape[-2]),
        int(prepared["target"].shape[-1]),
    ):
        raise RuntimeError("cropped moving support does not match valid Top-2 target windows")
    return mask


@torch.no_grad()
def _flow_velocity(
    model,
    prepared: dict,
    source_noise: torch.Tensor,
    *,
    t_value: float,
    use_amp: bool,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Run the native flow_loss and capture its exact predicted velocity tensor."""
    captured = {}

    def hook(_module, _inputs, output):
        if not isinstance(output, dict) or "predicted_latent" not in output:
            raise RuntimeError("transition forward contract lacks predicted_latent")
        captured["predicted_latent"] = output["predicted_latent"].detach()

    handle = model.transition.register_forward_hook(hook)
    try:
        with torch.autocast(
            device_type=prepared["history"].device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            loss, _ = model.flow_loss(
                prepared["history"],
                prepared["target"],
                prepared["physics"],
                history_context=prepared["context"],
                trajectory=prepared["trajectory"],
                window_origins=prepared["origins"],
                t_override=float(t_value),
                source_noise=source_noise,
                return_endpoint=False,
                force_conditioned=True,
            )
    finally:
        handle.remove()

    if "predicted_latent" not in captured:
        raise RuntimeError("transition hook did not capture predicted velocity")
    hist_frames = int(prepared["history"].shape[1])
    pred = captured["predicted_latent"][:, hist_frames:].float()
    target_velocity = (
        prepared["target"].float() * float(model.rescale_factor) - source_noise.float()
    )
    if pred.shape != target_velocity.shape:
        raise RuntimeError("captured predicted velocity shape differs from native FM target")
    return float(loss.detach().cpu()), pred, target_velocity


def _native_check(new_state: dict) -> dict:
    grouped = new_state["grouped"]
    overall = grouped["overall"]
    elems = int(overall["moving"]["elements"]) + int(overall["non_moving"]["elements"])
    native = new_state["native_loss_weighted_sum"] / new_state["native_loss_elements"]
    grouped_mse = overall["global_mse"]
    if elems != int(new_state["native_loss_elements"]):
        raise RuntimeError("native/grouped FM element counts differ")
    return {
        "native_flow_loss": native,
        "grouped_global_mse": grouped_mse,
        "abs_difference": abs(native - grouped_mse),
        "elements": elems,
    }


@torch.no_grad()
def _run_variant(
    name: str,
    model,
    ds: MSPWorldModelCacheDataset,
    device: torch.device,
    *,
    t_values: list[float],
    seed: int,
    use_amp: bool,
    n_eval: int,
    motion_weight_lambda: float,
) -> dict:
    model.eval().requires_grad_(False)
    states = {
        t: {
            "acc": new_grouped_velocity_accumulator(num_frames=6),
            "native_loss_weighted_sum": 0.0,
            "native_loss_elements": 0,
        }
        for t in t_values
    }
    zero_route_samples = 0

    for i in range(n_eval):
        sample = ds[i]
        prepared = _prepare_sample(sample, device)
        if not bool(prepared["flat_valid"].any()):
            zero_route_samples += 1
            continue
        motion_mask = _motion_mask_windows(sample, prepared, device)
        sample_seed = deterministic_sample_seed(
            str(sample["sample_id"]), int(seed), stream=SOURCE_STREAM
        )
        global_noise = safe._official_seeded_noise_like(prepared["physics_full"], sample_seed)
        source_noise = crop_coherent_source_noise(
            global_noise, prepared["plan"], prepared["flat_valid"]
        )

        for t in t_values:
            loss, pred_v, target_v = _flow_velocity(
                model,
                prepared,
                source_noise,
                t_value=t,
                use_amp=use_amp,
            )
            state = states[t]
            update_grouped_velocity_accumulator(
                state["acc"], pred_v, target_v, motion_mask
            )
            nelem = int(pred_v.numel())
            state["native_loss_weighted_sum"] += loss * nelem
            state["native_loss_elements"] += nelem

        if i % 8 == 0:
            print("motion_fm_probe", name, i, sample["sample_id"])

    result = {
        "num_samples_requested": int(n_eval),
        "zero_route_samples": int(zero_route_samples),
        "source_seed_stream": SOURCE_STREAM,
        "t_values": {},
    }
    for t in t_values:
        state = states[t]
        grouped = finalize_grouped_velocity_accumulator(
            state["acc"], motion_weight_lambda=motion_weight_lambda
        )
        check_state = {
            "grouped": grouped,
            "native_loss_weighted_sum": state["native_loss_weighted_sum"],
            "native_loss_elements": state["native_loss_elements"],
        }
        check = _native_check(check_state)
        # Hook capture must reproduce flow_loss up to normal AMP reduction error.
        if check["abs_difference"] > 5e-5:
            raise RuntimeError(
                f"{name} t={t}: grouped FM does not reproduce native loss; {check}"
            )
        result["t_values"][f"{t:.2f}"] = {
            "grouped": grouped,
            "native_recomposition_check": check,
        }
    return result


def _metric(row: dict, group: str, key: str):
    value = row[group].get(key)
    return None if value is None else float(value)


def _comparisons(frozen: dict, trained: dict, t_values: list[float]) -> dict:
    out = {}
    for t in t_values:
        key = f"{t:.2f}"
        fr = frozen["t_values"][key]["grouped"]["overall"]
        tr = trained["t_values"][key]["grouped"]["overall"]
        row = {}
        for group in ("moving", "non_moving"):
            for metric in ("mse", "nmse", "cosine_macro"):
                a = _metric(fr, group, metric)
                b = _metric(tr, group, metric)
                row[f"{group}_{metric}_frozen"] = a
                row[f"{group}_{metric}_v7"] = b
                row[f"{group}_{metric}_delta"] = (
                    b - a if a is not None and b is not None else None
                )
                row[f"{group}_{metric}_ratio_v7_over_frozen"] = (
                    b / a if a not in (None, 0.0) and b is not None else None
                )
        row["motion_cell_fraction"] = tr["motion_cell_fraction"]
        row["motion_error_share_frozen"] = fr["motion_squared_error_share"]
        row["motion_error_share_v7"] = tr["motion_squared_error_share"]
        out[key] = row
    return out


def _mean_t_summary(variant: dict, t_values: list[float]) -> dict:
    rows = [variant["t_values"][f"{t:.2f}"]["grouped"]["overall"] for t in t_values]

    def mean(vals):
        vals = [float(v) for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "global_mse": mean([r["global_mse"] for r in rows]),
        "moving_mse": mean([r["moving"]["mse"] for r in rows]),
        "non_moving_mse": mean([r["non_moving"]["mse"] for r in rows]),
        "moving_nmse": mean([r["moving"]["nmse"] for r in rows]),
        "non_moving_nmse": mean([r["non_moving"]["nmse"] for r in rows]),
        "moving_cosine": mean([r["moving"]["cosine_macro"] for r in rows]),
        "non_moving_cosine": mean([r["non_moving"]["cosine_macro"] for r in rows]),
        "motion_cell_fraction": mean([r["motion_cell_fraction"] for r in rows]),
        "motion_error_share": mean([r["motion_squared_error_share"] for r in rows]),
        "effective_motion_weight_mass_lambda2": mean(
            [r["effective_motion_weight_mass"] for r in rows]
        ),
    }


def _fmt(value, digits=6):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _print_report(report: dict, t_values: list[float]) -> None:
    print("\n=== P0-F9 v8 MOTION-ASSOCIATED FM ERROR DECOMPOSITION ===")
    print(
        f"{'t':>5} {'variant':<14} {'global':>9} {'movMSE':>9} {'nonMSE':>9} "
        f"{'movNMSE':>9} {'nonNMSE':>9} {'movCos':>8} {'nonCos':>8} "
        f"{'movCell%':>9} {'errShare%':>10}"
    )
    for t in t_values:
        key = f"{t:.2f}"
        for variant in ("frozen_sparse", "v7_step400"):
            row = report["variants"][variant]["t_values"][key]["grouped"]["overall"]
            print(
                f"{t:5.2f} {variant:<14} {_fmt(row['global_mse']):>9} "
                f"{_fmt(row['moving']['mse']):>9} {_fmt(row['non_moving']['mse']):>9} "
                f"{_fmt(row['moving']['nmse']):>9} {_fmt(row['non_moving']['nmse']):>9} "
                f"{_fmt(row['moving']['cosine_macro'],4):>8} "
                f"{_fmt(row['non_moving']['cosine_macro'],4):>8} "
                f"{_fmt(100.0 * row['motion_cell_fraction'],3):>9} "
                f"{_fmt(100.0 * row['motion_squared_error_share'],3):>10}"
            )

    print("\n=== REPORT HORIZONS (1s / 2s / 3s) ===")
    for t in t_values:
        key = f"{t:.2f}"
        for variant in ("frozen_sparse", "v7_step400"):
            by_frame = report["variants"][variant]["t_values"][key]["grouped"]["by_frame"]
            print(f"[{variant} t={t:.2f}]")
            for horizon, fi in REPORT_FRAMES.items():
                row = by_frame[fi]
                print(
                    f"  {horizon}s movMSE={_fmt(row['moving']['mse'])} "
                    f"nonMSE={_fmt(row['non_moving']['mse'])} "
                    f"movNMSE={_fmt(row['moving']['nmse'])} "
                    f"nonNMSE={_fmt(row['non_moving']['nmse'])} "
                    f"movCell={_fmt(100.0 * row['motion_cell_fraction'],3)}% "
                    f"errShare={_fmt(100.0 * row['motion_squared_error_share'],3)}%"
                )

    print("\n=== MULTI-T MEAN SUMMARY ===")
    for variant in ("frozen_sparse", "v7_step400"):
        s = report["multi_t_mean"][variant]
        print(
            f"{variant:<14} global={_fmt(s['global_mse'])} "
            f"movMSE={_fmt(s['moving_mse'])} nonMSE={_fmt(s['non_moving_mse'])} "
            f"movNMSE={_fmt(s['moving_nmse'])} nonNMSE={_fmt(s['non_moving_nmse'])} "
            f"movCell={_fmt(100.0*s['motion_cell_fraction'],3)}% "
            f"errShare={_fmt(100.0*s['motion_error_share'],3)}% "
            f"lambda2Mass={_fmt(100.0*s['effective_motion_weight_mass_lambda2'],3)}%"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--occfm-ckpt", required=True)
    p.add_argument("--trained-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--t-values", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--motion-weight-lambda", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()

    t_values = sorted(set(float(x) for x in a.t_values))
    if not t_values or any(not 0.0 < t < 1.0 for t in t_values):
        raise ValueError("all --t-values must lie strictly inside (0,1)")
    if float(a.motion_weight_lambda) < 0.0:
        raise ValueError("--motion-weight-lambda must be non-negative")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = MSPWorldModelCacheDataset(a.cache)
    _validate_cache(ds)
    n_eval = len(ds) if a.max_samples is None else min(len(ds), int(a.max_samples))
    if n_eval <= 0:
        raise ValueError("--max-samples leaves no samples")

    ck = torch.load(a.trained_checkpoint, map_location="cpu", weights_only=False)
    arch = _validate_checkpoint(ck, ds, a.occfm_ckpt)
    trained_state, state_source = _checkpoint_state(ck, use_ema=bool(a.use_ema))

    frozen = _make_model(arch, device)
    reuse = load_shape_safe(frozen.transition, a.occfm_ckpt, verbose=True)
    official_reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)
    safe._assert_new_conditioning_is_noop(frozen)
    frozen_result = _run_variant(
        "frozen_sparse",
        frozen,
        ds,
        device,
        t_values=t_values,
        seed=a.seed,
        use_amp=bool(a.amp and device.type == "cuda"),
        n_eval=n_eval,
        motion_weight_lambda=float(a.motion_weight_lambda),
    )
    del frozen
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trained = _make_model(arch, device)
    trained.load_state_dict(trained_state, strict=True)
    trained_result = _run_variant(
        "v7_step400",
        trained,
        ds,
        device,
        t_values=t_values,
        seed=a.seed,
        use_amp=bool(a.amp and device.type == "cuda"),
        n_eval=n_eval,
        motion_weight_lambda=float(a.motion_weight_lambda),
    )
    del trained
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report = {
        "protocol": PROTOCOL,
        "purpose": "test whether native FM error is concentrated on motion-associated latent support before any motion-weighted training",
        "num_samples": int(n_eval),
        "cache": str(Path(a.cache).resolve()),
        "cache_index_sha256": file_sha256(ds.root / "index.json"),
        "occfm_checkpoint": str(Path(a.occfm_ckpt).resolve()),
        "occfm_checkpoint_sha256": file_sha256(a.occfm_ckpt),
        "trained_checkpoint": str(Path(a.trained_checkpoint).resolve()),
        "trained_checkpoint_sha256": file_sha256(a.trained_checkpoint),
        "trained_state_source": state_source,
        "trained_step": int(ck.get("step", -1)),
        "official_reuse_fraction": float(official_reuse_fraction),
        "t_values": t_values,
        "motion_support_contract": {
            "source": "eval_gt_moving_support",
            "meaning": "Moving-v2 dual-box motion-associated support; not exact instance identity",
            "occupancy_shape": "[T,X,Y,Z]",
            "mapping": "any_Z_then_exact_block_any_pool_to_50x50_no_extra_dilation",
            "crop_contract": "same frozen Top-2 WindowPlan; overlap counted as in FM training",
            "used_as_model_condition": False,
        },
        "statistics": {
            "dtype": "FP32 accumulation",
            "mse": "micro exact sum over channel elements",
            "nmse": "group SSE / group target-velocity squared sum",
            "cosine": "macro mean over valid sample-horizon group vectors",
            "motion_weight_lambda_for_mass_preview_only": float(a.motion_weight_lambda),
        },
        "variants": {
            "frozen_sparse": frozen_result,
            "v7_step400": trained_result,
        },
    }
    report["comparisons"] = _comparisons(frozen_result, trained_result, t_values)
    report["multi_t_mean"] = {
        "frozen_sparse": _mean_t_summary(frozen_result, t_values),
        "v7_step400": _mean_t_summary(trained_result, t_values),
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report, t_values)
    print("saved", out)


if __name__ == "__main__":
    main()
