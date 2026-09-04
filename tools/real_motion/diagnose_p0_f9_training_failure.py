#!/usr/bin/env python3
"""Diagnose why P0-F9 finetuning makes OccFM forecasting worse.

This is a no-training diagnostic.  It reuses the exact audited P0-F9 validation
cache and step checkpoint, then decomposes the failure along two axes:

1) physical motion actions inside the frozen MSP support:
   CLEAR old dynamic occupancy, KEEP valid dynamic occupancy, WRITE new dynamic
   occupancy.  This exposes stale/ghost voxels, wrong clears, missed writes, and
   false writes rather than relying on mIoU alone.

2) parameter-source decomposition:
   - frozen_sparse: released OccFM-loaded weights + fresh zero-impact P0-F9 parts
   - trained_full: the actual trained/EMA P0-F9 checkpoint
   - trained_loaded_backbone_only: only tensors that were shape-loaded from the
     released OccFM checkpoint use trained values; every non-loaded tensor is
     reset to its fresh sparse initialization
   - trained_nonofficial_only: released OccFM-loaded tensors are reset to their
     frozen values; only tensors not loaded from OccFM use trained values

The last two variants form an exact state-dict partition by the checkpoint reuse
contract.  The script also probes the native CFM task at fixed t=0.5 with the
same coherent global Gaussian source geometry, and runs the real NFE=10 rollout.

No optimizer step is taken anywhere in this script.
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

import numpy as np
import torch

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.models.p0_f9 import make_p0_f9_model
from real_motion.motion_edit_diagnostics import MotionEditAccumulator
from real_motion.native_forecast import crop_coherent_source_noise, deterministic_sample_seed
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_occfm_config, load_official_vae
from real_motion.repair_target import apply_dynamic_repair
from real_motion.windows import crop_windows
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe

PROTOCOL = "p0_f9_training_failure_diagnostic_v1"
VARIANTS = (
    "frozen_sparse",
    "trained_full",
    "trained_loaded_backbone_only",
    "trained_nonofficial_only",
)
REPORT_FRAMES = {"1.0": 1, "2.0": 3, "3.0": 5}
CONDITION_PREFIXES = (
    "transition.prior_proj.",
    "transition.context_proj.",
    "transition.physics_fusion.",
)


def _cpu_clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _checkpoint_state(ck: dict, use_ema: bool) -> tuple[dict[str, torch.Tensor], str]:
    if use_ema:
        ema = ck.get("ema")
        if not ema or "state_dict" not in ema:
            raise RuntimeError("trained P0-F9 checkpoint lacks EMA state")
        return _cpu_clone_state(ema["state_dict"]), "ema"
    if "state_dict" not in ck:
        raise RuntimeError("trained P0-F9 checkpoint lacks raw state_dict")
    return _cpu_clone_state(ck["state_dict"]), "raw"


def _partition_states(
    frozen_state: dict[str, torch.Tensor],
    trained_state: dict[str, torch.Tensor],
    loaded_transition_keys: set[str],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    if set(frozen_state) != set(trained_state):
        missing = sorted(set(frozen_state) - set(trained_state))
        extra = sorted(set(trained_state) - set(frozen_state))
        raise RuntimeError(
            f"trained/frozen state key mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    loaded_model_keys = {f"transition.{k}" for k in loaded_transition_keys}
    unknown_loaded = sorted(loaded_model_keys - set(frozen_state))
    if unknown_loaded:
        raise RuntimeError(f"official loaded keys absent from model state: {unknown_loaded[:5]}")

    variants = {
        "frozen_sparse": _cpu_clone_state(frozen_state),
        "trained_full": _cpu_clone_state(trained_state),
        "trained_loaded_backbone_only": _cpu_clone_state(frozen_state),
        "trained_nonofficial_only": _cpu_clone_state(frozen_state),
    }
    for key in frozen_state:
        if key in loaded_model_keys:
            variants["trained_loaded_backbone_only"][key] = trained_state[key].clone()
        else:
            variants["trained_nonofficial_only"][key] = trained_state[key].clone()

    nonofficial = sorted(set(frozen_state) - loaded_model_keys)
    condition_keys = sorted(k for k in nonofficial if k.startswith(CONDITION_PREFIXES))
    other_nonofficial = sorted(set(nonofficial) - set(condition_keys))
    partition = {
        "official_loaded_state_keys": len(loaded_model_keys),
        "nonofficial_state_keys": len(nonofficial),
        "condition_state_keys": condition_keys,
        "other_nonofficial_state_keys": other_nonofficial,
    }
    return variants, partition


def _drift_group(
    frozen_state: dict[str, torch.Tensor],
    trained_state: dict[str, torch.Tensor],
    keys: list[str],
) -> dict:
    n = 0
    delta_sq = 0.0
    frozen_sq = 0.0
    trained_sq = 0.0
    rows = []
    for key in keys:
        a = frozen_state[key]
        b = trained_state[key]
        if not (torch.is_tensor(a) and a.is_floating_point() and torch.is_tensor(b)):
            continue
        af = a.float()
        bf = b.float()
        d = bf - af
        numel = int(d.numel())
        if numel == 0:
            continue
        ds = float(d.square().sum().item())
        fs = float(af.square().sum().item())
        ts = float(bf.square().sum().item())
        n += numel
        delta_sq += ds
        frozen_sq += fs
        trained_sq += ts
        drms = math.sqrt(ds / numel)
        frms = math.sqrt(fs / numel)
        rows.append(
            {
                "key": key,
                "numel": numel,
                "delta_rms": drms,
                "frozen_rms": frms,
                "trained_rms": math.sqrt(ts / numel),
                "relative_delta_rms": drms / max(frms, 1e-12),
                "max_abs_delta": float(d.abs().max().item()),
            }
        )
    rows.sort(key=lambda x: x["delta_rms"], reverse=True)
    return {
        "numel": n,
        "delta_rms": math.sqrt(delta_sq / n) if n else float("nan"),
        "frozen_rms": math.sqrt(frozen_sq / n) if n else float("nan"),
        "trained_rms": math.sqrt(trained_sq / n) if n else float("nan"),
        "relative_delta_rms": (
            math.sqrt(delta_sq / n) / max(math.sqrt(frozen_sq / n), 1e-12)
            if n else float("nan")
        ),
        "top_delta_tensors": rows[:20],
    }


def _drift_report(frozen_state, trained_state, loaded_transition_keys) -> dict:
    loaded_model = sorted(f"transition.{k}" for k in loaded_transition_keys)
    nonofficial = sorted(set(frozen_state) - set(loaded_model))
    condition = sorted(k for k in nonofficial if k.startswith(CONDITION_PREFIXES))
    other = sorted(set(nonofficial) - set(condition))
    return {
        "official_loaded_backbone": _drift_group(frozen_state, trained_state, loaded_model),
        "nonofficial_total": _drift_group(frozen_state, trained_state, nonofficial),
        "new_condition_modules": _drift_group(frozen_state, trained_state, condition),
        "other_nonofficial": _drift_group(frozen_state, trained_state, other),
    }


def _prepare_probe_sample(s, device):
    prepared = safe._prepare_sparse_sample(s, device, with_context=True)
    plan = prepared["plan"]
    target_full = s["gt_future_latent"].unsqueeze(0).to(device)
    target_w = crop_windows(target_full, plan)
    B, K = prepared["B"], prepared["K"]
    effective = prepared["flat_valid"]
    prepared["target"] = target_w.reshape(B * K, *target_w.shape[2:])[effective]
    return prepared


def _new_rollout_state():
    return {
        "metrics": safe._new_metrics(),
        "edit_all": MotionEditAccumulator(tuple(DYNAMIC_CLASS_IDS)),
        "edit_by_frame": [MotionEditAccumulator(tuple(DYNAMIC_CLASS_IDS)) for _ in range(6)],
    }


def _flow_probe_one(model, ds, device, *, seed: int, use_amp: bool, n_eval: int) -> dict:
    model.eval()
    loss_sum = 0.0
    cosine_sum = 0.0
    target_rms_sum = 0.0
    pred_rms_sum = 0.0
    windows = 0
    for i in range(n_eval):
        s = ds[i]
        p = _prepare_probe_sample(s, device)
        if not bool(p["flat_valid"].any()):
            continue
        sample_seed = deterministic_sample_seed(str(s["sample_id"]), seed, stream="fm_probe")
        global_noise = safe._official_seeded_noise_like(p["physics_full"], sample_seed)
        source_noise = crop_coherent_source_noise(global_noise, p["plan"], p["flat_valid"])
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            loss, info = model.flow_loss(
                p["history"],
                p["target"],
                p["physics"],
                history_context=p["context"],
                trajectory=p["trajectory"],
                window_origins=p["origins"],
                t_override=0.5,
                source_noise=source_noise,
                return_endpoint=False,
                force_conditioned=True,
            )
        nwin = int(p["history"].shape[0])
        loss_sum += float(loss.detach().cpu()) * nwin
        cosine_sum += float(info["cosine"]) * nwin
        target_rms_sum += float(info["target_rms"]) * nwin
        pred_rms_sum += float(info["pred_rms"]) * nwin
        windows += nwin
        if i % 16 == 0:
            print("fm_probe", i, s["sample_id"])
    if windows <= 0:
        raise RuntimeError("native FM probe found no valid sparse windows")
    return {
        "t": 0.5,
        "fm_mse": loss_sum / windows,
        "velocity_cosine": cosine_sum / windows,
        "target_rms": target_rms_sum / windows,
        "pred_rms": pred_rms_sum / windows,
        "pred_over_target_rms": (pred_rms_sum / windows) / max(target_rms_sum / windows, 1e-12),
        "num_windows": windows,
        "source_seed_stream": "fm_probe",
    }


@torch.no_grad()
def _rollout_probe_one(
    model,
    vae,
    ds,
    device,
    *,
    seed: int,
    use_amp: bool,
    guidance_scale: float,
    n_eval: int,
) -> dict:
    model.eval()
    state = _new_rollout_state()
    for i in range(n_eval):
        s = ds[i]
        payload = safe._sample_payload(s, device)
        prepared = _prepare_probe_sample(s, device)
        proposal = safe._decode_sparse_prediction(
            model,
            vae,
            s,
            prepared,
            seed=seed,
            use_amp=use_amp,
            guidance_scale=guidance_scale,
            physics_condition=True,
            context_condition=True,
        )
        state["edit_all"].update(
            payload["anchor"], proposal, payload["gt"], payload["write_bev"]
        )
        for fi in range(6):
            state["edit_by_frame"][fi].update(
                payload["anchor"][fi],
                proposal[fi],
                payload["gt"][fi],
                payload["write_bev"][fi],
            )
        for horizon, fi in safe.REPORT.items():
            final = apply_dynamic_repair(
                payload["anchor"][fi],
                proposal[fi],
                payload["write_bev"][fi],
                dynamic_class_ids=DYNAMIC_CLASS_IDS,
                free_label=safe.FREE,
            )
            safe._update(
                state["metrics"],
                horizon,
                final,
                payload["gt"][fi],
                payload["moving"][fi],
            )
        if i % 8 == 0:
            print("rollout_probe", i, s["sample_id"])

    metrics = safe._report(state["metrics"])
    by_frame = {str(fi): state["edit_by_frame"][fi].compute() for fi in range(6)}
    report_h = {h: by_frame[str(fi)] for h, fi in REPORT_FRAMES.items()}
    return {
        "takeover_metrics": metrics,
        "physical_edits_all_6_frames": state["edit_all"].compute(),
        "physical_edits_by_frame": by_frame,
        "physical_edits_report_horizons": report_h,
        "guidance_scale": float(guidance_scale),
        "sample_steps": int(model.sample_steps),
    }


def _oracle_edit_reference(ds, device, *, n_eval: int) -> dict:
    all_acc = MotionEditAccumulator(tuple(DYNAMIC_CLASS_IDS))
    by_frame = [MotionEditAccumulator(tuple(DYNAMIC_CLASS_IDS)) for _ in range(6)]
    for i in range(n_eval):
        s = ds[i]
        payload = safe._sample_payload(s, device)
        all_acc.update(payload["anchor"], payload["gt"], payload["gt"], payload["write_bev"])
        for fi in range(6):
            by_frame[fi].update(
                payload["anchor"][fi], payload["gt"][fi], payload["gt"][fi], payload["write_bev"][fi]
            )
    rows = {str(fi): by_frame[fi].compute() for fi in range(6)}
    return {
        "all_6_frames": all_acc.compute(),
        "by_frame": rows,
        "report_horizons": {h: rows[str(fi)] for h, fi in REPORT_FRAMES.items()},
        "note": "GT proposal gives perfect action decisions; counts expose the required CLEAR/KEEP/WRITE population.",
    }


def _build_model_from_state(state, arch, device):
    model = make_p0_f9_model(
        20,
        sample_steps=int(arch.get("sample_steps", 10)),
        unconditional_probability=float(arch.get("unconditional_probability", 0.0)),
        guidance_scale=float(arch.get("guidance_scale", 1.0)),
        hist_last=safe.HIST_LAST,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model


def _compact_physical(row: dict) -> dict:
    keys = (
        "clear_recall",
        "stale_dynamic_rate",
        "clear_precision",
        "keep_presence_recall",
        "wrong_clear_rate",
        "write_recall",
        "write_precision",
        "false_write_rate_on_stable_non_dynamic",
        "proposal_dynamic_precision",
        "proposal_dynamic_recall",
        "dynamic_volume_ratio_proposal_over_gt",
    )
    return {k: row[k] for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--occfm-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--trained-sparse-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-windows", type=int, default=0)
    a = p.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P0-F9 training-failure diagnostic requires CUDA")
    ds = safe.MSPWorldModelCacheDataset(a.cache)
    safe._validate_cache(ds, a.vae_ckpt)
    n_eval = len(ds) if int(a.max_windows) <= 0 else min(len(ds), int(a.max_windows))
    if n_eval <= 0:
        raise RuntimeError("no validation windows selected")

    ck = torch.load(a.trained_sparse_ckpt, map_location="cpu", weights_only=False)
    arch = safe._require_trained_checkpoint_match(ck, ds, a.vae_ckpt)
    trained_state, weight_source = _checkpoint_state(ck, a.use_ema)

    cfg = load_occfm_config(UP, "tools/cfgs/occfm_fut.yaml")
    if int(cfg.DATA_CONFIG.HIST_LAST) != safe.HIST_LAST:
        raise RuntimeError("pinned official OccFM HIST_LAST changed")

    # Canonical frozen sparse state: released shape-compatible OccFM weights plus
    # fresh P0-F9 modules.  It is the exact reference for the state partition.
    canonical = make_p0_f9_model(
        20,
        sample_steps=int(arch.get("sample_steps", 10)),
        unconditional_probability=float(arch.get("unconditional_probability", 0.0)),
        guidance_scale=float(arch.get("guidance_scale", 1.0)),
        hist_last=safe.HIST_LAST,
    )
    reuse = load_shape_safe(canonical.transition, a.occfm_ckpt, verbose=True)
    reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("released OccFM-Fut checkpoint was not reused as expected")
    frozen_state = _cpu_clone_state(canonical.state_dict())
    del canonical

    variant_states, partition = _partition_states(
        frozen_state, trained_state, set(reuse.get("loaded_keys", ()))
    )
    drift = _drift_report(frozen_state, trained_state, set(reuse.get("loaded_keys", ())))

    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)
    use_amp = bool(a.amp and device.type == "cuda")

    oracle_reference = _oracle_edit_reference(ds, device, n_eval=n_eval)
    variants = {}
    for name in VARIANTS:
        print(f"\n===== evaluating {name} =====")
        model = _build_model_from_state(variant_states[name], arch, device)
        fm = _flow_probe_one(model, ds, device, seed=a.seed, use_amp=use_amp, n_eval=n_eval)
        rollout = _rollout_probe_one(
            model,
            vae,
            ds,
            device,
            seed=a.seed,
            use_amp=use_amp,
            guidance_scale=float(arch.get("guidance_scale", 1.0)),
            n_eval=n_eval,
        )
        variants[name] = {"native_fm_probe": fm, **rollout}
        del model
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "protocol": PROTOCOL,
        "num_windows": n_eval,
        "cache_index_sha256": file_sha256(ds.root / "index.json"),
        "official_occfm_checkpoint": str(Path(a.occfm_ckpt).resolve()),
        "official_occfm_checkpoint_sha256": file_sha256(a.occfm_ckpt),
        "trained_checkpoint": str(Path(a.trained_sparse_ckpt).resolve()),
        "trained_checkpoint_sha256": file_sha256(a.trained_sparse_ckpt),
        "trained_checkpoint_step": int(ck.get("step", -1)),
        "trained_weight_source": weight_source,
        "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
        "official_transition_reuse_fraction": float(reuse_fraction),
        "official_transition_loaded_tensors": int(reuse.get("loaded", 0)),
        "official_transition_target_tensors": int(reuse.get("target_total", 0)),
        "state_partition": partition,
        "parameter_drift": drift,
        "native_fm_contract": {
            "target": "absolute_gt_future_latent",
            "source": "one_global_gaussian_field_then_same_top2_crop",
            "t": 0.5,
            "conditioned": True,
        },
        "physical_edit_contract": {
            "scope": "only_voxels_inside_frozen_MSP_write_support",
            "clear": "anchor_dynamic_and_gt_non_dynamic",
            "keep": "anchor_dynamic_and_gt_dynamic",
            "write": "anchor_non_dynamic_and_gt_dynamic",
            "stale_dynamic_rate": "proposal_still_dynamic_on_required_clear_voxels",
            "wrong_clear_rate": "proposal_non_dynamic_on_required_keep_voxels",
            "instance_level_duplicate_warning": (
                "semantic occupancy has no persistent instance IDs; stale/ghost is a voxel-level proxy, not a duplicate-car count"
            ),
        },
        "gt_action_reference": oracle_reference,
        "variants": variants,
        "interpretation_contract": {
            "trained_loaded_backbone_only_bad": (
                "training drift in tensors inherited from released OccFM is sufficient to damage native forecasting"
            ),
            "trained_nonofficial_only_bad": (
                "new/nonofficial P0-F9 branches are themselves sufficient to damage the proposal"
            ),
            "trained_full_worse_than_both_parts": (
                "failure is dominated by interaction between inherited-backbone drift and learned new branches"
            ),
            "fm_worse_and_rollout_worse": (
                "finetuning degraded the native CFM task itself"
            ),
            "fm_better_but_rollout_worse": (
                "local fixed-t FM optimization improved while NFE10/decode physical forecasting worsened; objective-to-rollout mismatch is implicated"
            ),
        },
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== P0-F9 TRAINING FAILURE DECOMPOSITION ===")
    print(
        f"{'variant':38s} {'FM_MSE':>10s} {'cos':>8s} {'Overall':>9s} {'Moving':>9s} "
        f"{'clearR':>8s} {'stale':>8s} {'wrongClr':>9s} {'writeR':>8s} {'writeP':>8s}"
    )
    for name in VARIANTS:
        row = variants[name]
        fm = row["native_fm_probe"]
        o, m = safe._metric_pair(row["takeover_metrics"])
        e = row["physical_edits_all_6_frames"]
        print(
            f"{name:38s} {fm['fm_mse']:10.6f} {fm['velocity_cosine']:8.4f} "
            f"{o:9.4f} {m:9.4f} {e['clear_recall']:8.4f} {e['stale_dynamic_rate']:8.4f} "
            f"{e['wrong_clear_rate']:9.4f} {e['write_recall']:8.4f} {e['write_precision']:8.4f}"
        )

    print("\n=== PHYSICAL EDITS @ 1s / 2s / 3s ===")
    for name in VARIANTS:
        print(f"[{name}]")
        for h in ("1.0", "2.0", "3.0"):
            e = variants[name]["physical_edits_report_horizons"][h]
            c = _compact_physical(e)
            print(
                f"  {h}s clearR={c['clear_recall']:.4f} stale={c['stale_dynamic_rate']:.4f} "
                f"wrongClr={c['wrong_clear_rate']:.4f} writeR={c['write_recall']:.4f} "
                f"writeP={c['write_precision']:.4f} dynP={c['proposal_dynamic_precision']:.4f} "
                f"dynR={c['proposal_dynamic_recall']:.4f} vol={c['dynamic_volume_ratio_proposal_over_gt']:.4f}"
            )

    print("\n=== PARAMETER DRIFT ===")
    for name, row in drift.items():
        print(
            f"{name:28s} numel={row['numel']:10d} delta_rms={row['delta_rms']:.6g} "
            f"frozen_rms={row['frozen_rms']:.6g} rel={row['relative_delta_rms']:.6g}"
        )
    if partition["other_nonofficial_state_keys"]:
        print("other_nonofficial_state_keys:", partition["other_nonofficial_state_keys"])
    print("saved", out)


if __name__ == "__main__":
    main()
