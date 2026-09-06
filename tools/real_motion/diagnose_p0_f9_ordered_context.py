#!/usr/bin/env python3
"""Deployment diagnostic for the MCTRL / MT ordered-context experiment.

This mirrors the established P0-F9 training-failure decomposition while building
the architecture declared by the checkpoint, including the ordered-context
module.  It reports fixed-t native FM, NFE=10 rollout Overall/Moving, physical
CLEAR/WRITE/stale/dynamic-volume metrics, and parameter-source drift.
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

import torch

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.models.p0_f9 import make_p0_f9_model
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_occfm_config, load_official_vae
from tools.real_motion import diagnose_p0_f9_training_failure as base
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe


PROTOCOL = "p0_f9_ordered_context_deployment_diagnostic_v1"
ORDERED_PREFIX = "transition.ordered_context_proj."


def _make_model(arch: dict, device):
    return make_p0_f9_model(
        20,
        sample_steps=int(arch.get("sample_steps", 10)),
        unconditional_probability=float(arch.get("unconditional_probability", 0.0)),
        guidance_scale=float(arch.get("guidance_scale", 1.0)),
        hist_last=safe.HIST_LAST,
        ordered_context=bool(arch.get("ordered_context_module", False)),
        ordered_context_enabled=bool(arch.get("ordered_context_enabled", True)),
    ).to(device)


def _partition_states(frozen_state, trained_state, loaded_transition_keys):
    variants, partition = base._partition_states(
        frozen_state, trained_state, loaded_transition_keys
    )
    condition = sorted(
        set(partition["condition_state_keys"])
        | {k for k in frozen_state if k.startswith(ORDERED_PREFIX)}
    )
    loaded_model_keys = {f"transition.{k}" for k in loaded_transition_keys}
    nonofficial = sorted(set(frozen_state) - loaded_model_keys)
    partition["condition_state_keys"] = condition
    partition["other_nonofficial_state_keys"] = sorted(set(nonofficial) - set(condition))
    return variants, partition


def _drift_report(frozen_state, trained_state, loaded_transition_keys):
    loaded_model = sorted(f"transition.{k}" for k in loaded_transition_keys)
    nonofficial = sorted(set(frozen_state) - set(loaded_model))
    base_prefixes = tuple(base.CONDITION_PREFIXES) + (ORDERED_PREFIX,)
    condition = sorted(k for k in nonofficial if k.startswith(base_prefixes))
    other = sorted(set(nonofficial) - set(condition))
    ordered = sorted(k for k in nonofficial if k.startswith(ORDERED_PREFIX))
    return {
        "official_loaded_backbone": base._drift_group(frozen_state, trained_state, loaded_model),
        "nonofficial_total": base._drift_group(frozen_state, trained_state, nonofficial),
        "new_condition_modules": base._drift_group(frozen_state, trained_state, condition),
        "ordered_context_only": base._drift_group(frozen_state, trained_state, ordered),
        "other_nonofficial": base._drift_group(frozen_state, trained_state, other),
    }


def _build_model_from_state(state, arch, device):
    model = _make_model(arch, device)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model


def main() -> None:
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
        raise RuntimeError("ordered-context deployment diagnostic requires CUDA")
    ds = safe.MSPWorldModelCacheDataset(a.cache)
    safe._validate_cache(ds, a.vae_ckpt)
    n_eval = len(ds) if int(a.max_windows) <= 0 else min(len(ds), int(a.max_windows))
    if n_eval <= 0:
        raise RuntimeError("no validation windows selected")

    ck = torch.load(a.trained_sparse_ckpt, map_location="cpu", weights_only=False)
    arch = safe._require_trained_checkpoint_match(ck, ds, a.vae_ckpt)
    if not bool(arch.get("ordered_context_module", False)):
        raise RuntimeError("checkpoint does not declare ordered-context architecture")
    trained_state, weight_source = base._checkpoint_state(ck, a.use_ema)

    cfg = load_occfm_config(UP, "tools/cfgs/occfm_fut.yaml")
    if int(cfg.DATA_CONFIG.HIST_LAST) != safe.HIST_LAST:
        raise RuntimeError("pinned official OccFM HIST_LAST changed")

    canonical = _make_model(arch, torch.device("cpu"))
    reuse = load_shape_safe(canonical.transition, a.occfm_ckpt, verbose=True)
    reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("released OccFM-Fut checkpoint was not reused as expected")
    frozen_state = base._cpu_clone_state(canonical.state_dict())
    del canonical

    variant_states, partition = _partition_states(
        frozen_state, trained_state, set(reuse.get("loaded_keys", ()))
    )
    drift = _drift_report(frozen_state, trained_state, set(reuse.get("loaded_keys", ())))

    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)
    use_amp = bool(a.amp)
    oracle_reference = base._oracle_edit_reference(ds, device, n_eval=n_eval)

    variants = {}
    for name in base.VARIANTS:
        print(f"\n===== evaluating {name} =====")
        model = _build_model_from_state(variant_states[name], arch, device)
        fm = base._flow_probe_one(model, ds, device, seed=a.seed, use_amp=use_amp, n_eval=n_eval)
        rollout = base._rollout_probe_one(
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
        "trained_checkpoint": str(Path(a.trained_sparse_ckpt).resolve()),
        "trained_checkpoint_sha256": file_sha256(a.trained_sparse_ckpt),
        "trained_checkpoint_step": int(ck.get("phase_step", ck.get("step", -1))),
        "trained_weight_source": weight_source,
        "architecture": arch,
        "ordered_context_enabled": bool(arch.get("ordered_context_enabled", False)),
        "official_occfm_checkpoint_sha256": file_sha256(a.occfm_ckpt),
        "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
        "official_transition_reuse_fraction": float(reuse_fraction),
        "state_partition": partition,
        "parameter_drift": drift,
        "gt_action_reference": oracle_reference,
        "variants": variants,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== P0-F9 ORDERED CONTEXT DEPLOYMENT DECOMPOSITION ===")
    print(
        f"{'variant':40s} {'FM_MSE':>10s} {'cos':>8s} {'Overall':>9s} {'Moving':>9s} "
        f"{'clearR':>8s} {'stale':>8s} {'wrongClr':>9s} {'writeR':>8s} {'writeP':>8s}"
    )
    for name in base.VARIANTS:
        row = variants[name]
        fm = row["native_fm_probe"]
        metrics = row["takeover_metrics"]
        edit = row["physical_edits_all_6_frames"]
        print(
            f"{name:40s} {fm['fm_mse']:10.6f} {fm['velocity_cosine']:8.4f} "
            f"{metrics['overall']['mIoU']:9.4f} {metrics['moving']['mIoU']:9.4f} "
            f"{edit['clear_recall']:8.4f} {edit['stale_dynamic_rate']:8.4f} "
            f"{edit['wrong_clear_rate']:9.4f} {edit['write_recall']:8.4f} {edit['write_precision']:8.4f}"
        )

    print("\n=== PHYSICAL EDITS @ 1s / 2s / 3s ===")
    for name in base.VARIANTS:
        print(f"[{name}]")
        rows = variants[name]["physical_edits_report_horizons"]
        for h in ("1.0", "2.0", "3.0"):
            r = rows[h]
            print(
                f"  {h}s clearR={r['clear_recall']:.4f} stale={r['stale_dynamic_rate']:.4f} "
                f"wrongClr={r['wrong_clear_rate']:.4f} writeR={r['write_recall']:.4f} "
                f"writeP={r['write_precision']:.4f} dynP={r['proposal_dynamic_precision']:.4f} "
                f"dynR={r['proposal_dynamic_recall']:.4f} vol={r['dynamic_volume_ratio_proposal_over_gt']:.4f}"
            )

    print("\n=== PARAMETER DRIFT ===")
    for key in (
        "official_loaded_backbone",
        "nonofficial_total",
        "new_condition_modules",
        "ordered_context_only",
        "other_nonofficial",
    ):
        r = drift[key]
        print(
            f"{key:28s} numel={r['numel']:10d} delta_rms={r['delta_rms']:.6g} "
            f"frozen_rms={r['frozen_rms']:.6g} rel={r['relative_delta_rms']:.6g}"
        )
    print("saved", out)


if __name__ == "__main__":
    main()
