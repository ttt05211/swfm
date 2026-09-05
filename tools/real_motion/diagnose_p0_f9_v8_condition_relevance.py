#!/usr/bin/env python3
"""P0-F9 v8 diagnostic: do the learned context / physics conditions matter at deployment?

No training is performed.  The exact v7 checkpoint is replayed three times on the
same validation samples and with the same per-sample Gaussian source:

- full: normal v7 deployment (context + physics condition)
- no_context: remove only the 40x40 history-context condition
- no_physics: zero only the Strong-W2Det latent condition inside the WM

The sparse routing, NFE=10 sampler, VAE decoder, Strong-W2Det fallback outside
Top-2, takeover fusion and evaluation support are unchanged.  The purpose is to
measure functional relevance of the two new conditioning paths before changing
architecture.  In particular, a no-context drop supports investigating a better
temporal context representation, but does not by itself prove that an ordered
residual context module will improve results.
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

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.model_ema import ModelEMA
from real_motion.models.p0_f9 import P0_F9_PROTOCOL, make_p0_f9_model
from real_motion.motion_edit_diagnostics import MotionEditAccumulator
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.repair_target import apply_dynamic_repair
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe

PROTOCOL = "p0_f9_v8_condition_relevance_v1"
V7_TRAINING_PROTOCOL = "p0_f9_v7_native_fm_only_v1"
VARIANTS = {
    "full": {"physics_condition": True, "context_condition": True},
    "no_context": {"physics_condition": True, "context_condition": False},
    "no_physics": {"physics_condition": False, "context_condition": True},
}
REPORT_FRAMES = {"1.0": 1, "2.0": 3, "3.0": 5}


def _validate_checkpoint(ck: dict, ds: MSPWorldModelCacheDataset, occfm_ckpt: str, vae_ckpt: str) -> dict:
    arch = ck.get("architecture", {})
    if arch.get("protocol") != P0_F9_PROTOCOL or int(arch.get("stage", -1)) != 1:
        raise RuntimeError("checkpoint is not audited P0-F9 Stage-1")
    if arch.get("training_protocol") != V7_TRAINING_PROTOCOL:
        raise RuntimeError("checkpoint is not the v7 native-FM-only control")
    if arch.get("training_objective") != "native_flow_matching_velocity_mse_only":
        raise RuntimeError("checkpoint objective is not native FM-only")
    if bool(arch.get("semantic_auxiliary", True)):
        raise RuntimeError("checkpoint unexpectedly used semantic auxiliary training")
    if int(arch.get("native_backbone_hist_last", -1)) != safe.HIST_LAST:
        raise RuntimeError("checkpoint HIST_LAST contract differs")
    if ck.get("val_cache_index_sha256") != file_sha256(ds.root / "index.json"):
        raise RuntimeError("checkpoint validation cache differs from --cache")
    if ck.get("upstream_checkpoint_sha256") != file_sha256(occfm_ckpt):
        raise RuntimeError("--occfm-ckpt differs from checkpoint parent")
    if ck.get("vae_checkpoint_sha256") != file_sha256(vae_ckpt):
        raise RuntimeError("--vae-ckpt differs from checkpoint/cache latent provenance")
    return arch


def _state_from_checkpoint(ck: dict, use_ema: bool) -> tuple[dict, str]:
    if use_ema:
        ema = ck.get("ema")
        if not isinstance(ema, dict) or "state_dict" not in ema:
            raise RuntimeError("checkpoint has no EMA state")
        return ema["state_dict"], "ema"
    if "state_dict" not in ck:
        raise RuntimeError("checkpoint has no raw state_dict")
    return ck["state_dict"], "raw"


def _make_model(arch: dict, state: dict, device: torch.device):
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


def _new_variant_state():
    return {
        "metrics": safe._new_metrics(),
        "edit_all": MotionEditAccumulator(tuple(DYNAMIC_CLASS_IDS)),
        "edit_by_frame": [MotionEditAccumulator(tuple(DYNAMIC_CLASS_IDS)) for _ in range(6)],
    }


def _update_variant(state: dict, payload: dict, proposal, *, horizon: float, frame_index: int):
    anchor = payload["anchor"][frame_index]
    gt = payload["gt"][frame_index]
    moving = payload["moving"][frame_index]
    write = payload["write_bev"][frame_index]
    final = apply_dynamic_repair(
        anchor,
        proposal[frame_index],
        write,
        dynamic_class_ids=DYNAMIC_CLASS_IDS,
        free_label=safe.FREE,
    )
    safe._update(state["metrics"], horizon, final, gt, moving)


def _finalize_variant(state: dict) -> dict:
    by_frame = {str(fi): state["edit_by_frame"][fi].compute() for fi in range(6)}
    return {
        "deployment": safe._report(state["metrics"]),
        "physical_edits_all_6_frames": state["edit_all"].compute(),
        "physical_edits_by_frame": by_frame,
        "physical_edits_report_horizons": {
            h: by_frame[str(fi)] for h, fi in REPORT_FRAMES.items()
        },
    }


def _pair(report: dict) -> tuple[float, float]:
    return (
        float(report["deployment"]["overall"]["mIoU"]),
        float(report["deployment"]["moving"]["mIoU"]),
    )


def _delta(full: dict, ablated: dict) -> dict:
    fo, fm = _pair(full)
    ao, am = _pair(ablated)
    return {
        "overall_full_minus_ablated": fo - ao,
        "moving_full_minus_ablated": fm - am,
    }


def _print_table(results: dict):
    full_o, full_m = _pair(results["full"])
    print("=== P0-F9 v8 CONDITION RELEVANCE ===")
    print(f"{'variant':<16} {'Overall':>10} {'Moving':>10} {'dOverall':>10} {'dMoving':>10}")
    for name in VARIANTS:
        o, m = _pair(results[name])
        print(f"{name:<16} {o:10.4f} {m:10.4f} {o-full_o:+10.4f} {m-full_m:+10.4f}")

    print("\n=== PHYSICAL EDITS @ 1s / 2s / 3s ===")
    for name in VARIANTS:
        print(f"[{name}]")
        rows = results[name]["physical_edits_report_horizons"]
        for h in ("1.0", "2.0", "3.0"):
            r = rows[h]
            print(
                f"  {h}s clearR={r['clear_recall']:.4f} stale={r['stale_dynamic_rate']:.4f} "
                f"wrongClr={r['wrong_clear_rate']:.4f} writeR={r['write_recall']:.4f} "
                f"writeP={r['write_precision']:.4f} dynP={r['proposal_dynamic_precision']:.4f} "
                f"dynR={r['proposal_dynamic_recall']:.4f} "
                f"vol={r['dynamic_volume_ratio_proposal_over_gt']:.4f}"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--occfm-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--trained-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-windows", type=int, default=0,
                   help="0=full validation set; small positive value is smoke only")
    a = p.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("condition relevance diagnostic requires CUDA")

    ds = MSPWorldModelCacheDataset(a.cache)
    safe._validate_cache(ds, a.vae_ckpt)
    n_eval = len(ds) if int(a.max_windows) <= 0 else min(len(ds), int(a.max_windows))

    ck = torch.load(a.trained_checkpoint, map_location="cpu", weights_only=False)
    arch = _validate_checkpoint(ck, ds, a.occfm_ckpt, a.vae_ckpt)
    state, state_source = _state_from_checkpoint(ck, bool(a.use_ema))
    model = _make_model(arch, state, device)

    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)

    use_amp = bool(a.amp and device.type == "cuda")
    states = {name: _new_variant_state() for name in VARIANTS}

    with torch.inference_mode():
        for i in range(n_eval):
            sample = ds[i]
            payload = safe._sample_payload(sample, device)
            prepared = safe._prepare_sparse_sample(sample, device, with_context=True)

            proposals = {}
            for name, cfg in VARIANTS.items():
                proposals[name] = safe._decode_sparse_prediction(
                    model,
                    vae,
                    sample,
                    prepared,
                    seed=int(a.seed),
                    use_amp=use_amp,
                    guidance_scale=float(a.guidance_scale),
                    physics_condition=bool(cfg["physics_condition"]),
                    context_condition=bool(cfg["context_condition"]),
                )

            for name, proposal in proposals.items():
                states[name]["edit_all"].update(
                    payload["anchor"], proposal, payload["gt"], payload["write_bev"]
                )
                for fi in range(6):
                    states[name]["edit_by_frame"][fi].update(
                        payload["anchor"][fi],
                        proposal[fi],
                        payload["gt"][fi],
                        payload["write_bev"][fi],
                    )
                for horizon, fi in safe.REPORT.items():
                    _update_variant(
                        states[name], payload, proposal, horizon=float(horizon), frame_index=int(fi)
                    )

            if i % 8 == 0:
                print("condition_probe", i, sample["sample_id"])

    results = {name: _finalize_variant(states[name]) for name in VARIANTS}
    comparisons = {
        "context_value_full_minus_no_context": _delta(results["full"], results["no_context"]),
        "physics_value_full_minus_no_physics": _delta(results["full"], results["no_physics"]),
    }

    report = {
        "protocol": PROTOCOL,
        "num_windows": int(n_eval),
        "seed": int(a.seed),
        "guidance_scale": float(a.guidance_scale),
        "sample_steps": int(model.sample_steps),
        "checkpoint": str(Path(a.trained_checkpoint).resolve()),
        "checkpoint_state_source": state_source,
        "checkpoint_step": int(ck.get("step", -1)),
        "checkpoint_sha256": file_sha256(a.trained_checkpoint),
        "cache_index_sha256": file_sha256(ds.root / "index.json"),
        "occfm_checkpoint_sha256": file_sha256(a.occfm_ckpt),
        "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
        "variant_contract": VARIANTS,
        "results": results,
        "comparisons": comparisons,
        "interpretation_contract": {
            "context": "full > no_context means the existing context path carries useful deployment information; it does not prove an ordered-context replacement will help",
            "physics": "full > no_physics means the learned physics condition carries useful deployment information inside the WM windows",
            "fusion": "all variants use the same original takeover fusion; this diagnostic does not change the final fusion contract",
        },
    }

    _print_table(results)
    print("\n=== CONDITION CONTRIBUTION (full - ablated) ===")
    for name, row in comparisons.items():
        print(name, json.dumps(row, sort_keys=True))

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", out)

    del model, vae, vae_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
