#!/usr/bin/env python3
"""Run and automatically audit a short, real-GPU P0-F8 v2 smoke test.

This entrypoint deliberately calls the public P0-F8 v2 trainer instead of a
mocked or reduced model.  It fixes the smoke schedule to one validation at the
final step and then reloads the produced checkpoint with strict model-state
validation.  Passing this command establishes an engineering/runtime gate; it
does not establish that P0-F8 improves deployment metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import torch

from real_motion.models.p0_f8 import make_p0_f8_model
from tools.real_motion.p0_f8_train_impl_v2 import F8_PROTOCOL

SMOKE_PROTOCOL = "p0_f8_anchor_relative_edit_wm_v2_gpu_smoke_v1"
REQUIRED_V2_FIELDS = (
    "balanced_false_edit_rate",
    "pool_false_edit_rate",
    "dynamic_keep_fraction_realized",
    "num_lovasz_voxels",
    "num_dynamic_keeps",
    "num_background_keeps",
    "num_pool_keeps",
    "num_pool_dynamic_keeps",
    "num_pool_background_keeps",
)


def validate_smoke_checkpoint(ck: dict, *, expected_steps: int) -> dict:
    """Validate the saved v2 protocol and population-accounting contract."""
    arch = ck.get("architecture") or {}
    if arch.get("protocol") != F8_PROTOCOL:
        raise RuntimeError("smoke checkpoint is not P0-F8 v2")
    if int(ck.get("step", -1)) != int(expected_steps):
        raise RuntimeError(
            f"smoke checkpoint step {ck.get('step')} != expected {expected_steps}"
        )
    history = ck.get("training_history") or []
    if not history:
        raise RuntimeError("smoke checkpoint has no validation history")
    latest = history[-1]
    if int(latest.get("step", -1)) != int(expected_steps):
        raise RuntimeError("latest validation record is not the final smoke step")
    train = latest.get("train") or {}
    val = latest.get("val") or {}
    missing_train = [key for key in REQUIRED_V2_FIELDS if key not in train]
    missing_val = [key for key in REQUIRED_V2_FIELDS if key not in val]
    if missing_train:
        raise RuntimeError(f"smoke train record lacks v2 fields {missing_train}")
    if missing_val:
        raise RuntimeError(f"smoke validation record lacks v2 fields {missing_val}")
    if int(val["num_lovasz_voxels"]) < int(val["num_supervised_voxels"]):
        raise RuntimeError("Lovasz population is smaller than balanced CE population")
    dynamic_fraction = float(val["dynamic_keep_fraction_realized"])
    if not 0.0 <= dynamic_fraction <= 1.0:
        raise RuntimeError("invalid realized dynamic KEEP fraction")
    edit_lambda = float(ck.get("edit_lambda", 0.0))
    if not edit_lambda > 0.0:
        raise RuntimeError("smoke checkpoint has no positive edit lambda")
    if not isinstance(ck.get("state_dict"), dict) or not ck["state_dict"]:
        raise RuntimeError("smoke checkpoint has no model state")
    return {
        "protocol": SMOKE_PROTOCOL,
        "status": "PASS",
        "steps": int(expected_steps),
        "p0_f8_protocol": arch["protocol"],
        "edit_lambda": edit_lambda,
        "num_supervised_voxels": int(val["num_supervised_voxels"]),
        "num_lovasz_voxels": int(val["num_lovasz_voxels"]),
        "num_pool_keeps": int(val["num_pool_keeps"]),
        "dynamic_keep_fraction_realized": dynamic_fraction,
        "pool_false_edit_rate": float(val["pool_false_edit_rate"]),
        "checkpoint_reload": "pending_strict_model_load",
    }


def _strict_reload_model(ck: dict) -> None:
    arch = ck["architecture"]
    model = make_p0_f8_model(
        20,
        sample_steps=int(arch.get("sample_steps", 10)),
        source_noise_std=float(arch.get("source_noise_std", 0.0)),
        keep_bias=float(arch.get("keep_bias", 2.0)),
    )
    model.load_state_dict(ck["state_dict"], strict=True)


def _training_command(args) -> list[str]:
    train = ROOT / "tools" / "real_motion" / "train_p0_f8_anchor_relative_edit_wm.py"
    cmd = [
        sys.executable,
        str(train),
        "--train-cache",
        args.train_cache,
        "--val-cache",
        args.val_cache,
        "--train-edit-targets",
        args.train_edit_targets,
        "--val-edit-targets",
        args.val_edit_targets,
        "--upstream-ckpt",
        args.upstream_ckpt,
        "--vae-ckpt",
        args.vae_ckpt,
        "--output-dir",
        args.output_dir,
        "--steps",
        str(args.steps),
        "--val-every",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--backbone-lr-scale",
        str(args.backbone_lr_scale),
        "--weight-decay",
        str(args.weight_decay),
        "--min-train-windows",
        str(args.min_train_windows),
        "--sample-steps",
        str(args.sample_steps),
        "--keep-ratio",
        str(args.keep_ratio),
        "--keep-when-no-edit",
        str(args.keep_when_no_edit),
        "--keep-bias",
        str(args.keep_bias),
        "--lovasz-weight",
        str(args.lovasz_weight),
        "--edit-grad-ratio",
        str(args.edit_grad_ratio),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--amp" if args.amp else "--no-amp",
    ]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--train-edit-targets", required=True)
    p.add_argument("--val-edit-targets", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--backbone-lr-scale", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--min-train-windows", type=int, default=4000)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--keep-ratio", type=float, default=1.0)
    p.add_argument("--keep-when-no-edit", type=int, default=64)
    p.add_argument("--keep-bias", type=float, default=2.0)
    p.add_argument("--lovasz-weight", type=float, default=0.5)
    p.add_argument("--edit-grad-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timeout-seconds", type=int, default=3600)
    args = p.parse_args()

    if not 1 <= args.steps <= 50:
        raise ValueError("GPU smoke steps must be in [1,50]")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    out = Path(args.output_dir)
    occupied = [
        out / "latest.pt",
        out / "best.pt",
        out / "last.pt",
        out / "training_report.json",
        out / "smoke_report.json",
    ]
    if any(path.exists() for path in occupied):
        raise RuntimeError(
            "smoke output directory already contains run artifacts; use a new directory"
        )

    cmd = _training_command(args)
    print("smoke_training_command", json.dumps(cmd))
    try:
        proc = subprocess.run(cmd, cwd=ROOT, timeout=args.timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"P0-F8 v2 GPU smoke exceeded hard timeout {args.timeout_seconds}s"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"P0-F8 v2 trainer exited with code {proc.returncode}")

    latest_path = out / "latest.pt"
    if not latest_path.is_file():
        raise RuntimeError("P0-F8 v2 smoke did not produce latest.pt")
    ck = torch.load(latest_path, map_location="cpu", weights_only=False)
    report = validate_smoke_checkpoint(ck, expected_steps=args.steps)
    _strict_reload_model(ck)
    report["checkpoint_reload"] = "strict_model_load_passed"
    report["checkpoint"] = str(latest_path.resolve())
    report["training_report"] = str((out / "training_report.json").resolve())
    report_path = out / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("P0-F8_V2_GPU_SMOKE_PASS", json.dumps(report))


if __name__ == "__main__":
    main()
