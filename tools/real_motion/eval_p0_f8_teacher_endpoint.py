#!/usr/bin/env python3
"""Evaluate the non-causal P0-F8 teacher-repair-endpoint ceiling."""
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

from real_motion.edit_repair import (
    DYNAMIC_IDS,
    apply_anchor_relative_actions,
    horizon_from_flat_indices,
    new_effective_action_stats,
    report_effective_action_stats,
    update_effective_action_stats,
)
from real_motion.msp import latent_support_to_bev
from real_motion.msp_wm_cache import (
    MSP_WM_CACHE_VERSION_V3,
    MSPWorldModelCacheDataset,
)
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.repair_target import apply_dynamic_repair
from tools.real_motion import eval_p0_f8_anchor_relative_edit_wm as common
from tools.real_motion.train_p0_f8_teacher_endpoint import (
    ENDPOINT_SOURCE,
    TEACHER_PROTOCOL,
    TeacherEndpointEditModel,
)

EVAL_PROTOCOL = "p0_f8_teacher_repair_endpoint_eval_v1"
FREE = 17


def decision_gate(
    report: dict,
    *,
    min_delta_overall: float,
    min_delta_moving: float,
    min_delta_moving_1s: float,
) -> dict:
    """Apply the predeclared point-estimate gate without hiding raw metrics."""
    delta_overall = float(report["delta_Overall_vs_strong_anchor"])
    delta_moving = float(report["delta_Moving_vs_strong_anchor"])
    teacher_h1 = float(
        report["teacher_endpoint_ceiling"]["moving"]["per_horizon"][1.0]["mIoU"]
    )
    anchor_h1 = float(
        report["strong_w2det_anchor"]["moving"]["per_horizon"][1.0]["mIoU"]
    )
    delta_h1 = teacher_h1 - anchor_h1
    checks = {
        "delta_Overall": {
            "value": delta_overall,
            "minimum": float(min_delta_overall),
            "pass": bool(delta_overall >= min_delta_overall),
        },
        "delta_Moving": {
            "value": delta_moving,
            "minimum": float(min_delta_moving),
            "pass": bool(delta_moving >= min_delta_moving),
        },
        "delta_Moving_1s": {
            "value": delta_h1,
            "minimum": float(min_delta_moving_1s),
            "pass": bool(delta_h1 >= min_delta_moving_1s),
        },
    }
    passed = all(item["pass"] for item in checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "interpretation": (
            "PASS: representation/action path has useful headroom; run causal P0-F8 v2."
            if passed
            else "FAIL: do not launch the full causal run before fixing the representation/action path."
        ),
        "statistical_note": (
            "Point-estimate gate only; paired scene bootstrap requires per-scene outputs."
        ),
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--action-chunk", type=int, default=65536)
    p.add_argument("--min-delta-overall", type=float, default=0.0)
    p.add_argument("--min-delta-moving", type=float, default=3.0)
    p.add_argument("--min-delta-moving-1s", type=float, default=-0.5)
    p.add_argument("--fail-on-gate", action="store_true")
    args = p.parse_args()
    if args.action_chunk <= 0:
        raise ValueError("action-chunk must be positive")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ds = MSPWorldModelCacheDataset(args.cache)
    if ds.version != MSP_WM_CACHE_VERSION_V3:
        raise RuntimeError("teacher evaluator requires a P0-F5/v3 cache")
    if int(ds.metadata.get("topk", -1)) != 2:
        raise RuntimeError("teacher evaluator is frozen to Top-2")
    if ds.metadata.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("cache does not use Strong W2Det")
    if ds.metadata.get("repair_endpoint_contract") != common.REPAIR_CONTRACT:
        raise RuntimeError("cache repair endpoint contract mismatch")
    if not bool(ds.metadata.get("include_eval_payload", False)):
        raise RuntimeError("teacher evaluation requires validation eval payload")

    vae_sha = file_sha256(args.vae_ckpt)
    expected_vae = ds.metadata.get("vae_checkpoint_sha256")
    if expected_vae and vae_sha != expected_vae:
        raise RuntimeError("VAE checkpoint differs from validation cache")
    ck = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("architecture") or {}
    if arch.get("protocol") != TEACHER_PROTOCOL:
        raise RuntimeError("checkpoint is not the P0-F8 teacher endpoint ceiling")
    if arch.get("endpoint_source") != ENDPOINT_SOURCE:
        raise RuntimeError("teacher checkpoint endpoint source mismatch")
    if arch.get("causal_deployment_eligible") is not False:
        raise RuntimeError("teacher checkpoint must be explicitly non-causal")
    if ck.get("vae_checkpoint_sha256") != vae_sha:
        raise RuntimeError("teacher checkpoint was trained with a different VAE")

    model = TeacherEndpointEditModel(
        keep_bias=float(arch.get("keep_bias", 2.0))
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    vae_model, _ = load_official_vae(UP, args.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)
    use_amp = bool(args.amp and device.type == "cuda")

    anchor_state = common._new_metrics()
    teacher_state = common._new_metrics()
    oracle_state = common._new_metrics()
    action_stats = new_effective_action_stats()
    write_ratios = []
    valid_windows = []

    for i in range(len(ds)):
        sample = ds[i]
        missing = [key for key in common.EVAL_KEYS if key not in sample]
        if missing:
            raise RuntimeError(f"{sample['sample_id']}: eval payload missing {missing}")
        endpoint = sample["repair_target_latent"].unsqueeze(0).to(device).float()
        if tuple(endpoint.shape[1:]) != (6, 16, 50, 50):
            raise RuntimeError(f"{sample['sample_id']}: invalid teacher endpoint shape")

        write_lat = sample["msp_write_support_latent"].bool()
        write_bev = latent_support_to_bev(
            write_lat, (200, 200)
        ).cpu().numpy().astype(bool)
        support_idx_np = common._support_flat_indices(write_bev, depth=16)
        support_idx = torch.from_numpy(support_idx_np).to(
            device=device, dtype=torch.long
        )
        anchor_future = sample["eval_strong_anchor_occ"].cpu().numpy()
        anchor_slots = common._anchor_slots_at_indices(
            anchor_future, support_idx_np
        ).to(device)
        horizons = horizon_from_flat_indices(support_idx).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            sparse_semantic = vae.decode_logits_at_flat_indices(
                endpoint, [support_idx]
            )[0]
            action_parts = []
            for start in range(0, int(support_idx.numel()), int(args.action_chunk)):
                end = min(start + int(args.action_chunk), int(support_idx.numel()))
                action_parts.append(
                    model.edit_head(
                        sparse_semantic[start:end],
                        anchor_slots[start:end],
                        horizons[start:end],
                    ).argmax(dim=-1)
                )
        actions = (
            torch.cat(action_parts, dim=0)
            if action_parts
            else torch.empty(0, device=device, dtype=torch.long)
        )
        actions_np = actions.cpu().numpy().astype(np.int64, copy=False)
        final_all = apply_anchor_relative_actions(
            anchor_future, support_idx_np, actions_np, free_label=FREE
        )
        repair_target = sample["eval_repair_target_occ"].cpu().numpy()
        update_effective_action_stats(
            action_stats,
            anchor_occ=anchor_future,
            final_occ=final_all,
            repair_target_occ=repair_target,
            flat_indices=support_idx_np,
            actions=actions_np,
        )
        write_ratios.append(float(write_lat.float().mean().item()))
        valid_windows.append(int(sample["window_valid"].sum().item()))

        gt_future = sample["eval_future_gt_occ"].cpu().numpy()
        moving_support = sample["eval_gt_moving_support"].cpu().numpy().astype(bool)
        for horizon, frame_index in common.REPORT.items():
            gt = gt_future[frame_index]
            anchor = anchor_future[frame_index]
            common._update(
                anchor_state, horizon, anchor, gt, moving_support[frame_index]
            )
            common._update(
                teacher_state, horizon, final_all[frame_index], gt,
                moving_support[frame_index]
            )
            oracle = apply_dynamic_repair(
                anchor,
                gt,
                write_bev[frame_index],
                dynamic_class_ids=DYNAMIC_IDS,
                free_label=FREE,
            )
            if not np.array_equal(oracle, repair_target[frame_index]):
                raise RuntimeError(
                    f"{sample['sample_id']} horizon={horizon}: "
                    "cached same-support oracle mismatch"
                )
            common._update(
                oracle_state, horizon, oracle, gt, moving_support[frame_index]
            )
        if i % 8 == 0:
            print(
                "eval_teacher", i, sample["sample_id"],
                "support_voxels", int(actions_np.size)
            )

    anchor_report = common._report(anchor_state)
    teacher_report = common._report(teacher_state)
    oracle_report = common._report(oracle_state)
    anchor_moving = float(anchor_report["moving"]["mIoU"])
    teacher_moving = float(teacher_report["moving"]["mIoU"])
    oracle_moving = float(oracle_report["moving"]["mIoU"])
    anchor_overall = float(anchor_report["overall"]["mIoU"])
    teacher_overall = float(teacher_report["overall"]["mIoU"])
    action_report = report_effective_action_stats(action_stats)

    report = {
        "protocol": {
            "name": EVAL_PROTOCOL,
            "diagnostic_only": True,
            "causal_deployment_eligible": False,
            "uses_future_gt": True,
            "endpoint_source": ENDPOINT_SOURCE,
            "num_windows": len(ds),
            "topk": 2,
            "mean_valid_top2_windows": float(np.mean(valid_windows)),
            "mean_write_latent_ratio": float(np.mean(write_ratios)),
            "write_budget_ratio": float(
                ds.metadata.get("write_budget_ratio", float("nan"))
            ),
            "fusion": (
                "exact Strong W2Det default; KEEP/CLEAR/WRITE only inside "
                "the identical causal MSP support"
            ),
            **action_report,
            "endpoint_oracle_consistency": "bit-exact checked on reported horizons",
        },
        "strong_w2det_anchor": anchor_report,
        "teacher_endpoint_ceiling": teacher_report,
        "same_support_gt_repair_oracle": oracle_report,
        "delta_Overall_vs_strong_anchor": teacher_overall - anchor_overall,
        "delta_Moving_vs_strong_anchor": teacher_moving - anchor_moving,
        "oracle_delta_Moving_vs_strong_anchor": oracle_moving - anchor_moving,
        "remaining_Moving_headroom_to_oracle": oracle_moving - teacher_moving,
        "checkpoint": str(Path(args.teacher_ckpt).resolve()),
        "best_val_objective": ck.get("best_val_objective"),
    }
    report["decision_gate"] = decision_gate(
        report,
        min_delta_overall=float(args.min_delta_overall),
        min_delta_moving=float(args.min_delta_moving),
        min_delta_moving_1s=float(args.min_delta_moving_1s),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.fail_on_gate and report["decision_gate"]["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
