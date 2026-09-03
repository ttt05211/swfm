#!/usr/bin/env python3
"""Evaluate P0-F8 KEEP/CLEAR/WRITE sparse edit deployment."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.context import crop_prediction_and_context
from real_motion.edit_repair import (
    DYNAMIC_IDS,
    DYNAMIC_TO_SLOT,
    KEEP,
    apply_anchor_relative_actions,
    horizon_from_flat_indices,
    new_effective_action_stats,
    report_effective_action_stats,
    update_effective_action_stats,
)
from real_motion.metrics.moving_miou_v2 import (
    MovingMIoUV2MultiHorizon,
    REPORT_HORIZONS_S,
    SemanticIoUAccumulator,
)
from real_motion.models.p0_f8 import make_p0_f8_model
from real_motion.msp import latent_support_to_bev
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V3, MSPWorldModelCacheDataset
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.repair_target import apply_dynamic_repair
from real_motion.windows import WindowPlan, crop_windows, scatter_windows
from tools.real_motion.train_p0_f8_anchor_relative_edit_wm import F8_PROTOCOL
from tools.real_motion.train_p0_f8_frozen_causal_endpoint_probe import (
    PROBE_PROTOCOL,
    load_probe_head_into_model,
)

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
PRED_HW = (20, 20)
CONTEXT_HW = (40, 40)
FULL_HW = (50, 50)
FREE = 17
REPAIR_CONTRACT = "strong_anchor_outside_support_gt_dynamic_inside_support_v1"
EVAL_KEYS = (
    "eval_future_gt_occ",
    "eval_strong_anchor_occ",
    "eval_repair_target_occ",
    "eval_gt_moving_support",
)
SWEEP_PROTOCOL = "p0_f8_keep_logit_margin_sweep_v1"


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


def _anchor_slots_at_indices(anchor: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    values = anchor.reshape(-1)[idx]
    slots = np.zeros(values.shape, dtype=np.int64)
    for cid, slot in DYNAMIC_TO_SLOT.items():
        slots[values == int(cid)] = int(slot)
    return torch.from_numpy(slots)


def _support_flat_indices(write_bev: np.ndarray, depth: int = 16) -> np.ndarray:
    support = np.broadcast_to(np.asarray(write_bev, dtype=bool)[..., None], (*write_bev.shape, depth))
    return np.flatnonzero(support.reshape(-1)).astype(np.int64, copy=False)


def normalize_keep_logit_margins(values) -> list[float]:
    margins = [float(x) for x in values]
    if not margins:
        raise ValueError("at least one keep-logit margin is required")
    if any(not math.isfinite(x) or x < 0.0 for x in margins):
        raise ValueError("keep-logit margins must be finite and non-negative")
    if len(set(margins)) != len(margins):
        raise ValueError("keep-logit margins must be unique")
    return margins


def actions_with_keep_margin(
    action_logits: torch.Tensor,
    keep_logit_margin: float,
) -> torch.Tensor:
    """Apply a conservative inference-only bias to KEEP before argmax."""
    margin = float(keep_logit_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("keep-logit margin must be finite and non-negative")
    adjusted = action_logits.clone()
    adjusted[:, KEEP] = adjusted[:, KEEP] + margin
    return adjusted.argmax(dim=-1)


def margin_decision_gate(
    *,
    anchor_report: dict,
    trained_report: dict,
    min_delta_overall: float,
    min_delta_moving: float,
    min_delta_moving_1s: float,
) -> dict:
    delta_overall = float(
        trained_report["overall"]["mIoU"] - anchor_report["overall"]["mIoU"]
    )
    delta_moving = float(
        trained_report["moving"]["mIoU"] - anchor_report["moving"]["mIoU"]
    )
    delta_moving_1s = float(
        trained_report["moving"]["per_horizon"][1.0]["mIoU"]
        - anchor_report["moving"]["per_horizon"][1.0]["mIoU"]
    )
    checks = {
        "delta_Overall": {
            "value": delta_overall,
            "minimum": float(min_delta_overall),
            "comparison": ">=",
            "pass": bool(delta_overall >= float(min_delta_overall)),
        },
        "delta_Moving": {
            "value": delta_moving,
            "minimum_exclusive": float(min_delta_moving),
            "comparison": ">",
            "pass": bool(delta_moving > float(min_delta_moving)),
        },
        "delta_Moving_1s": {
            "value": delta_moving_1s,
            "minimum": float(min_delta_moving_1s),
            "comparison": ">=",
            "pass": bool(delta_moving_1s >= float(min_delta_moving_1s)),
        },
    }
    return {
        "status": "PASS" if all(x["pass"] for x in checks.values()) else "FAIL",
        "checks": checks,
    }


def select_passing_margin(results: list[dict]) -> dict:
    """Select only among predeclared gate passes, with deterministic ranking."""
    passing = [item for item in results if item["decision_gate"]["status"] == "PASS"]
    best_moving = max(results, key=lambda x: float(x["delta_Moving_vs_strong_anchor"]))
    best_overall = max(results, key=lambda x: float(x["delta_Overall_vs_strong_anchor"]))
    if not passing:
        return {
            "status": "FAIL",
            "selected_margin": None,
            "passing_margins": [],
            "best_delta_moving_margin": float(best_moving["keep_logit_margin"]),
            "best_delta_overall_margin": float(best_overall["keep_logit_margin"]),
            "ranking": "no selection outside the predeclared gate",
        }
    selected = max(
        passing,
        key=lambda x: (
            float(x["delta_Moving_vs_strong_anchor"]),
            float(x["delta_Overall_vs_strong_anchor"]),
            -float(x["action_statistics"]["effective_false_edit_rate"]),
            -float(x["keep_logit_margin"]),
        ),
    )
    return {
        "status": "PASS",
        "selected_margin": float(selected["keep_logit_margin"]),
        "passing_margins": [float(x["keep_logit_margin"]) for x in passing],
        "best_delta_moving_margin": float(best_moving["keep_logit_margin"]),
        "best_delta_overall_margin": float(best_overall["keep_logit_margin"]),
        "ranking": (
            "among gate passes: delta Moving desc, delta Overall desc, "
            "effective false-edit rate asc, margin asc"
        ),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--sparse-ckpt", required=True)
    p.add_argument(
        "--edit-head-probe-ckpt",
        default=None,
        help=(
            "Optional head-only checkpoint from the frozen causal-endpoint probe. "
            "The causal transition still comes from --sparse-ckpt."
        ),
    )
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--action-chunk", type=int, default=65536)
    p.add_argument(
        "--keep-logit-margins",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "Inference-only margins added to the KEEP logit. Multiple values are "
            "evaluated in one causal-WM/VAE forward pass."
        ),
    )
    p.add_argument("--min-delta-overall", type=float, default=0.0)
    p.add_argument("--min-delta-moving", type=float, default=0.0)
    p.add_argument("--min-delta-moving-1s", type=float, default=-0.5)
    p.add_argument("--fail-on-no-passing-margin", action="store_true")
    a = p.parse_args()
    if a.action_chunk <= 0:
        raise ValueError("action-chunk must be positive")
    margins = normalize_keep_logit_margins(a.keep_logit_margins)
    sweep_mode = len(margins) > 1

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = MSPWorldModelCacheDataset(a.cache)
    if ds.version != MSP_WM_CACHE_VERSION_V3:
        raise RuntimeError("P0-F8 evaluator requires a P0-F5/v3 cache")
    if int(ds.metadata.get("topk", -1)) != 2:
        raise RuntimeError("P0-F8 evaluator is frozen to Top-2")
    if ds.metadata.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("cache does not use Strong W2Det")
    if ds.metadata.get("repair_endpoint_contract") != REPAIR_CONTRACT:
        raise RuntimeError("cache repair endpoint contract mismatch")
    if not bool(ds.metadata.get("include_eval_payload", False)):
        raise RuntimeError("P0-F8 final evaluation requires validation eval payload")
    vae_sha256 = file_sha256(a.vae_ckpt)
    expected_vae = ds.metadata.get("vae_checkpoint_sha256")
    if expected_vae and vae_sha256 != expected_vae:
        raise RuntimeError("VAE checkpoint differs from routed cache")

    ck = torch.load(a.sparse_ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("architecture", {})
    if arch.get("protocol") != F8_PROTOCOL:
        raise RuntimeError("Sparse-WM checkpoint is not P0-F8")
    if arch.get("repair_endpoint_contract") != REPAIR_CONTRACT:
        raise RuntimeError("checkpoint repair endpoint contract mismatch")
    edit_lambda = ck.get("edit_lambda", arch.get("edit_lambda"))
    if edit_lambda is None or float(edit_lambda) <= 0:
        raise RuntimeError("P0-F8 checkpoint lacks a positive edit lambda")

    model = make_p0_f8_model(
        20,
        sample_steps=int(arch.get("sample_steps", 10)),
        source_noise_std=float(arch.get("source_noise_std", 0.0)),
        keep_bias=float(arch.get("keep_bias", 2.0)),
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    edit_head_overlay = None
    if a.edit_head_probe_ckpt:
        causal_sha256 = file_sha256(a.sparse_ckpt)
        probe_ck = torch.load(
            a.edit_head_probe_ckpt, map_location="cpu", weights_only=False
        )
        probe_arch = load_probe_head_into_model(
            model,
            probe_ck,
            causal_sha256=causal_sha256,
            vae_sha256=vae_sha256,
        )
        edit_head_overlay = {
            "protocol": PROBE_PROTOCOL,
            "checkpoint": str(Path(a.edit_head_probe_ckpt).resolve()),
            "checkpoint_sha256": file_sha256(a.edit_head_probe_ckpt),
            "step": int(probe_ck.get("step", 0)),
            "best_val_objective": probe_ck.get("best_val_objective"),
            "endpoint_source": probe_arch.get("endpoint_source"),
            "source_causal_checkpoint_sha256": causal_sha256,
            "source_causal_checkpoint_step": probe_arch.get(
                "source_causal_checkpoint_step"
            ),
            "lovasz_weight": probe_arch.get("lovasz_weight"),
            "edit_sampling": probe_arch.get("edit_sampling"),
            "keep_bias": probe_arch.get("keep_bias"),
        }
    model.eval()

    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    va = OccFMVAEAdapter(vae_model)
    anchor_state = _new_metrics()
    oracle_state = _new_metrics()
    margin_states = [_new_metrics() for _ in margins]
    valid_windows = []
    write_ratios = []
    margin_action_stats = [new_effective_action_stats() for _ in margins]
    use_amp = bool(a.amp and device.type == "cuda")

    for i in range(len(ds)):
        s = ds[i]
        missing = [k for k in EVAL_KEYS if k not in s]
        if missing:
            raise RuntimeError(f"{s['sample_id']}: eval payload missing {missing}")

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
            fused = scatter_windows(pad.reshape(B, K, *pred.shape[1:]), plan, base=anchor_full)

        write_lat = s["msp_write_support_latent"].bool()
        write_bev = latent_support_to_bev(write_lat, (200, 200)).cpu().numpy().astype(bool)
        support_idx_np = _support_flat_indices(write_bev, depth=16)
        support_idx = torch.from_numpy(support_idx_np).to(device=device, dtype=torch.long)
        anchor_future = s["eval_strong_anchor_occ"].cpu().numpy()
        anchor_slots = _anchor_slots_at_indices(anchor_future, support_idx_np).to(device)
        horizons = horizon_from_flat_indices(support_idx).to(device)

        sparse_semantic = va.decode_logits_at_flat_indices(fused.float(), [support_idx])[0]
        action_parts = [[] for _ in margins]
        for start in range(0, int(support_idx.numel()), int(a.action_chunk)):
            end = min(start + int(a.action_chunk), int(support_idx.numel()))
            action_logits = model.edit_head(
                sparse_semantic[start:end],
                anchor_slots[start:end],
                horizons[start:end],
            )
            for margin_idx, margin in enumerate(margins):
                action_parts[margin_idx].append(
                    actions_with_keep_margin(action_logits, margin).cpu()
                )
        actions_by_margin = [
            (
                torch.cat(parts, dim=0).numpy().astype(np.int64, copy=False)
                if parts else np.empty(0, dtype=np.int64)
            )
            for parts in action_parts
        ]
        repair_target_future = s["eval_repair_target_occ"].cpu().numpy()
        valid_windows.append(int(valid.sum().item()))
        write_ratios.append(float(write_lat.float().mean().item()))
        gt_future = s["eval_future_gt_occ"].cpu().numpy()
        moving_support = s["eval_gt_moving_support"].cpu().numpy().astype(bool)

        for h, fi in REPORT.items():
            gt = gt_future[fi]
            anchor = anchor_future[fi]
            _update(anchor_state, h, anchor, gt, moving_support[fi])
            oracle = apply_dynamic_repair(
                anchor,
                gt,
                write_bev[fi],
                dynamic_class_ids=DYNAMIC_IDS,
                free_label=FREE,
            )
            if not np.array_equal(oracle, repair_target_future[fi]):
                raise RuntimeError(
                    f"{s['sample_id']} horizon={h}: cached same-support oracle mismatch"
                )
            _update(oracle_state, h, oracle, gt, moving_support[fi])

        for margin_idx, actions_np in enumerate(actions_by_margin):
            final_all = apply_anchor_relative_actions(
                anchor_future,
                support_idx_np,
                actions_np,
                free_label=FREE,
            )
            update_effective_action_stats(
                margin_action_stats[margin_idx],
                anchor_occ=anchor_future,
                final_occ=final_all,
                repair_target_occ=repair_target_future,
                flat_indices=support_idx_np,
                actions=actions_np,
            )
            for h, fi in REPORT.items():
                _update(
                    margin_states[margin_idx],
                    h,
                    final_all[fi],
                    gt_future[fi],
                    moving_support[fi],
                )

        if i % 8 == 0:
            print(
                "eval",
                i,
                s["sample_id"],
                "support_voxels",
                int(support_idx_np.size),
                "keep_margins",
                margins,
            )

    anchor_report = _report(anchor_state)
    oracle_report = _report(oracle_state)
    anchor_overall = float(anchor_report["overall"]["mIoU"])
    anchor_moving = float(anchor_report["moving"]["mIoU"])
    oracle_moving = float(oracle_report["moving"]["mIoU"])
    margin_results = []
    for margin, state, stats in zip(margins, margin_states, margin_action_stats):
        trained_report = _report(state)
        action_report = report_effective_action_stats(stats)
        delta_overall = float(trained_report["overall"]["mIoU"] - anchor_overall)
        delta_moving = float(trained_report["moving"]["mIoU"] - anchor_moving)
        gate = margin_decision_gate(
            anchor_report=anchor_report,
            trained_report=trained_report,
            min_delta_overall=float(a.min_delta_overall),
            min_delta_moving=float(a.min_delta_moving),
            min_delta_moving_1s=float(a.min_delta_moving_1s),
        )
        margin_results.append({
            "keep_logit_margin": float(margin),
            "trained_sparse_wm": trained_report,
            "delta_Overall_vs_strong_anchor": delta_overall,
            "delta_Moving_vs_strong_anchor": delta_moving,
            "remaining_Moving_headroom_to_oracle": float(
                oracle_moving - trained_report["moving"]["mIoU"]
            ),
            "action_statistics": action_report,
            "decision_gate": gate,
        })

    common_protocol = {
        "num_windows": len(ds),
        "topk": 2,
        "prediction_hw": list(PRED_HW),
        "history_context_hw": list(CONTEXT_HW),
        "slot_compute_ratio": float(np.mean(valid_windows) * 400.0 / 2500.0),
        "mean_write_latent_ratio": float(np.mean(write_ratios)),
        "write_budget_ratio": float(ds.metadata.get("write_budget_ratio", float("nan"))),
        "anchor": "strong occupancy-only W2Det",
        "history": "full occupancy history latent",
        "training_objective": (
            "fresh edit head on frozen deterministic causal deployment endpoints; "
            "balanced action CE + result-semantic Lovasz"
            if edit_head_overlay is not None
            else "uniform FM MSE + calibrated balanced action CE + result-semantic Lovasz"
        ),
        "edit_head_source": (
            PROBE_PROTOCOL if edit_head_overlay is not None else F8_PROTOCOL
        ),
        "edit_head_overlay": edit_head_overlay,
        "edit_lambda": (
            float(edit_lambda) if edit_head_overlay is None else None
        ),
        "edit_lambda_calibration": (
            ck.get("edit_lambda_calibration")
            if edit_head_overlay is None else None
        ),
        "source_causal_edit_lambda": float(edit_lambda),
        "lovasz_weight": (
            arch.get("lovasz_weight")
            if edit_head_overlay is None
            else edit_head_overlay["lovasz_weight"]
        ),
        "edit_sampling": (
            arch.get("edit_sampling")
            if edit_head_overlay is None
            else edit_head_overlay["edit_sampling"]
        ),
        "fusion": (
            "exact Strong W2Det default; KEEP/CLEAR/WRITE only inside causal MSP support"
        ),
        "endpoint_oracle_consistency": "bit-exact checked on reported horizons",
    }
    if sweep_mode:
        selection = select_passing_margin(margin_results)
        selection["statistical_note"] = (
            "Diagnostic point-estimate selection on this validation cache; lock the margin "
            "before evaluation on an independent split or paired scene bootstrap."
        )
        report = {
            "protocol": {
                "name": SWEEP_PROTOCOL,
                **common_protocol,
                "keep_logit_margins": margins,
                "margin_application": "additive_to_KEEP_logit_before_argmax",
                "shared_forward": (
                    "one causal-WM sample and one sparse VAE decode per window; "
                    "all margins reuse the same logits"
                ),
                "decision_gate": {
                    "delta_Overall_minimum": float(a.min_delta_overall),
                    "delta_Moving_minimum_exclusive": float(a.min_delta_moving),
                    "delta_Moving_1s_minimum": float(a.min_delta_moving_1s),
                },
            },
            "strong_w2det_anchor": anchor_report,
            "same_support_gt_repair_oracle": oracle_report,
            "oracle_delta_Moving_vs_strong_anchor": oracle_moving - anchor_moving,
            "margin_results": margin_results,
            "selection": selection,
            "checkpoint": str(Path(a.sparse_ckpt).resolve()),
            "edit_head_probe_checkpoint": (
                None if edit_head_overlay is None else edit_head_overlay["checkpoint"]
            ),
            "best_val_objective": (
                ck.get("best_val_objective")
                if edit_head_overlay is None
                else edit_head_overlay["best_val_objective"]
            ),
        }
    else:
        result = margin_results[0]
        report = {
            "protocol": {
                "name": "p0_f8_anchor_relative_edit_eval_v1",
                **common_protocol,
                "keep_logit_margin": float(margins[0]),
                **result["action_statistics"],
            },
            "strong_w2det_anchor": anchor_report,
            "trained_sparse_wm": result["trained_sparse_wm"],
            "same_support_gt_repair_oracle": oracle_report,
            "delta_Overall_vs_strong_anchor": result[
                "delta_Overall_vs_strong_anchor"
            ],
            "delta_Moving_vs_strong_anchor": result["delta_Moving_vs_strong_anchor"],
            "oracle_delta_Moving_vs_strong_anchor": oracle_moving - anchor_moving,
            "remaining_Moving_headroom_to_oracle": result[
                "remaining_Moving_headroom_to_oracle"
            ],
            "decision_gate": result["decision_gate"],
            "checkpoint": str(Path(a.sparse_ckpt).resolve()),
            "edit_head_probe_checkpoint": (
                None if edit_head_overlay is None else edit_head_overlay["checkpoint"]
            ),
            "best_val_objective": (
                ck.get("best_val_objective")
                if edit_head_overlay is None
                else edit_head_overlay["best_val_objective"]
            ),
        }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if a.fail_on_no_passing_margin:
        gate_status = (
            report["selection"]["status"]
            if sweep_mode else report["decision_gate"]["status"]
        )
        if gate_status != "PASS":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
