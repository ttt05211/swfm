#!/usr/bin/env python3
"""H0: no-training horizon supervision/gradient attribution for V17-RL.

This diagnostic answers one narrow question before changing the loss:

    Is the frozen V17-RL training objective systematically under-driving later
    future horizons, and if so is that due to label counts, SmoothL1, the
    transport-overlap term, or their interaction at the residual head?

It uses only the frozen V17 validation cache and one V17-RL checkpoint.  It does
not read future occupancy, rerun Strong/KTA, alter the model, or train anything.

The frozen RL motion objective is

    L_motion = L_position + lambda_overlap * L_overlap

where L_position is the historical micro-averaged SmoothL1 over every valid
source/horizon/XY coordinate and L_overlap is the historical micro-averaged
transport Soft-IoU loss over every eligible source/horizon pair.  H0 decomposes
*those exact global reductions* into six additive horizon contributions.  This
is important: a per-horizon mean by itself would erase the label-count imbalance
that we are trying to diagnose.

For every horizon H0 reports both:

1. direct residual-output gradients (what supervision reaches the six predicted
   displacement slots before the shared head/backbone), and
2. residual-head parameter gradients (which additionally include the learned
   future-query representation for that horizon).

Existence BCE is reported only by label count and is intentionally excluded from
``L_motion`` attribution: it has no computational path to ``residual_xy_m`` or
``residual_head`` parameters.  It can affect shared decoder/backbone parameters,
but that is a different question from the proposed horizon-normalized residual
loss.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from real_motion.local_st_world_model_v17 import (
    MODEL_PROTOCOL_V17,
    LocalSpatialTemporalWorldModelV17,
    config_from_mapping_v17,
    soft_transport_overlap_loss,
)
from real_motion.motion_transport import FUTURE_FRAMES
from tools.real_motion.train_p0_f9_v17_local_stwm import (
    flatten_supervised,
    forward_model,
    load_cache,
    make_dataset,
    true_moving_mask,
    unpack,
)

PROTOCOL = "p0_f9_v17_h0_horizon_gradient_attribution_v1"
EXPECTED_CHECKPOINT_EPOCH = 5
EXPECTED_VARIANT = "RL"
EXPECTED_OVERLAP_WEIGHT = 0.25
HORIZONS_S = tuple(0.5 * (i + 1) for i in range(FUTURE_FRAMES))


def position_horizon_term(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    horizon_index: int,
    *,
    global_valid_labels: int,
) -> dict:
    """Return one horizon's exact additive contribution to frozen SmoothL1.

    ``motion_transport_loss`` applies ``reduction='mean'`` to ``pred[valid]`` of
    shape [K,2], so the global denominator is exactly ``2 * K_global``.  The
    returned ``objective_contribution`` uses that denominator, while
    ``horizon_mean`` is only a descriptive within-horizon mean.
    """
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError("pred/target must be matching [B,6,2] tensors")
    if valid.shape != pred.shape[:2]:
        raise ValueError("valid shape mismatch")
    h = int(horizon_index)
    if h < 0 or h >= pred.shape[1]:
        raise ValueError("horizon_index out of range")
    if int(global_valid_labels) <= 0:
        raise ValueError("global_valid_labels must be positive")
    m = valid[:, h].bool()
    n = int(m.sum().item())
    if n:
        element_sum = F.smooth_l1_loss(
            pred[:, h][m], target[:, h][m].to(pred.dtype), reduction="sum", beta=1.0
        )
        horizon_mean = element_sum / float(2 * n)
    else:
        element_sum = pred[:, h].sum() * 0.0
        horizon_mean = element_sum
    contribution = element_sum / float(2 * int(global_valid_labels))
    return {
        "labels": n,
        "element_sum": element_sum,
        "horizon_mean": horizon_mean,
        "objective_contribution": contribution,
    }


def overlap_horizon_term(
    pred: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    valid: torch.Tensor,
    horizon_index: int,
    *,
    global_eligible_labels: int,
    patch_resolution_m: float,
) -> dict:
    """Return one horizon's exact additive contribution to frozen overlap loss."""
    if int(global_eligible_labels) <= 0:
        raise ValueError("global_eligible_labels must be positive")
    h = int(horizon_index)
    if h < 0 or h >= FUTURE_FRAMES:
        raise ValueError("horizon_index out of range")
    hvalid = torch.zeros_like(valid, dtype=torch.bool)
    hvalid[:, h] = valid[:, h].bool()
    mean_loss, stats = soft_transport_overlap_loss(
        pred.float(),
        target.float(),
        footprint.float(),
        hvalid,
        patch_resolution_m=float(patch_resolution_m),
    )
    n = int(stats["transport_overlap_labels"])
    contribution = mean_loss * (float(n) / float(global_eligible_labels))
    return {
        "labels": n,
        "horizon_mean": mean_loss,
        "soft_iou": float(stats["transport_soft_iou"]),
        "objective_contribution": contribution,
    }


def _flat_grad(grads: Iterable[torch.Tensor | None], params: Iterable[torch.Tensor]) -> torch.Tensor:
    chunks = []
    for grad, param in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
        else:
            chunks.append(grad.detach().float().reshape(-1))
    if not chunks:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(chunks)


def _loss_gradients(
    loss: torch.Tensor,
    pred: torch.Tensor,
    head_params: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    grads = torch.autograd.grad(
        loss,
        [pred, *head_params],
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    pred_grad = grads[0]
    if pred_grad is None:
        pred_grad = torch.zeros_like(pred)
    return pred_grad.detach().float(), _flat_grad(grads[1:], head_params)


def head_gradient_attribution(horizon_vectors: list[torch.Tensor]) -> dict:
    """Summarize shared-head gradient magnitude, alignment, and signed share.

    ``projection_share`` is dot(g_h, g_total) / ||g_total||^2.  Its six values
    sum to one (up to floating-point error) and can be negative when a horizon
    directly opposes the final shared-head update.
    """
    if not horizon_vectors:
        raise ValueError("horizon_vectors cannot be empty")
    vecs = [v.detach().double().reshape(-1).cpu() for v in horizon_vectors]
    width = int(vecs[0].numel())
    if any(int(v.numel()) != width for v in vecs):
        raise ValueError("head gradient vectors have inconsistent sizes")
    total = torch.stack(vecs, dim=0).sum(dim=0)
    total_norm = float(torch.linalg.vector_norm(total))
    norms = [float(torch.linalg.vector_norm(v)) for v in vecs]
    norm_sum = float(sum(norms))
    total_sq = float(torch.dot(total, total))
    rows = []
    for v, norm in zip(vecs, norms):
        if norm > 0.0 and total_norm > 0.0:
            cosine = float(torch.dot(v, total) / (norm * total_norm))
        else:
            cosine = float("nan")
        projection = float(torch.dot(v, total) / total_sq) if total_sq > 0.0 else float("nan")
        rows.append({
            "l2": norm,
            "norm_share": norm / norm_sum if norm_sum > 0.0 else float("nan"),
            "cosine_to_total": cosine,
            "projection_share": projection,
        })
    return {
        "total_l2": total_norm,
        "sum_horizon_l2": norm_sum,
        "rows": rows,
        "projection_share_sum": float(sum(r["projection_share"] for r in rows))
        if total_sq > 0.0 else float("nan"),
    }


def squared_energy_shares(squared_norms: Iterable[float]) -> list[float]:
    xs = [float(x) for x in squared_norms]
    if any(x < -1e-15 for x in xs):
        raise ValueError("squared norms must be non-negative")
    total = float(sum(max(0.0, x) for x in xs))
    if total <= 0.0:
        return [float("nan")] * len(xs)
    return [max(0.0, x) / total for x in xs]


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if abs(float(b)) > 1e-30 else float("nan")


def _load_frozen_rl(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != MODEL_PROTOCOL_V17:
        raise RuntimeError(f"checkpoint protocol mismatch: {ck.get('protocol')}")
    if str(ck.get("variant")) != EXPECTED_VARIANT or not bool(ck.get("use_representation", False)):
        raise RuntimeError("H0 requires the frozen V17-RL representation checkpoint")
    if int(ck.get("epoch", -1)) != EXPECTED_CHECKPOINT_EPOCH:
        raise RuntimeError(
            f"H0 requires epoch {EXPECTED_CHECKPOINT_EPOCH}, got {ck.get('epoch')}"
        )
    weight = float(ck.get("overlap_weight", float("nan")))
    if not math.isclose(weight, EXPECTED_OVERLAP_WEIGHT, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"H0 requires frozen RL overlap weight {EXPECTED_OVERLAP_WEIGHT}, got {weight}"
        )
    cfg = config_from_mapping_v17(ck.get("model_config"))
    if not bool(cfg.use_representation):
        raise RuntimeError("checkpoint model_config disables V17 representation")
    model = LocalSpatialTemporalWorldModelV17(cfg).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return ck, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", required=True, help="frozen V17-RL epoch-5 checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()
    if a.batch_size <= 0 or a.num_workers < 0:
        raise ValueError("invalid batch/runtime arguments")

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not bool(a.no_amp)
    meta, records = load_cache(a.val_cache)
    flat = flatten_supervised(records)
    patch_resolution_m = float(meta.get("patch_resolution_m", 0.8))
    valid_all = flat["target_valid"].bool()
    footprint_all = flat["target_source_mask_tube"][:, -1].bool()
    footprint_present = footprint_all.flatten(1).any(dim=1)
    eligible_all = valid_all & footprint_present[:, None]
    moving_all = true_moving_mask(flat["target_displacement_xy_m"], valid_all)
    global_valid_labels = int(valid_all.sum().item())
    global_eligible_labels = int(eligible_all.sum().item())
    if global_valid_labels <= 0 or global_eligible_labels <= 0:
        raise RuntimeError("validation cache has no valid/overlap-eligible labels")

    ck, model = _load_frozen_rl(a.checkpoint, device)
    head_params = [model.residual_head.weight, model.residual_head.bias]
    head_width = sum(int(x.numel()) for x in head_params)

    loader = DataLoader(
        make_dataset(flat),
        batch_size=int(a.batch_size),
        shuffle=False,
        num_workers=int(a.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    valid_counts = [int(valid_all[:, h].sum().item()) for h in range(FUTURE_FRAMES)]
    moving_counts = [int(moving_all[:, h].sum().item()) for h in range(FUTURE_FRAMES)]
    eligible_counts = [int(eligible_all[:, h].sum().item()) for h in range(FUTURE_FRAMES)]

    # Loss accumulators use raw sums/counts for descriptive horizon means and
    # globally normalized contributions for exact objective attribution.
    pos_raw = [0.0] * FUTURE_FRAMES
    tm_pos_raw = [0.0] * FUTURE_FRAMES
    tm_coord_count = [0] * FUTURE_FRAMES
    tm_error_sum = [0.0] * FUTURE_FRAMES
    ov_raw = [0.0] * FUTURE_FRAMES
    pos_contrib = [0.0] * FUTURE_FRAMES
    ov_contrib = [0.0] * FUTURE_FRAMES

    # Squared direct-output gradients.  Horizon slots are disjoint at this
    # tensor, so squared-energy shares have an exact interpretation.
    pos_out_sq = [0.0] * FUTURE_FRAMES
    ov_out_sq = [0.0] * FUTURE_FRAMES
    combined_out_sq = [0.0] * FUTURE_FRAMES

    # Residual head is shared over the six future queries, so vectors must be
    # accumulated over batches before norms/projections are taken.
    head_pos = [torch.zeros(head_width, dtype=torch.float64) for _ in range(FUTURE_FRAMES)]
    head_ov = [torch.zeros(head_width, dtype=torch.float64) for _ in range(FUTURE_FRAMES)]

    full_position_check = 0.0
    full_overlap_check = 0.0
    seen = 0

    for bi, raw in enumerate(loader, start=1):
        b = unpack(raw, device)
        out = forward_model(model, b, use_representation=True, amp=amp, device=device)
        pred = out["residual_xy_m"]
        target = b["target_residual_xy_m"].to(pred.dtype)
        valid = b["target_valid"].bool()
        footprint = b["target_source_mask_tube"][:, -1].float()
        present = footprint.flatten(1).any(dim=1)
        eligible = valid & present[:, None]
        moving = true_moving_mask(b["target_displacement_xy_m"], valid)

        # Independent full-loss checks use detached predictions for overlap so
        # they do not enlarge the retained autograd graph.
        if bool(valid.any()):
            full_position_check += float(
                F.smooth_l1_loss(
                    pred[valid], target[valid], reduction="sum", beta=1.0
                ).detach().float().cpu()
            ) / float(2 * global_valid_labels)
        with torch.no_grad():
            full_ov, full_ov_stats = soft_transport_overlap_loss(
                pred.detach().float(),
                b["target_residual_xy_m"].detach().float(),
                footprint.detach().float(),
                valid,
                patch_resolution_m=patch_resolution_m,
            )
        full_n = int(full_ov_stats["transport_overlap_labels"])
        full_overlap_check += float(full_ov.detach().cpu()) * full_n / float(global_eligible_labels)

        for h in range(FUTURE_FRAMES):
            pt = position_horizon_term(
                pred,
                target,
                valid,
                h,
                global_valid_labels=global_valid_labels,
            )
            ot = overlap_horizon_term(
                pred,
                b["target_residual_xy_m"],
                footprint,
                valid,
                h,
                global_eligible_labels=global_eligible_labels,
                patch_resolution_m=patch_resolution_m,
            )
            pos_raw[h] += float(pt["element_sum"].detach().float().cpu())
            pos_contrib[h] += float(pt["objective_contribution"].detach().float().cpu())
            n_ov = int(ot["labels"])
            if n_ov:
                ov_raw[h] += float(ot["horizon_mean"].detach().float().cpu()) * n_ov
            ov_contrib[h] += float(ot["objective_contribution"].detach().float().cpu())

            m = moving[:, h]
            if bool(m.any()):
                tm_loss_sum = F.smooth_l1_loss(
                    pred[:, h][m].detach().float(),
                    b["target_residual_xy_m"][:, h][m].detach().float(),
                    reduction="sum",
                    beta=1.0,
                )
                tm_pos_raw[h] += float(tm_loss_sum.cpu())
                tm_coord_count[h] += int(m.sum().item()) * 2
                tm_error_sum[h] += float(
                    torch.linalg.vector_norm(
                        pred[:, h][m].detach().float()
                        - b["target_residual_xy_m"][:, h][m].detach().float(),
                        dim=-1,
                    ).sum().cpu()
                )

            gp_out, gp_head = _loss_gradients(pt["objective_contribution"], pred, head_params)
            if n_ov:
                go_out, go_head = _loss_gradients(ot["objective_contribution"], pred, head_params)
            else:
                go_out = torch.zeros_like(gp_out)
                go_head = torch.zeros_like(gp_head)
            gc_out = gp_out + EXPECTED_OVERLAP_WEIGHT * go_out

            vh = valid[:, h]
            eh = eligible[:, h]
            if bool(vh.any()):
                pos_out_sq[h] += float((gp_out[:, h][vh].double() ** 2).sum().cpu())
                combined_out_sq[h] += float((gc_out[:, h][vh].double() ** 2).sum().cpu())
            if bool(eh.any()):
                ov_out_sq[h] += float((go_out[:, h][eh].double() ** 2).sum().cpu())

            head_pos[h] += gp_head.detach().double().cpu()
            head_ov[h] += go_head.detach().double().cpu()

        seen += int(pred.shape[0])
        if bi == 1 or bi % 4 == 0 or bi == len(loader):
            print(f"h0_horizon_gradient {bi}/{len(loader)} sources={seen}", flush=True)

        # All gradients were obtained through autograd.grad; no .grad state is
        # expected to be populated.  Dropping these references releases the
        # retained per-batch graph before the next forward pass.
        del out, pred

    if seen != int(flat["features"].shape[0]):
        raise RuntimeError("DataLoader did not traverse every supervised validation source")

    position_total = float(sum(pos_contrib))
    overlap_total = float(sum(ov_contrib))
    combined_total = position_total + EXPECTED_OVERLAP_WEIGHT * overlap_total
    pos_decomp_err = abs(position_total - full_position_check)
    ov_decomp_err = abs(overlap_total - full_overlap_check)
    if pos_decomp_err > 2e-5 or ov_decomp_err > 2e-5:
        raise RuntimeError(
            f"horizon loss decomposition failed: position={pos_decomp_err:.3g} overlap={ov_decomp_err:.3g}"
        )

    weighted_head_ov = [EXPECTED_OVERLAP_WEIGHT * x for x in head_ov]
    combined_head = [head_pos[h] + weighted_head_ov[h] for h in range(FUTURE_FRAMES)]
    pos_head_attr = head_gradient_attribution(head_pos)
    ov_head_attr = head_gradient_attribution(weighted_head_ov)
    combined_head_attr = head_gradient_attribution(combined_head)
    pos_energy = squared_energy_shares(pos_out_sq)
    ov_energy = squared_energy_shares(ov_out_sq)
    combined_energy = squared_energy_shares(combined_out_sq)

    horizon_rows = []
    for h, seconds in enumerate(HORIZONS_S):
        n_valid = valid_counts[h]
        n_moving = moving_counts[h]
        n_eligible = eligible_counts[h]
        pos_mean = pos_raw[h] / float(2 * n_valid) if n_valid else float("nan")
        tm_pos_mean = tm_pos_raw[h] / float(tm_coord_count[h]) if tm_coord_count[h] else float("nan")
        tm_ade = tm_error_sum[h] / float(n_moving) if n_moving else float("nan")
        ov_mean = ov_raw[h] / float(n_eligible) if n_eligible else float("nan")
        weighted_ov_sq = (EXPECTED_OVERLAP_WEIGHT ** 2) * ov_out_sq[h]
        row = {
            "horizon_s": float(seconds),
            "valid_labels": n_valid,
            "valid_label_share": n_valid / float(global_valid_labels),
            "true_moving_labels": n_moving,
            "true_moving_fraction_of_valid": n_moving / float(n_valid) if n_valid else float("nan"),
            "overlap_eligible_labels": n_eligible,
            "overlap_label_share": n_eligible / float(global_eligible_labels),
            "position_smooth_l1_mean": pos_mean,
            "true_moving_position_smooth_l1_mean": tm_pos_mean,
            "true_moving_learned_ade_m": tm_ade,
            "overlap_loss_mean": ov_mean,
            "overlap_soft_iou": 1.0 - ov_mean if np.isfinite(ov_mean) else float("nan"),
            "position_objective_contribution": pos_contrib[h],
            "position_loss_share": pos_contrib[h] / position_total if position_total > 0 else float("nan"),
            "overlap_objective_contribution": ov_contrib[h],
            "overlap_loss_share": ov_contrib[h] / overlap_total if overlap_total > 0 else float("nan"),
            "weighted_overlap_objective_contribution": EXPECTED_OVERLAP_WEIGHT * ov_contrib[h],
            "combined_motion_objective_contribution": pos_contrib[h] + EXPECTED_OVERLAP_WEIGHT * ov_contrib[h],
            "combined_motion_loss_share": (
                (pos_contrib[h] + EXPECTED_OVERLAP_WEIGHT * ov_contrib[h]) / combined_total
                if combined_total > 0 else float("nan")
            ),
            "position_output_grad_rms_on_valid": math.sqrt(pos_out_sq[h] / float(2 * n_valid))
            if n_valid else float("nan"),
            "weighted_overlap_output_grad_rms_on_eligible": math.sqrt(weighted_ov_sq / float(2 * n_eligible))
            if n_eligible else float("nan"),
            "combined_output_grad_rms_on_valid": math.sqrt(combined_out_sq[h] / float(2 * n_valid))
            if n_valid else float("nan"),
            "position_output_grad_energy_share": pos_energy[h],
            "weighted_overlap_output_grad_energy_share": ov_energy[h],
            "combined_output_grad_energy_share": combined_energy[h],
            "position_head_grad_l2": pos_head_attr["rows"][h]["l2"],
            "weighted_overlap_head_grad_l2": ov_head_attr["rows"][h]["l2"],
            "combined_head_grad_l2": combined_head_attr["rows"][h]["l2"],
            "combined_head_grad_norm_share": combined_head_attr["rows"][h]["norm_share"],
            "combined_head_grad_cosine_to_total": combined_head_attr["rows"][h]["cosine_to_total"],
            "combined_head_grad_projection_share": combined_head_attr["rows"][h]["projection_share"],
        }
        horizon_rows.append(row)

    short = horizon_rows[0]
    long = horizon_rows[-1]
    ratios = {
        "valid_labels_0p5s_over_3p0s": _safe_ratio(short["valid_labels"], long["valid_labels"]),
        "position_mean_0p5s_over_3p0s": _safe_ratio(
            short["position_smooth_l1_mean"], long["position_smooth_l1_mean"]
        ),
        "combined_output_grad_rms_0p5s_over_3p0s": _safe_ratio(
            short["combined_output_grad_rms_on_valid"], long["combined_output_grad_rms_on_valid"]
        ),
        "combined_head_grad_l2_0p5s_over_3p0s": _safe_ratio(
            short["combined_head_grad_l2"], long["combined_head_grad_l2"]
        ),
        "combined_loss_contribution_0p5s_over_3p0s": _safe_ratio(
            short["combined_motion_objective_contribution"], long["combined_motion_objective_contribution"]
        ),
    }

    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "checkpoint_variant": str(ck.get("variant")),
        "val_cache": str(Path(a.val_cache).resolve()),
        "num_windows": len(records),
        "num_supervised_sources": int(flat["features"].shape[0]),
        "amp_bfloat16": bool(amp),
        "patch_resolution_m": patch_resolution_m,
        "overlap_weight": EXPECTED_OVERLAP_WEIGHT,
        "global_counts": {
            "valid_labels": global_valid_labels,
            "overlap_eligible_labels": global_eligible_labels,
            "true_moving_labels": int(moving_all.sum().item()),
        },
        "global_motion_objective": {
            "position_smooth_l1": position_total,
            "overlap_loss": overlap_total,
            "weighted_overlap_loss": EXPECTED_OVERLAP_WEIGHT * overlap_total,
            "combined_motion_loss": combined_total,
            "position_decomposition_check": full_position_check,
            "overlap_decomposition_check": full_overlap_check,
            "position_decomposition_abs_error": pos_decomp_err,
            "overlap_decomposition_abs_error": ov_decomp_err,
        },
        "residual_head_gradient": {
            "parameter_count": head_width,
            "position_total_l2": pos_head_attr["total_l2"],
            "weighted_overlap_total_l2": ov_head_attr["total_l2"],
            "combined_total_l2": combined_head_attr["total_l2"],
            "combined_sum_horizon_l2": combined_head_attr["sum_horizon_l2"],
            "combined_projection_share_sum": combined_head_attr["projection_share_sum"],
        },
        "horizons": horizon_rows,
        "short_vs_long_ratios": ratios,
        "contracts": {
            "position": "exact_frozen_micro_smooth_l1_decomposed_by_horizon_v1",
            "overlap": "exact_frozen_micro_transport_overlap_decomposed_by_horizon_v1",
            "combined": "position_plus_0p25_overlap_no_existence_v1",
            "output_gradient_share": "squared_l2_energy_share_over_disjoint_residual_horizon_slots_v1",
            "head_projection_share": "dot_g_h_g_total_over_norm_g_total_squared_sums_to_one_v1",
            "diagnostic_only": True,
            "no_training": True,
            "no_future_occupancy_read": True,
        },
    }

    print("\n=== V17-RL H0 HORIZON GRADIENT ATTRIBUTION ===")
    print(
        f"checkpoint_epoch={result['checkpoint_epoch']} sources={result['num_supervised_sources']} "
        f"valid={global_valid_labels} overlapEligible={global_eligible_labels} "
        f"lambda={EXPECTED_OVERLAP_WEIGHT}"
    )
    print(
        f"global: position={position_total:.6f} overlap={overlap_total:.6f} "
        f"weightedOverlap={EXPECTED_OVERLAP_WEIGHT*overlap_total:.6f} "
        f"motion={combined_total:.6f}"
    )
    print(
        "h    valid    tm   posMean  ovIoU  posLoss%  ovLoss%  motion%  "
        "outGradRMS outEnergy% headL2 headNorm% headProj% cosTotal"
    )
    for r in horizon_rows:
        print(
            f"{r['horizon_s']:>3.1f} {r['valid_labels']:>7d} {r['true_moving_labels']:>5d} "
            f"{r['position_smooth_l1_mean']:>8.5f} {100*r['overlap_soft_iou']:>7.2f} "
            f"{100*r['position_loss_share']:>8.2f} {100*r['overlap_loss_share']:>7.2f} "
            f"{100*r['combined_motion_loss_share']:>8.2f} "
            f"{r['combined_output_grad_rms_on_valid']:>10.6g} "
            f"{100*r['combined_output_grad_energy_share']:>9.2f} "
            f"{r['combined_head_grad_l2']:>7.4g} "
            f"{100*r['combined_head_grad_norm_share']:>8.2f} "
            f"{100*r['combined_head_grad_projection_share']:>8.2f} "
            f"{r['combined_head_grad_cosine_to_total']:>8.3f}"
        )

    print("\n=== 0.5s / 3.0s RATIOS ===")
    for k, v in ratios.items():
        print(f"{k}={v:.6f}")
    print("\n=== DECOMPOSITION CHECKS ===")
    print(f"position_abs_error={pos_decomp_err:.9g}")
    print(f"overlap_abs_error={ov_decomp_err:.9g}")
    print(f"head_projection_share_sum={combined_head_attr['projection_share_sum']:.9f}")
    print("decision=diagnostic_only; do not enable horizon weighting until these results are interpreted")

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
