"""Incremental P0-F4 -> P0-F5 cache upgrade helpers.

The upgrade keeps every frozen routing/context/anchor latent from the already
built P0-F4 cache and replaces only the future target with the VAE encoding of
the occupancy-space sparse repair endpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .msp_wm_cache import MSP_WM_CACHE_VERSION_V2

P0_F4_ANCHOR_CONTRACT = "strong_w2det_occ_only_v1"
P0_F4_HISTORY_CONTRACT = "full_native_occ_history_6f"
P0_F4_TARGET = "full_future_gt_latent"
P0_F5_REPAIR_CONTRACT = "strong_anchor_outside_support_gt_dynamic_inside_support_v1"
P0_F5_LOSS_CONTRACT = (
    "strong_anchor_to_occ_repair_endpoint_local_flow_full_history_context_no_auxiliary_losses"
)
P0_F5_TARGET = "occupancy_sparse_repair_endpoint_vae_latent"

REUSED_TENSOR_KEYS = (
    "full_history_latent",
    "anchor_future_latent",
    "window_origins",
    "window_valid",
    "msp_write_support_latent",
    "trajectory",
)


def validate_p0_f4_upgrade_source(version: str, metadata: Mapping) -> None:
    """Reject any source cache that is not the frozen P0-F4 contract."""
    if version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError(f"incremental P0-F5 upgrade requires P0-F4/v2 cache, got {version}")
    if metadata.get("anchor_contract") != P0_F4_ANCHOR_CONTRACT:
        raise RuntimeError("source cache is not Strong-W2Det anchored")
    if metadata.get("history_contract") != P0_F4_HISTORY_CONTRACT:
        raise RuntimeError("source cache does not contain full-history latents")
    if metadata.get("target") != P0_F4_TARGET:
        raise RuntimeError("source cache is not the frozen P0-F4 full-GT target contract")
    if int(metadata.get("topk", -1)) != 2:
        raise RuntimeError("source cache is not frozen Top-2")
    if list(metadata.get("window_hw", [])) != [20, 20]:
        raise RuntimeError("source cache prediction window is not 20x20")
    if list(metadata.get("context_hw", [])) != [40, 40]:
        raise RuntimeError("source cache history context is not 40x40")
    if metadata.get("vae_mode") != "mean":
        raise RuntimeError("source cache must use deterministic VAE posterior means")


def make_p0_f5_upgrade_metadata(
    source_metadata: Mapping,
    *,
    source_cache: str | Path,
    source_index_sha256: str,
) -> dict:
    """Preserve frozen P0-F4 provenance while switching only target semantics."""
    out = dict(source_metadata)
    out.update({
        "protocol": "p0_f5_strong_w2det_occ_repair_endpoint_top2_v1",
        "repair_endpoint_contract": P0_F5_REPAIR_CONTRACT,
        "loss_contract": P0_F5_LOSS_CONTRACT,
        "source": "strong_w2det_anchor_latent",
        "target": P0_F5_TARGET,
        "latent_loss_mask": "none",
        "incremental_upgrade": True,
        "incremental_upgrade_from": str(Path(source_cache).resolve()),
        "incremental_upgrade_source_index_sha256": str(source_index_sha256),
        "incremental_reused_tensor_keys": list(REUSED_TENSOR_KEYS),
        "incremental_new_gpu_encode": ["repair_target_latent"],
    })
    return out


def build_upgraded_sample(
    source_sample: Mapping,
    repair_target_latent: torch.Tensor,
    *,
    repair_target_occ: torch.Tensor | None = None,
) -> dict:
    """Copy frozen P0-F4 tensors bit-exactly and install only the new target."""
    missing = [k for k in ("sample_id", "scene_name", *REUSED_TENSOR_KEYS) if k not in source_sample]
    if missing:
        raise KeyError(f"P0-F4 source sample missing upgrade keys {missing}")
    if not torch.is_tensor(repair_target_latent) or repair_target_latent.ndim != 4:
        raise ValueError("repair_target_latent must be [T,C,H,W]")
    anchor = source_sample["anchor_future_latent"]
    if tuple(repair_target_latent.shape) != tuple(anchor.shape):
        raise ValueError("repair target latent must match cached anchor latent shape")

    out = {
        "sample_id": str(source_sample["sample_id"]),
        "scene_name": str(source_sample["scene_name"]),
        **{
            k: source_sample[k].clone() if torch.is_tensor(source_sample[k]) else source_sample[k]
            for k in REUSED_TENSOR_KEYS
        },
        "repair_target_latent": repair_target_latent.detach().cpu().clone(),
    }

    # Preserve compact validation payload exactly. P0-F5 adds only the repair
    # endpoint occupancy so the evaluator can assert target/oracle identity.
    for key, value in source_sample.items():
        if str(key).startswith("eval_"):
            out[key] = value.clone() if torch.is_tensor(value) else value
    if repair_target_occ is not None:
        if not torch.is_tensor(repair_target_occ) or repair_target_occ.ndim != 4:
            raise ValueError("repair_target_occ must be [T,X,Y,Z]")
        out["eval_repair_target_occ"] = repair_target_occ.detach().cpu().clone()
    return out
