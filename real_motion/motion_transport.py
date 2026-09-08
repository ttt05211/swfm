"""Causal object-motion prediction for explicit rigid occupancy transport.

P0-F9 v12 deliberately separates *where an observed source object moves* from
voxel generation.  Inputs are occupancy-only Strong-W2Det components and their
backward history; nuScenes annotations are used only to build training targets
and evaluation diagnostics.

The first learned contract is intentionally small: predict six planar center
residuals relative to the Strong/KTA constant-velocity anchor plus six existence
logits.  Yaw is excluded because the v11 rigid oracle showed that future center
accounts for the overwhelming majority of the rigid-transport headroom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import OccupancyGrid
from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .nuscenes_adapter import category_to_dynamic_class

MOTION_TRANSPORT_CACHE_VERSION = "p0_f9_motion_transport_v1"
HISTORY_FRAMES = 6
FUTURE_FRAMES = 6
CLASS_TO_SLOT = {int(c): i for i, c in enumerate(DYNAMIC_CLASS_IDS)}

# 2 current position + 2 current velocity + speed + log size + xyz extent +
# KTA match + class one-hot + 6x2 backward offsets + 6 validity + 5x2 segment
# velocities.  All metric quantities are represented in the current t0 ego
# frame, making the feature independent of global-map orientation.
FEATURE_NAMES = (
    "current_x_norm", "current_y_norm",
    "current_vx_norm", "current_vy_norm", "current_speed_norm",
    "log_voxel_count", "extent_x_norm", "extent_y_norm", "extent_z_norm",
    "kta_matched",
    *tuple(f"class_{int(c)}" for c in DYNAMIC_CLASS_IDS),
    *tuple(f"hist_offset_{t}_{a}" for t in range(HISTORY_FRAMES) for a in ("x", "y")),
    *tuple(f"hist_valid_{t}" for t in range(HISTORY_FRAMES)),
    *tuple(f"hist_vel_{t}_{a}" for t in range(HISTORY_FRAMES - 1) for a in ("x", "y")),
)
FEATURE_DIM = len(FEATURE_NAMES)


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    return arr @ np.asarray(T, dtype=np.float64)[:3, :3].T + np.asarray(T, dtype=np.float64)[:3, 3]


def world_points_to_t0(points_world: np.ndarray, t0_ego_to_world: np.ndarray) -> np.ndarray:
    return _transform_points(np.linalg.inv(np.asarray(t0_ego_to_world, dtype=np.float64)), points_world)


def world_vec_to_t0(vec_world: np.ndarray, t0_ego_to_world: np.ndarray) -> np.ndarray:
    R = np.asarray(t0_ego_to_world, dtype=np.float64)[:3, :3]
    return np.asarray(vec_world, dtype=np.float64) @ R


def t0_xy_to_world(xy_t0: np.ndarray, t0_ego_to_world: np.ndarray, z_world: float) -> np.ndarray:
    xy = np.asarray(xy_t0, dtype=np.float64)
    if xy.shape != (2,):
        raise ValueError("xy_t0 must have shape [2]")
    # Use t0-ground-plane XY and recover world XY through the full ego pose.
    p0 = np.asarray([xy[0], xy[1], 0.0], dtype=np.float64)
    pw = _transform_points(np.asarray(t0_ego_to_world, dtype=np.float64), p0[None])[0]
    pw[2] = float(z_world)
    return pw


def dynamic_annotations(nusc, sample_token: str) -> list[dict]:
    sample = nusc.get("sample", str(sample_token))
    out = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cid = category_to_dynamic_class(ann["category_name"])
        if cid is None:
            continue
        out.append({
            "instance_token": str(ann["instance_token"]),
            "class_id": int(cid),
            "center_world": np.asarray(ann["translation"], dtype=np.float64),
        })
    out.sort(key=lambda r: (int(r["class_id"]), str(r["instance_token"])))
    return out


def annotation_map(nusc, sample_token: str) -> dict[str, dict]:
    return {str(r["instance_token"]): r for r in dynamic_annotations(nusc, sample_token)}


def match_sources_to_annotations(
    components: Sequence[Mapping], annotations: Sequence[Mapping], *, max_distance_m: float = 4.0
) -> list[str | None]:
    """Deterministic one-to-one same-class source/GT matching for labels only."""
    pairs = []
    for ci, comp in enumerate(components):
        cc = np.asarray(comp["centroid_world"], dtype=np.float64)
        for ai, ann in enumerate(annotations):
            if int(comp["class_id"]) != int(ann["class_id"]):
                continue
            d = float(np.linalg.norm(cc[:2] - np.asarray(ann["center_world"], dtype=np.float64)[:2]))
            if d <= float(max_distance_m):
                pairs.append((d, ci, ai, str(ann["instance_token"])))
    pairs.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    used_c, used_a = set(), set()
    tokens: list[str | None] = [None] * len(components)
    for _, ci, ai, token in pairs:
        if ci in used_c or ai in used_a:
            continue
        used_c.add(ci); used_a.add(ai)
        tokens[int(ci)] = str(token)
    return tokens


def backward_component_tracks(
    components_by_frame: Sequence[Sequence[Mapping]], *, frame_dt_s: float = 0.5,
    max_speed_mps: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Causally backtrack t0 Strong components through adjacent history frames.

    Matching is one-to-one, same-class and nearest-neighbour under the same
    25 m/s-style gate used by Strong-W2Det.  No annotation identity is used.
    Returns world centers ``[N,T,3]`` and validity ``[N,T]``.
    """
    if len(components_by_frame) != HISTORY_FRAMES:
        raise ValueError(f"expected {HISTORY_FRAMES} history frames")
    current = list(components_by_frame[-1])
    N = len(current)
    centers = np.zeros((N, HISTORY_FRAMES, 3), dtype=np.float64)
    valid = np.zeros((N, HISTORY_FRAMES), dtype=bool)
    for i, comp in enumerate(current):
        centers[i, -1] = np.asarray(comp["centroid_world"], dtype=np.float64)
        valid[i, -1] = True

    gate = float(frame_dt_s) * float(max_speed_mps)
    for t in range(HISTORY_FRAMES - 2, -1, -1):
        prev = list(components_by_frame[t])
        pairs = []
        for i, comp0 in enumerate(current):
            if not valid[i, t + 1]:
                continue
            ref = centers[i, t + 1]
            for j, comp in enumerate(prev):
                if int(comp["class_id"]) != int(comp0["class_id"]):
                    continue
                p = np.asarray(comp["centroid_world"], dtype=np.float64)
                d = float(np.linalg.norm(ref[:2] - p[:2]))
                if d <= gate:
                    pairs.append((d, i, j))
        pairs.sort(key=lambda x: (x[0], x[1], x[2]))
        used_i, used_j = set(), set()
        for _, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i); used_j.add(j)
            centers[i, t] = np.asarray(prev[j]["centroid_world"], dtype=np.float64)
            valid[i, t] = True
    return centers, valid


def component_extent_xyz_m(component: Mapping, grid: OccupancyGrid = OccupancyGrid()) -> np.ndarray:
    idx = np.asarray(component["voxel_indices"], dtype=np.int64)
    if len(idx) == 0:
        return np.zeros(3, dtype=np.float32)
    span = idx.max(axis=0) - idx.min(axis=0) + 1
    return (span * np.asarray(grid.voxel_size, dtype=np.float64)).astype(np.float32)


def build_source_features(
    current_components: Sequence[Mapping],
    current_velocities_world: Mapping[int, np.ndarray],
    track_centers_world: np.ndarray,
    track_valid: np.ndarray,
    t0_ego_to_world: np.ndarray,
    *, frame_dt_s: float = 0.5,
    grid: OccupancyGrid = OccupancyGrid(),
) -> np.ndarray:
    N = len(current_components)
    if track_centers_world.shape != (N, HISTORY_FRAMES, 3) or track_valid.shape != (N, HISTORY_FRAMES):
        raise ValueError("history-track shape mismatch")
    out = np.zeros((N, FEATURE_DIM), dtype=np.float32)
    for i, comp in enumerate(current_components):
        cur_world = np.asarray(comp["centroid_world"], dtype=np.float64)
        cur_t0 = world_points_to_t0(cur_world[None], t0_ego_to_world)[0]
        v_world = np.asarray(current_velocities_world.get(i, np.zeros(3)), dtype=np.float64)
        v_t0 = world_vec_to_t0(v_world, t0_ego_to_world)
        extent = component_extent_xyz_m(comp, grid)
        feat = [
            cur_t0[0] / 40.0, cur_t0[1] / 40.0,
            v_t0[0] / 20.0, v_t0[1] / 20.0,
            float(np.linalg.norm(v_t0[:2])) / 20.0,
            math.log1p(float(comp.get("voxel_count", len(comp["voxel_indices"])))) / 8.0,
            extent[0] / 10.0, extent[1] / 10.0, extent[2] / 5.0,
            1.0 if i in current_velocities_world else 0.0,
        ]
        cls = [0.0] * len(DYNAMIC_CLASS_IDS)
        cls[CLASS_TO_SLOT[int(comp["class_id"])]] = 1.0
        feat.extend(cls)

        hist_t0 = np.zeros((HISTORY_FRAMES, 3), dtype=np.float64)
        for t in range(HISTORY_FRAMES):
            if track_valid[i, t]:
                hist_t0[t] = world_points_to_t0(track_centers_world[i, t][None], t0_ego_to_world)[0]
        offsets = hist_t0[:, :2] - cur_t0[None, :2]
        offsets[~track_valid[i]] = 0.0
        feat.extend((offsets / 20.0).reshape(-1).tolist())
        feat.extend(track_valid[i].astype(np.float32).tolist())

        seg = np.zeros((HISTORY_FRAMES - 1, 2), dtype=np.float64)
        for t in range(HISTORY_FRAMES - 1):
            if track_valid[i, t] and track_valid[i, t + 1]:
                seg[t] = (hist_t0[t + 1, :2] - hist_t0[t, :2]) / float(frame_dt_s)
        feat.extend((seg / 20.0).reshape(-1).tolist())
        arr = np.asarray(feat, dtype=np.float32)
        if arr.shape != (FEATURE_DIM,):
            raise AssertionError(f"feature shape {arr.shape} != {(FEATURE_DIM,)}")
        out[i] = arr
    return out


def build_motion_targets(
    current_components: Sequence[Mapping], current_velocities_world: Mapping[int, np.ndarray],
    source_tokens: Sequence[str | None], future_maps: Sequence[Mapping[str, Mapping]],
    t0_ego_to_world: np.ndarray, *, frame_dt_s: float = 0.5,
) -> dict[str, np.ndarray]:
    N = len(current_components)
    if len(source_tokens) != N or len(future_maps) != FUTURE_FRAMES:
        raise ValueError("target input length mismatch")
    anchors = np.zeros((N, FUTURE_FRAMES, 2), dtype=np.float32)
    residual = np.zeros_like(anchors)
    target_xy = np.zeros_like(anchors)
    existence = np.zeros((N, FUTURE_FRAMES), dtype=np.float32)
    target_valid = np.zeros((N, FUTURE_FRAMES), dtype=bool)
    supervised = np.asarray([t is not None for t in source_tokens], dtype=bool)
    for i, comp in enumerate(current_components):
        cur_world = np.asarray(comp["centroid_world"], dtype=np.float64)
        cur_t0 = world_points_to_t0(cur_world[None], t0_ego_to_world)[0, :2]
        v_world = np.asarray(current_velocities_world.get(i, np.zeros(3)), dtype=np.float64)
        v_t0 = world_vec_to_t0(v_world, t0_ego_to_world)[:2]
        token = source_tokens[i]
        for h in range(FUTURE_FRAMES):
            dt = (h + 1) * float(frame_dt_s)
            anchor = cur_t0 + v_t0 * dt
            anchors[i, h] = anchor.astype(np.float32)
            if token is None:
                continue
            annh = future_maps[h].get(str(token))
            if annh is None:
                continue
            p = world_points_to_t0(np.asarray(annh["center_world"], dtype=np.float64)[None], t0_ego_to_world)[0, :2]
            target_xy[i, h] = p.astype(np.float32)
            residual[i, h] = (p - anchor).astype(np.float32)
            existence[i, h] = 1.0
            target_valid[i, h] = True
    return {
        "anchors_xy_t0_m": anchors,
        "target_xy_t0_m": target_xy,
        "target_residual_xy_m": residual,
        "existence": existence,
        "target_valid": target_valid,
        "supervised_source": supervised,
    }


class MotionTransportHead(nn.Module):
    """Tiny per-source MLP: KTA residual center trajectory + existence."""
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 128,
                 future_frames: int = FUTURE_FRAMES):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.future_frames = int(future_frames)
        self.backbone = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.residual_head = nn.Linear(hidden_dim, self.future_frames * 2)
        self.existence_head = nn.Linear(hidden_dim, self.future_frames)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must be [N,{self.feature_dim}]")
        x = self.backbone(features)
        return {
            "residual_xy_m": self.residual_head(x).view(-1, self.future_frames, 2),
            "existence_logits": self.existence_head(x),
        }


def motion_transport_loss(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    pred = outputs["residual_xy_m"]
    target = batch["target_residual_xy_m"].to(pred.dtype)
    valid = batch["target_valid"].bool()
    if bool(valid.any()):
        traj = F.smooth_l1_loss(pred[valid], target[valid], reduction="mean", beta=1.0)
    else:
        traj = pred.sum() * 0.0
    logits = outputs["existence_logits"]
    labels = batch["existence"].to(logits.dtype)
    supervised = batch.get("supervised_source")
    if supervised is None:
        emask = torch.ones_like(labels, dtype=torch.bool)
    else:
        emask = supervised.bool().unsqueeze(-1).expand_as(labels)
    if bool(emask.any()):
        exist = F.binary_cross_entropy_with_logits(logits[emask], labels[emask])
    else:
        exist = logits.sum() * 0.0
    total = traj + exist
    return total, {
        "loss": float(total.detach().cpu()),
        "trajectory_smooth_l1": float(traj.detach().cpu()),
        "existence_bce": float(exist.detach().cpu()),
        "trajectory_labels": int(valid.sum().item()),
        "existence_labels": int(emask.sum().item()),
    }


def trajectory_errors(pred_residual: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> dict:
    target = batch["target_residual_xy_m"].to(pred_residual.dtype)
    valid = batch["target_valid"].bool()
    anchor_err = target.norm(dim=-1)
    learned_err = (pred_residual - target).norm(dim=-1)
    out = {}
    for name, err in (("kta", anchor_err), ("learned", learned_err)):
        vals = err[valid]
        out[f"{name}_ade_m"] = float(vals.mean().item()) if vals.numel() else float("nan")
        # FDE is the latest valid horizon for each supervised source.
        fdes = []
        for i in range(err.shape[0]):
            ids = torch.nonzero(valid[i], as_tuple=False).flatten()
            if ids.numel():
                fdes.append(err[i, ids[-1]])
        out[f"{name}_fde_m"] = float(torch.stack(fdes).mean().item()) if fdes else float("nan")
    return out
