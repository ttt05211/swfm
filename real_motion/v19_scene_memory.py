"""V19 causal scene-memory primitives.

This module is intentionally additive to the frozen Clean-E14/V18 path.

It provides two pieces of causal state:

1. DynamicSourceMemory
   Tracks motion-capable Strong components across the six observed history
   frames. Current-t0 sources are kept in the exact frozen Strong order;
   recently missing tracks are appended afterwards as dormant memory sources.
   No annotation identity is used.

2. StaticWorldMemory
   Accumulates only lidar-observed non-motion-capable occupancy in a sparse
   world-coordinate voxel table. Observed free cells invalidate stale memory,
   while observed dynamic cells neither enter nor erase the static map.

The module does not change V18 predictions by itself. It only builds the
causal memory state and V18-compatible tensors consumed by the V19 adapters and
diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from .local_st_world_model import (
    build_local_semantic_tubes,
    history_offsets_from_features,
)
from .local_st_world_model_v17 import (
    frame_motion_features_from_flat,
    target_source_mask_from_tube,
)
from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .motion_transport import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FUTURE_FRAMES,
    HISTORY_FRAMES,
    world_points_to_t0,
    world_vec_to_t0,
)
from .runtime_fastpath import extract_instances_cropped_exact
from .strong_w2det import StrongW2DetConfig


PROVENANCE_OBSERVED_CURRENT = "observed_current"
PROVENANCE_OBSERVED_HISTORY = "observed_history"
PROVENANCE_PREDICTED_BIRTH = "predicted_birth"
PROVENANCE_PERSISTENT_PREDICTION = "persistent_prediction"
SOURCE_STATUS_DIM = 4

_DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
_DYNAMIC_SET = set(_DYNAMIC_IDS)
_NAME_TO_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    mat = np.asarray(T, dtype=np.float64)
    return arr @ mat[:3, :3].T + mat[:3, 3]


def _voxel_centers(indices: np.ndarray, grid) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError("voxel indices must be [N,3]")
    origin = np.asarray(
        [grid.x_min, grid.y_min, grid.z_min], dtype=np.float64
    )
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    return origin[None] + (idx.astype(np.float64) + 0.5) * step[None]


def _canonical_local_xyz(component: Mapping, ego_to_world: np.ndarray, grid) -> np.ndarray:
    idx = np.asarray(component["voxel_indices"], dtype=np.int64)
    if len(idx) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts_ego = _voxel_centers(idx, grid)
    pts_world = _transform_points(ego_to_world, pts_ego)
    center = np.asarray(component["centroid_world"], dtype=np.float64)
    return pts_world - center[None]


def _shape_extent(local_xyz: np.ndarray, grid) -> np.ndarray:
    pts = np.asarray(local_xyz, dtype=np.float64)
    if len(pts) == 0:
        return np.asarray(grid.voxel_size, dtype=np.float64)
    span = pts.max(axis=0) - pts.min(axis=0)
    return span + np.asarray(grid.voxel_size, dtype=np.float64)


@dataclass
class SourceTrack:
    """One causal source-memory track in world coordinates."""

    track_id: int
    class_id: int
    canonical_xyz_local: np.ndarray
    centers_world: np.ndarray
    valid_history: np.ndarray
    velocity_world: np.ndarray
    last_observed_frame: int
    confidence: float
    provenance: str
    current_component_index: int | None = None
    last_component_voxel_count: int = 0
    last_real_observation_age_s_override: float | None = None

    def __post_init__(self):
        self.track_id = int(self.track_id)
        self.class_id = int(self.class_id)
        if self.class_id not in _DYNAMIC_SET:
            raise ValueError(f"class {self.class_id} is not motion-capable")
        self.canonical_xyz_local = np.asarray(
            self.canonical_xyz_local, dtype=np.float64
        )
        if self.canonical_xyz_local.ndim != 2 or (
            self.canonical_xyz_local.shape[1] != 3
        ):
            raise ValueError("canonical_xyz_local must be [N,3]")
        self.centers_world = np.asarray(self.centers_world, dtype=np.float64)
        self.valid_history = np.asarray(self.valid_history, dtype=bool)
        if self.centers_world.shape != (HISTORY_FRAMES, 3):
            raise ValueError("centers_world must be [6,3]")
        if self.valid_history.shape != (HISTORY_FRAMES,):
            raise ValueError("valid_history must be [6]")
        self.velocity_world = np.asarray(self.velocity_world, dtype=np.float64)
        if self.velocity_world.shape != (3,):
            raise ValueError("velocity_world must be [3]")
        self.last_observed_frame = int(self.last_observed_frame)
        self.confidence = float(self.confidence)
        self.last_component_voxel_count = int(self.last_component_voxel_count)
        if self.last_real_observation_age_s_override is not None:
            self.last_real_observation_age_s_override = float(
                self.last_real_observation_age_s_override
            )
            if self.last_real_observation_age_s_override < 0:
                raise ValueError("last real observation age must be non-negative")

    @property
    def observed_at_anchor(self) -> bool:
        if self.last_real_observation_age_s_override is not None:
            return self.last_real_observation_age_s_override <= 1e-8
        return bool(self.valid_history[-1])

    def state_age_s(self, frame_dt_s: float) -> float:
        return float(
            (HISTORY_FRAMES - 1 - self.last_observed_frame)
            * float(frame_dt_s)
        )

    def last_real_observation_age_s(self, frame_dt_s: float) -> float:
        if self.last_real_observation_age_s_override is not None:
            return float(self.last_real_observation_age_s_override)
        return self.state_age_s(frame_dt_s)

    def anchor_center_world(self, frame_dt_s: float) -> np.ndarray:
        last = np.asarray(
            self.centers_world[self.last_observed_frame], dtype=np.float64
        )
        age = self.state_age_s(frame_dt_s)
        return last + np.asarray(self.velocity_world, dtype=np.float64) * age

    def history_status(self, frame_dt_s: float) -> np.ndarray:
        age = self.last_real_observation_age_s(frame_dt_s)
        return np.asarray(
            [
                1.0 if self.observed_at_anchor else 0.0,
                min(age / 1.5, 2.0),
                float(np.clip(self.confidence, 0.0, 1.0)),
                1.0 if self.provenance == PROVENANCE_PREDICTED_BIRTH else 0.0,
            ],
            dtype=np.float32,
        )


def _last_velocity(
    centers_world: np.ndarray,
    valid: np.ndarray,
    frame_dt_s: float,
) -> np.ndarray:
    ids = np.flatnonzero(np.asarray(valid, dtype=bool))
    if len(ids) < 2:
        return np.zeros(3, dtype=np.float64)
    a, b = int(ids[-2]), int(ids[-1])
    dt = float(b - a) * float(frame_dt_s)
    if dt <= 0:
        return np.zeros(3, dtype=np.float64)
    return (
        np.asarray(centers_world[b], dtype=np.float64)
        - np.asarray(centers_world[a], dtype=np.float64)
    ) / dt


def _match_active_tracks(
    tracks: Sequence[SourceTrack],
    components: Sequence[Mapping],
    frame_index: int,
    *,
    frame_dt_s: float,
    max_speed_mps: float,
    max_missing_frames: int,
) -> list[tuple[int, int]]:
    """Deterministic same-class greedy association across short gaps."""
    pairs = []
    for ti, tr in enumerate(tracks):
        gap = int(frame_index - tr.last_observed_frame)
        missing_between = gap - 1
        if gap <= 0 or missing_between > int(max_missing_frames):
            continue
        pred = (
            np.asarray(tr.centers_world[tr.last_observed_frame], dtype=np.float64)
            + np.asarray(tr.velocity_world, dtype=np.float64)
            * (gap * float(frame_dt_s))
        )
        gate = float(max_speed_mps) * gap * float(frame_dt_s)
        for ci, comp in enumerate(components):
            if int(comp["class_id"]) != int(tr.class_id):
                continue
            cc = np.asarray(comp["centroid_world"], dtype=np.float64)
            d = float(np.linalg.norm(cc[:2] - pred[:2]))
            if d <= gate:
                pairs.append((d, ti, ci, int(tr.track_id)))
    pairs.sort(key=lambda x: (x[0], x[3], x[2]))
    used_t, used_c = set(), set()
    out = []
    for _, ti, ci, _ in pairs:
        if ti in used_t or ci in used_c:
            continue
        used_t.add(ti)
        used_c.add(ci)
        out.append((int(ti), int(ci)))
    return out


def build_dynamic_source_memory(
    history_semantics: Sequence[np.ndarray],
    history_poses: Sequence[np.ndarray],
    *,
    grid,
    strong_cfg: StrongW2DetConfig,
    frame_dt_s: float = 0.5,
    max_missing_s: float = 1.5,
    confidence_tau_s: float = 1.0,
) -> tuple[list[SourceTrack], list[list[dict]]]:
    """Build six-frame causal source tracks.

    Return order is deliberate: every t0-current source appears first in the
    exact frozen Strong component order, followed by recently missing tracks.
    """
    if len(history_semantics) != HISTORY_FRAMES:
        raise ValueError("expected six history semantic grids")
    if len(history_poses) != HISTORY_FRAMES:
        raise ValueError("expected six history poses")
    if max_missing_s < 0:
        raise ValueError("max_missing_s must be non-negative")
    max_missing_frames = int(
        math.floor(max_missing_s / float(frame_dt_s) + 1e-8)
    )

    comps_by_frame = [
        extract_instances_cropped_exact(
            np.asarray(sem, dtype=np.uint8),
            np.asarray(pose, dtype=np.float64),
            grid=grid,
            cfg=strong_cfg,
        )
        for sem, pose in zip(history_semantics, history_poses)
    ]

    tracks: list[SourceTrack] = []
    next_id = 0
    for comp in comps_by_frame[0]:
        centers = np.zeros((HISTORY_FRAMES, 3), dtype=np.float64)
        valid = np.zeros(HISTORY_FRAMES, dtype=bool)
        centers[0] = np.asarray(comp["centroid_world"], dtype=np.float64)
        valid[0] = True
        tracks.append(
            SourceTrack(
                track_id=next_id,
                class_id=int(comp["class_id"]),
                canonical_xyz_local=_canonical_local_xyz(
                    comp, np.asarray(history_poses[0]), grid
                ),
                centers_world=centers,
                valid_history=valid,
                velocity_world=np.zeros(3, dtype=np.float64),
                last_observed_frame=0,
                confidence=1.0,
                provenance=PROVENANCE_OBSERVED_HISTORY,
                current_component_index=None,
                last_component_voxel_count=int(comp["voxel_count"]),
            )
        )
        next_id += 1

    for t in range(1, HISTORY_FRAMES):
        comps = comps_by_frame[t]
        matched = _match_active_tracks(
            tracks,
            comps,
            t,
            frame_dt_s=float(frame_dt_s),
            max_speed_mps=float(strong_cfg.max_match_speed_mps),
            max_missing_frames=max_missing_frames,
        )
        used_components = set()
        for ti, ci in matched:
            tr = tracks[ti]
            comp = comps[ci]
            tr.centers_world[t] = np.asarray(
                comp["centroid_world"], dtype=np.float64
            )
            tr.valid_history[t] = True
            tr.last_observed_frame = t
            tr.velocity_world = _last_velocity(
                tr.centers_world, tr.valid_history, float(frame_dt_s)
            )
            tr.canonical_xyz_local = _canonical_local_xyz(
                comp, np.asarray(history_poses[t]), grid
            )
            tr.last_component_voxel_count = int(comp["voxel_count"])
            tr.confidence = 1.0
            if t == HISTORY_FRAMES - 1:
                tr.current_component_index = int(ci)
                tr.provenance = PROVENANCE_OBSERVED_CURRENT
            used_components.add(int(ci))

        for ci, comp in enumerate(comps):
            if ci in used_components:
                continue
            centers = np.zeros((HISTORY_FRAMES, 3), dtype=np.float64)
            valid = np.zeros(HISTORY_FRAMES, dtype=bool)
            centers[t] = np.asarray(comp["centroid_world"], dtype=np.float64)
            valid[t] = True
            tracks.append(
                SourceTrack(
                    track_id=next_id,
                    class_id=int(comp["class_id"]),
                    canonical_xyz_local=_canonical_local_xyz(
                        comp, np.asarray(history_poses[t]), grid
                    ),
                    centers_world=centers,
                    valid_history=valid,
                    velocity_world=np.zeros(3, dtype=np.float64),
                    last_observed_frame=t,
                    confidence=1.0,
                    provenance=(
                        PROVENANCE_OBSERVED_CURRENT
                        if t == HISTORY_FRAMES - 1
                        else PROVENANCE_OBSERVED_HISTORY
                    ),
                    current_component_index=(
                        int(ci) if t == HISTORY_FRAMES - 1 else None
                    ),
                    last_component_voxel_count=int(comp["voxel_count"]),
                )
            )
            next_id += 1

    current_by_index: dict[int, SourceTrack] = {}
    dormant: list[SourceTrack] = []
    for tr in tracks:
        if tr.observed_at_anchor:
            if tr.current_component_index is None:
                raise RuntimeError("t0-observed track lacks current component index")
            if tr.current_component_index in current_by_index:
                raise RuntimeError("duplicate t0 component assignment")
            tr.confidence = 1.0
            current_by_index[int(tr.current_component_index)] = tr
        else:
            age = tr.state_age_s(float(frame_dt_s))
            if age <= float(max_missing_s) + 1e-8:
                tr.confidence = float(
                    math.exp(-age / max(float(confidence_tau_s), 1e-6))
                )
                dormant.append(tr)

    n_current = len(comps_by_frame[-1])
    if set(current_by_index) != set(range(n_current)):
        raise RuntimeError(
            "source-memory tracking failed to preserve all t0 Strong components"
        )
    current_ordered = [current_by_index[i] for i in range(n_current)]
    dormant.sort(
        key=lambda tr: (
            -int(tr.last_observed_frame),
            int(tr.class_id),
            int(tr.track_id),
        )
    )
    return current_ordered + dormant, comps_by_frame


def _feature_from_track(
    tr: SourceTrack,
    t0_ego_to_world: np.ndarray,
    *,
    frame_dt_s: float,
    grid,
) -> np.ndarray:
    cur_world = tr.anchor_center_world(float(frame_dt_s))
    cur_t0 = world_points_to_t0(cur_world[None], t0_ego_to_world)[0]
    v_t0 = world_vec_to_t0(tr.velocity_world, t0_ego_to_world)
    extent = _shape_extent(tr.canonical_xyz_local, grid)

    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    feat[_NAME_TO_INDEX["current_x_norm"]] = float(cur_t0[0] / 40.0)
    feat[_NAME_TO_INDEX["current_y_norm"]] = float(cur_t0[1] / 40.0)
    feat[_NAME_TO_INDEX["current_vx_norm"]] = float(v_t0[0] / 20.0)
    feat[_NAME_TO_INDEX["current_vy_norm"]] = float(v_t0[1] / 20.0)
    feat[_NAME_TO_INDEX["current_speed_norm"]] = float(
        np.linalg.norm(v_t0[:2]) / 20.0
    )
    feat[_NAME_TO_INDEX["log_voxel_count"]] = float(
        math.log1p(max(int(tr.last_component_voxel_count), 1)) / 8.0
    )
    feat[_NAME_TO_INDEX["extent_x_norm"]] = float(extent[0] / 10.0)
    feat[_NAME_TO_INDEX["extent_y_norm"]] = float(extent[1] / 10.0)
    feat[_NAME_TO_INDEX["extent_z_norm"]] = float(extent[2] / 5.0)
    feat[_NAME_TO_INDEX["kta_matched"]] = (
        1.0 if np.linalg.norm(tr.velocity_world[:2]) > 0.0 else 0.0
    )
    feat[_NAME_TO_INDEX[f"class_{int(tr.class_id)}"]] = 1.0

    hist_t0 = np.zeros((HISTORY_FRAMES, 3), dtype=np.float64)
    for t in range(HISTORY_FRAMES):
        if tr.valid_history[t]:
            hist_t0[t] = world_points_to_t0(
                tr.centers_world[t][None], t0_ego_to_world
            )[0]
            feat[_NAME_TO_INDEX[f"hist_valid_{t}"]] = 1.0
            feat[_NAME_TO_INDEX[f"hist_offset_{t}_x"]] = float(
                (hist_t0[t, 0] - cur_t0[0]) / 20.0
            )
            feat[_NAME_TO_INDEX[f"hist_offset_{t}_y"]] = float(
                (hist_t0[t, 1] - cur_t0[1]) / 20.0
            )

    for t in range(HISTORY_FRAMES - 1):
        if tr.valid_history[t] and tr.valid_history[t + 1]:
            vel = (
                hist_t0[t + 1, :2] - hist_t0[t, :2]
            ) / float(frame_dt_s)
            feat[_NAME_TO_INDEX[f"hist_vel_{t}_x"]] = float(vel[0] / 20.0)
            feat[_NAME_TO_INDEX[f"hist_vel_{t}_y"]] = float(vel[1] / 20.0)
    return feat


def prepare_causal_arrays_from_tracks(
    tracks: Sequence[SourceTrack],
    history_semantics: Sequence[np.ndarray],
    history_poses: Sequence[np.ndarray],
    *,
    grid,
    free_label: int,
    frame_dt_s: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Build V18-compatible tensors directly from persistent source tracks."""
    if len(history_semantics) != HISTORY_FRAMES:
        raise ValueError("expected six history semantic grids")
    if len(history_poses) != HISTORY_FRAMES:
        raise ValueError("expected six history poses")
    t0_pose = np.asarray(history_poses[-1], dtype=np.float64)
    N = len(tracks)
    features_np = (
        np.stack(
            [
                _feature_from_track(
                    tr,
                    t0_pose,
                    frame_dt_s=float(frame_dt_s),
                    grid=grid,
                )
                for tr in tracks
            ],
            axis=0,
        ).astype(np.float32)
        if tracks
        else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    )
    features = torch.from_numpy(features_np)

    source_xy = np.zeros((N, 2), dtype=np.float32)
    kta = np.zeros((N, FUTURE_FRAMES, 2), dtype=np.float32)
    anchors = np.zeros_like(kta)
    class_ids = np.zeros(N, dtype=np.int64)
    valid = np.zeros((N, HISTORY_FRAMES), dtype=bool)
    status = np.zeros((N, SOURCE_STATUS_DIM), dtype=np.float32)
    for i, tr in enumerate(tracks):
        cur_world = tr.anchor_center_world(float(frame_dt_s))
        cur_t0 = world_points_to_t0(cur_world[None], t0_pose)[0]
        vt0 = world_vec_to_t0(tr.velocity_world, t0_pose)
        source_xy[i] = cur_t0[:2].astype(np.float32)
        class_ids[i] = int(tr.class_id)
        valid[i] = tr.valid_history
        status[i] = tr.history_status(float(frame_dt_s))
        for h in range(FUTURE_FRAMES):
            d = vt0[:2] * ((h + 1) * float(frame_dt_s))
            kta[i, h] = d.astype(np.float32)
            anchors[i, h] = (cur_t0[:2] + d).astype(np.float32)

    offsets = history_offsets_from_features(features)
    tube_np = build_local_semantic_tubes(
        history_semantics,
        history_poses,
        source_xy,
        offsets,
        valid,
        grid=grid,
        free_label=int(free_label),
    )
    tube = torch.from_numpy(tube_np)
    class_t = torch.from_numpy(class_ids)
    valid_t = torch.from_numpy(valid)
    frame_motion = frame_motion_features_from_flat(features)
    source_mask = target_source_mask_from_tube(
        tube,
        class_t,
        valid_t,
        features,
    )
    return {
        "features": features,
        "local_semantic_tube": tube,
        "kta_displacement_xy_m": torch.from_numpy(kta),
        "anchors_xy_t0_m": torch.from_numpy(anchors),
        "frame_motion_features": frame_motion,
        "target_source_mask_tube": source_mask,
        "source_class_id": class_t,
        "track_valid": valid_t,
        "history_status": torch.from_numpy(status),
        "source_xy_t0_m": torch.from_numpy(source_xy),
    }



def persistent_tracks_from_v18_predictions(
    current_components: Sequence[Mapping],
    source_world_points: Sequence[np.ndarray],
    current_pose: np.ndarray,
    anchors_xy_t0_m: np.ndarray | torch.Tensor,
    pred_residual_xy_m: np.ndarray | torch.Tensor,
    pred_yaw_delta_rad: np.ndarray | torch.Tensor,
    *,
    frame_dt_s: float = 0.5,
    real_observation_age_at_block_end_s: float | None = None,
) -> list[SourceTrack]:
    """Promote one V18 block's source trajectories into persistent memory.

    The six predicted future centres become the next block's six-frame source
    history directly.  No connected-component extraction or identity matching
    is performed.  Canonical geometry is materialized at the last predicted
    yaw so the next block can again predict a relative SE(2) motion.
    """
    from .local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS

    anchors = (
        anchors_xy_t0_m.detach().cpu().numpy()
        if torch.is_tensor(anchors_xy_t0_m)
        else np.asarray(anchors_xy_t0_m)
    )
    residual = (
        pred_residual_xy_m.detach().cpu().numpy()
        if torch.is_tensor(pred_residual_xy_m)
        else np.asarray(pred_residual_xy_m)
    )
    yaw = (
        pred_yaw_delta_rad.detach().cpu().numpy()
        if torch.is_tensor(pred_yaw_delta_rad)
        else np.asarray(pred_yaw_delta_rad)
    )
    N = len(current_components)
    if anchors.shape != (N, FUTURE_FRAMES, 2):
        raise ValueError("anchors must be [N,6,2]")
    if residual.shape != anchors.shape or yaw.shape != (N, FUTURE_FRAMES):
        raise ValueError("V18 prediction shape mismatch")
    if len(source_world_points) != N:
        raise ValueError("source_world_points count mismatch")

    pose = np.asarray(current_pose, dtype=np.float64)
    world_to_t0 = np.linalg.inv(pose)
    real_age = (
        FUTURE_FRAMES * float(frame_dt_s)
        if real_observation_age_at_block_end_s is None
        else float(real_observation_age_at_block_end_s)
    )
    tracks = []
    yaw_enabled = set(int(x) for x in YAW_ENABLED_CLASS_IDS)
    for i, comp in enumerate(current_components):
        src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
        src_z_t0 = float((world_to_t0 @ np.r_[src_center, 1.0])[2])
        centers = np.zeros((HISTORY_FRAMES, 3), dtype=np.float64)
        for h in range(FUTURE_FRAMES):
            xy = np.asarray(anchors[i, h], dtype=np.float64) + np.asarray(
                residual[i, h], dtype=np.float64
            )
            p0 = np.asarray(
                [xy[0], xy[1], src_z_t0, 1.0], dtype=np.float64
            )
            centers[h] = (pose @ p0)[:3]

        pts_world = np.asarray(source_world_points[i], dtype=np.float64)
        local = pts_world - src_center[None]
        final_yaw = float(yaw[i, -1]) if int(comp["class_id"]) in yaw_enabled else 0.0
        cy, sy = math.cos(final_yaw), math.sin(final_yaw)
        R = np.asarray([[cy, -sy], [sy, cy]], dtype=np.float64)
        canonical = local.copy()
        if len(canonical):
            canonical[:, :2] = canonical[:, :2] @ R.T

        valid = np.ones(HISTORY_FRAMES, dtype=bool)
        vel = _last_velocity(centers, valid, float(frame_dt_s))
        tracks.append(
            SourceTrack(
                track_id=i,
                class_id=int(comp["class_id"]),
                canonical_xyz_local=canonical,
                centers_world=centers,
                valid_history=valid,
                velocity_world=vel,
                last_observed_frame=HISTORY_FRAMES - 1,
                confidence=1.0,
                provenance=PROVENANCE_PERSISTENT_PREDICTION,
                current_component_index=i,
                last_component_voxel_count=int(
                    comp.get("voxel_count", len(comp["voxel_indices"]))
                ),
                last_real_observation_age_s_override=real_age,
            )
        )
    return tracks

@dataclass
class StaticVoxelState:
    class_id: int
    last_observed_frame: int
    observation_count: int


class StaticWorldMemory:
    """Sparse lidar-observed world-coordinate static occupancy memory."""

    def __init__(self, voxel_size_xyz: Sequence[float]):
        step = np.asarray(voxel_size_xyz, dtype=np.float64)
        if step.shape != (3,) or np.any(step <= 0):
            raise ValueError("voxel_size_xyz must be positive [3]")
        self.step = step
        self._voxels: dict[tuple[int, int, int], StaticVoxelState] = {}

    def _keys(self, points_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_world, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_world must be [N,3]")
        return np.floor(pts / self.step[None]).astype(np.int64)

    def update(
        self,
        semantics: np.ndarray,
        observed: np.ndarray,
        ego_to_world: np.ndarray,
        *,
        grid,
        free_label: int,
        frame_index: int,
    ) -> None:
        sem = np.asarray(semantics)
        obs = np.asarray(observed, dtype=bool)
        if sem.shape != tuple(grid.shape_hwd) or obs.shape != sem.shape:
            raise ValueError("static memory grid/observation shape mismatch")

        dynamic = np.isin(sem, np.asarray(_DYNAMIC_IDS, dtype=sem.dtype))
        usable = obs & ~dynamic
        idx = np.argwhere(usable)
        if len(idx) == 0:
            return
        pts_ego = _voxel_centers(idx, grid)
        pts_world = _transform_points(ego_to_world, pts_ego)
        keys = self._keys(pts_world)
        labels = sem[idx[:, 0], idx[:, 1], idx[:, 2]]

        for key_arr, lab in zip(keys, labels):
            key = tuple(int(x) for x in key_arr)
            lab_i = int(lab)
            if lab_i == int(free_label):
                self._voxels.pop(key, None)
                continue
            prev = self._voxels.get(key)
            self._voxels[key] = StaticVoxelState(
                class_id=lab_i,
                last_observed_frame=int(frame_index),
                observation_count=(
                    1 if prev is None else int(prev.observation_count) + 1
                ),
            )

    @classmethod
    def from_history(
        cls,
        history_semantics: Sequence[np.ndarray],
        history_observed: Sequence[np.ndarray],
        history_poses: Sequence[np.ndarray],
        *,
        grid,
        free_label: int,
    ) -> "StaticWorldMemory":
        if not (
            len(history_semantics)
            == len(history_observed)
            == len(history_poses)
            == HISTORY_FRAMES
        ):
            raise ValueError("static memory expects six semantic/obs/pose frames")
        mem = cls(grid.voxel_size)
        for t, (sem, obs, pose) in enumerate(
            zip(history_semantics, history_observed, history_poses)
        ):
            mem.update(
                sem,
                obs,
                pose,
                grid=grid,
                free_label=int(free_label),
                frame_index=t,
            )
        return mem

    def __len__(self) -> int:
        return len(self._voxels)

    def render(
        self,
        future_ego_to_world: np.ndarray,
        *,
        grid,
        free_label: int,
    ) -> np.ndarray:
        out = np.full(
            tuple(grid.shape_hwd), int(free_label), dtype=np.uint8
        )
        if not self._voxels:
            return out
        keys = np.asarray(list(self._voxels.keys()), dtype=np.int64)
        labels = np.asarray(
            [v.class_id for v in self._voxels.values()], dtype=np.uint8
        )
        pts_world = (keys.astype(np.float64) + 0.5) * self.step[None]
        world_to_future = np.linalg.inv(
            np.asarray(future_ego_to_world, dtype=np.float64)
        )
        pts = _transform_points(world_to_future, pts_world)
        origin = np.asarray(
            [grid.x_min, grid.y_min, grid.z_min], dtype=np.float64
        )
        step = np.asarray(grid.voxel_size, dtype=np.float64)
        idx = np.floor((pts - origin[None]) / step[None]).astype(np.int64)
        shape = np.asarray(grid.shape_hwd, dtype=np.int64)
        valid = np.all((idx >= 0) & (idx < shape[None]), axis=1)
        idx = idx[valid]
        labels = labels[valid]
        if len(idx):
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = labels
        return out


def protected_add_only(
    base: np.ndarray,
    proposal: np.ndarray,
    *,
    free_label: int,
    protected_mask: np.ndarray | None = None,
    confidence_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Add proposal occupancy only into currently free, unprotected cells."""
    out = np.asarray(base).copy()
    prop = np.asarray(proposal)
    if prop.shape != out.shape:
        raise ValueError("proposal/base shape mismatch")
    candidate = (prop != int(free_label)) & (out == int(free_label))
    if protected_mask is not None:
        protected = np.asarray(protected_mask, dtype=bool)
        if protected.shape != out.shape:
            raise ValueError("protected mask shape mismatch")
        candidate &= ~protected
    if confidence_mask is not None:
        conf = np.asarray(confidence_mask, dtype=bool)
        if conf.shape != out.shape:
            raise ValueError("confidence mask shape mismatch")
        candidate &= conf
    out[candidate] = prop[candidate]
    return out
