"""Corrected learned-motion contract for rigid source-shape transport.

v12 accidentally trained the motion head to align a Strong component centroid to
an annotation-box center.  That destroys the observed source-shape offset.  v13
instead predicts *object displacement residuals*:

    residual = (GT_future_center - GT_t0_center) - KTA_displacement

At deployment, the predicted displacement is applied to the observed Strong
source component itself, so the source shape keeps its original offset and the
GT-center oracle matches the v11 rigid-transport definition.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .motion_transport import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FUTURE_FRAMES,
    HISTORY_FRAMES,
    MotionTransportHead,
    annotation_map,
    backward_component_tracks,
    build_source_features,
    dynamic_annotations,
    match_sources_to_annotations,
    motion_transport_loss,
    trajectory_errors,
    world_points_to_t0,
    world_vec_to_t0,
)

MOTION_TRANSPORT_CACHE_VERSION = "p0_f9_motion_transport_v2"
TARGET_CONTRACT = "gt_displacement_minus_kta_displacement_preserve_source_offset_v2"
SOURCE_XY_FEATURE_SCALE_M = 40.0
VELOCITY_FEATURE_SCALE_MPS = 20.0


def displacement_residual_from_absolute_centers(
    source_component_xy_t0: np.ndarray,
    kta_anchor_xy_t0: np.ndarray,
    gt_t0_xy_t0: np.ndarray,
    gt_future_xy_t0: np.ndarray,
) -> np.ndarray:
    """Return the rigid-transport residual without aligning component/box centers.

    ``source_component_xy_t0`` is the observed Strong component centroid.
    ``kta_anchor_xy_t0`` is that centroid after causal constant-velocity KTA.
    The GT quantities are annotation centers and are used only to derive the
    *displacement* of the object between t0 and the future horizon.
    """
    source = np.asarray(source_component_xy_t0, dtype=np.float64)
    anchor = np.asarray(kta_anchor_xy_t0, dtype=np.float64)
    gt0 = np.asarray(gt_t0_xy_t0, dtype=np.float64)
    gtf = np.asarray(gt_future_xy_t0, dtype=np.float64)
    kta_disp = anchor - source
    gt_disp = gtf - gt0
    return (gt_disp - kta_disp).astype(np.float32)


def build_motion_targets(
    current_components: Sequence[Mapping],
    current_velocities_world: Mapping[int, np.ndarray],
    source_tokens: Sequence[str | None],
    t0_annotation_map: Mapping[str, Mapping],
    future_maps: Sequence[Mapping[str, Mapping]],
    t0_ego_to_world: np.ndarray,
    *,
    frame_dt_s: float = 0.5,
) -> dict[str, np.ndarray]:
    """Build v13 displacement-preserving KTA-residual supervision.

    GT boxes are supervision only.  They define how far the underlying object
    moved, never where the visible Strong component centroid should be forced to
    lie.  This is the key contract needed by rigid shape transport.
    """
    n = len(current_components)
    if len(source_tokens) != n or len(future_maps) != FUTURE_FRAMES:
        raise ValueError("target input length mismatch")

    source_xy = np.zeros((n, 2), dtype=np.float32)
    anchors = np.zeros((n, FUTURE_FRAMES, 2), dtype=np.float32)
    kta_disp = np.zeros((n, FUTURE_FRAMES, 2), dtype=np.float32)
    gt_t0_xy = np.zeros((n, 2), dtype=np.float32)
    target_xy = np.zeros((n, FUTURE_FRAMES, 2), dtype=np.float32)
    target_disp = np.zeros((n, FUTURE_FRAMES, 2), dtype=np.float32)
    residual = np.zeros((n, FUTURE_FRAMES, 2), dtype=np.float32)
    existence = np.zeros((n, FUTURE_FRAMES), dtype=np.float32)
    target_valid = np.zeros((n, FUTURE_FRAMES), dtype=bool)
    supervised = np.zeros(n, dtype=bool)

    for i, comp in enumerate(current_components):
        cur_world = np.asarray(comp["centroid_world"], dtype=np.float64)
        cur_t0 = world_points_to_t0(cur_world[None], t0_ego_to_world)[0, :2]
        source_xy[i] = cur_t0.astype(np.float32)
        v_world = np.asarray(current_velocities_world.get(i, np.zeros(3)), dtype=np.float64)
        v_t0 = world_vec_to_t0(v_world, t0_ego_to_world)[:2]

        token = source_tokens[i]
        ann0 = None if token is None else t0_annotation_map.get(str(token))
        if ann0 is not None:
            supervised[i] = True
            p0 = world_points_to_t0(
                np.asarray(ann0["center_world"], dtype=np.float64)[None], t0_ego_to_world
            )[0, :2]
            gt_t0_xy[i] = p0.astype(np.float32)

        for h in range(FUTURE_FRAMES):
            dt = (h + 1) * float(frame_dt_s)
            kd = v_t0 * dt
            kta_disp[i, h] = kd.astype(np.float32)
            anchor = cur_t0 + kd
            anchors[i, h] = anchor.astype(np.float32)
            if ann0 is None:
                continue
            annh = future_maps[h].get(str(token))
            if annh is None:
                continue
            pf = world_points_to_t0(
                np.asarray(annh["center_world"], dtype=np.float64)[None], t0_ego_to_world
            )[0, :2]
            gd = pf - p0
            target_xy[i, h] = pf.astype(np.float32)
            target_disp[i, h] = gd.astype(np.float32)
            residual[i, h] = (gd - kd).astype(np.float32)
            existence[i, h] = 1.0
            target_valid[i, h] = True

    return {
        "source_centroid_xy_t0_m": source_xy,
        "anchors_xy_t0_m": anchors,
        "kta_displacement_xy_m": kta_disp,
        "gt_t0_xy_t0_m": gt_t0_xy,
        "target_xy_t0_m": target_xy,
        "target_displacement_xy_m": target_disp,
        "target_residual_xy_m": residual,
        "existence": existence,
        "target_valid": target_valid,
        "supervised_source": supervised,
    }


def upgrade_v1_record_targets(
    record: Mapping,
    gt_t0_xy_t0_m: np.ndarray,
) -> dict:
    """Retarget an already-built v12 cache record without recomputing features.

    The expensive six-frame Strong tracking/features are reused verbatim.  The
    only rewritten tensors are target-definition tensors derived from existing
    absolute future GT centers plus t0 GT centers supplied by the upgrader.
    """
    features = record["features"].float().numpy()
    source_xy = features[:, :2].astype(np.float64) * SOURCE_XY_FEATURE_SCALE_M
    anchors = record["anchors_xy_t0_m"].float().numpy().astype(np.float64)
    future_xy = record["target_xy_t0_m"].float().numpy().astype(np.float64)
    valid = record["target_valid"].bool().numpy()
    gt0 = np.asarray(gt_t0_xy_t0_m, dtype=np.float64)
    if gt0.shape != source_xy.shape:
        raise ValueError("gt_t0_xy_t0_m shape mismatch")

    kta_disp = anchors - source_xy[:, None, :]
    target_disp = np.zeros_like(future_xy, dtype=np.float64)
    residual = np.zeros_like(future_xy, dtype=np.float64)
    for i in range(len(source_xy)):
        for h in range(FUTURE_FRAMES):
            if not valid[i, h]:
                continue
            target_disp[i, h] = future_xy[i, h] - gt0[i]
            residual[i, h] = target_disp[i, h] - kta_disp[i, h]

    out = dict(record)
    import torch
    out["source_centroid_xy_t0_m"] = torch.from_numpy(source_xy.astype(np.float32))
    out["kta_displacement_xy_m"] = torch.from_numpy(kta_disp.astype(np.float32))
    out["gt_t0_xy_t0_m"] = torch.from_numpy(gt0.astype(np.float32))
    out["target_displacement_xy_m"] = torch.from_numpy(target_disp.astype(np.float32))
    out["target_residual_xy_m"] = torch.from_numpy(residual.astype(np.float32))
    return out
