"""V20 Dormant-source preparation and synthetic-occlusion contract."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from .v19_scene_memory import (
    SourceTrack,
    build_dynamic_source_memory,
    prepare_causal_arrays_from_tracks,
)
from .strong_w2det import StrongW2DetConfig


@dataclass(frozen=True)
class SyntheticOcclusionResult:
    history_semantics: tuple[np.ndarray, ...]
    history_observed: tuple[np.ndarray, ...]
    dormant_track: SourceTrack
    dormant_inputs: dict[str, torch.Tensor]
    removed_voxels: int
    current_components_after: int


def _nearest_dormant_track(
    tracks: Sequence[SourceTrack],
    *,
    class_id: int,
    reference_center_world: np.ndarray,
    frame_dt_s: float,
) -> SourceTrack:
    candidates = [tr for tr in tracks if not tr.observed_at_anchor and int(tr.class_id) == int(class_id)]
    if not candidates:
        raise RuntimeError("synthetic occlusion did not produce a dormant source")
    ref = np.asarray(reference_center_world, dtype=np.float64)
    return min(
        candidates,
        key=lambda tr: float(
            np.linalg.norm(tr.anchor_center_world(float(frame_dt_s))[:2] - ref[:2])
        ),
    )


def recompute_synthetic_t0_occlusion(
    history_semantics: Sequence[np.ndarray],
    history_observed: Sequence[np.ndarray],
    history_poses: Sequence[np.ndarray],
    *,
    current_source_index: int,
    grid,
    free_label: int,
    frame_dt_s: float,
    strong_cfg: StrongW2DetConfig,
    max_missing_s: float = 1.5,
) -> SyntheticOcclusionResult:
    """Remove one real t0 source and rebuild the *entire* causal source path.

    This intentionally does not flip a visibility bit.  After removing the
    selected component's t0 voxels from semantic OCC and observation mask, it
    reruns:
      Strong extraction -> six-frame source tracking -> motion features/KTA ->
      local semantic tube -> source mask -> frame motion features.

    The resulting tensors are therefore suitable for Dormant training without
    leaking the original t0 source representation.
    """
    if len(history_semantics) != 6 or len(history_observed) != 6 or len(history_poses) != 6:
        raise ValueError("synthetic occlusion requires six history frames")
    sem = [np.asarray(x, dtype=np.uint8).copy() for x in history_semantics]
    obs = [np.asarray(x, dtype=bool).copy() for x in history_observed]
    if any(a.shape != b.shape for a, b in zip(sem, obs)):
        raise ValueError("semantic/observed shape mismatch")

    # V20 Dormant ancestry means genuinely observed source evidence. Unknown
    # semantic GT cells are not allowed to instantiate memory tracks.
    source_sem = [
        np.where(o, s, int(free_label)).astype(np.uint8)
        for s, o in zip(sem, obs)
    ]
    clean_tracks, clean_comps = build_dynamic_source_memory(
        source_sem,
        history_poses,
        grid=grid,
        strong_cfg=strong_cfg,
        frame_dt_s=float(frame_dt_s),
        max_missing_s=float(max_missing_s),
    )
    current = clean_comps[-1]
    si = int(current_source_index)
    if si < 0 or si >= len(current):
        raise IndexError("current_source_index outside t0 Strong sources")
    comp = current[si]
    vox = np.asarray(comp["voxel_indices"], dtype=np.int64)
    if vox.ndim != 2 or vox.shape[1] != 3 or len(vox) == 0:
        raise RuntimeError("selected t0 source has no voxel geometry")
    sem[-1][vox[:, 0], vox[:, 1], vox[:, 2]] = int(free_label)
    source_sem[-1][vox[:, 0], vox[:, 1], vox[:, 2]] = int(free_label)
    # Removing the observation itself is essential: observed-free would claim
    # evidence that the object is absent, which is not the intended occlusion.
    obs[-1][vox[:, 0], vox[:, 1], vox[:, 2]] = False

    rebuilt_tracks, rebuilt_comps = build_dynamic_source_memory(
        source_sem,
        history_poses,
        grid=grid,
        strong_cfg=strong_cfg,
        frame_dt_s=float(frame_dt_s),
        max_missing_s=float(max_missing_s),
    )
    dormant = _nearest_dormant_track(
        rebuilt_tracks,
        class_id=int(comp["class_id"]),
        reference_center_world=np.asarray(comp["centroid_world"], dtype=np.float64),
        frame_dt_s=float(frame_dt_s),
    )
    if dormant.observed_at_anchor or dormant.current_component_index is not None:
        raise RuntimeError("rebuilt synthetic target is not genuinely dormant")

    arrays = prepare_causal_arrays_from_tracks(
        [dormant],
        source_sem,
        history_poses,
        grid=grid,
        free_label=int(free_label),
        frame_dt_s=float(frame_dt_s),
    )
    return SyntheticOcclusionResult(
        history_semantics=tuple(sem),
        history_observed=tuple(obs),
        dormant_track=dormant,
        dormant_inputs=arrays,
        removed_voxels=int(len(vox)),
        current_components_after=int(len(rebuilt_comps[-1])),
    )


def split_current_and_dormant_tracks(
    tracks: Sequence[SourceTrack],
) -> tuple[tuple[SourceTrack, ...], tuple[SourceTrack, ...]]:
    """Keep detected t0 tracks on frozen V18; pack only truly dormant tracks."""
    current = tuple(tr for tr in tracks if tr.detected_at_anchor)
    dormant = tuple(tr for tr in tracks if not tr.detected_at_anchor)
    if any(not tr.observed_at_anchor and tr.current_component_index is not None for tr in dormant):
        raise RuntimeError("dormant track unexpectedly owns current component index")
    return current, dormant


def assert_current_source_predictions_unchanged(
    frozen_v18: dict[str, torch.Tensor],
    combined_current: dict[str, torch.Tensor],
) -> None:
    """Hard invariant: adding Dormant logic cannot change current-source V18."""
    for key in ("residual_xy_m", "existence_logits", "yaw_delta_rad"):
        if key not in frozen_v18 or key not in combined_current:
            raise KeyError(key)
        if not torch.equal(frozen_v18[key], combined_current[key]):
            n = int((frozen_v18[key] != combined_current[key]).sum().item())
            raise AssertionError(f"Dormant path changed current V18 {key}: {n} entries")


@dataclass(frozen=True)
class DormantRenderReport:
    future_semantic: np.ndarray
    tracks: int
    active_track_horizons: int
    rendered_voxels: int
    out_of_bounds_voxels: int
    collision_voxels: int


def _deduplicate_indices(idx: np.ndarray) -> np.ndarray:
    x = np.asarray(idx, dtype=np.int64)
    if x.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    order = np.lexsort((x[:, 2], x[:, 1], x[:, 0]))
    s = x[order]
    keep = np.ones(len(s), dtype=bool)
    keep[1:] = np.any(s[1:] != s[:-1], axis=1)
    return s[keep]


def render_dormant_sources(
    outputs: Mapping[str, torch.Tensor],
    tracks: Sequence[SourceTrack],
    *,
    kta_displacement_xy_m: torch.Tensor | np.ndarray,
    t0_ego_to_world: np.ndarray,
    future_ego_to_world: np.ndarray,
    grid,
    frame_dt_s: float,
    existence_threshold: float = 0.5,
    free_label: int = 17,
) -> DormantRenderReport:
    """Render learned Dormant trajectories with the frozen source geometry.

    The target convention exactly matches Stage-3 training:
      anchor_t0_xy + KTA displacement + learned residual.
    Yaw is a residual around each track's last observed source geometry.
    No GT identity, future semantic occupancy, or future observation mask is
    consumed here.
    """
    n = len(tracks)
    res = outputs["residual_xy_m"].detach().float().cpu().numpy()
    yaw = outputs["yaw_delta_rad"].detach().float().cpu().numpy()
    exist = torch.sigmoid(
        outputs["existence_logits"].detach().float()
    ).cpu().numpy()
    if res.shape != (n, 6, 2) or yaw.shape != (n, 6) or exist.shape != (n, 6):
        raise ValueError("Dormant output/track count mismatch")
    kta = np.asarray(
        kta_displacement_xy_m.detach().cpu().numpy()
        if isinstance(kta_displacement_xy_m, torch.Tensor)
        else kta_displacement_xy_m,
        dtype=np.float64,
    )
    if kta.shape != (n, 6, 2):
        raise ValueError("kta_displacement_xy_m must be [N,6,2]")
    t0 = np.asarray(t0_ego_to_world, dtype=np.float64)
    futures = np.asarray(future_ego_to_world, dtype=np.float64)
    if t0.shape != (4, 4) or futures.shape != (6, 4, 4):
        raise ValueError("Dormant render poses must be t0[4,4], future[6,4,4]")
    inv_t0 = np.linalg.inv(t0)
    inv_future = np.stack([np.linalg.inv(x) for x in futures], axis=0)
    origin = np.asarray(
        [grid.x_min, grid.y_min, grid.z_min], dtype=np.float64
    )
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    shape = np.asarray(grid.shape_hwd, dtype=np.int64)
    out = np.full((6,) + tuple(shape.tolist()), int(free_label), dtype=np.uint8)

    anchors_t0 = []
    for tr in tracks:
        aw = tr.anchor_center_world(float(frame_dt_s))
        anchors_t0.append((inv_t0 @ np.r_[aw, 1.0])[:3])
    anchors_t0 = (
        np.asarray(anchors_t0, dtype=np.float64)
        if anchors_t0 else np.zeros((0, 3), dtype=np.float64)
    )

    order = sorted(
        range(n),
        key=lambda i: (-float(tracks[i].confidence), int(tracks[i].track_id)),
    )
    active_h = rendered = oob = collisions = 0
    for i in order:
        tr = tracks[i]
        local = np.asarray(tr.canonical_xyz_local, dtype=np.float64)
        if local.ndim != 2 or local.shape[1] != 3:
            raise ValueError("Dormant track canonical geometry must be [N,3]")
        if len(local) == 0:
            continue
        for h in range(6):
            if float(exist[i, h]) < float(existence_threshold):
                continue
            active_h += 1
            target_t0 = anchors_t0[i].copy()
            target_t0[:2] += kta[i, h] + res[i, h]
            target_world = (t0 @ np.r_[target_t0, 1.0])[:3]
            theta = float(yaw[i, h])
            c, s = math.cos(theta), math.sin(theta)
            moved_local = local.copy()
            moved_local[:, 0] = c * local[:, 0] - s * local[:, 1]
            moved_local[:, 1] = s * local[:, 0] + c * local[:, 1]
            world = moved_local + target_world[None]
            ego = (
                world @ inv_future[h, :3, :3].T
                + inv_future[h, :3, 3][None]
            )
            idx = np.floor((ego - origin[None]) / step[None]).astype(np.int64)
            valid = ((idx >= 0) & (idx < shape[None])).all(axis=1)
            oob += int((~valid).sum())
            idx = _deduplicate_indices(idx[valid])
            if len(idx) == 0:
                continue
            free = out[h, idx[:, 0], idx[:, 1], idx[:, 2]] == int(free_label)
            collisions += int((~free).sum())
            good = idx[free]
            if len(good):
                out[h, good[:, 0], good[:, 1], good[:, 2]] = int(tr.class_id)
                rendered += int(len(good))
    return DormantRenderReport(
        future_semantic=out,
        tracks=int(n),
        active_track_horizons=int(active_h),
        rendered_voxels=int(rendered),
        out_of_bounds_voxels=int(oob),
        collision_voxels=int(collisions),
    )
