"""V20 Dormant-source preparation and synthetic-occlusion contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
