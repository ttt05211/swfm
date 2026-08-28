"""Causal real-motion decomposition utilities.

The final method must never use future GT to build these masks.

Motion contract
---------------
``STATIC`` is the deterministic background route. Non-motion-eligible occupied
classes are transported by ego motion and are never promoted to MOVING by
component jitter.

``MOVING`` is stricter: it requires explicit historical displacement of a
motion-eligible thing component.

A motion-eligible thing that is not explicitly moving is ``UNCERTAIN`` rather
than hard-static. This preserves generator write access for parked-to-moving or
otherwise future-ambiguous objects. Low persistence alone is never evidence of
motion.

Occ3D ``mask_lidar`` is still used to compute observation-conditioned
persistence as a diagnostic/stationarity signal, but semantic route eligibility
controls the hard-static safety boundary. Free space is never protected.
Occ3D semantic arrays follow the official ``[X,Y,Z]`` axis order.
"""
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

STATIC = np.uint8(0)
MOVING = np.uint8(1)
UNCERTAIN = np.uint8(2)
FREE = np.uint8(3)

# nuScenes/Occ3D thing classes for which object/component displacement is
# physically meaningful. Semantic identity only decides whether to *test*
# motion; MOVING still requires measured historical displacement.
DEFAULT_MOTION_ELIGIBLE_CLASS_IDS = (2, 3, 4, 5, 6, 7, 9, 10)


@dataclass(frozen=True)
class PersistenceMotionConfig:
    free_label: int = 17
    static_min_persistence: float = 0.80
    # Retained for backward-compatible config loading/audits. The formal route
    # no longer maps low persistence directly to MOVING.
    moving_max_persistence: float = 0.50
    min_observed_frames: int = 2
    min_static_observations: int = 3
    motion_eligible_class_ids: tuple = DEFAULT_MOTION_ELIGIBLE_CLASS_IDS
    use_component_tracks: bool = True
    history_dt_s: float = 0.5
    voxel_size_xy_m: tuple = (0.4, 0.4)
    component_max_step_m: float = 4.0
    moving_speed_mps: float = 0.5
    # Retained for compatibility/diagnostics; formal component tracking only
    # promotes explicit MOVING and never declares STATIC by centroid speed.
    static_speed_mps: float = 0.2
    min_track_frames: int = 2
    min_component_bev_cells: int = 1


class MotionMasks(NamedTuple):
    confident_static: np.ndarray
    moving: np.ndarray
    uncertain: np.ndarray
    free: np.ndarray
    persistence: np.ndarray

    @property
    def wm_candidate(self) -> np.ndarray:
        """Generator-write occupancy: explicit MOVING plus eligible UNCERTAIN."""
        return self.moving | self.uncertain


def _validate_observation(history_semantics, history_observed):
    x = np.asarray(history_semantics)
    if history_observed is None:
        # Synthetic/unit-test fallback. Formal nuScenes preparation passes
        # Occ3D mask_lidar explicitly and therefore never infers observation
        # from semantic free/non-free labels.
        return x != 17
    obs = np.asarray(history_observed, dtype=bool)
    if obs.shape != x.shape:
        raise ValueError(
            f"history_observed shape {obs.shape} must match semantics {x.shape}"
        )
    return obs


def _motion_eligible_mask(current_semantics, cfg: PersistenceMotionConfig):
    return np.isin(
        np.asarray(current_semantics),
        np.asarray(tuple(int(c) for c in cfg.motion_eligible_class_ids), dtype=np.int64),
    )


def decompose_ego_aligned_history(
    history_semantics: np.ndarray,
    cfg: PersistenceMotionConfig = PersistenceMotionConfig(),
    history_observed: np.ndarray | None = None,
):
    """Compute observation-conditioned same-class persistence.

    Args:
        history_semantics: ``[T,X,Y,Z]`` integer semantic labels, already
            transformed into the current frame's occupancy grid.
        history_observed: aligned ``[T,X,Y,Z]`` observation mask. Formal Occ3D
            preparation supplies ``mask_lidar``; an unobserved voxel is neither
            positive nor negative evidence for persistence.

    Returns:
        state: preliminary ``[X,Y,Z]`` persistence state. The final hard-static
            versus uncertain routing is applied in :func:`decompose_masks`.
        occupied: current-frame semantic occupancy mask.
        persistence: same-class fraction over genuinely observed frames.
    """
    x = np.asarray(history_semantics)
    if x.ndim != 4 or x.shape[0] < cfg.min_observed_frames:
        raise ValueError("history_semantics must be [T,X,Y,Z] with enough frames")
    if cfg.min_static_observations < 1:
        raise ValueError("min_static_observations must be >= 1")

    observed = _validate_observation(x, history_observed)
    current = x[-1]
    occupied = current != cfg.free_label
    same = x == current[None, ...]

    observed_count = observed.sum(axis=0)
    denom = np.maximum(observed_count, 1)
    persistence = (same & observed).sum(axis=0) / denom

    enough_static_evidence = observed_count >= int(cfg.min_static_observations)
    occ_static = occupied & enough_static_evidence & (
        persistence >= cfg.static_min_persistence
    )
    occ_uncertain = occupied & ~occ_static

    state = np.full(current.shape, FREE, dtype=np.uint8)
    state[occ_static] = STATIC
    state[occ_uncertain] = UNCERTAIN
    return state, occupied, persistence.astype(np.float32)


def _components_8(mask_xy: np.ndarray):
    m = np.asarray(mask_xy, dtype=bool)
    X, Y = m.shape
    seen = np.zeros_like(m)
    out = []
    for x0, y0 in np.argwhere(m):
        if seen[x0, y0]:
            continue
        stack = [(int(x0), int(y0))]
        seen[x0, y0] = True
        cells = []
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx, yy = x + dx, y + dy
                    if (
                        0 <= xx < X and 0 <= yy < Y and m[xx, yy]
                        and not seen[xx, yy]
                    ):
                        seen[xx, yy] = True
                        stack.append((xx, yy))
        out.append(np.asarray(cells, dtype=np.int64))
    return out


def _frame_components(
    sem: np.ndarray,
    free_label: int,
    min_cells: int,
    observed: np.ndarray | None = None,
    eligible_classes=None,
):
    comps = {}
    allowed = None if eligible_classes is None else {int(c) for c in eligible_classes}
    obs = None if observed is None else np.asarray(observed, dtype=bool)
    for cls in np.unique(sem):
        cls = int(cls)
        if cls == free_label or (allowed is not None and cls not in allowed):
            continue
        vox = sem == cls
        if obs is not None:
            vox &= obs
        bev = vox.any(axis=2)
        rows = []
        for cells in _components_8(bev):
            if len(cells) < min_cells:
                continue
            rows.append(cells)
        comps[cls] = rows
    return comps


def _centroid_metric(cells, cfg: PersistenceMotionConfig):
    vx, vy = cfg.voxel_size_xy_m
    # Official Occ3D BEV indices are [x_index, y_index]. Absolute origin
    # cancels in displacement, so only axis order and voxel scale matter here.
    return np.array([
        (cells[:, 0].mean() + 0.5) * vx,
        (cells[:, 1].mean() + 0.5) * vy,
    ], dtype=np.float64)


def _promote_explicit_component_motion(
    base_state,
    history_semantics,
    history_observed,
    cfg: PersistenceMotionConfig,
):
    """Promote only motion-eligible components with explicit displacement."""
    hist = np.asarray(history_semantics)
    obs = _validate_observation(hist, history_observed)
    cur = hist[-1]
    eligible = tuple(int(c) for c in cfg.motion_eligible_class_ids)
    frame_comps = [
        _frame_components(
            sem,
            cfg.free_label,
            cfg.min_component_bev_cells,
            observed=ob,
            eligible_classes=eligible,
        )
        for sem, ob in zip(hist, obs)
    ]

    state = np.asarray(base_state).copy()
    for cls, current_list in frame_comps[-1].items():
        for cells in current_list:
            cur_c = _centroid_metric(cells, cfg)
            ref = cur_c
            track = [cur_c]
            for ti in range(len(hist) - 2, -1, -1):
                candidates = frame_comps[ti].get(cls, [])
                if not candidates:
                    break
                centroids = [_centroid_metric(c, cfg) for c in candidates]
                d = np.asarray([np.linalg.norm(c - ref) for c in centroids])
                j = int(d.argmin())
                if float(d[j]) > cfg.component_max_step_m:
                    break
                ref = centroids[j]
                track.append(ref)

            intervals = len(track) - 1
            if len(track) < cfg.min_track_frames or intervals <= 0:
                continue
            speed = float(
                np.linalg.norm(track[0] - track[-1])
                / (intervals * cfg.history_dt_s)
            )
            if speed < cfg.moving_speed_mps:
                continue

            cell_mask = np.zeros(cur.shape[:2], dtype=bool)
            cell_mask[cells[:, 0], cells[:, 1]] = True
            vox = (cur == cls) & cell_mask[:, :, None]
            state[vox] = MOVING
    return state


def decompose_masks(
    history_semantics: np.ndarray,
    cfg: PersistenceMotionConfig = PersistenceMotionConfig(),
    history_observed: np.ndarray | None = None,
) -> MotionMasks:
    """Return the formal deterministic/generative routing partition.

    Final routing deliberately separates *current physical motion* from *safe
    deterministic transport*:

    - non-motion-eligible occupied background -> ``confident_static``;
    - eligible component with explicit historical displacement -> ``moving``;
    - every other eligible occupied thing -> ``uncertain``;
    - free -> ``free``.

    Thus a parked car is not called MOVING, but it is also never hard-locked as
    STATIC; the generator retains permission to predict a future start/turn.
    """
    hist = np.asarray(history_semantics)
    state, occupied, persistence = decompose_ego_aligned_history(
        hist, cfg, history_observed=history_observed
    )
    if cfg.use_component_tracks:
        state = _promote_explicit_component_motion(
            state, hist, history_observed, cfg
        )
        state[~occupied] = FREE

    current = hist[-1]
    eligible = occupied & _motion_eligible_mask(current, cfg)
    moving = eligible & (state == MOVING)
    uncertain = eligible & ~moving
    confident_static = occupied & ~eligible
    free = ~occupied

    total = (
        confident_static.astype(np.uint8)
        + moving.astype(np.uint8)
        + uncertain.astype(np.uint8)
        + free.astype(np.uint8)
    )
    if not np.all(total == 1):
        raise RuntimeError("motion masks do not form an exact partition")
    return MotionMasks(confident_static, moving, uncertain, free, persistence)


def split_semantics(
    current_semantics: np.ndarray,
    masks: MotionMasks,
    free_label: int = 17,
):
    """Split one semantic occupancy grid into static and WM-candidate grids."""
    cur = np.asarray(current_semantics)
    if cur.shape != masks.confident_static.shape:
        raise ValueError("current_semantics and motion masks must align")
    static = np.full_like(cur, free_label)
    candidate = np.full_like(cur, free_label)
    static[masks.confident_static] = cur[masks.confident_static]
    candidate[masks.wm_candidate] = cur[masks.wm_candidate]
    return static, candidate
