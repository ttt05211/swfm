"""Causal real-motion decomposition utilities.

The final method must never use future GT to build these masks.

Motion contract
---------------
``STATIC`` means there is sufficient *observed* history supporting persistence.
``MOVING`` is stronger: it requires explicit historical displacement of a
motion-eligible thing component.  Low persistence alone is never evidence of
motion.  Everything else that is currently occupied is ``UNCERTAIN``.

Free space is not a confident-static occupancy and must never be protected by
composition.  The high-level :func:`decompose_masks` API returns explicit masks
so callers cannot accidentally freeze free space.
"""
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

STATIC = np.uint8(0)
MOVING = np.uint8(1)
UNCERTAIN = np.uint8(2)
FREE = np.uint8(3)

# nuScenes/Occ3D thing classes for which object/component displacement is
# physically meaningful.  Semantic identity only decides whether to *test*
# motion; MOVING still requires measured historical displacement.
DEFAULT_MOTION_ELIGIBLE_CLASS_IDS = (2, 3, 4, 5, 6, 7, 9, 10)


@dataclass(frozen=True)
class PersistenceMotionConfig:
    free_label: int = 17
    static_min_persistence: float = 0.80
    # Kept for backward-compatible config loading/audits.  The formal detector
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
        """Observed-moving plus motion-uncertain occupancy."""
        return self.moving | self.uncertain


def _validate_observation(history_semantics, history_observed):
    x = np.asarray(history_semantics)
    if history_observed is None:
        # Synthetic/unit-test fallback.  Formal nuScenes preparation passes
        # Occ3D mask_lidar explicitly and therefore never infers observation
        # from semantic free/non-free labels.
        return x != 17
    obs = np.asarray(history_observed, dtype=bool)
    if obs.shape != x.shape:
        raise ValueError(
            f"history_observed shape {obs.shape} must match semantics {x.shape}"
        )
    return obs


def decompose_ego_aligned_history(
    history_semantics: np.ndarray,
    cfg: PersistenceMotionConfig = PersistenceMotionConfig(),
    history_observed: np.ndarray | None = None,
):
    """Low-level observation-conditioned persistence state for aligned history.

    Args:
        history_semantics: ``[T,H,W,D]`` integer semantic labels, already
            transformed into the current frame's occupancy grid.
        history_observed: aligned ``[T,H,W,D]`` observation mask.  Formal Occ3D
            preparation supplies ``mask_lidar``; an unobserved voxel is neither
            positive nor negative evidence for persistence.

    Returns:
        state: ``[H,W,D]`` with STATIC/UNCERTAIN/FREE.  MOVING is introduced
            only later by explicit component displacement.
        occupied: current-frame semantic occupancy mask.
        persistence: same-class fraction over genuinely observed frames.
    """
    x = np.asarray(history_semantics)
    if x.ndim != 4 or x.shape[0] < cfg.min_observed_frames:
        raise ValueError("history_semantics must be [T,H,W,D] with enough frames")
    if cfg.min_static_observations < 1:
        raise ValueError("min_static_observations must be >= 1")

    observed = _validate_observation(x, history_observed)
    current = x[-1]
    occupied = current != cfg.free_label
    same = x == current[None, ...]

    observed_count = observed.sum(axis=0)
    denom = np.maximum(observed_count, 1)
    persistence = (same & observed).sum(axis=0) / denom

    # Low persistence means insufficient stationary evidence, not motion.
    enough_static_evidence = observed_count >= int(cfg.min_static_observations)
    occ_static = occupied & enough_static_evidence & (
        persistence >= cfg.static_min_persistence
    )
    occ_uncertain = occupied & ~occ_static

    state = np.full(current.shape, FREE, dtype=np.uint8)
    state[occ_static] = STATIC
    state[occ_uncertain] = UNCERTAIN
    return state, occupied, persistence.astype(np.float32)


def _components_8(mask_hw: np.ndarray):
    m = np.asarray(mask_hw, dtype=bool)
    H, W = m.shape
    seen = np.zeros_like(m)
    out = []
    for y0, x0 in np.argwhere(m):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if (
                        0 <= yy < H and 0 <= xx < W and m[yy, xx]
                        and not seen[yy, xx]
                    ):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
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
    # Absolute grid origin cancels in displacement; x is column, y is row.
    return np.array([
        (cells[:, 1].mean() + 0.5) * vx,
        (cells[:, 0].mean() + 0.5) * vy,
    ], dtype=np.float64)


def _promote_explicit_component_motion(
    base_state,
    history_semantics,
    history_observed,
    cfg: PersistenceMotionConfig,
):
    """Promote only motion-eligible components with explicit displacement.

    Stuff/background classes are never turned into MOVING by connected-component
    centroid jitter.  For eligible thing classes, semantic identity only gates
    whether tracking is attempted; the state transition to MOVING still requires
    a causal, matched historical displacement above ``moving_speed_mps``.
    """
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
            # Greedy backward association, one physical step at a time.  No
            # future frame or GT instance token is used.
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
    """Return the safe, explicit four-way decomposition contract.

    ``confident_static`` requires enough genuinely observed history with high
    same-class persistence. ``moving`` requires explicit displacement of a
    motion-eligible thing. All other current occupancy is ``uncertain``.
    """
    state, occupied, persistence = decompose_ego_aligned_history(
        history_semantics, cfg, history_observed=history_observed
    )
    if cfg.use_component_tracks:
        state = _promote_explicit_component_motion(
            state, history_semantics, history_observed, cfg
        )
        state[~occupied] = FREE

    confident_static = occupied & (state == STATIC)
    moving = occupied & (state == MOVING)
    uncertain = occupied & (state == UNCERTAIN)
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
