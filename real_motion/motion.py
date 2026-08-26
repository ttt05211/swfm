"""Causal real-motion decomposition utilities.

The final method must never use future GT to build these masks.

Important contract
------------------
``STATIC`` is a motion *state for occupied voxels*. Free space is not a
confident-static occupancy and must never be protected by composition.  The
high-level :func:`decompose_masks` API returns explicit masks so callers cannot
accidentally write ``state == STATIC`` and freeze free space.
"""
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

STATIC = np.uint8(0)
MOVING = np.uint8(1)
UNCERTAIN = np.uint8(2)
FREE = np.uint8(3)


@dataclass(frozen=True)
class PersistenceMotionConfig:
    free_label: int = 17
    static_min_persistence: float = 0.80
    moving_max_persistence: float = 0.50
    min_observed_frames: int = 2
    # Component-track hysteresis prevents one moving object from being split
    # into static interior + moving boundary voxels by pure persistence.
    use_component_tracks: bool = True
    history_dt_s: float = 0.5
    voxel_size_xy_m: tuple = (0.4, 0.4)
    component_max_step_m: float = 4.0
    moving_speed_mps: float = 0.5
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


def decompose_ego_aligned_history(
    history_semantics: np.ndarray,
    cfg: PersistenceMotionConfig = PersistenceMotionConfig(),
):
    """Low-level state map for an ego-aligned history.

    Args:
        history_semantics: ``[T,H,W,D]`` integer semantic labels. Frames must
            already be rigidly transformed into the last frame's ego grid.

    Returns:
        state: ``[H,W,D]`` with STATIC/MOVING/UNCERTAIN/FREE.
        occupied: current-frame occupied mask.
        persistence: fraction of observed history carrying the current label.

    Notes:
        This is intentionally a simple occupancy-only causal baseline. A
        stronger tracker/KTA detector can replace it while keeping the explicit
        mask contract from :func:`decompose_masks`.
    """
    x = np.asarray(history_semantics)
    if x.ndim != 4 or x.shape[0] < cfg.min_observed_frames:
        raise ValueError("history_semantics must be [T,H,W,D] with enough frames")

    current = x[-1]
    occupied = current != cfg.free_label
    same = x == current[None, ...]

    # A historical free cell is not evidence that the current occupied voxel
    # persisted. Current occupancy is included in every denominator position.
    observed = (x != cfg.free_label) | occupied[None, ...]
    denom = np.maximum(observed.sum(axis=0), 1)
    persistence = (same & observed).sum(axis=0) / denom

    state = np.full(current.shape, FREE, dtype=np.uint8)
    occ_static = occupied & (persistence >= cfg.static_min_persistence)
    occ_moving = occupied & (persistence <= cfg.moving_max_persistence)
    occ_uncertain = occupied & ~(occ_static | occ_moving)
    state[occ_static] = STATIC
    state[occ_moving] = MOVING
    state[occ_uncertain] = UNCERTAIN
    return state, occupied, persistence.astype(np.float32)


def _components_8(mask_hw: np.ndarray):
    m=np.asarray(mask_hw,dtype=bool); H,W=m.shape
    seen=np.zeros_like(m); out=[]
    for y0,x0 in np.argwhere(m):
        if seen[y0,x0]: continue
        stack=[(int(y0),int(x0))]; seen[y0,x0]=True; cells=[]
        while stack:
            y,x=stack.pop(); cells.append((y,x))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy==0 and dx==0: continue
                    yy,xx=y+dy,x+dx
                    if 0<=yy<H and 0<=xx<W and m[yy,xx] and not seen[yy,xx]:
                        seen[yy,xx]=True; stack.append((yy,xx))
        out.append(np.asarray(cells,dtype=np.int64))
    return out


def _frame_components(sem: np.ndarray, free_label: int, min_cells: int):
    comps={}
    for cls in np.unique(sem):
        cls=int(cls)
        if cls==free_label: continue
        bev=(sem==cls).any(axis=2)
        rows=[]
        for cells in _components_8(bev):
            if len(cells)<min_cells: continue
            rows.append(cells)
        comps[cls]=rows
    return comps


def _centroid_metric(cells, cfg: PersistenceMotionConfig):
    vx,vy=cfg.voxel_size_xy_m
    # Absolute grid origin cancels in displacement; x is column, y is row.
    return np.array([(cells[:,1].mean()+0.5)*vx,
                     (cells[:,0].mean()+0.5)*vy],dtype=np.float64)


def _component_track_state(history_semantics, persistence, cfg: PersistenceMotionConfig):
    """Classify complete current semantic components from causal centroid tracks."""
    hist=np.asarray(history_semantics); cur=hist[-1]
    frame_comps=[_frame_components(x,cfg.free_label,cfg.min_component_bev_cells) for x in hist]
    state=np.full(cur.shape,FREE,dtype=np.uint8)
    for cls,current_list in frame_comps[-1].items():
        for cells in current_list:
            cur_c=_centroid_metric(cells,cfg); ref=cur_c; track=[cur_c]
            # Greedy backward association, one physical step at a time. No
            # future frame or GT instance token is used.
            for ti in range(len(hist)-2,-1,-1):
                candidates=frame_comps[ti].get(cls,[])
                if not candidates: break
                centroids=[_centroid_metric(c,cfg) for c in candidates]
                d=np.asarray([np.linalg.norm(c-ref) for c in centroids])
                j=int(d.argmin())
                if float(d[j])>cfg.component_max_step_m: break
                ref=centroids[j]; track.append(ref)
            intervals=len(track)-1
            speed=float('nan')
            if intervals>0:
                speed=float(np.linalg.norm(track[0]-track[-1])/(intervals*cfg.history_dt_s))

            cell_mask=np.zeros(cur.shape[:2],dtype=bool)
            cell_mask[cells[:,0],cells[:,1]]=True
            vox=(cur==cls)&cell_mask[:,:,None]
            mean_p=float(persistence[vox].mean()) if vox.any() else 0.0
            if len(track)>=cfg.min_track_frames and speed>=cfg.moving_speed_mps:
                s=MOVING
            elif (len(track)>=cfg.min_track_frames and speed<=cfg.static_speed_mps
                  and mean_p>=cfg.static_min_persistence):
                s=STATIC
            else:
                s=UNCERTAIN
            state[vox]=s
    return state


def decompose_masks(
    history_semantics: np.ndarray,
    cfg: PersistenceMotionConfig = PersistenceMotionConfig(),
) -> MotionMasks:
    """Return the safe, explicit four-way decomposition contract.

    ``confident_static`` is *always* intersected with current occupancy. This is
    the public API cache builders and composition code should consume.
    """
    state, occupied, persistence = decompose_ego_aligned_history(history_semantics, cfg)
    if cfg.use_component_tracks:
        state = _component_track_state(history_semantics, persistence, cfg)
        state[~occupied] = FREE
    confident_static = occupied & (state == STATIC)
    moving = occupied & (state == MOVING)
    uncertain = occupied & (state == UNCERTAIN)
    free = ~occupied

    # Partition invariant over the current grid.
    total = confident_static.astype(np.uint8) + moving.astype(np.uint8) + uncertain.astype(np.uint8) + free.astype(np.uint8)
    if not np.all(total == 1):
        raise RuntimeError("motion masks do not form an exact partition")
    return MotionMasks(confident_static, moving, uncertain, free, persistence)


def split_semantics(current_semantics: np.ndarray, masks: MotionMasks, free_label: int = 17):
    """Split one semantic occupancy grid into static and WM-candidate grids."""
    cur = np.asarray(current_semantics)
    if cur.shape != masks.confident_static.shape:
        raise ValueError("current_semantics and motion masks must align")
    static = np.full_like(cur, free_label)
    candidate = np.full_like(cur, free_label)
    static[masks.confident_static] = cur[masks.confident_static]
    candidate[masks.wm_candidate] = cur[masks.wm_candidate]
    return static, candidate
