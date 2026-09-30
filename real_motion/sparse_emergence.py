"""Causal, bounded point-set emergence. GT belongs only to supervision/audit.

History alignment is an INPUT resampler, not a replacement V18 renderer.
New point positions/classes are decoded; no history shape/prototype is copied.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import affine_transform, distance_transform_edt

PROTOCOL = "p0_f9_sparse_emergence_screen_v1"
FEATURE_PROTOCOL = "six_past_full_occ_future_ego_local_columns_v1"
REPORT = (1, 3, 5)
THRESHOLDS = (.05, .10, .20, .40, .60, .80, .95)


@dataclass(frozen=True)
class EmergenceConfig:
    patch_cells: int = 4
    context_cells: int = 8
    max_patches: int = 64
    queries: int = 4
    points_per_query: int = 16
    target_points: int = 256
    width: int = 64
    heads: int = 4
    layers: int = 2
    evidence_radius_m: float = 3.2
    refinement: bool = False

    def validate(self):
        if (min(self.patch_cells, self.context_cells, self.max_patches, self.queries,
                self.points_per_query, self.target_points, self.width, self.heads, self.layers) < 1
                or self.context_cells < self.patch_cells or self.context_cells % 2
                or (self.context_cells - self.patch_cells) % 2 or self.width % self.heads
                or self.queries != 4 or not np.isfinite(self.evidence_radius_m) or self.evidence_radius_m <= 0):
            raise ValueError("invalid emergence architecture/support budget")


def history_in_query_frame(history, history_poses, future_pose, grid):
    """Inverse nearest-centre sampling; unknown=18, known-free=17.

    Uses ALL supplied <=t0 semantic occupancy, as V18 does. This does not
    silently change the observation protocol to mask_lidar-filtered semantics.
    Future ego pose is the existing trajectory-conditioning input, not future
    semantic evidence. Nothing here accepts GT future occupancy.
    """
    h = np.asarray(history)
    if h.shape != (6, *grid.shape_hwd) or len(history_poses) != 6 or h.min() < 0 or h.max() > 17:
        raise ValueError("six-history semantic contract mismatch")
    step = np.asarray(grid.voxel_size, np.float64)
    origin = np.asarray((grid.x_min, grid.y_min, grid.z_min), np.float64)
    frames = []
    for sem, pose in zip(h, history_poses):
        t = np.linalg.inv(np.asarray(pose, np.float64)) @ np.asarray(future_pose, np.float64)
        mat = t[:3, :3] * step[None, :] / step[:, None]
        off = (t[:3, :3] @ (origin + .5*step) + t[:3, 3] - origin) / step - .5
        frames.append(affine_transform(sem, mat, off, output_shape=grid.shape_hwd,
                      order=0, mode="constant", cval=18, prefilter=False).astype(np.uint8))
    return np.stack(frames)


@dataclass
class EmergenceInputs:
    history: np.ndarray  # [N,6,context,context,Z], uint8
    query: np.ndarray  # [N,24], float32; base semantic histogram + pose/time/XY
    patch_xy: np.ndarray  # [N,2], integer lower corners, future ego grid
    t0_labels: np.ndarray
    history_static_seen: np.ndarray


def prepare_inputs(aligned_history, baseline, grid, horizon_s, relative_pose, config=EmergenceConfig()):
    """Deterministic GT-free support: near historical occupied evidence OR entry.

    Spatially stratified cap, never a GT/top-positive shortlist. The same
    candidate budget is used in training, calibration and validation.
    """
    config.validate()
    hist, base = np.asarray(aligned_history), np.asarray(baseline)
    shape = tuple(grid.shape_hwd)
    if hist.shape != (6, *shape) or base.shape != shape or hist.min() < 0 or hist.max() > 18:
        raise ValueError("aligned history/base shape or label mismatch")
    if base.min() < 0 or base.max() > 17 or not 0 < horizon_s <= 3:
        raise ValueError("invalid frozen query")
    p, c = config.patch_cells, config.context_cells
    if shape[0] % p or shape[1] % p:
        raise ValueError("grid XY must be divisible by patch_cells; no silent edge truncation")
    occupied = (hist < 17).any(axis=(0, 3))
    near = distance_transform_edt(~occupied, sampling=grid.voxel_size[:2]) <= config.evidence_radius_m if occupied.any() else np.zeros(shape[:2], bool)
    entry = (hist[-1] == 18).all(axis=2)
    support = (near | entry) & (base == 17).any(axis=2)
    nx, ny = shape[0]//p, shape[1]//p
    active = np.flatnonzero(support.reshape(nx, p, ny, p).any(axis=(1, 3)))
    if len(active) > config.max_patches:
        # Uniform fixed positions through spatial order, not a class/GT rank.
        active = active[np.linspace(0, len(active)-1, config.max_patches, dtype=int)]
    xy = np.column_stack((active//ny*p, active % ny*p)).astype(np.int16)
    pad = (c-p)//2
    padded = np.pad(hist, ((0,0), (pad,pad), (pad,pad), (0,0)), constant_values=18)
    tubes = np.stack([padded[:, x:x+c, y:y+c] for x, y in xy]) if len(xy) else np.empty((0,6,c,c,shape[2]), np.uint8)
    q = np.zeros((len(xy), 24), np.float32)
    rel = np.asarray(relative_pose, np.float64)
    if rel.shape != (4,4) or not np.isfinite(rel).all(): raise ValueError("relative query pose invalid")
    for i, (x,y) in enumerate(xy):
        block = base[x:x+p, y:y+p]
        q[i,:18] = np.bincount(block.ravel(), minlength=18)/block.size
    q[:,18:20] = (xy + p/2)/np.asarray(shape[:2])*2-1
    q[:,20] = horizon_s/3
    q[:,21:24] = (rel[0,3]/40, rel[1,3]/40, np.arctan2(rel[1,0], rel[0,0])/np.pi)
    static_ids = np.asarray((0,1,8,11,12,13,14,15,16))
    seen = np.zeros(shape, bool)
    for frame in hist:
        seen |= np.isin(frame, static_ids)
    return EmergenceInputs(tubes.astype(np.uint8), q, xy, hist[-1].copy(), seen)


def target_sets(inputs, baseline, gt, config=EmergenceConfig()):
    """Only residual occupied GT in the SAME causal patches; padding is masked.

    Class-balanced deterministic sampling prevents a large ground surface
    from exhausting the geometric target budget. No sampled target becomes
    an input or a test-time candidate/anchor.
    """
    b, g = np.asarray(baseline), np.asarray(gt)
    if b.shape != g.shape or g.min() < 0 or g.max() > 17: raise ValueError("target grid mismatch")
    n, p, z, k = len(inputs.patch_xy), config.patch_cells, b.shape[2], config.target_points
    xyz = np.zeros((n,k,3), np.float32); labels = np.zeros((n,k), np.int64); mask = np.zeros((n,k), bool)
    counts = np.zeros(n, np.int32); dyn = np.zeros(n, bool)
    for i,(x,y) in enumerate(inputs.patch_xy):
        block = g[x:x+p,y:y+p]; residual = (b[x:x+p,y:y+p] == 17) & (block < 17)
        ijk = np.argwhere(residual); cls = block[residual]
        counts[i] = len(ijk); dyn[i] = np.isin(cls, (2,3,4,5,6,7,9,10)).any()
        buckets = [np.flatnonzero(cls == cid).tolist() for cid in np.unique(cls)]
        chosen = []
        while len(chosen) < min(k, len(ijk)):
            for bucket in buckets:
                if bucket and len(chosen) < k: chosen.append(bucket.pop(0))
        m = len(chosen)
        if m:
            xyz[i,:m] = (ijk[chosen]+.5)/np.asarray((p,p,z))*2-1
            labels[i,:m] = cls[chosen]; mask[i,:m] = True
    return {"xyz": xyz, "labels": labels, "mask": mask, "count": counts, "dynamic": dyn}


def sample_rows(target, count, rng):
    """Represent dynamic/static-positive/empty strata; correct presence prior."""
    groups = (np.flatnonzero(target["dynamic"]),
              np.flatnonzero((target["count"] > 0) & ~target["dynamic"]),
              np.flatnonzero(target["count"] == 0))
    allocation = [min(len(g), count//3) for g in groups]
    for i in range(3): allocation[i] += min(len(groups[i])-allocation[i], count-sum(allocation))
    ids, weights = [], []
    for group, k in zip(groups, allocation):
        if k:
            picked = rng.choice(group, k, replace=False)
            ids.extend(picked.tolist()); weights.extend([len(group)/k]*k)
    return np.asarray(ids, np.int64), np.asarray(weights, np.float32)


def compose_points(base, inputs, xyz, semantic, confidence, threshold, config=EmergenceConfig()):
    """No GT input. Stable max-confidence collisions; never overwrite V18."""
    b = np.asarray(base)
    n, k = len(inputs.patch_xy), config.queries*config.points_per_query
    points, labels, score = np.asarray(xyz), np.asarray(semantic), np.asarray(confidence)
    if points.shape != (n,k,3) or labels.shape != (n,k) or score.shape != labels.shape:
        raise ValueError("point output shape mismatch")
    if not np.isfinite(points).all() or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("nonfinite/out-of-range decoded points/confidence")
    if (not np.issubdtype(labels.dtype,np.integer) or np.any((labels < 0) | (labels > 16))
            or threshold is not None and not 0 <= threshold <= 1):
        raise ValueError("invalid semantic/threshold")
    out = b.copy()
    if threshold is None or not n: return out
    ijk = np.floor((points+1)*.5*np.asarray((config.patch_cells,config.patch_cells,b.shape[2]))).astype(np.int64)
    local_ok = ((ijk >= 0) & (ijk < np.asarray((config.patch_cells,config.patch_cells,b.shape[2])))).all(-1)
    ijk[...,:2] += inputs.patch_xy[:,None,:]
    valid = local_ok & (score >= threshold) & ((ijk >= 0) & (ijk < np.asarray(b.shape))).all(-1)
    flat = np.ravel_multi_index(ijk[valid].T, b.shape) if valid.any() else np.empty(0, np.int64)
    s, cls = score[valid], labels[valid]
    free = b.ravel()[flat] == 17; flat,s,cls = flat[free],s[free],cls[free]
    order = np.lexsort((np.arange(len(flat)), -s, flat))
    flat,cls = flat[order],cls[order]
    unique = np.r_[True, flat[1:] != flat[:-1]] if len(flat) else np.empty(0,bool)
    out.ravel()[flat[unique]] = cls[unique]
    return out


def nondegradation_gate(report, *, tolerance=1e-9):
    """Measured gate, NOT a mathematical guarantee from add-only composition."""
    delta = report["delta_vs_v18_pp"]
    metrics = ("mIoU", "IoU", "MovingMacro", "MovingMicro")
    rows = [delta] + list(delta["per_horizon"].values())
    base_rows = [report.get("baseline", {})] + list(report.get("baseline", {}).get("per_horizon", {}).values())
    pred_rows = [report.get("selected", {})] + list(report.get("selected", {}).get("per_horizon", {}).values())
    finite_nonnegative = True
    for i, row in enumerate(rows):
        for key in metrics:
            v = row[key]
            if v is not None and np.isfinite(v):
                finite_nonnegative &= v >= -tolerance
            else:
                # Undefined Moving support may be unchanged, but a corrupt
                # finite->NaN result must never pass the screen gate.
                b = base_rows[i].get(key) if i < len(base_rows) else None
                p = pred_rows[i].get(key) if i < len(pred_rows) else None
                finite_nonnegative &= (key.startswith("Moving") and b is not None and p is not None
                                       and np.isnan(b) and np.isnan(p))
    q = report["quality"]
    generated = q.get("unseen_static_semantic_tp",0) + q.get("birth_semantic_tp",0) + q.get("dormant_semantic_tp",0)
    return {"all_metrics_all_horizons_nonnegative": bool(finite_nonnegative),
            "nonzero_correct_generated_geometry": generated > 0,
            "added": q.get("added",0), "generated_semantic_tp": generated,
            "pass": bool(finite_nonnegative and generated > 0)}
