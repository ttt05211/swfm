"""Fixed causal corruption recipe. Never consumes targets or global RNG."""
from dataclasses import asdict, dataclass
import hashlib
import itertools
import numpy as np
from real_motion.canonical_causal_repair import grid_arrays
from real_motion.source_evidence_audit import transform_points


@dataclass(frozen=True)
class Recipe:
    probability: float = .5
    patches: int = 4
    xy_side_m: tuple = (2., 6.)
    z_side_m: tuple = (.8, 2.4)
    max_changed_fraction: float = .15
    persistent_shift_voxels: int = 1
    frame_jitter_voxels: int = 1
    protocol: str = 'causal_world_patch_corruption_v1'

    def specification(self):
        value = asdict(self)
        for field in ('xy_side_m', 'z_side_m'): value[field] = list(value[field])
        return value


RECIPE = Recipe()
# Fixed plausible confusions, NOT fitted on DEV/STC validation predictions.
CONFUSION = np.array([15, 15, 6, 10, 10, 10, 2, 2, 1, 10, 4,
                      13, 14, 11, 12, 16, 15, 17], np.uint8)


def local_rng(record, epoch, seed):
    identity = f'{seed}:{epoch}:{record["scene_name"]}:{record["t0_token"]}'
    digest = hashlib.sha256(identity.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def patch_cells(center, sides, pose, t0_pose, grid):
    """A single persistent box in t0 axes, projected into EACH history frame."""
    origin, step, shape = grid_arrays(grid)
    offsets = np.array(list(itertools.product((-1., 1.), repeat=3)))*sides/2
    corners = center + offsets @ np.asarray(t0_pose)[:3, :3].T
    inverse = np.linalg.inv(pose)
    local = transform_points(corners, inverse)
    lo = np.maximum(np.floor((local.min(0)-origin)/step).astype(int), 0)
    hi = np.minimum(np.ceil((local.max(0)-origin)/step).astype(int)+1, shape)
    if np.any(hi <= lo): return np.empty((0, 3), np.int64)
    cells = np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(lo, hi)), indexing='ij'), -1).reshape(-1, 3)
    world = transform_points(origin+(cells+.5)*step, pose)
    offset = (world-center) @ np.asarray(t0_pose)[:3, :3]
    return cells[(np.abs(offset) <= sides/2+1e-8).all(1)]


def corrupt(raw, record, epoch, seed, grid, *, recipe=RECIPE, force_family=None):
    # Deliberately whitelist historical fields; labels/poses for future horizons
    # and cached clean component/background artifacts cannot affect corruption.
    labels = np.asarray(raw['history_occ']); known = np.asarray(raw['history_observed'])
    poses = np.asarray(raw['history_poses'])
    if labels.ndim != 4 or len(labels) != 4 or known.shape != labels.shape or known.dtype != bool:
        raise ValueError('four integer semantic histories and matching bool known required')
    if labels.dtype.kind not in 'ui' or poses.shape != (4, 4, 4) or not np.isfinite(poses).all():
        raise ValueError('invalid historical semantics/poses')
    origin, step, shape = grid_arrays(grid)
    if tuple(shape) != labels.shape[1:]: raise ValueError('augmentation grid differs')
    rng = local_rng(record, epoch, seed)
    result = dict(history_occ=labels.copy(), history_observed=known.copy(), history_poses=poses.copy())
    audit = dict(selected=False, family='clean', changed_voxels=[0]*4,
                 known_unchanged=True, clean_geometry_reusable=True, epoch=int(epoch))
    if force_family is None and rng.random() >= recipe.probability: return result, audit
    family = force_family or ('dropout', 'semantic_false_occupancy', 'geometry')[int(rng.integers(3))]
    if family not in ('dropout', 'semantic_false_occupancy', 'geometry'): raise ValueError('unknown corruption family')
    anchors = np.argwhere((labels[-1] < 17) & known[-1])
    if not len(anchors):
        audit.update(selected=True, family=family, no_occupied_anchor=True)
        return result, audit
    occupied = ((labels < 17) & known).sum((1, 2, 3))
    # Per-frame cap is relative to INPUT occupied count, not future target size.
    caps = np.maximum(1, np.floor(occupied*recipe.max_changed_fraction).astype(int))
    t0 = poses[-1]
    persistent = rng.integers(-recipe.persistent_shift_voxels, recipe.persistent_shift_voxels+1, 3)*step
    persistent[2] = 0  # frame-local subvoxel/height flicker below may move Z.
    for _ in range(recipe.patches):
        anchor = anchors[int(rng.integers(len(anchors)))]
        center = transform_points((origin+(anchor+.5)*step)[None], t0)[0]
        sides = np.array([rng.uniform(*recipe.xy_side_m), rng.uniform(*recipe.xy_side_m),
                          rng.uniform(*recipe.z_side_m)])
        fake_class = int(labels[-1][tuple(anchor)])
        for frame in range(4):
            at = patch_cells(center, sides, poses[frame], t0, grid)
            if not len(at): continue
            valid = known[frame][tuple(at.T)] & (labels[frame][tuple(at.T)] < 18)
            at = at[valid]
            if not len(at): continue
            candidate = result['history_occ'][frame].copy()
            idx = tuple(at.T); values = candidate[idx]
            if family == 'dropout': candidate[idx] = 17
            elif family == 'semantic_false_occupancy':
                candidate[idx] = CONFUSION[values]
                # Spatially contiguous false surface, not isolated salt noise.
                bottom = at[:, 2].min()
                free = (values == 17) & (at[:, 2] <= bottom+1)
                candidate[tuple(at[free].T)] = fake_class
            else:
                src = at[values < 17]; source_values = candidate[tuple(src.T)].copy()
                jitter = rng.integers(-recipe.frame_jitter_voxels, recipe.frame_jitter_voxels+1, 3)*step
                shift_world = (persistent+jitter) @ t0[:3, :3].T
                shift = np.rint((shift_world @ poses[frame][:3, :3])/step).astype(int)
                dest = src+shift
                inside = ((dest >= 0) & (dest < shape)).all(1)
                src, dest, source_values = src[inside], dest[inside], source_values[inside]
                good = known[frame][tuple(dest.T)] & (labels[frame][tuple(dest.T)] < 18)
                src, dest, source_values = src[good], dest[good], source_values[good]
                candidate[tuple(src.T)] = 17
                candidate[tuple(dest.T)] = source_values
            # Reject an oversized whole patch rather than randomly scattering
            # its edits to meet the budget. Preserve invalid/unknown semantics.
            if np.count_nonzero(candidate != labels[frame]) <= caps[frame]:
                result['history_occ'][frame] = candidate
    changes = np.count_nonzero(result['history_occ'] != labels, axis=(1, 2, 3)).tolist()
    audit.update(selected=True, family=family, changed_voxels=changes,
                 clean_geometry_reusable=not any(changes))
    return result, audit
