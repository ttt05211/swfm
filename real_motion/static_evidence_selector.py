"""Causal, whole-patch static evidence selection (no learned shape generation).

Feature construction has no future-GT argument. Supervision is deliberately a
separate function. The proposal uses the exact oldest-to-newest V19 memory.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

PROTOCOL = "p0_f9_static_evidence_selector_v1"
FEATURE_PROTOCOL = "causal_history_cross5_semantic_height_utility_v1"
DYNAMIC_IDS = (2, 3, 4, 5, 6, 7, 9, 10)
PATCH_CELLS = 4
HISTORY_DIM = 22  # 18 semantic fractions, coverage, height mean/std, occupancy
CONTEXT_DIM = 47  # baseline22 + proposal22 + x/y/horizon
THRESHOLD = 0.5
NEIGHBORS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


@lru_cache(maxsize=8)
def patch_layout(shape, patch_cells=PATCH_CELLS):
    shape = tuple(int(v) for v in shape)
    if len(shape) != 3 or min(shape) < 1 or int(patch_cells) < 1:
        raise ValueError("invalid patch grid")
    x, y, z = shape
    nx, ny = (x + patch_cells - 1) // patch_cells, (y + patch_cells - 1) // patch_cells
    xy = (np.arange(x)[:, None] // patch_cells) * ny + np.arange(y)[None, :] // patch_cells
    flat = np.repeat(xy.reshape(-1), z).astype(np.int32)
    capacity = np.bincount(flat, minlength=nx * ny).astype(np.float32)
    flat.setflags(write=False); capacity.setflags(write=False)
    return flat, capacity, (nx, ny)


def patch_descriptors(semantics, known, *, patch_cells=PATCH_CELLS):
    sem, known = np.asarray(semantics), np.asarray(known, bool)
    if sem.shape != known.shape or sem.ndim != 3 or sem.min() < 0 or sem.max() > 17:
        raise ValueError("invalid semantic/known grid")
    patch, capacity, _ = patch_layout(sem.shape, patch_cells)
    ids = np.flatnonzero(known)
    n = len(capacity)
    counts = np.bincount(patch[ids].astype(np.int64) * 18 + sem.reshape(-1)[ids],
                         minlength=n * 18).reshape(n, 18).astype(np.float32)
    out = np.zeros((n, HISTORY_DIM), np.float32)
    out[:, :18] = counts / capacity[:, None]
    out[:, 18] = counts.sum(1) / capacity
    occupied = ids[sem.reshape(-1)[ids] != 17]
    pid = patch[occupied]
    heights = (occupied % sem.shape[2] + .5) / sem.shape[2]
    mass = np.bincount(pid, minlength=n).astype(np.float32)
    mean = np.bincount(pid, weights=heights, minlength=n) / np.maximum(mass, 1)
    second = np.bincount(pid, weights=heights ** 2, minlength=n) / np.maximum(mass, 1)
    out[:, 19] = mean
    out[:, 20] = np.sqrt(np.maximum(0, second - mean ** 2))
    out[:, 21] = mass / capacity
    return out


@dataclass
class SelectorInputs:
    history: np.ndarray  # [N,6,5,22], float16; temporal x spatial evidence
    context: np.ndarray  # [N,47], float16; causal future query
    patch_ids: np.ndarray
    voxel_flat: np.ndarray  # ONLY baseline-free static proposal voxels
    voxel_patch_rows: np.ndarray
    labels: np.ndarray


def assemble_inputs(history_descriptors, baseline, memory, horizon_s, *, patch_cells=PATCH_CELLS):
    """Assemble the same causal input for training and deployment."""
    b, m = np.asarray(baseline), np.asarray(memory)
    if b.shape != m.shape or not (0 < float(horizon_s) <= 3.0):
        raise ValueError("invalid query")
    if b.min() < 0 or b.max() > 17 or m.min() < 0 or m.max() > 17:
        raise ValueError("semantic labels must be 0..17")
    if np.isin(m, DYNAMIC_IDS).any():
        raise ValueError("static memory contains a dynamic class")
    patch, _, (nx, ny) = patch_layout(b.shape, patch_cells)
    eligible = (b == 17) & (m != 17)
    flat = np.flatnonzero(eligible).astype(np.int32)
    active = np.unique(patch[flat])
    rows = np.searchsorted(active, patch[flat]).astype(np.int32)
    hist = np.asarray(history_descriptors, np.float32)
    if hist.shape != (6, nx * ny, HISTORY_DIM) or not np.isfinite(hist).all():
        raise ValueError("history descriptor contract mismatch")
    x, y = active // ny, active % ny
    tokens = np.zeros((len(active), 6, len(NEIGHBORS), HISTORY_DIM), np.float16)
    for ni, (dx, dy) in enumerate(NEIGHBORS):
        xx, yy = x + dx, y + dy
        valid = (xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny)
        tokens[valid, :, ni] = hist[:, xx[valid] * ny + yy[valid]].transpose(1, 0, 2)
    base_features = patch_descriptors(b, np.ones(b.shape, bool), patch_cells=patch_cells)
    proposal_features = patch_descriptors(m, eligible, patch_cells=patch_cells)
    coords = np.column_stack(((x + .5) / nx * 2 - 1, (y + .5) / ny * 2 - 1,
                              np.full(len(active), float(horizon_s) / 3)))
    context = np.concatenate((base_features[active], proposal_features[active], coords), axis=1)
    if context.shape != (len(active), CONTEXT_DIM) or not np.isfinite(context).all():
        raise ValueError("query descriptor contract mismatch")
    return SelectorInputs(tokens, context.astype(np.float16), active, flat, rows,
                          m.reshape(-1)[flat].astype(np.uint8))


def prepare_selector_inputs(history_semantics, history_observed, history_poses, future_poses,
                            baseline_predictions, *, grid, workers=8):
    """Six-frame sparse projections shared by memory and input descriptors.

    Imports are lazy so the core math can be tested without Torch. Latest free
    observations clear old evidence, exactly as in the formal V19 mosaic.
    """
    from .geometry import relative_transform
    from .v19_innovation import (prepare_history_alignment_frame, _xyz_to_indices,
                                _semantic_choices_from_pretransformed)
    if not all(len(x) == 6 for x in (history_semantics, history_observed, history_poses,
                                    future_poses, baseline_predictions)) or workers < 1:
        raise ValueError("expected six history and six future states")
    prepared = [prepare_history_alignment_frame(s, o, p, grid=grid, dynamic_class_ids=DYNAMIC_IDS)
                for s, o, p in zip(history_semantics, history_observed, history_poses)]
    shape = tuple(grid.shape_hwd)
    _, y, z = shape

    def one(h):
        memory = np.full(shape, 17, np.uint8)
        descriptors = []
        for pose, xyz, labels, static in prepared:
            transform = relative_transform(pose, np.asarray(future_poses[h], np.float64))
            dst = xyz @ transform[:3, :3].T + transform[:3, 3]
            ix, iy, iz, valid = _xyz_to_indices(dst, grid)
            all_flat = ((ix[valid] * y + iy[valid]) * z + iz[valid]).astype(np.int64)
            clear = valid & static
            clear_flat = ((ix[clear] * y + iy[clear]) * z + iz[clear]).astype(np.int64)
            memory.reshape(-1)[clear_flat] = 17
            _, _, _, values, write = _semantic_choices_from_pretransformed(
                labels, ix, iy, iz, dst, valid, static, grid=grid, free_label=17)
            memory.reshape(-1)[write] = values
            plane = np.full(shape, 17, np.uint8)
            known = np.zeros(shape, bool)
            known.reshape(-1)[all_flat] = True
            _, _, _, observed_labels, observed_write = _semantic_choices_from_pretransformed(
                labels, ix, iy, iz, dst, valid, np.ones(len(labels), bool), grid=grid, free_label=17)
            plane.reshape(-1)[observed_write] = observed_labels
            descriptors.append(patch_descriptors(plane, known))
        inputs = assemble_inputs(descriptors, baseline_predictions[h], memory, .5 * (h + 1))
        return inputs, memory

    with ThreadPoolExecutor(max_workers=min(workers, 6)) as pool:
        rows = list(pool.map(one, range(6)))
    return [r[0] for r in rows], [r[1] for r in rows]


def supervision_counts(inputs, future_gt):
    """GT ONLY here, not in feature construction; counts for one utility loss."""
    g = np.asarray(future_gt)
    if g.min() < 0 or g.max() > 17 or (len(inputs.voxel_flat) and inputs.voxel_flat.max() >= g.size):
        raise ValueError("invalid future GT")
    correct = g.reshape(-1)[inputs.voxel_flat] == inputs.labels
    n = len(inputs.patch_ids)
    good = np.bincount(inputs.voxel_patch_rows, weights=correct, minlength=n).astype(np.float32)
    total = np.bincount(inputs.voxel_patch_rows, minlength=n).astype(np.float32)
    return good, total - good


def sample_training_patches(correct, wrong, count, rng):
    """Balanced utility signs, WITH inverse sampling weights for calibration.

    No dev sampling, future input or threshold tuning. Rare useful patches are
    represented without pretending their artificial 50/50 prevalence is real.
    """
    a, b = np.asarray(correct), np.asarray(wrong)
    if a.shape != b.shape or count < 1 or np.any(a < 0) or np.any(b < 0):
        raise ValueError("invalid training population")
    groups = [np.flatnonzero(a > b), np.flatnonzero(a <= b)]
    if count == 1 and all(len(g) for g in groups):
        raise ValueError("at least two samples required to represent both utility strata")
    counts = [min(len(g), count // 2) for g in groups]
    for k in range(2):
        counts[k] += min(len(groups[k]) - counts[k], count - sum(counts))
    indices, weights = [], []
    for g, k in zip(groups, counts):
        if k:
            selected = rng.choice(g, k, replace=False)
            indices.extend(selected.tolist())
            weights.extend([len(g) / k] * k)
    return np.asarray(indices, np.int64), np.asarray(weights, np.float32)


def compose_selected_static(baseline, inputs, probability):
    b = np.asarray(baseline)
    p = np.asarray(probability)
    if p.shape != (len(inputs.patch_ids),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid selector probabilities")
    if np.isin(inputs.labels, DYNAMIC_IDS).any() or np.any(inputs.labels >= 17):
        raise ValueError("non-static proposal")
    if np.any(b.reshape(-1)[inputs.voxel_flat] != 17):
        raise ValueError("proposal was not built against this V18 free population")
    out = b.copy()
    take = p[inputs.voxel_patch_rows] >= THRESHOLD
    out.reshape(-1)[inputs.voxel_flat[take]] = inputs.labels[take]
    return out


def sparse_static_counts(base_counts, labels, gt, selected):
    """Exact add-only raw counts without retaining dense dev predictions.

    Static additions cannot change any frozen dynamic-class Moving numerator
    or denominator, even when they wrongly land on a dynamic GT voxel.
    """
    labels, gt, selected = np.asarray(labels), np.asarray(gt), np.asarray(selected, bool)
    if labels.shape != gt.shape or labels.shape != selected.shape or np.isin(labels, DYNAMIC_IDS).any():
        raise ValueError("invalid static edit")
    if labels.size and (labels.min() < 0 or labels.max() >= 17 or gt.min() < 0 or gt.max() > 17):
        raise ValueError("invalid semantic labels")
    lab, g = labels[selected].astype(np.int64), gt[selected].astype(np.int64)
    conf = np.bincount(g * 18 + lab, minlength=324).reshape(18, 18)
    old = np.bincount(g * 18 + 17, minlength=324).reshape(18, 18)
    d = conf - old
    oi, ou, si, su, mi, mu = base_counts
    return (int(oi + d[:17, :17].sum()), int(ou - d[17, 17]),
            np.asarray(si) + np.diag(d)[:17],
            np.asarray(su) + (d.sum(0) + d.sum(1) - np.diag(d))[:17],
            np.asarray(mi).copy(), np.asarray(mu).copy())


def acceptance_gate(delta):
    """Predeclared one-screen contract; never tuned against dev results."""
    horizons = delta["per_horizon"]
    tests = {"mIoU_ge_0_5pp": delta["mIoU"] >= .5,
             "all_report_horizons_nonnegative": all(horizons[str(h)]["mIoU"] >= -1e-10 for h in (1.0, 2.0, 3.0)),
             "moving_unchanged": all(abs(delta[k]) <= 1e-10 and all(abs(v[k]) <= 1e-10 for v in horizons.values())
                                     for k in ("MovingMacro", "MovingMicro"))}
    return {**tests, "pass": all(tests.values())}
