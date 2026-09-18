"""Bit-exact-oriented runtime fast paths for the frozen Strong/KTA + V18 pipeline.

These functions are engineering replacements for slow reference implementations.
They do not change the method contract.  Every fast path has a reference
equivalent and is intended to be gated by np.array_equal tests before paper
runtime numbers are accepted.

The main optimizations are:
- sparse 5x5x1 majority fill: evaluate only unknown voxels instead of running
  one full-grid scipy uniform_filter per semantic class;
- cropped connected components: run scipy label only inside the bounding box of
  each present dynamic class;
- direct A1 CLEAR mask construction: one dense boolean mask per horizon instead
  of allocating one 200x200x16 mask per source component.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import generate_binary_structure, label, uniform_filter

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .rigid_transport import RasterizedRigidComponent
from .strong_w2det import StrongW2DetConfig, _transform_points


def majority_fill_sparse_5x5x1(
    semantics: np.ndarray,
    unknown_mask: np.ndarray,
    *,
    kernel: tuple[int, int, int] = (5, 5, 1),
    min_fraction: float = 0.3,
) -> np.ndarray:
    """Bit-exact batched equivalent of the frozen majority fill.

    A first attempt used direct integer neighborhood counts only at unknown
    voxels.  That is mathematically equivalent, but it is *not* bit-exact to
    scipy.ndimage.uniform_filter(float32): separable float32 filtering can round
    at slightly different points and flip threshold/tie decisions.

    This implementation therefore preserves scipy's exact filtering path while
    reducing Python overhead: all present semantic classes are stacked as a
    leading channel dimension and filtered in one 4-D call with filter size
    (1, 5, 5, 1).  Because the class-axis kernel is one, each channel is
    independent and numerically identical to the frozen per-class reference.
    """
    if tuple(int(x) for x in kernel) != (5, 5, 1):
        raise ValueError("runtime majority fill only supports frozen kernel=(5,5,1)")

    sem = np.asarray(semantics)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if sem.shape != unknown.shape:
        raise ValueError("semantics/unknown shape mismatch")
    if not bool(unknown.any()):
        return sem.copy()
    if sem.ndim != 3:
        raise ValueError("semantic occupancy must be 3-D")

    known = ~unknown
    denom = uniform_filter(
        known.astype(np.float32),
        size=(5, 5, 1),
        mode="constant",
    )
    denom = np.maximum(denom, np.float32(1e-6))

    classes = np.unique(sem[known])
    if len(classes) == 0:
        return sem.copy()

    class_masks = (
        (sem[None, ...] == classes[:, None, None, None])
        & known[None, ...]
    ).astype(np.float32)
    scores = uniform_filter(
        class_masks,
        size=(1, 5, 5, 1),
        mode="constant",
    )
    scores = scores / denom[None, ...]

    # Reference loops np.unique(...) in ascending order and only updates on
    # strict '>'.  np.argmax returns the first maximum, exactly matching that
    # tie rule.
    best_idx = np.argmax(scores, axis=0)
    best_score = np.take_along_axis(
        scores, best_idx[None, ...], axis=0
    )[0]
    best_label = classes[best_idx]

    fill = unknown & (best_score >= float(min_fraction))
    out = sem.copy()
    out[fill] = best_label[fill]
    return out

def extract_instances_cropped_exact(
    semantics: np.ndarray,
    ego_to_world: np.ndarray,
    *,
    grid,
    cfg: StrongW2DetConfig,
) -> list[dict]:
    """Same-class connected components with class-local cropped labeling.

    Component ordering remains class-major then scipy component-id, matching the
    frozen reference.  Coordinates from np.argwhere are C-order and therefore
    preserve the reference centroid summation order after the crop offset is
    restored.
    """
    sem = np.asarray(semantics)
    if tuple(sem.shape) != tuple(grid.shape_hwd):
        raise ValueError("semantic grid shape mismatch")
    structure = generate_binary_structure(3, int(cfg.connectivity))
    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    out: list[dict] = []

    dyn = np.isin(sem, np.asarray(DYNAMIC_CLASS_IDS, dtype=sem.dtype))
    all_idx = np.argwhere(dyn)
    if len(all_idx) == 0:
        return out
    all_labels = sem[all_idx[:, 0], all_idx[:, 1], all_idx[:, 2]]

    for cls in DYNAMIC_CLASS_IDS:
        cls_pts = all_idx[all_labels == int(cls)]
        if len(cls_pts) < int(cfg.min_component_voxels):
            continue
        lo = cls_pts.min(axis=0)
        hi = cls_pts.max(axis=0) + 1
        sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
        crop = sem[sl] == int(cls)
        comp_map, n = label(crop, structure=structure)
        for comp_id in range(1, int(n) + 1):
            local = np.argwhere(comp_map == comp_id)
            if len(local) < int(cfg.min_component_voxels):
                continue
            idx = local.astype(np.int64, copy=False) + lo[None].astype(np.int64)
            pts_ego = origin + (idx.astype(np.float64) + 0.5) * step
            centroid_world = _transform_points(
                ego_to_world, pts_ego.mean(axis=0, keepdims=True)
            )[0]
            out.append(
                {
                    "class_id": int(cls),
                    "voxel_indices": idx,
                    "centroid_world": centroid_world,
                    "voxel_count": int(len(idx)),
                }
            )
    return out


def component_lists_equal(a: Sequence[dict], b: Sequence[dict]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if int(x["class_id"]) != int(y["class_id"]):
            return False
        if int(x["voxel_count"]) != int(y["voxel_count"]):
            return False
        if not np.array_equal(
            np.asarray(x["voxel_indices"], dtype=np.int64),
            np.asarray(y["voxel_indices"], dtype=np.int64),
        ):
            return False
        # The downstream matching/KTA contract depends on exact centroid values.
        if not np.array_equal(
            np.asarray(x["centroid_world"], dtype=np.float64),
            np.asarray(y["centroid_world"], dtype=np.float64),
        ):
            return False
    return True


def compose_component_replacements_fast_exact(
    anchor_occ: np.ndarray,
    baseline_components: Iterable[RasterizedRigidComponent],
    replacement_components: Iterable[RasterizedRigidComponent],
    *,
    dynamic_class_ids: Sequence[int],
    free_label: int,
    grid,
    precomputed_clear_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Frozen A1 compositor without one dense mask allocation per component."""
    anchor = np.asarray(anchor_occ)
    if tuple(anchor.shape) != tuple(grid.shape_hwd):
        raise ValueError("anchor_occ shape differs from occupancy grid")
    out = anchor.copy()

    if precomputed_clear_mask is None:
        clear = np.zeros(grid.shape_hwd, dtype=bool)
        for comp in baseline_components:
            idx = np.asarray(comp.voxel_indices, dtype=np.int64)
            if len(idx):
                clear[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    else:
        clear = np.asarray(precomputed_clear_mask, dtype=bool)
        if clear.shape != anchor.shape:
            raise ValueError("precomputed_clear_mask shape mismatch")

    lut = np.zeros(256, dtype=bool)
    for cid in dynamic_class_ids:
        lut[int(cid)] = True
    out[clear & lut[out.astype(np.uint8, copy=False)]] = int(free_label)

    # Preserve exact frozen source/input write order.
    for comp in replacement_components:
        idx = np.asarray(comp.voxel_indices, dtype=np.int64)
        if len(idx):
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = int(comp.class_id)
    return out


def baseline_clear_mask(
    baseline_components: Iterable[RasterizedRigidComponent],
    *,
    grid,
) -> np.ndarray:
    clear = np.zeros(grid.shape_hwd, dtype=bool)
    for comp in baseline_components:
        idx = np.asarray(comp.voxel_indices, dtype=np.int64)
        if len(idx):
            clear[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return clear
