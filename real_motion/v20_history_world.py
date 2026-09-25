"""V20 three-dimensional historical-evidence world-model contracts.

This module owns geometry, responsibility labels and protected composition.
It intentionally has no dataset-specific loader so every training/evaluation
entry point shares the same causal contract.

Inference may use:
  * six historical semantic OCC grids;
  * six historical lidar-observation masks;
  * six historical ego poses;
  * the same future ego trajectory conditioning already permitted by V18.

Future semantic occupancy, future observation masks and GT identities are
supervision/evaluation only and are rejected by the inference-input audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .motion_transport import FUTURE_FRAMES, HISTORY_FRAMES

V20_PROTOCOL = "p0_f9_v20_3d_history_world_model_v1"
FREE_LABEL = 17
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)
DYNAMIC_SET = frozenset(DYNAMIC_IDS)


@dataclass(frozen=True)
class CanonicalLattice:
    """Fixed canonical superset lattice Ωmax.

    The tensor shape and indices are fixed over the run.  High-resolution model
    work is still expected to be restricted by a per-window query mask.
    """

    origin_xyz_m: tuple[float, float, float]
    voxel_size_xyz_m: tuple[float, float, float]
    shape_xyz: tuple[int, int, int]

    def __post_init__(self):
        if len(self.origin_xyz_m) != 3 or len(self.voxel_size_xyz_m) != 3:
            raise ValueError("origin/voxel size must be xyz triples")
        if len(self.shape_xyz) != 3 or min(self.shape_xyz) <= 0:
            raise ValueError("shape_xyz must be positive")
        if min(self.voxel_size_xyz_m) <= 0:
            raise ValueError("voxel size must be positive")

    @property
    def max_xyz_m(self) -> np.ndarray:
        return np.asarray(self.origin_xyz_m, dtype=np.float64) + (
            np.asarray(self.shape_xyz, dtype=np.float64)
            * np.asarray(self.voxel_size_xyz_m, dtype=np.float64)
        )

    def world_to_index(self, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(xyz_world, dtype=np.float64)
        if p.shape[-1] != 3:
            raise ValueError("xyz_world last dimension must be 3")
        origin = np.asarray(self.origin_xyz_m, dtype=np.float64)
        step = np.asarray(self.voxel_size_xyz_m, dtype=np.float64)
        idx = np.floor((p - origin) / step).astype(np.int64)
        shp = np.asarray(self.shape_xyz, dtype=np.int64)
        valid = ((idx >= 0) & (idx < shp)).all(axis=-1)
        return idx, valid

    def index_to_world_center(self, idx_xyz: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx_xyz, dtype=np.float64)
        if idx.shape[-1] != 3:
            raise ValueError("idx last dimension must be 3")
        return (
            np.asarray(self.origin_xyz_m, dtype=np.float64)
            + (idx + 0.5) * np.asarray(self.voxel_size_xyz_m, dtype=np.float64)
        )


@dataclass(frozen=True)
class QueryMaskReport:
    mask: np.ndarray
    requested_voxels: int
    in_bounds_voxels: int
    out_of_bounds_voxels: int
    out_of_bounds_fraction: float


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    if T.shape != (4, 4) or p.shape[-1] != 3:
        raise ValueError("expected T[4,4] and points[...,3]")
    return p @ T[:3, :3].T + T[:3, 3]


def grid_centers_xyz(
    shape_xyz: Sequence[int],
    origin_xyz_m: Sequence[float],
    voxel_size_xyz_m: Sequence[float],
) -> np.ndarray:
    shape = tuple(int(x) for x in shape_xyz)
    origin = np.asarray(origin_xyz_m, dtype=np.float64)
    step = np.asarray(voxel_size_xyz_m, dtype=np.float64)
    axes = [
        origin[d] + (np.arange(shape[d], dtype=np.float64) + 0.5) * step[d]
        for d in range(3)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return np.stack((x, y, z), axis=-1)


def future_union_query_mask(
    lattice: CanonicalLattice,
    *,
    future_ego_to_world: np.ndarray,
    native_shape_xyz: Sequence[int],
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
) -> QueryMaskReport:
    """Rasterize the union of six future native OCC views into Ωmax.

    This uses future ego *pose only*.  It never inspects future semantics or
    future lidar masks.  Duplicate mapped cells are naturally unioned.
    """
    poses = np.asarray(future_ego_to_world, dtype=np.float64)
    if poses.shape != (FUTURE_FRAMES, 4, 4):
        raise ValueError("future_ego_to_world must be [6,4,4]")
    local = grid_centers_xyz(
        native_shape_xyz, native_origin_xyz_m, native_voxel_size_xyz_m
    ).reshape(-1, 3)
    out = np.zeros(lattice.shape_xyz, dtype=bool)
    requested = 0
    in_bounds = 0
    oob = 0
    for T in poses:
        world = transform_points(T, local)
        idx, valid = lattice.world_to_index(world)
        requested += int(len(idx))
        in_bounds += int(valid.sum())
        oob += int((~valid).sum())
        good = idx[valid]
        if len(good):
            out[good[:, 0], good[:, 1], good[:, 2]] = True
    return QueryMaskReport(
        mask=out,
        requested_voxels=requested,
        in_bounds_voxels=in_bounds,
        out_of_bounds_voxels=oob,
        out_of_bounds_fraction=float(oob / max(requested, 1)),
    )


@dataclass(frozen=True)
class AlignedHistoryEvidence:
    semantic: np.ndarray
    observed: np.ndarray
    observed_free: np.ndarray
    unknown: np.ndarray
    conflict: np.ndarray
    out_of_bounds_samples: int


def align_history_once_to_canonical(
    lattice: CanonicalLattice,
    *,
    history_semantic: np.ndarray,
    history_observed: np.ndarray,
    history_ego_to_world: np.ndarray,
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
    free_label: int = FREE_LABEL,
) -> AlignedHistoryEvidence:
    """Align each historical frame exactly once into the canonical lattice.

    The return keeps time as an explicit axis [T,X,Y,Z].  Unknown is never
    conflated with observed-free.  If more than one native voxel quantizes to a
    canonical cell, observed occupied evidence takes precedence over free;
    disagreeing occupied labels are marked as conflict.
    """
    sem = np.asarray(history_semantic)
    obs = np.asarray(history_observed, dtype=bool)
    poses = np.asarray(history_ego_to_world, dtype=np.float64)
    if sem.shape != obs.shape or sem.ndim != 4:
        raise ValueError("history semantic/observed must be [T,X,Y,Z]")
    if sem.shape[0] != HISTORY_FRAMES or poses.shape != (HISTORY_FRAMES, 4, 4):
        raise ValueError("V20 requires exactly six historical frames")

    native_shape = sem.shape[1:]
    local = grid_centers_xyz(
        native_shape, native_origin_xyz_m, native_voxel_size_xyz_m
    ).reshape(-1, 3)
    cshape = (HISTORY_FRAMES,) + tuple(lattice.shape_xyz)
    out_sem = np.full(cshape, int(free_label), dtype=np.uint8)
    out_obs = np.zeros(cshape, dtype=bool)
    conflict = np.zeros(cshape, dtype=bool)
    oob = 0

    for t in range(HISTORY_FRAMES):
        src_sem = sem[t].reshape(-1)
        src_obs = obs[t].reshape(-1)
        world = transform_points(poses[t], local)
        idx, valid = lattice.world_to_index(world)
        valid &= src_obs
        oob += int((src_obs & ~lattice.world_to_index(world)[1]).sum())
        for cell, label in zip(idx[valid], src_sem[valid]):
            key = (t, int(cell[0]), int(cell[1]), int(cell[2]))
            was = out_obs[key]
            old = int(out_sem[key])
            lab = int(label)
            if not was:
                out_sem[key] = lab
                out_obs[key] = True
            elif old == int(free_label) and lab != int(free_label):
                out_sem[key] = lab
            elif old != int(free_label) and lab != int(free_label) and old != lab:
                conflict[key] = True

    observed_free = out_obs & (out_sem == int(free_label))
    unknown = ~out_obs
    if bool((observed_free & unknown).any()):
        raise RuntimeError("observed-free and unknown must be disjoint")
    return AlignedHistoryEvidence(
        semantic=out_sem,
        observed=out_obs,
        observed_free=observed_free,
        unknown=unknown,
        conflict=conflict,
        out_of_bounds_samples=int(oob),
    )


class DynamicResponsibility(IntEnum):
    IGNORE = 0
    CURRENT_ANCESTRAL = 1
    DORMANT_ANCESTRAL = 2
    BIRTH = 3


@dataclass(frozen=True)
class DynamicPartition:
    labels: np.ndarray
    current: tuple[int, ...]
    dormant: tuple[int, ...]
    birth: tuple[int, ...]
    ignore: tuple[int, ...]

    def assert_mutually_exclusive(self) -> None:
        groups = [set(self.current), set(self.dormant), set(self.birth), set(self.ignore)]
        union = set().union(*groups)
        if len(union) != sum(len(g) for g in groups):
            raise RuntimeError("dynamic responsibility groups overlap")


def partition_future_dynamic_instances(
    *,
    num_future_instances: int,
    matches_t0: Mapping[int, bool],
    matches_earlier_history: Mapping[int, bool],
    ambiguous: Iterable[int] = (),
) -> DynamicPartition:
    """Strict GT-only responsibility partition used for labels/evaluation.

    A BIRTH is *not* merely absent at t0: it must lack a reliable observed
    ancestor throughout all six history frames.
    """
    n = int(num_future_instances)
    amb = {int(x) for x in ambiguous}
    labels = np.full(n, int(DynamicResponsibility.IGNORE), dtype=np.uint8)
    current, dormant, birth, ignore = [], [], [], []
    for i in range(n):
        if i in amb:
            ignore.append(i)
            continue
        at_t0 = bool(matches_t0.get(i, False))
        earlier = bool(matches_earlier_history.get(i, False))
        if at_t0:
            labels[i] = int(DynamicResponsibility.CURRENT_ANCESTRAL)
            current.append(i)
        elif earlier:
            labels[i] = int(DynamicResponsibility.DORMANT_ANCESTRAL)
            dormant.append(i)
        else:
            labels[i] = int(DynamicResponsibility.BIRTH)
            birth.append(i)
    out = DynamicPartition(
        labels=labels,
        current=tuple(current),
        dormant=tuple(dormant),
        birth=tuple(birth),
        ignore=tuple(ignore),
    )
    out.assert_mutually_exclusive()
    return out


def static_supervision_mask(
    future_gt_semantic: np.ndarray,
    future_gt_observed: np.ndarray,
    *,
    free_label: int = FREE_LABEL,
    dynamic_class_ids: Sequence[int] = DYNAMIC_IDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (valid, target) for static semantic supervision.

    Supervise only future observed voxels. Dynamic occupied GT is ignored.
    Observed free remains a valid negative/semantic class.
    """
    gt = np.asarray(future_gt_semantic)
    obs = np.asarray(future_gt_observed, dtype=bool)
    if gt.shape != obs.shape:
        raise ValueError("future GT semantic/observed shape mismatch")
    dyn = np.isin(gt, np.asarray(dynamic_class_ids, dtype=gt.dtype))
    valid = obs & ~dyn
    target = np.asarray(gt, dtype=np.uint8).copy()
    target[~valid] = int(free_label)
    return valid, target


def audit_inference_inputs(inputs: Mapping[str, object]) -> None:
    """Fail fast on accidental future-label leakage."""
    forbidden_exact = {
        "future_semantic",
        "future_occ",
        "future_occupancy",
        "future_mask_lidar",
        "future_observed",
        "future_instance_id",
        "future_instance_ids",
        "gt_instance_id",
        "gt_instance_ids",
    }
    bad = []
    for key in inputs:
        low = str(key).lower()
        if low in forbidden_exact:
            bad.append(str(key))
        elif low.startswith("future_gt") or low.startswith("gt_future"):
            bad.append(str(key))
    if bad:
        raise RuntimeError("future supervision leaked into inference inputs: " + ", ".join(sorted(bad)))


def protected_add_only(
    base_v18: torch.Tensor,
    *,
    static_world: torch.Tensor | None = None,
    birth: torch.Tensor | None = None,
    dormant: torch.Tensor | None = None,
    free_label: int = FREE_LABEL,
) -> torch.Tensor:
    """Compose V20 in fixed priority without overwriting V18 occupied voxels.

    Priority:
        V18 current source > Dormant > Birth > Static.

    All added branches write only where the original V18 tensor was free.
    Later lower-priority branches additionally cannot overwrite a higher-priority
    V20 branch.
    """
    out = base_v18.clone()
    base_free = base_v18.eq(int(free_label))

    def add(proposal: torch.Tensor | None) -> None:
        nonlocal out
        if proposal is None:
            return
        if proposal.shape != out.shape:
            raise ValueError("V20 proposal shape must match base V18")
        write = base_free & out.eq(int(free_label)) & proposal.ne(int(free_label))
        out[write] = proposal[write].to(out.dtype)

    add(dormant)
    add(birth)
    add(static_world)
    return out


def assert_zero_contribution_identity(
    base_v18: torch.Tensor,
    final_prediction: torch.Tensor,
) -> None:
    if not torch.equal(base_v18, final_prediction):
        diff = int((base_v18 != final_prediction).sum().item())
        raise AssertionError(
            f"zero-contribution V20 must equal V18 elementwise; differing voxels={diff}"
        )
