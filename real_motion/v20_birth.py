"""V20 Birth-query targets, decoding and causal rendering."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch

from .v20_history_world import DYNAMIC_IDS, FREE_LABEL
from .v20_training import dynamic_local_to_global


def cuboid_shape_target(
    size_lwh_m: Sequence[float],
    *,
    shape_size_xyz: Sequence[int],
    voxel_size_m: float,
) -> torch.Tensor:
    """Training-only local occupancy target from the GT 3D annotation box.

    The local lattice is centered on the object annotation center and aligned
    with the object's yaw. It is intentionally class-agnostic geometry.
    """
    l, w, h = (float(x) for x in size_lwh_m)
    sx, sy, sz = (int(x) for x in shape_size_xyz)
    step = float(voxel_size_m)
    if min(l, w, h, step) <= 0 or min(sx, sy, sz) <= 0:
        raise ValueError("invalid Birth shape target geometry")
    xs = (np.arange(sx, dtype=np.float32) - (sx - 1) / 2.0) * step
    ys = (np.arange(sy, dtype=np.float32) - (sy - 1) / 2.0) * step
    zs = (np.arange(sz, dtype=np.float32) - (sz - 1) / 2.0) * step
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    occ = (
        (np.abs(xx) <= l / 2.0)
        & (np.abs(yy) <= w / 2.0)
        & (np.abs(zz) <= h / 2.0)
    )
    return torch.from_numpy(occ.astype(np.float32))


def birth_targets_from_cache_row(
    row: Mapping,
    *,
    shape_size_xyz: Sequence[int],
    shape_voxel_size_m: float,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    births = [
        r for r in row["dynamic_supervision"]
        if str(r["responsibility_name"]) == "BIRTH"
    ]
    if not births:
        return {
            "class_id": torch.empty(0, dtype=torch.long, device=device),
            "existence": torch.empty((0, 6), dtype=torch.float32, device=device),
            "trajectory_xyz_yaw": torch.empty((0, 6, 4), dtype=torch.float32, device=device),
            "shape": torch.empty(
                (0,) + tuple(int(x) for x in shape_size_xyz),
                dtype=torch.float32,
                device=device,
            ),
        }
    cls, ex, traj, shape = [], [], [], []
    for r in births:
        cls.append(int(r["class_id"]))
        ex.append(np.asarray(r["existence"], dtype=np.float32))
        traj.append(np.asarray(r["trajectory_xyz_yaw_t0"], dtype=np.float32))
        shape.append(
            cuboid_shape_target(
                r["size_lwh_m"],
                shape_size_xyz=shape_size_xyz,
                voxel_size_m=float(shape_voxel_size_m),
            )
        )
    return {
        "class_id": torch.as_tensor(cls, dtype=torch.long, device=device),
        "existence": torch.as_tensor(np.stack(ex), dtype=torch.float32, device=device),
        "trajectory_xyz_yaw": torch.as_tensor(np.stack(traj), dtype=torch.float32, device=device),
        "shape": torch.stack(shape).to(device=device),
    }


@dataclass(frozen=True)
class BirthRenderReport:
    future_semantic: np.ndarray
    selected_queries: int
    active_query_horizons: int
    rendered_voxels: int
    out_of_bounds_voxels: int


def _local_shape_points(
    active: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    idx = np.argwhere(active)
    if len(idx) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    shape = np.asarray(active.shape, dtype=np.float64)
    return (
        idx.astype(np.float64) - (shape[None] - 1.0) / 2.0
    ) * float(voxel_size_m)


def render_birth_queries(
    outputs: Mapping[str, torch.Tensor],
    *,
    future_ego_to_canonical: np.ndarray,
    native_shape_xyz: Sequence[int],
    native_origin_xyz_m: Sequence[float],
    native_voxel_size_xyz_m: Sequence[float],
    shape_voxel_size_m: float,
    existence_threshold: float = 0.5,
    shape_threshold: float = 0.5,
    free_label: int = FREE_LABEL,
) -> BirthRenderReport:
    """Render B=1 Birth queries directly into each future ego Occ3D grid.

    Query identity is persistent across all six horizons. Queries are ordered by
    class confidence; higher-confidence Birth objects win only against other
    Birth queries. The global compositor still gives V18 and Dormant priority.
    """
    cls_logits = outputs["class_logits"].detach().float().cpu()
    exist = torch.sigmoid(outputs["existence_logits"].detach().float()).cpu()
    traj = outputs["trajectory_xyz_yaw"].detach().float().cpu()
    shape = torch.sigmoid(outputs["shape_logits"].detach().float()).cpu()
    if cls_logits.shape[0] != 1:
        raise ValueError("Birth renderer currently expects B=1")
    noobj = len(DYNAMIC_IDS)
    probs = cls_logits[0].softmax(-1)
    local_cls = probs.argmax(-1)
    scores = probs.gather(1, local_cls[:, None])[:, 0]
    selected = [
        q for q in range(cls_logits.shape[1])
        if int(local_cls[q]) != noobj
    ]
    selected.sort(key=lambda q: (-float(scores[q]), int(q)))
    global_cls = (
        dynamic_local_to_global(local_cls[selected])
        if selected else torch.empty(0, dtype=torch.long)
    )

    fposes = np.asarray(future_ego_to_canonical, dtype=np.float64)
    if fposes.shape != (6, 4, 4):
        raise ValueError("future_ego_to_canonical must be [6,4,4]")
    inv_future = np.stack([np.linalg.inv(x) for x in fposes], axis=0)
    origin = np.asarray(native_origin_xyz_m, dtype=np.float64)
    step = np.asarray(native_voxel_size_xyz_m, dtype=np.float64)
    nshape = np.asarray(tuple(int(x) for x in native_shape_xyz), dtype=np.int64)
    out = np.full((6,) + tuple(nshape.tolist()), int(free_label), dtype=np.uint8)
    active_h = rendered = oob = 0

    for order_i, q in enumerate(selected):
        cid = int(global_cls[order_i])
        local_pts = _local_shape_points(
            (shape[0, q].numpy() >= float(shape_threshold)),
            float(shape_voxel_size_m),
        )
        if not len(local_pts):
            continue
        for h in range(6):
            if float(exist[0, q, h]) < float(existence_threshold):
                continue
            active_h += 1
            x, y, z, yaw = [float(v) for v in traj[0, q, h].tolist()]
            cy, sy = math.cos(yaw), math.sin(yaw)
            rot = np.asarray(
                [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            canon = local_pts @ rot.T + np.asarray([x, y, z], dtype=np.float64)[None]
            homog = np.concatenate((canon, np.ones((len(canon), 1))), axis=1)
            ego = (inv_future[h] @ homog.T).T[:, :3]
            idx = np.floor((ego - origin[None]) / step[None]).astype(np.int64)
            valid = ((idx >= 0) & (idx < nshape[None])).all(axis=1)
            oob += int((~valid).sum())
            idx = idx[valid]
            if len(idx):
                # Higher-confidence query already written wins within Birth.
                empty = out[h, idx[:, 0], idx[:, 1], idx[:, 2]] == int(free_label)
                good = idx[empty]
                out[h, good[:, 0], good[:, 1], good[:, 2]] = cid
                rendered += int(len(good))
    return BirthRenderReport(
        future_semantic=out,
        selected_queries=int(len(selected)),
        active_query_horizons=int(active_h),
        rendered_voxels=int(rendered),
        out_of_bounds_voxels=int(oob),
    )
