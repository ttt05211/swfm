"""Frozen Moving-mIoU v2 (interval_displacement_v2).

Contract:
- motion decision uses GT instance centers in WORLD XY;
- rasterization uses the same t0/th boxes transformed into the TARGET FUTURE
  ego grid;
- Occ3D arrays use official [X,Y,Z] axis order;
- support is their union with 0.5 m margin;
- only dynamic semantic classes are scored;
- mIoU is computed independently at 1s/2s/3s, then arithmetic-averaged.
"""
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

PROTOCOL = "interval_displacement_v2"
SPEED_THRESHOLD_MPS = 0.5
BOX_MARGIN_M = 0.5
REPORT_HORIZONS_S = (1.0, 2.0, 3.0)

NUSCENES_LABELS = (
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation", "free",
)
DYNAMIC_CLASS_IDS = (2, 3, 4, 5, 6, 7, 9, 10)


@dataclass(frozen=True)
class Box3D:
    token: str
    class_id: int
    center_xyz: tuple
    size_lwh: tuple
    yaw: float


@dataclass(frozen=True)
class GridSpec:
    x_min: float = -40.0
    y_min: float = -40.0
    z_min: float = -1.0
    voxel_size: tuple = (0.4, 0.4, 0.4)
    # Legacy field name; order is (X,Y,Z), matching Occ3D labels.npz.
    shape_hwd: tuple = (200, 200, 16)


def interval_speed_xy(center0_world, centerh_world, dt):
    if dt <= 0:
        raise ValueError("dt must be positive")
    return float(np.linalg.norm(
        np.asarray(centerh_world, dtype=np.float64)[:2]
        - np.asarray(center0_world, dtype=np.float64)[:2]
    ) / dt)


def is_moving_world(center0_world, centerh_world, dt,
                    threshold=SPEED_THRESHOLD_MPS):
    return interval_speed_xy(center0_world, centerh_world, dt) >= float(threshold)


def rasterize_oriented_box(box: Box3D, grid: GridSpec = GridSpec(), margin=BOX_MARGIN_M):
    """Rasterize a target-ego box into an Occ3D ``[X,Y,Z]`` mask."""
    NX, NY, NZ = grid.shape_hwd
    vx, vy, vz = grid.voxel_size
    xs = grid.x_min + (np.arange(NX) + 0.5) * vx
    ys = grid.y_min + (np.arange(NY) + 0.5) * vy
    zs = grid.z_min + (np.arange(NZ) + 0.5) * vz
    cx, cy, cz = box.center_xyz
    l, w, h = box.size_lwh
    radius = 0.5 * math.hypot(l + 2*margin, w + 2*margin)
    xi = np.where((xs >= cx-radius) & (xs <= cx+radius))[0]
    yi = np.where((ys >= cy-radius) & (ys <= cy+radius))[0]
    zi = np.where((zs >= cz-h/2-margin) & (zs <= cz+h/2+margin))[0]
    out = np.zeros((NX, NY, NZ), dtype=bool)
    if not len(xi) or not len(yi) or not len(zi):
        return out
    xx_metric, yy_metric = np.meshgrid(xs[xi], ys[yi], indexing="xy")
    dx, dy = xx_metric-cx, yy_metric-cy
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    lx = c*dx + s*dy
    ly = -s*dx + c*dy
    inside = (np.abs(lx) <= l/2+margin) & (np.abs(ly) <= w/2+margin)
    row_y, col_x = np.where(inside)
    for z in zi:
        out[xi[col_x], yi[row_y], z] = True
    return out


def moving_support_from_world_motion(
    center0_world,
    centerh_world,
    box0_in_future_ego: Box3D,
    boxh_in_future_ego: Box3D,
    dt,
    grid: GridSpec = GridSpec(),
    speed_threshold=SPEED_THRESHOLD_MPS,
    margin=BOX_MARGIN_M,
):
    """Frozen dual-box support with explicit world-vs-raster coordinate contract."""
    if box0_in_future_ego.token != boxh_in_future_ego.token:
        raise ValueError("instance token mismatch")
    if not is_moving_world(center0_world, centerh_world, dt, speed_threshold):
        return np.zeros(grid.shape_hwd, dtype=bool)
    return (
        rasterize_oriented_box(box0_in_future_ego, grid, margin)
        | rasterize_oriented_box(boxh_in_future_ego, grid, margin)
    )


# Backward-compatible helper; inputs are assumed to be in one rigid frame, so
# planar displacement magnitude is invariant. New dataset adapters should use
# moving_support_from_world_motion() instead.
def moving_support(box0_future_grid: Box3D, boxh_future_grid: Box3D,
                   dt, grid: GridSpec = GridSpec(),
                   speed_threshold=SPEED_THRESHOLD_MPS, margin=BOX_MARGIN_M):
    return moving_support_from_world_motion(
        box0_future_grid.center_xyz, boxh_future_grid.center_xyz,
        box0_future_grid, boxh_future_grid, dt, grid, speed_threshold, margin,
    )


class MovingMIoUV2Accumulator:
    def __init__(self, dynamic_classes: Sequence[int] = DYNAMIC_CLASS_IDS):
        classes = tuple(int(c) for c in dynamic_classes)
        if classes != tuple(DYNAMIC_CLASS_IDS):
            raise ValueError(
                f"Moving-mIoU v2 uses frozen dynamic classes {DYNAMIC_CLASS_IDS}; got {classes}"
            )
        self.classes = classes
        self.inter = {c: 0 for c in self.classes}
        self.union = {c: 0 for c in self.classes}

    def update(self, pred, gt, support):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        support = np.asarray(support, dtype=bool)
        if pred.shape != gt.shape or pred.shape != support.shape:
            raise ValueError("shape mismatch")
        for c in self.classes:
            p = (pred == c) & support
            g = (gt == c) & support
            self.inter[c] += int((p & g).sum())
            self.union[c] += int((p | g).sum())

    def compute(self):
        ious = {c: (self.inter[c]/self.union[c] if self.union[c] else np.nan)
                for c in self.classes}
        vals = [v for v in ious.values() if not np.isnan(v)]
        return {
            "mIoU": 100 * float(np.mean(vals)) if vals else float("nan"),
            "per_class": {c: 100*v for c, v in ious.items()},
        }


class MovingMIoUV2MultiHorizon:
    """Enforce horizon-first aggregation from the frozen contract."""
    def __init__(self, dynamic_classes: Sequence[int] = DYNAMIC_CLASS_IDS,
                 horizons_s=REPORT_HORIZONS_S):
        if tuple(float(h) for h in horizons_s) != REPORT_HORIZONS_S:
            raise ValueError(f"frozen report horizons are {REPORT_HORIZONS_S}")
        self.horizons = REPORT_HORIZONS_S
        self.acc = {h: MovingMIoUV2Accumulator(dynamic_classes) for h in self.horizons}

    def update(self, horizon_s, pred, gt, support):
        h = float(horizon_s)
        if h not in self.acc:
            raise KeyError(f"horizon {h} not in frozen report horizons {self.horizons}")
        self.acc[h].update(pred, gt, support)

    def compute(self):
        per_horizon = {h: self.acc[h].compute() for h in self.horizons}
        vals = [per_horizon[h]["mIoU"] for h in self.horizons]
        if any(np.isnan(v) for v in vals):
            return {"mIoU": float("nan"), "per_horizon": per_horizon}
        return {"mIoU": float(np.mean(vals)), "per_horizon": per_horizon}


def semantic_miou(pred, gt, classes=tuple(range(17)), mask=None):
    """Generic semantic mIoU helper; free class 17 is excluded by default."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError("shape mismatch")
    if mask is None:
        mask = np.ones_like(gt, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    vals = []
    per = {}
    for c in classes:
        p = (pred == c) & mask
        g = (gt == c) & mask
        union = int((p | g).sum())
        if union == 0:
            per[int(c)] = float("nan")
            continue
        iou = int((p & g).sum()) / union
        per[int(c)] = 100 * iou
        vals.append(iou)
    return {"mIoU": 100*float(np.mean(vals)) if vals else float("nan"), "per_class": per}


class SemanticIoUAccumulator:
    """Dataset-level semantic IoU accumulator with optional per-update mask."""
    def __init__(self, classes=tuple(range(17))):
        self.classes = tuple(int(c) for c in classes)
        self.inter = {c: 0 for c in self.classes}
        self.union = {c: 0 for c in self.classes}

    def update(self, pred, gt, mask=None):
        pred, gt = np.asarray(pred), np.asarray(gt)
        if pred.shape != gt.shape:
            raise ValueError("shape mismatch")
        if mask is None:
            mask = np.ones_like(gt, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != gt.shape:
                raise ValueError("mask shape mismatch")
        for c in self.classes:
            p = (pred == c) & mask
            g = (gt == c) & mask
            self.inter[c] += int((p & g).sum())
            self.union[c] += int((p | g).sum())

    def compute(self):
        per = {}
        vals = []
        for c in self.classes:
            if self.union[c] == 0:
                per[c] = float("nan")
            else:
                v = self.inter[c] / self.union[c]
                per[c] = 100.0 * v
                vals.append(v)
        return {"mIoU": 100.0 * float(np.mean(vals)) if vals else float("nan"),
                "per_class": per}
