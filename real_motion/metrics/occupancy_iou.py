"""Binary occupancy IoU for Occ3D-style semantic occupancy grids.

Unlike semantic mIoU, this metric ignores the occupied semantic label and only
asks whether a voxel is occupied or free.  With the default Occ3D convention,
label 17 is free and every other label is occupied.

Aggregation follows the forecasting protocol used in this repository:
intersections/unions are accumulated dataset-wide inside each report horizon,
then the 1s/2s/3s IoUs are arithmetic-averaged.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .moving_miou_v2 import REPORT_HORIZONS_S


class OccupancyIoUAccumulator:
    """Dataset-level binary occupied-vs-free IoU accumulator."""

    def __init__(self, free_label: int = 17):
        self.free_label = int(free_label)
        self.intersection = 0
        self.union = 0

    def update(self, pred, gt, mask=None) -> None:
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        if pred.shape != gt.shape:
            raise ValueError("shape mismatch")
        if mask is None:
            mask_arr = np.ones_like(gt, dtype=bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != gt.shape:
                raise ValueError("mask shape mismatch")

        pred_occ = (pred != self.free_label) & mask_arr
        gt_occ = (gt != self.free_label) & mask_arr
        self.intersection += int((pred_occ & gt_occ).sum())
        self.union += int((pred_occ | gt_occ).sum())

    def compute(self) -> dict:
        value = self.intersection / self.union if self.union else float("nan")
        return {
            "IoU": 100.0 * float(value) if np.isfinite(value) else float("nan"),
            "intersection": int(self.intersection),
            "union": int(self.union),
        }


class OccupancyIoUMultiHorizon:
    """Horizon-first binary occupancy IoU with 1s/2s/3s mean reporting."""

    def __init__(
        self,
        free_label: int = 17,
        horizons_s: Sequence[float] = REPORT_HORIZONS_S,
    ):
        horizons = tuple(float(h) for h in horizons_s)
        if horizons != REPORT_HORIZONS_S:
            raise ValueError(f"report horizons must be {REPORT_HORIZONS_S}; got {horizons}")
        self.horizons = horizons
        self.acc = {h: OccupancyIoUAccumulator(free_label=free_label) for h in horizons}

    def update(self, horizon_s: float, pred, gt, mask=None) -> None:
        h = float(horizon_s)
        if h not in self.acc:
            raise KeyError(f"horizon {h} not in report horizons {self.horizons}")
        self.acc[h].update(pred, gt, mask=mask)

    def compute(self) -> dict:
        per_horizon = {h: self.acc[h].compute() for h in self.horizons}
        values = [float(per_horizon[h]["IoU"]) for h in self.horizons]
        if any(not np.isfinite(v) for v in values):
            mean = float("nan")
        else:
            mean = float(np.mean(values))
        return {"IoU": mean, "per_horizon": per_horizon}
