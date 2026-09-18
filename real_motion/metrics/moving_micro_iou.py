"""Companion micro Moving-IoU diagnostic on the frozen Moving-mIoU v2 support.

This module deliberately does *not* replace or modify Moving-mIoU v2.  It uses
exactly the same true-moving support, dynamic semantic classes, and report
horizons, but changes only the class aggregation:

    macro Moving-mIoU: mean_c IoU_c
    micro Moving-IoU:  sum_c intersection_c / sum_c union_c

The reported ``micro_IoU`` preserves the frozen temporal contract by computing
micro IoU independently at 1s/2s/3s and taking the arithmetic mean over those
three horizons. ``pooled_micro_IoU`` is also exposed as a secondary diagnostic
that pools intersections/unions across all three horizons before division.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .moving_miou_v2 import DYNAMIC_CLASS_IDS, REPORT_HORIZONS_S


PROTOCOL = "moving_micro_iou_companion_v1"


class MovingMicroIoUAccumulator:
    """Dataset-level micro IoU over dynamic classes inside a supplied support."""

    def __init__(self, dynamic_classes: Sequence[int] = DYNAMIC_CLASS_IDS):
        classes = tuple(int(c) for c in dynamic_classes)
        if classes != tuple(DYNAMIC_CLASS_IDS):
            raise ValueError(
                f"Moving micro-IoU uses frozen dynamic classes {DYNAMIC_CLASS_IDS}; "
                f"got {classes}"
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
        total_inter = int(sum(self.inter.values()))
        total_union = int(sum(self.union.values()))
        micro = (
            100.0 * float(total_inter) / float(total_union)
            if total_union > 0
            else float("nan")
        )
        return {
            "micro_IoU": micro,
            "total_intersection": total_inter,
            "total_union": total_union,
            "per_class_intersection": dict(self.inter),
            "per_class_union": dict(self.union),
        }


class MovingMicroIoUMultiHorizon:
    """Horizon-first micro Moving-IoU companion for 1s/2s/3s."""

    def __init__(
        self,
        dynamic_classes: Sequence[int] = DYNAMIC_CLASS_IDS,
        horizons_s=REPORT_HORIZONS_S,
    ):
        horizons = tuple(float(h) for h in horizons_s)
        if horizons != REPORT_HORIZONS_S:
            raise ValueError(f"frozen report horizons are {REPORT_HORIZONS_S}")
        self.horizons = REPORT_HORIZONS_S
        self.acc = {
            h: MovingMicroIoUAccumulator(dynamic_classes) for h in self.horizons
        }

    def update(self, horizon_s, pred, gt, support):
        h = float(horizon_s)
        if h not in self.acc:
            raise KeyError(
                f"horizon {h} not in frozen report horizons {self.horizons}"
            )
        self.acc[h].update(pred, gt, support)

    def compute(self):
        per_horizon = {h: self.acc[h].compute() for h in self.horizons}
        vals = [float(per_horizon[h]["micro_IoU"]) for h in self.horizons]
        horizon_first = (
            float(np.mean(vals)) if not any(np.isnan(v) for v in vals) else float("nan")
        )
        total_inter = int(
            sum(per_horizon[h]["total_intersection"] for h in self.horizons)
        )
        total_union = int(
            sum(per_horizon[h]["total_union"] for h in self.horizons)
        )
        pooled = (
            100.0 * float(total_inter) / float(total_union)
            if total_union > 0
            else float("nan")
        )
        return {
            "protocol": PROTOCOL,
            "micro_IoU": horizon_first,
            "pooled_micro_IoU": pooled,
            "per_horizon": per_horizon,
            "aggregation": "per_horizon_micro_iou_then_arithmetic_mean",
        }
