"""Dependency-light helpers for the final P0-F0.6 historical-source audit.

This audit is intentionally terminal: it only decides whether t0-outside-grid
future-moving records had any causal same-class occupancy source in earlier
history frames. It does not introduce another routing branch or model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

HISTORY_SEEN = "history_seen"
HISTORY_NEVER_SEEN = "history_never_seen"


@dataclass(frozen=True)
class HistoryFrameEvidence:
    age_s: float
    annotation_present: bool
    box_in_grid: bool
    exact_same_class_voxels: int
    margin_same_class_voxels: int

    def __post_init__(self):
        if float(self.age_s) <= 0:
            raise ValueError("history frame age_s must be positive")
        if int(self.exact_same_class_voxels) < 0 or int(self.margin_same_class_voxels) < 0:
            raise ValueError("occupancy voxel counts must be non-negative")
        if int(self.margin_same_class_voxels) < int(self.exact_same_class_voxels):
            raise ValueError("margin occupancy count cannot be smaller than exact count")
        if not self.annotation_present and (
            self.box_in_grid or self.exact_same_class_voxels or self.margin_same_class_voxels
        ):
            raise ValueError("absent annotation cannot have box/occupancy evidence")
        if not self.box_in_grid and (
            self.exact_same_class_voxels or self.margin_same_class_voxels
        ):
            raise ValueError("outside-grid box cannot contain in-grid occupancy evidence")

    @property
    def source_seen(self) -> bool:
        return bool(self.margin_same_class_voxels > 0)


@dataclass(frozen=True)
class HistoricalSourceSummary:
    category: str
    seen_frame_count: int
    annotation_frame_count: int
    in_grid_frame_count: int
    last_seen_age_s: float | None
    oldest_seen_age_s: float | None


def summarize_history_source(
    frames: Sequence[HistoryFrameEvidence],
) -> HistoricalSourceSummary:
    """Collapse past-frame evidence into the frozen seen/never-seen decision.

    A source is ``history_seen`` iff at least one *past* frame contains
    same-class occupancy inside the same GT instance box (0.5 m margin allowed).
    The function assumes the caller excludes t0.
    """
    rows = list(frames)
    if not rows:
        raise ValueError("at least one past history frame is required")
    ages = [float(r.age_s) for r in rows]
    if len(set(ages)) != len(ages):
        raise ValueError("history frame ages must be unique")

    seen = [r for r in rows if r.source_seen]
    category = HISTORY_SEEN if seen else HISTORY_NEVER_SEEN
    last_seen = min((float(r.age_s) for r in seen), default=None)
    oldest_seen = max((float(r.age_s) for r in seen), default=None)
    return HistoricalSourceSummary(
        category=category,
        seen_frame_count=len(seen),
        annotation_frame_count=sum(bool(r.annotation_present) for r in rows),
        in_grid_frame_count=sum(bool(r.box_in_grid) for r in rows),
        last_seen_age_s=last_seen,
        oldest_seen_age_s=oldest_seen,
    )
