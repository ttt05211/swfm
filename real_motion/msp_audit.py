"""Dependency-light helpers for auditing P0-F0 C-source attribution.

The P0-F0 C bucket means a future-moving GT instance was not associated with a
causal t0 MSP candidate. It does *not* imply a future birth: the frozen
Moving-mIoU v2 record set contains only instances common to t0 and the future
horizon. This module separates label-association failures from a genuine lack
of an occupancy candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

from .msp import MSPCandidate

C_T0_ANNOTATION_MISSING = "C0_unexpected_t0_annotation_missing"
C_NO_SAME_CLASS_CANDIDATE = "C1_no_same_class_candidate"
C_DISTANCE_GATE = "C2_candidate_outside_match_gate"
C_ONE_TO_ONE_CONFLICT = "C3_candidate_within_gate_but_unmatched"

CATEGORIES = (
    C_T0_ANNOTATION_MISSING,
    C_NO_SAME_CLASS_CANDIDATE,
    C_DISTANCE_GATE,
    C_ONE_TO_ONE_CONFLICT,
)

DISTANCE_BINS = (
    "lt_1m", "1_to_2m", "2_to_4m", "4_to_6m", "6_to_10m", "ge_10m", "no_candidate"
)


@dataclass(frozen=True)
class CSourceClassification:
    category: str
    nearest_distance_m: float | None
    nearest_candidate_index: int | None
    nearest_assigned_token: str | None
    candidate_within_gate: bool


def distance_bin(distance_m: float | None) -> str:
    if distance_m is None or not math.isfinite(float(distance_m)):
        return "no_candidate"
    d = float(distance_m)
    if d < 1.0:
        return "lt_1m"
    if d < 2.0:
        return "1_to_2m"
    if d < 4.0:
        return "2_to_4m"
    if d < 6.0:
        return "4_to_6m"
    if d < 10.0:
        return "6_to_10m"
    return "ge_10m"


def classify_unmatched_instance(
    *,
    class_id: int,
    center_xy_t0_m: np.ndarray,
    candidates: Sequence[MSPCandidate],
    candidate_tokens: Sequence[str | None],
    match_max_distance_m: float,
    t0_annotation_present: bool = True,
) -> CSourceClassification:
    """Explain why one GT instance landed in P0-F0 bucket C.

    This mirrors ``match_candidates_to_instances`` but does not use future GT.
    A candidate within the matching gate can leave the GT instance unmatched
    only if that candidate was consumed by another GT instance under the frozen
    greedy one-to-one matcher. A candidate outside the gate is only an ambiguous
    same-class candidate; it is not automatically the correct physical source.
    """
    if match_max_distance_m <= 0:
        raise ValueError("match_max_distance_m must be positive")
    if len(candidate_tokens) != len(candidates):
        raise ValueError("candidate_tokens and candidates length mismatch")
    if not t0_annotation_present:
        return CSourceClassification(
            C_T0_ANNOTATION_MISSING, None, None, None, False
        )

    center = np.asarray(center_xy_t0_m, dtype=np.float64)
    if center.shape != (2,):
        raise ValueError("center_xy_t0_m must be [2]")
    same = []
    for i, cand in enumerate(candidates):
        if int(cand.class_id) != int(class_id):
            continue
        d = float(
            np.linalg.norm(np.asarray(cand.centroid_xy_m, dtype=np.float64) - center)
        )
        same.append((d, i))
    if not same:
        return CSourceClassification(
            C_NO_SAME_CLASS_CANDIDATE, None, None, None, False
        )
    same.sort(key=lambda x: (x[0], x[1]))
    d, i = same[0]
    assigned = candidate_tokens[i]
    if d > float(match_max_distance_m):
        return CSourceClassification(
            C_DISTANCE_GATE, d, i, assigned, False
        )
    if assigned is None:
        raise RuntimeError(
            "unmatched GT has an unassigned same-class candidate inside the match gate; "
            "greedy matcher invariant is broken"
        )
    return CSourceClassification(
        C_ONE_TO_ONE_CONFLICT, d, i, assigned, True
    )


def mask_touches_xy_boundary(mask_xyz: np.ndarray) -> bool:
    m = np.asarray(mask_xyz, dtype=bool)
    if m.ndim != 3:
        raise ValueError("mask must be [X,Y,Z]")
    if not m.any():
        return False
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
