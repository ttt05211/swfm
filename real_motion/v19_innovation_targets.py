"""V19 innovation-target contracts and diagnostic ancestry helpers.

The first innovation head is intentionally supervised only on occupancy that
cannot be delegated to a reliable causal ancestor:

* future_birth_dynamic
* source_shape_innovation
* never_seen_static

Known-ancestor failures (transport/model misses, source extraction misses and
history-static mismatches), memory-addressable content and unresolved
attribution ambiguity are excluded from positive innovation supervision.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


# Legacy V19 training target retained for reproducibility of earlier experiments.
# New work should use NOVELTY_POSITIVE_CATEGORIES: source-shape residuals have a
# reliable current-source ancestor and therefore belong to Transport refinement,
# not ancestor-free Novelty.
INNOVATION_POSITIVE_CATEGORIES = (
    "future_birth_dynamic",
    "source_shape_innovation",
    "never_seen_static",
)

NOVELTY_POSITIVE_CATEGORIES = (
    "never_seen_static",
    "future_birth_dynamic",
)

TRANSPORT_REFINEMENT_CATEGORIES = (
    "current_source_transportable_miss",
    "source_shape_innovation",
    "t0_unrepresented_dynamic",
)

MEMORY_ADDRESSABLE_CATEGORIES = (
    "history_source_recoverable",
    "history_static_recoverable",
)

KNOWN_ANCESTOR_MODEL_MISS_CATEGORIES = (
    "t0_unrepresented_dynamic",
    "current_source_transportable_miss",
    "history_static_seen_mismatch",
)

AMBIGUOUS_CATEGORIES = (
    "dynamic_other_ambiguous",
    "static_other_ambiguous",
)

DECOMPOSITION_CATEGORIES = (
    "history_source_recoverable",
    "t0_unrepresented_dynamic",
    "current_source_transportable_miss",
    "future_birth_dynamic",
    "source_shape_innovation",
    "dynamic_other_ambiguous",
    "history_static_recoverable",
    "history_static_seen_mismatch",
    "never_seen_static",
    "static_other_ambiguous",
)

DECOMPOSITION_GROUPS = {
    "core_innovation": INNOVATION_POSITIVE_CATEGORIES,
    "novelty": NOVELTY_POSITIVE_CATEGORIES,
    "transport_refinement": TRANSPORT_REFINEMENT_CATEGORIES,
    "memory_addressable": MEMORY_ADDRESSABLE_CATEGORIES,
    "known_ancestor_model_miss": KNOWN_ANCESTOR_MODEL_MISS_CATEGORIES,
    "ambiguous": AMBIGUOUS_CATEGORIES,
}


def match_future_components_many_to_one(
    components: Sequence[Mapping],
    annotations: Mapping[str, Mapping],
    *,
    max_distance_m: float,
) -> list[tuple[str | None, float]]:
    """Link future GT occupancy components to same-class annotation identities.

    This helper is diagnostic/supervision-only. It intentionally permits many
    occupancy fragments to map to one GT instance token. A one-to-one tracker
    is appropriate for deployment state, but would incorrectly mark fragmented
    future GT occupancy as innovation ambiguity.

    Returns (token, nearest_same_class_distance_m) for each component.
    token is None when the nearest same-class annotation exceeds the gate.
    """
    gate = float(max_distance_m)
    if gate <= 0:
        raise ValueError("max_distance_m must be positive")

    anns = list(annotations.values())
    rows: list[tuple[str | None, float]] = []
    for comp in components:
        cid = int(comp["class_id"])
        cc = np.asarray(comp["centroid_world"], dtype=np.float64)
        if cc.shape != (3,):
            raise ValueError("component centroid_world must be [3]")
        candidates = []
        for ann in anns:
            if int(ann["class_id"]) != cid:
                continue
            ac = np.asarray(ann["center_world"], dtype=np.float64)
            if ac.shape != (3,):
                raise ValueError("annotation center_world must be [3]")
            d = float(np.linalg.norm(cc[:2] - ac[:2]))
            candidates.append((d, str(ann["instance_token"])))
        candidates.sort(key=lambda x: (x[0], x[1]))
        if not candidates:
            rows.append((None, float("inf")))
            continue
        d, tok = candidates[0]
        rows.append((str(tok) if d <= gate else None, float(d)))
    return rows


def annotation_distance_bin(distance_m: float) -> str:
    """Coarse audit bin for unresolved dynamic occupancy components."""
    d = float(distance_m)
    if not np.isfinite(d):
        return "no_same_class_annotation"
    if d <= 4.0:
        return "le_4m"
    if d <= 6.0:
        return "4_to_6m"
    if d <= 10.0:
        return "6_to_10m"
    return "gt_10m"
