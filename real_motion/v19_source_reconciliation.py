"""Source-memory reconciliation for V19.

V18's detected source state remains authoritative whenever a source is present
in the current block state. Memory is used for identity/lifecycle continuity
and only contributes geometry when a remembered source is not re-detected.

This is intentionally tracking-by-detection, not replacement of detection:

    detected current sources  -> exact frozen V18 path
    matched memory            -> identity/lifecycle bookkeeping only
    new detected source       -> exact frozen V18 path + new memory identity
    unmatched memory          -> optional low-confidence add-only recovery

No GT identity or future annotation is used.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

from .v19_scene_memory import SourceTrack


@dataclass(frozen=True)
class SourceReconciliationConfig:
    """Deterministic source-memory reconciliation hyperparameters."""

    max_center_distance_m: float = 4.0
    max_memory_age_s: float = 6.0
    confidence_tau_s: float = 3.0
    min_memory_confidence: float = 0.15

    def __post_init__(self):
        if self.max_center_distance_m <= 0:
            raise ValueError("max_center_distance_m must be positive")
        if self.max_memory_age_s < 0:
            raise ValueError("max_memory_age_s must be non-negative")
        if self.confidence_tau_s <= 0:
            raise ValueError("confidence_tau_s must be positive")
        if not (0.0 <= self.min_memory_confidence <= 1.0):
            raise ValueError("min_memory_confidence must be in [0,1]")


@dataclass(frozen=True)
class SourceMatch:
    detected_index: int
    memory_index: int
    class_id: int
    center_distance_m: float


@dataclass(frozen=True)
class SourceReconciliationResult:
    matches: tuple[SourceMatch, ...]
    unmatched_detected: tuple[int, ...]
    unmatched_memory: tuple[int, ...]

    @property
    def num_detected(self) -> int:
        return len(self.matches) + len(self.unmatched_detected)

    @property
    def num_memory(self) -> int:
        return len(self.matches) + len(self.unmatched_memory)


@dataclass(frozen=True)
class MemoryOnlySelection:
    tracks: tuple[SourceTrack, ...]
    memory_indices: tuple[int, ...]
    dropped_by_age: tuple[int, ...]
    dropped_by_confidence: tuple[int, ...]
    effective_confidence: tuple[float, ...]


def effective_memory_confidence(
    track: SourceTrack,
    *,
    frame_dt_s: float,
    confidence_tau_s: float,
    extra_age_s: float = 0.0,
) -> float:
    """Age-decayed memory confidence without mutating the track."""
    age = (
        float(track.last_real_observation_age_s(float(frame_dt_s)))
        + float(extra_age_s)
    )
    base = float(np.clip(track.confidence, 0.0, 1.0))
    return float(base * math.exp(-age / float(confidence_tau_s)))


def reconcile_detected_sources(
    detected_components: Sequence[Mapping],
    memory_tracks: Sequence[SourceTrack],
    *,
    frame_dt_s: float,
    config: SourceReconciliationConfig = SourceReconciliationConfig(),
) -> SourceReconciliationResult:
    """Same-class nearest one-to-one reconciliation at one block anchor.

    Detection is authoritative. The routine returns association metadata only;
    it never changes detected component order or any frozen V18 tensor.
    """
    candidates = []
    for di, det in enumerate(detected_components):
        dc = np.asarray(det["centroid_world"], dtype=np.float64)
        if dc.shape != (3,):
            raise ValueError("detected centroid_world must be [3]")
        cid = int(det["class_id"])
        for mi, mem in enumerate(memory_tracks):
            if int(mem.class_id) != cid:
                continue
            mc = np.asarray(
                mem.anchor_center_world(float(frame_dt_s)), dtype=np.float64
            )
            d = float(np.linalg.norm(dc[:2] - mc[:2]))
            if d <= float(config.max_center_distance_m):
                candidates.append(
                    (d, int(di), int(mi), int(mem.track_id), cid)
                )

    # Fully deterministic: nearest first, then detected order, stable track id,
    # then memory list position.
    candidates.sort(key=lambda x: (x[0], x[1], x[3], x[2]))
    used_d, used_m = set(), set()
    matches = []
    for dist, di, mi, _, cid in candidates:
        if di in used_d or mi in used_m:
            continue
        used_d.add(di)
        used_m.add(mi)
        matches.append(
            SourceMatch(
                detected_index=di,
                memory_index=mi,
                class_id=cid,
                center_distance_m=float(dist),
            )
        )

    # Store matches in frozen detected-source order for easy auditing.
    matches.sort(key=lambda x: x.detected_index)
    unmatched_d = tuple(
        i for i in range(len(detected_components)) if i not in used_d
    )
    unmatched_m = tuple(
        i for i in range(len(memory_tracks)) if i not in used_m
    )
    return SourceReconciliationResult(
        matches=tuple(matches),
        unmatched_detected=unmatched_d,
        unmatched_memory=unmatched_m,
    )


def select_memory_only_tracks(
    memory_tracks: Sequence[SourceTrack],
    reconciliation: SourceReconciliationResult,
    *,
    frame_dt_s: float,
    config: SourceReconciliationConfig = SourceReconciliationConfig(),
) -> MemoryOnlySelection:
    """Select unmatched memory tracks that survive causal age/confidence gates.

    Returned tracks are copies with detected_at_anchor_override=False and
    confidence replaced by the age-decayed value used by the recovery branch.
    Matched memory is never allowed to alter the detected V18 path.
    """
    selected, selected_idx, selected_conf = [], [], []
    dropped_age, dropped_conf = [], []
    for mi in reconciliation.unmatched_memory:
        tr = memory_tracks[int(mi)]
        age = float(tr.last_real_observation_age_s(float(frame_dt_s)))
        if age > float(config.max_memory_age_s) + 1e-8:
            dropped_age.append(int(mi))
            continue
        conf = effective_memory_confidence(
            tr,
            frame_dt_s=float(frame_dt_s),
            confidence_tau_s=float(config.confidence_tau_s),
        )
        if conf < float(config.min_memory_confidence):
            dropped_conf.append(int(mi))
            continue
        selected.append(
            replace(
                tr,
                confidence=float(conf),
                detected_at_anchor_override=False,
                current_component_index=None,
            )
        )
        selected_idx.append(int(mi))
        selected_conf.append(float(conf))
    return MemoryOnlySelection(
        tracks=tuple(selected),
        memory_indices=tuple(selected_idx),
        dropped_by_age=tuple(dropped_age),
        dropped_by_confidence=tuple(dropped_conf),
        effective_confidence=tuple(selected_conf),
    )



@dataclass(frozen=True)
class DetectedTrackAssignment:
    """Stable identities for detected sources after reconciliation."""

    track_ids: tuple[int, ...]
    matched_memory_index: tuple[int | None, ...]
    next_track_id: int


def assign_detected_track_ids(
    reconciliation: SourceReconciliationResult,
    memory_tracks: Sequence[SourceTrack],
    *,
    num_detected: int,
    next_track_id: int | None = None,
) -> DetectedTrackAssignment:
    """Preserve matched memory IDs and allocate IDs only to new detections.

    This closes the lifecycle loop needed for multi-block rollout without
    changing detected-source ordering or prediction tensors.
    """
    n = int(num_detected)
    if n < 0:
        raise ValueError("num_detected must be non-negative")
    by_detected: list[int | None] = [None] * n
    memory_index: list[int | None] = [None] * n
    for match in reconciliation.matches:
        di = int(match.detected_index)
        mi = int(match.memory_index)
        if di < 0 or di >= n:
            raise ValueError("reconciliation detected index out of range")
        if mi < 0 or mi >= len(memory_tracks):
            raise ValueError("reconciliation memory index out of range")
        if by_detected[di] is not None:
            raise ValueError("duplicate detected assignment")
        by_detected[di] = int(memory_tracks[mi].track_id)
        memory_index[di] = mi

    existing = [int(tr.track_id) for tr in memory_tracks]
    fresh = max(existing, default=-1) + 1
    if next_track_id is not None:
        fresh = max(fresh, int(next_track_id))

    for di in range(n):
        if by_detected[di] is None:
            by_detected[di] = int(fresh)
            fresh += 1

    ids = tuple(int(x) for x in by_detected)
    if len(set(ids)) != len(ids):
        raise RuntimeError("reconciled detected track IDs are not unique")
    return DetectedTrackAssignment(
        track_ids=ids,
        matched_memory_index=tuple(memory_index),
        next_track_id=int(fresh),
    )

def reconciliation_summary(
    result: SourceReconciliationResult,
    selection: MemoryOnlySelection | None = None,
) -> dict[str, float | int]:
    distances = [float(x.center_distance_m) for x in result.matches]
    out: dict[str, float | int] = {
        "detected_sources": int(result.num_detected),
        "memory_sources": int(result.num_memory),
        "matched": int(len(result.matches)),
        "unmatched_detected": int(len(result.unmatched_detected)),
        "unmatched_memory": int(len(result.unmatched_memory)),
        "mean_match_distance_m": (
            float(np.mean(distances)) if distances else float("nan")
        ),
        "max_match_distance_m": (
            float(np.max(distances)) if distances else float("nan")
        ),
    }
    if selection is not None:
        out.update(
            {
                "selected_memory_only": int(len(selection.tracks)),
                "dropped_memory_age": int(len(selection.dropped_by_age)),
                "dropped_memory_confidence": int(
                    len(selection.dropped_by_confidence)
                ),
                "mean_selected_confidence": (
                    float(np.mean(selection.effective_confidence))
                    if selection.effective_confidence
                    else float("nan")
                ),
            }
        )
    return out
