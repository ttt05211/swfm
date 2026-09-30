"""Dependency-light geometry and counterfactuals for source evidence auditing.

No network, annotation lookup or future labels enter causal registration. GT
candidate selection is a separate, explicitly hindsight-assisted diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

PROTOCOL = "p0_f9_source_evidence_geometry_v1"


def transform_points(points, transform):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(transform, dtype=np.float64)
    return p @ t[:3, :3].T + t[:3, 3]


def planar_move(points, source_center, target_center, yaw):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    src = np.asarray(source_center, dtype=np.float64)
    dst = np.asarray(target_center, dtype=np.float64)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    r = np.array([[c, -s], [s, c]])
    out = p.copy()
    out[:, :2] = (p[:, :2] - src[:2]) @ r.T + dst[:2]
    # Match the frozen planar renderer: world Z is NOT translated.
    return out


def raster_flat(points_world, world_to_ego, origin, step, shape):
    p = transform_points(points_world, world_to_ego)
    if not np.isfinite(p).all() or not np.isfinite(step).all() or np.any(np.asarray(step) <= 0):
        raise ValueError("nonfinite points or invalid voxel size")
    idx = np.floor((p - np.asarray(origin)) / np.asarray(step)).astype(np.int64)
    valid = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    flat = np.ravel_multi_index(idx[valid].T, shape) if valid.any() else np.empty(0, np.int64)
    return np.unique(flat), int((~valid).sum())


def associate_backwards(frames, current, velocity, *, dt=0.5, speed_gate=25.0,
                        ambiguity_gap_m=0.4):
    """Same-class causal one-to-one tracks, anchored in exact t0 source order.

    No new/dormant source is invented. Ambiguous nearest matches fail closed.
    Missing frames retain the last state and may be bridged within five frames.
    """
    if not frames or dt <= 0 or speed_gate <= 0 or ambiguity_gap_m < 0:
        raise ValueError("invalid association configuration")
    if len(frames[-1]) != len(current):
        raise ValueError("last history frame must be the t0 source population")
    centers = [np.asarray(c["centroid_world"], float).copy() for c in current]
    velocities = [np.asarray(velocity.get(i, np.zeros(3)) if isinstance(velocity, dict)
                             else velocity[i], float).copy() for i in range(len(current))]
    last = [len(frames) - 1] * len(current)
    links = [[None] * len(frames) for _ in current]
    audit = {"matched": 0, "ambiguous": 0, "unmatched": 0}
    for i in range(len(current)):
        links[i][-1] = i
    for f in range(len(frames) - 2, -1, -1):
        pairs = []
        for i, src in enumerate(current):
            elapsed = (last[i] - f) * dt
            expected = centers[i] - velocities[i] * elapsed
            choices = []
            for j, comp in enumerate(frames[f]):
                if int(comp["class_id"]) != int(src["class_id"]):
                    continue
                d = float(np.linalg.norm(np.asarray(comp["centroid_world"])[:2] - expected[:2]))
                physical_distance = float(np.linalg.norm(np.asarray(comp["centroid_world"])[:2] - centers[i][:2]))
                if d <= speed_gate * elapsed and physical_distance <= speed_gate * elapsed:
                    choices.append((d, j))
            choices.sort()
            if not choices:
                audit["unmatched"] += 1
            elif len(choices) > 1 and choices[1][0] - choices[0][0] < ambiguity_gap_m:
                audit["ambiguous"] += 1
            else:
                pairs.append((choices[0][0], i, choices[0][1]))
        used = set()
        for _, i, j in sorted(pairs):
            if j in used:
                audit["unmatched"] += 1
                continue
            used.add(j)
            new = np.asarray(frames[f][j]["centroid_world"], float)
            velocities[i] = (centers[i] - new) / ((last[i] - f) * dt)
            centers[i] = new.copy()
            last[i] = f
            links[i][f] = j
            audit["matched"] += 1
    return links, audit


@dataclass
class Registration:
    points: np.ndarray
    accepted: bool
    yaw_rad: float
    median_error_m: float
    inlier_fraction: float


def register_history_shape(points, reference, *, allow_yaw=True, max_points=512,
                           iterations=5, correspondence_m=1.6, max_yaw_rad=math.pi / 4):
    """Bounded deterministic trimmed XY ICP; only <=t0 geometry is accepted.

    Registration never adjusts Z, never uses GT boxes, and fails closed if
    there are fewer than six correspondences or fewer than 50% inliers.
    """
    p = np.asarray(points, np.float64).reshape(-1, 3)
    q = np.asarray(reference, np.float64).reshape(-1, 3)
    if max_points < 6 or iterations < 1 or correspondence_m <= 0 or max_yaw_rad < 0:
        raise ValueError("invalid registration configuration")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("nonfinite registration points")
    if len(p) < 6 or len(q) < 6:
        return Registration(p.copy(), False, 0.0, float("inf"), 0.0)
    # Sort before deterministic subsampling; extraction/worker order is irrelevant.
    p = p[np.lexsort(p.T[::-1])]
    q = q[np.lexsort(q.T[::-1])]
    a = p[np.linspace(0, len(p) - 1, min(max_points, len(p)), dtype=int), :2]
    b = q[np.linspace(0, len(q) - 1, min(max_points, len(q)), dtype=int), :2]
    r = np.eye(2)
    t = b.mean(0) - a.mean(0)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(b)
    except ImportError:
        tree = None

    def nearest(x):
        if tree is not None:
            d, j = tree.query(x, k=1, workers=1)
            return d, j
        squared = np.sum((x[:, None] - b[None]) ** 2, axis=-1)
        j = squared.argmin(1)
        return np.sqrt(squared[np.arange(len(x)), j]), j

    for _ in range(iterations):
        d, j = nearest(a @ r.T + t)
        keep = d <= correspondence_m
        if int(keep.sum()) < 6:
            return Registration(p.copy(), False, 0.0, float(np.median(d)), float(keep.mean()))
        # Trim gross outliers, but retain at least six point pairs.
        ids = np.flatnonzero(keep)
        ids = ids[np.argsort(d[ids], kind="stable")[:max(6, int(len(ids) * 0.8))]]
        aa, bb = a[ids], b[j[ids]]
        if allow_yaw:
            u, _, vt = np.linalg.svd((aa - aa.mean(0)).T @ (bb - bb.mean(0)))
            r_new = vt.T @ u.T
            if np.linalg.det(r_new) < 0:
                vt[-1] *= -1
                r_new = vt.T @ u.T
            angle = math.atan2(r_new[1, 0], r_new[0, 0])
            if abs(angle) > max_yaw_rad:
                return Registration(p.copy(), False, angle, float(np.median(d)), float(keep.mean()))
            r = r_new
        t = bb.mean(0) - aa.mean(0) @ r.T
    d, _ = nearest(a @ r.T + t)
    fraction = float((d <= correspondence_m).mean())
    out = p.copy()
    out[:, :2] = p[:, :2] @ r.T + t
    return Registration(out, fraction >= 0.5, math.atan2(r[1, 0], r[0, 0]),
                        float(np.median(d)), fraction)


def protected_add_indices(baseline, indices, class_id, *, free_label=17, copy=True):
    out = np.asarray(baseline).copy() if copy else np.asarray(baseline)
    flat = out.reshape(-1)
    ids = np.asarray(indices, np.int64)
    if np.any((ids < 0) | (ids >= flat.size)):
        raise ValueError("proposal index outside grid")
    ids = ids[flat[ids] == free_label]
    flat[ids] = int(class_id)
    return out


def choose_candidate_gt_assisted(baseline, gt, candidates, class_id, *, free_label=17):
    """Choose ONE whole history observation or abstain, not individual GT voxels.

    Utility = newly semantic-correct voxels minus newly incorrect voxels.
    The score is not mIoU and sequential source interactions are not globally
    optimized: this is a GT-assisted diagnostic, NOT a mathematical ceiling.
    """
    p = np.asarray(baseline).reshape(-1)
    g = np.asarray(gt).reshape(-1)
    if np.asarray(baseline).shape != np.asarray(gt).shape:
        raise ValueError("candidate baseline/GT shapes differ")
    best, score = np.empty(0, np.int64), 0
    for candidate in candidates:
        ids = np.unique(np.asarray(candidate, np.int64))
        if np.any((ids < 0) | (ids >= p.size)):
            raise ValueError("candidate index outside grid")
        ids = ids[p[ids] == free_label]
        value = int(2 * np.count_nonzero(g[ids] == int(class_id)) - len(ids))
        if value > score:  # deterministic first-candidate tie break, including abstain
            best, score = ids, value
    return best, score


def metric_count_delta(base_counts, baseline, prediction, gt, moving, dynamic_ids, *, free_label=17):
    """Exact frozen raw-count update, including deletions and semantic relabels."""
    p0, p, g = map(np.asarray, (baseline, prediction, gt))
    m = np.asarray(moving, bool)
    if p0.shape != p.shape or p.shape != g.shape or g.shape != m.shape:
        raise ValueError("metric shapes differ")
    if free_label != 17 or any(x.size and (x.min() < 0 or x.max() > 17) for x in (p0, p, g)):
        raise ValueError("frozen labels must be 0..17")
    changed = p0 != p
    def conf(x, mask):
        code = g[mask].astype(np.int64) * 18 + x[mask].astype(np.int64)
        return np.bincount(code, minlength=324).reshape(18, 18)
    d = conf(p, changed) - conf(p0, changed)
    md = conf(p, changed & m) - conf(p0, changed & m)
    oi, ou, si, su, mi, mu = base_counts
    ids = np.asarray(dynamic_ids, dtype=int)
    return (int(oi + d[:17, :17].sum()), int(ou + d.sum() - d[17, 17]),
            np.asarray(si) + np.diag(d)[:17],
            np.asarray(su) + (d.sum(0) + d.sum(1) - np.diag(d))[:17],
            np.asarray(mi) + np.diag(md)[ids],
            np.asarray(mu) + (md.sum(0) + md.sum(1) - np.diag(md))[ids])


def edit_quality(baseline, prediction, gt, *, free_label=17):
    b, p, g = map(np.asarray, (baseline, prediction, gt))
    changed = b != p
    added = (b == free_label) & (p != free_label)
    removed = (b != free_label) & (p == free_label)
    return {"changed": int(changed.sum()), "added": int(added.sum()),
            "added_occ_tp": int((added & (g != free_label)).sum()),
            "added_semantic_tp": int((added & (p == g)).sum()),
            "removed": int(removed.sum()),
            "removed_false_occupancy": int((removed & (g == free_label)).sum()),
            "removed_true_occupancy": int((removed & (g != free_label)).sum()),
            "corrected": int((changed & (p == g)).sum()),
            "damaged": int((changed & (b == g)).sum()),
            "remaining_occupied_fn": int(((p == free_label) & (g != free_label)).sum())}


def route_diagnostic(gains):
    """Conservative resource triage, not an acceptance claim or metric tuning."""
    motion = float(gains["T0_GT_MOTION"])
    history = float(gains["HISTORY_GT_ALIGN_GT_MOTION"]) - motion
    causal = float(gains["HISTORY_CAUSAL_ALIGN_PRED_MOTION"])
    if not np.isfinite([motion, history, causal]).all():
        return "inconclusive_nonfinite_metrics"
    if history >= 0.5:
        return "probe_history_evidence_selection" if causal <= 0 else "probe_history_source_reconstruction"
    if motion >= 1.0:
        return "prioritize_observed_source_motion"
    return "stop_current_evidence_representation_or_inspect_attribution"
