"""Sparse class-conditioned generation and ownership-constrained refinement.

No function constructing proposals accepts future occupancy or annotations.
Ground truth is restricted to action_targets and metric/calibration callers.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PROTOCOL = "p0_f9_causal_column_generation_refinement_v1"
FEATURE_PROTOCOL = "six_history_full_z_frontier_anchor_source_owned_columns_v1"
FREE, UNKNOWN = 17, 18
GENERATE, REFINE = 0, 1
KEEP, ADD, REMOVE = 0, 1, 2
CONTEXT_DIM = 12


def _valid_semantics(values):
    # Integer labels dominate the hot path; range checks are exactly equivalent
    # to membership in 0..17 but avoid allocating a full-grid isin lookup result.
    if np.asarray(values).dtype.kind in 'iu':
        return not np.any(values < 0) and not np.any(values > FREE)
    return np.isin(values, np.arange(18)).all()


@dataclass(frozen=True)
class ColumnConfig:
    width: int = 64
    heads: int = 4
    layers: int = 2
    semantic_dim: int = 8
    patch: int = 7
    z_bins: int = 16
    entry_radius_m: float = 3.2
    boundary_padding_cells: int = 1

    def validate(self):
        if (min(self.width, self.heads, self.layers, self.semantic_dim, self.z_bins) < 1
                or self.width % self.heads or self.patch != 7 or self.boundary_padding_cells != 1
                or not np.isfinite(self.entry_radius_m) or self.entry_radius_m <= 0):
            raise ValueError("invalid column architecture/support contract")


@dataclass
class ColumnPlan:
    xy: np.ndarray                 # [N,2], candidate columns in future grid
    kind: np.ndarray               # [N], GENERATE or REFINE
    actor: np.ndarray              # [N], -3 generation, -2 static, >=0 source
    classes: np.ndarray            # [N], inherited class, never predicted GT
    flat: np.ndarray               # [N,Z]
    base: np.ndarray               # [N,Z], immutable V18 labels
    fallback: np.ndarray           # [N,Z], next source/background after removal
    legal: np.ndarray              # [N,Z,3], KEEP always legal
    context: np.ndarray            # [N,12], causal geometry/time/pose
    evidence_xy: np.ndarray | None = None  # GEN reads frontier anchor, not unseen query centre

    def __post_init__(self):
        if self.evidence_xy is None: self.evidence_xy = np.asarray(self.xy).copy()

    def __len__(self): return len(self.kind)

    def subset(self, indices):
        return ColumnPlan(**{k: np.asarray(v)[indices].copy() for k, v in vars(self).items()})

    def validate(self):
        n, z = self.base.shape
        if (self.xy.shape != (n, 2) or self.evidence_xy.shape != (n, 2)
                or not np.isfinite(self.evidence_xy).all() or np.any(self.evidence_xy < 0)
                or self.flat.shape != (n, z) or self.fallback.shape != (n, z)
                or self.legal.shape != (n, z, 3) or self.context.shape != (n, CONTEXT_DIM)
                or any(getattr(self, k).shape != (n,) for k in ("kind", "actor", "classes"))
                or not self.legal[..., KEEP].all() or not np.isfinite(self.context).all()
                or np.any(self.flat < 0) or np.any((self.classes < 0) | (self.classes >= FREE))
                or np.any((self.kind == GENERATE) != (self.actor == -3))
                or np.any((self.kind == REFINE) & (self.actor < -2))
                or not np.isin(self.kind, (GENERATE, REFINE)).all()
                or not _valid_semantics(self.base) or not _valid_semantics(self.fallback)):
            raise ValueError("invalid column plan")
        if (np.any(self.legal[..., ADD] & (self.base != FREE))
                or np.any(self.legal[..., REMOVE] & ((self.kind[:, None] != REFINE)
                                                   | (self.base != self.classes[:, None])
                                                   | (self.fallback == self.base)))):
            raise ValueError("illegal source ownership/add/remove mask")
        keys = np.column_stack((self.actor, self.xy))
        # No ordering is consumed here, only equality/uniqueness. Integer row
        # byte keys avoid NumPy's costly per-field lexicographic structured sort.
        unique = (np.unique(np.ascontiguousarray(keys).view(np.dtype((np.void, keys.dtype.itemsize*3))))
                  if keys.dtype.kind in 'iu' else np.unique(keys, axis=0))
        if len(unique) != n:
            raise ValueError("duplicate actor-column query")


def action_targets(plan, future_gt):
    """Label the ACTUAL edit outcome, including restored background semantics.

    Removing class c is NOT automatically correct when GT != c: if removal
    exposes another wrong class, or destroys occupied GT, KEEP wins the tie.
    """
    plan.validate()
    gt = np.asarray(future_gt)
    if gt.size <= int(plan.flat.max(initial=-1)) or not _valid_semantics(gt):
        raise ValueError("invalid future supervision grid")
    g = gt.reshape(-1)[plan.flat]
    y = np.zeros_like(plan.base, dtype=np.int64)
    y[plan.legal[..., ADD] & (g == plan.classes[:, None])] = ADD
    y[plan.legal[..., REMOVE] & (g == plan.fallback) & (g != plan.base)] = REMOVE
    return y


def sparse_layout(plan):
    plan.validate()
    flat, first, inverse = np.unique(plan.flat, return_index=True, return_inverse=True)
    baseline = plan.base.reshape(-1)[first].copy()
    rows = inverse.reshape(plan.flat.shape)
    if not np.array_equal(baseline[rows], plan.base):
        raise ValueError("conflicting baseline labels in overlapping queries")
    return id(plan), flat, baseline, rows


def compose_sparse(plan, actions, *, enable_generation=True, enable_refine=True, layout=None):
    """Same layered semantics as full recomposition; no deletion of other owners.

    Only originally visible owners may REMOVE. Hidden sources cannot remove
    themselves simultaneously, so their stored fallback stays well-defined.
    Refinement additions preserve frozen source order; generation fills only
    still-free cells AFTER refinement. Refine may not overwrite original V18.
    """
    if layout is None: layout = sparse_layout(plan)
    if layout[0] != id(plan): raise ValueError("sparse layout belongs to a different plan")
    action = np.asarray(actions, np.int64)
    if action.shape != plan.base.shape or not np.isin(action, (KEEP, ADD, REMOVE)).all():
        raise ValueError("invalid column actions")
    if not np.take_along_axis(plan.legal, action[..., None], axis=-1).all():
        raise ValueError("attempted illegal source edit")
    _, flat, baseline, rows = layout
    out = baseline.copy()
    if enable_refine:
        remove = (plan.kind[:, None] == REFINE) & (action == REMOVE)
        rr = rows[remove]
        if len(rr) != len(np.unique(rr)):
            raise ValueError("multiple visible owners claim the same removal")
        out[rr] = plan.fallback[remove]
        for actor in np.unique(plan.actor[plan.kind == REFINE]):
            take = (plan.kind[:, None] == REFINE) & (plan.actor[:, None] == actor) & (action == ADD)
            out[rows[take]] = np.broadcast_to(plan.classes[:, None], plan.base.shape)[take]
    if enable_generation:
        add = (plan.kind[:, None] == GENERATE) & (action == ADD)
        # One generation query per XY; duplicate generation actors are rejected.
        take = add & (out[rows] == FREE)
        out[rows[take]] = np.broadcast_to(plan.classes[:, None], plan.base.shape)[take]
    return flat, baseline, out


def compose_dense(baseline, plan, actions, **kwargs):
    ids, before, after = compose_sparse(plan, actions, **kwargs)
    out = np.asarray(baseline).copy()
    if not np.array_equal(out.reshape(-1)[ids], before):
        raise ValueError("plan/V18 baseline mismatch")
    out.reshape(-1)[ids] = after
    return out


def actions_from_probabilities(plan, probability, thresholds):
    """Thresholds: generation ADD, refinement ADD, refinement REMOVE.

    None/disabled maps to infinity, not 1.0 (rounded softmax can equal 1).
    An action must also beat KEEP. Legality is imposed before softmax by model.
    """
    p = np.asarray(probability, np.float32)
    if (p.shape != (*plan.base.shape, 3) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(-1), 1, rtol=0, atol=1e-5)):
        raise ValueError("invalid calibrated action probabilities")
    if len(thresholds) != 3 or any(t is not None and (not np.isfinite(t) or not .5 <= t <= 1) for t in thresholds):
        raise ValueError("invalid fixed calibration thresholds")
    gate = np.asarray([np.inf if t is None else t for t in thresholds])
    y = np.zeros_like(plan.base, np.int64)
    add_gate = np.where(plan.kind == GENERATE, gate[0], gate[1])[:, None]
    add = plan.legal[..., ADD] & (p[..., ADD] >= add_gate) & (p[..., ADD] > p[..., KEEP])
    remove = plan.legal[..., REMOVE] & (p[..., REMOVE] >= gate[2]) & (p[..., REMOVE] > p[..., KEEP])
    y[add] = ADD; y[remove] = REMOVE
    return y


def sample_queries(plan, targets, per_kind, rng):
    """GT affects TRAIN sampling only; inverse probabilities preserve priors."""
    if per_kind < 2: raise ValueError("need >=2 TRAIN queries per kind")
    selected, weights = [], []
    # Do not let static road columns starve source-local dynamic refinement.
    populations = ((np.flatnonzero(plan.kind == GENERATE), per_kind),
                   (np.flatnonzero((plan.kind == REFINE)&(plan.actor < 0)), max(2, per_kind//2)),
                   (np.flatnonzero((plan.kind == REFINE)&(plan.actor >= 0)), max(2, per_kind//2)))
    for population, budget in populations:
        buckets = [population[(targets[population] != KEEP).any(axis=1)],
                   population[(targets[population] == KEEP).all(axis=1)]]
        for bucket in buckets:
            if not len(bucket): continue
            count = min(len(bucket), max(1, budget//2))
            ids = rng.choice(bucket, count, replace=False)
            selected.extend(ids.tolist()); weights.extend([len(bucket)/count]*count)
    return np.asarray(selected, np.int64), np.asarray(weights, np.float32)


def sparse_counts(base_counts, before, after, gt, moving, dynamic_ids):
    """Exact integer deltas for both additions AND removals/restored classes."""
    if np.array_equal(before, after): return base_counts
    from .source_evidence_audit import metric_count_delta
    return metric_count_delta(base_counts, before, after, gt, moving, dynamic_ids)


def acceptance_gate(reports):
    """Identity never counts as success; no guaranteed future non-degradation."""
    joint, gen, ref = (reports[k] for k in ("joint", "generation", "refine"))
    def nonnegative(row):
        d = row["delta_vs_v18_pp"]
        values = [d[k] for k in ("IoU", "mIoU", "MovingMacro", "MovingMicro")]
        values += [h[k] for h in d["per_horizon"].values() for k in ("IoU", "mIoU", "MovingMacro", "MovingMicro")]
        return all(v is not None and np.isfinite(v) and v >= -1e-10 for v in values)
    checks = {"generation_real_correct_additions": gen["quality"].get("added_semantic_tp", 0) > 0,
              "refine_real_correct_edits": ref["quality"].get("corrected", 0) > 0,
              "all_three_variants_nonnegative": all(nonnegative(r) for r in (gen, ref, joint)),
              "joint_mIoU_strictly_positive": joint["delta_vs_v18_pp"]["mIoU"] is not None
                  and joint["delta_vs_v18_pp"]["mIoU"] > 1e-10}
    return {**checks, "pass": all(checks.values())}
