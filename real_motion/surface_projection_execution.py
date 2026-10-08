"""Ephemeral float64 projection reuse; no persistent/learned cache.

Keep the canonical reference and its geometry-cache namespace untouched.
Integer legality uses the same kernel/arithmetic; a SurfacePlan additionally
owns the unrounded static coordinates for this ONE live forecast only.
"""
from dataclasses import dataclass
import numpy as np
from .canonical_causal_repair import RepairPlan, STATIC, FREE, grid_arrays
from .source_evidence_audit import transform_points, planar_move


@dataclass
class SurfacePlan(RepairPlan):
    static_rows: np.ndarray
    static_phase: list
    static_z: list
    static_destinations: list


def map_surface_evidence(evidence, prepared, grid, *, kernels=None, executor=None):
    origin, step, shape = grid_arrays(grid)
    n = len(evidence)
    flat = np.full((n, 6), -1, np.int64)
    base = np.full((n, 6), FREE, np.uint8); fallback = base.copy()
    legal = np.zeros((n, 6, 2), bool); context = np.zeros((n, 6, 8), np.float32)
    actors = evidence.actor; dynamic = actors >= 0
    rows = [(a, np.flatnonzero(actors == a)) for a in np.unique(actors[dynamic])]
    static = np.flatnonzero(actors == STATIC)
    selection = (slice(int(static[0]),int(static[-1])+1)
                 if len(static) and int(static[-1])-int(static[0])+1==len(static) else static)
    inverse = np.linalg.inv(prepared.state['current_pose'])
    plan = SurfacePlan(flat, base, fallback, legal, context, static,
                       [None]*6, [None]*6, [None]*6)

    def horizon(h):
        points = evidence.world.copy()
        for actor, ids in rows:
            points[ids] = planar_move(points[ids], prepared.state['current'][actor]['centroid_world'],
                                      prepared.targets[h][actor], prepared.yaws[h][actor])
            context[ids, h, 6] = prepared.yaws[h][actor] / np.pi
        mapped = transform_points(points, prepared.state['world_to_future'][h])
        coordinate = (mapped - origin) / step
        cells = np.floor(coordinate).astype(np.int64)
        # Tiny GEMV/GEMM row counts can round differently. Retain the reference
        # static-subset arithmetic for that case instead of inventing a phase
        # from rounded float32 context or destination indices.
        sm = (transform_points(evidence.world[static], prepared.state['world_to_future'][h])
              if 0 < len(static) < 4 and len(static) != n else mapped[selection])
        if 0 < len(static) < 4 and len(static) != n:
            sc = (sm-origin)/step; si=np.floor(sc).astype(np.int64)
        else:
            sc,si=coordinate[selection],cells[selection]
        # Contiguous per-horizon arrays, not three redundant [N,6,3]
        # snapshots. Do not retain the full learned/dynamic projection.
        plan.static_phase[h] = sc-si-.5
        plan.static_z[h] = sm[:,2].copy()
        good_static=((si>=0)&(si<shape)).all(1)
        sf=(si[:,0]*shape[1]+si[:,1])*shape[2]+si[:,2]
        plan.static_destinations[h]=np.where(good_static,sf,-1)
        rel = inverse @ prepared.raw['future_poses'][h]
        context[:, h, :2] = mapped[:, :2] / 40
        context[:, h, 2] = .5 * (h + 1) / 3
        context[:, h, 3:6] = (rel[0, 3] / 40, rel[1, 3] / 40,
                             np.arctan2(rel[1, 0], rel[0, 0]) / np.pi)
        if kernels is not None:
            kernels.ccr_plan(cells, actors, evidence.classes, prepared.baseline[h],
                             prepared.owners[h], prepared.fallbacks[h], h, plan, context)
            return
        good = ((cells >= 0) & (cells < shape)).all(1); ids = np.flatnonzero(good)
        q = cells[good]; at = (q[:, 0] * shape[1] + q[:, 1]) * shape[2] + q[:, 2]
        flat[ids, h] = at; labels = np.asarray(prepared.baseline[h]).ravel()[at]
        owner = np.asarray(prepared.owners[h]).ravel()[at]
        cls = evidence.classes[ids]; actor = actors[ids]; dyn = dynamic[ids]
        owns = np.where(dyn, owner == actor, (owner < 0) & (labels == cls))
        restored = np.where(dyn, np.asarray(prepared.fallbacks[h]).ravel()[at], FREE)
        base[ids, h] = labels; fallback[ids, h] = np.where(owns, restored, labels)
        legal[ids, h, 0] = labels == FREE
        legal[ids, h, 1] = owns & (labels == cls) & (restored != labels)
        context[ids, h, 7] = np.where(dyn, owner == actor, owner < 0)
        st = ids[~dyn]
        if len(st):
            left = np.unique(flat[st[evidence.classes[st] == 11], h])
            right = np.unique(flat[st[evidence.classes[st] == 13], h])
            conflict = np.intersect1d(left, right, assume_unique=True)
            legal[st[np.isin(flat[st, h], conflict)], h, 0] = False
    if executor is None or n < 16384:
        for h in range(6): horizon(h)
    else:
        list(executor.map(horizon, range(6)))
    return plan


class SurfaceMapExecution:
    """Own no resources: wrap the existing bounded canonical CPU executor."""
    def __init__(self, reference):
        self.reference = reference
        self.kernels, self.pool = reference.kernels, reference.pool
    def build(self, prepared, grid): return self.reference.build(prepared, grid)
    def map(self, evidence, prepared, grid):
        return map_surface_evidence(evidence, prepared, grid, kernels=self.kernels, executor=self.pool)
    def close(self): self.reference.close()
