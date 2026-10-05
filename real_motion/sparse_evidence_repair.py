"""Experimental evidence-only repair; NOT installed in the Local checkpoint path.

Four causally registered observations -> sparse canonical memory -> tiny heads.
The renderer is an ADD-only diagnostic, not an exact replacement for the old
KEEP/ADD/REMOVE compositor. No future labels enter memory, heads or rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch
from torch import nn

from .source_evidence_audit import planar_move, raster_flat

PROTOCOL = 'synthetic_sparse_evidence_repair_v1'
FREE = 17
STATIC = -2
POINT_DIM = 12  # relative XYZ, four presence, four visibility, last-seen age


@dataclass(frozen=True)
class CanonicalObservation:
    occupied: np.ndarray  # integer rows [actor, class, x, y, z]
    visible: np.ndarray   # includes occupied AND observed-free cells


@dataclass(frozen=True)
class EvidenceMemory:
    keys: np.ndarray
    presence: np.ndarray
    visibility: np.ndarray
    ambiguous_voxels: int = 0

    def __len__(self):
        return len(self.keys)

    @cached_property
    def base_features(self):
        # XYZ are canonical cells, not future positions or GT box coordinates.
        age = np.argmax(self.presence[:, ::-1], axis=1).astype(np.float32) / 3
        # Explicit candidate halos have no occupied observation. They must not
        # accidentally appear "last observed at t0" because argmax(all-zero)=0.
        age[~self.presence.any(1)] = 1.
        return np.concatenate((self.keys[:, 2:].astype(np.float32) / 16,
            self.presence.astype(np.float32), self.visibility.astype(np.float32), age[:, None]), axis=1)

    @cached_property
    def neighbor_features(self):
        # Six face neighbors, same actor AND class, all four observations.
        # Fixed causal integer lookup ONCE, not a learned/global scene backbone.
        rows = np.repeat(self.keys, 6, axis=0).copy()
        delta = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
        rows[:, 2:] += np.tile(delta, (len(self),1))
        ids, valid = _locate(self.keys, rows)
        values = np.zeros((len(rows),4), np.float32)
        values[valid] = self.presence[ids[valid]]
        counts = values.reshape(len(self),6,4).mean(1)
        return np.concatenate((self.base_features, counts),axis=1)

    def features(self, *, neighborhood=False):
        return self.neighbor_features if neighborhood else self.base_features

    def tensors(self, device='cpu', *, neighborhood=False):
        return dict(features=torch.as_tensor(self.features(neighborhood=neighborhood), device=device),
            actor=torch.as_tensor(self.keys[:, 0], device=device),
            classes=torch.as_tensor(self.keys[:, 1], device=device))


def _rows(value):
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[1] != 5 or a.dtype.kind not in 'iu':
        raise ValueError('canonical rows must be integer [actor,class,x,y,z]')
    if np.any((a[:, 0] < 0) & (a[:, 0] != STATIC)) or np.any((a[:, 1] < 0) | (a[:, 1] >= FREE)):
        raise ValueError('invalid causal actor or inherited class')
    return np.asarray(a, np.int64)


def _key_view(rows):
    return np.ascontiguousarray(rows, dtype=np.int64).view(
        np.dtype([(f'k{i}',np.int64) for i in range(5)])).reshape(-1)


def _locate(keys, rows):
    """Vectorized full signed-integer row lookup, no unsafe packed-key overflow."""
    k, q = _key_view(keys), _key_view(rows)
    index = np.searchsorted(k, q)
    valid = index < len(k)
    valid[valid] &= k[index[valid]] == q[valid]
    return index, valid


def build_memory(observations):
    """Accept exactly four REGISTERED frames; never infer registration from GT.

    Keys keep actor, inherited class and full Z. An unknown/free-only cell does
    not become a repair candidate. Visibility distinguishes absence from unknown.
    """
    if len(observations) != 4:
        raise ValueError('strict four-history evidence required')
    occupied = [_rows(f.occupied) for f in observations]
    visible = [_rows(f.visible) for f in observations]
    keys = np.unique(np.concatenate(occupied, axis=0), axis=0)
    for actor in np.unique(keys[:, 0]):
        if actor >= 0 and len(np.unique(keys[keys[:, 0] == actor, 1])) != 1:
            raise ValueError('dynamic source class changed across registered observations')
    # Static semantics may disagree across histories. Do not choose the larger
    # class ID or duplicate a cell with two inherited labels; fail closed.
    _, inverse, count = np.unique(keys[:, [0, 2, 3, 4]], axis=0,
        return_inverse=True, return_counts=True)
    ambiguous = int(np.sum(count > 1))
    keys = keys[count[inverse] == 1]
    presence = np.zeros((len(keys), 4), bool)
    visibility = presence.copy()
    for t, (occ, vis) in enumerate(zip(occupied, visible)):
        _, legal = _locate(np.unique(vis, axis=0), occ)
        if not legal.all():
            raise ValueError('occupied evidence must be observed, not unknown')
        indices, valid = _locate(keys, occ)
        presence[indices[valid],t] = True
        indices, valid = _locate(keys, vis)
        visibility[indices[valid],t] = True
    return EvidenceMemory(keys, presence, visibility, ambiguous)


class SparseRepairHead(nn.Module):
    """One shared point encoder; choices expose shape/time expressivity costs.

    once: one canonical score transported six times (rigid-shape assumption).
    actor_gate: one point score + six scalar actor confidence corrections.
    cached_future: point encoding ONCE + six tiny point/time interaction heads.
    repeated_future: same weights/math as cached_future, encodes points six times.
    local_consensus: cached_future with four causal face-neighbor consensus bits.

    Static points use zero source context, but keep class/XYZ/presence/visibility.
    The heads all preserve a live gradient path to existing V18 source latents.
    """
    MODES = ('once', 'actor_gate', 'cached_future', 'repeated_future', 'local_consensus')

    def __init__(self, mode='cached_future', *, source_dim=128, width=32):
        super().__init__()
        if mode not in self.MODES or source_dim < 1 or width < 4:
            raise ValueError('invalid sparse repair architecture')
        self.mode, self.source_dim, self.width = mode, source_dim, width
        self.neighborhood = mode == 'local_consensus'
        self.point_dim = POINT_DIM + 4*self.neighborhood
        self.classes = nn.Embedding(17, 8)
        self.source = nn.Linear(source_dim, width, bias=False)
        self.point = nn.Sequential(nn.Linear(self.point_dim + 8, width), nn.SiLU(), nn.Linear(width, width))
        self.future = nn.Linear(source_dim, width, bias=False)
        self.time = nn.Linear(2, width)
        self.score = nn.Linear(width, 1)
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(width, 1))
        # At initialization all scores = -4: exactly no ADD at threshold 0.5.
        nn.init.zeros_(self.score.weight)
        nn.init.constant_(self.score.bias, -4.)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _context(self, actor, source_context, future_queries):
        if source_context.ndim != 2 or source_context.shape[1] != self.source_dim:
            raise ValueError('source history context mismatch')
        if future_queries.shape != (len(source_context), 6, self.source_dim):
            raise ValueError('six live future source queries required')
        # Index zero is the static zero context. Dynamic actor i maps to i+1.
        index = torch.where(actor >= 0, actor + 1, 0).long()
        zeros = source_context.new_zeros((1, self.width))
        c = torch.cat((zeros, self.source(source_context)), dim=0)
        if self.mode == 'once':
            return index, c, None
        f = torch.cat((zeros[:, None].expand(-1, 6, -1), self.future(future_queries)), dim=0)
        h = source_context.new_tensor([.5, 1., 1.5, 2., 2.5, 3.])
        f = f + self.time(torch.stack((h / 3, (h / 3).square()), dim=-1))[None]
        return index, c, f

    def encode(self, features, classes):
        return self.point(torch.cat((features, self.classes(classes.long())), dim=-1))

    def forward(self, features, actor, classes, source_context, future_queries):
        if features.shape != (len(actor), self.point_dim) or classes.shape != actor.shape:
            raise ValueError('canonical feature shape mismatch')
        index, c, f = self._context(actor, source_context, future_queries)
        if self.mode == 'repeated_future':
            # Same point encoder and same point/future fusion as cached_future.
            return torch.cat([self.score(torch.nn.functional.silu(
                self.encode(features, classes) + c[index] + f[index, h])) for h in range(6)], dim=1)
        p = self.encode(features, classes) + c[index]
        if self.mode == 'once':
            return self.score(torch.nn.functional.silu(p)).expand(-1, 6)
        if self.mode == 'actor_gate':
            return self.score(torch.nn.functional.silu(p)) + self.gate(f)[index, :, 0]
        return self.score(torch.nn.functional.silu(p[:, None] + f[index])).squeeze(-1)


def render_add_only(memory, probabilities, baseline, source_centers, future_centers, yaw,
                    world_to_future, origin, step, *, exists=None, threshold=.5):
    """Reference CPU renderer, inherited classes, original occupied cells protected.

    static first, then ascending source order (last source wins ADD collisions).
    Z is never translated by source motion, matching the frozen planar renderer.
    No historical/unknown point invention, deletion or overwrite of V18 cells.
    """
    base = np.asarray(baseline)
    p = np.asarray(probabilities)
    centers = np.asarray(source_centers)
    future = np.asarray(future_centers)
    nsource = len(centers)
    if (base.ndim != 4 or base.shape[0] != 6 or p.shape != (len(memory), 6)
            or centers.shape != (nsource, 3) or future.shape != (6, nsource, 3)
            or np.shape(yaw) != (6, nsource) or np.shape(world_to_future) != (6, 4, 4)
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not 0 <= threshold <= 1 or np.any((base < 0) | (base > FREE))):
        raise ValueError('invalid repair renderer inputs')
    if (np.shape(origin) != (3,) or np.shape(step) != (3,) or np.any(np.asarray(step) <= 0)
            or any(not np.isfinite(v).all() for v in (centers, future, yaw, world_to_future, origin, step))):
        raise ValueError('nonfinite repair geometry or invalid voxel size')
    exists = np.ones((6, nsource), bool) if exists is None else np.asarray(exists, bool)
    if exists.shape != (6, nsource) or np.any(memory.keys[:, 0] >= nsource):
        raise ValueError('actor/existence mismatch')
    out, oob, protected, ambiguous = base.copy(), 0, 0, 0
    # Current evidence is already carried by V18; this branch cannot relabel it.
    eligible = ~memory.presence[:, -1]
    for actor in np.unique(memory.keys[:, 0]):
        ids = np.flatnonzero((memory.keys[:, 0] == actor) & eligible)
        for h in range(6):
            if actor >= 0 and not exists[h, actor]:
                continue
            selected = ids[p[ids, h] >= threshold]
            cells = memory.keys[selected, 2:]
            pts = (cells.astype(np.float64) + .5) * np.asarray(step)
            if actor >= 0:
                pts += centers[actor]
                pts = planar_move(pts, centers[actor], future[h, actor], yaw[h, actor])
            # Do not lose class alignment by deduplicating before class lookup.
            rasterized = []
            for cls in np.unique(memory.keys[selected, 1]):
                take = memory.keys[selected, 1] == cls
                flat, missed = raster_flat(pts[take], world_to_future[h], origin, step, base.shape[1:])
                oob += missed
                rasterized.append((int(cls),flat))
            # Ego rotation/pitch can collapse two differently labelled static
            # cells into one voxel. Do not pick the larger inherited class ID.
            conflict = np.empty(0,np.int64)
            if actor == STATIC and len(rasterized) > 1:
                flat, count = np.unique(np.concatenate([f for _,f in rasterized]),return_counts=True)
                conflict = flat[count>1]; ambiguous += len(conflict)
            for cls, flat in rasterized:
                if len(conflict):
                    flat = flat[~np.isin(flat,conflict)]
                legal = base[h].reshape(-1)[flat] == FREE
                protected += int((~legal).sum())
                out[h].reshape(-1)[flat[legal]] = int(cls)
    return out, dict(out_of_bounds_points=oob, protected_original_voxels=protected,
        ambiguous_static_raster_voxels=ambiguous,
        added=int(np.sum((base == FREE) & (out != FREE))))
