"""Surface-consistent CCR: integrated evidence encoding and horizon readout.

No post-hoc score rectifier, new candidates, GT input, or six spatial encoders.
The deterministic atlas is history-only. Learned encoding and projection phase
conditioning belong inside the existing Dense Forecast timing boundary.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from .canonical_causal_repair import (
    CanonicalRepairHead, FEATURE_DIM, STATIC, grid_arrays,
)
from .source_evidence_audit import transform_points

PROTOCOL = "surface_consistent_ccr_v1"
SURFACE_DIM = 12
PHASE_DIM = 6
NEIGHBORS = 16
RADIUS = 2.5
FEATURE_NAMES = (
    "fit_valid", "relative_surface_height", "plane_residual_std",
    "height_spread", "slope_x", "slope_y", "support_fraction",
    "nearest_direct_distance", "opposite_surface_distance",
    "opposite_surface_present", "same_column_fraction", "recent_support",
)


class SurfaceAtlas:
    """Read-only, class-specific metric history surfaces, retaining all layers.

    K bounds neighbourhood DESCRIPTORS, never the candidate population. Z is
    scaled more strongly so separate stacked surfaces are not flattened into
    one BEV height. No query uses a future label, visibility, or learned pose.
    """

    def __init__(self, world, classes, presence, actors, current_pose, grid):
        self.origin, self.step, _ = grid_arrays(grid)
        self.inverse = np.linalg.inv(np.asarray(current_pose, np.float64))
        self.metric = {}
        self.trees = {}
        self.recent = {}
        self.scale = np.array([1., 1., 2.]) / self.step
        for cls in (11, 13):
            direct = (np.asarray(actors) == STATIC) & (np.asarray(classes) == cls)
            direct &= np.asarray(presence, bool).any(1)
            points = transform_points(np.asarray(world)[direct], self.inverse)
            if not np.isfinite(points).all():
                raise ValueError("nonfinite history surface")
            self.metric[cls] = points
            self.recent[cls] = np.asarray(presence, bool)[direct, -1].astype(np.float64)
            self.trees[cls] = cKDTree(points * self.scale) if len(points) else None

    @classmethod
    def from_compact(cls, support, prep, grid):
        """Reuse unchanged v2 compact cache; never mutate cached arrays."""
        points, classes, presence = [], [], []
        origin, step, _ = grid_arrays(grid)
        for layout in support.layouts:
            if int(layout['actor']) != STATIC:
                continue
            flags = np.asarray(layout['flags'], np.uint8)
            world = layout.get('static_world')
            if world is None:
                at = layout.get('at')
                if at is None:
                    at = np.stack(np.unravel_index(layout['keys'], tuple(layout['shape'])), 1)
                    at = at + np.asarray(layout['lo'])
                world = transform_points(origin + (at + .5) * step, prep.state['current_pose'])
                last = np.asarray(layout['last'])
                real = last >= 0
                world[real] = np.asarray(layout['points'])[last[real]]
            points.append(world)
            classes.append(np.full(len(flags), int(layout['cls']), np.uint8))
            presence.append(((flags[:, None] >> np.arange(4)) & 1).astype(bool))
        world = np.concatenate(points) if points else np.empty((0, 3), np.float64)
        labels = np.concatenate(classes) if classes else np.empty(0, np.uint8)
        seen = np.concatenate(presence) if presence else np.empty((0, 4), bool)
        return cls(world, labels, seen, np.full(len(world), STATIC), prep.state['current_pose'], grid)

    def describe(self, evidence):
        result = np.zeros((len(evidence), SURFACE_DIM), np.float32)
        for cls in (11, 13):
            ids = np.flatnonzero((evidence.actor == STATIC) & (evidence.classes == cls))
            tree = self.trees[cls]
            if not len(ids) or tree is None:
                continue
            query = transform_points(evidence.world[ids], self.inverse)
            scaled_query = query * self.scale
            workers = int(getattr(self, 'query_workers', 1)) if len(ids) >= 4096 else 1
            if not 1 <= workers <= 8:
                raise ValueError('bounded surface query workers must be in 1..8')
            distance, index = tree.query(scaled_query, k=NEIGHBORS,
                                         distance_upper_bound=RADIUS, workers=workers)
            valid = np.isfinite(distance)
            safe = np.minimum(index, len(self.metric[cls]) - 1)
            delta = (self.metric[cls][safe] - query[:, None]) / self.step
            delta = np.where(valid[..., None], delta, 0.)
            weight = np.where(valid, 1. / (1. + np.where(valid, distance, 0.) ** 2), 0.)
            total = weight.sum(1).clip(1e-12)
            mean = (weight[..., None] * delta).sum(1) / total[:, None]
            centered = delta - mean[:, None]
            def cov(a, b):
                return (weight * centered[..., a] * centered[..., b]).sum(1) / total
            xx, yy, xy = cov(0, 0) + 1e-3, cov(1, 1) + 1e-3, cov(0, 1)
            xz, yz = cov(0, 2), cov(1, 2)
            determinant = (xx * yy - xy * xy).clip(1e-9)
            gx = (xz * yy - yz * xy) / determinant
            gy = (yz * xx - xz * xy) / determinant
            intercept = mean[:, 2] - gx * mean[:, 0] - gy * mean[:, 1]
            error = delta[..., 2] - (intercept[:, None] + gx[:, None] * delta[..., 0]
                                     + gy[:, None] * delta[..., 1])
            rms = np.sqrt((weight * error ** 2).sum(1) / total)
            count = valid.sum(1)
            # Unknown is explicit; absence is NOT evidence of a zero-height road.
            nearest = np.where(count > 0, distance[:, 0], RADIUS) / RADIUS
            other = self.trees[13 if cls == 11 else 11]
            opposite = np.full(len(ids), np.inf)
            if other is not None:
                opposite, _ = other.query(scaled_query, k=1,
                                           distance_upper_bound=RADIUS, workers=workers)
            opposite_seen = np.isfinite(opposite)
            opposite = np.where(opposite_seen, opposite / RADIUS, 1.)
            recent = (weight * self.recent[cls][safe]).sum(1) / total
            column = valid & (np.abs(delta[..., 0]) < .55) & (np.abs(delta[..., 1]) < .55)
            values = np.column_stack((
                count >= 3, -intercept, rms, np.sqrt(np.maximum(cov(2, 2), 0.)),
                gx, gy, count / NEIGHBORS, nearest, opposite, opposite_seen,
                column.sum(1) / np.maximum(count, 1), recent,
            ))
            result[ids] = np.clip(values, -4., 4.).astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite surface descriptor")
        return result


def augment_evidence(evidence, atlas):
    if evidence.features.shape[1] != FEATURE_DIM:
        raise ValueError("surface augmentation requires unaugmented CCR features")
    return replace(evidence, features=np.concatenate((evidence.features, atlas.describe(evidence)), 1))


def augment_projection(evidence, plan, current_pose, world_to_future, grid):
    """Six cheap LIVE descriptors; never persist phases in a history cache."""
    if evidence.features.shape[1] != FEATURE_DIM + SURFACE_DIM or plan.context.shape[-1] != 8:
        raise ValueError("surface feature/context contract mismatch")
    origin, step, shape = grid_arrays(grid)
    result = np.zeros((*plan.context.shape[:2], PHASE_DIM), np.float32)
    ids = np.flatnonzero(evidence.actor == STATIC)
    if len(ids):
        dz = evidence.features[ids, FEATURE_DIM + 1].astype(np.float64) * step[2]
        z_axis = np.asarray(current_pose, np.float64)[:3, 2]
        for h, matrix in enumerate(world_to_future):
            snapshot = getattr(plan, 'static_phase', None)
            if snapshot is None:
                mapped = transform_points(evidence.world[ids], matrix)
                coordinate = (mapped - origin) / step
                cells = np.floor(coordinate).astype(np.int64)
                good = ((cells >= 0) & (cells < shape)).all(1)
                flat = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]
                expected = np.where(good, flat, -1)
                phase = coordinate - cells - .5
                mapped_z = mapped[:,2]
            else:
                if not np.array_equal(ids, plan.static_rows):
                    raise RuntimeError('ephemeral surface projection row identity changed')
                phase = snapshot[h]
                mapped_z = plan.static_z[h]
                expected = plan.static_destinations[h]
            if not np.array_equal(expected, plan.flat[ids, h]):
                raise RuntimeError("surface projection changed the canonical destination")
            height = dz * float(np.asarray(matrix)[2, :3] @ z_axis) / step[2]
            result[ids, h] = np.column_stack((phase, mapped_z / 40., height, height - phase[:, 2]))
    return replace(plan, context=np.concatenate((plan.context, result), -1))


class SurfaceCanonicalRepairHead(CanonicalRepairHead):
    """CCR with shared encoding and a replaced static conditional readout.

    Surface information enters the representation, not a correction to output
    logits. Dynamic readout is unchanged. In the frozen-motion validation only
    static input projections/readout are trained; clean joint runs may train ALL
    parameters of this same model with no teacher/residual head dependency.
    """

    add_only_weighted = True

    def __init__(self, source_dim=128, width=64):
        super().__init__(source_dim, width)
        self.surface = nn.Linear(SURFACE_DIM, width, bias=False)
        self.phase = nn.Linear(PHASE_DIM, width, bias=False)
        self.static_context = copy.deepcopy(self.context)
        self.static_readout = copy.deepcopy(self.readout)
        nn.init.zeros_(self.surface.weight)
        nn.init.zeros_(self.phase.weight)
        self.static_only_training = False

    def initialize_from(self, baseline):
        parent = set(baseline.state_dict())
        missing, unexpected = self.load_state_dict(baseline.state_dict(), strict=False)
        if unexpected or set(missing) != set(self.state_dict()) - parent:
            raise RuntimeError("invalid surface CCR initialization")
        self.static_context.load_state_dict(baseline.context.state_dict())
        self.static_readout.load_state_dict(baseline.readout.state_dict())

    def freeze_validation(self):
        self.requires_grad_(False)
        for module in (self.surface, self.phase, self.static_context, self.static_readout):
            module.requires_grad_(True)
        self.static_only_training = True

    def inference_batch(self,output,actors):
        # CPU metadata already exists before upload. Canonical rows are grouped
        # by actor, so most chunks need only ONE readout. No GPU .all()/sync,
        # candidate reorder, altered matrix batch shape, or learned cache.
        static=np.asarray(actors)<0
        role=True if static.all() else False if not static.any() else None
        return {**output,'_surface_inference_role':role}

    def encode(self, features, labels, actors, classes, output):
        if features.shape[-1] != FEATURE_DIM + SURFACE_DIM:
            raise ValueError("surface CCR requires augmented evidence")
        encoded = super().encode(features[:, :FEATURE_DIM], labels, actors, classes, output)
        hint=output.get('_surface_inference_role')
        if hint is False:
            return encoded
        geometry = self.surface(features[:, FEATURE_DIM:])
        if hint is True:
            return encoded+geometry
        return torch.where((actors < 0)[:, None], encoded + geometry, encoded)

    def decode(self, encoded, actors, context, base, fallback, legal, output):
        if context.shape[-1] != 8 + PHASE_DIM:
            raise ValueError("surface CCR requires live projection phase")
        hint=output.get('_surface_inference_role')
        if hint is False:
            return super().decode(encoded,actors,context[...,:8],base,fallback,legal,output)
        extra = torch.cat((context[..., :8], self.semantic(base.long()),
                           self.semantic(fallback.long()), legal.to(context.dtype)), -1)
        static = self.static_readout(encoded[:, None] + self.static_context(extra)
                                     + self.phase(context[..., 8:]))
        if hint is True:
            return static
        # Frozen dynamic validation does not need a backward graph through
        # unused original static logits. Clean joint training retains it.
        original = super().decode(encoded.detach() if self.static_only_training else encoded,
                                  actors, context[..., :8], base, fallback, legal, output)
        return torch.where((actors < 0)[:, None, None], static, original)

    def probabilities(self, logits, actors):
        score = torch.zeros_like(logits, dtype=torch.float32)
        score[..., 0] = logits[..., 0].float().sigmoid()
        return score
