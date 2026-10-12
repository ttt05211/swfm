"""History-only geometry views, NOT persistent learned features."""
from types import SimpleNamespace
import numpy as np
from real_motion.canonical_causal_repair import CanonicalEvidence, grid_arrays, FEATURE_DIM
from real_motion.source_evidence_audit import transform_points
from real_motion.surface_canonical_repair import SurfaceAtlas
from tools.joint_occ_experiments.data import HistoryInputs, history_only_payload


def normalized_xy(points, grid):
    origin, step, shape = grid_arrays(grid)
    # Tensor fields use [C,X,Y], whereas grid_sample takes [column(Y),row(X)].
    q = 2*(np.asarray(points)[..., :2]-origin[:2])/(step[:2]*shape[:2])-1
    return q[..., ::-1].copy().astype(np.float32)


def compact_geometry(support, state, grid):
    """Same support/order/metric points, without the old 27 descriptors."""
    origin, step, _ = grid_arrays(grid)
    world, actors, classes, presence = [], [], [], []
    cursor = 0
    for layout in support.layouts:
        flags = np.asarray(layout['flags'], np.uint8)
        if int(layout['start']) != cursor or int(layout['stop']) != cursor+len(flags):
            raise RuntimeError('noncontiguous canonical/source order')
        cursor += len(flags)
        if 'static_world' in layout:
            points = np.array(layout['static_world'], copy=True)
        else:
            at = layout.get('at')
            if at is None:
                at = np.stack(np.unravel_index(layout['keys'], tuple(layout['shape'])), 1)+layout['lo']
            points = transform_points(origin+(np.asarray(at)+.5)*step, state['current_pose'])
            last = np.asarray(layout['last']); real = last >= 0
            points[real] = np.asarray(layout['points'])[last[real]]
        world.append(points)
        actors.append(np.full(len(flags), int(layout['actor']), np.int32))
        classes.append(np.full(len(flags), int(layout['cls']), np.uint8))
        presence.append(((flags[:, None] >> np.arange(4)) & 1).astype(bool))
    cat = lambda rows, shape, dtype: np.concatenate(rows).astype(dtype, copy=False) if rows else np.empty(shape, dtype)
    n = cursor
    # Zero ABI padding only: neither calculated nor consumed by the new head.
    return CanonicalEvidence(np.zeros((n, FEATURE_DIM+12), np.float32),
        np.full((n, 4), 18, np.uint8), cat(actors, (0,), np.int32),
        cat(classes, (0,), np.uint8), cat(world, (0, 3), np.float64),
        cat(presence, (0, 4), bool), {**support.audit, 'legacy_27_descriptors_used': False})


def history_reads(raw, causal, evidence, grid, *, hw=10, patch_size=16.):
    """Source crops in t0-oriented axes; point reads inverse-register each frame."""
    if len(raw['history_occ']) != 4 or len(raw['history_poses']) != 4:
        raise ValueError('exact FOUR real histories required')
    state, registrations = causal['state'], causal['registrations']
    origin, step, shape = grid_arrays(grid)
    poses = np.asarray(raw['history_poses']); inverses = np.linalg.inv(poses)
    current_inverse = np.linalg.inv(state['current_pose'])
    sources, points = len(state['current']), len(evidence)
    if len(registrations) != sources: raise RuntimeError('source/registration order differs')
    crop = np.zeros((sources, 4, hw, hw, 2), np.float32)
    mask = np.zeros((sources, 4, hw, hw), np.float32)
    center = np.zeros((sources, 4, 2), np.float32)
    valid = np.zeros((sources, 4), bool)
    axis = ((np.arange(hw)+.5)/hw-.5)*patch_size
    offsets = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1)
    offsets = np.concatenate((offsets, np.zeros((*offsets.shape[:2], 1))), -1)
    for actor in range(sources):
        for frame, registration in enumerate(registrations[actor]):
            if registration is None: continue
            ijk = np.asarray(registration[1], np.int64)
            if not len(ijk): continue
            metric = transform_points(origin+(ijk+.5)*step, poses[frame])
            centroid = metric.mean(0)
            center[actor, frame] = transform_points(centroid[None], current_inverse)[0, :2]
            valid[actor, frame] = True
            query = offsets @ np.asarray(state['current_pose'])[:3, :3].T + centroid
            native = transform_points(query.reshape(-1, 3), inverses[frame]).reshape(hw, hw, 3)
            crop[actor, frame] = normalized_xy(native, grid)
            # Rasterize actual component support into the crop, retaining tiny
            # objects even when their voxel does not contain a sampling centre.
            local = (metric-centroid) @ np.asarray(state['current_pose'])[:3, :3]
            cells = np.floor((local[:, :2]/patch_size+.5)*hw).astype(np.int64)
            inside = ((cells >= 0) & (cells < hw)).all(1)
            mask[actor, frame, cells[inside, 0], cells[inside, 1]] = 1
    point_xy = np.zeros((points, 4, 2), np.float32)
    point_z = np.zeros((points, 4), np.float32)
    point_valid = np.zeros((points, 4), bool)
    for actor in np.unique(evidence.actor):
        ids = np.flatnonzero(evidence.actor == actor)
        for frame in range(4):
            registration = None if actor < 0 else registrations[int(actor)][frame]
            if actor >= 0 and registration is None: continue
            reg = np.eye(4) if actor < 0 else registration[0]
            native = transform_points(evidence.world[ids], inverses[frame] @ np.linalg.inv(reg))
            cells = np.floor((native-origin)/step).astype(np.int64)
            good = ((cells >= 0) & (cells < shape)).all(1)
            observed = np.zeros(len(ids), bool)
            observed[good] = np.asarray(raw['history_observed'][frame], bool)[tuple(cells[good].T)]
            label = np.asarray(raw['history_occ'][frame])[tuple(cells[good].T)]
            observed[good] &= (label >= 0) & (label < 18)
            point_xy[ids, frame] = normalized_xy(native, grid)
            point_z[ids, frame] = native[:, 2]/8
            point_valid[ids, frame] = good & observed
    return dict(source_grid=crop, source_mask=mask, source_center=center, source_valid=valid,
                point_grid=point_xy, point_z=point_z, point_valid=point_valid)


class SharedHistoryInputs(HistoryInputs):
    """Reuse bounded single-flight geometry cache; no raw or learned feature bank."""
    def build(self, raw):
        causal = history_only_payload(raw['_column_causal_preparation'])
        historical = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
        prep = SimpleNamespace(raw=historical, state=causal['state'], registrations=causal['registrations'])
        support = causal['support']
        evidence = compact_geometry(support, causal['state'], self.provider.pcfg.grid)
        atlas = SurfaceAtlas.from_compact(support, prep, self.provider.pcfg.grid)
        atlas.query_workers = 1
        evidence.features[:, FEATURE_DIM:] = atlas.describe(evidence)
        joint = getattr(self.provider, 'joint', None)
        hw = joint.transport.config.tube_hw//2 if joint is not None else 10
        causal['shared_reads'] = history_reads(historical, causal, evidence, self.provider.pcfg.grid, hw=hw)
        return causal, evidence
