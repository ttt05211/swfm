"""Predicted source provenance -> conservative trajectory-continuity handoff.

No adapters, labels, annotation identities, GT masks or metric arguments. Keep
redetected geometry/order; do NOT resurrect missing sources or overwrite shapes.
Ambiguous split/merge provenance fails closed to the existing redetection path.
"""
from dataclasses import dataclass
import numpy as np

PROTOCOL = 'predicted_visible_owner_unique_source_trajectory_handoff_v1'


@dataclass(frozen=True)
class PredictedSourceHandoff:
    owners: np.ndarray             # visible original-source owner at 3s, -1 background
    class_ids: np.ndarray
    centers_world: np.ndarray      # [source,4,3], predicted 1.5/2/2.5/3s rigid centers
    dt_s: float = .5
    max_speed_mps: float = 25.
    history_visible: np.ndarray | None = None
    max_centroid_offset_m: float = 4.

    def __post_init__(self):
        owners = np.asarray(self.owners)
        classes = np.asarray(self.class_ids)
        centers = np.asarray(self.centers_world, np.float64)
        if owners.ndim != 3 or owners.dtype.kind not in 'iu':
            raise ValueError('owner grid must be integer XYZ')
        if classes.ndim != 1 or classes.dtype.kind not in 'iu':
            raise ValueError('source classes must be an integer vector')
        if centers.shape != (len(classes), 4, 3) or not np.isfinite(centers).all():
            raise ValueError('predicted source trajectories must be finite [N,4,3]')
        if np.any(owners < -1) or np.any(owners >= len(classes)):
            raise ValueError('owner index outside source population')
        if not np.isfinite(self.dt_s) or self.dt_s <= 0 or not np.isfinite(self.max_speed_mps) or self.max_speed_mps <= 0:
            raise ValueError('positive finite cadence/speed gate required')
        if not np.isfinite(self.max_centroid_offset_m) or self.max_centroid_offset_m <= 0:
            raise ValueError('positive finite centroid offset gate required')
        visible = np.ones((len(classes), 4), bool) if self.history_visible is None else np.asarray(self.history_visible)
        if visible.shape != (len(classes), 4) or visible.dtype.kind != 'b':
            raise ValueError('predicted visibility must be boolean [N,4]')
        # Defensive copies: downstream preparation must never modify first block.
        for name, value in (('owners', owners), ('class_ids', classes), ('centers_world', centers), ('history_visible', visible)):
            copy = value.copy(); copy.flags.writeable = False
            object.__setattr__(self, name, copy)

    def reconcile(self, current, velocities, tracks, valid):
        n = len(current)
        if tracks.shape != (n, 6, 3) or valid.shape != (n, 6):
            raise ValueError('six-slot motion ABI required')
        result_velocity = {int(i): np.asarray(v, np.float64).copy() for i, v in velocities.items()}
        result_tracks, result_valid = np.asarray(tracks, np.float64).copy(), np.asarray(valid, bool).copy()
        # Use actual visible ownership, not GT or nearest-center guesses. Any
        # source split across multiple components or component containing multiple
        # owners is ambiguous. Ignore wrong-class provenance, including edits.
        candidates, inverse = [], {}
        for i, component in enumerate(current):
            idx = np.asarray(component['voxel_indices'])
            if idx.ndim != 2 or idx.shape[1] != 3 or idx.dtype.kind not in 'iu':
                raise ValueError('invalid redetected voxel indices')
            if np.any(idx < 0) or np.any(idx >= np.asarray(self.owners.shape)):
                raise ValueError('redetected voxels outside owner grid')
            ids = np.unique(self.owners[tuple(idx.T)])
            ids = [int(j) for j in ids if j >= 0 and int(self.class_ids[j]) == int(component['class_id'])]
            candidates.append(ids)
            for j in ids: inverse.setdefault(j, []).append(i)
        audit = dict(protocol=PROTOCOL, current_sources=n, original_sources=len(self.class_ids),
            matched_sources=0, unmatched_sources=0, ambiguous_merge_sources=0,
            ambiguous_split_sources=0, rejected_speed_sources=0, rejected_visibility_sources=0,
            rejected_centroid_offset_sources=0, source_identity_pairs=[],
            velocity_correction_mps_sum=0., centroid_shape_offset_m_sum=0.,
            missing_original_sources=sum(j not in inverse for j in range(len(self.class_ids))),
            memory_only_sources_added=0)
        for i, ids in enumerate(candidates):
            if not ids: audit['unmatched_sources'] += 1; continue
            if len(ids) != 1: audit['ambiguous_merge_sources'] += 1; continue
            j = ids[0]
            if len(inverse[j]) != 1: audit['ambiguous_split_sources'] += 1; continue
            if not self.history_visible[j, -2:].all():
                audit['rejected_visibility_sources'] += 1; continue
            predicted = self.centers_world[j]
            segments = np.diff(predicted[:, :2], axis=0)/self.dt_s
            if np.any(np.linalg.norm(segments, axis=1) > self.max_speed_mps):
                audit['rejected_speed_sources'] += 1; continue
            center = np.asarray(current[i]['centroid_world'], np.float64)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError('nonfinite redetected centroid')
            # Keep the CURRENT source centroid/shape authoritative. The constant
            # shape-origin offset cancels in every displacement/velocity; do not
            # differentiate changing raster/refinement centroids frame by frame.
            offset = center-predicted[-1]
            if np.linalg.norm(offset[:2]) > self.max_centroid_offset_m:
                audit['rejected_centroid_offset_sources'] += 1; continue
            linked = predicted+offset
            velocity = (linked[-1]-linked[-2])/self.dt_s; velocity[2] = 0.
            old = result_velocity.get(i, np.zeros(3))
            audit['velocity_correction_mps_sum'] += float(np.linalg.norm((velocity-old)[:2]))
            audit['centroid_shape_offset_m_sum'] += float(np.linalg.norm(offset[:2]))
            result_velocity[i] = velocity
            result_tracks[i] = 0.; result_valid[i] = False
            result_tracks[i, -4:] = linked; result_valid[i, -4:] = self.history_visible[j]
            result_tracks[i, ~result_valid[i]] = 0.
            audit['matched_sources'] += 1
            audit['source_identity_pairs'].append([i, j])
        return result_velocity, result_tracks, result_valid, audit


def handoff_from_prepared(first, final_prediction, *, dt_s=.5, max_speed_mps=25.):
    """Whitelist predicted render outputs, not PreparedColumns labels/raw GT."""
    owners = np.asarray(first.owners[-1]).copy()
    classes = np.asarray([int(c['class_id']) for c in first.state['current']], np.int64)
    final = np.asarray(final_prediction)
    if final.shape != owners.shape: raise ValueError('prediction/owner shape mismatch')
    # Provenance cannot survive a changed semantic or removed voxel.
    known = owners >= 0
    if np.any(owners[known] >= len(classes)): raise ValueError('invalid visible source owner')
    idx = np.flatnonzero(known.reshape(-1))
    flat = owners.reshape(-1)
    wrong = final.reshape(-1)[idx] != classes[flat[idx]]
    flat[idx[wrong]] = -1
    centers = np.asarray(first.targets[-4:], np.float64).reshape(4, len(classes), 3).transpose(1, 0, 2).copy()
    # The frozen SE(2) renderer preserves world Z even when the t0 ego frame
    # has pitch/roll. Its intermediate target's Z is NOT a vertical trajectory.
    # Do not leak that unused coordinate into re-expressed history offsets.
    centers[:, :, 2] = np.asarray([c['centroid_world'][2] for c in first.state['current']])[:, None]
    visible = np.zeros((len(classes), 4), bool)
    if len(first.owners) < 4: raise ValueError('four predicted owner grids required')
    for f, grid in enumerate(first.owners[-4:]):
        ids = np.asarray(grid); valid = ids[ids >= 0]
        if np.any(valid >= len(classes)): raise ValueError('invalid historical visible owner')
        visible[np.unique(valid), f] = True
    return PredictedSourceHandoff(owners, classes, centers, dt_s, max_speed_mps, visible)
