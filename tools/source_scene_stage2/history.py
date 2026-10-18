"""Track-valid native reads and UNREGISTERED historical memory points."""
import numpy as np
from real_motion.canonical_causal_repair import grid_arrays
from real_motion.source_evidence_audit import transform_points
from tools.joint_occ_experiments.data import HistoryInputs, PURE_STATE
from tools.history_validity.data import policy_raw


class DirectHistory(HistoryInputs):
    def build(self, raw):
        raw = policy_raw(raw, 'dense_valid')
        fixed = raw['_column_causal_preparation']; identity = raw['_shared_source_track']
        state = {k: fixed['prepared_state'][k] for k in PURE_STATE}
        origin, step, shape = grid_arrays(self.provider.pcfg.grid)
        if shape[0] % 2 or shape[1] % 2: raise ValueError('even XY grid required')
        coarse = (int(shape[0]//2), int(shape[1]//2), int(shape[2]))
        poses = np.asarray(raw['history_poses'], np.float64); inverses = np.linalg.inv(poses)
        valid = np.asarray(raw['history_observed'], bool).copy()
        labels = np.asarray(raw['history_occ']); valid &= (labels >= 0)&(labels < 18)
        points, actors, frames, cells, coverage = [], [], [], [], []
        for actor, supports in enumerate(identity.voxels):
            for frame, ijk in enumerate(supports):
                if ijk is None: continue
                ijk = np.asarray(ijk, np.int64)
                ijk = np.unique(ijk[((ijk >= 0)&(ijk < shape)).all(1)], axis=0)
                ijk = ijk[valid[frame][tuple(ijk.T)]]
                if not len(ijk): continue
                valid[frame][tuple(ijk.T)] = False
                at = ijk.copy(); at[:, :2] //= 2
                flat = np.ravel_multi_index(at.T, coarse)
                unique, inverse, count = np.unique(flat, return_inverse=True, return_counts=True)
                world = transform_points(origin+(ijk+.5)*step, poses[frame])
                mean = np.stack([np.bincount(inverse, weights=world[:, axis])/count for axis in range(3)], 1)
                points.append(mean); actors.append(np.full(len(unique), actor, np.int64))
                frames.append(np.full(len(unique), frame, np.int64)); cells.append(unique)
                coverage.append(count.astype(np.float32)/4)
        sx, sy, sz = map(int, shape)
        cat = lambda values, empty: np.concatenate(values) if values else empty
        memory = dict(static_valid=valid.reshape(4, sx//2, 2, sy//2, 2, sz).mean((2, 4)).astype(np.float32),
            world=cat(points, np.empty((0, 3), np.float64)), actor=cat(actors, np.empty(0, np.int64)),
            frame=cat(frames, np.empty(0, np.int64)), cell=cat(cells, np.empty(0, np.int64)),
            coverage=cat(coverage, np.empty(0, np.float32)), centers_world=identity.centers.copy())
        n = len(state['current']); reference = np.zeros((n, 4, 2), np.float32)
        xy_t0 = np.zeros_like(reference)
        for actor, flags in enumerate(identity.track_valid):
            for f in np.flatnonzero(flags):
                world = identity.centers[actor, f][None]
                native = transform_points(world, inverses[f])[0]
                reference[actor, f] = (2*(native[:2]-origin[:2])/(step[:2]*shape[:2])-1)[::-1]
                xy_t0[actor, f] = transform_points(world, inverses[-1])[0, :2]
        history_to_t0 = inverses[-1][None] @ poses
        t0_to_history = inverses @ poses[-1][None]
        jacobian = t0_to_history[:, :2, :2]*(2/(step[:2]*shape[:2]))[None, :, None]
        reads = dict(reference_grid=reference, reference_xy_t0=xy_t0,
            reference_valid=identity.track_valid.copy(), t0_xy_to_native_grid=jacobian[:, ::-1].copy().astype(np.float32),
            history_to_t0=history_to_t0.astype(np.float32),
            scene_bounds=np.r_[origin[:2], origin[:2]+step[:2]*shape[:2]].astype(np.float32))
        return dict(state=state, shared_reads=reads, memory_reads=memory), None
