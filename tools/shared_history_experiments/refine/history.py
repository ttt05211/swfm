"""Pure causal geometry for source reads and ALL-class history feature transport."""
import numpy as np
from real_motion.canonical_causal_repair import grid_arrays
from real_motion.source_evidence_audit import transform_points
from tools.joint_occ_experiments.data import HistoryInputs, PURE_STATE
from tools.shared_history_experiments.history import history_reads


class NoCanonicalPoints:
    actor = np.empty(0, np.int32)
    world = np.empty((0, 3), np.float64)
    def __len__(self): return 0


class ResidualHistoryInputs(HistoryInputs):
    def build(self, raw):
        fixed = raw['_column_causal_preparation']
        causal = dict(state={k: fixed['prepared_state'][k] for k in PURE_STATE},
                      registrations=fixed['registrations'])
        historical = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
        origin, step, shape = grid_arrays(self.provider.pcfg.grid)
        if shape[0] % 2 or shape[1] % 2: raise ValueError('even XY grid required by shared stride2')
        coarse = (int(shape[0]//2), int(shape[1]//2), int(shape[2]))
        valid = np.asarray(raw['history_observed'], bool).copy()
        labels = np.asarray(raw['history_occ'])
        valid &= (labels >= 0) & (labels < 18)
        points, actors, frames, cells, coverage = [], [], [], [], []
        for actor, registrations in enumerate(causal['registrations']):
            for frame, registration in enumerate(registrations):
                if registration is None: continue
                matrix, ijk = registration
                ijk = np.asarray(ijk, np.int64)
                if not len(ijk): continue
                good = ((ijk >= 0) & (ijk < shape)).all(1)
                ijk = np.unique(ijk[good], axis=0)
                ijk = ijk[valid[frame][tuple(ijk.T)]]
                if not len(ijk): continue
                valid[frame][tuple(ijk.T)] = False  # source features follow their motion, not static ego warp
                at = ijk.copy(); at[:, :2] //= 2
                flat = np.ravel_multi_index(at.T, coarse)
                unique, inverse, count = np.unique(flat, return_inverse=True, return_counts=True)
                world = transform_points(transform_points(origin+(ijk+.5)*step,
                    raw['history_poses'][frame]), matrix)
                mean = np.stack([np.bincount(inverse, weights=world[:, axis])/count for axis in range(3)], 1)
                points.append(mean); actors.append(np.full(len(unique), actor, np.int64))
                frames.append(np.full(len(unique), frame, np.int64)); cells.append(unique)
                coverage.append(count.astype(np.float32)/4)
        sx, sy, sz = (int(n) for n in shape)
        static_valid = valid.reshape(4, sx//2, 2, sy//2, 2, sz).mean((2, 4)).astype(np.float32)
        cat = lambda values, empty: np.concatenate(values) if values else empty
        causal['memory_reads'] = dict(static_valid=static_valid,
            world=cat(points, np.empty((0, 3), np.float64)),
            actor=cat(actors, np.empty(0, np.int64)), frame=cat(frames, np.empty(0, np.int64)),
            cell=cat(cells, np.empty(0, np.int64)), coverage=cat(coverage, np.empty(0, np.float32)))
        joint = getattr(self.provider, 'joint', None)
        hw = joint.transport.config.tube_hw//2 if joint is not None else 10
        causal['shared_reads'] = history_reads(historical, causal, NoCanonicalPoints(),
                                               self.provider.pcfg.grid, hw=hw)
        return causal, None  # no old descriptors, SurfaceAtlas, halo or labels ABI
