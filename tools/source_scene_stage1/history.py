"""New source references, identical old registration-conditioned memory reads."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tools.history_validity.data import policy_raw
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.shared_history_experiments.full_gt.engine import history_inputs
from tools.joint_occ_experiments.data import PURE_STATE
from real_motion.canonical_causal_repair import grid_arrays
from real_motion.source_evidence_audit import transform_points


class StageHistory(ResidualHistoryInputs):
    def __init__(self, provider, *, arm, ram_mib=512):
        super().__init__(provider, ram_mib=ram_mib)
        if arm not in ('baseline', 'reader'): raise ValueError('unknown arm')
        self.arm = arm

    def build(self, raw):
        raw = policy_raw(raw, 'dense_valid')
        if self.arm == 'baseline': return super().build(raw)
        fixed = raw['_column_causal_preparation']
        causal = dict(state={k: fixed['prepared_state'][k] for k in PURE_STATE},
                      registrations=fixed['registrations'])
        grid = self.provider.pcfg.grid
        origin, step, shape = grid_arrays(grid)
        if shape[0] % 2 or shape[1] % 2: raise ValueError('even grid required')
        coarse = (int(shape[0]//2), int(shape[1]//2), int(shape[2]))
        # Deliberately retain the original memory algorithm byte-for-byte in
        # arithmetic/order. Only source tube/crop construction below is removed.
        valid = np.asarray(raw['history_observed'], bool).copy()
        labels = np.asarray(raw['history_occ']); valid &= (labels >= 0) & (labels < 18)
        points, actors, frames, cells, coverage = [], [], [], [], []
        poses = np.asarray(raw['history_poses'], np.float64)
        for actor, registrations in enumerate(causal['registrations']):
            for frame, registration in enumerate(registrations):
                if registration is None: continue
                matrix, ijk = registration; ijk = np.asarray(ijk, np.int64)
                if not len(ijk): continue
                good = ((ijk >= 0) & (ijk < shape)).all(1)
                ijk = np.unique(ijk[good], axis=0)
                ijk = ijk[valid[frame][tuple(ijk.T)]]
                if not len(ijk): continue
                valid[frame][tuple(ijk.T)] = False
                at = ijk.copy(); at[:, :2] //= 2
                flat = np.ravel_multi_index(at.T, coarse)
                unique, inverse, count = np.unique(flat, return_inverse=True, return_counts=True)
                world = transform_points(transform_points(origin+(ijk+.5)*step, poses[frame]), matrix)
                mean = np.stack([np.bincount(inverse, weights=world[:, axis])/count for axis in range(3)], 1)
                points.append(mean); actors.append(np.full(len(unique), actor, np.int64))
                frames.append(np.full(len(unique), frame, np.int64)); cells.append(unique)
                coverage.append(count.astype(np.float32)/4)
        sx, sy, sz = map(int, shape)
        static_valid = valid.reshape(4, sx//2, 2, sy//2, 2, sz).mean((2, 4)).astype(np.float32)
        cat = lambda values, empty: np.concatenate(values) if values else empty
        causal['memory_reads'] = dict(static_valid=static_valid,
            world=cat(points, np.empty((0, 3), np.float64)), actor=cat(actors, np.empty(0, np.int64)),
            frame=cat(frames, np.empty(0, np.int64)), cell=cat(cells, np.empty(0, np.int64)),
            coverage=cat(coverage, np.empty(0, np.float32)))
        sources = len(causal['state']['current'])
        reference = np.zeros((sources, 4, 2), np.float32)
        xy_t0 = np.zeros_like(reference); reference_valid = np.zeros((sources, 4), bool)
        history_to_t0 = np.linalg.inv(poses[-1])[None] @ poses
        for actor, registrations in enumerate(causal['registrations']):
            for frame, registration in enumerate(registrations):
                if registration is None: continue
                ijk = np.asarray(registration[1], np.int64)
                if not len(ijk): continue
                native = origin+(ijk+.5)*step
                reference[actor, frame] = (2*(native.mean(0)[:2]-origin[:2])/(step[:2]*shape[:2])-1)[::-1]
                world = transform_points(native, poses[frame])
                xy_t0[actor, frame] = transform_points(world.mean(0)[None], np.linalg.inv(poses[-1]))[0, :2]
                reference_valid[actor, frame] = True
        # Offset [dx0,dy0,0] -> native XY -> normalized [gridY,gridX].
        t0_to_history = np.linalg.inv(poses) @ poses[-1][None]
        jacobian = t0_to_history[:, :2, :2]* (2/(step[:2]*shape[:2]))[None, :, None]
        causal['shared_reads'] = dict(reference_grid=reference, reference_xy_t0=xy_t0,
            reference_valid=reference_valid, t0_xy_to_native_grid=jacobian[:, ::-1].copy().astype(np.float32),
            history_to_t0=history_to_t0.astype(np.float32),
            scene_bounds=np.r_[origin[:2], origin[:2]+step[:2]*shape[:2]].astype(np.float32))
        return causal, None


def batches(provider, source, groups, history, *, workers=2, depth=2):
    """Bounded CPU prefetch; no rebinding of legacy module globals or CUDA work."""
    if not 1 <= workers <= 8 or not 1 <= depth <= 4: raise ValueError('prefetch bounds')
    iterator, pending = iter(groups), deque()
    def load(record):
        raw = provider.load_raw_columns(source, record, include_gt=True)
        inputs = history_inputs(source, record, raw)
        return record, policy_raw(raw, 'dense_valid'), inputs, history.get(record, raw)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit():
            group = next(iterator, None)
            if group is None: return False
            pending.append([pool.submit(load, r) for r in group]); return True
        for _ in range(depth):
            if not submit(): break
        while pending:
            rows = [f.result() for f in pending.popleft()]
            submit(); yield rows
