from collections import OrderedDict
from pathlib import Path
import numpy as np
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens, gt_moving_support_sequence
from real_motion.stc_camera_protocol import semantics
from real_motion.ego_navigation import navigation_commands
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from tools.real_motion import geniedrive_eval_alignment as genie, joint_long_rollout_common as rollout


def window(record):
    return WindowTokens(str(record['scene_name']), tuple(record['history_tokens'][-4:]),
                        str(record['t0_token']), tuple(record['future_tokens']))


class NuscInputs:
    def __init__(self, dataroot, info, *, stc_root=None, cache_mib=256):
        self.source = NuScenesWindowSource(dataroot, info_pkl=info)
        self.stc = Path(stc_root).resolve() if stc_root else None
        self.cache = OrderedDict(); self.bytes = 0; self.limit = cache_mib*2**20; self.inventory = {}

    def path(self, kind, scene, token):
        return ((self.stc if kind == 'stc' else Path(self.source.dataroot)/'gts') / scene / token / 'labels.npz')

    def stat(self, path):
        st = path.stat(); value = (st.st_size, st.st_mtime_ns)
        if not path.is_file() or not st.st_size: raise ValueError('empty/non-file label: '+str(path))
        if str(path) in self.inventory and self.inventory[str(path)] != value:
            raise RuntimeError('label file changed: '+str(path))
        self.inventory[str(path)] = value
        return value

    def preflight(self, jobs, kind='occ'):
        paths = set()
        for w in jobs:
            paths.update(self.path(kind, w.scene_name, t) for t in w.history_tokens)
            paths.update(self.path('occ', w.scene_name, t) for t in w.future_tokens)
        for path in sorted(paths): self.stat(path)
        return [[str(p), *self.inventory[str(p)]] for p in sorted(paths)]

    def frame(self, kind, scene, token):
        path = self.path(kind, scene, token); self.stat(path); key = (kind, scene, token)
        if key in self.cache: self.cache.move_to_end(key); return self.cache[key]
        with np.load(path, allow_pickle=False) as z:
            sem = semantics(z['semantics'])
            # Camera branch MUST NOT read real visibility, even if present in NPZ.
            observed = np.ones(sem.shape, bool) if kind == 'stc' else np.array(z['mask_lidar'], bool, copy=True)
        if observed.shape != sem.shape: raise ValueError('historical observation shape differs')
        value = (sem, observed); size = sum(v.nbytes for v in value)
        if size <= self.limit:
            while self.cache and self.bytes+size > self.limit:
                _, old = self.cache.popitem(last=False); self.bytes -= sum(v.nbytes for v in old)
            self.cache[key] = value; self.bytes += size
        return value

    def history(self, w, kind='occ'):
        loaded = [self.frame(kind, w.scene_name, t) for t in w.history_tokens]
        raw = dict(history_occ=np.stack([r[0] for r in loaded]), history_observed=np.stack([r[1] for r in loaded]),
            history_poses=[self.source.pose(t) for t in w.history_tokens])
        times = np.array([self.source.nusc.get('sample', t)['timestamp']/1e6 for t in w.history_tokens])
        record = dict(scene_name=w.scene_name, t0_token=w.t0_token, history_tokens=w.history_tokens,
                      future_tokens=w.future_tokens[:6])
        return record, raw, times

    def poses(self, w): return [self.source.pose(t) for t in w.future_tokens]
    def commands(self, w): return navigation_commands(self.source.nusc, w.future_tokens)
    def targets(self, w, indices):
        result = []
        for i in indices:
            path = self.path('occ', w.scene_name, w.future_tokens[i]); self.stat(path)
            with np.load(path, allow_pickle=False) as z: result.append(semantics(z['semantics']))
        return result

    def moving(self, w, indices, horizons, grid, workers):
        support = gt_moving_support_sequence(self.source.nusc, w.t0_token,
            tuple(w.future_tokens[i] for i in indices), tuple(horizons), grid=grid, workers=workers)
        return moving_support_masks(support, grid.shape_hwd)


def nusc_jobs(inputs, records, parent, population, task, *, genie_info=None, long_alignment='geniedrive'):
    by = {(str(r['scene_name']), str(r['t0_token'])): r for r in records}
    if len(by) != 4369 or len(by) != len(records): raise RuntimeError('complete unique VAL4369 cache required')
    if task == 'long6':
        if long_alignment == 'geniedrive':
            if population != 'all' or not genie_info: raise ValueError('GenieDrive needs population=all and pinned metadata')
            pairs, audit = genie.select_population(genie_info, inputs.source, records)
        else:
            pairs, audit = rollout.select_long_population(records, inputs.source.iter_windows(history=4, future=12), parent, population)
        jobs = [w for w, _ in pairs]
        audit['timestamp_audit'] = rollout.summarize_timestamps([rollout.validate_timestamps(inputs.source.nusc, w) for w in jobs])
        return jobs, audit
    keys = list(by) if population == 'all' else [tuple(k) for k in parent]
    if population == 'dev64':
        from real_motion.v21_source_induction import select_scene_balanced_round_robin
        keys = select_scene_balanced_round_robin(keys, 64)
    jobs = [window(by[k]) for k in keys]; missing = []
    if task == 'stc':
        missing = [[w.scene_name, w.t0_token] for w in jobs
            if any(not inputs.path('stc', w.scene_name, t).is_file() for t in w.history_tokens)]
        absent = {tuple(k) for k in missing}
        jobs = [w for w in jobs if (w.scene_name, w.t0_token) not in absent]
    if not jobs: raise RuntimeError('empty evaluation population')
    return jobs, dict(population=population, requested=len(keys), windows=len(jobs),
        missing_STC_history_keys=missing, planner_trimmed=False, selected_keys=[[w.scene_name, w.t0_token] for w in jobs])
