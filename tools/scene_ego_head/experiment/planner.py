"""Read ONLY identity and external poses, never old camera/latent/label arrays."""
import json
from pathlib import Path
import re
import numpy as np
from real_motion.stc_camera_protocol import PLAN_TAG, PLAN_SHA, validate_pose
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
from real_motion.v21_source_induction import stable_json_fingerprint


class PlannerCache:
    def __init__(self, root):
        self.root = Path(root).resolve()
        path = self.root/'manifest.json'
        before = sha256(path)
        m = json.loads(path.read_text(encoding='utf-8'))
        if sha256(path) != before: raise RuntimeError('planner manifest changed while reading')
        if (m.get('format') != 'stochocc_transport_cache_v2' or m.get('come_protocol_setting') != 'camera_pred'
                or m.get('come_protocol_tag') != PLAN_TAG or m.get('future_ego') != 'predicted'
                or m.get('come_trajectory_sha256') != PLAN_SHA or m.get('hist_last') != 4
                or m.get('predicted_yaw_semantics') != 'each_future_json_yaw_offset_relative_to_current_pose_non_cumulative'):
            raise ValueError('exact v4 BEV-Planner+yaw cache required')
        keys = m.get('keys', [])
        if not keys or len(keys) != len(set(keys)) or len(keys) != m.get('num_windows'):
            raise ValueError('invalid planner population')
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+__[A-Za-z0-9_-]+', k) for k in keys):
            raise ValueError('unsafe planner identity')
        self.keys = {tuple(k.split('__')) for k in keys}
        self.rows, self.inventory = {}, {}
        self.metadata = dict(root=str(self.root), manifest_sha256=before, original_json_declared_sha256=PLAN_SHA)

    def prepare(self, requested):
        keys = [tuple(k) for k in requested]
        if not keys or len(keys) != len(set(keys)): raise ValueError('empty/duplicate fixed evaluation population')
        common = [k for k in keys if k in self.keys]
        if not common: raise ValueError('no planner-covered evaluation windows')
        for key in common:
            path = self.root/'records'/('__'.join(key)+'.npz')
            stat = path.stat(); signature = (stat.st_size, stat.st_mtime_ns)
            with np.load(path, allow_pickle=False) as z:
                if (str(z['format'].item()) != 'stochocc_transport_cache_v2' or str(z['protocol_tag'].item()) != PLAN_TAG
                        or str(z['scene'].item()) != key[0]): raise ValueError('planner record identity/tag differs')
                hist, future = tuple(z['history_tokens'].tolist()), tuple(z['future_tokens'].tolist())
                if len(hist) != 4 or hist[-1] != key[1] or len(future) != 6 or len(set(hist+future)) != 10:
                    raise ValueError('planner record history/future identity differs')
                raw = z['future_e2g']
                if raw.shape != (6, 4, 4): raise ValueError('six planner poses required')
                poses = np.stack([validate_pose(p) for p in raw])
            end = path.stat()
            if signature != (end.st_size, end.st_mtime_ns): raise RuntimeError('planner archive changed')
            self.rows[key] = dict(history=hist, future=future, poses=poses)
            self.inventory[key] = signature
        self.requested = keys
        self.metadata.update(requested=len(keys), covered=len(common), selected_keys=[list(k) for k in common],
            missing_keys=[list(k) for k in keys if k not in self.keys],
            selected_pose_identity_fingerprint=stable_json_fingerprint([
                [list(k), self.rows[k]['history'], self.rows[k]['future'], self.rows[k]['poses'].tolist(), self.inventory[k]]
                for k in common]), scoring='ALL routes same planner-covered intersection; full grid, no alignment')

    def poses(self, record, current):
        key = (record['scene_name'], record['t0_token']); row = self.rows[key]
        if (tuple(record['history_tokens'])[-4:] != row['history'] or tuple(record['future_tokens']) != row['future']):
            raise ValueError('live record/planner time identity differs')
        path = self.root/'records'/('__'.join(key)+'.npz'); st = path.stat()
        if (st.st_size, st.st_mtime_ns) != self.inventory[key]: raise RuntimeError('planner archive changed since preflight')
        p = row['poses']; current = validate_pose(current)
        rel = current[:3, :3].T[None] @ p[:, :3, :3]
        if not np.allclose(p[:, 2, 3], current[2, 3], atol=1e-6, rtol=0) or not np.allclose(rel[:, :, 2], [0, 0, 1], atol=1e-6, rtol=0):
            raise ValueError('external planner must preserve t0 z/tilt, no GT completion')
        return p.copy()
