"""Prepare both policies from identical CCs; rebuild every dependent field."""
import numpy as np
import torch
from real_motion.ego_navigation import ego_history_features
from real_motion.shared_source_track import kta_and_anchors
from tools.source_scene_stage2 import track as original
from tools.source_scene_stage2.adapter import DirectPredictor
from tools.fpn_unet.adapter import forward_window
from tools.fpn_unet_stc_age.processing import weight_history
from . import track as candidate


class TrackPredictor(DirectPredictor):
    def prepare_pair(self, record, raw, *, times):
        if set(raw) != {'history_occ', 'history_observed', 'history_poses'}:
            raise ValueError('only historical observations/poses allowed in association')
        if (len(raw['history_occ']) != 4 or len(record['history_tokens']) != 4 or
                str(record['history_tokens'][-1]) != str(record['t0_token']) or len(record['future_tokens']) != 6):
            raise ValueError('four histories ending at t0 and six destinations required')
        from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
        poses = np.asarray(raw['history_poses'], np.float64)
        # No old association first, no monkey patch. Extract each component set ONCE.
        frames = [runtime.extract_instances_cropped_exact(sem, pose, grid=self.provider.pcfg.grid,
                    cfg=self.provider.strong) for sem, pose in zip(raw['history_occ'], poses)]
        current = frames[-1]
        points = runtime._precompute_source_world(current, poses[-1], self.provider.pcfg.grid)
        xyz = np.asarray([c['centroid_world'] for c in current]).reshape(-1, 3)
        identities = dict(baseline=original.build(frames, poses, times, self.provider.strong,
                        dt_s=self.provider.pcfg.frame_dt_s),
            forward=candidate.build(frames, poses, times, self.provider.strong, dt_s=self.provider.pcfg.frame_dt_s))
        rows = {}
        for name, identity in identities.items():
            native, kta, anchors = kta_and_anchors(current, identity, poses[-1], dt_s=self.provider.pcfg.frame_dt_s)
            rec = {**{k:record[k] for k in ('scene_name', 't0_token', 'future_tokens')},
                'history_tokens':tuple(record['history_tokens']),
                'source_class_id':np.asarray([c['class_id'] for c in current], np.int64),
                'source_centroid_xy_t0_m':native[:, :2].astype(np.float32),
                'kta_displacement_xy_m':kta, 'anchors_xy_t0_m':anchors}
            state = dict(current_pose=poses[-1], current=current, source_world_points=points,
                source_rel_xy=[v[:, :2]-xyz[i, None, :2] for i, v in enumerate(points)], source_z_t0=native[:, 2])
            value = dict(raw, _shared_source_track=identity, _column_causal_preparation=dict(prepared_state=state))
            rows[name] = rec, value, (ego_history_features(poses, times), None, None), self.history.build(value)
        old, new = identities.values()
        audit = dict(candidate=new.audit, source_count=len(current),
            changed_last_pairs=int(np.count_nonzero(old.links[:, 2] != new.links[:, 2])),
            changed_history_links=int(np.count_nonzero(old.links != new.links)),
            changed_KTA_sources=int(np.count_nonzero(np.any(rows['baseline'][0]['kta_displacement_xy_m'] !=
                rows['forward'][0]['kta_displacement_xy_m'], axis=(1, 2)))))
        return rows, audit

    @torch.no_grad()
    def infer(self, row, poses, *, stc_age=False):
        if stc_age:
            rec, raw, ego, history = row
            history, _ = weight_history(history, raw['_shared_source_track'].times)
            row = rec, raw, ego, history
        # Explicit poses force fresh Strong using THIS policy's pair_velocity.
        return forward_window(self.body, self.provider, row, poses=poses, fresh=True)
