"""Rebuild identities/KTA/reads/Strong; reuse only immutable t0 component data."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.ego_navigation import ego_history_features
from real_motion.shared_source_track import kta_and_anchors
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.source_scene_stage2.adapter import DirectProvider, DirectPredictor
from tools.source_scene_stage2.history import DirectHistory
from tools.source_scene_stage1.history import batches as original_batches
from tools.joint_occ_experiments.data import PURE_STATE
from tools.fpn_unet.adapter import forward_window as original_forward
from tools.causal_forward_track import track

HISTORICAL = ('history_occ','history_observed','history_poses')


def identity_from_raw(raw,provider,times,*,current=None):
    if set(raw) != set(HISTORICAL):raise ValueError('association accepts four histories ONLY')
    from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
    frames = [runtime.extract_instances_cropped_exact(raw['history_occ'][f],raw['history_poses'][f],
        grid=provider.pcfg.grid,cfg=provider.strong) for f in range(3)]
    frames.append(current if current is not None else runtime.extract_instances_cropped_exact(
        raw['history_occ'][-1],raw['history_poses'][-1],grid=provider.pcfg.grid,cfg=provider.strong))
    return frames,track.build(frames,raw['history_poses'],times,provider.strong,dt_s=provider.pcfg.frame_dt_s)


def updated_record(record,raw):
    state = raw['_column_causal_preparation']['prepared_state']; current = state['current']
    native,kta,anchors = kta_and_anchors(current,raw['_shared_source_track'],state['current_pose'])
    if (len(record.get('features',current)) != len(current)
            or not np.array_equal(record['source_class_id'],[c['class_id'] for c in current])
            or not np.allclose(record['source_centroid_xy_t0_m'],native[:,:2],atol=2e-4,rtol=0)):
        raise RuntimeError('t0 component order/count/centers changed; batch budget would differ')
    return dict(record,kta_displacement_xy_m=kta,anchors_xy_t0_m=anchors)


class ForwardProvider(DirectProvider):
    def attach_tracks(self,root,*,rank,implementation):
        from real_motion.v21_source_induction import stable_json_fingerprint
        from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
        new = (self.joint.experiment or {}).get('candidate_implementation',{})
        identity = stable_json_fingerprint(dict(protocol=track.PROTOCOL,grid=asdict(self.pcfg.grid),
            strong=asdict(self.strong),frame_dt_s=self.pcfg.frame_dt_s,implementation={**implementation,**new}))
        root = Path(root).parent/'forward_identity_history'/f'rank{rank}'
        self.track_cache = CausalGeometryCache(root,identity,max_bytes=8*2**30,ram_bytes=128*2**20,reserve_bytes=2*2**30)
        self.track_writer_lease = evaluation_lock(self.track_cache.cache_root);self.track_writer_lease.__enter__()
        return dict(protocol=track.PROTOCOL,root=str(root),history_only=True,old_associations_consumed=False,
            registration_computed=False,old_backgrounds_consumed=False,fresh_Strong_every_forward=True)

    def load_raw_columns(self,source,record,*,include_gt):
        if (self.ccr_history_cache_mode != 'require' or self.ccr_val_history_cache_mode != 'require'
                or source is not self.ccr_history_cache_source and source is not self.ccr_val_history_cache_source):
            raise RuntimeError('old component caches READ ONLY; no legacy builder fallback')
        raw = PilotProvider.load_raw_columns(self,source,record,include_gt=include_gt)
        state = {k:raw['_column_causal_preparation']['prepared_state'][k] for k in PURE_STATE}
        times = np.asarray([source.nusc.get('sample',str(t))['timestamp']/1e6 for t in record['history_tokens'][-4:]])
        history = {k:raw[k] for k in HISTORICAL}
        key_raw = dict(history,future_poses=np.array([0],np.uint8))  # history-only cache digest.
        identity,hit = self.track_cache.get_or_build((record['scene_name'],record['t0_token'],tuple(times)),key_raw,
            lambda:identity_from_raw(history,self,times,current=state['current'])[1])
        value = dict(raw,_shared_source_track=identity,_shared_track_cache_hit=hit,
            _column_causal_preparation=dict(prepared_state=state,source_track=identity))
        updated_record(record,value)  # validate t0, NOT old KTA/anchor associations.
        return value


class ForwardPredictor(DirectPredictor):
    def prepare(self,record,raw,*,times=None):
        if (len(record['history_tokens']) != 4 or str(record['history_tokens'][-1]) != str(record['t0_token'])
                or len(record['future_tokens']) != 6):raise ValueError('four histories ending t0 -> six futures')
        from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
        frames,identity = identity_from_raw(raw,self.provider,times)
        current = frames[-1];pose = np.asarray(raw['history_poses'][-1])
        native,kta,anchors = kta_and_anchors(current,identity,pose)
        points = runtime._precompute_source_world(current,pose,self.provider.pcfg.grid)
        xyz = np.asarray([c['centroid_world'] for c in current]).reshape(-1,3)
        rec = {**{k:record[k] for k in ('scene_name','t0_token','future_tokens')},
            'history_tokens':tuple(record['history_tokens']),
            'source_class_id':np.asarray([c['class_id'] for c in current],np.int64),
            'source_centroid_xy_t0_m':native[:,:2].astype(np.float32),
            'kta_displacement_xy_m':kta,'anchors_xy_t0_m':anchors}
        state = dict(current_pose=pose,current=current,source_world_points=points,
            source_rel_xy=[v[:,:2]-xyz[i,None,:2] for i,v in enumerate(points)],source_z_t0=native[:,2])
        value = dict(raw,_shared_source_track=identity,_column_causal_preparation=dict(prepared_state=state))
        return rec,value,(ego_history_features(raw['history_poses'],times),None,None),self.history.build(value)


def batches(*args,**kwargs):
    iterator = original_batches(*args,**kwargs)
    try:
        for rows in iterator:
            yield [(updated_record(rec,raw),raw,ego,history) for rec,raw,ego,history in rows]
    finally:iterator.close()


def forward_window(model,provider,row,*,poses=None,fresh=False):
    # Never use the old cached GT background after changing last-pair velocity.
    return original_forward(model,provider,row,poses=poses,fresh=True)
