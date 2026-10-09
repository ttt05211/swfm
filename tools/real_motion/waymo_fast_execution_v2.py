"""Opt-in second exact execution; old fast files/fingerprints remain untouched."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from contextlib import nullcontext
import time
import numpy as np
import torch

from real_motion.waymo_geometry_execution_v2 import (TubeFrameCache, build_tubes, NativeSurfaceAtlas,
    registration_reference, register_presorted)
from real_motion.source_evidence_audit import associate_backwards
from tools.real_motion.causal_column_common import pose_motion
from real_motion.waymo_native_execution import prepare_waymo_native
from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution
from tools.real_motion.waymo_fast_execution import FastWaymoSurfaceProvider, FastSurfaceBlockExecution
from tools.real_motion.joint_column_common import JointColumnProvider
from tools.real_motion.joint_surface_long_rollout_common import require_history_only
from tools.real_motion import joint_long_rollout_common as rollout


def causal_geometry_v2(raw,state,frames):
    components=[list(f.components) for f in frames]
    links,audit=associate_backwards(components,state['current'],state['velocities'],dt=.5)
    registrations=[[None]*4 for _ in state['current']]
    for i,comp in enumerate(state['current']):
        registrations[i][-1]=(np.eye(4),np.asarray(comp['voxel_indices'],np.int64))
        reference=None
        for f,j in enumerate(links[i][:-1]):
            if j is None: continue
            if reference is None: reference=registration_reference(state['source_world_points'][i])
            p=frames[f].registration_points[j]
            result=register_presorted(p,frames[f].registration_sorted[j],reference,allow_yaw=int(comp['class_id'])!=7)
            key='registration_accepted' if result.accepted else 'registration_rejected'
            audit[key]=audit.get(key,0)+1
            if result.accepted:
                r=pose_motion(np.zeros(3),np.zeros(3),result.yaw_rad)
                r[:2,3]=result.points[:,:2].mean(0)-p[:,:2].mean(0)@r[:2,:2].T
                registrations[i][f]=(r,np.asarray(components[f][j]['voxel_indices'],np.int64))
    return dict(current=state['current'],registrations=registrations,audit=audit,
                footprints=None,memory=None,prepared_state=state)


def build_state(raw,frames,pcfg,strong,device,native,pool):
    """Same original V18 equations/ABI; CPU tubes overlap the live Strong pass."""
    legacy=rollout.legacy; history=list(raw['history_occ']); poses=list(raw['history_poses'])
    future=list(raw['future_poses']); grid=pcfg.grid
    if len(history)!=4 or len(future)!=6 or pcfg.free_label!=17:
        raise ValueError('unchanged FOUR history / SIX future / free17 required')
    if any(x.shape!=tuple(grid.shape_hwd) or not np.isin(x,np.arange(18)).all() for x in history):
        raise ValueError('invalid semantic grid')
    components=[[],[],*[list(f.components) for f in frames]]
    current,previous=components[-1],components[-2]; timing={}; tick=time.perf_counter()
    velocities=legacy.match_instances(previous,current,float(pcfg.frame_dt_s),max_speed_mps=strong.max_match_speed_mps)
    tracks,valid=legacy.backward_component_tracks(components,frame_dt_s=float(pcfg.frame_dt_s),
        max_speed_mps=float(strong.max_match_speed_mps))
    feature_np=legacy.build_source_features(current,velocities,tracks,valid,poses[-1],
        frame_dt_s=float(pcfg.frame_dt_s),grid=grid)
    features=torch.from_numpy(feature_np)
    source_xy,kta,anchors_xy=legacy._kta_tensors(current,velocities,poses[-1],float(pcfg.frame_dt_s))
    timing['track_features_kta']=time.perf_counter()-tick; tick=time.perf_counter()
    finish_tubes=build_tubes(history,poses,frames,source_xy,legacy.history_offsets_from_features(features),
        valid,grid,native,pool,defer=True)
    timing['tube_submit']=time.perf_counter()-tick; tick=time.perf_counter()
    world=legacy._precompute_source_world(current,poses[-1],grid)
    relative_xy=[np.asarray(p,np.float64)[:,:2]-np.asarray(c['centroid_world'],np.float64)[None,:2]
                 for p,c in zip(world,current)]
    z=np.asarray([legacy.world_points_to_t0(np.asarray(c['centroid_world'],np.float64)[None],poses[-1])[0,2]
                  for c in current],np.float64)
    timing['source_geometry']=time.perf_counter()-tick; tick=time.perf_counter(); strong_profile={}
    anchors,baseline=legacy._strong_all_horizons(history[-1],poses[-1],future,current,velocities,world,
        frame_dt_s=float(pcfg.frame_dt_s),grid=grid,cfg=strong,runtime_device=device,profile=strong_profile)
    timing['strong_prior']=time.perf_counter()-tick
    timing.update({'strong.'+k.removesuffix('_ms'):v/1000 for k,v in strong_profile.items()})
    tick=time.perf_counter(); clear=[legacy.baseline_clear_mask(rows,grid=grid) for rows in baseline]
    clear_flat=[legacy.baseline_clear_flat_indices(rows,grid=grid) for rows in baseline]
    timing['clear_masks']=time.perf_counter()-tick; tick=time.perf_counter()
    tube=torch.from_numpy(finish_tubes()); timing['tube_wait']=time.perf_counter()-tick; tick=time.perf_counter()
    class_ids=torch.as_tensor([int(c['class_id']) for c in current],dtype=torch.long)
    frame_motion=legacy.frame_motion_features_from_flat(features)
    masks=legacy.target_source_mask_from_tube(tube,class_ids,torch.from_numpy(valid),features)
    rec=dict(features=features,local_semantic_tube=tube,kta_displacement_xy_m=torch.from_numpy(kta),
        frame_motion_features=frame_motion,target_source_mask_tube=masks,source_class_id=class_ids,
        anchors_xy_t0_m=torch.from_numpy(anchors_xy),source_centroid_xy_t0_m=torch.from_numpy(source_xy))
    timing['mask_and_tensor_pack']=time.perf_counter()-tick
    return dict(rec=rec,window=None,scene=None,current_sem=history[-1],previous_sem=history[-2],
        current_pose=poses[-1],previous_pose=poses[-2],future_poses=future,current=current,previous=previous,
        velocities=velocities,components_by_frame=components,motion_handoff_audit=None,source_world_points=world,
        source_rel_xy=relative_xy,source_z_t0=z,anchors=anchors,baseline_by_hi=baseline,
        baseline_clear_by_hi=clear,baseline_clear_flat_by_hi=clear_flat,
        world_to_future=[np.linalg.inv(np.asarray(p,np.float64)) for p in future],gpu=None),timing


class FastV2WaymoProvider(FastWaymoSurfaceProvider):
    def __init__(self,joint,pcfg,device,workers,*,geometry_mib=1024):
        super().__init__(joint,pcfg,device,workers,geometry_mib=geometry_mib)
        self.native=prepare_waymo_native()
        self.geometry=TubeFrameCache(pcfg.grid,self.strong,ram_mib=geometry_mib)
        self.prepare_pool=ThreadPoolExecutor(workers,thread_name_prefix='waymo-tubes')

    def prepare_columns(self,source,record,*,include_gt,raw_window=None,outputs=None):
        if include_gt or raw_window is None or 'features' in record:
            raise RuntimeError('v2 requires actual raw FOUR history inputs')
        raw=raw_window; require_history_only(raw); tick=time.perf_counter()
        frames=raw.get('_waymo_frame_geometry')
        if frames is None: frames=self.geometry.window(record,raw)
        raw['_waymo_frame_geometry']=frames
        timing=dict(frame_geometry_wait=time.perf_counter()-tick); tick=time.perf_counter()
        state,detail=build_state(raw,frames,self.pcfg,self.strong,self.device,self.native,self.prepare_pool)
        timing['state_tracks_tubes_strong']=time.perf_counter()-tick
        timing.update({'state.'+k:v for k,v in detail.items()})
        record={**state['rec'],**record}; state['rec']=record; tick=time.perf_counter()
        raw['_column_causal_preparation']=causal_geometry_v2(raw,state,frames)
        timing['live_history_registration']=time.perf_counter()-tick; tick=time.perf_counter()
        prep=JointColumnProvider.prepare_columns(self,None,record,include_gt=False,raw_window=raw,outputs=outputs)
        timing['live_motion_and_layers']=time.perf_counter()-tick; self.fast_prepare_stages=timing
        return prep

    def close(self): self.prepare_pool.shutdown(wait=True,cancel_futures=True)


class FastV2SurfaceExecution(FastSurfaceBlockExecution):
    def atlas(self,prep,evidence):
        atlas=NativeSurfaceAtlas(evidence.world,evidence.classes,evidence.presence,evidence.actor,
            prep.state['current_pose'],self.provider.pcfg.grid)
        atlas.chunk_rows=self.surface_chunk; atlas.fit_pool=self.cpu.pool; atlas.native=self.provider.native
        return atlas


@torch.no_grad()
def paired_speed_v2(source,windows,joint,pcfg,device,*,workers,graphs,parallel_majority,
                    surface_chunk,geometry_mib,repeats=2,stop_event=None):
    """Same ordered population, first-frame-only warm, actual full SIX output."""
    if not windows or repeats<1: raise ValueError('nonempty speed population required')
    providers=dict(fast_v1=FastWaymoSurfaceProvider(joint,pcfg,device,workers,geometry_mib=geometry_mib),
        fast_v2=FastV2WaymoProvider(joint,pcfg,device,workers,geometry_mib=geometry_mib))
    executors={arm:(FastSurfaceBlockExecution if arm=='fast_v1' else FastV2SurfaceExecution)(provider,
        workers=workers,query_workers=workers,graphs=graphs,surface_chunk=surface_chunk)
        for arm,provider in providers.items()}
    durations=defaultdict(list); details=defaultdict(lambda:defaultdict(float))
    majority=ParallelNativeMajority(min(4,workers)) if parallel_majority else None
    def sync():
        if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
    def run(arm,window):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('paired speed stopped')
        record,raw=source.prediction_inputs(window); tick=time.perf_counter()
        with strong_majority_execution(majority) if majority is not None else nullcontext():
            prep=providers[arm].prepare_columns(None,record,include_gt=False,raw_window=raw)
        timing={'history_and_transport_prepare':time.perf_counter()-tick,
            **{'prepare.'+k:v for k,v in providers[arm].fast_prepare_stages.items()}}
        dense,_,stages,scores=executors[arm].predict(prep); timing.update(stages)
        return prep,dense,scores,timing
    try:
        for w in windows:
            a=run('fast_v1',w); b=run('fast_v2',w)
            # STRICT bytes for every motion input, not a tolerance-only gate.
            for k in ('features','local_semantic_tube','frame_motion_features','target_source_mask_tube',
                      'kta_displacement_xy_m','anchors_xy_t0_m','source_centroid_xy_t0_m','source_class_id'):
                x,y=a[0].state['rec'][k].cpu().numpy(),b[0].state['rec'][k].cpu().numpy()
                if x.shape!=y.shape or x.dtype!=y.dtype or x.tobytes()!=y.tobytes():
                    raise RuntimeError('Waymo v2 motion input byte gate failed: '+k)
            rollout.assert_dense_equal(a[0].baseline,b[0].baseline); rollout.assert_dense_equal(a[1],b[1])
            if a[2].shape!=b[2].shape or a[2].dtype!=b[2].dtype or a[2].tobytes()!=b[2].tobytes():
                raise RuntimeError('Waymo v2 probability bytes differ')
        for repeat in range(repeats):
            for arm in (('fast_v1','fast_v2') if repeat%2==0 else ('fast_v2','fast_v1')):
                providers[arm].geometry.clear(); run(arm,windows[0]); sync(); tick=time.perf_counter()
                for w in windows:
                    prep,dense,scores,stages=run(arm,w)
                    for k,v in stages.items(): details[arm][k]+=v
                    del prep,dense,scores
                sync(); durations[arm].append((time.perf_counter()-tick)/len(windows))
        means={k:float(np.mean(v)) for k,v in durations.items()}
        return dict(windows=len(windows),repeats=repeats,seconds_per_window=means,
            speedup=means['fast_v1']/means['fast_v2'],repeat_seconds_per_window=dict(durations),
            stages_seconds_per_window={k:{n:v/(len(windows)*repeats) for n,v in d.items()} for k,d in details.items()},
            probability_and_six_dense_bytes_exact=True,motion_inputs_bytes_exact=True,no_future_GT_reads=True,
            no_saved_scientific_updates=True,scope='v1/v2 full same-window execution; NOT formal FPS',
            geometry_timing_policy='clear each pass; only first history warm; each later frame fresh')
    finally:
        for e in executors.values(): e.close()
        providers['fast_v2'].close()
        if majority is not None: majority.close()
