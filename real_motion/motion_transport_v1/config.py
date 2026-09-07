from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml
from real_motion.geometry import OccupancyGrid
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.kta import KTAConfig
from real_motion.motion import PersistenceMotionConfig
DEFAULT_CONFIG=Path(__file__).resolve().parents[2]/'configs'/'real_motion'/'motion_transport_v1.yaml'
def _get(d,path,default=None):
    cur=d
    for k in path.split('.'):
        if not isinstance(cur,dict) or k not in cur:return default
        cur=cur[k]
    return cur
def _set(d,path,value):
    ks=path.split('.');cur=d
    for k in ks[:-1]:cur=cur.setdefault(k,{})
    cur[ks[-1]]=value
def load_config(path=None,overrides=()):
    p=Path(path or DEFAULT_CONFIG);cfg=yaml.safe_load(p.read_text()) or {}
    for item in overrides or ():
        if '=' not in item:raise ValueError(f'override must be key=value: {item}')
        k,v=item.split('=',1);_set(cfg,k.strip(),yaml.safe_load(v))
    validate_config(cfg);cfg=deepcopy(cfg);cfg.setdefault('runtime',{})['config_path']=str(p.resolve());return cfg
def make_grid_unchecked(c):
    r=list(map(float,_get(c,'grid.occ_range',[-40,-40,-1,40,40,5.4])));v=tuple(map(float,_get(c,'grid.voxel_size',[.4,.4,.4])));shape=(round((r[3]-r[0])/v[0]),round((r[4]-r[1])/v[1]),round((r[5]-r[2])/v[2]));return OccupancyGrid(r[0],r[1],r[2],v,shape)
def validate_config(c):
    from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS,PROTOCOL
    fixed={'spec_version':'MT-V1-SPEC-2','method':'motion_transport_v1','input.history_frames':6,'input.future_frames':6,'input.class_count':18,'input.free_label':17,'input.shape_xyz':[200,200,16],'input.future_objects_as_input':False,'crop.shape_xyz':[64,64,16],'crop.input_channels':84,'crop.embedding_tokens':19,'network.backbone':'stpn_adapted','network.output':'cumulative_kta_residual_xy_yaw','network.output_hard_clip':False,'network.autoregressive':False,'renderer.hard':'exact_kta_stable_forward_floor','renderer.soft':'inverse_trilinear_ordered_alpha','renderer.align_corners':False,'training.ddp':True,'training.accumulation_steps':4,'training.source_microbatch':16,'training.profile_warmup_microsteps':50,'training.profile_measure_microsteps':200,'training.augmentation.type':'local_F_y_coordinate_mirror','training.augmentation.restore_delta_y_and_yaw_before_losses':True,'routing.train_all_source_warmup_fraction':.2,'routing.train_later_all_source_probability':.5,'routing.train_later_sparse_budget':16,'routing.train_sparse_selection':'uniform_without_replacement_no_gt_no_msp','evaluation.moving_metric':'existing_moving_miou_v2'}
    for p,e in fixed.items():
        if _get(c,p)!=e:raise ValueError(f'SPEC-2 config mismatch {p}: {_get(c,p)!r} != {e!r}')
    if tuple(map(int,_get(c,'input.movable_classes',[])))!=tuple(DYNAMIC_CLASS_IDS):raise ValueError('movable class contract differs from Moving v2')
    if PROTOCOL!='interval_displacement_v2':raise ValueError('unexpected Moving v2 protocol')
    if any(bool(_get(c,x,False)) for x in ('network.vae','network.pretrained_occfm','network.pretrained_motionnet','network.shape_head','network.birth_head','network.visibility_head','network.utility_gate')):raise ValueError('forbidden pretrained/unregistered component enabled')
    if _get(c,'routing.preserve_original_msp_candidates_and_features') is not True:raise ValueError('MSP candidate preservation required')
    if _get(c,'data.supervision_valid_mask')!='occ3d_mask_lidar':raise ValueError('supervision mask contract mismatch')
    if float(_get(c,'training.max_hours',0))>4:raise ValueError('max training budget >4h')
    eps=float(_get(c,'renderer.probability_epsilon',0))
    if not 0<eps<1/18:raise ValueError('invalid renderer epsilon')
    lam=_get(c,'loss.lambda_reference');E=_get(c,'training.epochs_locked')
    if lam is not None and not float(lam)>0:raise ValueError('lambda_reference must be positive')
    if E is not None and int(E)<=0:raise ValueError('epochs_locked must be positive')
    if tuple(make_grid_unchecked(c).shape_hwd)!=tuple(map(int,_get(c,'input.shape_xyz'))):raise ValueError('grid shape mismatch')
def make_grid(c):return make_grid_unchecked(c)
def make_strong_cfg(c):return StrongW2DetConfig(free_label=int(_get(c,'input.free_label',17)),min_component_voxels=int(_get(c,'kta.min_component_voxels',6)),max_match_speed_mps=float(_get(c,'kta.max_match_speed_mps',25)),connectivity=int(_get(c,'kta.connectivity',2)),fill_kernel=tuple(map(int,_get(c,'kta.fill_kernel',[5,5,1]))),fill_min_fraction=float(_get(c,'kta.fill_min_fraction',.3)))
def make_msp_motion_cfg(c):
    g=make_grid(c);return PersistenceMotionConfig(free_label=int(_get(c,'input.free_label',17)),static_min_persistence=float(_get(c,'msp.static_min_persistence',.8)),moving_max_persistence=float(_get(c,'msp.moving_max_persistence',.5)),min_observed_frames=2,min_static_observations=int(_get(c,'msp.min_static_observations',3)),motion_eligible_class_ids=tuple(map(int,_get(c,'input.movable_classes'))),use_component_tracks=True,history_dt_s=float(_get(c,'input.dt_seconds',.5)),voxel_size_xy_m=(g.voxel_size[0],g.voxel_size[1]),component_max_step_m=float(_get(c,'msp.component_max_step_m',4)),moving_speed_mps=float(_get(c,'msp.moving_speed_mps',.5)),static_speed_mps=float(_get(c,'msp.static_speed_mps',.2)),min_track_frames=int(_get(c,'msp.min_track_frames',2)),min_component_bev_cells=int(_get(c,'msp.min_component_bev_cells',1)))
def make_msp_kta_cfg(c):return KTAConfig(free_label=int(_get(c,'input.free_label',17)),history_dt_s=float(_get(c,'input.dt_seconds',.5)),max_match_distance_m=float(_get(c,'msp.kta_max_match_distance_m',6)),min_component_cells=1)
get=_get
