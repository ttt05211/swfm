"""History-only frozen feature extraction; isolated ego-head bank and recovery."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from real_motion.ego_trajectory_head import EgoHeadConfig, HistoryEgoTrajectoryHead, PROTOCOL
from real_motion.ego_navigation import ego_history_features, checked_poses
from real_motion.canonical_causal_repair import build_canonical_evidence
from real_motion.surface_canonical_repair import augment_evidence
from real_motion.waymo_geometry_execution_v2 import build_tubes, NativeSurfaceAtlas
from tools.real_motion import joint_long_rollout_common as rollout

FEATURE_FIELDS = ('objects','object_geometry','object_valid','surfaces','surface_geometry','surface_valid','ego_history')


def implementation_fingerprint(root):
    """Pin feature equations and geometry execution, not merely the new head."""
    root=Path(root)
    files=[*sorted((root/'real_motion').rglob('*.py')),*sorted((root/'real_motion/native').glob('*.cpp')),
           *sorted((root/'tools/real_motion').glob('*.py'))]
    return fingerprint({p.relative_to(root).as_posix():digest_file(p) for p in files})


def row_fingerprint(row):
    h=hashlib.sha256()
    for k in FEATURE_FIELDS:
        a=row['features'][k].detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    for k in ('target','commands'):
        a=np.asarray(row[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    h.update(fingerprint(row['key']).encode())
    return h.hexdigest()


def atomic_save(path, value):
    """Only caller-owned NEW artifacts; never used on the frozen WM checkpoint."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temporary.open('xb') as f:
            torch.save(value,f);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@torch.no_grad()
def extract_history_features(provider, record, raw, config, *, timestamps_s=None):
    """No Strong, rendering, future pose, future mask, annotation or target reads.

    Reuses the original V18 history equations and Surface-CCR **encode** only.
    A whitelist physically removes future fields before any preparation call.
    Object limit and static sampling are ego-head pooling ONLY, never WM caps.
    """
    raw={k:raw[k] for k in ('history_occ','history_observed','history_poses')}
    if any(len(raw[k])!=4 for k in raw):raise ValueError('FOUR causal histories required')
    poses=checked_poses(raw['history_poses'],4); legacy=rollout.legacy
    pcfg,strong=provider.pcfg,provider.strong;grid=pcfg.grid;device=provider.device
    if provider.joint.training or any(p.requires_grad for p in provider.joint.parameters()):
        raise RuntimeError('shared WM/CCR must be frozen evaluation-only')
    frames=provider.geometry.window(record,raw)
    components=[[],[],*[list(f.components) for f in frames]]
    current,previous=components[-1],components[-2]
    velocities=legacy.match_instances(previous,current,pcfg.frame_dt_s,max_speed_mps=strong.max_match_speed_mps)
    tracks,valid=legacy.backward_component_tracks(components,frame_dt_s=pcfg.frame_dt_s,
        max_speed_mps=strong.max_match_speed_mps)
    features=torch.from_numpy(legacy.build_source_features(current,velocities,tracks,valid,poses[-1],
        frame_dt_s=pcfg.frame_dt_s,grid=grid))
    source_xy,kta,_=legacy._kta_tensors(current,velocities,poses[-1],pcfg.frame_dt_s)
    tube=torch.from_numpy(build_tubes(list(raw['history_occ']),list(poses),frames,source_xy,
        legacy.history_offsets_from_features(features),valid,grid,provider.native,provider.prepare_pool))
    classes=torch.as_tensor([int(c['class_id']) for c in current],dtype=torch.long)
    rec=dict(features=features,local_semantic_tube=tube,kta_displacement_xy_m=torch.from_numpy(kta),
        frame_motion_features=legacy.frame_motion_features_from_flat(features),
        target_source_mask_tube=legacy.target_source_mask_from_tube(tube,classes,torch.from_numpy(valid),features))
    outputs=provider.joint.motion(rec,device)
    n=config.object_slots;side=config.surface_side
    bank=dict(objects=np.zeros((n,config.object_dim),np.float32),object_geometry=np.zeros((n,8),np.float32),
        object_valid=np.zeros(n,bool),surfaces=np.zeros((side**2,config.surface_dim),np.float32),
        surface_geometry=np.zeros((side**2,3),np.float32),surface_valid=np.zeros(side**2,bool),
        ego_history=ego_history_features(poses,timestamps_s))
    if outputs['history_source_context'].shape!=(len(current),config.object_dim):
        raise RuntimeError('frozen object encoder/head ABI mismatch')
    # Stable nearest-object pooling. Source ordering in actual WM is untouched.
    order=np.argsort(np.sum(source_xy**2,axis=1),kind='stable')[:n]
    world=legacy._precompute_source_world(current,poses[-1],grid)
    inv=np.linalg.inv(poses[-1]); context=outputs['history_source_context'].float().cpu().numpy()
    for slot,idx in enumerate(order):
        c=current[idx]; points=np.asarray(world[idx],np.float64)
        local=points@inv[:3,:3].T+inv[:3,3];v=legacy.world_vec_to_t0(np.asarray(velocities.get(int(idx),np.zeros(3))),poses[-1])
        bank['objects'][slot]=context[idx]
        bank['object_geometry'][slot]=np.r_[source_xy[idx]/40,v[:2]/10,c['class_id']/16,
            np.ptp(local,axis=0)/10]
        bank['object_valid'][slot]=True
    # Only static historical evidence is needed here; no future CCR phases.
    prep=SimpleNamespace(raw=raw,state=dict(current_pose=poses[-1],current=[]),registrations=[])
    evidence=build_canonical_evidence(prep,grid)
    atlas=NativeSurfaceAtlas(evidence.world,evidence.classes,evidence.presence,evidence.actor,poses[-1],grid)
    atlas.native=provider.native;atlas.chunk_rows=1024;atlas.fit_pool=provider.prepare_pool
    local=evidence.world@inv[:3,:3].T+inv[:3,3]
    origin=np.array([grid.x_min,grid.y_min]); extent=np.asarray(grid.shape_hwd[:2])*np.asarray(grid.voxel_size[:2])
    cells=np.floor((local[:,:2]-origin)/extent*side).astype(np.int64)
    inside=((cells>=0)&(cells<side)).all(1)&evidence.presence.any(1)
    cell_id=cells[:,0]*side+cells[:,1];chosen=[]
    for k in range(side**2):
        ids=np.flatnonzero(inside&(cell_id==k))
        if len(ids):chosen.extend(ids[np.linspace(0,len(ids)-1,min(64,len(ids)),dtype=np.int64)])
    chosen=np.asarray(chosen,np.int64)
    if len(chosen):
        small=replace(evidence,features=evidence.features[chosen],labels=evidence.labels[chosen],
            actor=evidence.actor[chosen],classes=evidence.classes[chosen],world=evidence.world[chosen],
            presence=evidence.presence[chosen],layouts=None)
        small=augment_evidence(small,atlas)
        encoded=[]
        for start in range(0,len(chosen),1024):
            sl=slice(start,start+1024)
            t=lambda a:torch.as_tensor(a[sl],device=device)
            encoded.append(provider.joint.columns.encode(t(small.features).float(),t(small.labels),
                t(small.actor),t(small.classes),outputs).float().cpu().numpy())
        encoded=np.concatenate(encoded)
        for k in range(side**2):
            subset=cell_id[chosen]==k
            if not subset.any():continue
            if encoded.shape[1]!=config.surface_dim:raise RuntimeError('frozen surface encoder/head ABI mismatch')
            bank['surfaces'][k]=encoded[subset].mean(0);bank['surface_valid'][k]=True
            bank['surface_geometry'][k,:2]=((origin+(np.array([k//side,k%side])+.5)*extent/side)/40)
            bank['surface_geometry'][k,2]=np.log1p(int((inside&(cell_id==k)).sum()))/10
    return {k:torch.from_numpy(v) for k,v in bank.items()}


def stack_features(rows, device):
    return {k:torch.stack([r[k] for r in rows]).to(device) for k in FEATURE_FIELDS}


def save_head(path, head, optimizer, *, contract, epoch, cursor, order, updates, generator):
    atomic_save(path,dict(protocol=PROTOCOL,contract=contract,config=asdict(head.config),
        state_dict=head.state_dict(),optimizer=optimizer.state_dict(),epoch=int(epoch),cursor=int(cursor),
        order=list(order),updates=int(updates),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        sampling_rng=generator.bit_generator.state,python_rng=random.getstate()))


def restore_head(path, head, optimizer, contract, generator):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('protocol')!=PROTOCOL or saved.get('contract')!=contract or saved.get('config')!=asdict(head.config):
        raise RuntimeError('ego resume contract changed: bank/model/population/schedule must match')
    head.load_state_dict(saved['state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng']);generator.bit_generator.state=saved['sampling_rng']
    random.setstate(saved['python_rng'])
    if saved['cuda_rng']:
        if len(saved['cuda_rng'])!=torch.cuda.device_count():raise RuntimeError('resume CUDA device count mismatch')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved
