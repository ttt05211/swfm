#!/usr/bin/env python3
"""Read-only complete WM/CCR evaluation of the independently trained ego head."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from dataclasses import asdict
import json
import signal
import threading
import time

import numpy as np
import torch

from real_motion.ego_trajectory_head import HistoryEgoTrajectoryHead,EgoHeadConfig,PROTOCOL
from real_motion.ego_navigation import navigation_commands,relative_se2,poses_from_se2,replace_future_geometry,COMMAND_PROTOCOL
from real_motion.stc_camera_protocol import STCFourSettingSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.waymo_i2world import WaymoMetrics
from real_motion.runtime_config import load_runtime_config,make_prepare_config
from tools.real_motion.ego_trajectory_common import extract_history_features,stack_features,digest_file,fingerprint,implementation_fingerprint
from tools.real_motion.joint_surface_checkpoint_selection import load_evaluation_model,AVERAGE_EPOCHS
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest
from tools.real_motion.waymo_zero_shot_common import write_json
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock


@torch.no_grad()
def evaluate(source,windows,predictor,head,commands_for,contract,*,saved=None,save=None,progress=None,
             stop_event=None,feature_extractor=extract_history_features):
    arms=tuple(f'{m}_{a}' for m in contract['modalities'] for a in ('gt','external','internal'))
    counts={k:WaymoMetrics() for k in arms};cursor=0;trajectory={};elapsed=0.
    if saved is not None:
        check=dict(saved);digest=check.pop('fingerprint',None)
        if digest!=fingerprint(check) or saved['contract_fingerprint']!=fingerprint(contract):raise RuntimeError('ego evaluation resume contract/fingerprint changed')
        cursor=saved['completed_windows']
        if type(cursor)!=int or not 0<=cursor<=len(windows) or set(saved['counts'])!=set(arms):raise ValueError('invalid evaluation cursor/arms')
        for k,arr in saved['counts'].items():
            a=np.asarray(arr)
            if a.dtype.kind not in 'ui' or a.shape!=(3,18,18) or (a<0).any() or not (a.sum((1,2))==cursor*np.prod(source.shape)).all():raise ValueError('corrupt metric prefix')
            counts[k]=WaymoMetrics(a,cursor)
        trajectory=saved['trajectory'];elapsed=saved['seconds']
    device=next(head.parameters()).device;head.eval();checked=set()
    def persist(status):
        s=dict(status=status,completed_windows=cursor,contract_fingerprint=fingerprint(contract),
               counts={k:v.counts.tolist() for k,v in counts.items()},trajectory=trajectory,seconds=elapsed)
        s['fingerprint']=fingerprint(s)
        if save:save(s)
        return s
    persist('running')
    for w in windows[cursor:]:
        if stop_event is not None and stop_event.is_set():break
        tick=time.perf_counter();commands=commands_for(w);predictions={};local_traj={}
        for modality in contract['modalities']:
            # External Pred raw is chosen BEFORE head inference. It contains NO
            # future GT poses, occupancy or visibility. Extractor strips future.
            rec,raw=source.prediction_inputs(w,modality+'_pred')
            times=[source.catalog[t].timestamp/1e6 for t in w.history]
            features=feature_extractor(predictor.provider,rec,raw,head.config,timestamps_s=times)
            se2=head(stack_features([features],device),np.asarray(commands)[None])['se2'][0].cpu().numpy()
            poses=poses_from_se2(raw['history_poses'][-1],se2)
            internal=replace_future_geometry(raw,poses)
            for arm,inputs in (('internal',internal),('external',raw)):
                key=modality+'_'+arm
                _,dense,_,_=predictor.full(rec,inputs,verify=key not in checked)
                predictions[key]=dense;checked.add(key)
            gt_rec,gt_raw=source.prediction_inputs(w,modality+'_gt')
            key=modality+'_gt';_,dense,_,_=predictor.full(gt_rec,gt_raw,verify=key not in checked)
            predictions[key]=dense;checked.add(key)
            # Only diagnostic errors: no GT pose alignment or selection.
            target=relative_se2(raw['history_poses'][-1],gt_raw['future_poses'])
            ext=relative_se2(raw['history_poses'][-1],raw['future_poses'])
            def errors(a):return dict(xy_m=np.linalg.norm(a[:,:2]-target[:,:2],axis=1).tolist(),
                yaw_deg=(np.abs(np.arctan2(np.sin(a[:,2]-target[:,2]),np.cos(a[:,2]-target[:,2])))*180/np.pi).tolist())
            local_traj[modality]=dict(internal=errors(se2),external=errors(ext))
        # Future occupancy labels are read AFTER every full six-frame forecast.
        targets=source.metric_targets(w)
        for k in arms:counts[k].add(predictions[k],targets)
        for m,row in local_traj.items():trajectory.setdefault(m,[]).append(row)
        cursor+=1;elapsed+=time.perf_counter()-tick
        if cursor%8==0:persist('running')
        if progress:progress(dict(window=cursor,total=len(windows),seconds=time.perf_counter()-tick))
    state=persist('complete' if cursor==len(windows) else 'stopped')
    reports={k:v.report() for k,v in counts.items()}
    trajectory_summary={}
    for m,rows in trajectory.items():
        trajectory_summary[m]={}
        for arm in ('internal','external'):
            trajectory_summary[m][arm]={}
            for key in ('xy_m','yaw_deg'):
                x=np.asarray([r[arm][key] for r in rows])
                trajectory_summary[m][arm][key]=dict(mean=x.mean(0).tolist(),median=np.median(x,0).tolist(),p90=np.quantile(x,.9,axis=0).tolist())
    return dict(state=state,reports=reports,trajectory=trajectory_summary,contract=contract,
        declaration='Internal Pred is GT-derived navigation-conditioned, NOT navigation-free. NO GT pose alignment.')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataroot','stc-root','plan-cache','head-checkpoint','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--population',choices=('dev64','dev512','all'),default='dev64')
    p.add_argument('--population-manifest');p.add_argument('--modalities',choices=('occ','occ,stc'),default='occ,stc')
    p.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'configs/real_motion_occfm.yaml'))
    p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-interim',action='store_true',help='explicitly evaluate an unfinished ego-head screen');return p


def main(stop_event=None,argv=None):
    a=parser().parse_args(argv);out=Path(a.out_dir).resolve();head_path=Path(a.head_checkpoint).resolve()
    if not 1<=a.cpu_workers<=8:raise ValueError('bounded CPU worker count required')
    for p in map(lambda x:Path(x).resolve(),(a.dataroot,a.stc_root,a.plan_cache,head_path.parent)):
        if out==p or out.is_relative_to(p):raise ValueError('new evaluation directory outside original inputs/training required')
    if out.exists() and not a.resume:raise FileExistsError('NEW output required, or --resume SAME evaluation directory')
    if a.resume and not (out/'state.json').is_file():raise ValueError('missing evaluation recovery state')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for real full-model evaluation')
    torch.set_num_threads(1);head_sha=digest_file(head_path)
    saved_head=torch.load(head_path,map_location='cpu',weights_only=False)
    if saved_head.get('protocol')!=PROTOCOL:raise ValueError('wrong ego head artifact')
    training=saved_head['contract'];checkpoint=Path(training['frozen_checkpoint'])
    if saved_head['config']!=training['head_config']:raise RuntimeError('head model/training ABI mismatch')
    if not a.allow_interim and (saved_head['epoch']!=training['schedule']['epochs'] or saved_head['cursor']!=0):
        raise ValueError('head screen unfinished; explicit --allow-interim required')
    if implementation_fingerprint(Path(__file__).resolve().parents[2])!=training['feature_geometry_implementation']:
        raise RuntimeError('feature/geometry implementation changed since head training')
    if digest_file(checkpoint)!=training['frozen_sha256']:raise RuntimeError('frozen WM checkpoint mismatch')
    if digest_file(a.config)!=training['runtime_config_sha256']:raise RuntimeError('runtime geometry config changed')
    cfg=EgoHeadConfig(**saved_head['config']);head=HistoryEgoTrajectoryHead(cfg).cuda().eval()
    head.load_state_dict(saved_head['state_dict'],strict=True)
    meta,joint=load_evaluation_model(checkpoint,device='cuda')
    if meta.get('source_epochs')!=list(AVERAGE_EPOCHS):raise RuntimeError('wrong frozen mean recipe')
    source=STCFourSettingSource.from_files(a.dataroot,a.stc_root,a.plan_cache,cache_mib=512)
    parent=None
    if a.population!='all':
        if not a.population_manifest:raise ValueError('fixed dev manifest required')
        manifest,_,_=load_manifest(a.population_manifest);parent=manifest['parent_keys']
    windows,population=source.select(a.population,parent);inventory=source.preflight(windows)
    nusc=NuScenesWindowSource(a.dataroot).nusc
    contract=dict(protocol=PROTOCOL+'_complete_forecast_evaluation',windows=len(windows),population=population,
        frozen_checkpoint_sha256=training['frozen_sha256'],head_sha256=head_sha,head_training_contract=training,
        head_completed_epochs=saved_head['epoch'],head_update=saved_head['updates'],modalities=a.modalities.split(','),
        inventory=inventory,commands=COMMAND_PROTOCOL,thresholds=[.5,None],future_GT_pose_alignment=False,
        runtime_environment={k:v for k,v in sorted(__import__('os').environ.items()) if k.startswith('SWFM_')},
        future_GT_labels='metrics only AFTER all forecasts',GT_derived_navigation_condition=True,
        implementation={p:digest_file(Path(__file__).resolve().parents[2]/p) for p in
            ('real_motion/ego_trajectory_head.py','real_motion/ego_navigation.py','tools/real_motion/ego_trajectory_common.py',
             'tools/real_motion/eval_p0_f9_surface_ego_head.py')})
    old=None
    if a.resume:
        if json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:raise RuntimeError('evaluation contract changed')
        old=json.loads((out/'state.json').read_text(encoding='utf-8'))
    else:out.mkdir(parents=True);write_json(out/'contract.json',contract)
    pcfg=make_prepare_config(load_runtime_config(a.config));predictor=Predictor(joint,pcfg,'cuda',workers=a.cpu_workers,geometry_mib=512)
    try:
        with evaluation_lock(out):
            result=evaluate(source,windows,predictor,head,lambda w:navigation_commands(nusc,w.future),contract,
                saved=old,save=lambda s:write_json(out/'state.json',s),stop_event=stop_event,
                progress=lambda r:print('EGO_COMPLETE_FORECAST '+json.dumps(r),flush=True))
    finally:predictor.close()
    if digest_file(checkpoint)!=training['frozen_sha256'] or digest_file(head_path)!=head_sha:raise RuntimeError('source checkpoint changed during evaluation')
    write_json(out/'evaluation.json',result)
    lines=['===== FROZEN WM / INTERNAL EGO HEAD =====',f'status={result["state"]["status"]}; windows={result["state"]["completed_windows"]}/{len(windows)}',
        'Only ego head trained; existing WM/CCR frozen. FOUR history / SIX futures; ADD0.5 / REMOVEoff.',
        'Internal ego uses GT-derived destination navigation commands. NOT navigation-free; NO GT pose alignment.',
        'setting       horizon       IoU      mIoU']
    for name,report in result['reports'].items():
        for h,r in [*report['horizons'].items(),('Avg',report['average'])]:
            fmt=lambda v:'NA' if v is None else f'{v:.6f}'
            lines.append(f'{name:14s} {h:>6s} {fmt(r["IoU"]):>10s} {fmt(r["standard_mIoU"]):>10s}')
    lines+=['trajectory_errors='+json.dumps(result['trajectory']),f'seconds={result["state"]["seconds"]:.2f}',
        'Diagnostic dev population, no automatic best-head selection, promotion, threshold tuning or retry.']
    summary='\n'.join(lines)+'\n';(out/'summary.txt').write_text(summary,encoding='utf-8');print(summary)
    return 0 if result['state']['status']=='complete' else 130


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
