#!/usr/bin/env python3
"""One fixed dev64 pass: prior/old320/A/B, external planner and GT reference."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import signal
import threading
import time

import numpy as np
import torch

from real_motion.ego_trajectory_head import HistoryEgoTrajectoryHead,EgoHeadConfig
from real_motion.ego_navigation import navigation_commands,relative_se2,poses_from_se2,replace_future_geometry
from real_motion.stc_camera_protocol import STCFourSettingSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.waymo_i2world import WaymoMetrics
from real_motion.runtime_config import load_runtime_config,make_prepare_config
from tools.real_motion.ego_trajectory_common import (
    extract_history_features,stack_features,digest_file,fingerprint,implementation_fingerprint,
)
from tools.real_motion.surface_ego_ablation_common import (
    PROTOCOL,HEAD_NAMES,TRAIN_NAMES,historical_prior,verify_sources,trajectory_report,
)
from tools.real_motion.joint_surface_checkpoint_selection import load_evaluation_model
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.waymo_zero_shot_common import write_json

ROOT=Path(__file__).resolve().parents[2]


@torch.no_grad()
def evaluate(source,windows,predictor,models,commands_for,contract,*,saved=None,save=None,
             stop_event=None,progress=None,feature_extractor=extract_history_features):
    routes=('gt','external',*HEAD_NAMES)
    arms=tuple(f'{m}_{r}' for m in contract['modalities'] for r in routes)
    counts={k:WaymoMetrics() for k in arms};cursor=0;errors={};elapsed=0.;pose_reuses=0
    if saved is not None:
        checked=dict(saved);digest=checked.pop('fingerprint',None)
        if digest!=fingerprint(checked) or saved['contract_fingerprint']!=fingerprint(contract):
            raise RuntimeError('paired evaluation contract/state fingerprint changed')
        cursor=saved['completed_windows']
        if type(cursor)!=int or not 0<=cursor<=len(windows) or set(saved['counts'])!=set(arms):
            raise RuntimeError('invalid complete-window evaluation cursor/arms')
        for k,arr in saved['counts'].items():
            a=np.asarray(arr)
            if (a.dtype.kind not in 'ui' or a.shape!=(3,18,18) or (a<0).any()
                    or not (a.sum((1,2))==cursor*np.prod(source.shape)).all()):
                raise RuntimeError('corrupt paired integer metric prefix')
            counts[k]=WaymoMetrics(a,cursor)
        errors=saved['errors'];elapsed=saved['seconds'];pose_reuses=saved['identical_pose_reuses']
        expected={f'{m}_{r}' for m in contract['modalities'] for r in ('external',*HEAD_NAMES)}
        if (cursor and set(errors)!=expected) or any(len(v)!=cursor for v in errors.values()):
            raise RuntimeError('trajectory prefix does not match complete windows')
    device=next(models['old320'].parameters()).device
    for model in models.values():model.eval()
    verified=set()
    def persist(status):
        state=dict(status=status,completed_windows=cursor,contract_fingerprint=fingerprint(contract),
            counts={k:v.counts.tolist() for k,v in counts.items()},errors=errors,seconds=elapsed,
            identical_pose_reuses=pose_reuses)
        state['fingerprint']=fingerprint(state)
        if save:save(state)
        return state
    persist('running')
    for window in windows[cursor:]:
        if stop_event is not None and stop_event.is_set():break
        tick=time.perf_counter();commands=np.asarray(commands_for(window));dense={};local_errors={};reuses=0
        for modality in contract['modalities']:
            rec,raw=source.prediction_inputs(window,modality+'_pred')
            times=[source.catalog[t].timestamp/1e6 for t in window.history]
            features=feature_extractor(predictor.provider,rec,raw,models['old320'].config,timestamps_s=times)
            bank=stack_features([features],device)
            trajectories={'prior':historical_prior(bank)[0].cpu().numpy()}
            trajectories.update({name:model(bank,commands[None])['se2'][0].cpu().numpy() for name,model in models.items()})
            # Compute every learned head BEFORE loading the GT-conditioned route.
            raws={name:replace_future_geometry(raw,poses_from_se2(raw['history_poses'][-1],se2))
                  for name,se2 in trajectories.items()}
            raws['external']=raw
            _,gt_raw=source.prediction_inputs(window,modality+'_gt');raws['gt']=gt_raw
            completed={}
            for route in (*HEAD_NAMES,'external','gt'):
                inputs=raws[route];key=f'{modality}_{route}'
                pose_key=np.asarray(inputs['future_poses'],np.float64).tobytes()
                if pose_key in completed:
                    dense[key]=completed[pose_key];reuses+=1
                else:
                    _,prediction,_,_=predictor.full(rec,inputs,verify=key not in verified)
                    dense[key]=prediction;completed[pose_key]=prediction
                verified.add(key)
            target=relative_se2(raw['history_poses'][-1],gt_raw['future_poses'])
            trajectories['external']=relative_se2(raw['history_poses'][-1],raw['future_poses'])
            for route,pred in trajectories.items():
                local_errors[f'{modality}_{route}']=dict(pred=pred.tolist(),target=target.tolist(),commands=commands.tolist())
        # No future occupancy/mask scoring read before ALL 12 predictions finish.
        targets=source.metric_targets(window)
        deltas={k:WaymoMetrics() for k in arms}
        for key in arms:deltas[key].add(dense[key],targets)
        # All-or-nothing integer and trajectory commit for this window.
        for key in arms:counts[key].counts+=deltas[key].counts;counts[key].windows+=1
        for key,row in local_errors.items():errors.setdefault(key,[]).append(row)
        cursor+=1;pose_reuses+=reuses;elapsed+=time.perf_counter()-tick
        if cursor%8==0:persist('running')
        if progress:progress(dict(window=cursor,total=len(windows),seconds=time.perf_counter()-tick))
    state=persist('complete' if cursor==len(windows) else 'stopped')
    trajectory={k:trajectory_report(np.asarray([r['pred'] for r in rows]),np.asarray([r['target'] for r in rows]),
                    np.asarray([r['commands'] for r in rows])) for k,rows in errors.items() if rows}
    return dict(state=state,reports={k:v.report() for k,v in counts.items()},trajectory=trajectory,contract=contract)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataroot','stc-root','plan-cache','pair-dir','out-dir','population-manifest'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--config',default=str(ROOT/'configs/real_motion_occfm.yaml'))
    p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    return p


def main(stop_event=None,argv=None):
    a=parser().parse_args(argv);out=Path(a.out_dir).resolve();pairdir=Path(a.pair_dir).resolve()
    if not 1<=a.cpu_workers<=8:raise ValueError('bounded CPU workers required')
    paths=(a.dataroot,a.stc_root,a.plan_cache,pairdir)
    if any(out==Path(p).resolve() or out.is_relative_to(Path(p).resolve()) for p in paths):
        raise ValueError('separate NEW evaluation directory required')
    if out.exists() and not a.resume:raise FileExistsError('NEW output required or --resume SAME directory')
    if a.resume and not (out/'state.json').is_file():raise ValueError('missing paired evaluation recovery')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for actual dev64 evaluation')
    torch.set_num_threads(1);pairpath=pairdir/'pair_last.pt';pair_sha=digest_file(pairpath)
    saved=torch.load(pairpath,map_location='cpu',weights_only=False);training=saved['contract']
    if (saved.get('protocol')!=PROTOCOL or saved['updates']!=training['schedule']['max_updates']
            or saved['config']!=training['head_config']):
        raise RuntimeError('completed fixed-budget A/B checkpoint required')
    if training!=json.loads((pairdir/'contract.json').read_text(encoding='utf-8')):
        raise RuntimeError('paired checkpoint/contract mismatch')
    audit=training['source'];verify_sources(audit)
    if implementation_fingerprint(ROOT)!=audit['current_implementation']:
        raise RuntimeError('paired feature/evaluation code changed since training')
    original=audit['original_training']
    env={k:v for k,v in sorted(__import__('os').environ.items()) if k.startswith('SWFM_')}
    if env!=original['runtime_environment']:
        raise RuntimeError('feature execution flags differ from source bank; use the wrapper to restore the original child environment')
    if digest_file(a.config)!=original['runtime_config_sha256']:raise RuntimeError('runtime geometry config changed')
    old=torch.load(Path(audit['source_dir'])/'head_last.pt',map_location='cpu',weights_only=False)
    cfg=EgoHeadConfig(**saved['config']);models={k:HistoryEgoTrajectoryHead(cfg).cuda().eval() for k in ('old320',*TRAIN_NAMES)}
    models['old320'].load_state_dict(old['state_dict'],strict=True)
    for name in TRAIN_NAMES:models[name].load_state_dict(saved['models'][name],strict=True)
    _,joint=load_evaluation_model(audit['frozen_checkpoint'],device='cuda')
    source=STCFourSettingSource.from_files(a.dataroot,a.stc_root,a.plan_cache,cache_mib=512)
    manifest,_,_=load_manifest(a.population_manifest)
    windows,population=source.select('dev64',manifest['parent_keys']);inventory=source.preflight(windows)
    nusc=NuScenesWindowSource(a.dataroot).nusc
    contract=dict(protocol=PROTOCOL+'_dev64',training=training,pair_checkpoint_sha256=pair_sha,
        windows=len(windows),population=population,inventory=inventory,modalities=['occ','stc'],
        routes=['gt','external',*HEAD_NAMES],thresholds=[.5,None],GT_derived_destination_navigation=True,
        no_GT_pose_alignment=True,future_occupancy='scoring only AFTER all predictions',
        runtime_environment=env)
    previous=None
    if a.resume:
        if json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:raise RuntimeError('paired eval contract changed')
        previous=json.loads((out/'state.json').read_text(encoding='utf-8'))
    else:out.mkdir(parents=True);write_json(out/'contract.json',contract)
    predictor=Predictor(joint,make_prepare_config(load_runtime_config(a.config)),'cuda',workers=a.cpu_workers,geometry_mib=512)
    try:
        with evaluation_lock(out):
            result=evaluate(source,windows,predictor,models,lambda w:navigation_commands(nusc,w.future),contract,
                saved=previous,save=lambda s:write_json(out/'state.json',s),stop_event=stop_event,
                progress=lambda s:print('EGO_AB_DEV64 '+json.dumps(s),flush=True))
            verify_sources(audit)
            if digest_file(pairpath)!=pair_sha:raise RuntimeError('paired weights changed during read-only evaluation')
            write_json(out/'evaluation.json',result)
            summary=summary_text(result,pairdir)
            (out/'summary.txt').write_text(summary,encoding='utf-8');print(summary,flush=True)
    finally:predictor.close()
    return 0 if result['state']['status']=='complete' else 130


def summary_text(result,pairdir):
    training=(Path(pairdir)/'training_summary.txt').read_text(encoding='utf-8')
    lines=[training.rstrip(),'','===== FIXED DEV64 / PAIRED EGO HEAD =====',
        f'status={result["state"]["status"]}; windows={result["state"]["completed_windows"]}',
        'GT-derived navigation-conditioned Pred; no alignment/mask/threshold tuning; WM/CCR frozen.',
        'setting           horizon       IoU      mIoU']
    fmt=lambda v:'NA' if v is None else f'{v:.6f}'
    for name,r in result['reports'].items():
        for h,row in [*r['horizons'].items(),('Avg',r['average'])]:
            lines.append(f'{name:18} {h:>6} {fmt(row["IoU"]):>10} {fmt(row["standard_mIoU"]):>10}')
    lines+=['','===== DEV64 TRAJECTORY ERRORS =====','candidate          ADE_m    FDE_3s_m    XY@1/2/3s_m   yaw@1/2/3s_deg']
    for name,r in result['trajectory'].items():
        xy='/'.join(f'{r["xy_mean_m"][i]:.3f}' for i in (1,3,5))
        yaw='/'.join(f'{r["yaw_mean_deg"][i]:.3f}' for i in (1,3,5))
        lines.append(f'{name:18} {r["ADE_m"]:.6f} {r["FDE_3s_m"]:.6f} {xy} {yaw}')
    lines += [f'seconds={result["state"]["seconds"]:.2f}',
        'A vs old320 tests budget+cosine duration; B vs A tests the supervised objective.',
        'Prior and TRAIN/dev gap are diagnostics, not a deployable selection rule.',
        'No dev-best checkpoint, automatic retry, full-data training or promotion. Quality eval time is NOT FPS.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
