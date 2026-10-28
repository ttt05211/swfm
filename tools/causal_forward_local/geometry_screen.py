"""Input-only temporal-prior and multi-history-velocity candidates, latest body."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.shared_source_track import kta_and_anchors
from tools.causal_forward_all.common import Epoch2Predictor
from tools.shared_history_experiments.refine.geometry import TransportPrior,centers,inverse_grid
from tools.stc_overnight.bundle import Bundle
from .portable import load_weights
from .cache import AT,write
from .calibrate import confusion,scores


def velocity_row(row,amount):
    rec,raw,ego,history=row;track=raw['_shared_source_track'];velocities={k:np.array(v,copy=True) for k,v in track.pair_velocity.items()}
    for i,original in velocities.items():
        good=track.track_valid[i]
        if good.sum()<3:continue
        t=track.times[good];p=track.centers[i,good];u=t-t.mean()
        fitted=(u[:,None]*(p-p.mean(0))).sum(0)/(u*u).sum();fitted[2]=0.
        velocities[i]=(1-amount)*original+amount*fitted
    updated=replace(track,pair_velocity=velocities)
    _,kta,anchors=kta_and_anchors(history[0]['state']['current'],updated,history[0]['state']['current_pose'])
    return dict(rec,kta_displacement_xy_m=kta,anchors_xy_t0_m=anchors),dict(raw,_shared_source_track=updated),ego,history


def position_row(row,amount=.5,velocity_amount=.5):
    """Causal CV current-state estimate; historical feature measurements NEVER shift.

    Current shape hypothesis and future anchors move together. Direct Memory
    still starts at actual historical points/centers and reaches the NEW target,
    hence the position correction is applied ONCE, not twice.
    """
    rec,raw,ego,history=velocity_row(row,velocity_amount);track=raw['_shared_source_track']
    causal=dict(history[0]);state=dict(causal['state']);offset=np.zeros((len(state['current']),3),np.float64)
    for i in track.pair_velocity:
        good=track.track_valid[i]
        if good.sum()<3:continue
        t=track.times[good];p=track.centers[i,good];u=t-t.mean()
        slope=(u[:,None]*(p-p.mean(0))).sum(0)/(u*u).sum()
        intercept=p.mean(0)-slope*t.mean();offset[i,:2]=amount*(intercept-track.centers[i,-1])[:2]
    state['current']=[dict(s,centroid_world=np.asarray(s['centroid_world'])+offset[i]) for i,s in enumerate(state['current'])]
    state['source_world_points']=[np.asarray(v)+offset[i] for i,v in enumerate(state['source_world_points'])]
    state['source_rel_xy']=[v[:,:2]-state['current'][i]['centroid_world'][None,:2] for i,v in enumerate(state['source_world_points'])]
    native,kta,anchors=kta_and_anchors(state['current'],track,state['current_pose']);state['source_z_t0']=native[:,2]
    causal['state']=state;fixed=dict(raw['_column_causal_preparation'],prepared_state=state)
    rec=dict(rec,source_centroid_xy_t0_m=native[:,:2].astype(np.float32),kta_displacement_xy_m=kta,anchors_xy_t0_m=anchors)
    # shared_reads, memory_reads.world and memory_reads.centers_world remain the
    # ACTUAL observed evidence. Altering them here would double the offset.
    return rec,dict(raw,_column_causal_preparation=fixed),ego,(causal,*history[1:])


@torch.no_grad()
def static_probability(row,grid,device,fallback):
    """Ego-align native historical static predictions to t0; no GT geometry/labels."""
    raw=row[1];sem=torch.as_tensor(raw['history_occ'],device=device).long()
    valid=torch.as_tensor(raw['history_observed'],device=device).bool()&(sem<18)
    for cls in DYNAMIC_CLASS_IDS:valid&=sem!=cls
    poses=torch.as_tensor(np.asarray(raw['history_poses']),device=device,dtype=torch.float64);query=centers(grid,device)
    total=torch.zeros_like(fallback);mass=torch.zeros_like(fallback[:,:1])
    for f in range(4):
        probability=F.one_hot(sem[f].clamp(0,17),18).permute(3,0,1,2).float()*valid[f][None]
        sampling=inverse_grid(query,torch.linalg.inv(poses[f])@poses[-1],grid)[None]
        votes=F.grid_sample(probability[None],sampling,mode='nearest',align_corners=False,padding_mode='zeros')
        total+=votes;mass+=votes.sum(1,keepdim=True)
    return torch.where(mass>0,total/mass.clamp_min(1),fallback)


@torch.no_grad()
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--weights',default='../_received/forward_adaln_epoch2_local.zip')
    ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--domains',nargs='+',default=['stc','gt']);ap.add_argument('--state-center-filter',action='store_true');a=ap.parse_args(argv)
    torch.set_num_threads(1);torch.manual_seed(21);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'counts').mkdir(exist_ok=True)
    body,c,info=load_weights(a.weights,a.device);b=Bundle(a.bundle,cache_mib=32);p=Epoch2Predictor(body,info['prepare_config'],a.device,2)
    free=json.loads((Path(a.cache)/'calibration/fit.json').read_text())['chosen_free_bias']
    names=(['raw','position_half','position_velocity_half'] if a.state_center_filter else
        ['raw','temporal_prior25','temporal_prior50','velocity_halfLS','velocity_fullLS'])
    all_names=names+[n+'_free' for n in names]
    contract=dict(protocol='latest_forward_input_only_temporal_and_velocity_screen_v1',source_sha256=info['manifest']['source_checkpoint_sha256'],
        held_keys=c['eval_keys'][32:],GT_future_ego=True,known='dense_valid',full_grid=True,Moving_available=False,
        geometry_methods='static probability mean4 with .25/.5 t0-anchored mixture; pair vs LS velocity .5/1',
        state_center_filter=a.state_center_filter,
        identity='original forward identity; no new matching or missing last-pair velocity',
        geometry_target_future_labels_used=False,readout_free_bias=free,readout_free_bias_GT_fitted_on_first32=True,
        raw_weights_unchanged=True,threshold_search=False,FPS=False)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:raise ValueError('new geometry screen output required')
    write(out/'contract.json',contract);totals={};tic=time.monotonic()
    try:
        for domain in a.domains:
            for i in range(32,64):
                name=f'{domain}_{i:03d}';path=out/'counts'/(name+'.json')
                if path.exists():item=json.loads(path.read_text())
                else:
                    r=b.by_key[tuple(c['eval_keys'][i])];rec,raw,times=b.history(r,domain,'dense_valid');row=p.prepare(rec,raw,times=times)
                    base=p.forward(row,poses=np.asarray(r['future_poses']));pred={}
                    def collect(n,logits):
                        pred[n]=logits[list(AT)].argmax(1).byte().cpu().numpy()
                        occupied,classes=logits[:,:17].max(1);calibrated=torch.where(occupied>=logits[:,17]+free,classes,17)
                        pred[n+'_free']=calibrated[list(AT)].byte().cpu().numpy()
                    collect('raw',base['logits'])
                    if a.state_center_filter:
                        del base
                        for velocity,n in ((0.,'position_half'),(.5,'position_velocity_half')):
                            fw=p.forward(position_row(row,.5,velocity),poses=np.asarray(r['future_poses']));collect(n,fw['logits']);del fw
                    else:
                        renderer=TransportPrior(row[1],base['state'],p.provider.pcfg.grid,a.device)
                        old=renderer.static;pooled=static_probability(row,p.provider.pcfg.grid,a.device,old)
                        for amount,n in ((.25,'temporal_prior25'),(.5,'temporal_prior50')):
                            renderer.static=(1-amount)*old+amount*pooled
                            prior=renderer.render(base['poses'],base['targets'],base['yaws'])
                            logits=body.columns(base['memory'],prior,range(6),base['shape']);collect(n,logits);del prior,logits
                        del renderer,old,pooled,base
                        for amount,n in ((.5,'velocity_halfLS'),(1.,'velocity_fullLS')):
                            fw=p.forward(velocity_row(row,amount),poses=np.asarray(r['future_poses']));collect(n,fw['logits']);del fw
                    # Every candidate completes six futures before any GT occupancy access.
                    gt=b.targets(r).reshape(3,-1);count={n:[confusion(v[h],gt[h]).tolist() for h in range(3)] for n,v in pred.items()}
                    item=dict(key=[r['scene_name'],r['t0_token']],confusion=count);write(path,item)
                    del pred,gt,row,raw
                for split in ('held32','screen16' if i<48 else 'confirmation16'):
                    key=domain+'/'+split;entry=totals.setdefault(key,dict(windows=0,counts={n:np.zeros((3,18,18),np.int64) for n in all_names}))
                    entry['windows']+=1
                    for n,q in item['confusion'].items():entry['counts'][n]+=np.asarray(q,np.int64)
                if (i+1)%4==0:
                    reports={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
                    write(out/'results.json',dict(status='running',contract=contract,reports=reports))
                    print('GEOMETRY_SCREEN',name,'seconds',round(time.monotonic()-tic,1),
                        {n:v['average'] for n,v in reports[domain+'/held32']['reports'].items() if not n.endswith('_free')},flush=True)
        reports={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
        write(out/'results.json',dict(status='complete',contract=contract,reports=reports))
    finally:p.close();b.close()


if __name__=='__main__':main()
