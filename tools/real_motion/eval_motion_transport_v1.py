#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.motion_transport_v1.config import load_config,get
from real_motion.motion_transport_v1.data import ManifestDataset,load_manifest
from real_motion.motion_transport_v1.model import MotionTransportV1
from real_motion.motion_transport_v1.evaluation import evaluate
from real_motion.motion_transport_v1.routing import sha256_file,route_sources,infer_source_scores
from real_motion.motion_transport_v1.source_adapter import decompose_strong_sources,extract_original_msp_candidates,map_msp_to_sources
from real_motion.motion_transport_v1.crop_history import build_history_crops
from real_motion.motion_transport_v1.compositor import hard_kta_identity,compose_hard
from real_motion.motion_transport_v1.engine import _hash_json,_stable_config,seed_all

def _budgets(s):return tuple('all' if x.strip()=='all' else int(x) for x in s.split(',') if x.strip())
def _stats(x):
    a=np.asarray(x,float);return {'mean_ms':float(a.mean()*1000) if len(a) else float('nan'),'p50_ms':float(np.quantile(a,.5)*1000) if len(a) else float('nan'),'p95_ms':float(np.quantile(a,.95)*1000) if len(a) else float('nan')}
def _sum_mapping(maps):
    out={}
    for m in maps:
        for k,v in m.items():out[int(k)]=out.get(int(k),0)+int(v)
    return out
def _miou(inter,union):
    vals=[inter.get(k,0)/u for k,u in union.items() if u>0];return float(np.mean(vals)) if vals else float('nan')
def bootstrap_delta(per_scene,*,baseline='hard:0',candidate='hard:16',repeats=2000,seed=3407):
    scenes=[s for s,r in per_scene.items() if baseline in r and candidate in r]
    if not scenes:return None
    rng=np.random.default_rng(seed);out={'overall':[],'moving':[]}
    for _ in range(int(repeats)):
        sample=rng.choice(scenes,size=len(scenes),replace=True)
        for metric in ('overall','moving'):
            vals=[]
            for key in (baseline,candidate):
                hs=[]
                for h in ('1.0','2.0','3.0'):
                    inter=_sum_mapping([per_scene[s][key][metric][h]['inter'] for s in sample]);union=_sum_mapping([per_scene[s][key][metric][h]['union'] for s in sample]);hs.append(_miou(inter,union))
                vals.append(float(np.nanmean(hs)))
            out[metric].append(100.0*(vals[1]-vals[0]))
    return {m:{'mean_pp':float(np.mean(v)),'ci95_pp':[float(np.quantile(v,.025)),float(np.quantile(v,.975))],'p_gt_0':float(np.mean(np.asarray(v)>0))} for m,v in out.items()}
@torch.no_grad()
def latency_profile(pipe,src,ds,cfg,budgets,*,warmup=100,windows=500,seed=3407):
    limit=min(len(ds),int(warmup)+int(windows));stage={str(q):{k:[] for k in ('input_load','kta_source','msp','route','crop','network','hard_compose','end_to_end')} for q in budgets};calls={str(q):0 for q in budgets};peak={str(q):0. for q in budgets}
    for di in range(limit):
        t0=time.perf_counter();w,c=ds[di];input_t=time.perf_counter()-t0;t=time.perf_counter();d=decompose_strong_sources(c,grid=pipe.grid,cfg=pipe.strong_cfg,frame_dt_s=float(get(cfg,'input.dt_seconds',.5)),crop_radius_limit_m=float(get(cfg,'crop.max_source_xy_radius_about_centroid_m',11.2)));kta_t=time.perf_counter()-t;t=time.perf_counter();cand=extract_original_msp_candidates(c,grid=pipe.grid,motion_cfg=pipe.motion_cfg,kta_cfg=pipe.msp_kta_cfg);map_msp_to_sources(d,cand,shape_xyz=tuple(pipe.grid.shape_hwd));scores=infer_source_scores(cand,d.sources,pipe.msp,pipe.device);msp_t=time.perf_counter()-t
        for q in budgets:
            key=str(q);start=time.perf_counter();t=time.perf_counter();sel=route_sources(d.sources,q,strategy='msp',scores=scores,seed=seed+di);route_t=time.perf_counter()-t
            if pipe.device.type=='cuda':torch.cuda.reset_peak_memory_stats(pipe.device)
            if not sel:crop_t=net_t=0.;t=time.perf_counter();_=hard_kta_identity(c,d,grid=pipe.grid);comp_t=time.perf_counter()-t
            else:
                t=time.perf_counter();crops=build_history_crops(c,d.sources,sel,grid=pipe.grid,crop_shape_xyz=tuple(get(cfg,'crop.shape_xyz',[64,64,16])),xy_resolution_m=float(get(cfg,'crop.xy_resolution_m',.4)),mirror=False,device=pipe.device);crop_t=time.perf_counter()-t
                if pipe.device.type=='cuda':torch.cuda.synchronize(pipe.device)
                t=time.perf_counter();delta,_=pipe.source_network(crops,source_microbatch=int(get(cfg,'training.source_microbatch',16)))
                if pipe.device.type=='cuda':torch.cuda.synchronize(pipe.device)
                net_t=time.perf_counter()-t;t=time.perf_counter();_=compose_hard(c,d,delta,sel,grid=pipe.grid);comp_t=time.perf_counter()-t
            total=time.perf_counter()-start+input_t+kta_t+msp_t
            if di>=warmup:
                for n,v in [('input_load',input_t),('kta_source',kta_t),('msp',msp_t),('route',route_t),('crop',crop_t),('network',net_t),('hard_compose',comp_t),('end_to_end',total)]:stage[key][n].append(v)
                calls[key]+=len(sel);peak[key]=max(peak[key],torch.cuda.max_memory_allocated(pipe.device)/(1024**3) if pipe.device.type=='cuda' else 0.)
    report={q:{'timing':{k:_stats(v) for k,v in st.items()},'stpn_source_calls':calls[q],'peak_memory_gib':peak[q]} for q,st in stage.items()}
    if '16' in report and 'all' in report:
        a=report['16']['timing']['end_to_end']['mean_ms'];b=report['all']['timing']['end_to_end']['mean_ms'];report['q16_latency_reduction_vs_all']=1-a/b if b>0 else float('nan')
    return report
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--override',action='append',default=[]);p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--split',default='dev');p.add_argument('--weights',choices=('ema','raw'),default='ema');p.add_argument('--budgets',default='0,4,8,16,32,all');p.add_argument('--strategies',default='msp');p.add_argument('--max-windows',type=int,default=None);p.add_argument('--latency',action='store_true');p.add_argument('--latency-warmup',type=int,default=100);p.add_argument('--latency-windows',type=int,default=500);a=p.parse_args();cfg=load_config(a.config,a.override);seed=int(get(cfg,'training.initial_seed',3407));seed_all(seed);manifest,msp=get(cfg,'paths.manifest'),get(cfg,'paths.msp_checkpoint');ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck.get('spec_version')!='MT-V1-SPEC-2' or ck.get('config_hash')!=_hash_json(_stable_config(cfg)):raise RuntimeError('checkpoint spec/config mismatch')
    if ck.get('manifest_sha256')!=sha256_file(manifest) or ck.get('msp_sha256')!=sha256_file(msp):raise RuntimeError('checkpoint provenance mismatch')
    device=torch.device('cuda',0) if torch.cuda.is_available() else torch.device('cpu');src=NuScenesWindowSource(get(cfg,'paths.dataroot'),info_pkl=get(cfg,'paths.info_pkl'),verbose=False);ds=ManifestDataset(src,load_manifest(manifest),a.split);pipe=MotionTransportV1(cfg,msp_checkpoint=msp,device=device)
    if a.weights=='raw':pipe.source_network.load_state_dict(ck['model_state_dict'],strict=True)
    else:
        st=pipe.source_network.state_dict()
        for k,v in ck['ema_state_dict']['shadow'].items():
            if k in st:st[k].copy_(v.to(dtype=st[k].dtype,device=st[k].device))
    pipe.source_network.eval();budgets=_budgets(a.budgets);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);allrep={}
    for strategy in [x.strip() for x in a.strategies.split(',') if x.strip()]:
        rep=evaluate(pipe,src,ds,cfg,budgets=budgets,strategy=strategy,include_soft_main=(16 in budgets),seed=seed,max_windows=a.max_windows)
        if strategy=='msp' and '0' in rep['hard'] and '16' in rep['hard'] and a.max_windows is None:rep['scene_bootstrap_q16_vs_kta']=bootstrap_delta(rep['per_scene_confusion'],repeats=int(get(cfg,'evaluation.bootstrap_repeats',2000)),seed=seed)
        allrep[strategy]=rep
    (out/'metrics.json').write_text(json.dumps(allrep,indent=2))
    if a.latency:(out/'latency.json').write_text(json.dumps(latency_profile(pipe,src,ds,cfg,budgets,warmup=a.latency_warmup,windows=a.latency_windows,seed=seed),indent=2))
    print(json.dumps({k:{'delta_vs_kta':v.get('delta_vs_kta'),'bootstrap':v.get('scene_bootstrap_q16_vs_kta')} for k,v in allrep.items()},indent=2))
if __name__=='__main__':main()
