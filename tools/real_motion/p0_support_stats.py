#!/usr/bin/env python3
"""P0-B: per-horizon arrival coverage, sparsity, and window connectivity."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from real_motion.prepared import PreparedShardDataset
from real_motion.support import build_motion_tube,MotionTubeConfig,downsample_support
from real_motion.windows import WindowPlanner,window_coverage
from real_motion.runtime_config import add_config_args,load_runtime_config,get_cfg,save_resolved_config

def ratio(n,d):return float(n/d) if d else 1.0
def summarize(samples,radii,extra,schedule,window,maxw):
    F=6;mk=lambda:[{'inter':0,'gt':0,'active':0,'dense':0,'l_inter':0,'l_gt':0,'l_active':0,'l_dense':0} for _ in range(F)];stats={r:mk() for r in radii};scheduled=mk();moving=[{'moving_vox':0,'occupied_vox':0,'moving_bev':0,'dense_bev':0,'moving_latent':0,'dense_latent':0} for _ in range(F)];planner=WindowPlanner((window,window),maxw);wr=[]
    for s in samples:
        kta=torch.from_numpy(np.asarray(s['kta_support'])).bool();fg=np.asarray(s['future_gt_occ']);fm=np.asarray(s['future_moving_occ']);gt=torch.from_numpy((fm!=17).any(axis=-1))
        for h in range(F):
            moving[h]['moving_vox']+=int((fm[h]!=17).sum());moving[h]['occupied_vox']+=int((fg[h]!=17).sum());mb=gt[h];moving[h]['moving_bev']+=int(mb.sum());moving[h]['dense_bev']+=mb.numel();ml=downsample_support(mb.unsqueeze(0),(50,50),extra_radius=extra)[0];moving[h]['moving_latent']+=int(ml.sum());moving[h]['dense_latent']+=ml.numel()
        tube=build_motion_tube(kta,MotionTubeConfig(radii=list(schedule),latent_extra_radius=0));gl=downsample_support(gt,(50,50),extra_radius=extra);tl=downsample_support(tube,(50,50),extra_radius=extra);hl=downsample_support(torch.from_numpy(np.asarray(s['history_candidate_support'])).bool(),(50,50),extra_radius=extra);req=tl.unsqueeze(0);ctx=torch.cat([hl,tl],0).unsqueeze(0);plan=planner.plan(req,context_support=ctx);hu=hl.any(0);ru=tl.any(0);conn=torch.zeros_like(ru);nw=int(plan.valid.sum());withh=0
        for ki in range(plan.valid.shape[1]):
            if not bool(plan.valid[0,ki]):continue
            y,x=[int(v) for v in plan.origins[0,ki].tolist()];has=bool(hu[y:y+window,x:x+window].any())
            if has:withh+=1;conn[y:y+window,x:x+window]|=ru[y:y+window,x:x+window]
        rc=int(ru.sum());wr.append({'future_window_coverage':float(window_coverage(req,plan)[0]),'history_plus_future_context_coverage':float(window_coverage(ctx,plan)[0]),'future_windows_with_any_history_ratio':withh/nw if nw else 1.0,'future_required_cells_in_history_connected_windows_ratio':float(conn.sum())/rc if rc else 1.0,'num_windows':nw,'slot_compute_ratio':nw*window*window/2500.0})
        for h in range(F):
            d=scheduled[h];d['inter']+=int((gt[h]&tube[h]).sum());d['gt']+=int(gt[h].sum());d['active']+=int(tube[h].sum());d['dense']+=tube[h].numel();d['l_inter']+=int((gl[h]&tl[h]).sum());d['l_gt']+=int(gl[h].sum());d['l_active']+=int(tl[h].sum());d['l_dense']+=tl[h].numel()
        for r in radii:
            tr=build_motion_tube(kta,MotionTubeConfig(radii=[r]*F,latent_extra_radius=0));trl=downsample_support(tr,(50,50),extra_radius=extra)
            for h in range(F):
                d=stats[r][h];d['inter']+=int((gt[h]&tr[h]).sum());d['gt']+=int(gt[h].sum());d['active']+=int(tr[h].sum());d['dense']+=tr[h].numel();d['l_inter']+=int((gl[h]&trl[h]).sum());d['l_gt']+=int(gl[h].sum());d['l_active']+=int(trl[h].sum());d['l_dense']+=trl[h].numel()
    out={'constant_radius_scan':{},'true_moving_sparsity':[],'scheduled_radius':[],'proposed_window_backend':{}}
    for r in radii:out['constant_radius_scan'][str(r)]=[{'horizon_s':.5*(h+1),'coverage_bev':ratio(d['inter'],d['gt']),'active_ratio_bev':ratio(d['active'],d['dense']),'coverage_latent':ratio(d['l_inter'],d['l_gt']),'active_ratio_latent':ratio(d['l_active'],d['l_dense'])} for h,d in enumerate(stats[r])]
    for h,d in enumerate(scheduled):out['scheduled_radius'].append({'horizon_s':.5*(h+1),'radius':int(schedule[h]),'coverage_bev':ratio(d['inter'],d['gt']),'active_ratio_bev':ratio(d['active'],d['dense']),'coverage_latent':ratio(d['l_inter'],d['l_gt']),'active_ratio_latent':ratio(d['l_active'],d['l_dense'])})
    if wr:
        for k in wr[0]:
            v=[r[k] for r in wr];out['proposed_window_backend'][k]={'mean':float(np.mean(v)),'p05':float(np.quantile(v,.05)),'min':float(np.min(v)),'max':float(np.max(v))}
        out['proposed_window_backend'].update({'window_hw':[window,window],'max_windows':maxw})
    for h,d in enumerate(moving):out['true_moving_sparsity'].append({'horizon_s':.5*(h+1),'moving_voxel_over_occupied':ratio(d['moving_vox'],d['occupied_vox']),'moving_bev_over_dense':ratio(d['moving_bev'],d['dense_bev']),'moving_latent_over_dense':ratio(d['moving_latent'],d['dense_latent'])})
    return out
def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--prepared',required=True);p.add_argument('--radii',default='0,1,2,3,4,5,6');p.add_argument('--latent-extra-radius',type=int,default=None);p.add_argument('--schedule',default=None);p.add_argument('--max-windows',type=int,default=None);p.add_argument('--window-size',type=int,default=None);p.add_argument('--window-slots',type=int,default=None);p.add_argument('--output',required=True);a=p.parse_args();cfg=load_runtime_config(a.config,a.override);radii=[int(x) for x in a.radii.split(',')];schedule=tuple(int(x) for x in (a.schedule.split(',') if a.schedule else get_cfg(cfg,'MOTION.KTA_TUBE_RADII')));extra=int(a.latent_extra_radius if a.latent_extra_radius is not None else get_cfg(cfg,'MOTION.LATENT_EXTRA_RADIUS',1));window=int(a.window_size or get_cfg(cfg,'MODEL.WINDOW_HW',[20,20])[0]);slots=int(a.window_slots or get_cfg(cfg,'MODEL.MAX_WINDOWS',8));ds=PreparedShardDataset(a.prepared);n=len(ds) if a.max_windows is None else min(len(ds),a.max_windows);res=summarize((ds[i] for i in range(n)),radii,extra,schedule,window,slots);res['num_windows']=n;op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(res,indent=2),encoding='utf-8');save_resolved_config(cfg,op.with_suffix('.resolved.yaml'));print('saved',a.output)
if __name__=='__main__':main()
