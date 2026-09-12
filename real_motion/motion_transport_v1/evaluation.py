from __future__ import annotations
from collections import Counter
import math,numpy as np,torch
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS,REPORT_HORIZONS_S,SemanticIoUAccumulator,MovingMIoUV2MultiHorizon,MovingMIoUV2Accumulator,GridSpec,rasterize_oriented_box,Box3D,SPEED_THRESHOLD_MPS,BOX_MARGIN_M
from real_motion.nuscenes_adapter import gt_moving_support_for_horizon,category_to_dynamic_class,box3d_from_dict
from real_motion.geometry import quaternion_yaw
from .routing import route_sources
from .targets import build_training_targets,gt_moving_source_ids
from .compositor import hard_kta_identity,soft_argmax_scene
REPORT={1.0:1,2.0:3,3.0:5};GROUP_NAMES=('observed-moving','dormant-to-moving','stationary','stopping','ambiguous-causal-source','no-causal-source')
def _ann_map(nusc,token):
    out={}
    for at in nusc.get('sample',token)['anns']:
        a=nusc.get('sample_annotation',at);out[str(a['instance_token'])]=a
    return out
def _box_future(a,T,cid):
    c=np.asarray(a['translation'],float);ce=(np.linalg.inv(T)@np.r_[c,1])[:3];yaw=quaternion_yaw(a['rotation'])-math.atan2(T[1,0],T[0,0]);w,l,h=a['size'];return Box3D(str(a['instance_token']),int(cid),tuple(ce),(float(l),float(w),float(h)),float(yaw))
def _union(a0,ah,T,cid,grid):
    gs=GridSpec(grid.x_min,grid.y_min,grid.z_min,grid.voxel_size,grid.shape_hwd);return rasterize_oriented_box(_box_future(a0,T,cid),gs,BOX_MARGIN_M)|rasterize_oriented_box(_box_future(ah,T,cid),gs,BOX_MARGIN_M)
def _inside_world(points,ann):
    p=np.asarray(points,float);c=np.asarray(ann['translation'],float);yaw=quaternion_yaw(ann['rotation']);cy,sy=math.cos(yaw),math.sin(yaw);d=p-c;lx=cy*d[:,0]+sy*d[:,1];ly=-sy*d[:,0]+cy*d[:,1];w,l,h=map(float,ann['size']);return (np.abs(lx)<=l/2)&(np.abs(ly)<=w/2)&(np.abs(d[:,2])<=h/2)
def stationary_movable_support(nusc,t0,th,dt,T,grid):
    a0,ah=_ann_map(nusc,t0),_ann_map(nusc,th);sup=np.zeros(grid.shape_hwd,bool)
    for tok in sorted(set(a0)&set(ah)):
        c0=category_to_dynamic_class(a0[tok]['category_name']);ch=category_to_dynamic_class(ah[tok]['category_name'])
        if c0 is None or ch!=c0:continue
        speed=np.linalg.norm(np.asarray(ah[tok]['translation'])[:2]-np.asarray(a0[tok]['translation'])[:2])/dt
        if speed<SPEED_THRESHOLD_MPS:sup|=_union(a0[tok],ah[tok],T,c0,grid)
    return sup
def _groups(source,window,causal,decomp,targets,fi,h,moving_records,grid):
    out={n:np.zeros(grid.shape_hwd,bool) for n in GROUP_NAMES};a0=_ann_map(source.nusc,window.t0_token);ah=_ann_map(source.nusc,window.future_tokens[fi]);T=np.asarray(causal.future_ego_to_world[fi]);by={int(s.source_id):s for s in decomp.sources};matched=set()
    for sid,t in targets.motion_targets.items():
        if t.instance_token is None or not bool(t.valid[fi]):continue
        tok=str(t.instance_token);s=by.get(int(sid));x0=a0.get(tok);xh=ah.get(tok)
        if s is None or x0 is None or xh is None:continue
        cid=category_to_dynamic_class(xh['category_name'])
        if cid is None or cid!=s.class_id:continue
        matched.add(tok);future_speed=float(t.gt_speed_mps[fi]);hist_speed=float(np.linalg.norm(s.velocity_world[:2]))
        if np.isfinite(future_speed) and future_speed<SPEED_THRESHOLD_MPS:name='stopping' if hist_speed>=SPEED_THRESHOLD_MPS else 'stationary'
        elif s.dormant_fraction>s.observed_moving_fraction and s.dormant_fraction>0:name='dormant-to-moving'
        elif hist_speed<SPEED_THRESHOLD_MPS and s.observed_moving_fraction<=0:name='dormant-to-moving'
        else:name='observed-moving'
        out[name]|=_union(x0,xh,T,cid,grid)
    gs=GridSpec(grid.x_min,grid.y_min,grid.z_min,grid.voxel_size,grid.shape_hwd)
    for r in moving_records:
        tok=str(r['instance_token'])
        if tok in matched:continue
        ann0=a0.get(tok);cid=int(r['class_id']);has_causal=False
        if ann0 is not None:
            for s in decomp.sources:
                if int(s.class_id)==cid and len(s.points_world) and bool(_inside_world(s.points_world,ann0).any()):has_causal=True;break
        name='ambiguous-causal-source' if has_causal else 'no-causal-source';out[name]|=rasterize_oriented_box(box3d_from_dict(r['box0_future_ego']),gs,BOX_MARGIN_M)|rasterize_oriented_box(box3d_from_dict(r['boxh_future_ego']),gs,BOX_MARGIN_M)
    return out
def _state():return {'overall':{h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},'moving':MovingMIoUV2MultiHorizon(),'stationary':{h:MovingMIoUV2Accumulator() for h in REPORT_HORIZONS_S},'groups':{n:{h:MovingMIoUV2Accumulator() for h in REPORT_HORIZONS_S} for n in GROUP_NAMES},'tp':0,'fp':0,'fn':0}
def _update(st,h,p,g,mov,sta,groups):
    st['overall'][h].update(p,g);st['moving'].update(h,p,g,mov);st['stationary'][h].update(p,g,sta)
    for n in GROUP_NAMES:st['groups'][n][h].update(p,g,groups[n])
    dyn=np.asarray(DYNAMIC_CLASS_IDS);pm=np.isin(p,dyn);gm=np.isin(g,dyn);st['tp']+=int((pm&gm).sum());st['fp']+=int((pm&~gm).sum());st['fn']+=int((~pm&gm).sum())
def _mean(rows):
    v=[r['mIoU'] for r in rows.values() if not np.isnan(r['mIoU'])];return float(np.mean(v)) if v else float('nan')
def _report(st):
    oh={h:st['overall'][h].compute() for h in REPORT_HORIZONS_S};sh={h:st['stationary'][h].compute() for h in REPORT_HORIZONS_S};gh={n:{h:st['groups'][n][h].compute() for h in REPORT_HORIZONS_S} for n in GROUP_NAMES};tp,fp,fn=st['tp'],st['fp'],st['fn'];return {'overall':{'mIoU':_mean(oh),'per_horizon':oh},'moving':st['moving'].compute(),'stationary_movable':{'mIoU':_mean(sh),'per_horizon':sh},'motion_groups':{n:{'mIoU':_mean(gh[n]),'per_horizon':gh[n]} for n in GROUP_NAMES},'dynamic_precision':tp/max(1,tp+fp),'dynamic_recall':tp/max(1,tp+fn),'dynamic_tp':tp,'dynamic_fp':fp,'dynamic_fn':fn}
def _snap(st):return {'overall':{str(h):{'inter':st['overall'][h].inter,'union':st['overall'][h].union} for h in REPORT_HORIZONS_S},'moving':{str(h):{'inter':st['moving'].acc[h].inter,'union':st['moving'].acc[h].union} for h in REPORT_HORIZONS_S},'stationary':{str(h):{'inter':st['stationary'][h].inter,'union':st['stationary'][h].union} for h in REPORT_HORIZONS_S}}
def _validity_audit_update(audit,targets,moving_support=None,fi=None):
    valid=np.asarray(targets.future_valid,bool);sem=np.asarray(targets.future_semantics);obs=np.asarray(targets.future_observed,bool) if targets.future_observed is not None else np.zeros_like(valid,bool)
    if moving_support is None:
        audit['valid_voxels']+=int(valid.sum());audit['invalid_voxels']+=int((~valid).sum());audit['valid_observed']+=int((valid&obs).sum());audit['valid_unobserved']+=int((valid&~obs).sum());audit['valid_free_observed']+=int((valid&obs&(sem==17)).sum());audit['valid_free_unobserved']+=int((valid&~obs&(sem==17)).sum());audit['valid_nonfree_observed']+=int((valid&obs&(sem!=17)).sum());audit['valid_nonfree_unobserved']+=int((valid&~obs&(sem!=17)).sum());return
    if fi is None:raise ValueError('moving-support audit requires horizon index')
    v=valid[fi];o=obs[fi];m=np.asarray(moving_support,bool);audit['moving_support_valid']+=int((m&v).sum());audit['moving_support_observed']+=int((m&v&o).sum());audit['moving_support_unobserved']+=int((m&v&~o).sum())
@torch.no_grad()
def evaluate(pipe,source,dataset,cfg,*,budgets=(0,16,'all'),strategy='msp',include_soft_main=True,seed=3407,max_windows=None):
    states={str(q):_state() for q in budgets};soft_state=_state() if include_soft_main else None;scene_states={};calls={str(q):0 for q in budgets};source_counts=[];class_counts=Counter();fallback=Counter();sel_cls=Counter();covers=[];gtv=gtt=0;main=int(cfg.get('evaluation',{}).get('main_budget_sources',16));limit=min(len(dataset),max_windows if max_windows is not None else len(dataset));pipe.source_network.eval();soft_windows=0;validity={k:0 for k in ('valid_voxels','invalid_voxels','valid_observed','valid_unobserved','valid_free_observed','valid_free_unobserved','valid_nonfree_observed','valid_nonfree_unobserved','moving_support_valid','moving_support_observed','moving_support_unobserved')}
    for di in range(limit):
        w,c=dataset[di];d,cands,scores=pipe.prepare_scene(c);targets=build_training_targets(source,w,d,best_coverage_min=float(cfg['targets']['best_box_source_coverage_min']),second_coverage_max=float(cfg['targets']['second_box_source_coverage_max']),max_points=int(cfg['targets']['motion_points_per_source_max']),class_count=int(cfg['input']['class_count']));gtids=gt_moving_source_ids(targets) if strategy in ('gt_moving','gt_moving_diagnostic') else None;preds={};softpred=None;by={s.source_id:s for s in d.sources};_validity_audit_update(validity,targets)
        for q in budgets:
            sel=route_sources(d.sources,q,strategy=strategy,scores=scores,seed=seed+di,gt_moving_source_ids=gtids)
            if not sel:
                pred=hard_kta_identity(c,d,grid=pipe.grid)
                if include_soft_main and q==main:softpred=pred.copy()
            else:
                _,_,pred,soft,_=pipe.forward_selected(c,d,sel,mirror=False,soft=(include_soft_main and q==main),hard=True)
                if include_soft_main and q==main:softpred=soft_argmax_scene(soft)
            preds[str(q)]=pred;calls[str(q)]+=len(sel)
            for sid in sel:sel_cls[by[int(sid)].class_id]+=1
        if include_soft_main and softpred is not None:soft_windows+=1
        for s in d.sources:
            class_counts[s.class_id]+=1;fallback[s.fallback_reason or 'eligible']+=1;covers.append(s.mapping_coverage);gtt+=1;t=targets.motion_targets.get(s.source_id);gtv+=int(t is not None and np.asarray(t.valid,bool).any())
        source_counts.append({'sample_id':c.sample_id,'scene_name':c.scene_name,'eligible':sum(s.crop_eligible for s in d.sources),'total':len(d.sources),'msp_candidates':len(cands)})
        for h,fi in REPORT.items():
            gt=targets.future_semantics[fi];mov,records,_=gt_moving_support_for_horizon(source.nusc,w.t0_token,w.future_tokens[fi],h,grid=pipe.grid);_validity_audit_update(validity,targets,mov,fi);sta=stationary_movable_support(source.nusc,w.t0_token,w.future_tokens[fi],h,np.asarray(c.future_ego_to_world[fi]),pipe.grid);groups=_groups(source,w,c,d,targets,fi,h,records,pipe.grid)
            for q in budgets:
                key=str(q);_update(states[key],h,preds[key][fi],gt,mov,sta,groups);row=scene_states.setdefault(c.scene_name,{}).setdefault('hard:'+key,_state());_update(row,h,preds[key][fi],gt,mov,sta,groups)
            if include_soft_main and softpred is not None:_update(soft_state,h,softpred[fi],gt,mov,sta,groups)
    reports={q:_report(st) for q,st in states.items()};out={'protocol':{'spec_version':'MT-V1-SPEC-2','moving_metric':'interval_displacement_v2','strategy':strategy,'num_windows':limit,'num_scenes':len(scene_states),'hard_deployment':True,'hard_windows':limit,'soft_main_windows':soft_windows if include_soft_main else 0},'hard':reports,'soft_q16_argmax':_report(soft_state) if soft_state else None,'source_calls':calls,'source_counts':source_counts,'source_audit':{'per_class':{str(k):int(v) for k,v in sorted(class_counts.items())},'selected_calls_per_class':{str(k):int(v) for k,v in sorted(sel_cls.items())},'crop_fallback':dict(fallback),'mean_msp_mapping_coverage':float(np.mean(covers)) if covers else 0.,'gt_motion_label_source_coverage':gtv/max(1,gtt)},'supervision_audit':validity,'per_scene_confusion':{sc:{k:_snap(v) for k,v in rows.items()} for sc,rows in scene_states.items()}}
    if '0' in reports:
        b=reports['0'];out['delta_vs_kta']={q:{'overall_pp':r['overall']['mIoU']-b['overall']['mIoU'],'moving_pp':r['moving']['mIoU']-b['moving']['mIoU'],'stationary_movable_pp':r['stationary_movable']['mIoU']-b['stationary_movable']['mIoU']} for q,r in reports.items()}
    return out
