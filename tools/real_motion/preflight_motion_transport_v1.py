#!/usr/bin/env python3
from __future__ import annotations
import argparse,inspect,json,sys
from dataclasses import fields
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.strong_w2det import strong_w2det_sequence
from real_motion.motion_transport_v1.config import load_config,get
from real_motion.motion_transport_v1.contracts import CausalInputs
from real_motion.motion_transport_v1.data import ManifestDataset,load_manifest
from real_motion.motion_transport_v1.model import MotionTransportV1
from real_motion.motion_transport_v1.targets import build_training_targets
from real_motion.motion_transport_v1.losses import occupancy_ce_full,motion_loss_sum,calibration_probe_like,calibrated_gradient_ratio,output_gradient_lambda_floor,gradient_summary
from real_motion.motion_transport_v1.compositor import render_soft_ordered,hard_kta_identity
from real_motion.motion_transport_v1.crop_history import build_history_crops
from real_motion.motion_transport_v1.routing import msp_checkpoint_provenance,sha256_file
FORBIDDEN=('future_semantics','future_gt','gt_box','annotation_velocity','future_instance','instance_id')
def _flat_grads(loss,params,retain=True):
    gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True);return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1) for p,g in zip(params,gs)])
def _cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
def _reference(c,pipe,cfg):return strong_w2det_sequence(c.history_semantics,c.history_ego_to_world,c.future_ego_to_world,frame_dt_s=float(get(cfg,'input.dt_seconds',.5)),grid=pipe.grid,cfg=pipe.strong_cfg)
def _identity_scan(ds,pipe,cfg,max_windows=None):
    n=len(ds) if max_windows is None else min(int(max_windows),len(ds));diff_windows=0;diff_voxels=0;failures=[]
    for i in range(n):
        _,c=ds[i];d,_,_=pipe.prepare_scene(c);ref=_reference(c,pipe,cfg);hard=hard_kta_identity(c,d,grid=pipe.grid);dv=int(np.count_nonzero(hard!=ref));diff_voxels+=dv
        if dv:
            diff_windows+=1
            if len(failures)<20:failures.append({'index':i,'sample_id':c.sample_id,'diff_voxels':dv})
    return {'windows_scanned':n,'diff_windows':diff_windows,'diff_voxels':diff_voxels,'failures_first20':failures,'pass':diff_windows==0}
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=None);p.add_argument('--override',action='append',default=[]);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=8);p.add_argument('--device',default='cuda');p.add_argument('--require-cuda',action='store_true');p.add_argument('--identity-scan',choices=('sampled','dev','all'),default='sampled');a=p.parse_args();cfg=load_config(a.config,a.override);dr,info,manifest,msp=[get(cfg,x) for x in ('paths.dataroot','paths.info_pkl','paths.manifest','paths.msp_checkpoint')]
    if not all((dr,info,manifest,msp)):raise RuntimeError('runtime paths incomplete')
    if a.require_cuda and not torch.cuda.is_available():raise RuntimeError('--require-cuda set but CUDA is unavailable')
    if a.require_cuda and a.device!='cuda':raise RuntimeError('--require-cuda requires --device cuda')
    mp=load_manifest(manifest);tr,dv=set(mp['train_scenes']),set(mp['dev_scenes']);prov=msp_checkpoint_provenance(msp);leak=sorted(set(prov['train_scene_names'])&dv)
    if tr&dv:raise RuntimeError('manifest scene leakage')
    if leak:raise RuntimeError(f'MSP train/dev leakage {leak[:8]}')
    if not prov['train_scene_names']:raise RuntimeError('MSP checkpoint has no auditable train scene list')
    device=a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu';src=NuScenesWindowSource(dr,info_pkl=info,verbose=False);train_ds=ManifestDataset(src,mp,'train');dev_ds=ManifestDataset(src,mp,'dev');pipe=MotionTransportV1(cfg,msp_checkpoint=msp,device=device);names={f.name for f in fields(CausalInputs)};forbidden=[n for n in names if any(x in n.lower() for x in FORBIDDEN)];res={'spec_version':'MT-V1-SPEC-2','manifest_sha256':sha256_file(manifest),'msp':prov,'environment':{'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':bool(torch.cuda.is_available()),'device':str(pipe.device),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'bf16_supported':bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),'require_cuda':bool(a.require_cuda)},'predict_signature':str(inspect.signature(pipe.predict)),'causal_fields':sorted(names),'forbidden_causal_fields':forbidden,'hard_identity_all':True,'zero_initialized_routed_identity_all':True,'actual_sparse_forward_all':True,'gradient_probe':None,'supervision_audit':{'valid':0,'invalid':0,'observed_valid':0,'unobserved_valid':0,'nonfree_observed_valid':0,'nonfree_unobserved_valid':0},'samples':[]}
    if a.identity_scan=='sampled':res['identity_scan']={'train':_identity_scan(train_ds,pipe,cfg,max_windows=max(1,a.samples))}
    elif a.identity_scan=='dev':res['identity_scan']={'dev':_identity_scan(dev_ds,pipe,cfg,max_windows=None)}
    else:res['identity_scan']={'train':_identity_scan(train_ds,pipe,cfg,max_windows=None),'dev':_identity_scan(dev_ds,pipe,cfg,max_windows=None)}
    res['identity_scan_all']=all(x['pass'] for x in res['identity_scan'].values())
    forward_calls=[]
    hook=pipe.source_network.register_forward_pre_hook(lambda module,args: forward_calls.append(int(args[0].num_sources)))
    try:
        for i in range(min(a.samples,len(train_ds))):
            w,c=train_ds[i];d,_,_=pipe.prepare_scene(c);ref=_reference(c,pipe,cfg);before=len(forward_calls);z=pipe.predict(c,budget=0);q0_calls=forward_calls[before:];before=len(forward_calls);q=pipe.predict(c,budget=int(get(cfg,'routing.main_eval_budget_sources',16)));q_calls=forward_calls[before:];dz=np.count_nonzero(z.hard_occupancy!=ref);dq=np.count_nonzero(q.hard_occupancy!=ref);expected=len(q.selected_source_ids);call_ok=(len(q0_calls)==0 and ((expected==0 and len(q_calls)==0) or q_calls==[expected]));res['hard_identity_all']&=dz==0;res['zero_initialized_routed_identity_all']&=dq==0;res['actual_sparse_forward_all']&=call_ok;res['samples'].append({'sample_id':c.sample_id,'sources':len(d.sources),'zero_diff_voxels':int(dz),'routed_zero_init_diff_voxels':int(dq),'selected':expected,'q0_actual_forward_calls':q0_calls,'q_actual_forward_calls':q_calls,'actual_sparse_forward_ok':call_ok})
    finally:hook.remove()
    for i in range(min(max(a.samples,16),len(train_ds))):
        w,c=train_ds[i];d,_,_=pipe.prepare_scene(c);sel=tuple(s.source_id for s in d.sources if s.crop_eligible)[:2]
        if not sel:continue
        targets=build_training_targets(src,w,d,best_coverage_min=float(get(cfg,'targets.best_box_source_coverage_min',.8)),second_coverage_max=float(get(cfg,'targets.second_box_source_coverage_max',.2)),max_points=int(get(cfg,'targets.motion_points_per_source_max',64)),class_count=int(get(cfg,'input.class_count',18)));valid=np.asarray(targets.future_valid,bool);obs=np.asarray(targets.future_observed,bool);sem=np.asarray(targets.future_semantics);au=res['supervision_audit'];au['valid']+=int(valid.sum());au['invalid']+=int((~valid).sum());au['observed_valid']+=int((valid&obs).sum());au['unobserved_valid']+=int((valid&~obs).sum());au['nonfree_observed_valid']+=int((valid&obs&(sem!=17)).sum());au['nonfree_unobserved_valid']+=int((valid&~obs&(sem!=17)).sum())
        if not any(int(np.asarray(targets.motion_targets.get(int(sid)).valid,bool).sum()) for sid in sel if int(sid) in targets.motion_targets):continue
        pipe.source_network.train();crops=build_history_crops(c,d.sources,sel,grid=pipe.grid,crop_shape_xyz=tuple(get(cfg,'crop.shape_xyz',[64,64,16])),xy_resolution_m=float(get(cfg,'crop.xy_resolution_m',.4)),mirror=False,device=pipe.device);delta,zero=pipe.source_network(crops,source_microbatch=int(get(cfg,'training.source_microbatch',16)));delta=delta.float();probe=calibration_probe_like(delta,float(get(cfg,'loss.gradient_calibration.probe_epsilon',1e-3)));soft=render_soft_ordered(c,d,probe,sel,grid=pipe.grid,class_count=int(get(cfg,'input.class_count',18)),halo_voxels=int(get(cfg,'renderer.query_bbox_halo_voxels',2)),query_chunk=int(get(cfg,'renderer.query_chunk',65536)));occ,_=occupancy_ce_full(soft,targets,eps=float(get(cfg,'renderer.probability_epsilon',1e-6)));mn,n,_=motion_loss_sum(probe,sel,d,targets,np.asarray(c.history_ego_to_world[-1]));mot=mn/max(1,n);params=list(pipe.source_network.head2.parameters());go_out=torch.autograd.grad(occ+zero,delta,retain_graph=True,allow_unused=True)[0];gm_out=torch.autograd.grad(mot+zero,delta,retain_graph=True,allow_unused=True)[0];go_head=_flat_grads(occ+zero,params,True);gm_head=_flat_grads(mot+zero,params,True);Go=float(go_head.norm());Gm=float(gm_head.norm());cos=_cos(go_head,gm_head);frac=float(get(cfg,'loss.gradient_calibration.max_ce_antagonistic_fraction_of_motion',.5));head_ratio=calibrated_gradient_ratio(Go,Gm,cos,max_ce_antagonistic_fraction_of_motion=frac);floor=output_gradient_lambda_floor(go_out,gm_out,max_ce_antagonistic_fraction_of_motion=frac,active_motion_grad_rel=float(get(cfg,'loss.gradient_calibration.active_motion_grad_rel',1e-4)));lam=max(head_ratio if np.isfinite(head_ratio) else 0.,floor);joint=go_out+lam*gm_out;motion_alignment=float((joint*gm_out).sum());res['gradient_probe']={'sample_id':c.sample_id,'selected_sources':list(map(int,sel)),'motion_pairs':int(n),'probe_epsilon':float(get(cfg,'loss.gradient_calibration.probe_epsilon',1e-3)),'ce_output_abs':gradient_summary(go_out),'motion_output_abs':gradient_summary(gm_out),'ce_head_abs':gradient_summary(go_head),'motion_head_abs':gradient_summary(gm_head),'head_cosine':cos,'head_ratio':head_ratio,'output_conflict_floor':floor,'lambda_safe':lam,'joint_dot_motion_gradient':motion_alignment,'ce_output_nonzero':bool(float(go_out.abs().sum())>0),'motion_output_nonzero':bool(float(gm_out.abs().sum())>0),'ce_head_nonzero':bool(float(go_head.abs().sum())>0),'motion_head_nonzero':bool(float(gm_head.abs().sum())>0)};break
    gp=res['gradient_probe'];grad_ok=bool(gp and gp['ce_output_nonzero'] and gp['motion_output_nonzero'] and gp['ce_head_nonzero'] and gp['motion_head_nonzero'] and np.isfinite(gp['lambda_safe']) and gp['lambda_safe']>0 and gp['joint_dot_motion_gradient']>0);ok=not forbidden and res['identity_scan_all'] and res['hard_identity_all'] and res['zero_initialized_routed_identity_all'] and res['actual_sparse_forward_all'] and grad_ok;res['pass']=bool(ok);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
    if not ok:raise RuntimeError('MT-V1 preflight failed')
if __name__=='__main__':main()
