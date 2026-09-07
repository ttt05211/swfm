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
from real_motion.motion_transport_v1.losses import occupancy_ce_full,motion_loss_sum
from real_motion.motion_transport_v1.routing import msp_checkpoint_provenance,sha256_file
FORBIDDEN=('future_semantics','future_gt','gt_box','annotation_velocity','future_instance','instance_id')
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=None);p.add_argument('--override',action='append',default=[]);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=8);p.add_argument('--device',default='cuda');a=p.parse_args();cfg=load_config(a.config,a.override);dr,info,manifest,msp=[get(cfg,x) for x in ('paths.dataroot','paths.info_pkl','paths.manifest','paths.msp_checkpoint')]
    if not all((dr,info,manifest,msp)):raise RuntimeError('runtime paths incomplete')
    mp=load_manifest(manifest);tr,dv=set(mp['train_scenes']),set(mp['dev_scenes']);prov=msp_checkpoint_provenance(msp);leak=sorted(set(prov['train_scene_names'])&dv)
    if tr&dv:raise RuntimeError('manifest scene leakage')
    if leak:raise RuntimeError(f'MSP train/dev leakage {leak[:8]}')
    if not prov['train_scene_names']:raise RuntimeError('MSP checkpoint has no auditable train scene list')
    device=a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu';src=NuScenesWindowSource(dr,info_pkl=info,verbose=False);ds=ManifestDataset(src,mp,'train');pipe=MotionTransportV1(cfg,msp_checkpoint=msp,device=device);names={f.name for f in fields(CausalInputs)};forbidden=[n for n in names if any(x in n.lower() for x in FORBIDDEN)];res={'spec_version':'MT-V1-SPEC-2','manifest_sha256':sha256_file(manifest),'msp':prov,'predict_signature':str(inspect.signature(pipe.predict)),'causal_fields':sorted(names),'forbidden_causal_fields':forbidden,'hard_identity_all':True,'zero_initialized_routed_identity_all':True,'soft_grad_nonzero':False,'sparse_call_count_exact':True,'samples':[]}
    for i in range(min(a.samples,len(ds))):
        w,c=ds[i];d,_,_=pipe.prepare_scene(c);ref=strong_w2det_sequence(c.history_semantics,c.history_ego_to_world,c.future_ego_to_world,frame_dt_s=float(get(cfg,'input.dt_seconds',.5)),grid=pipe.grid,cfg=pipe.strong_cfg);z=pipe.predict(c,budget=0);q=pipe.predict(c,budget=int(get(cfg,'routing.main_eval_budget_sources',16)));dz=np.count_nonzero(z.hard_occupancy!=ref);dq=np.count_nonzero(q.hard_occupancy!=ref);res['hard_identity_all']&=dz==0;res['zero_initialized_routed_identity_all']&=dq==0;res['sparse_call_count_exact']&=q.diagnostics['stpn_sources_executed']==len(q.selected_source_ids);res['samples'].append({'sample_id':c.sample_id,'sources':len(d.sources),'zero_diff_voxels':int(dz),'routed_zero_init_diff_voxels':int(dq),'selected':len(q.selected_source_ids),'stpn_executed':q.diagnostics['stpn_sources_executed']})
    for i in range(min(a.samples,len(ds))):
        w,c=ds[i];d,_,_=pipe.prepare_scene(c);sel=tuple(s.source_id for s in d.sources if s.crop_eligible)[:2]
        if not sel:continue
        t=build_training_targets(src,w,d,best_coverage_min=float(get(cfg,'targets.best_box_source_coverage_min',.8)),second_coverage_max=float(get(cfg,'targets.second_box_source_coverage_max',.2)),max_points=int(get(cfg,'targets.motion_points_per_source_max',64)));pipe.source_network.train();pipe.source_network.zero_grad(set_to_none=True);delta,zero,_,soft,_=pipe.forward_selected(c,d,sel,mirror=False,soft=True,hard=False);occ,_=occupancy_ce_full(soft,t,eps=float(get(cfg,'renderer.probability_epsilon',1e-6)));mn,n,_=motion_loss_sum(delta,sel,d,t,np.asarray(c.history_ego_to_world[-1]));(occ+mn/max(1,n)+zero).backward();g=pipe.source_network.head2.weight.grad;res['soft_grad_nonzero']=bool(g is not None and torch.isfinite(g).all() and float(g.abs().sum())>0);break
    ok=not forbidden and res['hard_identity_all'] and res['zero_initialized_routed_identity_all'] and res['soft_grad_nonzero'] and res['sparse_call_count_exact'];res['pass']=bool(ok);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
    if not ok:raise RuntimeError('MT-V1 preflight failed')
if __name__=='__main__':main()
