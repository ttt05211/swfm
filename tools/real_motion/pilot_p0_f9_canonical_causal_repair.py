#!/usr/bin/env python3
"""One REAL local CCR trial: support oracle, GT-only fit, six-frame FPS,
actual joint backward, geometry exactness and isolated CPU profile together.
No server training, old checkpoint writes, KD, threshold search or promotion.
"""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import copy
import cProfile
import pstats
import json
import time
from collections import defaultdict
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from real_motion.canonical_causal_repair import (CanonicalRepairHead,build_canonical_evidence,
    map_canonical_evidence,map_canonical_reference,materialize_canonical_features,
    repair_targets,compose_canonical,sampled_tasks,repair_loss,PROTOCOL)
from real_motion.local_replay_bundle import ReplayBundle,file_digest
from real_motion.column_execution import execution_session
from real_motion.column_gpu_sampling import GpuColumnSampler
from real_motion.causal_column_completion import actions_from_probabilities,compose_dense,ADD,REMOVE,REFINE
from real_motion.source_evidence_audit import edit_quality
from real_motion.runtime_config import make_prepare_config
from tools.real_motion import causal_column_common as columns
from tools.real_motion.height_field_screen_common import sync as synchronize
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.joint_column_full_common import build_fixed_geometry,train_full_batch,pack_records,MOTION_KEYS,LABEL_KEYS
from tools.real_motion.joint_column_common import motion_loss,set_lr
from tools.real_motion.shared_evidence_pilot_common import fresh_prior,GATES
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics,delta
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,finite_json


def tensor(value,device):return torch.as_tensor(np.ascontiguousarray(value),device=device)


def load_exported_config(path, expected_fingerprint):
    """Read the already-resolved replay artifact without rewriting provenance."""
    import yaml
    from real_motion.runtime_config import validate_runtime_config
    from real_motion.v21_source_induction import stable_json_fingerprint
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or stable_json_fingerprint(cfg) != expected_fingerprint:
        raise RuntimeError('config content mismatch')
    validate_runtime_config(cfg)
    return cfg


def probabilities(head,evidence,plan,output,device,*,chunk=8192):
    """One upload/encoding per canonical candidate, six readouts per chunk.
    Bounded activation/readback memory; no learned feature reused across calls.
    """
    result=[]
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        output=head.project_sources(output)
        shared = (head.encode_queries(evidence,evidence.neighbor_graph,np.arange(len(evidence)),output,device)
                  if hasattr(head,'encode_queries') and len(evidence) else None)
        for start in range(0,len(evidence),chunk):
            sl=slice(start,start+chunk);actor=tensor(evidence.actor[sl],device)
            encoded=(shared[sl] if shared is not None else
                     head.encode(tensor(evidence.features[sl],device),tensor(evidence.labels[sl],device),
                                 actor,tensor(evidence.classes[sl],device),output))
            logits=head.decode(encoded,actor,tensor(plan.context[sl],device),tensor(plan.base[sl],device),
                               tensor(plan.fallback[sl],device),tensor(plan.legal[sl],device),output)
            result.append(head.probabilities(logits,actor).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0,6,2),np.float32)


def loss_for(head,evidence,plan,output,target,valid,rng,device,*,per_group=768,remove_weight=.25,prepared=None,grid=None):
    ids,w=sampled_tasks(evidence,target,valid,rng,per_group=per_group)
    sampled=materialize_canonical_features(evidence,prepared,grid,ids)
    actor=tensor(evidence.actor[ids],device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        output=head.project_sources(output)
        encoded=(head.encode_queries(evidence,evidence.neighbor_graph,ids,output,device) if hasattr(head,'encode_queries') else
                 head.encode(tensor(sampled.features,device),tensor(sampled.labels,device),actor,
                             tensor(evidence.classes[ids],device),output))
        logits=head.decode(encoded,actor,tensor(plan.context[ids],device),tensor(plan.base[ids],device),
                           tensor(plan.fallback[ids],device),tensor(plan.legal[ids],device),output)
        loss=repair_loss(head,logits,actor,tensor(target[ids],device),tensor(w,device),remove_weight=remove_weight)
    return loss,len(ids)


def loss_for_causal(head,evidence,output,prepared,grid,gt,rng,device,conflicts,*,per_role=1024,remove_weight=.25):
    from real_motion.canonical_repair_context import sample_causal_points,map_sampled_canonical
    ids,importance=sample_causal_points(evidence,rng,per_role=per_role)
    sampled,plan=map_sampled_canonical(evidence,ids,prepared,grid,conflicts)
    target,valid=repair_targets(sampled,plan,gt)
    weight=importance[:,None,None]*valid
    actor=tensor(evidence.actor[ids],device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        output=head.project_sources(output)
        if hasattr(head,'encode_queries'):
            encoded=head.encode_queries(evidence,evidence.neighbor_graph,ids,output,device)
        else:
            encoded=head.encode(tensor(sampled.features,device),tensor(sampled.labels,device),actor,
                                tensor(sampled.classes,device),output)
        logits=head.decode(encoded,actor,tensor(plan.context,device),tensor(plan.base,device),
                           tensor(plan.fallback,device),tensor(plan.legal,device),output)
        loss=repair_loss(head,logits,actor,tensor(target,device),tensor(weight,device),remove_weight=remove_weight)
    return loss,len(ids)


def coverage(case):
    """Correct old edits representable by causal CCR candidates, same semantics.
    No old probabilities/GT choose inference candidates or head features.
    """
    e,p=case['evidence'],case['plan'];report=defaultdict(int)
    for h,old,action in case['old_rows']:
        gt=case['gt'][h].ravel()
        for role in ('static','dynamic','generation'):
            use=(old.actor==-2) if role=='static' else ((old.actor>=0) if role=='dynamic' else (old.actor==-3))
            for mode,label in (('ADD',ADD),('REMOVE',REMOVE)):
                edits=(action==label)&use[:,None]
                flats=old.flat[edits]
                sem=(np.broadcast_to(old.classes[:,None],old.base.shape)[edits] if mode=='ADD' else old.fallback[edits])
                correct=gt[flats]==sem
                pairs=np.unique(flats[correct]*18+sem[correct])
                legal=p.legal[:,h,0 if mode=='ADD' else 1].copy()
                if role=='static':legal &= e.actor== -2
                elif role=='dynamic':legal &= e.actor>=0
                cls=e.classes if mode=='ADD' else p.fallback[:,h]
                available=np.unique(p.flat[legal,h]*18+cls[legal])
                report[role+'_'+mode+'_correct_old_voxels']+=len(pairs)
                report[role+'_'+mode+'_covered_voxels']+=int(np.isin(pairs,available).sum())
    return dict(report)


def score(cases,head,device,*,variant='learned'):
    base=Metrics();metrics=Metrics();quality=defaultdict(int);scene=defaultdict(lambda:[Metrics(),Metrics()])
    with torch.no_grad():
      for case in cases:
        if variant=='old':dense=case['old_dense']
        elif variant=='oracle':
            y=case['target'].astype(np.float32);dense=compose_canonical(case['prep'].baseline,case['evidence'],case['plan'],y[...,0],y[...,1])
        else:
            p=probabilities(head,case['evidence'],case['plan'],case['output'],device)
            dense=compose_canonical(case['prep'].baseline,case['evidence'],case['plan'],p[...,0],p[...,1],role=variant if variant in ('static','dynamic') else 'all')
        for ri,h in enumerate(columns.REPORT):
            gt=case['gt'][h];moving=case['moving'][h];before=case['prep'].baseline[h]
            base.update(ri,before,gt,moving);metrics.update(ri,dense[h],gt,moving)
            a,b=scene[case['record']['scene_name']];a.update(ri,before,gt,moving);b.update(ri,dense[h],gt,moving)
            for k,v in edit_quality(before,dense[h],gt).items():quality[k]+=v
    bm=base.compute();mm=metrics.compute()
    return dict(windows=len(cases),baseline=bm,joint=mm,delta_vs_transport_pp=delta(mm,bm),quality=dict(quality),
        scene_delta={name:b.compute()['mIoU']-a.compute()['mIoU'] for name,(a,b) in scene.items()})


def fit(head,cases,device,passes,progress,remove_weight,*,causal_sampling=False):
    counts=np.zeros((2,2,2),np.int64)
    for case in cases:
        for role in range(2):
            for action in range(2):
                mask=case['valid'][...,action]&((case['evidence'].actor>=0)==bool(role))[:,None]
                positive=case['target'][...,action][mask].sum()
                counts[role,action]=[int(mask.sum()-positive),int(positive)]
    # TRAIN-only balancing; probability correction reverses this objective's
    # prior tilt. Unlike an uncalibrated pos_weight, fixed .5 still means .5.
    weight=np.sqrt(counts[...,0]/np.maximum(counts[...,1],1)).clip(1,32)
    head.positive_weight.copy_(tensor(weight.astype(np.float32),device))
    torch.manual_seed(20261006);rng=np.random.default_rng(20261006)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
    losses=[];started=time.perf_counter()
    for epoch in range(passes):
        lr=.002*(.1+.9*.5*(1+np.cos(np.pi*epoch/max(passes-1,1))))
        for group in optimizer.param_groups:group['lr']=float(lr)
        values=[];head.train()
        for ci in rng.permutation(len(cases)):
            c=cases[ci];optimizer.zero_grad(set_to_none=True)
            if causal_sampling:
                loss,_=loss_for_causal(head,c['evidence'],c['output'],c['prep'],c['grid'],c['gt'],rng,device,
                                      c['static_conflicts'],remove_weight=remove_weight)
            else:
                loss,_=loss_for(head,c['evidence'],c['plan'],c['output'],c['target'],c['valid'],rng,device,remove_weight=remove_weight)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.)
            if not torch.isfinite(norm) or not torch.isfinite(loss):raise RuntimeError('nonfinite CCR gradient')
            optimizer.step();values.append(float(loss.detach()))
        losses.append(float(np.mean(values)));progress(dict(event='fit',epoch=epoch+1,loss=losses[-1],lr=lr))
        if epoch==0 or (epoch+1)%8==0:print(f'CCR_FIT {epoch+1}/{passes} loss={losses[-1]:.5f}',flush=True)
    synchronize(device);head.eval()
    return dict(TRAIN_windows=len(cases),passes=passes,updates=passes*len(cases),loss_by_pass=losses,
        seconds=time.perf_counter()-started,TRAIN_role_action_counts=counts.tolist(),positive_weights=weight.tolist(),
        GT_only=True,KD=False,transport_frozen=True,remove_loss_weight=remove_weight,
        causal_preprojection_sampling=causal_sampling,
        timing_boundary='mini-fit sampled head backward/AdamW; cached CAUSAL evidence/poses, NOT complete online training')


def timed_full(case,teacher,provider,head,*,old=False,profile_stages=False,verify_outputs=False):
    device=provider.device;stages={}
    def call(name,fn):
        if profile_stages:synchronize(device)
        tick=time.perf_counter();value=fn()
        if profile_stages:synchronize(device)
        stages[name]=time.perf_counter()-tick;return value
    with torch.no_grad():
        synchronize(device);tick=time.perf_counter()
        template=SimpleNamespace(state=case['causal']['_column_causal_preparation']['prepared_state'])
        state=call('fresh_prior',lambda:fresh_prior(template,provider).state)
        raw={**case['causal'],'_column_causal_preparation':{**case['causal']['_column_causal_preparation'],'prepared_state':state}}
        output=call('live_motion',lambda:teacher.motion(case['record'],device))
        prep=call('live_render',lambda:provider.prepare_columns(None,case['record'],include_gt=False,raw_window=raw,outputs=output))
        if old:
            def predict():
                dense=[]
                for h in range(6):
                    plan=columns.candidate_plan(prep,h,provider.pcfg.grid,teacher.columns.config)
                    p=columns.predict_probabilities(teacher.columns,prep,h,plan,provider.pcfg.grid,device,256)
                    dense.append(compose_dense(prep.baseline[h],plan,actions_from_probabilities(plan,p,GATES)))
                return dense
            dense=call('old_all_candidates_reader_composition',predict)
        else:
            evidence=call('canonical_evidence_neighbours',lambda:build_canonical_evidence(prep,provider.pcfg.grid))
            plan=call('all_six_geometry_legality',lambda:map_canonical_evidence(evidence,prep,provider.pcfg.grid))
            p=call('once_encode_six_readout',lambda:probabilities(head,evidence,plan,output,device))
            dense=call('six_dense_composition',lambda:compose_canonical(prep.baseline,evidence,plan,p[...,0],p[...,1]))
        synchronize(device);seconds=time.perf_counter()-tick
        if len(dense)!=6 or any(x.shape!=tuple(provider.pcfg.grid.shape_hwd) for x in dense):raise RuntimeError('incomplete six-frame forecast')
        # Force observation of output after timing; no KEEP-only speed claims.
        changed=sum(int((a!=b).sum()) for a,b in zip(dense,prep.baseline))
    result=dict(mode='old_joint' if old else 'CCR',seconds=seconds,stages=stages,changed_voxels=changed,
                full_six_dense=True,generation_branch_replaced_by_causal_support=True)
    if verify_outputs:
        import hashlib
        result['dense_sha256']=[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in dense]
    return result


def joint_probe(provider,teacher,head,cases,device,*,fixed_cache=None,causal_sampling=False):
    """Both paths: full live motion/repair gradients + actual AdamW on clones.
    Different representations/objectives/sampling: NOT numerical equivalence.
    CCR candidate construction is charged on every step, not cached mini-fit.
    """
    old_model,old_joint=provider.model,provider.joint;report={}
    if causal_sampling and fixed_cache is None:raise ValueError('causal sampling requires fixed full-domain inputs')
    groups=[cases[i:i+4] for i in (0,4)]
    try:
      for mode in ('old_joint','CCR'):
        joint=copy.deepcopy(teacher).requires_grad_(True).train();new=copy.deepcopy(head).train()
        provider.joint=joint;provider.model=joint.transport
        params=list(joint.transport.parameters())+list(joint.columns.parameters() if mode=='old_joint' else new.parameters())
        optimizer=torch.optim.AdamW([
            dict(params=joint.transport.parameters(),lr=.0005,initial_lr=.0005,weight_decay=.0001),
            dict(params=joint.columns.parameters() if mode=='old_joint' else new.parameters(),
                 lr=.0003,initial_lr=.0003,weight_decay=.01)])
        rng=np.random.default_rng(20261006)
        sampler=GpuColumnSampler(device) if mode=='old_joint' else None
        seconds=windows=0;stages=defaultdict(float);trials=[]
        torch.cuda.reset_peak_memory_stats()
        with ThreadPoolExecutor(max_workers=2) as pool:
          for index,group in enumerate([groups[0],*groups]):
            synchronize(device);started=time.perf_counter();optimizer.zero_grad(set_to_none=True);local=defaultdict(float)
            set_lr(optimizer,index,3)
            if mode=='old_joint':
                rows=[(c['record'],{**c['causal'],'future_gt_occ':c['gt']}) for c in group]
                train_full_batch(joint,optimizer,provider,None,rows,rng,index+1,3,sampling_pool=pool,sampling_workers=2,
                                 column_feature_sampler=sampler,profile=True)
            else:
                tick=time.perf_counter();merged=pack_records([c['record'] for c in group],(*MOTION_KEYS,*LABEL_KEYS))
                output=joint.motion(merged,device)
                sizes=[len(c['record']['features']) for c in group]
                outputs=[{k:v for k,v in zip(output,values)} for values in zip(*(v.split(sizes) for v in output.values()))]
                local['motion_pack_forward']+=time.perf_counter()-tick;losses=[]
                for c,out in zip(group,outputs):
                    tick=time.perf_counter();prep=provider.prepare_columns(None,c['record'],include_gt=False,raw_window=c['causal'],outputs=out)
                    if fixed_cache is None:
                        evidence=build_canonical_evidence(prep,provider.pcfg.grid,materialize_features=False)
                    else:
                        evidence,graph=fixed_cache.get(prep,provider.pcfg.grid)
                        if hasattr(new,'encode_queries'):evidence.neighbor_graph=graph
                    if causal_sampling:
                        conflicts=fixed_cache.static_conflicts(evidence,prep,provider.pcfg.grid)
                    else:
                        plan=map_canonical_evidence(evidence,prep,provider.pcfg.grid)
                        target,valid=repair_targets(evidence,plan,c['gt'])
                    local['live_render_evidence_geometry_labels']+=time.perf_counter()-tick;tick=time.perf_counter()
                    if causal_sampling:
                        repair,_=loss_for_causal(new,evidence,out,prep,provider.pcfg.grid,c['gt'],rng,device,conflicts)
                    else:
                        repair,_=loss_for(new,evidence,plan,out,target,valid,rng,device,prepared=prep,grid=provider.pcfg.grid)
                    motion,_=motion_loss(out,c['record'],device,.8,materialize_stats=False)
                    losses.append(repair+motion);local['sample_encoder_decoder_losses']+=time.perf_counter()-tick
                tick=time.perf_counter();(sum(losses)/len(group)).backward()
                torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step();local['backward_clip_optimizer']+=time.perf_counter()-tick
                if not torch.isfinite(sum(losses)):raise RuntimeError('nonfinite actual joint CCR step')
                if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in joint.transport.parameters()):
                    raise RuntimeError('missing finite live motion gradients')
                del losses,output,outputs,out,prep,evidence,repair,motion
                if not causal_sampling:del plan,target,valid
            synchronize(device);duration=time.perf_counter()-started
            if index:
                seconds+=duration;windows+=len(group);trials.append(duration)
                for k,v in local.items():stages[k]+=v
        report[mode]=dict(seconds_per_window=seconds/windows,windows=windows,batch=4,batch_seconds=trials,
            stage_seconds_per_window={k:v/windows for k,v in stages.items()},
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,actual_backward=True,transport_frozen=False,
            all_sources_retained=True,saved_scientific_updates=False)
        if mode=='CCR':report[mode]['materialization']='full causal support/GT scan, features only on sampled TRAIN points; inference remains full population'
        if mode=='CCR' and causal_sampling:report[mode]['materialization']='full causal domain retained; GT-independent source/class/history stratification BEFORE live projection/GT labels; importance weighted MC; full inference unchanged'
        print(f'CCR_JOINT_SPEED {mode} seconds/window={seconds/windows:.5f}',flush=True)
        del joint,new,optimizer;torch.cuda.empty_cache()
      report['measured_speedup']=report['old_joint']['seconds_per_window']/report['CCR']['seconds_per_window']
      report['boundary']='warm immutable registration/Strong -> batched live motion, fresh support/features/GT labels, head, full backward and optimizer'
      report['scope']='same eight TRAIN representative windows and batch4; old column CE vs CCR masked role BCE, different sampling units; NOT equivalent-method speedup proof'
      if fixed_cache is not None:report['fixed_history_cache']=fixed_cache.stats()
      if causal_sampling:report['boundary']='warm immutable registration + full fixed canonical inputs -> live motion/render, GT-independent sample, fresh sampled projection/GT targets, spatial head, full backward/optimizer; cache lookup/IO charged'
      return report
    finally:provider.model=old_model;provider.joint=old_joint


def finish_report(report,out):
    q=report['learned']['joint'];old=report['old_joint']['joint']
    means={name:np.mean([s['seconds'] for s in report['speed'] if s['mode']==name]) for name in ('old_joint','CCR')}
    strata={}
    for role in ('representative','high_source_stress'):
        values={name:np.mean([s['seconds'] for s in report['speed'] if s['mode']==name and s['stratum']==role]) for name in means}
        strata[role]=dict(six_seconds=values,FPS={k:6/v for k,v in values.items()},speedup=values['old_joint']/values['CCR'])
    report['aggregate_speed']=dict(six_seconds=means,FPS={k:6/v for k,v in means.items()},speedup=means['old_joint']/means['CCR'],strata=strata,
        boundary='resident source tensors + registered FOUR histories -> fresh prior + live motion + fresh canonical evidence/neighbours + six temporal heads + all SIX dense frames',
        excludes='I/O, initial source extraction/registration, GT/metrics, warmup, correctness hashing; NOT raw-input E2E FPS',
        representative_and_stress_also_reported_separately=True)
    report['gate']=dict(mIoU_within_0_05pp=q['mIoU']>=old['mIoU']-.05,Moving_within_0_10pp=q['MovingMicro']>=old['MovingMicro']-.1,
        all_horizons_Moving_within_0_10pp=all(q['per_horizon'][h]['MovingMicro']>=old['per_horizon'][h]['MovingMicro']-.1
                                            for h in ('1.0','2.0','3.0')),
        speedup_ge_3=report['aggregate_speed']['speedup']>=3)
    report['status']='complete';report.pop('error',None)
    report['route']='local_candidate_only' if all(report['gate'].values()) else 'local_screen_failed_no_automatic_server_training'
    report['warning']='TRAIN16 / DEV12+4 replay is developer screening, not independent validation, not L40S timing. Final fixed-budget candidate, no best-by-dev.'
    (out/'pilot.json').write_text(json.dumps(finite_json(report),ensure_ascii=False,indent=2),encoding='utf-8')
    summary=['===== LOCAL CCR / REAL REPLAY =====',f'GPU: {report["GPU"]}',f'TRAIN16 x {report["fit"]["passes"]}; DEV12 (+4 stress)',
        f'old_joint mIoU={old["mIoU"]:.6f} MovingMicro={old["MovingMicro"]:.6f}',
        f'CCR mIoU={q["mIoU"]:.6f} MovingMicro={q["MovingMicro"]:.6f} delta_vs_old={q["mIoU"]-old["mIoU"]:+.6f}/{q["MovingMicro"]-old["MovingMicro"]:+.6f}',
        'speed='+json.dumps(finite_json(report['aggregate_speed'])),
        'joint_training_probe='+json.dumps(finite_json(report['joint_training_probe'])),
        'gate='+json.dumps(finite_json(report['gate'])),report['route'],report['warning']]
    (out/'summary.txt').write_text('\n'.join(summary)+'\n',encoding='utf-8');print('\n'.join(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--passes',type=int,default=64);p.add_argument('--cpu-workers',type=int,default=2)
    p.add_argument('--remove-weight',type=float,default=.25)
    p.add_argument('--resume-diagnostics',action='store_true',help='reuse completed local fit/scores/speed, never rerun training')
    p.add_argument('--only-joint-probe',action='store_true',help='continue only the eight-window joint timing after saved quality/speed')
    p.add_argument('--summary-only',action='store_true',help='format already-completed results; no CUDA/data/training rerun')
    a=p.parse_args()
    if a.summary_only:
        finish_report(json.loads((Path(a.out_dir)/'pilot.json').read_text(encoding='utf-8')),Path(a.out_dir));return
    if a.passes<1 or a.cpu_workers<1:p.error('positive budgets required')
    if a.only_joint_probe and not a.resume_diagnostics:p.error('--only-joint-probe requires finished head and saved diagnostics')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('actual CUDA/BF16 required')
    out=Path(a.out_dir)
    if out.exists() and not a.resume_diagnostics:p.error('fresh output required; original experiments are read-only')
    if a.resume_diagnostics and not (out/'candidate.pt').is_file():p.error('completed CCR candidate required for diagnostics continuation')
    out.mkdir(parents=True,exist_ok=a.resume_diagnostics);torch.set_num_threads(1);torch.manual_seed(20261006);device=torch.device('cuda')
    bundle=ReplayBundle(a.bundle);report=dict(status='running',protocol=PROTOCOL,thresholds=[.5,.95],history=4,future=6,
        local_only=True,no_KD=True,no_AE=True,no_old_checkpoint_edits=True,no_auto_server_training=True,
        scope='history-evidenced existing source/static repair, NOT never-seen dynamic generation')
    if a.resume_diagnostics:
        report=json.loads((out/'pilot.json').read_text(encoding='utf-8'))
        if (report['protocol']!=PROTOCOL or report['fit']['passes']!=a.passes
                or report['fit']['remove_loss_weight']!=a.remove_weight or report['thresholds']!=[.5,.95]):
            raise RuntimeError('diagnostic continuation contract mismatch')
        report['status']='resuming_diagnostics';report.pop('error',None)
    def save():(out/'pilot.json').write_text(json.dumps(finite_json(report),ensure_ascii=False,indent=2),encoding='utf-8')
    def progress(row):
        with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(finite_json(row),ensure_ascii=False)+'\n')
    save()
    try:
        for member in ('runtime.yaml','checkpoints/epoch_0019.pt','checkpoints/clean_e14.pt'):
            dest=out/Path(member).name
            if a.resume_diagnostics:
                if file_digest(dest)!=bundle.manifest['members'][member]['sha256']:raise RuntimeError('source snapshot changed')
            else:bundle.copy_member(member,dest)
        cfg=load_exported_config(out/'runtime.yaml',bundle.manifest['config_fingerprint'])
        _,teacher=load_joint(out/'epoch_0019.pt',device,reference_sha=CLEAN_SHA256,
            config_sha=bundle.manifest['config_fingerprint'],allow_diagnostic=True)
        teacher.eval().requires_grad_(False);provider=PilotProvider(out/'clean_e14.pt',CLEAN_SHA256,make_prepare_config(cfg),device,a.cpu_workers,teacher,None)
        teacher.columns.column_inference_optimized=True;teacher.columns.column_async_readback=True
        teacher.columns.column_probability_optimized=False;teacher.columns.column_sampling_workers=a.cpu_workers
        teacher.columns.column_inference_verify_remaining=0
        if a.only_joint_probe:
            saved=torch.load(out/'candidate.pt',map_location='cpu',weights_only=True)
            if (saved['teacher_sha256']!=bundle.manifest['teacher_sha256'] or saved['passes']!=a.passes
                    or saved['TRAIN_keys']!=report['TRAIN_keys']):raise RuntimeError('continued candidate/teacher mismatch')
            head=CanonicalRepairHead(teacher.columns.source_dim).to(device);head.load_state_dict(saved['head']);head.eval()
            cases=[]
            for index,meta in enumerate(bundle.manifest['windows']):
                if meta['split']!='train' or meta['stratum']!='representative':continue
                record,raw,labels=bundle.window(index,labels=True);raw['future_gt_occ']=None
                causal={**raw,'_column_causal_preparation':build_fixed_geometry(raw,record,provider.pcfg,provider.strong,a.cpu_workers,teacher.columns.config)}
                cases.append(dict(record=record,causal=causal,gt=labels['future_gt_occ'].numpy()))
                if len(cases)==8:break
            if report.get('joint_training_probe'):report.setdefault('preoptimization_joint_training_probe',report['joint_training_probe'])
            report['joint_training_probe']=joint_probe(provider,teacher,head,cases,device)
            for member in ('runtime.yaml','checkpoints/epoch_0019.pt','checkpoints/clean_e14.pt'):
                if file_digest(out/Path(member).name)!=bundle.manifest['members'][member]['sha256']:raise RuntimeError('original snapshot changed')
            finish_report(report,out);return
        cases=[];counts=defaultdict(int)
        with execution_session(teacher.columns,graphs=True,reuse=False):
          for index,meta in enumerate(bundle.manifest['windows']):
            record,raw,labels=bundle.window(index,labels=True);raw['future_gt_occ']=None
            started=time.perf_counter();causal={**raw,'_column_causal_preparation':build_fixed_geometry(raw,record,provider.pcfg,provider.strong,a.cpu_workers,teacher.columns.config)}
            cold=time.perf_counter()-started
            with torch.no_grad():
                output=teacher.motion(record,device);prep=provider.prepare_columns(None,record,include_gt=False,raw_window=causal,outputs=output)
            tick=time.perf_counter();evidence=build_canonical_evidence(prep,provider.pcfg.grid);plan=map_canonical_evidence(evidence,prep,provider.pcfg.grid)
            evidence_seconds=time.perf_counter()-tick;gt=labels['future_gt_occ'].numpy()
            reference=map_canonical_reference(evidence,prep,provider.pcfg.grid)
            for field in ('flat','base','fallback','legal','context'):
                if not np.array_equal(getattr(plan,field),getattr(reference,field)):
                    raise RuntimeError('batched projection/reference mismatch at '+field)
            del reference
            target,valid=repair_targets(evidence,plan,gt)
            zero=np.zeros(plan.flat.shape,np.float32)
            keep=compose_canonical(prep.baseline,evidence,plan,zero,zero)
            if any(not np.array_equal(x,y) for x,y in zip(keep,prep.baseline)):raise RuntimeError('KEEP baseline exactness failed')
            old_rows=[];old_dense=[]
            with torch.no_grad():
              for h in (() if a.resume_diagnostics else range(6)):
                old=columns.candidate_plan(prep,h,provider.pcfg.grid,teacher.columns.config)
                prob=columns.predict_probabilities(teacher.columns,prep,h,old,provider.pcfg.grid,device,256)
                action=actions_from_probabilities(old,prob,GATES)
                old_rows.append((h,old,action));old_dense.append(compose_dense(prep.baseline[h],old,action))
            c=dict(meta=meta,record=record,causal=causal,prep=prep,output=output,evidence=evidence,plan=plan,target=target,valid=valid,
                   gt=gt,moving=labels['moving_support'].numpy(),old_rows=old_rows,old_dense=old_dense)
            if meta['split']=='dev' and not a.resume_diagnostics:
                for k,v in coverage(c).items():counts[k]+=v
            c.pop('old_rows');cases.append(c)
            progress(dict(event='prepare',**meta,cold_geometry_seconds=cold,canonical_build_and_map_seconds=evidence_seconds,**evidence.audit))
            print(f'CCR_PREP {index+1}/{len(bundle.manifest["windows"])} {meta["split"]}/{meta["stratum"]} points={len(evidence)} seconds={evidence_seconds:.3f}',flush=True)
        train=[c for c in cases if c['meta']['split']=='train']
        dev=[c for c in cases if c['meta']['split']=='dev' and c['meta']['stratum']=='representative']
        stress=[c for c in cases if c['meta']['split']=='dev' and c['meta']['stratum']!='representative']
        if len(train)!=16 or len(dev)!=12 or len(stress)!=4:raise RuntimeError('frozen replay TRAIN16/DEV12+4 identity mismatch')
        head=CanonicalRepairHead(teacher.columns.source_dim).to(device)
        if a.resume_diagnostics:
            saved=torch.load(out/'candidate.pt',map_location='cpu',weights_only=True)
            if (saved['teacher_sha256']!=bundle.manifest['teacher_sha256'] or saved['passes']!=a.passes
                    or saved['TRAIN_keys']!=[c['meta']['key'] for c in train]
                    or report['DEV_keys']!=[c['meta']['key'] for c in dev]):raise RuntimeError('continued population/teacher mismatch')
            head.load_state_dict(saved['head'],strict=True);head.eval();del saved
            print('CCR_DIAGNOSTICS restored finished head/quality/speed; NO retraining',flush=True)
            report.setdefault('preoptimization_quality',{k:copy.deepcopy(report[k]) for k in ('learned','static','dynamic','stress')})
            for mode in ('learned','static','dynamic'):report[mode]=score(dev,head,device,variant=mode)
            report['stress']=score(stress,head,device);save()
        else:
            report.update(GPU=torch.cuda.get_device_name(),torch=str(torch.__version__),TRAIN_keys=[c['meta']['key'] for c in train],
                DEV_keys=[c['meta']['key'] for c in dev],coverage=dict(counts),oracle=score(dev,None,device,variant='oracle'),
                old_joint=score(dev,None,device,variant='old'),old_stress=score(stress,None,device,variant='old'));save()
            print('CCR_ORACLE '+json.dumps(finite_json(report['oracle']['delta_vs_transport_pp'])),flush=True)
            report['fit']=fit(head,train,device,a.passes,progress,a.remove_weight);save()
            for mode in ('learned','static','dynamic'):
                report[mode]=score(dev,head,device,variant=mode);save()
            report['stress']=score(stress,head,device);save()
            torch.save(dict(protocol=PROTOCOL,head=head.state_dict(),teacher_sha256=bundle.manifest['teacher_sha256'],
                TRAIN_keys=report['TRAIN_keys'],passes=a.passes,thresholds=[.5,.95],deployable=False),out/'candidate.pt')
        speed_cases=[train[0],dev[0],next(c for c in train if c['meta']['stratum']!='representative'),stress[0]]
        if a.resume_diagnostics:
            report.setdefault('preoptimization_speed',report['speed']);report['speed']=[]
            report.pop('segmented_profile',None)
        report.setdefault('speed',[])
        with execution_session(teacher.columns,graphs=True,reuse=False):
          for c in speed_cases:
            if len([s for s in report['speed'] if s['key']==c['meta']['key']])==4:continue
            for old in (True,False):timed_full(c,teacher,provider,head,old=old)
            for repeat in range(2):
              for old in ((True,False) if repeat==0 else (False,True)):
                if any(s['key']==c['meta']['key'] and s['repeat']==repeat+1 and s['mode']==('old_joint' if old else 'CCR') for s in report['speed']):continue
                row=timed_full(c,teacher,provider,head,old=old);row.update(key=c['meta']['key'],split=c['meta']['split'],stratum=c['meta']['stratum'],repeat=repeat+1)
                report['speed'].append(row);save();print(f'CCR_FPS {row["mode"]} {row["split"]}/{row["stratum"]} seconds={row["seconds"]:.4f} FPS={6/row["seconds"]:.2f}',flush=True)
        if not report.get('segmented_profile'):report['segmented_profile']=[timed_full(dev[0],teacher,provider,head,profile_stages=True)]
        profiler=cProfile.Profile();profiler.enable();ev=build_canonical_evidence(dev[0]['prep'],provider.pcfg.grid)
        map_canonical_evidence(ev,dev[0]['prep'],provider.pcfg.grid);profiler.disable()
        with (out/'cpu_profile.txt').open('w',encoding='utf-8') as f:pstats.Stats(profiler,stream=f).sort_stats('cumulative').print_stats(30)
        reptrain=[c for c in train if c['meta']['stratum']=='representative']
        report['joint_training_probe']=joint_probe(provider,teacher,head,reptrain,device)
        for member in ('runtime.yaml','checkpoints/epoch_0019.pt','checkpoints/clean_e14.pt'):
            if file_digest(out/Path(member).name)!=bundle.manifest['members'][member]['sha256']:raise RuntimeError('original snapshot changed')
        finish_report(report,out)
    except BaseException as error:report.update(status='failed',error=type(error).__name__+': '+str(error));save();raise
    finally:bundle.close()


if __name__=='__main__':main()
