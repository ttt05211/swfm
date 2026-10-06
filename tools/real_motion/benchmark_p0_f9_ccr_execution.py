#!/usr/bin/env python3
"""Paired REAL replay CCR execution, unchanged model/full support/thresholds.

One run checks byte exact geometry, probabilities, SIX dense forecasts, integer
metrics and actual live motion+repair backward, with cold/warm input cases.
No long training, distilled targets, dev selection or source artifact writes.
"""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
from real_motion.native_column_cpu import prepare_native, get_prepared_native
from real_motion.canonical_causal_repair import (CanonicalRepairHead, build_canonical_evidence,
    map_canonical_evidence, repair_targets, compose_canonical, repair_loss)
from real_motion.canonical_repair_context import (build_causal_strata, sample_causal_points, full_static_conflicts,FixedCanonicalCache)
from real_motion.canonical_repair_batch import batched_repair_losses
from real_motion.local_replay_bundle import ReplayBundle, file_digest
from real_motion.runtime_config import make_prepare_config
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.joint_column_full_common import build_fixed_geometry, pack_records, MOTION_KEYS, LABEL_KEYS
from tools.real_motion.joint_column_common import motion_loss
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import load_exported_config, probabilities, tensor
from tools.real_motion.shared_evidence_pilot_common import fresh_prior
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, finite_json
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics


def sync(device): torch.cuda.synchronize(device)


def exact(a,b,fields):
    for field in fields:
        x,y=getattr(a,field),getattr(b,field)
        if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():
            raise RuntimeError('CCR execution byte mismatch: '+field)


def fixed(prep,grid,kernels,executor=None):
    e=build_canonical_evidence(prep,grid,kernels=kernels,executor=executor)
    e.causal_strata=build_causal_strata(e)
    return e, full_static_conflicts(e,prep,grid)


def selected(e,ids):
    # Preserve sampler's FULL-domain indices/order; features are fixed inputs,
    # never learned activations or future labels.
    from real_motion.canonical_causal_repair import CanonicalEvidence
    return CanonicalEvidence(e.features[ids],e.labels[ids],e.actor[ids],e.classes[ids],
                             e.world[ids],e.presence[ids],e.audit)


@torch.no_grad()
def forecast(c,teacher,provider,head,kernels,*,fresh=True,executor=None):
    stage={}
    def call(name,fn):
        tick=time.perf_counter();v=fn();stage[name]=time.perf_counter()-tick;return v
    sync(provider.device);started=time.perf_counter()
    if fresh:
        template=SimpleNamespace(state=c['causal']['_column_causal_preparation']['prepared_state'])
        state=call('fresh_Strong_KTA',lambda:fresh_prior(template,provider).state)
        raw={**c['causal'],'_column_causal_preparation':{**c['causal']['_column_causal_preparation'],'prepared_state':state}}
    else:raw=c['causal']
    output=call('motion',lambda:teacher.motion(c['record'],provider.device))
    prep=call('live_render',lambda:provider.prepare_columns(None,c['record'],include_gt=False,raw_window=raw,outputs=output))
    e=call('full_canonical_inputs',lambda:build_canonical_evidence(prep,provider.pcfg.grid,kernels=kernels,executor=executor))
    plan=call('six_projection_legality',lambda:map_canonical_evidence(e,prep,provider.pcfg.grid,kernels=kernels,executor=executor))
    pr=call('full_point_encode_six_readouts',lambda:probabilities(head,e,plan,output,provider.device))
    dense=call('six_dense_composition',lambda:compose_canonical(prep.baseline,e,plan,pr[...,0],pr[...,1]))
    sync(provider.device);seconds=time.perf_counter()-started
    if len(dense)!=6 or any(x.shape!=tuple(provider.pcfg.grid.shape_hwd) for x in dense):
        raise RuntimeError('missing complete SIX dense outputs')
    # Force all outputs to be observed, compare OUTSIDE timed region.
    signature=[hashlib.sha256(x.tobytes()).hexdigest() for x in [pr,*dense]]
    return seconds,stage,signature


def train_trial(cases,teacher,provider,head,kernels,*,warm,repeats,executor=None,batched=False):
    torch.manual_seed(20261006)
    joint=copy.deepcopy(teacher).requires_grad_(True).train()
    new=copy.deepcopy(head).requires_grad_(True).train();old_model,old_joint=provider.model,provider.joint
    provider.model=joint.transport;provider.joint=joint
    optimizer=torch.optim.AdamW([dict(params=joint.transport.parameters(),lr=.0005,weight_decay=.0001),
                                dict(params=new.parameters(),lr=.0003,weight_decay=.01)])
    cache=FixedCanonicalCache(128,neighbors=False,kernels=kernels,executor=executor) if warm else None
    cache_build=0.
    if warm:
        tick=time.perf_counter()
        for c in cases:
            ev,_=cache.get(c['prep'],provider.pcfg.grid)
            cache.static_conflicts(ev,c['prep'],provider.pcfg.grid)
        cache_build=time.perf_counter()-tick
    rng=np.random.default_rng(20261006);durations=[];stage_sum=defaultdict(float);losses=[]
    windows=0;torch.cuda.reset_peak_memory_stats()
    try:
        groups=[list(range(i,min(i+4,len(cases)))) for i in range(0,len(cases),4)]
        for iteration in range(1+repeats):
            group=groups[iteration%len(groups)];cs=[cases[i] for i in group]
            sync(provider.device);started=time.perf_counter();optimizer.zero_grad(set_to_none=True);local=defaultdict(float)
            tick=time.perf_counter();merged=pack_records([c['record'] for c in cs],(*MOTION_KEYS,*LABEL_KEYS))
            output=joint.motion(merged,provider.device)
            sizes=[len(c['record']['features']) for c in cs]
            outputs=[dict(zip(output,vs)) for vs in zip(*(v.split(sizes) for v in output.values()))]
            local['live_motion']+=time.perf_counter()-tick;total=[];packed=[]
            for ci,c,out in zip(group,cs,outputs):
                tick=time.perf_counter()
                prep=provider.prepare_columns(None,c['record'],include_gt=False,raw_window=c['causal'],outputs=out)
                local['live_renderer']+=time.perf_counter()-tick;tick=time.perf_counter()
                if warm:
                    e,_=cache.get(prep,provider.pcfg.grid)
                    conflicts=cache.static_conflicts(e,prep,provider.pcfg.grid)
                else:e,conflicts=fixed(prep,provider.pcfg.grid,kernels,executor)
                local['fixed_inputs_conflicts']+=time.perf_counter()-tick;tick=time.perf_counter()
                ids,importance=sample_causal_points(e,rng,per_role=1024)
                sample=selected(e,ids);plan=map_canonical_evidence(sample,prep,provider.pcfg.grid,kernels=kernels)
                for h in range(6):plan.legal[(sample.actor<0)&np.isin(plan.flat[:,h],conflicts[h]),h,0]=False
                y,valid=repair_targets(sample,plan,c['gt']);weight=importance[:,None,None]*valid
                local['sample_live_projection_targets']+=time.perf_counter()-tick;tick=time.perf_counter()
                if batched:
                    packed.append((sample,plan,y,weight))
                    motion,_=motion_loss(out,c['record'],provider.device,.8,materialize_stats=False)
                    total.append(motion);local['head_motion_losses']+=time.perf_counter()-tick
                    continue
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    projected=new.project_sources(out);actor=tensor(sample.actor,provider.device)
                    enc=new.encode(tensor(sample.features,provider.device),tensor(sample.labels,provider.device),
                                   actor,tensor(sample.classes,provider.device),projected)
                    logits=new.decode(enc,actor,tensor(plan.context,provider.device),tensor(plan.base,provider.device),
                                      tensor(plan.fallback,provider.device),tensor(plan.legal,provider.device),projected)
                    repair=repair_loss(new,logits,actor,tensor(y,provider.device),tensor(weight,provider.device))
                motion,_=motion_loss(out,c['record'],provider.device,.8,materialize_stats=False)
                total.append(motion+repair);local['head_motion_losses']+=time.perf_counter()-tick
            if batched:
                tick=time.perf_counter()
                fields=list(zip(*packed))
                repair_terms=batched_repair_losses(new,fields[0],fields[1],output,sizes,fields[2],fields[3],provider.device)
                total=[x+y for x,y in zip(total,repair_terms)]
                local['head_motion_losses']+=time.perf_counter()-tick
            loss=sum(total)/len(cs);tick=time.perf_counter();loss.backward()
            params=[*joint.transport.parameters(),*new.parameters()]
            norm=torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step()
            sync(provider.device);local['backward_clip_optimizer']+=time.perf_counter()-tick;duration=time.perf_counter()-started
            # Checking gradients/reading diagnostics is OUTSIDE throughput timer.
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('nonfinite actual joint update')
            if not any(p.grad is not None and p.grad.abs().sum()>0 for p in joint.transport.parameters()):
                raise RuntimeError('motion gradients missing')
            if not any(p.grad is not None and p.grad.abs().sum()>0 for p in new.parameters()):
                raise RuntimeError('repair gradients missing')
            if iteration:
                durations.append(duration);losses.append(float(loss.detach()));windows+=len(cs)
                for k,v in local.items():stage_sum[k]+=v
            del output,outputs,out,loss,total,motion
            if batched:del packed,fields,repair_terms
            else:del repair,logits,enc,projected
        return dict(seconds_per_window=sum(durations)/windows,windows=windows,batch=4,
                    stage_seconds_per_window={k:v/windows for k,v in stage_sum.items()},losses=losses,
                    cache_prefill_seconds=cache_build,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    actual_joint_backward=True,transport_frozen=False,source_population_unchanged=True,
                    batched_point_head=batched,
                    saved_scientific_updates=False)
    finally:
        provider.model=old_model;provider.joint=old_joint
        if cache is not None:cache.close()
        del joint,new,optimizer;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--reference-run',required=True)
    p.add_argument('--out-dir',required=True);p.add_argument('--fps-windows',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--train-repeats',type=int,default=4)
    p.add_argument('--cpu-workers',type=int,default=4)
    p.add_argument('--training-only',action='store_true',help='eight TRAIN windows, skip FPS/dev; separate real joint backward diagnostic')
    a=p.parse_args();out=Path(a.out_dir);ref=Path(a.reference_run)
    if out.exists() or min(a.repeats,a.train_repeats,a.fps_windows,a.cpu_workers)<1 or a.cpu_workers>8:p.error('fresh output, positive fixed budgets and <=8 workers required')
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA required, no fake timing')
    out.mkdir(parents=True);torch.set_num_threads(1);device=torch.device('cuda')
    native=prepare_native(out/'native_build');kernels=get_prepared_native();bundle=ReplayBundle(a.bundle)
    pool=ThreadPoolExecutor(max_workers=a.cpu_workers)
    immutable={name:file_digest(ref/name) for name in ('candidate.pt','epoch_0019.pt','clean_e14.pt','runtime.yaml')}
    r=dict(status='running',protocol='p0_f9_ccr_exact_execution_v1',GPU=torch.cuda.get_device_name(),
           native=native,history=4,future=6,weights_changed=False,thresholds=[.5,.95],KD=False,AE=False,
           manifest_fingerprint=bundle.manifest['manifest_fingerprint'],local_screen_only=True,source_artifact_sha256=immutable)
    r['training_only']=a.training_only
    def save():
        (out/'speed.json').write_text(json.dumps(finite_json(r),indent=2,ensure_ascii=False),encoding='utf-8')
    save();cases=[]
    try:
        for member in ('runtime.yaml','checkpoints/epoch_0019.pt','checkpoints/clean_e14.pt'):
            if immutable[Path(member).name]!=bundle.manifest['members'][member]['sha256']:
                raise RuntimeError('reference/replay checkpoint or config mismatch')
        cfg=load_exported_config(ref/'runtime.yaml',bundle.manifest['config_fingerprint'])
        _,teacher=load_joint(ref/'epoch_0019.pt',device,reference_sha=CLEAN_SHA256,
                             config_sha=bundle.manifest['config_fingerprint'],allow_diagnostic=True)
        teacher.eval().requires_grad_(False)
        provider=PilotProvider(ref/'clean_e14.pt',CLEAN_SHA256,make_prepare_config(cfg),device,2,teacher,None)
        saved=torch.load(ref/'candidate.pt',map_location='cpu',weights_only=True)
        if saved['teacher_sha256']!=bundle.manifest['teacher_sha256']:raise RuntimeError('CCR teacher identity mismatch')
        head=CanonicalRepairHead(teacher.columns.source_dim).to(device);head.load_state_dict(saved['head'],strict=True)
        head.eval().requires_grad_(False);cold=defaultdict(float);metrics={name:Metrics() for name in ('reference','native')}
        for i,meta in enumerate(bundle.manifest['windows']):
            if a.training_only and (meta['split']!='train' or meta['stratum']!='representative'):continue
            if a.training_only and len(cases)==8:break
            record,raw,labels=bundle.window(i,labels=True);raw['future_gt_occ']=None
            tick=time.perf_counter();causal={**raw,'_column_causal_preparation':build_fixed_geometry(raw,record,provider.pcfg,provider.strong,2,teacher.columns.config)}
            cold['initial_source_registration_Strong']+=time.perf_counter()-tick
            with torch.no_grad():
                output=teacher.motion(record,device);prep=provider.prepare_columns(None,record,include_gt=False,raw_window=causal,outputs=output)
                tick=time.perf_counter();e=build_canonical_evidence(prep,provider.pcfg.grid);cold['reference_evidence']+=time.perf_counter()-tick
                tick=time.perf_counter();new=build_canonical_evidence(prep,provider.pcfg.grid,kernels=kernels,executor=pool);cold['native_parallel_evidence']+=time.perf_counter()-tick
                exact(e,new,('features','labels','actor','classes','world','presence'))
                tick=time.perf_counter();plan=map_canonical_evidence(e,prep,provider.pcfg.grid);cold['reference_six_plan']+=time.perf_counter()-tick
                tick=time.perf_counter();np_=map_canonical_evidence(new,prep,provider.pcfg.grid,kernels=kernels,executor=pool);cold['native_parallel_six_plan']+=time.perf_counter()-tick
                exact(plan,np_,('flat','base','fallback','legal','context'))
                if not a.training_only:
                    pr=probabilities(head,e,plan,output,device);pn=probabilities(head,new,np_,output,device)
                    if pr.tobytes()!=pn.tobytes():raise RuntimeError('full probability bytes changed')
                    d=compose_canonical(prep.baseline,e,plan,pr[...,0],pr[...,1])
                    dn=compose_canonical(prep.baseline,new,np_,pn[...,0],pn[...,1])
                    if any(x.tobytes()!=y.tobytes() for x,y in zip(d,dn)):raise RuntimeError('full SIX dense bytes changed')
            gt=labels['future_gt_occ'].numpy();moving=labels['moving_support'].numpy()
            if meta['split']=='dev' and meta['stratum']=='representative':
                for ri,h in enumerate((1,3,5)):
                    metrics['reference'].update(ri,d[h],gt[h],moving[h]);metrics['native'].update(ri,dn[h],gt[h],moving[h])
            cases.append(dict(meta=meta,record=record,causal=causal,prep=prep,gt=gt))
            print(f'CCR_EXACT {i+1}/{len(bundle.manifest["windows"])} points={len(e)} PASS',flush=True)
            del e,new,plan,np_,output
            if not a.training_only:del pr,pn,d,dn
        r['exactness']=dict(windows=len(cases),full_support_features_bytes=True,full_six_plan_bytes=True,
                            probabilities_bytes=not a.training_only,six_dense_bytes=not a.training_only)
        r['quality']={name:m.compute() for name,m in metrics.items()} if not a.training_only else {}
        r['cold_preparation_seconds']=dict(cold);save()
        dev=[c for c in cases if c['meta']['split']=='dev'];regular=[c for c in dev if c['meta']['stratum']=='representative']
        stress=[c for c in dev if c['meta']['stratum']!='representative']
        timed=regular[:max(a.fps_windows-min(2,len(stress)),1)]+stress[:min(2,len(stress))]
        samples=[]
        for ci,c in enumerate(timed):
            expected=None
            for repeat in range(1+a.repeats):
                names=('reference','native','native_parallel')
                for name in names[repeat%3:]+names[:repeat%3]:
                    sec,st,sig=forecast(c,teacher,provider,head,kernels if name!='reference' else None,
                                        executor=pool if name=='native_parallel' else None)
                    if expected is None:expected=sig
                    elif expected!=sig:raise RuntimeError('paired FPS output bytes changed')
                    if repeat:samples.append(dict(mode=name,seconds=sec,stages=st,key=c['meta']['key'],stratum=c['meta']['stratum']))
            print(f'CCR_FPS {ci+1}/{len(timed)} reference/native full SIX PASS',flush=True)
        r['six_frame']={}
        for name in ('reference','native','native_parallel'):
            rows=[x for x in samples if x['mode']==name]
            if not rows:continue
            sec=np.mean([x['seconds'] for x in rows])
            stages=set().union(*(x['stages'] for x in rows))
            r['six_frame'][name]=dict(seconds=float(sec),FPS=6/float(sec),windows=len(timed),repeats=a.repeats,
                host_stage_seconds={k:float(np.mean([x['stages'].get(k,0) for x in rows])) for k in stages})
        r['samples']=samples;r['inference_speedup']={name:r['six_frame']['reference']['seconds']/r['six_frame'][name]['seconds'] for name in ('native','native_parallel') if name in r['six_frame']};save()
        train=[c for c in cases if c['meta']['split']=='train'][:8];r['joint_training']={}
        for warm in (False,True):
            for name in ('reference','native','native_parallel'):
                key=name+('_warm' if warm else '_cold')
                r['joint_training'][key]=train_trial(train,teacher,provider,head,kernels if name!='reference' else None,
                                                      warm=warm,repeats=a.train_repeats,executor=pool if name=='native_parallel' else None)
                print(f'CCR_JOINT {key} seconds/window={r["joint_training"][key]["seconds_per_window"]:.5f}',flush=True);save()
        r['training_speedup']={backend:{name:r['joint_training']['reference_'+name]['seconds_per_window']/r['joint_training'][backend+'_'+name]['seconds_per_window']
                                      for name in ('cold','warm')} for backend in ('native','native_parallel')}
        for warm in (False,True):
            key='native_parallel_batched'+('_warm' if warm else '_cold')
            r['joint_training'][key]=train_trial(train,teacher,provider,head,kernels,warm=warm,repeats=a.train_repeats,executor=pool,batched=True)
            base=r['joint_training']['reference_'+('warm' if warm else 'cold')]
            r.setdefault('batched_training_speedup',{})['warm' if warm else 'cold']=base['seconds_per_window']/r['joint_training'][key]['seconds_per_window']
            print(f'CCR_JOINT {key} seconds/window={r["joint_training"][key]["seconds_per_window"]:.5f}',flush=True);save()
        for name,digest in immutable.items():
            if file_digest(ref/name)!=digest:raise RuntimeError('source checkpoint/config changed')
        r['exactness']['source_artifacts_unchanged']=True;r['status']='complete';r['native']=kernels.info()
        lines=['===== EXACT CANONICAL CCR EXECUTION / REAL REPLAY =====',f'GPU={r["GPU"]} 4 histories -> 6 futures; same weights/support/thresholds',
               f'exactness={r["exactness"]}',f'quality={r["quality"]}',f'six_frame={r["six_frame"]}',
               f'inference_speedup={r["inference_speedup"]}',f'joint_training={r["joint_training"]}',f'training_speedup={r["training_speedup"]}',f'batched_training_speedup={r["batched_training_speedup"]}',
               'FPS=6/mean completed SIX latency. Includes fresh Strong/KTA, live motion/render, full canonical inputs/legality/head/composition.',
               'Excludes disk/initial source extraction-registration/GT/metrics/compilation/warmup/hashes. NOT raw-sensor E2E.',
               'Cold training builds full fixed inputs; warm uses fixed causal descriptors only. Both include live motion/repair backward and AdamW.',
               'No long training, threshold search, checkpoint promotion or L40S speed claim. Point CCR quality gap versus old Local remains.']
        (out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');save();print('\n'.join(lines),flush=True)
    except BaseException as exc:
        r['status']='failed';r['error']=str(exc);save();raise
    finally:bundle.close();pool.shutdown(wait=True)


if __name__=='__main__':main()
