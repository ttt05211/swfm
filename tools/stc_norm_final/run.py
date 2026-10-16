"""Independent frozen evaluator, arbitrary 1/2/4 workers on ONE physical GPU.

Probe: same windows, 2/4 then4/2, warmup excluded from steady throughput.
Fixed norm192 and previous model are BOTH scored; no quality-driven selection.
"""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.final_semantic_evaluation import metrics
from tools.final_semantic_evaluation.predictor import Predictor
from tools.final_semantic_evaluation.sources import NuscInputs,window
from tools.shared_source_track.adapter import SharedTrackPredictor
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.shared_history_experiments.refine.geometry import aligned_memory
from tools.stc_overnight.bundle import Bundle
from tools.stc_overnight import verify_all as old
from .common import ROOT,PROTOCOL,BODY_SHA,ROUTES,read,write,parse_counts,serialized,norm_state,assign,merge,report,summary


def implementation():
    paths=list(ROOT.glob('*.py'))+[Path(old.__file__).parent/n for n in ('verify_all.py','replay.py','bundle.py','calibration.py')]
    return {str(p.relative_to(ROOT.parents[1])).replace('\\','/'):base.sha256(p) for p in paths}


def check_job(job,out):
    for name,digest in job['assets'].items():
        if base.sha256(out/'frozen'/name)!=digest:raise ValueError('frozen asset changed: '+name)
    if implementation()!=job['implementation']:raise ValueError('evaluator implementation changed')


@torch.no_grad()
def forecast(p,row,poses,patch,bias,group):
    parameters,original,adapted=patch
    assign(parameters,original)
    c=old.prepare_forecast(p,row,poses);old.temporal_static_prior(c,support_only=True)
    predictions={n:[] for n in ROUTES}
    try:
        for begin in range(0,6,group):
            hs=list(range(begin,min(begin+group,6)))
            prior=c['renderer'].render(c['poses'],c['targets'],c['yaws'],hs)
            memory=aligned_memory(p.body.columns,c['lifted'],c['output'],c['history'],c['raw'],
                c['targets'],c['yaws'],c['poses'],p.provider.pcfg.grid,hs)
            predictions[ROUTES[0]].append(prior.argmax(1).reshape(len(hs),*c['shape']).byte().cpu().numpy())
            for name,values in [(ROUTES[1],original),(ROUTES[2],adapted)]:
                assign(parameters,values)
                logits=p.body.columns(memory,prior,hs,c['shape'])
                x=logits+logits.new_tensor(bias)[None,:,None]
                predictions[name].append(x.argmax(1).reshape(len(hs),*c['shape']).byte().cpu().numpy())
                del logits,x
            assign(parameters,original);del prior,memory
        return {n:np.concatenate(v) for n,v in predictions.items()}
    finally:assign(parameters,original)


def records_for(job,bundle):
    if job['portable_subset']:rows=[r for r in bundle.records if r['role']=='reference64']
    else:
        for name,digest in job['metadata'].items():
            if base.sha256(job['paths'][name])!=digest:raise ValueError('metadata changed: '+name)
        _,rows=base.load_cache(job['paths']['dev_cache'])
    lookup={(r['scene_name'],r['t0_token']):r for r in rows}
    if len(lookup)!=len(rows) or any(tuple(k) not in lookup for k in job['keys']):raise ValueError('population differs')
    return lookup


def worker(out,rank,event):
    out=Path(out).resolve();job=read(out/'contract.json');check_job(job,out)
    if job['device'].startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('ONE visible GPU required')
    torch.set_num_threads(1);torch.manual_seed(21);folder=out/f'shard_{rank:02d}';folder.mkdir(exist_ok=True)
    path=folder/'state.json';expected=job['keys'][rank::job['processes']];digest=base.stable_json_fingerprint(job)
    state=dict(job_sha256=digest,rank=rank,keys=[],counts=serialized({n:metrics.empty(3) for n in ROUTES}),
        measured_windows=0,measured_seconds=0.,warmup_windows=0)
    if path.exists():state=read(path)
    if state['job_sha256']!=digest or state['rank']!=rank or state['keys']!=expected[:len(state['keys'])]:raise ValueError('worker resume differs')
    counts=parse_counts(state['counts'],len(state['keys']),job['voxel_count'],ROUTES)
    def publish(status):state.update(status=status,counts=serialized(counts));write(path,state)
    if state['keys']==expected:publish('complete');return 0
    b=Bundle(job['paths']['bundle'],cache_mib=64);body,saved,policy=b.load_model('clean1000',job['device'])
    if saved['source_sha256']!=BODY_SHA or policy!=job['source_policy']:raise ValueError('fixed source body differs')
    patch=norm_state(body,read(out/'frozen/norm192.json'),saved['source_sha256']);bias=read(out/'frozen/posterior_fit.json')['bias']
    p=(SharedTrackPredictor if policy=='shared_pinned' else Predictor)(body,b.pcfg,job['device'],workers=job['cpu_workers'])
    lookup=records_for(job,b);inputs=None
    if not job['portable_subset']:
        inputs=NuscInputs(job['paths']['dataroot'],job['paths']['dev_info'],stc_root=job['paths']['stc_root'],cache_mib=64)
        inputs.preflight([window(lookup[tuple(k)]) for k in expected],'stc')
    def predict(r):
        if inputs is None:rec,raw,times=b.history(r);poses=r['future_poses']
        else:w=window(r);rec,raw,times=inputs.history(w,'stc');poses=inputs.poses(w)
        row=p.prepare(rec,raw,times=times);row,_=old.candidate(row,p.provider,times)
        return forecast(p,row,poses,patch,bias,job['horizon_group'])
    fingerprint=tensor_state_fingerprint(body.state_dict());status='running';publish(status);tic=time.monotonic()
    try:
        if job['probe']:
            for _ in range(2):
                if event.is_set():raise InterruptedError('probe warmup stopped')
                predict(lookup[tuple(expected[0])]);state['warmup_windows']+=1
            write(folder/'ready.json',dict(job_sha256=digest,rank=rank))
            while not (out/'probe_start.json').exists():
                if event.is_set():raise InterruptedError('probe start cancelled')
                time.sleep(.1)
            if read(out/'probe_start.json')['job_sha256']!=digest:raise ValueError('probe barrier contract differs')
        measured_before=state['measured_seconds'];measure_begin=time.monotonic()
        for key in expected[len(state['keys']):]:
            if event.is_set():raise InterruptedError('stopped at complete window')
            r=lookup[tuple(key)];pred=predict(r)
            gt=b.targets(r) if inputs is None else inputs.targets(window(r),(1,3,5))
            for name in ROUTES:
                for h,at in enumerate((1,3,5)):metrics.add(counts[name],h,pred[name][at],gt[h])
            del pred,gt;gc.collect()
            state['measured_seconds']=measured_before+time.monotonic()-measure_begin
            state['measured_windows']+=1;state['keys'].append(key)
            if len(state['keys'])%8==0:publish(status)
            if len(state['keys'])%16==0:print('NORM192_SHARD',rank,len(state['keys']),'/',len(expected),'seconds',round(time.monotonic()-tic,1),flush=True)
        status='complete';return 0
    except InterruptedError:status='stopped_at_complete_window';return 130
    except Exception:status='failed';raise
    finally:
        assign(patch[0],patch[1]);p.close();b.close()
        if tensor_state_fingerprint(body.state_dict())!=fingerprint:raise RuntimeError('source body not restored')
        publish(status)


def parent(a,event):
    torch.set_num_threads(1)
    if a.device.startswith('cuda') and (torch.cuda.device_count()!=1 or ',' in os.environ.get('CUDA_VISIBLE_DEVICES','')):
        raise ValueError('ONE physical/visible GPU only; use CUDA_VISIBLE_DEVICES=0')
    out=Path(a.out).resolve();paths={n:str(Path(getattr(a,n)).resolve()) for n in ('bundle','dev_cache','dev_info','dataroot','stc_root')}
    if a.portable_subset and a.probe and a.limit and a.limit<a.processes:raise ValueError('probe needs at least one window per worker')
    bundle=Bundle(paths['bundle'],cache_mib=0);body,saved,policy=bundle.load_model('clean1000','cpu')
    patch=read(ROOT/'norm192.json');norm_state(body,patch,saved['source_sha256']);calibration=read(ROOT/'posterior_fit.json')
    if (calibration['source_sha256']!=saved['source_sha256'] or len(calibration['bias'])!=18
            or calibration['target_GT_labels_used_for_fitting'] is not False
            or not np.isfinite(calibration['bias']).all()):raise ValueError('posterior/body differs')
    if a.portable_subset:rows=[r for r in bundle.records if r['role']=='reference64'];metadata={}
    else:
        metadata={n:base.sha256(paths[n]) for n in ('dev_cache','dev_info')}
        if metadata!={n:saved['source_contract']['data'][n] for n in metadata}:raise ValueError('original metadata differs')
        _,rows=base.load_cache(paths['dev_cache'])
        if len(rows)!=4369 or len({r['scene_name'] for r in rows})!=150:raise ValueError('complete ALL4369/150 required')
    keys=[[r['scene_name'],r['t0_token']] for r in rows]
    if len(set(map(tuple,keys)))!=len(keys):raise ValueError('duplicate keys')
    if a.limit:
        if not a.probe:raise ValueError('--limit only for explicit speed probe, never a full score')
        keys=keys[:a.limit]
    if len(keys)<a.processes:raise ValueError('at least one window per worker required')
    job=dict(protocol=PROTOCOL,paths=paths,keys=keys,parent_windows=len(rows),portable_subset=a.portable_subset,
        probe=a.probe,processes=a.processes,physical_GPUs=1 if a.device.startswith('cuda') else 0,
        visible_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),horizon_group=a.horizon_group,device=a.device,cpu_workers=a.cpu_workers,
        source_sha256=saved['source_sha256'],source_policy=policy,manifest_sha256=hashlib.sha256(bundle.zip.read('manifest.json')).hexdigest(),
        metadata=metadata,voxel_count=int(np.prod(bundle.pcfg.grid.shape_hwd)),implementation=implementation(),
        assets={n:base.sha256(ROOT/n) for n in ('norm192.json','posterior_fit.json')},scientific_updates=0,
        GT_future_ego_conditioned=True,known='STC dense-valid; NO GT visibility',metric_mask='none',
        patch='fixed local target-history final200, ONLY192 norm affine parameters; NO fitting here',
        population='speed_probe' if a.probe else 'portable_dev64' if a.portable_subset else 'ALL4369/150')
    del body,saved;bundle.close()
    lock=out.parent/'.stc_norm_locks'/base.stable_json_fingerprint(str(out));lock.mkdir(parents=True,exist_ok=True)
    with evaluation_lock(lock),ExitStack() as stack:
        if a.resume:
            if read(out/'contract.json')!=job:raise ValueError('same directory/processes/assets/population needed for resume')
            check_job(job,out)
        else:
            out.mkdir(parents=True,exist_ok=False);(out/'frozen').mkdir()
            for name in job['assets']:(out/'frozen'/name).write_bytes((ROOT/name).read_bytes())
            write(out/'contract.json',job)
        children=[];states=[None]*a.processes;last=None;tic=time.monotonic()
        def publish(status,complete=False):
            nonlocal last
            for rank in range(a.processes):
                path=out/f'shard_{rank:02d}/state.json'
                if path.exists():states[rank]=read(path)
            counts,seen=merge(job,states,complete);r=report(job,counts,seen,status)
            r['wall_seconds_this_launch']=time.monotonic()-tic
            measured=sum(s.get('measured_windows',0) for s in states if s)
            span=max([s.get('measured_seconds',0.) for s in states if s] or [0.])
            r['speed']=dict(measured_windows=measured,max_worker_measured_seconds=span,
                steady_windows_per_second=measured/span if span else None,
                includes='history IO/prepare, BOTH frozen readouts, labels/metrics; probe warmup excluded; NOT model FPS')
            write(out/'results.json',r);write(out/'state.json',dict(contract=job,keys=seen,counts=serialized(counts)))
            (out/'summary.txt').write_text(summary(r),encoding='utf-8')
            if len(seen)!=last:print('NORM192_ALL',len(seen),'/',len(job['keys']),'workers',a.processes,flush=True);last=len(seen)
            return r
        publish('running')
        try:
            if not all(s is not None and s['keys']==job['keys'][rank::a.processes] for rank,s in enumerate(states)):
                for rank in range(a.processes):
                    handle=stack.enter_context((out/f'shard_{rank:02d}.log').open('a',encoding='utf-8'))
                    child=subprocess.Popen([sys.executable,'-u','-m','tools.stc_norm_final.run','--out',str(out),'--worker-rank',str(rank)],
                        cwd=ROOT.parents[1],env=os.environ.copy(),stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    children.append(child)
            while children and any(ch.poll() is None for ch in children):
                if event.is_set() or any(ch.poll() not in (None,0) for ch in children):
                    for ch in children:
                        if ch.poll() is None:ch.terminate()
                    for ch in children:ch.wait()
                    publish('stopped_or_failed');return 130 if event.is_set() else 1
                if job['probe'] and not (out/'probe_start.json').exists():
                    ready=[out/f'shard_{rank:02d}/ready.json' for rank in range(a.processes)]
                    if all(p.exists() for p in ready):
                        if any(read(p)!=dict(job_sha256=base.stable_json_fingerprint(job),rank=rank) for rank,p in enumerate(ready)):
                            raise ValueError('probe warmup barrier differs')
                        write(out/'probe_start.json',dict(job_sha256=base.stable_json_fingerprint(job)))
                publish('running');time.sleep(2)
            if any(ch.returncode!=0 for ch in children):publish('failed');return 1
            r=publish('complete',True);print(summary(r),flush=True);return 0
        finally:
            for ch in children:
                if ch.poll() is None:ch.terminate()
            for ch in children:ch.wait()


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--bundle',default='')
    ap.add_argument('--dev-cache',default='data/p0_f9_v18_se2_val_all_4369.pt');ap.add_argument('--dev-info',default='')
    ap.add_argument('--dataroot',default='');ap.add_argument('--stc-root',default='data/stc_camera/compact')
    ap.add_argument('--device',default='cuda');ap.add_argument('--processes',type=int,choices=(1,2,4),default=2)
    ap.add_argument('--horizon-group',type=int,choices=(1,2,3,6),default=6);ap.add_argument('--cpu-workers',type=int,default=2)
    ap.add_argument('--portable-subset',action='store_true');ap.add_argument('--probe',action='store_true');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--worker-rank',type=int)
    a=ap.parse_args(argv);event=threading.Event()
    if a.limit<0:raise ValueError('nonnegative limit required')
    def stop(*_):event.set()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    return worker(a.out,a.worker_rank,event) if a.worker_rank is not None else parent(a,event)


if __name__=='__main__':sys.exit(main() or 0)
