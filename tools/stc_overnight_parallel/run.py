"""Two disjoint GPU shards, H6 readout, immutable imported serial prefix."""
import argparse
from contextlib import contextmanager,ExitStack
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
from tools.stc_overnight.bundle import Bundle
from tools.stc_overnight import verify_all as old
from .common import PROTOCOL,read,write,routes,parse_counts,serialized,inspect_source,snapshot,request_source_stop,merge,result,summary,check_snapshot


@torch.no_grad()
def forecast(predictor,row,poses,bias,support,group_size):
    context=old.prepare_forecast(predictor,row,poses);outputs={}
    for prefix in ('','supported_') if support else ('',):
        if prefix:old.temporal_static_prior(context,support_only=True)
        for name in ['transport','semantic_refine']+(['calibrated_semantic_refine'] if bias is not None else []):outputs[prefix+name]=[]
        for hs,fw in old.iter_forecasts(context,group_size=group_size):
            for name,key in [('transport','prior'),('semantic_refine','logits')]:
                outputs[prefix+name].append(fw[key].argmax(1).reshape(len(hs),*fw['shape']).byte().cpu().numpy())
            if bias is not None:
                x=fw['logits']+fw['logits'].new_tensor(bias)[None,:,None]
                outputs[prefix+'calibrated_semantic_refine'].append(x.argmax(1).reshape(len(hs),*fw['shape']).byte().cpu().numpy())
            del fw
    return {n:np.concatenate(v) for n,v in outputs.items()}


@contextmanager
def source_guard(source,stop,event):
    lock=source.parent/'.stc_verify_locks'/base.stable_json_fingerprint(str(source));lock.mkdir(parents=True,exist_ok=True)
    if stop:request_source_stop(source)
    start=time.monotonic()
    while True:
        cm=evaluation_lock(lock)
        try:cm.__enter__();break
        except RuntimeError:
            if not stop:raise RuntimeError('old evaluator is still running; use --stop-source for exact-PID graceful handoff')
            if event.is_set() or time.monotonic()-start>120:raise RuntimeError('source lock not released; no force kill or duplicate evaluation started')
            print('HANDOFF waiting for old task to save a complete window...',flush=True);time.sleep(2)
    try:yield
    finally:cm.__exit__(None,None,None)


def records_for(job,bundle):
    c=job['legacy_contract'];paths=job['data_paths']
    if c['portable_subset']:records=[r for r in bundle.records if r['role']=='reference64']
    else:
        for name,digest in c['metadata'].items():
            if base.sha256(paths[name])!=digest:raise ValueError('metadata differs: '+name)
        _,records=base.load_cache(paths['dev_cache'])
    if [[r['scene_name'],r['t0_token']] for r in records]!=c['keys']:raise ValueError('population/order differs from original evaluator')
    return records


def worker(a,event):
    out=Path(a.out_dir).resolve();job=read(out/'contract.json');c=job['legacy_contract'];rank=a.worker_rank
    check_snapshot(job,out)
    if torch.cuda.device_count()!=1 and job['device'].startswith('cuda'):raise ValueError('ONE visible GPU required')
    torch.set_num_threads(1);torch.manual_seed(21)
    folder=out/f'shard_{rank:02d}';folder.mkdir(exist_ok=True);path=folder/'state.json'
    expected=c['keys'][job['imported_windows']:][rank::2];names=routes(c);job_sha=base.stable_json_fingerprint(job)
    state=dict(job_sha256=job_sha,rank=rank,keys=[],counts=serialized({n:metrics.empty(3) for n in names}))
    if path.exists():state=read(path)
    if state['job_sha256']!=job_sha or state['rank']!=rank or state['keys']!=expected[:len(state['keys'])]:raise ValueError('worker resume differs')
    counts=parse_counts(state['counts'],len(state['keys']),job['voxel_count'],names)
    def publish(status):
        state.update(status=status,counts=serialized(counts));write(path,state)
    if state['keys']==expected:publish('complete');return 0
    b=Bundle(job['data_paths']['bundle'],cache_mib=64);body,saved,policy=b.load_model(c['model'],job['device'])
    if saved['source_sha256']!=c['source_sha256'] or policy!=c['source_policy']:raise ValueError('worker body differs')
    records=records_for(job,b);lookup={(r['scene_name'],r['t0_token']):r for r in records}
    inputs=None if c['portable_subset'] else NuscInputs(job['data_paths']['dataroot'],job['data_paths']['dev_info'],
        stc_root=job['data_paths']['stc_root'],cache_mib=64)
    if inputs is not None:inputs.preflight([window(lookup[tuple(k)]) for k in expected],'stc')
    calibration=read(out/'source_snapshot/unlabeled_posterior_fit.json')['bias'] if c.get('posterior_alignment') else None
    p=(SharedTrackPredictor if policy=='shared_pinned' else Predictor)(body,b.pcfg,job['device'],workers=job['cpu_workers'])
    fingerprint=tensor_state_fingerprint(body.state_dict());tic=time.monotonic();status='running'
    publish(status)
    try:
        for i,key in enumerate(expected):
            if i<len(state['keys']):continue
            if event.is_set():raise InterruptedError('stop at complete paired window')
            r=lookup[tuple(key)]
            if inputs is None:rec,raw,times=b.history(r);poses=r['future_poses']
            else:
                w=window(r);rec,raw,times=inputs.history(w,'stc');poses=inputs.poses(w)
            row=p.prepare(rec,raw,times=times)
            original=forecast(p,row,poses,None,False,job['horizon_group'])
            modified,_=old.candidate(row,p.provider,times)
            changed=forecast(p,modified,poses,calibration,bool(c.get('history_support')),job['horizon_group'])
            target=b.targets(r) if inputs is None else inputs.targets(w,(1,3,5))
            for prefix,values in [('baseline',original),('candidate',changed)]:
                for route,pred in values.items():
                    for h,at in enumerate((1,3,5)):metrics.add(counts[prefix+'/'+route],h,pred[at],target[h])
            state['keys'].append(key)
            if len(state['keys'])%8==0:publish(status)
            if len(state['keys'])%16==0:print('STC_SHARD',rank,len(state['keys']),'/',len(expected),'seconds',round(time.monotonic()-tic,1),flush=True)
        status='complete';return 0
    except InterruptedError:status='stopped_at_complete_paired_window';return 130
    except Exception:status='failed_original_unchanged';raise
    finally:
        p.close();b.close()
        if tensor_state_fingerprint(body.state_dict())!=fingerprint:raise RuntimeError('frozen model changed')
        publish(status)


def parent(a,event):
    torch.set_num_threads(1);source=Path(a.source_dir).resolve();out=Path(a.out_dir).resolve()
    if out==source or source in out.parents or out in source.parents:raise ValueError('NEW independent sibling output required')
    gpu_ids=a.worker_gpus.split(',') if a.worker_gpus else None
    if gpu_ids and (len(gpu_ids)!=2 or len(set(gpu_ids))!=2 or any(not s.strip() for s in gpu_ids)):
        raise ValueError('--worker-gpus requires TWO distinct GPU IDs, e.g. 0,1')
    if gpu_ids and os.environ.get('CUDA_VISIBLE_DEVICES')!=a.worker_gpus:
        raise ValueError('parent CUDA_VISIBLE_DEVICES must equal --worker-gpus')
    if a.device.startswith('cuda') and torch.cuda.device_count()!=(2 if gpu_ids else 1):
        raise ValueError('expected TWO visible GPUs for separate cards, otherwise ONE shared GPU')
    c=inspect_source(source);paths={n:str(Path(getattr(a,n)).resolve()) for n in ['bundle','dev_cache','dev_info','dataroot','stc_root']}
    b=Bundle(paths['bundle'],cache_mib=0);body,saved,policy=b.load_model(c['model'],'cpu')
    if (saved['source_sha256']!=c['source_sha256'] or policy!=c['source_policy']
            or hashlib.sha256(b.zip.read('manifest.json')).hexdigest()!=c['manifest_sha256']):raise ValueError('bundle/body differs')
    volume=int(np.prod(b.pcfg.grid.shape_hwd));del body,saved
    for name,digest in c['metadata'].items():
        if base.sha256(paths[name])!=digest:raise ValueError('metadata differs before handoff: '+name)
    b.close()
    lock=out.parent/'.stc_parallel_locks'/base.stable_json_fingerprint(str(out));lock.mkdir(parents=True,exist_ok=True)
    with evaluation_lock(lock),source_guard(source,a.stop_source,event),ExitStack() as stack:
        if a.resume:
            job=read(out/'contract.json')
            if (job['legacy_contract']!=c or job['source_dir']!=str(source) or job['data_paths']!=paths or job['horizon_group']!=a.horizon_group
                    or job['device']!=a.device or job['cpu_workers']!=a.cpu_workers
                    or job.get('worker_GPUs')!=gpu_ids):raise ValueError('same parallel execution required for resume')
            for name,digest in job['implementation'].items():
                if base.sha256(Path(__file__).parent/name)!=digest:raise ValueError('parallel implementation changed')
            check_snapshot(job,out)
            imported=read(out/'source_snapshot/state.json')
        else:
            out.mkdir(parents=True,exist_ok=False);c=inspect_source(source)
            imported,hashes=snapshot(source,out,c,volume)
            job=dict(protocol=PROTOCOL,source_dir=str(source),legacy_contract=c,data_paths=paths,voxel_count=volume,
                source_snapshot_sha256=hashes,imported_windows=len(imported['keys']),horizon_group=a.horizon_group,
                device=a.device,cpu_workers=a.cpu_workers,processes=2,
                physical_GPUs=(2 if gpu_ids else 1) if a.device.startswith('cuda') else 0,worker_GPUs=gpu_ids,
                visible_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),partition='REMAINING ordered_keys rank::2, NO padding',
                scientific_updates=0,execution=f'imported H1 prefix + H{a.horizon_group} tail; not bitwise identity',
                implementation={p.name:base.sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
            write(out/'contract.json',job)
        if imported['contract']!=job['legacy_contract']:raise ValueError('imported snapshot contract differs')
        children=[];latest=[None,None];last_progress=None
        def publish(status,complete=False):
            nonlocal latest,last_progress
            for rank in range(2):
                path=out/f'shard_{rank:02d}/state.json'
                if path.exists():latest[rank]=read(path)
            counts,keys=merge(job,imported,latest,complete=complete);r=result(job,counts,keys,status)
            write(out/'results.json',r);write(out/'state.json',dict(contract=job,keys=keys,counts=serialized(counts)))
            (out/'summary.txt').write_text(summary(r),encoding='utf-8')
            if len(keys)!=last_progress:
                print('STC_PARALLEL',len(keys),'/',len(c['keys']),'imported=',job['imported_windows'],
                    'shards=',[len(s['keys']) if s else 0 for s in latest],flush=True);last_progress=len(keys)
            return r
        status='running';publish(status)
        try:
            if job['imported_windows']<len(c['keys']):
                for rank in range(2):
                    handle=stack.enter_context((out/f'shard_{rank:02d}.log').open('a',encoding='utf-8'))
                    env=os.environ.copy()
                    if gpu_ids:env['CUDA_VISIBLE_DEVICES']=gpu_ids[rank] # cuda:0 inside EACH child.
                    child=subprocess.Popen([sys.executable,'-u','-m','tools.stc_overnight_parallel.run',
                        '--source-dir',str(source),'--out-dir',str(out),'--worker-rank',str(rank)],
                        cwd=Path(__file__).resolve().parents[2],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    children.append(child);print('GPU_WORKER',rank,'pid=',child.pid,
                        'CUDA_VISIBLE_DEVICES=',env.get('CUDA_VISIBLE_DEVICES'),flush=True)
                signaled=False
                while any(p.poll() is None for p in children):
                    if (event.is_set() or any(p.poll() not in (None,0) for p in children)) and not signaled:
                        for child in children:
                            if child.poll() is None:child.terminate()
                        signaled=True
                    publish(status);time.sleep(2)
                codes=[p.wait() for p in children]
                if event.is_set():status='stopped_workers_saved';return 130
                if codes!=[0,0]:raise RuntimeError('shard failed; inspect shard_00.log/shard_01.log; '+str(codes))
            check_snapshot(job,out);publish('complete',complete=True);status='complete';return 0
        except Exception:status='failed_original_unchanged';raise
        finally:
            for child in children:
                if child.poll() is None:child.terminate()
            for child in children:child.wait()
            final=publish(status,complete=status=='complete');print(summary(final),flush=True)


def main(argv=None,event=None):
    ap=argparse.ArgumentParser();root=Path(__file__).resolve().parents[2]
    ap.add_argument('--source-dir',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--bundle');ap.add_argument('--dev-cache',default=str(root/'data/p0_f9_v18_se2_val_all_4369.pt'))
    ap.add_argument('--dataroot',default='/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes')
    ap.add_argument('--dev-info');ap.add_argument('--stc-root',default=str(root/'data/stc_camera/compact'))
    ap.add_argument('--horizon-group',type=int,choices=[1,2,3,6],default=6);ap.add_argument('--cpu-workers',type=int,choices=[1,2,3,4],default=2)
    ap.add_argument('--device',default='cuda');ap.add_argument('--resume',action='store_true');ap.add_argument('--stop-source',action='store_true')
    ap.add_argument('--worker-gpus',help='optional TWO physical GPU IDs, one process per GPU; otherwise share one visible GPU')
    ap.add_argument('--worker-rank',type=int,choices=[0,1]);a=ap.parse_args(argv);event=event or threading.Event()
    if a.worker_rank is not None:return worker(a,event)
    if not a.bundle:ap.error('--bundle is required')
    a.dev_info=a.dev_info or str(Path(a.dataroot)/'nuscenes_infos_val_temporal_v3_scene.pkl')
    return parent(a,event)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
