"""Frozen SAME-window GT H6 inference A/B. No training or new quality scores."""
import argparse
from pathlib import Path
import signal
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.stc_frozen_speed.freeze import load_frozen
from tools.stc_frozen_speed.run import read_history,workload,SCOPE,EXCLUDES
from tools.stc_frozen_speed.engine import sync
from tools.stc_norm_final.common import read,write
from tools.stc_overnight.bundle import Bundle
from tools.final_semantic_evaluation.sources import NuscInputs,window
from tools.final_semantic_evaluation.predictor import Predictor
from tools.shared_source_track.adapter import SharedTrackPredictor
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from real_motion.strong_majority_execution import ParallelNativeMajority
from . import engine


def aggregate(rows,backends,repeats,actual_cuda):
    result={}
    for name in backends:
        seconds=[t for row in rows for group in row['seconds'][name] for t in group]
        complete=bool(rows) and all(len(row['seconds'][name])==2 and all(len(g)==repeats for g in row['seconds'][name]) for row in rows)
        means=[float(np.mean([t for row in rows for t in row['seconds'][name][i]])) for i in range(2)] if rows else []
        changes=[v for row in rows for v in row['changed_voxels'][name]]
        result[name]=dict(mean_six_frame_ms=1000*float(np.mean(seconds)) if seconds else None,
            output_frame_fps=6*len(seconds)/sum(seconds) if actual_cuda and complete else None,
            window_fps=len(seconds)/sum(seconds) if actual_cuda and complete else None,
            pass_mean_ms=[1000*v for v in means],max_changed_voxels=max(changes,default=0),
            strict_byte_parity=complete and not any(changes),all_timings_retained=True,
            peak_allocated_mib=max((row['peak_allocated_mib'][name] for row in rows),default=0),
            peak_reserved_mib=max((row['peak_reserved_mib'][name] for row in rows),default=0),
            profile_mean_ms={k:1000*float(np.mean([row['profile'][name][k] for row in rows if name in row['profile']]))
                for k in next((row['profile'][name] for row in rows if name in row['profile']),{})})
    for name in backends:
        if name=='reference' or not rows:continue
        r=result[name];ref=result['reference'];r['speedup']=ref['mean_six_frame_ms']/r['mean_six_frame_ms']
        r['faster_each_pass']=all(x<y for x,y in zip(r['pass_mean_ms'],ref['pass_mean_ms']))
        r['strict_pass_gate']=r['faster_each_pass'] and r['strict_byte_parity'] and ref['strict_byte_parity']
        if name=='packed_splat_cpu4' and 'combined_cpu4' in result:
            previous=result['combined_cpu4']
            r['speedup_over_combined_cpu4']=previous['mean_six_frame_ms']/r['mean_six_frame_ms']
            r['faster_each_pass_than_combined_cpu4']=all(x<y for x,y in zip(r['pass_mean_ms'],previous['pass_mean_ms']))
        if engine.cpu4(name):
            r['Strong_six_frame_byte_parity']=bool(rows) and all(row['strong_audit']['strict_byte_parity'] for row in rows)
            r['Strong_max_changed_voxels']=max((max(row['strong_audit']['changed_voxels_by_horizon']) for row in rows),default=0)
            r['strict_pass_gate']=r['strict_pass_gate'] and r['Strong_six_frame_byte_parity']
    return result


def summary(r):
    lines=['===== FROZEN GT / HIGH-LEVEL INFERENCE REUSE A/B =====',
        f"status={r['status']}; windows={r['windows']}/{len(r['contract']['keys'])}; ONE GPU/ONE model process; H6",
        'SAME frozen norm192 model, dense-valid GT history, GT future ego; no training/labels/new quality scores.',
        'No custom CUDA/Triton/compile, no precision/model/threshold changes. Fresh Strong EVERY call.',
        'Two passes in alternating backend order; warmup/profile excluded; ALL timings retained even on byte FAIL.',
        'backend                 six_frame_ms output_frame_fps speedup max_changed_voxels strict_byte']
    for name,v in r['speed'].items():
        lines.append(f"{name:22s} {v['mean_six_frame_ms']} {v['output_frame_fps']} {v.get('speedup',1.)} {v['max_changed_voxels']} {v['strict_byte_parity']}")
        lines.append(name+' serialized_profile_mean_ms='+str(v['profile_mean_ms']))
        if 'speedup_over_combined_cpu4' in v:lines.append('vs previous combined_cpu4 speedup='+str(v['speedup_over_combined_cpu4'])+' faster_each_pass='+str(v['faster_each_pass_than_combined_cpu4']))
        if engine.cpu4(name):lines.append('EXISTING native majority CPU4 (different CPU thread budget): Strong six-frame byte parity='+str(v.get('Strong_six_frame_byte_parity'))+' max_changed_voxels='+str(v.get('Strong_max_changed_voxels')))
    lines += ['Original reference-repeat drift remains FAIL if present. No tolerance relaxation or automatic backend promotion.',
        'SCOPE: '+SCOPE,'EXCLUDES: '+EXCLUDES,
        'Not camera end-to-end FPS; new GT quality is not evaluated. No forecast/learned-field caching across calls.']
    return '\n'.join(lines)+'\n'


@torch.no_grad()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--source-fps',required=True);ap.add_argument('--bundle',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--windows',type=int,default=32);ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--backends',nargs='+',choices=engine.BACKENDS,default=list(engine.BACKENDS))
    ap.add_argument('--portable-subset',action='store_true');ap.add_argument('--device',default='cuda');ap.add_argument('--cpu-workers',type=int,default=2)
    for name in ('dev-cache','dev-info','dataroot'):ap.add_argument('--'+name,default='')
    a=ap.parse_args(argv)
    if a.windows<1 or a.repeats<1 or 'reference' not in a.backends or len(set(a.backends))!=len(a.backends):raise ValueError('positive fixed budget and unique reference/candidates required')
    if a.device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('ONE visible GPU required')
    source=Path(a.source_fps).resolve();proof=read(source/'results.json');contract=proof['contract']
    if proof['status']!='complete' or proof['windows']!=len(contract['keys']):raise ValueError('completed source FPS evidence required')
    if base.sha256(a.bundle)!=contract['bundle_sha256']:raise ValueError('original bundle required')
    if not a.portable_subset and (len(contract['keys'])!=128 or contract.get('modalities')!=['gt','stc']):raise ValueError('completed SAME GT/STC128 source benchmark required')
    if not a.portable_subset and a.cpu_workers!=contract['cpu_workers']:raise ValueError('reference CPU worker budget must match source FPS; CPU4 is a separately labeled candidate')
    torch.set_num_threads(1);torch.manual_seed(21);body=load_frozen(source/'frozen',a.device)
    fingerprint=tensor_state_fingerprint(body.state_dict())
    if fingerprint!=contract['merged_state_sha256']:raise ValueError('source merged model differs')
    receipt=read(source/'frozen/receipt.json');payload=torch.load(source/'frozen/final_model.pt',map_location='cpu',weights_only=False)
    bias=payload['posterior_bias'];policy=payload['source_contract'].get('source_policy','baseline');del payload
    bundle=Bundle(a.bundle,cache_mib=64)
    if a.portable_subset:
        records=[r for r in bundle.records if r['role']=='reference64'];inputs=None
        selected=base.select_scene_balanced_round_robin([(r['scene_name'],r['t0_token']) for r in records],a.windows)
    else:
        for name in ('dev_cache','dev_info'):
            if base.sha256(getattr(a,name))!=contract['metadata'][name]:raise ValueError('same metadata required')
        _,records=base.load_cache(a.dev_cache);selected=contract['keys'][:a.windows]
        inputs=NuscInputs(a.dataroot,a.dev_info,cache_mib=64)
    lookup={(r['scene_name'],r['t0_token']):r for r in records};records=[lookup[tuple(k)] for k in selected]
    if len(records)!=a.windows:raise ValueError('insufficient fixed population; no replacement')
    if inputs is not None:inputs.preflight([window(r) for r in records],'occ')
    job=dict(protocol='frozen_GT_high_level_reuse_v1',keys=[list(k) for k in selected],source_fps_sha256=base.sha256(source/'results.json'),
        frozen_checkpoint_sha256=receipt['frozen_checkpoint_sha256'],merged_state_sha256=fingerprint,
        backends=a.backends,repeats=a.repeats,cpu_workers=a.cpu_workers,passes=2,horizon_group=6,
        reference_native_majority_workers=min(a.cpu_workers,4),
        candidate_native_majority_workers=4 if any(map(engine.cpu4,a.backends)) else None,
        scientific_updates=0,history='GT dense-valid',GT_future_ego_conditioned=True,target_future_occupancy_read=False,
        fps_scope=SCOPE,fps_excludes=EXCLUDES,
        implementation={p.name:base.sha256(p) for p in Path(__file__).parent.glob('*.py')})
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);write(out/'contract.json',job)
    p=(SharedTrackPredictor if policy=='shared_pinned' else Predictor)(body,bundle.pcfg,a.device,workers=a.cpu_workers)
    majority=ParallelNativeMajority(4) if any(map(engine.cpu4,a.backends)) else None
    p.gt_reuse_majority=majority
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    rows=[];status='running';tic=time.perf_counter()
    def publish():
        result=dict(status=status,windows=len(rows),contract=job,speed=aggregate(rows,a.backends,a.repeats,a.device.startswith('cuda')),
            wall_seconds=time.perf_counter()-tic)
        write(out/'results.json',result);(out/'summary.txt').write_text(summary(result),encoding='utf-8');return result
    try:
        for i,r in enumerate(records):
            if event.is_set():raise InterruptedError('stop at complete benchmark window')
            rec,raw,times,poses=read_history(bundle,inputs,r,'gt');row=p.prepare(rec,raw,times=times);counts=workload(row)
            ready=engine.prepared(p,row,poses,times)
            audit=engine.strong_audit(p,ready) if majority is not None else None
            # No legacy paired GC control here: source benchmark already measured it.
            baseline=engine.fresh(p,ready,bias,'reference')
            for name in a.backends:engine.fresh(p,ready,bias,name)
            sample=dict(key=selected[i],workload=counts,strong_audit=audit,seconds={n:[] for n in a.backends},changed_voxels={n:[] for n in a.backends},
                peak_allocated_mib={},peak_reserved_mib={},profile={})
            for pass_index in range(2):
                order=a.backends if (i+pass_index)%2==0 else a.backends[::-1]
                for name in order:
                    seconds=[]
                    if p.provider.device.type=='cuda':torch.cuda.reset_peak_memory_stats(p.provider.device)
                    for _ in range(a.repeats):
                        sync(p.provider.device);start=time.perf_counter();actual=engine.fresh(p,ready,bias,name);sync(p.provider.device)
                        seconds.append(time.perf_counter()-start)
                        sample['changed_voxels'][name].append(int(np.count_nonzero(actual!=baseline)))
                    sample['seconds'][name].append(seconds)
                    sample['peak_allocated_mib'][name]=max(sample['peak_allocated_mib'].get(name,0.),torch.cuda.max_memory_allocated(p.provider.device)/2**20 if p.provider.device.type=='cuda' else 0.)
                    sample['peak_reserved_mib'][name]=max(sample['peak_reserved_mib'].get(name,0.),torch.cuda.max_memory_reserved(p.provider.device)/2**20 if p.provider.device.type=='cuda' else 0.)
            if i<4:
                for name in a.backends:
                    profile={};engine.fresh(p,ready,bias,name,profile);sample['profile'][name]=profile
            write(out/f'window_{i:03d}.json',sample);rows.append(sample)
            if (i+1)%4==0:publish();print('GT_REUSE_AB',i+1,'/',len(records),'seconds',round(time.perf_counter()-tic,1),flush=True)
            del ready,row,raw,rec,baseline,actual
        status='complete';print(summary(publish()),flush=True);return 0
    except InterruptedError:status='stopped_at_complete_window';publish();return 130
    except Exception:status='failed';publish();raise
    finally:
        if majority is not None:majority.close()
        p.close();bundle.close()
        if tensor_state_fingerprint(body.state_dict())!=fingerprint:raise RuntimeError('frozen source model changed')


if __name__=='__main__':raise SystemExit(main() or 0)
