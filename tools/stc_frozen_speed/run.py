"""Freeze verified STC+GT norm192; SINGLE GPU/process128-window inference benchmark."""
import argparse
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.final_semantic_evaluation.predictor import Predictor
from tools.final_semantic_evaluation.sources import NuscInputs,window
from tools.shared_source_track.adapter import SharedTrackPredictor
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.stc_overnight.bundle import Bundle
from tools.stc_norm_final.common import read,write,assign
from .freeze import ROOT,assets,freeze,validate_server,load_frozen
from .engine import sync,prepared,fresh,evaluation_ab,aggregate

PROTOCOL='frozen_STC_norm192_single_process128_speed_v1'
SCOPE='prepared four-history geometry/read coordinates/initial H2D -> live KTA + fresh Strong/CLEAR + Encoder/STWM + Transport/support + aligned memory + Refiner + calibration + SIX CPU semantics'
EXCLUDES='disk IO/history CC+association+registration/read-coordinate construction/initial H2D/labels+metrics/warmup/checkpoint writes/serialized profile; NOT raw-camera end-to-end FPS'


def summary(result):
    s=result['speed'];c=result['contract'];lines=['===== FROZEN STC NORM192 / SINGLE PROCESS INFERENCE =====',
        f"status={result['status']}; windows={result['windows']}/{len(c['keys'])}; ONE GPU/ONE model process; H{c['horizon_group']}",
        f"release={c['release']['version']}; scientific_updates=0; target-only final200 norm merged ONCE.",
        f"FPS mean six-frame latency_ms={s['mean_six_frame_ms']}; p90_ms={s['p90_six_frame_ms']}",
        f"window_fps={s['window_fps']}; output_frame_fps={s['output_frame_fps']} (SIX frames/window)",
        f"peak_allocated_MiB={s['peak_allocated_mib']}; peak_reserved_MiB={s['peak_reserved_mib']} (profile/allocator history included)",
        f"strict_byte_parity={s['strict_byte_parity']}; failed_windows={s['byte_failed_windows']}; max_changed_voxels={s['max_changed_voxels']}",
        'ALL timing windows retained even on byte FAIL; actual-execution WARM FPS, not bit-identical-reference/old-model FPS.',
        '===== SAME WINDOWS / FULL INPUT INFERENCE A/B (NOT FPS) =====']
    for name,value in s['evaluation_ab'].items():
        lines.append(f"{name}: seconds/window={value['seconds_per_window']}; forced_GC_seconds/window={value['mean_forced_gc_seconds']}")
    lines+=[f"single-vs-paired speedup={s.get('evaluation_speedup')}; A/B changed_voxels_max={s['evaluation_changed_voxels_max']}",
        'A/B includes history IO/association + inference; excludes labels/metrics/writes in BOTH arms.',
        'GC effect is measured directly; total speedup ALSO removes redundant old-model readout. No promise either improves.',
        'FPS scope: '+SCOPE,'FPS excludes: '+EXCLUDES,
        'GT future ego conditioned; STC dense-valid; no future occupancy read or quality-score selection.',
        'Original evaluator/checkpoints/history caches unchanged; frozen merged checkpoint and receipt are separate.']
    return '\n'.join(lines)+'\n'


@torch.no_grad()
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--bundle',required=True)
    ap.add_argument('--validated-run');ap.add_argument('--portable-subset',action='store_true')
    for name in ('dev-cache','dev-info','dataroot','stc-root'):ap.add_argument('--'+name,default='')
    ap.add_argument('--windows',type=int,default=128);ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--cpu-workers',type=int,default=2);ap.add_argument('--horizon-group',type=int,choices=(1,2,3,6),default=6)
    ap.add_argument('--device',default='cuda');a=ap.parse_args(argv)
    if a.windows<=0 or a.repeats<=0:raise ValueError('positive windows/repeats required')
    if a.device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('ONE visible GPU required')
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    torch.set_num_threads(1);torch.manual_seed(21);out=Path(a.out).resolve()
    evidence=validate_server(a.validated_run) if a.validated_run else None
    if not a.portable_subset and evidence is None:raise ValueError('matching completed ALL4369 --validated-run required for final release')
    bundle=Bundle(a.bundle,cache_mib=64);body,saved,policy=bundle.load_model('clean1000','cpu')
    if a.portable_subset:rows=[r for r in bundle.records if r['role']=='reference64'];metadata={}
    else:
        metadata={n:base.sha256(getattr(a,n)) for n in ('dev_cache','dev_info')}
        if metadata!={n:saved['source_contract']['data'][n] for n in metadata}:raise ValueError('original metadata differs')
        _,rows=base.load_cache(a.dev_cache)
        if len(rows)!=4369 or len({r['scene_name'] for r in rows})!=150:raise ValueError('complete parent population required')
    keys=[(r['scene_name'],r['t0_token']) for r in rows]
    if len(keys)!=len(set(keys)) or a.windows>len(keys):raise ValueError('unique fixed population insufficient; never duplicate/replace windows')
    chosen=base.select_scene_balanced_round_robin(keys,a.windows);lookup=dict(zip(keys,rows));rows=[lookup[tuple(k)] for k in chosen]
    release,patch,posterior=assets();contract=dict(protocol=PROTOCOL,release=release,keys=[list(k) for k in chosen],
        source_sha256=saved['source_sha256'],bundle_sha256=base.sha256(a.bundle),metadata=metadata,
        evidence=evidence,repeats=a.repeats,horizon_group=a.horizon_group,cpu_workers=a.cpu_workers,
        physical_GPU=1 if a.device.startswith('cuda') else 0,processes=1,fps_scope=SCOPE,fps_excludes=EXCLUDES,
        evaluation_ab='legacy paired+forced full GC vs frozen single+ordinary auto GC; alternate order; both exclude labels/metrics',
        target_future_occupancy_read=False,GT_future_ego_conditioned=True,scientific_updates=0,
        implementation={p.name:base.sha256(p) for p in sorted(ROOT.glob('*.py'))},
        numerical_policy='strict byte FAIL retained independently; no changed-voxel cutoff, timing filtering or FP32 fallback')
    lock=out.parent/'.stc_frozen_speed_locks'/base.stable_json_fingerprint(str(out));lock.mkdir(parents=True,exist_ok=True)
    with evaluation_lock(lock):
        out.mkdir(parents=True,exist_ok=False);write(out/'contract.json',contract)
        parameters,original,adapted,payload=freeze(body,saved,out/'frozen',evidence)
        body=load_frozen(out/'frozen','cpu')  # measured model is the materialized frozen artifact
        body.to(a.device);parameters={n:dict(body.named_parameters())[n] for n in parameters}
        original={n:v.to(a.device) for n,v in original.items()};adapted={n:v.to(a.device) for n,v in adapted.items()}
        norm=(parameters,original,adapted);bias=posterior['bias'];assign(parameters,adapted)
        fingerprint=tensor_state_fingerprint(body.state_dict())
        p=(SharedTrackPredictor if policy=='shared_pinned' else Predictor)(body,bundle.pcfg,a.device,workers=a.cpu_workers)
        inputs=None if a.portable_subset else NuscInputs(a.dataroot,a.dev_info,stc_root=a.stc_root,cache_mib=64)
        if inputs is not None:inputs.preflight([window(r) for r in rows],'stc')
        def read_input(r):
            if inputs is None:rec,raw,times=bundle.history(r);poses=r['future_poses']
            else:w=window(r);rec,raw,times=inputs.history(w,'stc');poses=inputs.poses(w)
            return rec,raw,times,poses
        samples=[];tic=time.perf_counter();status='running'
        def publish():
            speed=aggregate(samples,a.repeats,a.device.startswith('cuda'))
            speed['full_population_measured']=len(samples)==len(rows)
            speed['timing_complete']=speed['timing_complete'] and speed['full_population_measured']
            result=dict(status=status,windows=len(samples),contract=contract,speed=speed,
                wall_seconds=time.perf_counter()-tic)
            write(out/'results.json',result);(out/'summary.txt').write_text(summary(result),encoding='utf-8');return result
        try:
            for i,r in enumerate(rows):
                if event.is_set():raise InterruptedError('stopped after complete benchmark window')
                rec,raw,times,poses=read_input(r);row=p.prepare(rec,raw,times=times);ready=prepared(p,row,poses,times)
                for _ in range(3 if i==0 else 1):fresh(p,ready,bias,a.horizon_group)
                ab,reference=evaluation_ab(p,lambda:read_input(r),norm,bias,a.horizon_group,i)
                sync(p.provider.device)
                if p.provider.device.type=='cuda':torch.cuda.reset_peak_memory_stats(p.provider.device)
                seconds=[];changes=[]
                for repeat in range(a.repeats):
                    sync(p.provider.device);start=time.perf_counter();actual=fresh(p,ready,bias,a.horizon_group);sync(p.provider.device)
                    seconds.append(time.perf_counter()-start);changes.append(int(np.count_nonzero(actual!=reference)))
                profile={}
                if i<8:fresh(p,ready,bias,a.horizon_group,profile)
                sample=dict(key=list(chosen[i]),fps_seconds=seconds,fps_changed_voxels=changes,evaluation_ab=ab,profile=profile,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(p.provider.device)/2**20 if p.provider.device.type=='cuda' else None,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(p.provider.device)/2**20 if p.provider.device.type=='cuda' else None)
                write(out/f'window_{i:03d}.json',sample);samples.append(sample)
                if (i+1)%8==0:publish();print('FROZEN_STC_FPS',i+1,'/',len(rows),'seconds',round(time.perf_counter()-tic,1),flush=True)
                del ready,row,reference,actual,raw,rec  # refcount/normal auto GC; NO forced per-window gc.collect
            status='complete';result=publish();print(summary(result),flush=True);return 0
        except InterruptedError:status='stopped_at_complete_window';publish();return 130
        except Exception:status='failed';publish();raise
        finally:
            assign(parameters,adapted);p.close();bundle.close()
            if tensor_state_fingerprint(body.state_dict())!=fingerprint:raise RuntimeError('frozen model changed')


if __name__=='__main__':sys.exit(main() or 0)
