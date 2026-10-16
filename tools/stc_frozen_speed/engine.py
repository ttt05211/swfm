"""Single frozen forward. No forced per-window garbage collection or control route.

FPS boundary matches prior prepared-history -> fresh Strong -> six CPU outputs.
History association/read geometry and initial H2D are outside; learned F, live
KTA, static support, fresh Strong and all projections remain INSIDE the timer.
"""
import gc
import time
import numpy as np
import torch
from tools.stc_norm_final.common import assign
from tools.stc_norm_final.run import forecast as paired_forecast
from tools.stc_overnight import verify_all as old
from tools.shared_source_track.adapter import backgrounds
from real_motion.shared_source_track import from_raw
from tools.shared_history_experiments.refine.geometry import TransportPrior,aligned_memory
from tools.joint_occ_experiments.soft_geometry import source_targets,effective_yaw


def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def prepared(p,row,poses,times):
    rec,raw,ego,history=row;device=p.provider.device
    if '_shared_source_track' not in raw:
        _,track=from_raw(raw,p.provider,times,current=history[0]['state']['current'])
        raw=dict(raw,_shared_source_track=track)
    # Never carry cached future backgrounds, learned fields or future targets.
    state={k:v for k,v in history[0]['state'].items() if k!='column_backgrounds'}
    history=(dict(history[0],state=state),history[1]);row=(rec,raw,ego,history)
    upload=lambda x:torch.as_tensor(np.asarray(x),device=device)
    staged=dict(semantic=upload(raw['history_occ']),observed=upload(raw['history_observed']),
        reads={k:upload(v) for k,v in history[0]['shared_reads'].items()},times=upload(ego[0][:,-1]),
        classes=upload(rec['source_class_id']).long(),poses=upload(poses).double())
    if staged['poses'].shape!=(6,4,4):raise ValueError('complete six future conditions required')
    sync(device);return row,staged,np.asarray(times)


@torch.no_grad()
def fresh(p,ready,bias,group=6,profile=None):
    row,staged,times=ready;device=p.provider.device;body=p.body
    def stage(name,fn):
        if profile is None:return fn()
        sync(device);tic=time.perf_counter();value=fn();sync(device);profile[name]=time.perf_counter()-tic;return value
    modified,_=stage('live_KTA_and_age_weights',lambda:old.candidate(row,p.provider,times))
    rec,raw,ego,history=modified
    def network():
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            field=body.transport.history_encoder(staged['semantic'],staged['observed'])
            output=body.transport(field,staged['reads'],staged['times'],staged['classes'],
                torch.as_tensor(rec['kta_displacement_xy_m'],device=device).float())
            lifted=body.columns.lift_history(field)
        return output,lifted
    output,lifted=stage('shared_encoder_STWM_lift',network)
    state=history[0]['state'];poses=staged['poses']
    bg=stage('fresh_Strong_CLEAR',lambda:backgrounds(raw,state,poses,p.provider,raw['_shared_source_track']))
    state=dict(state,column_backgrounds=bg)
    targets=source_targets(output,rec,state['current_pose'],state['source_z_t0'])
    yaws=effective_yaw(output,[int(s['class_id']) for s in state['current']])
    def renderer():
        value=TransportPrior(raw,state,p.provider.pcfg.grid,device)
        old.temporal_static_prior(dict(renderer=value,raw=raw,poses=poses),support_only=True)
        return value
    transport=stage('prior_setup_and_static_support',renderer);result=[]
    for begin in range(0,6,group):
        hs=list(range(begin,min(begin+group,6)))
        prior=stage(f'transport_H{begin}',lambda:transport.render(poses,targets,yaws,hs))
        memory=stage(f'memory_H{begin}',lambda:aligned_memory(body.columns,lifted,output,history,raw,targets,yaws,poses,p.provider.pcfg.grid,hs))
        logits=stage(f'refiner_H{begin}',lambda:body.columns(memory,prior,hs,transport.shape))
        def decode():
            x=logits+logits.new_tensor(bias)[None,:,None]
            return x.argmax(1).reshape(len(hs),*transport.shape).byte().cpu().numpy()
        result.append(stage(f'calibration_argmax_CPU_H{begin}',decode));del prior,memory,logits
    return np.concatenate(result)


@torch.no_grad()
def single(p,row,poses,times,bias,group):
    # Ordinary reference path, no staged upload; only ONE frozen model readout.
    row,_=old.candidate(row,p.provider,times);c=old.prepare_forecast(p,row,poses)
    old.temporal_static_prior(c,support_only=True);outputs=[]
    for hs,fw in old.iter_forecasts(c,group_size=group):
        x=fw['logits']+fw['logits'].new_tensor(bias)[None,:,None]
        outputs.append(x.argmax(1).reshape(len(hs),*c['shape']).byte().cpu().numpy());del fw,x
    return np.concatenate(outputs)


def evaluation_ab(p,read_input,patch,bias,group,index):
    """Matched full-input inference: legacy paired+forced GC vs single+auto GC.

    Includes history IO/preparation; excludes labels/metrics/writes in BOTH.
    Legacy GC duration is measured separately, NOT assumed to cause slowdown.
    """
    results={};predictions={};names=('legacy_paired_forced_gc','frozen_single_auto_gc')
    for name in names if index%2==0 else names[::-1]:
        assign(patch[0],patch[2]);sync(p.provider.device);tic=time.perf_counter()
        rec,raw,times,poses=read_input();row=p.prepare(rec,raw,times=times)
        if name==names[0]:
            row,_=old.candidate(row,p.provider,times)
            pred=paired_forecast(p,row,poses,patch,bias,group)['norm192/semantic_refine']
            tick=time.perf_counter();reclaimed=gc.collect();gc_seconds=time.perf_counter()-tick
        else:
            pred=single(p,row,poses,times,bias,group);reclaimed=0;gc_seconds=0.
        sync(p.provider.device);seconds=time.perf_counter()-tic
        results[name]=dict(seconds=seconds,forced_gc_seconds=gc_seconds,reclaimed=reclaimed)
        predictions[name]=pred;assign(patch[0],patch[2])
    results['changed_voxels']=int(np.count_nonzero(predictions[names[0]]!=predictions[names[1]]))
    return results,predictions[names[0]]


def aggregate(rows,repeats,actual_cuda):
    samples=[x for row in rows for x in row['fps_seconds']]
    complete=bool(rows) and all(len(row['fps_seconds'])==repeats for row in rows)
    valid=bool(complete and actual_cuda and bool(samples) and np.isfinite(samples).all() and min(samples)>0)
    total=sum(samples);changes=[x for row in rows for x in row['fps_changed_voxels']]
    out=dict(windows=len(rows),repeats=repeats,measured_calls=len(samples),timing_complete=valid,
        mean_six_frame_ms=1000*np.mean(samples) if samples else None,p90_six_frame_ms=1000*np.percentile(samples,90) if samples else None,
        window_fps=len(samples)/total if valid else None,output_frame_fps=6*len(samples)/total if valid else None,
        strict_byte_parity=complete and not any(changes),max_changed_voxels=max(changes,default=0),
        byte_failed_windows=sum(any(row['fps_changed_voxels']) for row in rows),
        fps_label='single-process WARM actual execution; strict byte parity reported separately; no timing sample filtered',
        evaluation_ab={})
    for name in ('legacy_paired_forced_gc','frozen_single_auto_gc'):
        values=[row['evaluation_ab'][name] for row in rows]
        out['evaluation_ab'][name]=dict(seconds_per_window=float(np.mean([v['seconds'] for v in values])) if values else None,
            mean_forced_gc_seconds=float(np.mean([v['forced_gc_seconds'] for v in values])) if values else None)
    if rows:out['evaluation_speedup']=out['evaluation_ab']['legacy_paired_forced_gc']['seconds_per_window']/out['evaluation_ab']['frozen_single_auto_gc']['seconds_per_window']
    out['evaluation_changed_voxels_max']=max((row['evaluation_ab']['changed_voxels'] for row in rows),default=0)
    out['profile_mean_ms']={k:float(1000*np.mean([row['profile'][k] for row in rows if row['profile']]))
        for k in next((row['profile'] for row in rows if row['profile']),{})}
    out['workload_mean']={k:float(np.mean([row['workload'][k] for row in rows]))
        for k in next((row['workload'] for row in rows if 'workload' in row),{})}
    out['peak_allocated_mib']=max((row['peak_allocated_mib'] for row in rows if row.get('peak_allocated_mib') is not None),default=None)
    out['peak_reserved_mib']=max((row['peak_reserved_mib'] for row in rows if row.get('peak_reserved_mib') is not None),default=None)
    out['memory_scope']='allocator peak includes serialized profile; reserved memory may retain earlier untimed diagnostics'
    return out
