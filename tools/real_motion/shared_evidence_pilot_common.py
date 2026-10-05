"""One finite migration experiment; shared data for probes, speed and training."""
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace
import copy
import time
import numpy as np
import torch
from real_motion.column_device_geometry import DeviceColumnWindow, DevicePlan, targets, sample_training, actions, compose
from real_motion.shared_column_evidence import SharedEvidenceColumns, SharedHistorySession, SharedReadExecution
from real_motion.sparse_column_readout import token_probe
from real_motion.causal_column_model import column_loss
from real_motion.causal_column_completion import compose_dense, actions_from_probabilities
from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.local_training_profile import StageTimer
from real_motion.column_gpu_sampling import GpuColumnSampler
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.source_evidence_audit import edit_quality
from tools.real_motion import causal_column_common as columns
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics, delta
from tools.real_motion.joint_column_full_common import pack_records, MOTION_KEYS, LABEL_KEYS, train_full_batch
from tools.real_motion.joint_column_common import motion_loss

BUDGETS=(24,24,20,20,20,20)
GATES=(.5,.5,.95)


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def causal_template(raw,record):
    evidence=raw.get('_column_causal_preparation',{})
    state=evidence.get('prepared_state')
    if state is None or 'column_backgrounds' not in state:
        raise RuntimeError('complete fixed causal geometry must be prepared before live device forecast')
    if len(state['current'])!=len(record['features']) or [s['class_id'] for s in state['current']]!=record['source_class_id'].tolist():
        raise RuntimeError('causal source identity/order mismatch')
    # Labels/order alone cannot detect a same-class source permutation. Check
    # immutable source centres before bypassing the CPU live renderer.
    from real_motion.source_evidence_audit import transform_points
    world=np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1,3)
    centres=transform_points(world,np.linalg.inv(state['current_pose']))[:,:2]
    if not np.allclose(centres,np.asarray(record['source_centroid_xy_t0_m']),rtol=0,atol=2e-4):
        raise RuntimeError('causal source centre identity/order mismatch')
    return SimpleNamespace(raw=raw,state=state,registrations=evidence['registrations'],
        aligned_history_points=evidence.get('aligned_history_points'),fixed_candidate_geometry=evidence.get('fixed_candidate_geometry'),
        memory=evidence['memory'],footprints=evidence['footprints'])


def tensor_probability(model,window,h,plan,output,*,session=None,batch_size=256,timer=None,execution=None):
    result=[]
    timer=timer or StageTimer(window.device,False)
    finite=torch.ones((),device=window.device,dtype=torch.bool)
    for start in range(0,len(plan),batch_size):
        small=plan.subset(slice(start,start+batch_size))
        sparse=isinstance(model,SharedEvidenceColumns)
        ijk,labels,flags=timer.call('full_Z_inverse_lookup',window.lookup,h,small,sparse=sparse,gpu=True)
        source=timer.call('source_query_gather',window.source_features,h,small,output,gpu=True)
        with torch.autocast(device_type=window.device.type,dtype=torch.bfloat16,enabled=window.device.type=='cuda'):
            if execution is not None:
                batch=dict(base=small.base,fallback=small.fallback,context=small.context,
                    kind=small.kind.byte(),classes=small.classes.byte(),source_features=source)
                if sparse:
                    features,valid=timer.call('shared_history_encoding_gather',session.gather,ijk,gpu=True)
                    batch.update(features=features,valid=valid,flags=flags)
                else:batch=dict(history=labels.reshape(len(small),4,7,7,-1),flags=flags.reshape(len(small),4,7,7,-1),**batch)
                p,good=timer.call('graph_reader',execution.run,batch,small.legal,gpu=True)
                finite=finite&good
                # Graph outputs alias one reusable buffer. Retaining the view
                # would silently turn every earlier chunk into the last chunk.
                p=p.clone()
            elif sparse:
                g,r=timer.call('shared_encode_and_read',model.read,session,ijk,flags,small.base,small.fallback,small.context,small.kind,small.classes,source,gpu=True)
            else:
                n=len(small);z=model.config.z_bins
                g,r=timer.call('old_patch_encode_and_read',model,labels.reshape(n,4,7,7,z),flags.reshape(n,4,7,7,z),small.base,small.fallback,
                    small.context,small.kind,small.classes,source_features=source,validate_source=False,gpu=True)
            if execution is None:p=timer.call('calibrated_probabilities',model.calibrated_probabilities,g,r,small.kind,small.legal,validate=False,gpu=True)
        result.append(p)
    p=torch.cat(result) if result else torch.empty((*plan.base.shape,3),device=window.device)
    if not (finite&torch.isfinite(p).all()):raise RuntimeError('nonfinite sparse action probability')
    return p


def check_geometry(prep,window,*,horizons=range(6),features=True):
    """No timing: check ALL layers, ALL candidates, independent of predicted edits."""
    for h in horizons:
        for name,actual,expected in (('baseline',window.baseline[h],prep.baseline[h]),
                ('owner',window.owners[h],prep.owners[h]),('fallback',window.fallback[h],prep.fallbacks[h])):
            if not np.array_equal(actual.cpu().numpy(),expected):raise RuntimeError(f'device integer gate failed: {name} horizon={h}')
        cpu=columns.candidate_plan(prep,h,window.grid,window.config);gpu=window.candidates(h)
        for field in fields(DevicePlan):
            actual=getattr(gpu,field.name).cpu().numpy();expected=getattr(cpu,field.name)
            ok=np.allclose(actual,expected,rtol=0,atol=3e-7) if field.name=='context' else np.array_equal(actual,expected)
            if not ok:raise RuntimeError(f'device candidate gate failed: {field.name} horizon={h}')
        if features and len(cpu):
            # Bounded chunks, but COMPLETE populations on audited horizons.
            for start in range(0,len(cpu),256):
                small=cpu.subset(slice(start,start+256));d=gpu.subset(slice(start,start+256))
                _,label,flag=window.lookup(h,d,sparse=False)
                reference=columns.sample_column_features(prep,h,small,window.grid,window.config)
                if (not np.array_equal(label.cpu().numpy().reshape(reference['history'].shape),reference['history'])
                        or not np.array_equal(flag.cpu().numpy().reshape(reference['flags'].shape),reference['flags'])):
                    raise RuntimeError(f'device full-Z inverse sampling gate failed horizon={h}')
    return dict(integer_layers_exact=True,complete_candidate_population_exact=True,
        full_Z_labels_membership_exact=features,horizons=len(tuple(horizons)),
        near_floor_points=int(window.rounding_boundary_points))


def moving_support_masks(rows,shape):
    """Unpack the frozen adapter's (mask, instance records, exclusions) rows.

    Reject malformed support rather than broadcasting a sliced/raw array and
    silently changing the Moving metric's population.
    """
    if len(rows)!=6:raise ValueError('Moving support must contain all six horizons')
    masks=[]
    for h,row in enumerate(rows):
        if not isinstance(row,(tuple,list)) or len(row)!=3:
            raise ValueError(f'Moving support row must be (mask, records, exclusions), horizon={h}')
        mask=np.asarray(row[0])
        if mask.dtype!=np.dtype(bool) or mask.shape!=tuple(shape):
            raise ValueError(f'Moving support mask must be boolean with shape {tuple(shape)}, horizon={h}; got {mask.dtype} {mask.shape}')
        masks.append(mask)
    return masks


def evaluate(provider,source,records,teacher,student,*,probe=False,audit=False,batch_size=256,progress=None,stop_event=None):
    teacher.eval();student.eval()
    names=('teacher','probe','student') if probe else ('teacher','student')
    variants=tuple(n+'_'+v for n in names for v in ('generation','refine','joint'))
    base=Metrics();metrics={n:Metrics() for n in variants};quality={n:defaultdict(int) for n in variants}
    scenes=defaultdict(lambda:{n:Metrics() for n in ('baseline',*variants)})
    diagnostics=[];begun=time.perf_counter()
    with torch.inference_mode():
      for wi,(record,raw) in enumerate(prefetch_raw_columns(provider,source,records),1):
        if stop_event is not None and stop_event.is_set():raise InterruptedError('probe/eval stopped at window boundary')
        output=teacher.motion(record,provider.device)
        prep=provider.prepare_columns(source,record,include_gt=True,raw_window=raw,outputs=output)
        window=DeviceColumnWindow(prep,provider.pcfg.grid,student.config,provider.device).render(
            record['anchors_xy_t0_m'],output['residual_xy_m'],output['yaw_delta_rad'])
        if audit:
            diagnostics.append(check_geometry(prep,window,horizons=range(6) if wi<=2 else columns.REPORT,features=wi<=2))
        moving=gt_moving_support_sequence(source.nusc,prep.window.t0_token,prep.window.future_tokens,
            tuple(.5*(h+1) for h in range(6)),grid=provider.pcfg.grid,workers=provider.workers)
        moving=moving_support_masks(moving,provider.pcfg.grid.shape_hwd)
        session=SharedHistorySession(student,window.labels,window.visibility)
        for ri,h in enumerate(columns.REPORT):
            plan=columns.candidate_plan(prep,h,provider.pcfg.grid,teacher.columns.config)
            p=columns.predict_probabilities(teacher.columns,prep,h,plan,provider.pcfg.grid,provider.device,batch_size)
            probabilities={'teacher':p}
            if probe:
                with token_probe(teacher.columns):
                    probabilities['probe']=columns.predict_probabilities(teacher.columns,prep,h,plan,provider.pcfg.grid,provider.device,batch_size)
            dp=window.candidates(h)
            sp=tensor_probability(student,window,h,dp,output,session=session,batch_size=batch_size)
            # Device geometry with the OLD network must preserve finished voxels,
            # not merely allclose logits. Context has its own tight float guard.
            if audit:
                reference_device=tensor_probability(teacher.columns,window,h,dp,output,batch_size=batch_size)
                expected=compose_dense(prep.baseline[h],plan,actions_from_probabilities(plan,p,GATES))
                actual=compose(window.baseline[h],dp,actions(dp,reference_device,GATES)).cpu().numpy()
                if not np.array_equal(expected,actual):raise RuntimeError(f'device teacher voxel gate failed window={wi} horizon={h}')
            gt=prep.raw['future_gt_occ'][h];mask=np.asarray(moving[h],bool);scene=scenes[str(record['scene_name'])]
            base.update(ri,prep.baseline[h],gt,mask);scene['baseline'].update(ri,prep.baseline[h],gt,mask)
            probabilities['student']=sp
            for name,probability in probabilities.items():
                act=actions(dp,probability,GATES) if name=='student' else actions_from_probabilities(plan,probability,GATES)
                for variant,gen,ref in (('generation',True,False),('refine',False,True),('joint',True,True)):
                    key=name+'_'+variant
                    prediction=(compose(window.baseline[h],dp,act,generation=gen,refine=ref).cpu().numpy()
                        if name=='student' else compose_dense(prep.baseline[h],plan,act,enable_generation=gen,enable_refine=ref))
                    metrics[key].update(ri,prediction,gt,mask);scene[key].update(ri,prediction,gt,mask)
                    for k,v in edit_quality(prep.baseline[h],prediction,gt).items():quality[key][k]+=v
        if progress:progress(dict(event='shared_evaluation',window=wi,windows=len(records),memory=session.audit()))
        print(f'SHARED_EVAL {wi}/{len(records)} teacher'+('/36-token-probe' if probe else '')+'/shared',flush=True)
    report=columns.report_states(base,metrics,quality,scenes)
    report.update(windows=len(records),seconds=time.perf_counter()-begun,geometry_audits=diagnostics,
        thresholds=GATES,selection=False)
    return report


def training_step(joint,optimizer,provider,rows,generator,*,frozen=False,teacher=None,kd_weight=.25,profile=False):
    """Same live geometry and gradients; only migration freezes motion.

    Teacher only evaluates selected TRAIN queries. GT labels/sampling are not
    part of history encoding. KD cost is measured separately from joint speed.
    """
    timer=StageTimer(provider.device,profile);optimizer.zero_grad(set_to_none=True)
    joint.train();joint.transport.eval() if frozen else joint.transport.train()
    records=[r for r,_ in rows];sizes=[len(r['features']) for r in records]
    record=timer.call('pack_inputs',pack_records,records,(*MOTION_KEYS,*LABEL_KEYS))
    with torch.no_grad() if frozen else nullcontext():
        output=timer.call('motion_forward',joint.motion,record,provider.device,gpu=True)
    locals=[dict(zip(output,values)) for values in zip(*(v.split(sizes) for v in output.values()))]
    lm=output['residual_xy_m'].sum()*0 if frozen else timer.call('motion_loss',motion_loss,output,record,provider.device,materialize_stats=False,gpu=True)[0]
    terms=[];teacher_terms=[];counts=0;audits=[]
    predictions=[];batches=defaultdict(list)
    for (rec,raw),local in zip(rows,locals):
        template=causal_template(raw,rec)
        window=timer.call('device_pack',DeviceColumnWindow,template,provider.pcfg.grid,joint.columns.config,provider.device)
        timer.call('device_render',window.render,rec['anchors_xy_t0_m'],local['residual_xy_m'],local['yaw_delta_rad'],gpu=True)
        session=(SharedHistorySession(joint.columns,window.labels,window.visibility,
            mode=getattr(joint.columns,'shared_execution_mode','auto'))
            if isinstance(joint.columns,SharedEvidenceColumns) else None)
        # First gather the union for ALL SIX sampled horizons in this window.
        # Tiled training shares overlapping regions across horizons, never a
        # learned cache from a prior update. Deterministic GPU RNG is explicit.
        selected=[]
        for h in range(6):
            plan=timer.call('device_candidates',window.candidates,h,gpu=True)
            y=targets(plan,torch.as_tensor(raw['future_gt_occ'][h],device=provider.device))
            ids,w=timer.call('device_sample',sample_training,plan,y,BUDGETS[h],generator,gpu=True)
            if len(ids):selected.append((h,plan.subset(ids),y[ids],w))
        lookup=[]
        for h,p,y,w in selected:
            ijk,labels,flags=timer.call('inverse_lookup',window.lookup,h,p,sparse=session is not None,gpu=True)
            lookup.append((h,p,y,w,ijk,labels,flags))
        if session is not None and lookup:
            # Decide dense vs tiled on the complete sampled union, not on one
            # small chunk. Nothing is reused after optimizer.step().
            union=torch.cat([v[4] for v in lookup])
            timer.call('shared_encoding',session.gather,union,gpu=True)
        if lookup:
            # One grouped reader for all sampled horizons in a window. Sources,
            # horizon contexts and masks stay per-query; no invalid shared K/V
            # assumption for actor-dependent membership.
            p=DevicePlan(**{f.name:torch.cat([getattr(row[1],f.name) for row in lookup]) for f in fields(DevicePlan)})
            ijk=torch.cat([row[4] for row in lookup]);labels=torch.cat([row[5] for row in lookup]);flags=torch.cat([row[6] for row in lookup])
            source=torch.cat([window.source_features(h,small,local) for h,small,*_ in lookup])
            y=torch.cat([row[2] for row in lookup]);w=torch.cat([row[3] for row in lookup])
            with torch.autocast(device_type=provider.device.type,dtype=torch.bfloat16,enabled=provider.device.type=='cuda'):
                if session is not None:
                    g,r=timer.call('column_readout',joint.columns.read,session,ijk,flags,p.base,p.fallback,p.context,p.kind,p.classes,source,gpu=True)
                else:
                    g,r=timer.call('column_readout',joint.columns,labels.reshape(len(p),4,7,7,-1),flags.reshape(len(p),4,7,7,-1),
                        p.base,p.fallback,p.context,p.kind,p.classes,source_features=source,validate_source=False,gpu=True)
            predictions.append((g,r));counts+=len(p)
            for k,v in (('kind',p.kind),('legal',p.legal),('target',y),('weight',w)):batches[k].append(v)
            if teacher is not None and kd_weight:
                def teacher_prediction():
                    with torch.no_grad(),torch.autocast(device_type=provider.device.type,dtype=torch.bfloat16,enabled=provider.device.type=='cuda'):
                        full=[window.lookup(h,small,sparse=False)[1:] for h,small,*_ in lookup]
                        lab=torch.cat([v[0] for v in full]).reshape(len(p),4,7,7,-1)
                        bit=torch.cat([v[1] for v in full]).reshape_as(lab)
                        tg,tr=teacher(lab,bit,p.base,p.fallback,p.context,p.kind,p.classes,source_features=source,validate_source=False)
                        return teacher.calibrated_probabilities(tg,tr,p.kind,p.legal,validate=False)
                tp=timer.call('TRAIN_selected_teacher',teacher_prediction,gpu=True)
                sp=joint.columns.calibrated_probabilities(g,r,p.kind,p.legal,validate=False)
                mask=p.legal[...,1:].any(-1).float()*w[:,None]
                kl=(tp*(tp.clamp_min(1e-7).log()-sp.clamp_min(1e-7).log())).sum(-1)
                teacher_terms.append(((kl*mask).sum(),mask.sum()))
        if session is not None:audits.append(session.audit())
    if predictions:
        g=torch.cat([p[0] for p in predictions]);r=torch.cat([p[1] for p in predictions])
        b={k:torch.cat(v) for k,v in batches.items()}
        lc,_=timer.call('column_loss',column_loss,joint.columns,g,r,b['kind'],b['legal'],b['target'],b['weight'],materialize_stats=False,gpu=True)
    else:lc=lm*0
    kd=(sum(v for v,_ in teacher_terms)/sum(v for _,v in teacher_terms).clamp_min(1)) if teacher_terms else lm*0
    loss=lm+lc+kd_weight*kd
    if not torch.isfinite(loss):raise RuntimeError('nonfinite shared migration objective before update')
    updated=bool(loss.requires_grad)
    if updated:
        timer.call('backward',loss.backward,gpu=True)
        if not frozen:
            norm=timer.call('clip_motion',torch.nn.utils.clip_grad_norm_,joint.transport.parameters(),5.,gpu=True)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite motion gradient before update')
        norm=timer.call('clip_columns',torch.nn.utils.clip_grad_norm_,joint.columns.parameters(),1.,gpu=True)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite column gradient before update')
        timer.call('optimizer',optimizer.step,gpu=True)
    profile_result=timer.finish()
    scalar=torch.stack((loss.detach().float(),lm.detach().float(),lc.detach().float(),kd.detach().float())).cpu().tolist()
    if not np.isfinite(scalar).all():raise RuntimeError('nonfinite shared migration objective')
    return dict(loss=scalar[0],motion_loss=scalar[1],column_loss=scalar[2],kd_loss=scalar[3],
        windows=len(rows),sources=sum(sizes),sampled_columns=counts,optimizer_updated=updated,
        frozen_transport=frozen,teacher_sampled_queries_only=teacher is not None,memory=audits,**profile_result)


def make_optimizer(joint,*,frozen=False):
    for p in joint.transport.parameters():p.requires_grad_(not frozen)
    for p in joint.columns.parameters():p.requires_grad_(True)
    groups=[dict(params=joint.columns.parameters(),lr=3e-4,initial_lr=3e-4,weight_decay=.01)]
    if not frozen:groups.insert(0,dict(params=joint.transport.parameters(),lr=5e-4,initial_lr=5e-4,weight_decay=1e-4))
    return torch.optim.AdamW(groups)


def training_speed(teacher,student,provider,rows,*,repeats=2,seed=20261005):
    """Real forward/backward/optimizer, no saved scientific updates.

    Same records, batch<=4, sources<=128. Immutable preparation is staged once;
    raw loading is outside WARM compute timing, explicitly reported elsewhere.
    """
    groups=[];group=[];sources=0
    for row in rows:
        count=len(row[0]['features'])
        if group and (len(group)>=4 or sources+count>128):groups.append(group);group=[];sources=0
        group.append(row);sources+=count
    if group:groups.append(group)
    if not groups:raise ValueError('speed requires training windows')
    trials=[]
    modes=('current_joint','device_geometry_joint','shared_auto_joint','shared_dense_joint','shared_tiles_joint')
    previous_joint=provider.joint;previous_model=provider.model
    try:
      for repeat in range(repeats):
        for mode in (modes if repeat%2==0 else tuple(reversed(modes))):
            torch.manual_seed(seed);joint=copy.deepcopy(teacher)
            if mode.startswith('shared'):joint.columns=copy.deepcopy(student)
            optimizer=make_optimizer(joint);provider.joint=joint;provider.model=joint.transport
            old_sampler=GpuColumnSampler(provider.device,allow_cpu=provider.device.type!='cuda') if mode=='current_joint' else None
            generator=torch.Generator(device=provider.device).manual_seed(seed)
            rng=np.random.default_rng(seed)
            # Same one-batch warmup for each fresh model/optimizer.
            measured=[];total=0.;windows=0
            for i,group in enumerate([groups[0],*groups]):
                sync(provider.device);tick=time.perf_counter()
                if mode=='current_joint':
                    stat=train_full_batch(joint,optimizer,provider,None,group,rng,i+1,len(groups)+1,
                        profile=True,optimize_cpu=True,optimize_kernels=True,column_feature_sampler=old_sampler,sampling_workers=6)
                else:
                    # Exact same architecture for dense/tile comparison. Only
                    # per-forward execution strategy changes; no feature cache
                    # is reused across updates or between timed trials.
                    if mode.startswith('shared'):joint.columns.shared_execution_mode=mode.split('_')[1]
                    from tools.real_motion.joint_column_full_common import set_lr
                    set_lr(optimizer,i,len(groups)+1)
                    stat=training_step(joint,optimizer,provider,group,generator,profile=True)
                sync(provider.device);duration=time.perf_counter()-tick
                if i:total+=duration;windows+=len(group);measured.append(stat)
            trials.append(dict(mode=mode,repeat=repeat+1,seconds_per_window=total/windows,windows=windows,
                stage_seconds={k:sum(s.get('host_stage_seconds',{}).get(k,0) for s in measured)/windows
                    for k in {k for s in measured for k in s.get('host_stage_seconds',{})}},
                backward_included=True,transport_frozen=False,distillation_included=False,
                cuda_stream_timings_NOT_active_utilization=[s.get('cuda_stream_stage_seconds',{}) for s in measured]))
            print(f'SHARED_TRAIN_SPEED {mode} repeat={repeat+1} seconds/window={total/windows:.4f}',flush=True)
            del joint,optimizer
    finally:provider.joint=previous_joint;provider.model=previous_model
    return dict(trials=trials,boundary='warm immutable geometry -> live joint forward/geometry/sampling/backward/optimizer',
        no_saved_updates=True,cold_preparation_excluded=True,
        current_baseline_feature_backend='existing_verified_GPU_full_patch_sampler',
        sampling_comparison='same populations/budgets/importance; old NumPy vs new device RNG, not identical sampled query IDs')


def fresh_prior(prepared,provider):
    """Charge the existing Strong/KTA rebuild in six-frame generation FPS.

    Do not pretend the cached six-frame backgrounds are free deployment input.
    Current source extraction/registration and resident learned INPUT tensors
    remain explicit preparation outside this prepared-input boundary.
    """
    state={**prepared.state};state.pop('column_backgrounds',None)
    anchor,baseline=runtime._strong_all_horizons(state['current_sem'],state['current_pose'],
        state['future_poses'],state['current'],state['velocities'],state['source_world_points'],
        frame_dt_s=float(provider.pcfg.frame_dt_s),grid=provider.pcfg.grid,cfg=provider.strong,runtime_device=provider.device)
    from real_motion.runtime_fastpath import compose_component_replacements_fast_exact
    state['column_backgrounds']=[compose_component_replacements_fast_exact(a,b,[],dynamic_class_ids=columns.DYN,
        free_label=17,grid=provider.pcfg.grid,precomputed_clear_flat_indices=runtime.baseline_clear_flat_indices(b,grid=provider.pcfg.grid))
        for a,b in zip(anchor,baseline)]
    return SimpleNamespace(**{**vars(prepared),'state':state})


def fps_speed(provider,source,records,teacher,student,*,repeats=2,batch_size=256,stop_event=None):
    from real_motion.column_execution import execution_session
    from tools.real_motion.joint_execution_speed import generation_six
    trials=[];geometry_verified=0;profiles={}
    teacher.eval();student.eval();teacher.columns.column_inference_optimized=True
    teacher.columns.column_async_readback=True;teacher.columns.column_probability_optimized=False
    teacher.columns.column_sampling_workers=provider.workers
    with torch.inference_mode(),execution_session(teacher.columns,graphs=True,reuse=False) as old_execution:
      shared_execution=SharedReadExecution(student)
      for wi,(record,raw) in enumerate(prefetch_raw_columns(provider,source,records,include_gt=False),1):
        if raw.get('future_gt_occ') is not None:raise RuntimeError('FPS must NEVER load future GT')
        if stop_event is not None and stop_event.is_set():raise InterruptedError('FPS interrupted')
        template=causal_template(raw,record);gi=runtime._gpu_inputs(record,provider.device)
        def run(mode,timer=None):
            timer=timer or StageTimer(provider.device,False)
            if mode=='current_graph':return timer.call('current_whole_six_frame',generation_six,provider,source,record,raw,teacher.columns,GATES,
                    batch_size,transport_inputs=gi)[0]
            prepared=timer.call('fresh_Strong_KTA_prior',fresh_prior,template,provider)
            output=timer.call('motion_forward',runtime._model_forward,teacher.transport,gi,provider.device,return_latents=True,gpu=True)
            window=timer.call('fixed_input_pack',DeviceColumnWindow,prepared,provider.pcfg.grid,student.config,provider.device)
            timer.call('batched_SE2_ownership_fallback',window.render,
                record['anchors_xy_t0_m'],output['residual_xy_m'],output['yaw_delta_rad'],gpu=True)
            model=teacher.columns if mode=='device_geometry' else student
            session=SharedHistorySession(student,window.labels,window.visibility) if model is student else None
            result=[]
            for h in range(6):
                plan=timer.call('sparse_candidates',window.candidates,h,gpu=True)
                p=tensor_probability(model,window,h,plan,output,session=session,batch_size=batch_size,timer=timer,
                    execution=old_execution if model is teacher.columns else shared_execution)
                act=timer.call('threshold_actions',actions,plan,p,GATES,gpu=True)
                result.append(timer.call('dense_composition',compose,window.baseline[h],plan,act,gpu=True))
            return torch.stack(result)
        # Compile/capture/memory-allocation warmup outside timing; fresh learned
        # memory is still rebuilt INSIDE every generation, no output cache.
        expected=run('current_graph');actual=run('device_geometry')
        if not np.array_equal(np.stack(expected),actual.cpu().numpy()):raise RuntimeError('six-frame device geometry voxel gate failed')
        geometry_verified+=1;run('shared_auto')
        if wi==1:
            # Independently verify graph probabilities, including every chunk
            # and the short tail. Never infer correctness from mIoU alone.
            w=DeviceColumnWindow(template,provider.pcfg.grid,student.config,provider.device)
            output=runtime._model_forward(teacher.transport,gi,provider.device,return_latents=True)
            w.render(record['anchors_xy_t0_m'],output['residual_xy_m'],output['yaw_delta_rad'])
            for h in range(6):
                dp=w.candidates(h);ss=SharedHistorySession(student,w.labels,w.visibility)
                eager=tensor_probability(student,w,h,dp,output,session=ss,batch_size=batch_size)
                ss=SharedHistorySession(student,w.labels,w.visibility)
                captured=tensor_probability(student,w,h,dp,output,session=ss,batch_size=batch_size,execution=shared_execution)
                if not torch.equal(eager,captured):raise RuntimeError('shared reader graph probability byte gate failed')
        for repeat in range(repeats):
            modes=('current_graph','device_geometry','shared_auto')
            for mode in (modes if repeat%2==0 else tuple(reversed(modes))):
                sync(provider.device);tick=time.perf_counter();result=run(mode);sync(provider.device)
                seconds=time.perf_counter()-tick
                if len(result)!=6:raise RuntimeError('FPS requires SIX completed dense frames')
                # Both boundaries include final output transfer to CPU, avoiding
                # credit for keeping only the new method's output on device.
                transfer=time.perf_counter()
                if isinstance(result,torch.Tensor):result.cpu().numpy();sync(provider.device)
                seconds+=time.perf_counter()-transfer
                trials.append(dict(mode=mode,window=wi,repeat=repeat+1,six_frame_seconds=seconds,fps=6/seconds))
                print(f'SHARED_FPS window={wi}/{len(records)} {mode} six_seconds={seconds:.4f} FPS={6/seconds:.2f}',flush=True)
        if wi==1:
            # Separate diagnostic pass: event allocation/profile synchronization
            # never contaminates the repeated steady-state FPS measurements.
            for mode in ('current_graph','device_geometry','shared_auto'):
                timer=StageTimer(provider.device,True);run(mode,timer);profiles[mode]=timer.finish()
    return dict(trials=trials,separate_first_window_profile=profiles,device_geometry_six_frame_voxel_checks=geometry_verified,
        old_execution=old_execution.audit(),shared_execution=shared_execution.audit(),shared_graph_probability_bytes_exact=True,
        boundary='resident source input + registered causal history -> FRESH Strong/KTA prior + live motion + all SIX column forecasts + SIX dense outputs + final D2H',
        excludes='disk, GT, metrics, source input tensor construction, causal source extraction/registration',
        fresh_history_encoding_inside_timer=True,all_future_priors_device_native=False,
        note='existing Strong prior rebuild still charged, not claimed fully GPU-resident; shared model untrained during speed phase')
