"""Finite real-data sparse repair experiment; old teacher is read-only."""
from collections import defaultdict
import copy
import time
import numpy as np
import torch
from real_motion.sparse_evidence_repair import SparseRepairHead, FREE, STATIC, _locate
from real_motion.source_repair_evidence import build_evidence, map_evidence, compose_repair, oracle_probabilities
from real_motion.causal_column_completion import GENERATE, REFINE, KEEP, ADD, REMOVE, compose_dense, actions_from_probabilities
from real_motion.causal_column_sampling import ColumnHistoryIndex
from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.source_evidence_audit import edit_quality
from tools.real_motion import causal_column_common as columns
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.shared_evidence_pilot_common import GATES, sync, moving_support_masks


def probabilities(head, evidence, output, device, *, chunk=4096):
    memory = evidence.memory
    result = []
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        for start in range(0,len(memory),chunk):
            ids = slice(start,start+chunk)
            result.append(head(torch.as_tensor(memory.neighbor_features[ids],device=device),
                torch.as_tensor(memory.keys[ids,0],device=device),torch.as_tensor(memory.keys[ids,1],device=device),
                output['history_source_context'],output['future_transport_queries']).float().sigmoid().cpu().numpy())
    return np.concatenate(result) if result else np.empty((0,6),np.float32)


def old_horizon(teacher, prepared, h, grid, device, *, generation_only=False):
    plan = columns.candidate_plan(prepared,h,grid,teacher.config)
    if generation_only: plan = plan.subset(np.flatnonzero(plan.kind==GENERATE))
    p = columns.predict_probabilities(teacher,prepared,h,plan,grid,device,256)
    return plan,p,actions_from_probabilities(plan,p,GATES)


def fill_generation(baseline, plan, actions):
    """Original GEN proposals, rechecked against STILL-free repaired output."""
    if np.any(plan.kind!=GENERATE) or np.shape(actions)!=plan.base.shape or np.any(actions==REMOVE):
        raise ValueError('generation-only legal ADD/KEEP required')
    output=np.array(baseline,copy=True)
    take=(actions==ADD)&plan.legal[...,ADD]
    flat=plan.flat[take];classes=np.broadcast_to(plan.classes[:,None],plan.base.shape)[take]
    valid=output.reshape(-1)[flat]==FREE
    output.reshape(-1)[flat[valid]]=classes[valid]
    return output


def teacher_predictions(base, plan, actions):
    result = {}
    for name,keep in (
        ('teacher_generation',plan.kind==GENERATE),
        ('teacher_static_ADD',(plan.kind==REFINE)&(plan.actor==STATIC)),
        ('teacher_dynamic_ADD',(plan.kind==REFINE)&(plan.actor>=0)),
        ('teacher_ADD',plan.kind==REFINE),
        ('teacher_REMOVE',plan.kind==REFINE),
        ('teacher_refine',plan.kind==REFINE),
        ('teacher_joint',np.ones(len(plan),bool))):
        a = actions.copy(); a[~keep] = KEEP
        if name.endswith('_ADD'): a[a==REMOVE] = KEEP
        if name=='teacher_REMOVE': a[a==ADD] = KEEP
        result[name] = compose_dense(base,plan,a)
    return result


def evaluate(provider, source, records, teacher, head, *, progress=None, stop_event=None):
    """ONE old probability pass: decomposition + causal oracle + learned head."""
    names = ('teacher_generation','teacher_static_ADD','teacher_dynamic_ADD','teacher_ADD','teacher_REMOVE',
        'teacher_refine','teacher_joint','support_oracle','student_static','student_dynamic','student_refine','student_joint')
    base = Metrics(); metrics = {n:Metrics() for n in names}; quality = {n:defaultdict(int) for n in names}
    scenes = defaultdict(lambda:{n:Metrics() for n in ('baseline',*names)})
    coverage = defaultdict(int); audits=[]; started=time.perf_counter(); stage=defaultdict(float)
    teacher.eval(); head.eval()
    with torch.inference_mode():
      for wi,(record,raw) in enumerate(prefetch_raw_columns(provider,source,records),1):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('evaluation stopped at window boundary')
        tick=time.perf_counter(); output=teacher.motion(record,provider.device)
        prep=provider.prepare_columns(source,record,include_gt=True,raw_window=raw,outputs=output)
        stage['prepare_and_live_motion']+=time.perf_counter()-tick
        tick=time.perf_counter(); evidence=build_evidence(prep,provider.pcfg.grid); mapped=map_evidence(evidence,prep,provider.pcfg.grid)
        p=probabilities(head,evidence,output,provider.device); stage['sparse_evidence_map_and_head']+=time.perf_counter()-tick
        oracle=oracle_probabilities(evidence,mapped,raw['future_gt_occ'])
        support=gt_moving_support_sequence(source.nusc,prep.window.t0_token,prep.window.future_tokens,
            tuple(.5*(h+1) for h in range(6)),grid=provider.pcfg.grid,workers=provider.workers)
        moving=moving_support_masks(support,provider.pcfg.grid.shape_hwd)
        prep.column_history_index=ColumnHistoryIndex(prep,provider.pcfg.grid)
        for ri,h in enumerate(columns.REPORT):
            tick=time.perf_counter(); plan,op,act=old_horizon(teacher.columns,prep,h,provider.pcfg.grid,provider.device)
            stage['teacher_probability']+=time.perf_counter()-tick
            pred=teacher_predictions(prep.baseline[h],plan,act)
            for name,which in (('student_static','static'),('student_dynamic','dynamic'),('student_refine','all')):
                pred[name]=compose_repair(prep.baseline[h],evidence,mapped,p,h,actors=which)
            pred['support_oracle']=compose_repair(prep.baseline[h],evidence,mapped,oracle,h)
            # The UNCHANGED old generation writes last, and only into still-free
            # cells. It cannot overwrite student occupied additions.
            gen_plan=plan.subset(np.flatnonzero(plan.kind==GENERATE)); gen_act=act[plan.kind==GENERATE]
            pred['student_joint']=fill_generation(pred['student_refine'],gen_plan,gen_act)
            gt=raw['future_gt_occ'][h]; scene=scenes[str(record['scene_name'])]
            base.update(ri,prep.baseline[h],gt,moving[h]); scene['baseline'].update(ri,prep.baseline[h],gt,moving[h])
            for name in names:
                metrics[name].update(ri,pred[name],gt,moving[h]); scene[name].update(ri,pred[name],gt,moving[h])
                for k,v in edit_quality(prep.baseline[h],pred[name],gt).items(): quality[name][k]+=v
            for kind in ('static','dynamic'):
                old=pred['teacher_'+kind+'_ADD']; corrected=(old!=prep.baseline[h])&(old==gt)
                recovered=corrected&(pred['support_oracle']==old)
                coverage[kind+'_teacher_correct_ADD_voxels']+=int(corrected.sum())
                coverage[kind+'_teacher_correct_ADD_covered']+=int(recovered.sum())
        audits.append(evidence.audit)
        if progress:progress(dict(event='sparse_evaluation',window=wi,windows=len(records),evidence=evidence.audit))
        print(f'SPARSE_EVAL {wi}/{len(records)} teacher/decomposition/oracle/student',flush=True)
    report=columns.report_states(base,metrics,quality,scenes)
    report.update(windows=len(records),seconds=time.perf_counter()-started,stage_seconds=dict(stage),teacher_ADD_domain_coverage=dict(coverage),
        evidence_audit=audits,scope='selection population; support_oracle uses future GT ONLY for upper-bound diagnostics')
    return report


def sample_pairs(mapped, actor, rng, *, per_window=256):
    """Static/dynamic strata; exact inverse-probability natural-population risk.

    Positives NEVER decide candidates or sampled IDs. Oversampling small source
    strata is corrected rather than quietly changing the foreground prior.
    """
    population=np.argwhere(mapped>=0)
    if not len(population): return population,np.empty(0,np.float32)
    groups=[population[actor[population[:,0]]<0],population[actor[population[:,0]]>=0]]
    groups=[g for g in groups if len(g)]
    chosen=[];weights=[]
    for g in groups:
        n=min(max(1,per_window//len(groups)),len(g)); at=rng.choice(len(g),n,replace=False)
        chosen.append(g[at]);weights.append(np.full(n,len(g)/len(population)/n,np.float32))
    return np.concatenate(chosen),np.concatenate(weights)


def selected_teacher_probability(teacher, prep, evidence, mapped, pairs, grid, device):
    """Teacher runs on selected XY/source queries ONLY, not the full population."""
    result=np.zeros(len(pairs),np.float32); shape=tuple(grid.shape_hwd)
    for h in np.unique(pairs[:,1]):
        take=np.flatnonzero(pairs[:,1]==h); point=pairs[take,0]
        xyz=np.column_stack(np.unravel_index(mapped[point,h],shape))
        plan=columns.candidate_plan(prep,int(h),grid,teacher.config)
        # Five-column full signed integer keys avoid hash/packed-index aliasing.
        row_keys=np.column_stack((plan.actor,plan.classes,plan.xy,np.zeros(len(plan),np.int64)))
        query=np.column_stack((evidence.memory.keys[point,:2],xyz[:,:2],np.zeros(len(point),np.int64)))
        order=np.lexsort(row_keys.T[::-1]); index,valid=_locate(row_keys[order],query)
        if not valid.any():continue
        rows=order[index[valid]]; unique,inv=np.unique(rows,return_inverse=True)
        small=plan.subset(unique)
        p=columns.predict_probabilities(teacher,prep,int(h),small,grid,device,256)
        result[take[valid]]=p[inv,xyz[valid,2],ADD]
    return result


def training_step(provider, source_rows, teacher, head, optimizer, rng, *, kd_weight=.25, live_motion=False):
    """Frozen epoch19 migration; actual joint training is a later experiment.

    No cross-step learned-feature cache, no detached tensors retained in a
    closure. All per-window graphs die after this one optimizer update.
    """
    started=time.perf_counter(); head.train(); optimizer.zero_grad(set_to_none=True)
    losses=[]; ce=[]; kd=[]; motion=[]; positive=sampled=0; stages=defaultdict(float)
    for record,raw in source_rows:
        tick=time.perf_counter()
        with torch.enable_grad() if live_motion else torch.no_grad():
            output=teacher.motion(record,provider.device)
            prep=provider.prepare_columns(None,record,include_gt=True,raw_window=raw,outputs=output)
        lm=output['residual_xy_m'].new_zeros(())
        if live_motion:
            from tools.real_motion.joint_column_common import motion_loss
            lm,_=motion_loss(output,record,provider.device,.8,materialize_stats=False)
        evidence=build_evidence(prep,provider.pcfg.grid); mapped=map_evidence(evidence,prep,provider.pcfg.grid)
        stages['prepare_motion_evidence_map']+=time.perf_counter()-tick
        pairs,weight=sample_pairs(mapped,evidence.memory.keys[:,0],rng)
        if not len(pairs):
            if live_motion and lm.requires_grad:
                (lm/len(source_rows)).backward();losses.append(lm.detach());motion.append(lm.detach())
            continue
        ids,inv=np.unique(pairs[:,0],return_inverse=True)
        labels=oracle_probabilities(evidence,mapped,raw['future_gt_occ'])[pairs[:,0],pairs[:,1]]
        tick=time.perf_counter()
        with torch.no_grad():
            old=selected_teacher_probability(teacher.columns,prep,evidence,mapped,pairs,provider.pcfg.grid,provider.device) if kd_weight else labels
        stages['selected_teacher_KD']+=time.perf_counter()-tick
        tick=time.perf_counter()
        with torch.autocast(device_type=provider.device.type,dtype=torch.bfloat16,enabled=provider.device.type=='cuda'):
            m=evidence.memory
            logits=head(torch.as_tensor(m.neighbor_features[ids],device=provider.device),torch.as_tensor(m.keys[ids,0],device=provider.device),
                torch.as_tensor(m.keys[ids,1],device=provider.device),output['history_source_context'],output['future_transport_queries'])
            logit=logits[torch.as_tensor(inv,device=provider.device),torch.as_tensor(pairs[:,1],device=provider.device)].float()
            w=torch.as_tensor(weight,device=provider.device)
            bce=(torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.as_tensor(labels,device=provider.device),reduction='none')*w).sum()
            distill=(torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.as_tensor(old,device=provider.device),reduction='none')*w).sum()
            loss=bce+kd_weight*distill+lm
        # Backprop immediately, not after collecting four complete graphs.
        (loss/len(source_rows)).backward(); losses.append(loss.detach()); ce.append(bce.detach()); kd.append(distill.detach())
        motion.append(lm.detach())
        stages['head_forward_backward']+=time.perf_counter()-tick
        sampled+=len(pairs);positive+=int(labels.sum())
    if losses:
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite sparse head gradient')
        if live_motion:
            motion_norm=torch.nn.utils.clip_grad_norm_(teacher.transport.parameters(),5.)
            if not torch.isfinite(motion_norm):raise RuntimeError('nonfinite motion gradient')
        optimizer.step()
    sync(provider.device)
    scalar=lambda values:float(torch.stack(values).mean().cpu()) if values else 0.
    return dict(loss=scalar(losses),GT_BCE=scalar(ce),KD_BCE=scalar(kd),motion_loss=scalar(motion),sampled_pairs=sampled,
        sampled_positive=positive,windows=len(source_rows),optimizer_updated=bool(losses),seconds=time.perf_counter()-started,
        stage_seconds=dict(stages),transport_frozen=not live_motion)


def joint_training_speed(provider, teacher, head, rows, *, repeats=2, seed=20261005):
    """Disposable LIVE motion+refine backward/optimizer; no scientific updates.

    Same input groups. Head architectures/samplers/loss differ, so this is a
    runtime experiment, not an exact-resume or convergence equivalence claim.
    The later migration's costly selected-teacher KD is excluded from BOTH.
    """
    from real_motion.column_gpu_sampling import GpuColumnSampler
    from tools.real_motion.shared_evidence_pilot_common import make_optimizer
    from tools.real_motion.joint_column_full_common import train_full_batch
    groups=[];group=[];sources=0
    for row in rows:
        n=len(row[0]['features'])
        if group and (len(group)>=4 or sources+n>128):groups.append(group);group=[];sources=0
        group.append(row);sources+=n
    if group:groups.append(group)
    original_joint,original_model=provider.joint,provider.model;trials=[]
    try:
      for repeat in range(repeats):
        modes=('old_joint','sparse_joint') if repeat%2==0 else ('sparse_joint','old_joint')
        for mode in modes:
            torch.manual_seed(seed);joint=copy.deepcopy(teacher)
            for param in joint.transport.parameters():param.requires_grad_(True)
            joint.train();new=copy.deepcopy(head)
            for param in new.parameters():param.requires_grad_(True)
            provider.joint=joint;provider.model=joint.transport
            optimizer=(make_optimizer(joint) if mode=='old_joint' else torch.optim.AdamW([
                dict(params=joint.transport.parameters(),lr=5e-4,weight_decay=1e-4),dict(params=new.parameters(),lr=3e-4,weight_decay=.01)]))
            rng=np.random.default_rng(seed);sampler=GpuColumnSampler(provider.device,allow_cpu=False) if mode=='old_joint' else None
            seconds=windows=0;stats=[]
            for i,group in enumerate([groups[0],*groups]):
                sync(provider.device);tick=time.perf_counter()
                if mode=='old_joint':
                    stat=train_full_batch(joint,optimizer,provider,None,group,rng,i+1,len(groups)+1,
                        profile=True,optimize_cpu=True,optimize_kernels=True,column_feature_sampler=sampler,sampling_workers=6)
                else:
                    stat=training_step(provider,group,joint,new,optimizer,rng,kd_weight=0,live_motion=True)
                sync(provider.device)
                if i:seconds+=time.perf_counter()-tick;windows+=len(group);stats.append(stat)
            trials.append(dict(mode=mode,repeat=repeat+1,seconds_per_window=seconds/windows,windows=windows,
                stage_seconds_per_window={k:sum(s.get('stage_seconds',{}).get(k,0) for s in stats)/windows
                    for k in {k for s in stats for k in s.get('stage_seconds',{})}},
                transport_frozen=False,teacher_KD=False,saved_scientific_updates=False))
            print(f'SPARSE_JOINT_TRAIN {mode} seconds/window={seconds/windows:.5f}',flush=True)
            del optimizer,joint,new,sampler
    finally:provider.joint=original_joint;provider.model=original_model
    return dict(trials=trials,boundary='warm causal geometry -> LIVE motion + online head proposals/GT + backward/optimizer',
        raw_IO_excluded=True,not_identical_sampling_or_loss=True,
        old_includes_generation_and_REMOVE=True,new_measures_ADD_only_refine=True,
        warning='new joint training omits GEN loss/REMOVE; speed is not complete-method-equivalent until those paths are integrated')


def six_frame_speed(provider, source, records, teacher, head, *, repeats=2, stop_event=None):
    """Matched prepared-input six dense outputs, including fresh prior and motion.

    Old/refine and new/refine separately; FULL old/new both include unchanged
    generation. Evidence build, mapping, transfer and composition are timed.
    No GT, metric/annotations/disk or precomputed predictions in FPS.
    """
    from tools.real_motion.shared_evidence_pilot_common import fresh_prior, causal_template
    from real_motion.column_execution import execution_session
    teacher.eval();head.eval();trials=[]
    teacher.columns.column_inference_optimized=True;teacher.columns.column_async_readback=True
    teacher.columns.column_probability_optimized=False;teacher.columns.column_sampling_workers=provider.workers
    previous=teacher.columns.column_inference_verify_remaining if hasattr(teacher.columns,'column_inference_verify_remaining') else None
    with torch.inference_mode(),execution_session(teacher.columns,graphs=True,reuse=False):
      for wi,(record,raw) in enumerate(prefetch_raw_columns(provider,source,records,include_gt=False),1):
        if raw.get('future_gt_occ') is not None: raise RuntimeError('FPS cannot consume future GT')
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('FPS stopped')
        template=causal_template(raw,record); inputs=runtime._gpu_inputs(record,provider.device)
        # Outside timing: GEN-only batching must preserve the OLD threshold
        # actions, not merely approximate logits. Otherwise refuse the speed
        # result instead of quietly measuring a changed generation branch.
        output=runtime._model_forward(teacher.transport,inputs,provider.device,return_latents=True)
        audited=provider.prepare_columns(source,record,include_gt=False,raw_window=raw,outputs=output)
        for h in range(6):
            full,_,full_act=old_horizon(teacher.columns,audited,h,provider.pcfg.grid,provider.device)
            small,_,small_act=old_horizon(teacher.columns,audited,h,provider.pcfg.grid,provider.device,generation_only=True)
            if not np.array_equal(full_act[full.kind==GENERATE],small_act):
                raise RuntimeError('GEN-only batching changed old generation actions; speed result rejected')
        del output,audited
        def run(mode):
            clocks={};tick=time.perf_counter()
            fresh=fresh_prior(template,provider)
            current_raw={**raw,'_column_causal_preparation':{**raw['_column_causal_preparation'],'prepared_state':fresh.state}}
            output=runtime._model_forward(teacher.transport,inputs,provider.device,return_latents=True)
            prep=provider.prepare_columns(source,record,include_gt=False,raw_window=current_raw,outputs=output)
            clocks['prior_motion_layers']=time.perf_counter()-tick
            result=[];refine=mode.endswith('refine'); old=mode.startswith('old')
            if not old:
                tick=time.perf_counter();e=build_evidence(prep,provider.pcfg.grid);mapped=map_evidence(e,prep,provider.pcfg.grid)
                p=probabilities(head,e,output,provider.device);clocks['evidence_map_head']=time.perf_counter()-tick
            prep.column_history_index=ColumnHistoryIndex(prep,provider.pcfg.grid) if old or not refine else None
            tick=time.perf_counter()
            for h in range(6):
                if old:
                    plan,_,act=old_horizon(teacher.columns,prep,h,provider.pcfg.grid,provider.device)
                    dense=compose_dense(prep.baseline[h],plan,act,enable_generation=not refine)
                else:
                    dense=compose_repair(prep.baseline[h],e,mapped,p,h)
                    if not refine:
                        plan,_,act=old_horizon(teacher.columns,prep,h,provider.pcfg.grid,provider.device,generation_only=True)
                        dense=fill_generation(dense,plan,act)
                result.append(dense)
            clocks['column_generation_and_composition']=time.perf_counter()-tick
            if len(result)!=6 or any(r.shape!=tuple(provider.pcfg.grid.shape_hwd) for r in result):raise RuntimeError('six dense outputs required')
            return result,clocks
        modes=('old_refine','sparse_refine','old_joint','sparse_joint')
        # Warm-up all paths including graphs outside timing; no learned encoding
        # or predicted shape is reused in measured calls. No training RNG draws.
        teacher.columns.column_inference_verify_remaining=0
        for mode in modes:run(mode)
        for repeat in range(repeats):
            for mode in (modes if repeat%2==0 else tuple(reversed(modes))):
                sync(provider.device);tick=time.perf_counter();dense,clocks=run(mode);sync(provider.device)
                elapsed=time.perf_counter()-tick
                trials.append(dict(mode=mode,window=wi,repeat=repeat+1,six_frame_seconds=elapsed,FPS=6/elapsed,stages=clocks))
                print(f'SPARSE_FPS window={wi}/{len(records)} {mode} seconds={elapsed:.4f} FPS={6/elapsed:.2f}',flush=True)
    if previous is not None:teacher.columns.column_inference_verify_remaining=previous
    return dict(trials=trials,boundary='resident V18 inputs + registered causal history -> fresh Strong/KTA prior + live motion + evidence build + six dense outputs',
        exclusions='I/O, source extraction/registration, GT, metrics, graph capture/warmup; NOT raw-sensor end-to-end FPS',
        history_frames=4,future_frames=6,head='trained local_consensus + cached six future queries',
        generation_subset_six_horizon_actions_exact=True,actual_cuda=provider.device.type=='cuda')
