"""Full-TRAIN surface-consistent CCR validation, BEFORE clean joint training.

Reuse immutable v2 TRAIN/VAL geometry caches and the existing fast pipeline.
Keep motion/dynamic CCR fixed, replace the static conditional readout inside CCR.
Weighted TRAIN ADD, raw sigmoid@0.5, REMOVE off; no KD/AE/threshold selection.
"""
from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import time

import numpy as np
import torch

from real_motion.canonical_causal_repair import CanonicalRepairHead, FEATURE_DIM
from real_motion.ccr_frozen_b import frozen_b_probabilities
from real_motion.final_dataflow import prepare_history, forecast_six
from real_motion.surface_canonical_repair import (
    SurfaceAtlas, SurfaceCanonicalRepairHead, augment_evidence, augment_projection,
    PROTOCOL as MODEL_PROTOCOL, SURFACE_DIM, PHASE_DIM, NEIGHBORS, RADIUS,
)
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import ccr_screen_common as base
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
from tools.real_motion.point_ccr_v18_fps_common import load_point_head, select_population

PROTOCOL = 'p0_f9_surface_consistent_ccr_full_validation_v1'
SPEED_FULL_MONITOR_POPULATION = True  # select the fixed 18+2 arms from ALL dev64 keys
SPEED_NEEDS_QUALITY_REPORT = True  # completed evaluation survives a resume before FPS
SNAPSHOT_WARM_START_HEAD = True  # never depend on a mutable server last.pt during the run


def add_args(parser):
    base.add_args(parser)
    parser.add_argument('--surface-reference-execution', action='store_true',
                        help='diagnostic: original duplicate projection and single-thread tree queries')
    parser.add_argument('--surface-query-workers', type=int, default=4)
    parser.set_defaults(train_fraction=1., epochs=3, lr=3e-4, cpu_workers=10,
                        ccr_cpu_execution='native_parallel', ccr_fast_train=True,
                        ccr_history_cache_mode='require', fps_windows=20,
                        speed_repeats=3, descriptor_disk_mib=0)


def make_head(teacher, device):
    head=SurfaceCanonicalRepairHead(source_dim=teacher.columns.source_dim).to(device)
    head.freeze_validation()
    return head


def model_contract(head):
    return dict(mode='surface_consistent_CCR', protocol=MODEL_PROTOCOL,
                source_dim=head.source_dim, width=head.width,
                surface_dim=SURFACE_DIM, projection_dim=PHASE_DIM,
                static_readout_replaced=True, posthoc_logit_rectifier=False)


def warm_start_head(head,path,*,teacher_sha256,config_fingerprint,dev_manifest_fingerprint=None):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    original=load_point_head(saved,teacher_sha256=teacher_sha256,
                             config_fingerprint=config_fingerprint,source_dim=head.source_dim,
                             device=next(head.parameters()).device,
                             allow_completed_epoch_boundary=True)
    c=saved['contract']
    if dev_manifest_fingerprint is not None and c.get('dev_manifest_fingerprint') != dev_manifest_fingerprint:
        raise RuntimeError('Frozen B/validation manifest mismatch')
    prior=saved.get('reports',{}).get('train_prior')
    if not prior or not np.array_equal(np.asarray(prior.get('positive_weights')),
                                       original.positive_weight.cpu().numpy()):
        raise RuntimeError('Frozen B lacks matching persisted TRAIN positive weights')
    if not all(bool(torch.isfinite(v).all()) for v in original.state_dict().values()):
        raise RuntimeError('nonfinite Frozen B checkpoint')
    head.initialize_from(original)
    head.freeze_validation()
    return dict(checkpoint_sha256=sha256(path),source_epoch=int(saved.get('epoch',0)),
                source_updates=int(saved.get('updates',0)),
                positive_weights=original.positive_weight.cpu().tolist(),
                train_prior={**prior,'reused_for_surface_validation':True,
                             'note':'TRAIN-only weights fixed; raw decision; no DEV fit'})


def contract_extra(args,root):
    if args.train_fraction != 1. or not args.warm_start_head:
        raise ValueError('surface validation requires full TRAIN and an explicit Frozen B --warm-start-head')
    if args.ccr_add_only_natural_bce:
        raise ValueError('surface validation keeps TRAIN positive-weighted ADD; natural BCE is a different experiment')
    if not args.ccr_fast_train or args.ccr_history_cache_mode != 'require' or not args.ccr_history_cache:
        raise ValueError('reuse the completed TRAIN geometry cache with --ccr-fast-train / mode=require')
    if args.fps_windows != 20 or args.speed_repeats != 3:
        raise ValueError('official paired FPS requires the frozen 20-window population x3')
    result=base.contract_extra(args,root)
    files=('real_motion/surface_canonical_repair.py','real_motion/ccr_frozen_b.py','tools/real_motion/surface_ccr_screen_common.py',
           'tools/real_motion/train_p0_f9_surface_ccr.py')
    result.update(objective='importance_corrected_static_weighted_ADD_only',
                  thresholds=dict(CCR_ADD=.5,CCR_REMOVE=None,old_Local=(.5,.5,None)),
                  remove_loss_weight=0.,static_only_training=True,
                  surface_descriptor=dict(neighbors=NEIGHBORS,radius_voxels=RADIUS,z_distance_scale=2.),
                  surface_implementation=stable_json_fingerprint({p:sha256(root/p) for p in files}),
                  reference_rule='Frozen B raw weighted ADD0.5 REMOVEoff',
                  model_initialization='copied_shared_CCR_and_static_readout_zero_new_geometry_projections')
    return result


def _atlas(provider,prep,evidence=None):
    atlas=prep.raw.get('_surface_atlas_live')
    if atlas is None:
        compact=(prep.raw.get('_column_causal_preparation') or {}).get('_ccr_compact_support')
        if compact is not None:
            atlas=SurfaceAtlas.from_compact(compact,prep,provider.pcfg.grid)
        else:
            complete=evidence if evidence is not None else base._build_inputs_base(provider,prep)
            atlas=SurfaceAtlas(complete.world,complete.classes,complete.presence,
                               complete.actor,prep.state['current_pose'],provider.pcfg.grid)
        # Ephemeral window state only. Do not mutate/cache compact artifacts or
        # put learned activations, labels, sampled IDs, or future phases here.
        prep.raw['_surface_atlas_live']=atlas
    atlas.query_workers=getattr(provider,'surface_query_workers',1)
    return atlas


def setup(provider,args):
    base.setup(provider,args)
    provider.ccr_add_only_natural_bce=True  # selects ADD-only plumbing; loss uses head.add_only_weighted
    provider.ccr_old_gates=(.5,.5,None)
    provider.surface_fps_cpu_workers=int(args.ccr_cpu_workers)
    provider.surface_query_workers=(1 if getattr(args,'surface_reference_execution',False)
                                   else min(8,max(1,int(getattr(args,'surface_query_workers',4)))))
    if not getattr(args,'surface_reference_execution',False):
        from real_motion.surface_projection_execution import SurfaceMapExecution
        provider.ccr_execution=SurfaceMapExecution(provider.ccr_execution)
    provider.ccr_augment_evidence=lambda evidence,prep: augment_evidence(evidence,_atlas(provider,prep,evidence))
    provider.ccr_augment_plan=lambda evidence,plan,prep: augment_projection(
        evidence,plan,prep.state['current_pose'],prep.state['world_to_future'],provider.pcfg.grid)
    def sample(evidence,plan,prep):
        evidence=provider.ccr_augment_evidence(evidence,prep)
        return evidence,provider.ccr_augment_plan(evidence,plan,prep)
    provider.ccr_augment_sample=sample


def _reference(provider,head):
    if getattr(provider,'surface_reference_head',None) is None:
        original=CanonicalRepairHead(head.source_dim,head.width).to(provider.device)
        state=head.state_dict()
        original.load_state_dict({k:state[k] for k in original.state_dict()},strict=True)
        original.eval().requires_grad_(False)
        provider.surface_reference_head=original
    return provider.surface_reference_head


def calibrate_train(provider,source,records,teacher,head,**kwargs):
    raise RuntimeError('surface validation must reuse persisted Frozen B TRAIN weights, not recalibrate')


prepare_frozen_superbatch=base.prepare_frozen_superbatch
train_step=base.train_step
close=base.close


def _base_inputs(evidence,plan):
    return replace(evidence,features=evidence.features[:,:FEATURE_DIM]),replace(plan,context=plan.context[...,:8])


@torch.no_grad()
def evaluate(provider,source,records,teacher,head,**kwargs):
    reference=_reference(provider,head)
    def reference_probabilities(evidence,plan,output):
        plain,original=_base_inputs(evidence,plan)
        return frozen_b_probabilities(reference,plain,original,output,provider.device)
    provider.ccr_reference_probabilities=reference_probabilities
    report=base.evaluate(provider,source,records,teacher,head,**kwargs)
    provider.surface_last_evaluation=report
    # Metrics are from the same population and integer accumulation, not scalar
    # subtraction between overlapping subsets. Dynamic scores are separately
    # checked bytewise in the official six-frame execution below.
    report['decision_rule']='raw weighted ADD0.5 / REMOVEoff; no halo policy change'
    variants=report['variants']
    report['surface_class_delta_vs_B']={h:{cls:(
        variants['joint']['metrics']['per_horizon'][h]['semantic_per_class'][cls]-
        variants['frozen_B']['metrics']['per_horizon'][h]['semantic_per_class'][cls])
        for cls in ('11','13')} for h in ('1.0','2.0','3.0')}
    return report


@torch.no_grad()
def six_frame_speed(provider,source,records,teacher,head,*,repeats=3,stop_event=None,quality_report=None):
    if repeats != 3 or len(records) < 20:
        raise ValueError('frozen FPS population requires 20 eligible windows and three repeats')
    selected,population=select_population(records,tuple((str(r['scene_name']),str(r['t0_token'])) for r in records),
                                          windows=20,stress_windows=2)
    reference=_reference(provider,head)
    if quality_report is None:
        quality_report=getattr(provider,'surface_last_evaluation',None)
    if not quality_report or 'frozen_B' not in quality_report.get('variants',{}):
        raise RuntimeError('paired Frozen B quality report required; resume must pass the persisted evaluation')
    flags=[p.requires_grad for p in head.parameters()]
    training=head.training
    head.eval().requires_grad_(False)
    teacher.eval().requires_grad_(False)
    trials=[];prepare_seconds=0.;descriptor_seconds=0.;parity=0
    device=provider.device
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    try:
        with ThreadPoolExecutor(max_workers=getattr(provider,'surface_fps_cpu_workers',4)) as pool:
            for wi,record in enumerate(selected,1):
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError('surface paired FPS interrupted; completed training checkpoint preserved')
                tick=time.perf_counter()
                history=prepare_history(provider,source,record,kernels=base.execution_kernels(provider),executor=pool)
                prepare_seconds+=time.perf_counter()-tick
                plain=history.canonical_evidence
                tick=time.perf_counter()
                atlas=SurfaceAtlas(plain.world,plain.classes,plain.presence,plain.actor,history.current_pose,provider.pcfg.grid)
                enriched=augment_evidence(plain,atlas)
                descriptor_seconds+=time.perf_counter()-tick
                current=history.current_pose
                matrices=[np.linalg.inv(p) for p in history.future_poses]
                def probability(model,evidence,plan,output,where):
                    if model is head:
                        plan=augment_projection(evidence,plan,current,matrices,provider.pcfg.grid)
                        return frozen_b_probabilities(model,evidence,plan,output,where)
                    return frozen_b_probabilities(model,evidence,plan,output,where)
                def run(model,timed):
                    history.canonical_evidence=enriched if model is head else plain
                    initial_memory=None
                    if timed and device.type=='cuda':
                        sync();initial_memory=torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    sync();start=time.perf_counter()
                    out=forecast_six(history,provider,teacher.transport,model,probability,
                                     kernels=base.execution_kernels(provider),executor=pool)
                    sync();elapsed=time.perf_counter()-start
                    if initial_memory is not None:
                        peak=torch.cuda.max_memory_allocated(device)
                        out['memory_mib']=dict(peak_allocated=peak/2**20,
                            incremental_peak=(peak-initial_memory)/2**20,
                            peak_reserved=torch.cuda.max_memory_reserved(device)/2**20)
                    return out,elapsed
                expected={}
                for name,model in (('frozen_B',reference),('surface_CCR',head)):
                    expected[name],_=run(model,False)
                dyn=plain.actor>=0
                if not np.array_equal(expected['frozen_B']['probability'][dyn],expected['surface_CCR']['probability'][dyn]):
                    raise RuntimeError('surface validation changed frozen dynamic CCR probability bytes')
                parity+=1
                for repeat in range(repeats):
                    arms=(('frozen_B',reference),('surface_CCR',head))
                    if repeat%2:arms=arms[::-1]
                    for name,model in arms:
                        out,seconds=run(model,True)
                        if not np.array_equal(out['probability'],expected[name]['probability']) or any(
                                not np.array_equal(a,b) for a,b in zip(out['dense'],expected[name]['dense'])):
                            raise RuntimeError('surface six-frame repeated probability/dense outputs changed')
                        trials.append(dict(mode=name,window=wi,repeat=repeat+1,seconds=seconds,
                                           stages_seconds=out['stages_seconds'],memory_mib=out.get('memory_mib')))
                print(f'SURFACE_FPS {wi}/20 dynamic_probability_bytes=PASS',flush=True)
    finally:
        head.train(training)
        for p,flag in zip(head.parameters(),flags):p.requires_grad_(flag)
    means={name:float(np.mean([r['seconds'] for r in trials if r['mode']==name]))
           for name in ('frozen_B','surface_CCR')}
    return dict(trials=trials,population=population,windows=20,repeats=3,
                six_frame_mean_seconds=means,six_frame_amortized_FPS={k:6/v for k,v in means.items()},
                p90_six_ms={k:1000*float(np.percentile([r['seconds'] for r in trials if r['mode']==k],90)) for k in means},
                speedup=means['frozen_B']/means['surface_CCR'],dynamic_byte_parity_windows=parity,
                boundary='CausalHistoryState -> fresh Strong/KTA + live motion + learned CCR + live projection conditioning + SIX dense outputs',
                excludes='history-only deterministic representation / I/O / GT / metrics / warmup / parity hashing',
                history_prepare_seconds=prepare_seconds,surface_descriptor_prepare_seconds=descriptor_seconds,
                history_prepare_seconds_per_window=prepare_seconds/20,
                surface_descriptor_prepare_seconds_per_window=descriptor_seconds/20,
                memory_mib={k:{field:max(r['memory_mib'][field] for r in trials if r['mode']==k)
                               for field in ('peak_allocated','incremental_peak','peak_reserved')}
                            for k in means} if device.type=='cuda' else None,
                persistent_val_cache_used=False,
                quality_reference=quality_report['variants']['frozen_B']['metrics'])


def gate(new,old,speed):
    reference=speed['quality_reference']
    return dict(mIoU_improves_Frozen_B=new['mIoU']>reference['mIoU'],
                MovingMicro_not_below_Frozen_B=new['MovingMicro']>=reference['MovingMicro']-1e-8,
                all_horizons_Moving_not_below_Frozen_B=all(
                    new['per_horizon'][h]['MovingMicro']>=reference['per_horizon'][h]['MovingMicro']-1e-8
                    for h in ('1.0','2.0','3.0')),
                dynamic_bytes_20_windows=speed['dynamic_byte_parity_windows']==20,
                Dense_Forecast_FPS_ge_40=speed['six_frame_amortized_FPS']['surface_CCR']>=40.)


def brief(result):
    lines=['===== SURFACE-CONSISTENT CCR / FULL TRAIN VALIDATION =====',
           'status: '+result['status'],'protocol: '+PROTOCOL,
           '4 histories -> 6 futures; motion/dynamic CCR FROZEN; static conditional readout replaced INSIDE CCR.',
           'Complete TRAIN20430; three full passes by default; weighted ADD/raw0.5/REMOVEoff; no KD/AE/threshold sweep.',
           'TRAIN/VAL geometry reused READ ONLY; learned features and future phase are never persisted.']
    if result.get('training_population'):lines.append('TRAIN '+json.dumps(result['training_population']))
    reports=result.get('reports',{})
    for row in reports.get('epochs',[]):
        variants=row['evaluation']['variants'];new=variants['joint']['metrics'];ref=variants['frozen_B']['metrics']
        lines.append(f"epoch={row['epoch']} update={row['update']} mIoU={new['mIoU']:.6f} "
                     f"vs_B={new['mIoU']-ref['mIoU']:+.6f} Moving={new['MovingMicro']:.6f}")
    final=reports.get('final_dev512')
    if final:
        lines.append('===== FINAL DEV512 (development, not independent test) =====')
        ref=final['variants']['frozen_B']['metrics']
        for name,item in final['variants'].items():
            m=item['metrics']
            lines.append(f"{name}: IoU={m['IoU']:.6f} mIoU={m['mIoU']:.6f} MovingMicro={m['MovingMicro']:.6f} "
                         f"dMiOU_vs_B={m['mIoU']-ref['mIoU']:+.6f} dMoving_vs_B={m['MovingMicro']-ref['MovingMicro']:+.6f}")
        if 'surface_class_delta_vs_B' in final:
            lines.append('ROAD11_SIDEWALK13_dIoU_pp '+json.dumps(final['surface_class_delta_vs_B']))
    for key in ('training','gate'):
        if key in result:lines.append(key.upper()+' '+json.dumps(result[key]))
    if reports.get('speed'):
        speed=reports['speed'];lines.append('FORMAL_FPS '+json.dumps(speed['six_frame_amortized_FPS']))
        lines.append('SIX_LATENCY '+json.dumps(speed['six_frame_mean_seconds']))
        lines.append('boundary: '+speed['boundary'])
    lines+=['checkpoint: '+result.get('checkpoint','not saved yet'),
            'route: '+result.get('route','in_progress'),
            'Validation only: no automatic clean-joint training, promotion, full4369 rerun, or old checkpoint writes.']
    if result.get('error'):lines.append('error: '+result['error'])
    return '\n'.join(lines)+'\n'
