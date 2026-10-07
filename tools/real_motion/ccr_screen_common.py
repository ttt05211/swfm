"""Point CCR server screen: causal Monte Carlo, GT-only, frozen transport.

No distillation, learned-feature cache, GT candidate selection or dev tuning.
The old Local frontier GEN support is NOT preserved by this representation.
"""
from collections import defaultdict
from contextlib import nullcontext, contextmanager
from pathlib import Path
import copy
import json
import time

import numpy as np
import torch

from real_motion.canonical_causal_repair import (
    CanonicalRepairHead, build_canonical_evidence, build_compact_canonical_support,
    map_canonical_evidence, repair_targets, compose_canonical, repair_loss,
)
from real_motion.canonical_repair_context import (
    FixedCanonicalCache, build_fixed_canonical,
    full_static_conflicts, compact_static_conflicts, sample_causal_points, map_sampled_canonical,
    sample_compact_causal_points, map_sampled_compact_canonical,
)
from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.canonical_repair_batch import batched_repair_losses
from real_motion.final_dataflow import batch_frozen_motion, parallel_frozen_motion
from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.column_execution import execution_session
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.source_evidence_audit import edit_quality
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import causal_column_common as columns
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import probabilities, loss_for_causal, timed_full
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics, sha256
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks, GATES
from tools.real_motion.height_field_screen_common import sync
from real_motion.causal_column_completion import actions_from_probabilities, compose_dense

PROTOCOL = 'p0_f9_point_ccr_gt_screen_v1'
CCR_HISTORY_CACHE_PROTOCOL = 'p0_f9_ccr_history_geometry_v1'
SUPPORT_NOTE = 'canonical historical static/dynamic support; NOT equivalent to old frontier GEN'


def execution_kernels(provider):
    return getattr(getattr(provider,'ccr_execution',None),'kernels',None)


def build_inputs(provider,prep):
    execution=getattr(provider,'ccr_execution',None)
    return build_canonical_evidence(prep,provider.pcfg.grid) if execution is None else execution.build(prep,provider.pcfg.grid)


def map_inputs(provider,evidence,prep):
    execution=getattr(provider,'ccr_execution',None)
    return map_canonical_evidence(evidence,prep,provider.pcfg.grid) if execution is None else execution.map(evidence,prep,provider.pcfg.grid)


def add_args(parser):
    parser.add_argument('--descriptor-cache')
    parser.add_argument('--descriptor-disk-mib', type=int, default=16384)
    parser.add_argument('--descriptor-ram-mib', type=int, default=1024)
    parser.add_argument('--samples-per-role', type=int, default=1024)
    parser.add_argument('--ccr-cpu-execution',choices=('numpy','native','native_parallel'),default='numpy')
    parser.add_argument('--ccr-cpu-workers',type=int,default=4)
    parser.add_argument('--ccr-batched-head',action='store_true',help='same window-wise loss, batched point MLP; FP rounding may differ')
    parser.add_argument('--ccr-batched-motion',action='store_true',
                        help='one frozen V18 forward per packed window batch; enable only after parity gate')
    parser.add_argument('--ccr-fast-train',action='store_true',
                        help='max execution-only speed: batched frozen motion/head, lazy sampled features, minimal prefetched geometry')
    parser.add_argument('--ccr-prefetch-workers',type=int,default=16,
                        help='CPU window look-ahead workers for --ccr-fast-train (bounded to 32)')
    parser.add_argument('--ccr-motion-superbatch-updates',type=int,default=1,
                        help='prepare this many logical 4-window updates together; optimizer batch stays unchanged')
    parser.add_argument('--ccr-motion-streams',type=int,default=4,
                        help='independent CUDA streams for frozen V18 window forwards in fast mode')
    parser.add_argument('--ccr-history-cache',
                        help='persistent fixed causal-history geometry cache built by build_p0_f9_ccr_history_cache.py')
    parser.add_argument('--ccr-history-cache-mode',choices=('off','require'),default='off',
                        help='require=fail closed on any missing history-geometry window; never rebuild during training')
    parser.add_argument('--ccr-history-cache-ram-mib',type=int,default=1024,
                        help='bounded RAM LRU for verified persistent history-geometry artifacts')
    parser.add_argument('--warm-start-head',
                        help='Point CCR checkpoint used for WEIGHTS+positive_weight only; fresh optimizer/schedule/population')


def make_head(teacher, device):
    return CanonicalRepairHead(source_dim=teacher.columns.source_dim).to(device)


def warm_start_head(head, path, *, teacher_sha256, config_fingerprint, dev_manifest_fingerprint=None):
    """Load only Point-CCR weights/calibration; never optimizer/RNG/cursor.

    This is a new full-data continuation experiment, not an exact resume of the
    finite 20%-TRAIN screen. Reusing positive_weight keeps probability
    calibration fixed so the first question is training/data coverage.
    """
    saved=torch.load(path,map_location='cpu',weights_only=False);c=saved.get('contract',{})
    if (saved.get('protocol')!=PROTOCOL or c.get('protocol')!=PROTOCOL
            or saved.get('transport_frozen') is not True
            or c.get('teacher_sha256')!=teacher_sha256
            or c.get('config_fingerprint')!=config_fingerprint
            or (dev_manifest_fingerprint is not None and c.get('dev_manifest_fingerprint')!=dev_manifest_fingerprint)
            or c.get('model')!=model_contract(head)):
        raise RuntimeError('warm-start Point CCR checkpoint/teacher/config/model mismatch')
    state=saved.get('head')
    if not isinstance(state,dict):
        raise RuntimeError('warm-start checkpoint lacks Point CCR head state')
    head.load_state_dict(state,strict=True)
    if (not all(torch.isfinite(v).all() for v in head.state_dict().values())
            or not bool((head.positive_weight>=1).all())):
        raise RuntimeError('warm-start Point CCR has invalid weights/calibration')
    prior=saved.get('reports',{}).get('train_prior')
    if not prior or 'positive_weights' not in prior:
        raise RuntimeError('warm-start Point CCR lacks persisted TRAIN-only positive weights')
    initial=saved.get('reports',{}).get('initial_dev64')
    reusable_old=None
    if initial and initial.get('variants',{}).get('old_joint'):
        reusable_old={
            'baseline':initial.get('baseline'),
            'variants':{'old_joint':initial['variants']['old_joint']},
            'windows':initial.get('windows',64),
            'reused_reference_only':True,
            'source_checkpoint_sha256':sha256(path),
        }
    return dict(
        checkpoint_sha256=sha256(path),
        source_epoch=int(saved.get('epoch',0)),
        source_updates=int(saved.get('updates',0)),
        source_train_fraction=float(c.get('train_fraction',float('nan'))),
        positive_weights=np.asarray(head.positive_weight.detach().cpu()).tolist(),
        train_prior={**prior,'reused_for_full_data_warm_start':True,
                     'note':'kept fixed to isolate training/data coverage; not recalibrated on DEV'},
        reusable_initial_dev64_old=reusable_old,
    )


def ccr_history_cache_namespace(provider, args, root):
    """Identity of fixed history geometry reusable across CCR/joint training.

    Deliberately excludes trainable CCR parameters and frozen-V18 learned
    outputs.  Any code/config/data change that can alter Strong/source
    extraction, causal registration or compact support creates a new namespace.
    """
    files=(
        'real_motion/canonical_causal_repair.py',
        'real_motion/canonical_repair_context.py',
        'real_motion/causal_geometry_cache.py',
        'tools/real_motion/ccr_screen_common.py',
        'tools/real_motion/joint_column_full_common.py',
        'tools/real_motion/causal_column_common.py',
        'tools/real_motion/benchmark_p0_f9_v18_runtime.py',
        'real_motion/native/column_cpu.cpp',
    )
    identity=dict(
        protocol=CCR_HISTORY_CACHE_PROTOCOL,
        runtime_config_fingerprint=getattr(provider,'runtime_config_fingerprint',None),
        active_history_frames=int(provider.joint.transport.config.history_frames),
        train_cache_sha256=sha256(args.train_cache),
        train_info_sha256=sha256(args.train_info),
        dataroot=str(Path(args.dataroot).resolve()),
        implementation=stable_json_fingerprint({p:sha256(root/p) for p in files}),
    )
    if identity['runtime_config_fingerprint'] is None:
        raise RuntimeError('missing runtime config fingerprint for CCR history cache')
    return stable_json_fingerprint(identity)


def model_contract(head):
    return dict(mode='point_CCR', source_dim=head.source_dim, width=head.width)


def contract_extra(args, root):
    if min(args.descriptor_disk_mib, args.descriptor_ram_mib) < 0 or args.samples_per_role < 1:
        raise ValueError('invalid finite CCR cache/sampling budget')
    files = ('real_motion/canonical_causal_repair.py', 'real_motion/canonical_repair_context.py',
             'tools/real_motion/ccr_screen_common.py', 'tools/real_motion/train_p0_f9_point_ccr.py',
             'tools/real_motion/pilot_p0_f9_canonical_causal_repair.py','real_motion/canonical_repair_execution.py',
             'real_motion/canonical_repair_batch.py','real_motion/native/column_cpu.cpp')
    result = dict(objective='equal_window_role_action_BCE_causal_MC_GT_only',
                thresholds=dict(CCR_ADD=.5, CCR_REMOVE=.95, old_Local=(.5, .5, .95)),
                samples_per_role=args.samples_per_role, remove_loss_weight=.25,
                cpu_execution=getattr(args,'ccr_cpu_execution','numpy'),cpu_workers=getattr(args,'ccr_cpu_workers',4),
                batched_head=bool(getattr(args,'ccr_batched_head',False) or getattr(args,'ccr_fast_train',False)),
                fast_train=bool(getattr(args,'ccr_fast_train',False)),
                lazy_sampled_features=bool(getattr(args,'ccr_fast_train',False)),
                prefetch_workers=(min(32,max(1,getattr(args,'ccr_prefetch_workers',16)))
                                  if getattr(args,'ccr_fast_train',False) else None),
                motion_superbatch_updates=(max(1,int(getattr(args,'ccr_motion_superbatch_updates',1)))
                                           if getattr(args,'ccr_fast_train',False) else 1),
                motion_streams=(max(1,int(getattr(args,'ccr_motion_streams',4)))
                                if getattr(args,'ccr_fast_train',False) else 1),
                support=SUPPORT_NOTE,
                ccr_implementation=stable_json_fingerprint({p: sha256(root/p) for p in files}))
    # Preserve old checkpoint identity when the new execution-only batching is
    # disabled. Enabling it creates an explicit new training execution contract.
    if getattr(args,'ccr_batched_motion',False) or getattr(args,'ccr_fast_train',False):
        result['batched_motion'] = True
    warm=getattr(args,'warm_start_head',None)
    if warm:
        if not Path(warm).is_file():
            raise ValueError('missing --warm-start-head checkpoint')
        result['warm_start_head_sha256']=sha256(warm)
        result['warm_start_semantics']='weights+positive_weight_only_fresh_optimizer_schedule_population'
    return result


def _ccr_minimal_fixed_geometry(provider, raw, record):
    """Exact Point-CCR fixed geometry without legacy Local static/frontier work."""
    from tools.real_motion.joint_column_full_common import build_ccr_training_geometry
    prefetch=max(1,getattr(provider,'raw_prefetch_workers',1))
    workers=max(1,min(2,provider.workers//prefetch))
    profile={}
    causal=build_ccr_training_geometry(
        raw,record,provider.pcfg,provider.strong,workers,profile=profile)
    causal['_ccr_fast_profile']=profile
    return causal


def _ccr_fast_fixed_geometry(provider, raw, record):
    """Worker-owned history/fixed-transport preparation for fast Point CCR.

    No learned tensors, future GT labels, sampled IDs or action targets are
    cached here.  The canonical support is the same complete support; only
    feature materialization is deferred until the GT-independent sample IDs are
    drawn on the training thread.
    """
    from types import SimpleNamespace
    causal=_ccr_minimal_fixed_geometry(provider,raw,record)
    prep=SimpleNamespace(
        raw=raw,
        state={**causal['prepared_state'],'rec':record,'gpu':None},
        registrations=causal['registrations'])
    tick=time.perf_counter()
    # Window-level parallelism owns the scarce 10-core budget. Nested entity
    # pools make outer workers block and underutilize cores; keep each compact
    # lattice single-threaded and parallelize whole windows instead.
    compact=build_compact_canonical_support(
        prep,provider.pcfg.grid,kernels=execution_kernels(provider),executor=None)
    causal['_ccr_fast_profile']['compact_support']=time.perf_counter()-tick
    tick=time.perf_counter()
    conflicts=compact_static_conflicts(compact,prep,provider.pcfg.grid)
    causal['_ccr_fast_profile']['static_conflicts']=time.perf_counter()-tick
    causal['_ccr_fast_profile']['fast_total']=sum(
        float(v) for k,v in causal['_ccr_fast_profile'].items()
        if k not in ('total','geometry_workers') and isinstance(v,(int,float)))
    causal['_ccr_compact_support']=compact
    causal['_ccr_compact_conflicts']=conflicts
    return causal


@contextmanager
def _evaluation_geometry(provider, include_old):
    """Use minimal Point-CCR geometry for current-only eval; full geometry for old Local."""
    if not getattr(provider,'ccr_fast_train',False):
        yield
        return
    previous=getattr(provider,'fixed_geometry_builder',None)
    provider.fixed_geometry_builder=(None if include_old else _ccr_minimal_fixed_geometry)
    try:
        yield
    finally:
        provider.fixed_geometry_builder=previous


def prepare_frozen_superbatch(provider, rows, teacher):
    """Prepare several logical updates while preserving per-window V18 shapes.

    First use is strictly compared with the original sequential per-window
    execution. Any mismatch disables stream concurrency and reuses the exact
    sequential references instead of aborting or relaxing parity.
    """
    if not rows:
        raise ValueError('empty frozen-motion superbatch')
    streams=getattr(provider,'ccr_motion_streams',1)
    actual=parallel_frozen_motion(
        teacher,rows,provider.device,streams=streams,render_readback=True)
    if getattr(provider,'ccr_verify_batched_motion_remaining',0)>0:
        reference=parallel_frozen_motion(
            teacher,rows,provider.device,streams=1,render_readback=True)
        mismatch=None
        for wi,(got,ref) in enumerate(zip(actual,reference)):
            for key,value in ref.items():
                if key=="_column_render_numpy":
                    for name,array in value.items():
                        if not np.array_equal(got[key][name],array):
                            mismatch=f"window={wi} renderer={name}"
                            break
                elif isinstance(value,torch.Tensor) and not torch.equal(got[key],value):
                    diff=float((got[key].float()-value.float()).abs().max().detach().cpu())
                    mismatch=f"window={wi} tensor={key} max_abs={diff}"
                if mismatch is not None:break
            if mismatch is not None:break
        if mismatch is not None:
            print(f'CCR_MOTION_STREAM_FALLBACK {mismatch}; using exact sequential frozen V18',flush=True)
            provider.ccr_motion_streams=1
            actual=reference
        else:
            print(f'CCR_MOTION_STREAM_PARITY PASS windows={len(rows)} streams={streams}',flush=True)
        provider.ccr_verify_batched_motion_remaining=0
    return actual


def _independent_sample_loss(head, sample, plan, output, target, weight, device):
    def upload(value):
        return torch.as_tensor(np.ascontiguousarray(value),device=device)
    actor=upload(sample.actor)
    tensors={k:v for k,v in output.items() if isinstance(v,torch.Tensor)}
    with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        live=head.project_sources(tensors)
        encoded=head.encode(upload(sample.features),upload(sample.labels),actor,
                            upload(sample.classes),live)
        logits=head.decode(encoded,actor,upload(plan.context),upload(plan.base),
                           upload(plan.fallback),upload(plan.legal),live)
        return repair_loss(head,logits,actor,upload(target),upload(weight)).float()


def setup(provider, args):
    # Reclaimable OS file cache is not treated as available tensor memory.
    # Explicit modest RAM bounds avoid duplicating the entire 20k population.
    if args.descriptor_ram_mib > 8192:
        raise ValueError('CCR screen RAM cache is limited to 8GiB; do not use full-population RAM')
    if not 0 <= int(getattr(args,'ccr_history_cache_ram_mib',1024)) <= 16384:
        raise ValueError('CCR history cache RAM LRU must be in [0,16384] MiB')
    history_mode=getattr(args,'ccr_history_cache_mode','off')
    history_root=getattr(args,'ccr_history_cache',None)
    if history_mode!='off' and not history_root:
        raise ValueError('--ccr-history-cache-mode require needs --ccr-history-cache')
    fast=bool(getattr(args,'ccr_fast_train',False))
    requested_prefetch=min(32,max(1,int(getattr(args,'ccr_prefetch_workers',16))))
    super_updates=max(1,int(getattr(args,'ccr_motion_superbatch_updates',1)))
    motion_streams=max(1,int(getattr(args,'ccr_motion_streams',4)))
    # Compact support made producer work much cheaper; sampled feature/plan
    # materialization is now the dominant CPU work inside each optimizer step.
    # On the real 10-core allocation, 6 producer + 4 sampled workers balances
    # the pipeline substantially better than the old 8 + 2 split.
    sample_workers=min(4,max(1,args.cpu_workers//2)) if fast else 1
    prefetch=min(requested_prefetch,max(1,args.cpu_workers-sample_workers))
    if motion_streams>8:
        raise ValueError('CCR frozen-motion CUDA streams are capped at 8')
    if super_updates>8:
        raise ValueError('CCR frozen-motion superbatch is capped at 8 logical updates (32 windows / <=1024 sources)')
    if fast and prefetch>max(1,args.cpu_workers):
        prefetch=max(1,args.cpu_workers)
    if args.descriptor_cache and args.descriptor_disk_mib:
        import shutil
        parent = Path(args.descriptor_cache).resolve()
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < 2*2**30:
            raise RuntimeError('less than 2GiB free: disable descriptor disk writes or free space first')
    provider.ccr_execution=CanonicalCpuExecution(getattr(args,'ccr_cpu_execution','numpy'),getattr(args,'ccr_cpu_workers',4))
    provider.ccr_batched_head=bool(getattr(args,'ccr_batched_head',False) or fast)
    provider.ccr_batched_motion=bool(getattr(args,'ccr_batched_motion',False) or fast)
    provider.ccr_verify_batched_motion_remaining=1 if fast else 0
    provider.ccr_motion_superbatch_updates=super_updates if fast else 1
    provider.ccr_motion_streams=motion_streams if fast else 1
    provider.ccr_verify_batched_head_remaining=1 if fast else 0
    provider.ccr_verify_compact_remaining=1 if fast else 0
    provider.ccr_fast_train=fast
    provider.ccr_sample_workers=sample_workers
    if fast:
        provider.raw_prefetch_workers=provider.raw_prefetch_depth=prefetch
        # Outer window parallelism owns the CPU budget. Avoid nested raw-I/O
        # pools multiplying 16/32 window workers into hundreds of threads.
        provider.raw_io_workers=1
        provider.train_io_workers=prefetch
        provider.fixed_geometry_builder=_ccr_fast_fixed_geometry
    provider.ccr_cache = FixedCanonicalCache(args.descriptor_ram_mib, neighbors=False,
        disk_root=args.descriptor_cache, max_disk_mib=args.descriptor_disk_mib, async_writes=True,
        kernels=provider.ccr_execution.kernels,executor=provider.ccr_execution.pool,
        lazy_sampled=fast)
    if provider.ccr_cache.disk is not None:
        provider.ccr_cache.disk.reserve = 2*2**30
    provider.ccr_samples_per_role = args.samples_per_role
    if history_root:
        root=Path(__file__).resolve().parents[2]
        namespace=ccr_history_cache_namespace(provider,args,root)
        provider.ccr_history_cache=CausalGeometryCache(
            history_root,namespace,max_bytes=0,
            ram_bytes=int(args.ccr_history_cache_ram_mib)*2**20,reserve_bytes=0)
        provider.ccr_history_cache_mode=history_mode
        print('CCR_HISTORY_CACHE '+json.dumps({
            'mode':history_mode,'root':str(Path(history_root).resolve()),
            'namespace':provider.ccr_history_cache.namespace,
            'ram_mib':args.ccr_history_cache_ram_mib},sort_keys=True),flush=True)
    print('CCR_FIXED_INPUT_CACHE '+json.dumps(provider.ccr_cache.stats()), flush=True)
    if fast:
        print(f'CCR_FAST_TRAIN independent_motion_streams={motion_streams} batched_head=1 '
              f'compact_sampled_only=1 minimal_geometry=1 '
              f'cpu_split=producer{prefetch}+sample{sample_workers}/{args.cpu_workers} '
              f'motion_superbatch_updates={super_updates} '
              f'(logical optimizer batch remains 4 windows)',flush=True)


def close(provider, result):
    cache = getattr(provider, 'ccr_cache', None)
    if cache is not None:
        if cache.disk is not None:
            cache.disk.flush()
        result['descriptor_cache'] = cache.stats(); cache.close()
    history=getattr(provider,'ccr_history_cache',None)
    if history is not None:history.close()
    execution=getattr(provider,'ccr_execution',None)
    if execution is not None:execution.close()


@torch.no_grad()
def calibrate_train(provider, source, records, teacher, head, *, progress=None, stop_event=None):
    counts = np.zeros((2, 2, 2), np.int64)
    started = time.perf_counter()
    for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records), 1):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('TRAIN-only prior interrupted; restart prior before any update')
        output = teacher.motion(record, provider.device)
        prep = provider.prepare_columns(source, record, include_gt=True, raw_window=raw, outputs=output)
        evidence, _ = provider.ccr_cache.get(prep, provider.pcfg.grid)
        plan = map_inputs(provider,evidence,prep)
        targets, valid = repair_targets(evidence, plan, raw['future_gt_occ'])
        for role in range(2):
            for action in range(2):
                mask = valid[..., action] & ((evidence.actor >= 0) == bool(role))[:, None]
                positive = int(targets[..., action][mask].sum())
                counts[role, action] += (int(mask.sum())-positive, positive)
        if wi == 1 or wi % 32 == 0 or wi == len(records):
            print(f'CCR_TRAIN_PRIOR {wi}/{len(records)} unsampled TRAIN_only', flush=True)
    weights = np.sqrt(counts[..., 0]/np.maximum(counts[..., 1], 1)).clip(1, 32).astype(np.float32)
    head.positive_weight.copy_(torch.as_tensor(weights, device=provider.device))
    report = dict(population='TRAIN-only full unsampled legal CCR action support', windows=len(records),
                  counts=counts.tolist(), positive_weights=weights.tolist(), seconds=time.perf_counter()-started,
                  probability_correction='subtract log(pos_weight); not empirical calibration guarantee')
    if progress:
        progress(dict(event='ccr_train_prior', **report))
    return report


def train_step(provider, rows, teacher, head, optimizer, rng, *, candidate_pool=None, frozen_outputs=None):
    if not rows or any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError('nonempty whole-window batch and frozen epoch19 motion required')
    device = provider.device; sync(device); started = time.perf_counter()
    head.train(); teacher.eval(); optimizer.zero_grad(set_to_none=True)
    stages = defaultdict(float); losses = []; sampled = total = 0
    packed=[];outputs=[];sizes=[];pending=[]
    batched_motion = frozen_outputs; motion_refs = None
    if batched_motion is not None:
        if len(batched_motion)!=len(rows):
            raise RuntimeError('precomputed frozen V18 output/window count mismatch')
        stages['precomputed_batched_motion'] += 0.0
        if getattr(provider,'ccr_verify_batched_motion_remaining',0)>0:
            with torch.no_grad():
                motion_refs=[teacher.motion(record,device) for record,_ in rows]
            for got,ref in zip(batched_motion,motion_refs):
                for key,value in ref.items():
                    if not isinstance(value,torch.Tensor):continue
                    other=got[key]
                    if not torch.allclose(other,value,rtol=2e-5,atol=2e-6):
                        diff=float((other.float()-value.float()).abs().max().detach().cpu())
                        raise RuntimeError(f'precomputed superbatched V18 parity failed: {key} max_abs={diff}')
    elif getattr(provider,'ccr_batched_motion',False):
        tick = time.perf_counter()
        with torch.no_grad():
            batched_motion = batch_frozen_motion(teacher, rows, device, render_readback=True)
            if getattr(provider,'ccr_verify_batched_motion_remaining',0)>0:
                motion_refs=[teacher.motion(record,device) for record,_ in rows]
                for got,ref in zip(batched_motion,motion_refs):
                    for key,value in ref.items():
                        if not isinstance(value,torch.Tensor):continue
                        other=got[key]
                        if not torch.allclose(other,value,rtol=2e-5,atol=2e-6):
                            diff=float((other.float()-value.float()).abs().max().detach().cpu())
                            raise RuntimeError(f'batched frozen V18 parity failed: {key} max_abs={diff}')
        stages['batched_motion_forward'] += time.perf_counter()-tick

    def materialize(evidence,ids,importance,prep,conflicts,gt,compact=None,compact_conflicts=None):
        if compact is None:
            sample,plan=map_sampled_canonical(
                evidence,ids,prep,provider.pcfg.grid,conflicts,
                kernels=execution_kernels(provider))
        else:
            sample,plan=map_sampled_compact_canonical(
                compact,ids,prep,provider.pcfg.grid,kernels=execution_kernels(provider),
                static_conflicts=compact_conflicts)
        y,valid=repair_targets(sample,plan,gt)
        return sample,plan,y,importance[:,None,None]*valid

    for row_index, (record, raw) in enumerate(rows):
        tick = time.perf_counter()
        with torch.no_grad():
            output = (batched_motion[row_index] if batched_motion is not None
                      else teacher.motion(record, device))
            ref_prep=None
            if motion_refs is not None:
                ref_prep=provider.prepare_columns(
                    None,record,include_gt=True,raw_window=raw,outputs=motion_refs[row_index])
            prep = provider.prepare_columns(None, record, include_gt=True, raw_window=raw, outputs=output)
            if ref_prep is not None:
                for name in ('baseline','owners','fallbacks'):
                    a=getattr(prep,name);b=getattr(ref_prep,name)
                    if len(a)!=len(b) or any(not np.array_equal(x,y) for x,y in zip(a,b)):
                        raise RuntimeError(f'batched frozen V18 rendered {name} parity failed')
        stages['live_motion_render'] += time.perf_counter()-tick; tick = time.perf_counter()

        causal=raw.get('_column_causal_preparation') or {}
        compact=causal.get('_ccr_compact_support')
        compact_conflicts=causal.get('_ccr_compact_conflicts')
        prefetched=causal.get('_ccr_prefetched_fixed')
        evidence=conflicts=None
        if compact is not None:
            stages['compact_support_hits'] += 1
        elif prefetched is not None:
            evidence,_=prefetched
            conflicts=causal['_ccr_prefetched_conflicts']
            stages['prefetched_fixed_input_hits'] += 1
        else:
            evidence, _ = provider.ccr_cache.get(prep, provider.pcfg.grid)
            conflicts = provider.ccr_cache.static_conflicts(evidence, prep, provider.pcfg.grid)
        stages['fixed_input_hash_cache_conflicts'] += time.perf_counter()-tick; tick = time.perf_counter()

        if getattr(provider,'ccr_batched_head',False):
            rng_before=copy.deepcopy(rng.bit_generator.state) if compact is not None and getattr(provider,'ccr_verify_compact_remaining',0)>0 else None
            if compact is not None:
                ids,importance=sample_compact_causal_points(compact,rng,per_role=provider.ccr_samples_per_role)
                population=len(compact)
            else:
                ids,importance=sample_causal_points(evidence,rng,per_role=provider.ccr_samples_per_role)
                population=len(evidence)
            if rng_before is not None:
                # One real-window gate: compact support must reproduce the old
                # complete-evidence population, sample IDs/weights and sampled
                # plan/targets exactly. The live RNG is consumed only once.
                legacy,_=build_fixed_canonical(
                    prep,provider.pcfg.grid,neighbors=False,kernels=execution_kernels(provider),
                    executor=None,lazy_sampled=True)
                shadow=np.random.default_rng();shadow.bit_generator.state=rng_before
                old_ids,old_importance=sample_causal_points(
                    legacy,shadow,per_role=provider.ccr_samples_per_role)
                if not np.array_equal(ids,old_ids) or not np.array_equal(importance,old_importance):
                    raise RuntimeError('compact CCR sampler parity failed')
                old_conflicts=full_static_conflicts(legacy,prep,provider.pcfg.grid)
                old_item=materialize(legacy,old_ids,old_importance,prep,old_conflicts,raw['future_gt_occ'])
                new_item=materialize(None,ids,importance,prep,None,raw['future_gt_occ'],
                                     compact=compact,compact_conflicts=compact_conflicts)
                for ai,bi in zip(old_item[:2],new_item[:2]):
                    names=('features','labels','actor','classes','world','presence') if hasattr(ai,'features') else ('flat','base','fallback','legal','context')
                    for name in names:
                        if not np.array_equal(getattr(ai,name),getattr(bi,name)):
                            raise RuntimeError(f'compact CCR sampled parity failed: {name}')
                if not np.array_equal(old_item[2],new_item[2]) or not np.array_equal(old_item[3],new_item[3]):
                    raise RuntimeError('compact CCR target/weight parity failed')
                print(f'CCR_COMPACT_SAMPLE_PARITY PASS population={population} sampled={len(ids)}',flush=True)
                provider.ccr_verify_compact_remaining=0
                item=new_item
                pending.append((row_index,item,output,population,len(ids)))
                stages['compact_parity_gate']+=time.perf_counter()-tick
                continue
            if candidate_pool is not None and len(rows)>1:
                fut=candidate_pool.submit(
                    materialize,evidence,ids,importance,prep,conflicts,raw['future_gt_occ'],
                    compact,compact_conflicts)
                pending.append((row_index,fut,output,population,len(ids)))
            else:
                item=materialize(evidence,ids,importance,prep,conflicts,raw['future_gt_occ'],
                                 compact,compact_conflicts)
                pending.append((row_index,item,output,population,len(ids)))
            stages['sample_dispatch']+=time.perf_counter()-tick
            continue

        loss, n = loss_for_causal(head, evidence, output, prep, provider.pcfg.grid,
            raw['future_gt_occ'], rng, device, conflicts, per_role=provider.ccr_samples_per_role,
            kernels=execution_kernels(provider))
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite CCR loss; previous completed checkpoint preserved')
        stages['sample_live_projection_encoder_loss'] += time.perf_counter()-tick; tick = time.perf_counter()
        (loss/len(rows)).backward()
        stages['backward'] += time.perf_counter()-tick
        losses.append(loss.detach()); sampled += n; total += len(evidence)
        del loss, evidence, prep, output, conflicts

    if motion_refs is not None:
        provider.ccr_verify_batched_motion_remaining=0

    if pending:
        tick=time.perf_counter()
        for row_index,item,output,n_total,n_sample in pending:
            sample,plan,y,weight=(item.result() if hasattr(item,'result') else item)
            packed.append((sample,plan,y,weight));outputs.append(output)
            sizes.append(len(output['history_source_context']))
            sampled+=n_sample;total+=n_total
        stages['sample_materialize_wait']+=time.perf_counter()-tick
        fields=list(zip(*packed))
        merged={
            k:torch.cat([o[k] for o in outputs],0)
            for k,v in outputs[0].items() if isinstance(v,torch.Tensor)
        }
        tick=time.perf_counter()
        batch_losses=batched_repair_losses(
            head,fields[0],fields[1],merged,sizes,fields[2],fields[3],device)
        if getattr(provider,'ccr_verify_batched_head_remaining',0)>0:
            refs=torch.stack([
                _independent_sample_loss(head,row[0],row[1],output,row[2],row[3],device)
                for row,output in zip(packed,outputs)
            ])
            actual=torch.stack([v.float() for v in batch_losses])
            if not torch.allclose(actual,refs,rtol=5e-3,atol=5e-4):
                diff=float((actual-refs).abs().max().detach().cpu())
                raise RuntimeError(f'batched Point CCR loss parity failed: max_abs={diff}')
            provider.ccr_verify_batched_head_remaining=0
        loss=sum(batch_losses)/len(rows)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite batched CCR loss')
        stages['batched_head_forward_loss']+=time.perf_counter()-tick;tick=time.perf_counter()
        loss.backward();stages['backward']+=time.perf_counter()-tick
        losses=[v.detach() for v in batch_losses]
        del batch_losses,loss,merged,fields,packed,outputs

    tick = time.perf_counter(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
    if not torch.isfinite(norm):
        raise RuntimeError('nonfinite CCR gradient; previous completed checkpoint preserved')
    optimizer.step(); sync(device); stages['clip_optimizer_finish'] += time.perf_counter()-tick
    value = float(torch.stack(losses).mean()); optimizer.zero_grad(set_to_none=True)
    return dict(loss=value, grad_norm=float(norm), windows=len(rows), sampled_points=sampled,
                canonical_points=total, seconds=time.perf_counter()-started, stages_seconds=dict(stages),
                transport_frozen=True, GT_only=True, KD=False, optimizer_updated=True,
                allocated_after_mib=torch.cuda.memory_allocated(device)/2**20 if device.type == 'cuda' else 0.)


def old_execution(teacher, provider):
    teacher.columns.column_inference_optimized = True
    teacher.columns.column_async_readback = True
    teacher.columns.column_probability_optimized = False
    teacher.columns.column_sampling_workers = provider.workers
    teacher.columns.column_inference_verify_remaining = 3
    return execution_session(teacher.columns, graphs=True, reuse=False)


@torch.no_grad()
def evaluate(provider, source, records, teacher, head, *, include_old=False, progress=None, stop_event=None):
    teacher.eval(); head.eval(); names = ('static_repair', 'dynamic_repair', 'joint') + (('old_joint',) if include_old else ())
    base = Metrics(); metrics = {name: Metrics() for name in names}
    quality = {name: defaultdict(int) for name in names}
    # role(static/dynamic) x action(ADD/REMOVE): tp/fp/fn/valid
    action_counts=np.zeros((2,2,4),np.int64)
    scenes = defaultdict(lambda: {name: Metrics() for name in ('baseline', *names)})
    started = previous = time.perf_counter(); stages = defaultdict(float)
    with _evaluation_geometry(provider,include_old), (old_execution(teacher, provider) if include_old else nullcontext()):
        for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records), 1):
            tick = time.perf_counter(); stages['input_wait'] += tick-previous
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError('CCR evaluation interrupted; resume last completed training checkpoint')
            output = teacher.motion(record, provider.device)
            prep = provider.prepare_columns(source, record, include_gt=True, raw_window=raw, outputs=output)
            stages['prepare_motion_render'] += time.perf_counter()-tick; tick = time.perf_counter()
            # Fresh complete causal domain. Evaluation never samples by GT or
            # reuses TRAIN sampled plans/learned features.
            evidence = build_inputs(provider,prep)
            plan = map_inputs(provider,evidence,prep)
            p = probabilities(head, evidence, plan, output, provider.device)
            target,valid=repair_targets(evidence,plan,raw['future_gt_occ'])
            predicted=np.stack((p[...,0]>=.5,p[...,1]>=.95),axis=-1)&plan.legal
            roles=evidence.actor>=0
            for role in (0,1):
                role_mask=(roles==bool(role))[:,None]
                for action in (0,1):
                    mask=valid[...,action]&role_mask
                    y=target[...,action]&mask;z=predicted[...,action]&mask
                    action_counts[role,action]+=(
                        int((y&z).sum()),int((~y&z&mask).sum()),
                        int((y&~z).sum()),int(mask.sum()))
            predictions = {name: compose_canonical(prep.baseline, evidence, plan, p[..., 0], p[..., 1],
                role={'static_repair': 'static', 'dynamic_repair': 'dynamic', 'joint': 'all'}[name])
                for name in names if name != 'old_joint'}
            stages['fresh_canonical_all_six_probabilities_compose'] += time.perf_counter()-tick; tick = time.perf_counter()
            support = gt_moving_support_sequence(source.nusc, prep.window.t0_token, prep.window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            moving = moving_support_masks(support, provider.pcfg.grid.shape_hwd)
            for ri, h in enumerate(columns.REPORT):
                gt = raw['future_gt_occ'][h]; scene = scenes[str(record['scene_name'])]
                before = prep.baseline[h]; base.update(ri, before, gt, moving[h]); scene['baseline'].update(ri, before, gt, moving[h])
                if include_old:
                    old_plan = columns.candidate_plan(prep, h, provider.pcfg.grid, teacher.columns.config)
                    old_p = columns.predict_probabilities(teacher.columns, prep, h, old_plan, provider.pcfg.grid, provider.device, 256)
                    predictions['old_joint'] = {h: compose_dense(before, old_plan, actions_from_probabilities(old_plan, old_p, GATES))}
                for name in names:
                    dense = predictions[name][h]
                    metrics[name].update(ri, dense, gt, moving[h]); scene[name].update(ri, dense, gt, moving[h])
                    for key, value in edit_quality(before, dense, gt).items():
                        quality[name][key] += value
            sync(provider.device); previous = time.perf_counter(); stages['moving_old_reference_metrics'] += previous-tick
            if progress:
                progress(dict(event='ccr_evaluation', window=wi, windows=len(records), stages_seconds=dict(stages)))
            if wi == 1 or wi % 16 == 0 or wi == len(records):
                print(f'CCR_EVAL {wi}/{len(records)}', flush=True)
    report = columns.report_states(base, metrics, quality, scenes)
    action_learning={}
    for role,name in ((0,'static'),(1,'dynamic')):
        for action,label in ((0,'ADD'),(1,'REMOVE')):
            tp,fp,fn,valid_count=[int(x) for x in action_counts[role,action]]
            action_learning[f'{name}/{label}']=dict(
                tp=tp,fp=fp,fn=fn,valid=valid_count,
                precision=(tp/(tp+fp) if tp+fp else None),
                recall=(tp/(tp+fn) if tp+fn else None))
    report.update(windows=len(records), seconds=time.perf_counter()-started, stages_seconds=dict(stages),
                  support=SUPPORT_NOTE,action_learning=action_learning)
    return report


@torch.no_grad()
def six_frame_speed(provider, source, records, teacher, head, *, repeats=2, stop_event=None):
    teacher.eval(); head.eval(); trials = []
    with _evaluation_geometry(provider,True), old_execution(teacher, provider):
        for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records, include_gt=False), 1):
            if raw.get('future_gt_occ') is not None:
                raise RuntimeError('FPS cannot load future GT')
            case = dict(record=record, causal=raw)
            expected = {old: timed_full(case, teacher, provider, head, old=old, verify_outputs=True,execution=getattr(provider,'ccr_execution',None)) for old in (True, False)}
            for repeat in range(repeats):
                for old in ((True, False) if repeat % 2 == 0 else (False, True)):
                    if stop_event is not None and stop_event.is_set():
                        raise InterruptedError('CCR FPS stopped; training checkpoint preserved')
                    row = timed_full(case, teacher, provider, head, old=old, verify_outputs=True,execution=getattr(provider,'ccr_execution',None))
                    if row['dense_sha256'] != expected[old]['dense_sha256']:
                        raise RuntimeError('six-frame execution repeated dense outputs changed')
                    row.update(window=wi, repeat=repeat+1); trials.append(row)
                    print(f"CCR_FPS {row['mode']} {wi}/{len(records)} six_seconds={row['seconds']:.4f} FPS={6/row['seconds']:.2f}", flush=True)
    means = {mode: float(np.mean([r['seconds'] for r in trials if r['mode'] == mode])) for mode in ('old_joint', 'CCR')}
    return dict(trials=trials, six_frame_mean_seconds=means, six_frame_amortized_FPS={k: 6/v for k, v in means.items()},
                speedup=means['old_joint']/means['CCR'],
                boundary='resident source tensors + registered FOUR histories -> fresh prior + live motion + fresh full CCR domain + SIX dense compositions',
                excludes='I/O, initial source extraction/registration, GT/metrics, graph warmup; NOT raw-sensor E2E',
                old_generation_scope_preserved=False, fixed_descriptor_cache_used=False)


def gate(new, old, speed):
    # Diagnostic tolerances, NOT permission to deploy a degraded model.
    return dict(mIoU_within_0_20pp=new['mIoU'] >= old['mIoU']-.2,
                MovingMicro_within_0_20pp=new['MovingMicro'] >= old['MovingMicro']-.2,
                all_horizons_Moving_within_0_20pp=all(new['per_horizon'][h]['MovingMicro'] >= old['per_horizon'][h]['MovingMicro']-.2
                                                    for h in ('1.0', '2.0', '3.0')),
                paired_inference_speedup_ge_3=speed['speedup'] >= 3.)


def brief(result):
    warm=bool(result.get('warm_start'))
    lines = ['===== POINT CCR / GT-ONLY THREE-PASS SCREEN =====', 'status: '+result['status'], 'protocol: '+PROTOCOL,
             '4 histories -> 6 futures; epoch19 motion FROZEN; '+('WARM-START point head' if warm else 'RANDOM point head')+'; no KD/AE.',
             'Fixed CCR_ADD=0.5 / CCR_REMOVE=0.95; old Local=(0.5,0.5,0.95).',
             'Support: '+SUPPORT_NOTE]
    if 'training_population' in result:
        lines.append('TRAIN '+json.dumps(result['training_population']))
    reports = result.get('reports', {}); initial = reports.get('initial_dev64', {}).get('variants', {}).get('old_joint')
    for row in reports.get('epochs', []):
        m = row['evaluation']['variants']['joint']['metrics']
        a=row['evaluation'].get('action_learning',{})
        def rr(key):
            value=a.get(key,{}).get('recall')
            return 'n/a' if value is None else f'{100*value:.1f}%'
        lines.append(f"epoch={row['epoch']} update={row['update']} dev64_mIoU={m['mIoU']:.6f} MovingMicro={m['MovingMicro']:.6f}" +
            (f" vs_old_mIoU={m['mIoU']-initial['metrics']['mIoU']:+.6f} vs_old_Moving={m['MovingMicro']-initial['metrics']['MovingMicro']:+.6f}" if initial else '')+
            f" sADD_R={rr('static/ADD')} dADD_R={rr('dynamic/ADD')} dREM_R={rr('dynamic/REMOVE')}")
    final = reports.get('final_dev512')
    if final:
        lines.append('===== FINAL DEV512 (development population; NOT independent test) =====')
        old = final['variants']['old_joint']['metrics']
        for name, item in final['variants'].items():
            m = item['metrics']; q = item['quality']
            lines.append(f"{name}: IoU={m['IoU']:.6f} mIoU={m['mIoU']:.6f} MovingMacro={m['MovingMacro']:.6f} MovingMicro={m['MovingMicro']:.6f} "
                f"vs_old_mIoU={m['mIoU']-old['mIoU']:+.6f} vs_old_Moving={m['MovingMicro']-old['MovingMicro']:+.6f} "
                f"add={q.get('added', 0)} remove={q.get('removed', 0)} semantic_precision={q['addition_semantic_precision']}")
    if 'speed' in reports:
        lines += ['SIX_FRAME '+json.dumps(reports['speed']['six_frame_mean_seconds']),
                  'FPS=6/mean_six_frame_latency '+json.dumps(reports['speed']['six_frame_amortized_FPS']),
                  'FPS boundary: '+reports['speed']['boundary'], 'FPS excludes: '+reports['speed']['excludes']]
    for key in ('training', 'gate', 'descriptor_cache'):
        if key in result:
            lines.append(key.upper()+' '+json.dumps(result[key]))
    lines += ['checkpoint: '+result.get('checkpoint', 'not yet saved'), 'route: '+result.get('route', 'in_progress'),
              'No dev-best/threshold search/full4369/automatic extension/promotion; frozen screen speed is NOT joint training speed.']
    if 'error' in result:
        lines.append('error: '+result['error'])
    return '\n'.join(lines)+'\n'
