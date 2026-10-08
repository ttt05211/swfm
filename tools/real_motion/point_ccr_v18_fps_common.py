"""Read-only, paired frozen V18 and point CCR complete six-frame execution.

Resident causal source INPUT tensors are shared by all arms. Learned features,
point candidates/projections/probabilities and all six dense outputs are fresh.
No future GT, metrics, calibration, training or architecture promotion.
"""
from contextlib import nullcontext
from collections import defaultdict
import hashlib
import math
import time

import numpy as np
import torch

from real_motion.canonical_causal_repair import (
    CanonicalRepairHead, build_canonical_evidence, map_canonical_evidence, compose_canonical,
)
from real_motion.runtime_fastpath import baseline_clear_flat_indices, compose_component_replacements_fast_exact
from real_motion.v18_execution_trial import reuse_v18_projections
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.height_field_screen_recovery import validate_cursor
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import align_records
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import probabilities

PROTOCOL = 'p0_f9_point_ccr_v18_paired_fps_v2_ccr_cpu'
POINT_PROTOCOL = 'p0_f9_point_ccr_gt_screen_v1'
BOUNDARIES = ('fresh_prior', 'cached_prior')
ARMS = ('clean_e14_6h_original', 'clean_e14_6h_native',
        'epoch19_v18_4h_original', 'epoch19_v18_4h_native',
        'point_ccr_4h_original', 'point_ccr_4h_native',
        'point_ccr_4h_fused', 'point_ccr_4h_parallel')


def resolve_arm_models(provider, teacher):
    """Resolve the SAVED reference, not the provider's live joint model.

    JointColumnProvider.__init__ moves Clean-E14 to ``reference`` then assigns
    ``model = joint.transport``. Keep that live model for CCR preparation; do
    not mutate it to repair a benchmark label. Validate BOTH weight owners and
    history budgets before collecting any timing.
    """
    reference = getattr(provider, 'reference', None)
    transport = teacher.transport
    if reference is None or reference is transport:
        raise RuntimeError('distinct saved Clean-E14 reference required, not the live epoch19 transport')
    if (getattr(getattr(reference, 'config', None), 'history_frames', None) != 6
            or getattr(getattr(transport, 'config', None), 'history_frames', None) != 4):
        raise RuntimeError('Clean-E14 SIX-history / epoch19 FOUR-history model identity mismatch')
    if provider.model is not transport or provider.joint is not teacher:
        raise RuntimeError('CCR provider must retain its live epoch19 transport')
    return {arm:reference if arm.startswith('clean_e14_6h_') else transport for arm in ARMS}


def select_population(records, frozen_keys, *, windows=20, stress_windows=2):
    """Scene round-robin, then highest source-count stress keys; never GT/error."""
    if (type(windows) is not int or type(stress_windows) is not int
            or windows < 1 or stress_windows < 0 or stress_windows >= windows
            or windows > len(frozen_keys)):
        raise ValueError('invalid finite FPS population')
    pool = align_records(records, frozen_keys)
    grouped = defaultdict(list)
    for r in pool:
        grouped[str(r['scene_name'])].append(r)
    order = sorted(grouped)
    selected = []
    rank = 0
    while len(selected) < windows-stress_windows:
        for scene in order:
            if rank < len(grouped[scene]):
                selected.append(grouped[scene][rank])
                if len(selected) == windows-stress_windows: break
        rank += 1
    def key(r): return str(r['scene_name']), str(r['t0_token'])
    taken = {key(r) for r in selected}
    remaining = sorted((r for r in pool if key(r) not in taken), key=lambda r: (-len(r['features']), key(r)))
    selected += remaining[:stress_windows]
    metadata = [dict(key=list(key(r)), sources=len(r['features']),
                     stratum='scene_round_robin' if i < windows-stress_windows else 'high_source_stress')
                for i,r in enumerate(selected)]
    if len(selected) != windows or len({key(r) for r in selected}) != windows:
        raise RuntimeError('FPS population contains missing/duplicate identities')
    return selected, metadata


def load_point_head(saved, *, teacher_sha256, config_fingerprint, source_dim, device,
                    allow_completed_epoch_boundary=False):
    c = saved.get('contract', {})
    if (saved.get('protocol') != POINT_PROTOCOL or c.get('protocol') != POINT_PROTOCOL
            or saved.get('transport_frozen') is not True or saved.get('deployable') is not False
            or c.get('teacher_sha256') != teacher_sha256
            or c.get('config_fingerprint') != config_fingerprint
            or c.get('model') != dict(mode='point_CCR', source_dim=source_dim, width=64)
            or c.get('thresholds', {}).get('CCR_ADD') != .5
            or c.get('thresholds', {}).get('CCR_REMOVE') != .95):
        raise RuntimeError('point CCR checkpoint/teacher/config/threshold contract mismatch')
    validate_cursor(c, *[saved.get(k) for k in ('epoch','batch','updates','executed')])
    if c['epochs'] != 3:
        raise RuntimeError('Point CCR contract must retain the fixed THREE-pass schedule')
    if allow_completed_epoch_boundary:
        # Read-only diagnostics may inspect an explicitly completed epoch
        # boundary (e.g. epoch 1/2) without weakening the stricter FPS/deploy
        # loader. Never accept an in-progress batch cursor.
        if not 1 <= saved['epoch'] <= c['epochs'] or saved['batch'] != 0:
            raise RuntimeError('diagnostic Point CCR checkpoint must be a completed epoch boundary')
    elif saved['epoch'] != c['epochs'] or saved['batch'] != 0:
        raise RuntimeError('completed THREE-pass point CCR checkpoint required; not resume/training')
    prior = saved.get('reports', {}).get('train_prior', {})
    positive = np.asarray(prior.get('positive_weights', []), dtype=np.float64)
    if positive.shape != (2, 2) or not np.isfinite(positive).all():
        raise RuntimeError('missing persisted TRAIN-only CCR probability correction')
    head = CanonicalRepairHead(source_dim=source_dim).to(device)
    head.load_state_dict(saved['head'], strict=True)
    if (not all(torch.isfinite(v).all() for v in head.state_dict().values())
            or not bool((head.positive_weight >= 1).all())
            or not np.allclose(head.positive_weight.detach().cpu().numpy(),
                               positive, rtol=0, atol=1e-6)):
        raise RuntimeError('nonfinite/missing persisted TRAIN-only CCR probability correction')
    return head.eval().requires_grad_(False)


def rebuild_prior(case, provider, *, native, backgrounds):
    state = dict(case['raw']['_column_causal_preparation']['prepared_state'])
    state.pop('column_backgrounds', None)
    state['rec'] = case['record']; state['gpu'] = case['gpu']
    profile = {}
    anchors, components = runtime._strong_all_horizons(
        state['current_sem'], state['current_pose'], state['future_poses'], state['current'],
        state['velocities'], state['source_world_points'], frame_dt_s=provider.pcfg.frame_dt_s,
        grid=provider.pcfg.grid, cfg=provider.strong, runtime_device=provider.device,
        majority_backend='native' if native else 'dense_cuda', profile=profile)
    clear = [baseline_clear_flat_indices(rows, grid=provider.pcfg.grid) for rows in components]
    state.update(anchors=anchors, baseline_by_hi=components, baseline_clear_flat_by_hi=clear)
    if backgrounds:
        state['column_backgrounds'] = [compose_component_replacements_fast_exact(a, rows, [],
            dynamic_class_ids=runtime.DYNAMIC_CLASS_IDS, free_label=17, grid=provider.pcfg.grid,
            precomputed_clear_flat_indices=k) for a,rows,k in zip(anchors,components,clear)]
    return state, profile


def assert_prior_exact(actual, expected):
    for field in ('anchors','baseline_clear_flat_by_hi'):
        if len(actual[field]) != 6 or len(expected[field]) != 6:
            raise RuntimeError('Strong requires ALL six horizons')
        for a,b in zip(actual[field],expected[field]):
            if not np.array_equal(a,b): raise RuntimeError('Strong byte mismatch: '+field)
    if len(actual['baseline_by_hi']) != 6 or len(expected['baseline_by_hi']) != 6:
        raise RuntimeError('missing Strong source horizons')
    for a,b in zip(actual['baseline_by_hi'], expected['baseline_by_hi']):
        if len(a)!=len(b): raise RuntimeError('Strong source population mismatch')
        for x,y in zip(a,b):
            if (x.class_id != y.class_id or x.source_voxel_count != y.source_voxel_count
                    or not np.array_equal(x.voxel_indices,y.voxel_indices)):
                raise RuntimeError('Strong source/class/footprint mismatch')


def fingerprint_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def result_signature(dense, probability, output):
    # BF16 cannot be converted directly to NumPy. Byte-view hashing retains
    # EXACT dtype/shape/storage content instead of accepting float tolerances.
    return dict(dense=[fingerprint_array(a) for a in dense],
                probability=None if probability is None else fingerprint_array(probability),
                motion={k:dict(shape=list(v.shape), dtype=str(v.dtype),
                    sha256=fingerprint_array(v.detach().contiguous().view(torch.uint8).cpu().numpy()))
                    for k,v in output.items()})


@torch.no_grad()
def forecast(case, provider, model, head, *, native, boundary, kernels=None, executor=None):
    if boundary not in BOUNDARIES or case['raw'].get('future_gt_occ') is not None:
        raise RuntimeError('unknown FPS boundary or future occupancy in causal forecast')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('frozen read-only motion required')
    if head is not None and (head.training or any(p.requires_grad for p in head.parameters())):
        raise RuntimeError('frozen read-only point head required')
    device = provider.device
    def sync(): torch.cuda.synchronize(device)
    stages = {}; prior_profile = {}
    def call(name, function):
        tick=time.perf_counter(); value=function(); stages[name]=time.perf_counter()-tick; return value
    sync(); started=time.perf_counter()
    if boundary=='fresh_prior':
        state, prior_profile = call('fresh_strong_prior', lambda: rebuild_prior(case,provider,native=native,backgrounds=head is not None))
    else:
        state = dict(case['raw']['_column_causal_preparation']['prepared_state'])
        state.update(rec=case['record'],gpu=case['gpu'])
    with reuse_v18_projections(model) if native else nullcontext():
        output = call('motion_forward', lambda: runtime._model_forward(model,case['gpu'],device,return_latents=True))
    probability = None
    if head is None:
        dense = call('readback_rigid_A1', lambda: runtime._forecast_once(model,state,provider.pcfg,
            provider.strong,device,precomputed_out=output))
    else:
        raw = {**case['raw'], '_column_causal_preparation': {
            **case['raw']['_column_causal_preparation'], 'prepared_state': state}}
        prep = call('live_render_source_history', lambda: provider.prepare_columns(
            None,case['record'],include_gt=False,raw_window=raw,outputs=output))
        evidence = call('fresh_full_canonical_inputs', lambda: build_canonical_evidence(prep,provider.pcfg.grid,kernels=kernels,executor=executor))
        plan = call('six_live_projection_legality', lambda: map_canonical_evidence(evidence,prep,provider.pcfg.grid,kernels=kernels,executor=executor))
        probability = call('full_point_encoding_six_readouts', lambda: probabilities(head,evidence,plan,output,device))
        dense = call('six_dense_composition', lambda: compose_canonical(
            prep.baseline,evidence,plan,probability[...,0],probability[...,1],thresholds=(.5,.95)))
    sync(); elapsed=time.perf_counter()-started
    if (len(dense)!=6 or any(d.shape!=tuple(provider.pcfg.grid.shape_hwd) for d in dense)
            or not math.isfinite(elapsed) or elapsed<=0):
        raise RuntimeError('six FINISHED dense outputs required')
    # Correctness hashing and metadata are OUTSIDE the timer. No GT is read.
    signature=result_signature(dense,probability,output)
    return dict(seconds=elapsed,stages_seconds=stages,prior_profile_ms=prior_profile,
                signature=signature, six_complete_dense=True)


def aggregate(trials):
    groups = defaultdict(list)
    for row in trials:
        if row['arm'] not in ARMS or row['boundary'] not in BOUNDARIES or not row.get('six_complete_dense'):
            raise RuntimeError('invalid/incomplete FPS sample')
        groups[(row['boundary'],row['arm'])].append(row)
    result = {}
    for (boundary,arm), rows in groups.items():
        seconds = np.asarray([r['seconds'] for r in rows],np.float64)
        if not np.isfinite(seconds).all() or (seconds<=0).any():raise RuntimeError('invalid latency')
        stage_keys = set().union(*(r['stages_seconds'] for r in rows))
        item=dict(mean_six_ms=1000*float(seconds.mean()), FPS=6/float(seconds.mean()),
            p50_ms=1000*float(np.median(seconds)),p90_ms=1000*float(np.percentile(seconds,90)),
            windows=len({r['key'] for r in rows}),samples=len(rows),
            stage_mean_ms={k:1000*float(np.mean([r['stages_seconds'].get(k,0.) for r in rows])) for k in sorted(stage_keys)})
        item['by_stratum']={}
        for stratum in sorted({r['stratum'] for r in rows}):
            population=[r['seconds'] for r in rows if r['stratum']==stratum]
            item['by_stratum'][stratum]=dict(mean_six_ms=1000*float(np.mean(population)),FPS=6/float(np.mean(population)))
        result.setdefault(boundary,{})[arm]=item
    return result
