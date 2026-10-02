"""Full training: source batching + CPU-only causal batch look-ahead."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import time
import numpy as np
import torch
from real_motion.rigid_transport import rigid_source_points_world
from real_motion.source_evidence_audit import transform_points
from real_motion.causal_column_model import column_loss
from tools.real_motion.joint_column_common import (JointColumnProvider, select_online_columns,
    build_online_column_candidates, sample_online_column, assemble_online_columns, motion_loss, set_lr)
from tools.real_motion.causal_column_common import (causal_source_history, FEATURE_KEYS,
    history_grid_footprint_bev_sequence, build_future_static_memory_only, fixed_candidate_geometry,
    compose_component_replacements_fast_exact, DYN, FREE)
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from real_motion.local_training_profile import StageTimer

MOTION_KEYS = ('features', 'local_semantic_tube', 'kta_displacement_xy_m',
               'frame_motion_features', 'target_source_mask_tube')
LABEL_KEYS = ('supervised_source', 'se2_target_valid', 'target_source_residual_xy_m',
    'existence', 'target_yaw_rad', 'yaw_enabled', 'yaw_label_valid', 'target_source_displacement_xy_m')


def prepare_causal_evidence(raw, pcfg, strong, workers, *, state=None, column_config=None):
    """No Torch/CUDA/model/GT access: safe worker-owned immutable evidence."""
    started = time.perf_counter(); grid = pcfg.grid
    if state is None:
        current = runtime.extract_instances_cropped_exact(raw['history_occ'][-1], raw['history_poses'][-1], grid=grid, cfg=strong)
        previous = runtime.extract_instances_cropped_exact(raw['history_occ'][-2], raw['history_poses'][-2], grid=grid, cfg=strong)
        velocity = runtime.match_instances(previous, current, float(pcfg.frame_dt_s), max_speed_mps=strong.max_match_speed_mps)
        state = {'current': current, 'velocities': velocity,
            'source_world_points': [rigid_source_points_world(c['voxel_indices'], raw['history_poses'][-1], grid=grid) for c in current]}
    current = state['current']
    registrations, _, _, audit = causal_source_history(raw['history_occ'], raw['history_poses'], state, grid, strong, workers)
    aligned = [[None if reg is None else transform_points(
        rigid_source_points_world(reg[1], raw['history_poses'][f], grid=grid), reg[0])
        for f, reg in enumerate(row)] for row in registrations]
    footprint = history_grid_footprint_bev_sequence(raw['history_poses'], raw['future_poses'], grid, workers=workers)
    memory = build_future_static_memory_only(raw['history_occ'], raw['history_observed'], raw['history_poses'], raw['future_poses'],
        grid=grid, dynamic_class_ids=DYN, free_label=FREE, workers=workers)
    result = {'current': current, 'registrations': registrations, 'aligned_history_points': aligned,
            'audit': audit, 'footprints': footprint,
            'memory': memory, 'seconds': time.perf_counter()-started}
    if column_config is not None:
        result['fixed_candidate_geometry'] = fixed_candidate_geometry(memory, footprint, grid, column_config)
    return result


def build_fixed_geometry(raw, record, pcfg, strong, workers, column_config):
    # Runtime's CPU path contains no model forward/CUDA and uses the frozen
    # bit-exact Strong implementation. First-use exactness still gates it live.
    state = runtime._prepare_record(record, None, pcfg, strong, 'cpu', raw_window=raw)
    state['column_backgrounds'] = [compose_component_replacements_fast_exact(a, comps, [],
        dynamic_class_ids=DYN, free_label=FREE, grid=pcfg.grid, precomputed_clear_flat_indices=clear)
        for a, comps, clear in zip(state['anchors'], state['baseline_by_hi'], state['baseline_clear_flat_by_hi'])]
    evidence = prepare_causal_evidence(raw, pcfg, strong, workers, state=state, column_config=column_config)
    # No V18 records/labels, window adapters, GPU inputs or network output may
    # enter persistent artifacts. Reattach the CURRENT record after cache lookup.
    evidence['prepared_state'] = {k: v for k, v in state.items() if k not in ('rec', 'window', 'gpu')}
    return evidence


class FullJointColumnProvider(JointColumnProvider):
    def load_raw_columns(self, source, record, *, include_gt):
        raw = super().load_raw_columns(source, record, include_gt=include_gt)
        cache = getattr(self, 'causal_geometry_cache', None)
        tick = time.perf_counter()
        if cache is None:
            evidence = prepare_causal_evidence(raw, self.pcfg, self.strong, self.workers)
            hit = False
        else:
            evidence, hit = cache.get_or_build((str(record['scene_name']), str(record['t0_token'])), raw,
                lambda: prepare_causal_evidence(raw, self.pcfg, self.strong, min(self.workers, 3),
                    column_config=self.joint.columns.config), defer_write=True)
            raw['_causal_cache_deferred'] = not hit
        raw['_column_causal_preparation'] = evidence
        raw['_causal_geometry_cache_hit'] = hit; raw['_causal_geometry_seconds'] = time.perf_counter()-tick
        return raw

    def prepare_columns(self, source, record, *, include_gt, raw_window=None, outputs=None):
        prepared = super().prepare_columns(source, record, include_gt=include_gt, raw_window=raw_window, outputs=outputs)
        if prepared.raw.get('_causal_cache_deferred'):
            # Strong was computed on the ORIGINAL device in the main thread,
            # not replayed on CPU in the history worker. Only its completed CPU
            # arrays enter a bounded non-blocking disk writer. No model outputs,
            # GT labels, owner/fallback or live query graph enter the artifact.
            value = {**prepared.raw['_column_causal_preparation'],
                'prepared_state': {k: v for k, v in prepared.state.items() if k not in ('rec', 'window', 'gpu')}}
            self.causal_geometry_cache.store((str(record['scene_name']), str(record['t0_token'])),
                prepared.raw, value, asynchronous=True)
        return prepared


def prefetch_column_batches(provider, source, records, batch_size, source_budget=128, *, io_workers=2):
    """One next complete batch, CPU-only; never prefetch entire epoch to RAM."""
    if min(batch_size, source_budget, io_workers) < 1: raise ValueError('positive window/source/I/O budget required')
    def groups():
        rows = []; sources = 0
        for record in records:
            n = len(record['features'])
            if rows and (len(rows) >= batch_size or sources+n > source_budget):
                yield rows; rows = []; sources = 0
            rows.append(record); sources += n
        if rows: yield rows
    iterator = iter(groups())
    def group(): return next(iterator, [])
    io = None
    def load(rows):
        if io is not None:
            raws = list(io.map(lambda r: provider.load_raw_columns(source, r, include_gt=True), rows))
            return list(zip(rows, raws))
        return [(r, provider.load_raw_columns(source, r, include_gt=True)) for r in rows]
    rows = group()
    if not rows: return
    if getattr(provider, 'causal_geometry_cache', None) is not None:
        io = ThreadPoolExecutor(max_workers=io_workers)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(load, rows)
            try:
                while rows:
                    batch = pending.result(); rows = group()
                    pending = pool.submit(load, rows) if rows else None
                    yield batch
            finally:
                if pending is not None: pending.cancel()
    finally:
        # Join the outer loader before closing the inner window pool: an
        # in-flight next-batch loader may not have called io.map yet.
        if io is not None: io.shutdown(wait=True, cancel_futures=True)


def epoch_order(length, seed, epoch):
    if length < 1 or epoch < 0: raise ValueError('invalid epoch population')
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(length)


def pack_records(records, keys):
    return {k: torch.cat([torch.as_tensor(r[k]) for r in records]) for k in keys}


def train_full_batch(joint, optimizer, provider, source, rows, rng, update, schedule_steps,
                     *, probe=False, patch_resolution=.8, control=None, control_optimizer=None,
                     profile=False, cpu_profiles=None, sampling_pool=None, sampling_workers=0):
    """True batching, not repeated optimizer steps or stale feature replay."""
    joint.train(); optimizer.zero_grad(set_to_none=True); set_lr(optimizer, update-1, schedule_steps)
    timer = StageTimer(provider.device, profile)
    if provider.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(provider.device)
    records = [r for r, _ in rows]; sizes = [len(r['features']) for r in records]
    merged = timer.call('pack_inputs', pack_records, records, (*MOTION_KEYS, *LABEL_KEYS))
    output = timer.call('motion_forward', joint.motion, merged, provider.device, gpu=True)
    local_outputs = [{k: v for k, v in zip(output, values)} for values in zip(*(v.split(sizes) for v in output.values()))]
    batches = []; prep_seconds = selection_seconds = feature_wait_seconds = worker_seconds = 0.
    candidate_wait_seconds = candidate_worker_seconds = 0.
    workers = sampling_workers or max(1, min(int(getattr(provider, 'workers', 4)), 4))
    if workers < 1: raise ValueError('positive sampling worker budget required')
    def sample(prep, selected):
        started = time.perf_counter()
        if cpu_profiles is None:
            arrays = sample_online_column(prep, selected, provider.pcfg.grid, joint.columns.config)
        else:
            arrays = cpu_profiles.run('patch_workers', sample_online_column, prep, selected, provider.pcfg.grid, joint.columns.config)
        return arrays, time.perf_counter()-started
    def candidates(prep):
        started = time.perf_counter()
        if cpu_profiles is None:
            plans = build_online_column_candidates(prep, joint.columns.config, provider.pcfg.grid)
        else:
            plans = cpu_profiles.run('candidate_workers', build_online_column_candidates, prep, joint.columns.config, provider.pcfg.grid)
        return plans, time.perf_counter()-started
    # One bounded batch of CPU-only jobs. The sampler RNG is consumed serially
    # in EXACT original window/horizon order; no worker touches live tensors.
    with (ThreadPoolExecutor(max_workers=workers) if sampling_pool is None else nullcontext(sampling_pool)) as pool:
        planning = []; pending = []
        for (record, raw), local in zip(rows, local_outputs):
            tick = time.perf_counter()
            prep = timer.call('prepare_render', provider.prepare_columns, source, record,
                include_gt=True, raw_window=raw, outputs=local)
            prep_seconds += time.perf_counter()-tick
            planning.append((prep, pool.submit(candidates, prep)))
        # GPU motion supervision can overlap queued CPU patch sampling.
        lm, stats = timer.call('motion_loss', motion_loss, output, merged, provider.device, patch_resolution, gpu=True)
        for prep, job in planning:
            tick = time.perf_counter(); plans, seconds = job.result()
            candidate_wait_seconds += time.perf_counter()-tick; candidate_worker_seconds += seconds
            tick = time.perf_counter()
            selected = select_online_columns(prep, joint.columns.config, provider.pcfg.grid, rng, candidates=plans)
            selection_seconds += time.perf_counter()-tick
            pending.append((prep, selected, [pool.submit(sample, prep, item) for item in selected]))
        for prep, selected, jobs in pending:
            tick = time.perf_counter(); mapped = [job.result() for job in jobs]
            feature_wait_seconds += time.perf_counter()-tick
            worker_seconds += sum(seconds for _, seconds in mapped)
            b = timer.call('assemble_transfer', assemble_online_columns, prep, joint.columns, selected,
                [a for a, _ in mapped], provider.device)
            if b is not None: batches.append(b)
    lc = lm.new_zeros(()); column_stats = {}; link_grad = None; sampled = 0
    if batches:
        batch = timer.call('concatenate_columns', lambda: {k: torch.cat([b[k] for b in batches]) for k in batches[0]})
        sampled = len(batch['kind'])
        with torch.autocast(device_type=provider.device.type, dtype=torch.bfloat16, enabled=provider.device.type == 'cuda'):
            g, r = timer.call('column_forward', joint.columns,
                **{k: batch[k] for k in (*FEATURE_KEYS, 'source_features')}, gpu=True)
        lc, column_stats = timer.call('column_loss', column_loss, joint.columns, g, r,
            batch['kind'], batch['legal'], batch['target'], batch['weight'], gpu=True)
        if probe and len(output['future_transport_queries']) and output['future_transport_queries'].requires_grad:
            grad = torch.autograd.grad(lc, output['future_transport_queries'], retain_graph=True, allow_unused=True)[0]
            link_grad = float(grad.float().norm()) if grad is not None else 0.
    loss = lm+lc
    updated = loss.requires_grad and (bool(batches) or bool(merged['supervised_source'].any()))
    mn = cn = 0.
    if updated:
        timer.call('backward', loss.backward, gpu=True)
        mn = timer.call('clip_motion', torch.nn.utils.clip_grad_norm_, joint.transport.parameters(), 5., error_if_nonfinite=True, gpu=True)
        cn = timer.call('clip_columns', torch.nn.utils.clip_grad_norm_, joint.columns.parameters(), 1., error_if_nonfinite=True, gpu=True)
        timer.call('optimizer', optimizer.step, gpu=True)
    control_loss = None
    if control is not None:
        if control_optimizer is None: raise ValueError('control optimizer required')
        control.train(); control_optimizer.zero_grad(set_to_none=True); set_lr(control_optimizer, update-1, schedule_steps)
        values = runtime._gpu_inputs(merged, provider.device)
        with torch.autocast(device_type=provider.device.type, dtype=torch.bfloat16, enabled=provider.device.type == 'cuda'):
            co = control(values['features'], values['tube'], values['kta'], values['frame_motion'], values['source_mask'])
        control_loss, _ = motion_loss(co, merged, provider.device, patch_resolution)
        if control_loss.requires_grad and bool(merged['supervised_source'].any()):
            control_loss.backward(); torch.nn.utils.clip_grad_norm_(control.parameters(), 5., error_if_nonfinite=True)
            control_optimizer.step()
    timings = timer.finish()
    return {**timings, 'loss': float(loss.detach()), 'motion_loss': float(lm.detach()), 'column_loss': float(lc.detach()),
        'paired_control_motion_loss': float(control_loss.detach()) if control_loss is not None else None,
        'grad_norm': float(mn), 'column_grad_norm': float(cn), 'optimizer_updated': bool(updated),
        'source_query_gradient_norm': link_grad, 'gradient_probe': probe, 'sampled_columns': sampled,
        'windows': len(rows), 'sources': sum(sizes), 'prepare_main_seconds': prep_seconds,
        'online_sampling_seconds': candidate_wait_seconds+selection_seconds+feature_wait_seconds,
        'online_candidate_wait_seconds': candidate_wait_seconds,
        'online_candidate_worker_seconds_sum': candidate_worker_seconds,
        'online_selection_seconds': selection_seconds, 'online_feature_wait_seconds': feature_wait_seconds,
        'online_worker_seconds_sum': worker_seconds, 'online_sampling_workers': workers,
        'causal_geometry_cache_hits': sum(bool(raw.get('_causal_geometry_cache_hit')) for _, raw in rows if raw is not None),
        'causal_geometry_worker_seconds_sum': sum(float(raw.get('_causal_geometry_seconds', 0.)) for _, raw in rows if raw is not None),
        'peak_memory_mib': torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else None,
        **stats, **column_stats}
