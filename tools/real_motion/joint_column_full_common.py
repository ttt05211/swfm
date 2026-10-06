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
    build_online_column_candidates, sample_online_column, sample_online_columns, assemble_online_columns, motion_loss, set_lr,
    OnlineColumnCandidateBuilder, draw_online_column_indices, materialize_online_column, online_column_history_index)
from tools.real_motion.causal_column_common import (causal_source_history, FEATURE_KEYS,
    history_grid_footprint_bev_sequence, build_future_static_memory_only, fixed_candidate_geometry,
    compose_component_replacements_fast_exact, prepare_warm_columns_cpu, pose_motion, DYN, FREE)
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from real_motion.local_training_profile import StageTimer
from real_motion.native_column_cpu import backend_name, bundle_enabled
from real_motion.column_cpu_pipeline import horizon_pipeline_enabled, sampling_worker_budget, cpu_sampling_pool
from real_motion.v18_motion_gap import numpy

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
        state = {'current': current, 'previous': previous, 'velocities': velocity,
            'source_world_points': [rigid_source_points_world(c['voxel_indices'], raw['history_poses'][-1], grid=grid) for c in current]}
    current = state['current']
    # Same immediately previous Strong extraction already used for velocity.
    # Reuse it rather than recomputing its connected components a second time.
    registrations, _, _, audit = causal_source_history(raw['history_occ'], raw['history_poses'], state, grid, strong, workers,
        previous_instances=state.get('previous'))
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


def build_fixed_geometry(raw, record, pcfg, strong, workers, column_config, *, profile=None):
    # Runtime's CPU path contains no model forward/CUDA and uses the frozen
    # bit-exact Strong implementation. First-use exactness still gates it live.
    started = time.perf_counter()
    state = runtime._prepare_record(record, None, pcfg, strong, 'cpu', raw_window=raw)
    strong_at = time.perf_counter()
    state['column_backgrounds'] = [compose_component_replacements_fast_exact(a, comps, [],
        dynamic_class_ids=DYN, free_label=FREE, grid=pcfg.grid, precomputed_clear_flat_indices=clear)
        for a, comps, clear in zip(state['anchors'], state['baseline_by_hi'], state['baseline_clear_flat_by_hi'])]
    background_at = time.perf_counter()
    evidence = prepare_causal_evidence(raw, pcfg, strong, workers, state=state, column_config=column_config)
    if profile is not None:
        profile.update(strong_state=strong_at-started, static_backgrounds=background_at-strong_at,
            history_registration_and_static_memory=time.perf_counter()-background_at)
    # No V18 records/labels, window adapters, GPU inputs or network output may
    # enter persistent artifacts. Reattach the CURRENT record after cache lookup.
    evidence['prepared_state'] = {k: v for k, v in state.items() if k not in ('rec', 'window', 'gpu')}
    return evidence


def build_ccr_training_geometry(raw, record, pcfg, strong, workers, *, profile=None):
    """Minimal exact fixed geometry required by Point CCR training.

    Unlike legacy Local-column preparation, Point CCR never reads future static
    memory, ego footprints or old frontier candidate geometry.  Keep only the
    exact Strong transport state, backgrounds, causal registrations and audit.
    This removes a large deterministic CPU branch without changing the CCR
    support, labels, loss, motion, renderer or compositor.
    """
    started=time.perf_counter();grid=pcfg.grid
    state=runtime._prepare_record(record,None,pcfg,strong,'cpu',raw_window=raw)
    strong_at=time.perf_counter()
    state['column_backgrounds']=[compose_component_replacements_fast_exact(
        a,comps,[],dynamic_class_ids=DYN,free_label=FREE,grid=grid,
        precomputed_clear_flat_indices=clear)
        for a,comps,clear in zip(state['anchors'],state['baseline_by_hi'],state['baseline_clear_flat_by_hi'])]
    background_at=time.perf_counter()
    registrations,_,_,audit=causal_source_history(
        raw['history_occ'],raw['history_poses'],state,grid,strong,workers,
        previous_instances=state.get('previous'))
    history_at=time.perf_counter()
    result=dict(
        current=state['current'],registrations=registrations,audit=audit,
        footprints=None,memory=None,
        prepared_state={k:v for k,v in state.items() if k not in ('rec','window','gpu')})
    if profile is not None:
        profile.update(strong_state=strong_at-started,
                       static_backgrounds=background_at-strong_at,
                       history_registration=history_at-background_at,
                       skipped_old_static_memory_and_frontier=True,
                       total=history_at-started)
    return result


class FullJointColumnProvider(JointColumnProvider):
    def can_prepare_cpu(self, raw):
        causal = (raw or {}).get('_column_causal_preparation')
        return bool(getattr(self, 'columns_checked', False) and causal is not None
            and 'column_backgrounds' in causal.get('prepared_state', {}))

    def prepare_columns_cpu(self, record, raw, outputs):
        if not self.can_prepare_cpu(raw): raise RuntimeError('warm CPU preparation before renderer exactness/complete prefill')
        return prepare_warm_columns_cpu(record, raw, outputs, self.pcfg.grid)

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


class EvaluationJointColumnProvider(FullJointColumnProvider):
    """Bounded CPU look-ahead of exact FIXED geometry, with no disk writer.

    First main-thread renderer exactness still compares all six horizons with
    the original device path. Learned predictions/GT are never precomputed.
    """
    def load_raw_columns(self, source, record, *, include_gt):
        from tools.real_motion.causal_column_common import FrozenColumns
        started = time.perf_counter()
        raw = FrozenColumns.load_raw_columns(self, source, record, include_gt=include_gt)
        raw_at = time.perf_counter(); profile = {'raw_io': raw_at-started}
        # Parallel windows share a CPU quota: do not launch a large nested
        # geometry pool for EACH prefetched window. Floating-point work stays
        # unchanged and CUDA/model calls remain exclusively on the caller.
        geometry_workers = max(1, min(3, self.workers//getattr(self, 'raw_prefetch_workers', 1)))
        raw['_column_causal_preparation'] = build_fixed_geometry(raw, record, self.pcfg, self.strong,
            geometry_workers, self.joint.columns.config, profile=profile)
        profile.update(total=time.perf_counter()-started, geometry_workers=geometry_workers)
        raw['_evaluation_raw_prepare_seconds'] = profile
        return raw


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
    # Raw occupancy loading + fixed geometry are CPU-only and immutable.  They
    # may be prepared in parallel even when disk caching is disabled.  The old
    # cache-gated condition accidentally serialized cold Point-CCR training.
    if io_workers > 1:
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


def materialize_scalar_stats(values):
    """One detached scalar transfer, preserving each scalar's exact value."""
    tensors={k:v.detach().reshape(()) for k,v in values.items() if isinstance(v,torch.Tensor)}
    result=dict(values)
    if tensors:
        result.update(zip(tensors,torch.stack(list(tensors.values())).cpu().tolist()))
    return result


def train_full_batch(joint, optimizer, provider, source, rows, rng, update, schedule_steps,
                     *, probe=False, patch_resolution=.8, control=None, control_optimizer=None,
                     profile=False, cpu_profiles=None, sampling_pool=None, sampling_workers=0, optimize_cpu=True, optimize_kernels=True,
                     column_feature_sampler=None, continuation=None, distributed=None):
    """True batching, not repeated optimizer steps or stale feature replay."""
    joint.train(); optimizer.zero_grad(set_to_none=True)
    if continuation is None: set_lr(optimizer, update-1, schedule_steps)
    else:
        from tools.real_motion.joint_training_extension import set_extension_lr
        set_extension_lr(optimizer, update-1, continuation)
    if not rows:
        if distributed is None: raise RuntimeError('empty single-rank training batch')
        from tools.real_motion.joint_training_distributed import distributed_empty_batch
        return distributed_empty_batch(joint, optimizer, distributed, provider.device, probe=probe)
    timer = StageTimer(provider.device, profile)
    bundle=optimize_cpu and optimize_kernels and bundle_enabled()
    horizons=bundle and horizon_pipeline_enabled()
    if provider.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(provider.device)
    records = [r for r, _ in rows]; sizes = [len(r['features']) for r in records]
    merged = timer.call('pack_inputs', pack_records, records, (*MOTION_KEYS, *LABEL_KEYS))
    output = timer.call('motion_forward', joint.motion, merged, provider.device, gpu=True)
    local_outputs = [{k: v for k, v in zip(output, values)} for values in zip(*(v.split(sizes) for v in output.values()))]
    if optimize_cpu:
        # One D2H synchronization/copy per prediction head for the whole batch,
        # not two per window. Keep original future queries and outputs live.
        def render_readback():
            return {k: numpy(output[k]) for k in ('residual_xy_m', 'yaw_delta_rad')}
        render = timer.call('render_readback', render_readback)
        cursor = 0
        for size, local in zip(sizes, local_outputs):
            local['_column_render_numpy'] = {k: v[cursor:cursor+size] for k, v in render.items()}; cursor += size
    batches = []; supervision_parts = []; prep_seconds = selection_seconds = feature_wait_seconds = worker_seconds = 0.
    candidate_wait_seconds = candidate_worker_seconds = 0.
    materialize_seconds = index_worker_seconds = 0.
    candidate_jobs = feature_jobs = index_jobs = 0
    gpu_stats = {}
    candidate_columns=compact_columns=compact_bytes=materialized_columns=0
    workers = sampling_worker_budget(int(getattr(provider, 'workers', 4)), sampling_workers, horizons=horizons)
    def cpu_call(name, fn, *args, **kwargs):
        return fn(*args, **kwargs) if cpu_profiles is None else cpu_profiles.run(name, fn, *args, **kwargs)
    def candidate_builder(prep):
        started = time.perf_counter()
        builder = cpu_call('candidate_setup_workers', OnlineColumnCandidateBuilder,
            prep, joint.columns.config, provider.pcfg.grid, defer_context=True)
        return builder, time.perf_counter()-started
    def candidate_horizon(builder, h):
        started = time.perf_counter()
        return cpu_call('candidate_workers', builder.build, h), time.perf_counter()-started
    def history_index(prep, draws):
        started = time.perf_counter()
        index = cpu_call('history_index_workers', online_column_history_index, prep, draws, provider.pcfg.grid)
        return index, time.perf_counter()-started
    def pack_gpu_window(prep, selected):
        # Only NumPy here. All device work and live query gathering remain on
        # the main/autograd thread; caller RNG has already drawn all IDs.
        from real_motion.column_gpu_sampling import pack_column_window
        started = time.perf_counter()
        materialized = [materialize_online_column(row) for row in selected] if horizons else selected
        materialize_time = time.perf_counter()-started
        packed = pack_column_window(prep, materialized, provider.pcfg.grid, joint.columns.config, pose_motion)
        return materialized, packed, time.perf_counter()-started, materialize_time
    def sample_horizon(prep, draw, index_job):
        # The index was queued BEFORE every dependent job in the feature FIFO.
        # No worker submits children, and even a single worker cannot deadlock.
        index, _ = index_job.result()
        started = time.perf_counter(); tick = started
        selected = cpu_call('materialize_workers', materialize_online_column, draw)
        materialized = time.perf_counter()-tick
        arrays = cpu_call('patch_workers', sample_online_column, prep, selected,
            provider.pcfg.grid, joint.columns.config, history_index=index)
        return selected, arrays, time.perf_counter()-started, materialized
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
            plans = build_online_column_candidates(prep, joint.columns.config, provider.pcfg.grid,
                defer_context=optimize_cpu and optimize_kernels)
        else:
            plans = cpu_profiles.run('candidate_workers', build_online_column_candidates, prep, joint.columns.config, provider.pcfg.grid,
                defer_context=optimize_cpu and optimize_kernels)
        return plans, time.perf_counter()-started
    def sample_window(prep, selected):
        started = time.perf_counter()
        if cpu_profiles is None: arrays = sample_online_columns(prep, selected, provider.pcfg.grid, joint.columns.config)
        else: arrays = cpu_profiles.run('patch_workers', sample_online_columns, prep, selected, provider.pcfg.grid, joint.columns.config)
        return arrays, time.perf_counter()-started
    def prepare_cpu(record, raw, local):
        if cpu_profiles is None: return provider.prepare_columns_cpu(record, raw, local)
        return cpu_profiles.run('prepare_workers', provider.prepare_columns_cpu, record, raw, local)
    # One bounded batch of CPU-only jobs. The sampler RNG is consumed serially
    # in EXACT original window/horizon order; no worker touches live tensors.
    with (cpu_sampling_pool(workers, horizons=horizons) if sampling_pool is None else nullcontext(sampling_pool)) as pool:
        planning = []; pending = []; preparation = []
        submit_features = getattr(pool, 'submit_features', pool.submit)
        # Warm geometry workers never run CUDA, forward, RNG or latent gathers.
        # Cold/first-exactness calls remain on the original owning thread.
        for (record, raw), local in zip(rows, local_outputs):
            can_worker = (optimize_cpu and hasattr(provider, 'can_prepare_cpu') and provider.can_prepare_cpu(raw))
            if can_worker:
                preparation.append((record, raw, local, pool.submit(prepare_cpu, record, raw, local)))
            else:
                preparation.append((record, raw, local, None))
        for record, raw, local, prepared_job in preparation:
            tick = time.perf_counter()
            if prepared_job is None:
                prep = timer.call('prepare_render', provider.prepare_columns, source, record,
                    include_gt=True, raw_window=raw, outputs=local)
            else: prep = timer.call('prepare_render', prepared_job.result)
            prep.cpu_pipeline_optimized = optimize_cpu
            prep.cpu_kernels_optimized = optimize_cpu and optimize_kernels
            prep.cpu_bundle_optimized = bundle
            # Fixtures/legacy providers may reuse a prepared object. No index
            # from a previous pose/window is allowed to leak into this update.
            if hasattr(prep, 'column_history_index'): del prep.column_history_index
            prep_seconds += time.perf_counter()-tick
            planning.append((prep, pool.submit(candidate_builder if horizons else candidates, prep)))
        if horizons:
            split_planning = []
            for prep, job in planning:
                tick = time.perf_counter(); builder, seconds = job.result()
                candidate_wait_seconds += time.perf_counter()-tick; candidate_worker_seconds += seconds
                jobs = [pool.submit(candidate_horizon, builder, h) for h in range(6)]
                candidate_jobs += len(jobs); split_planning.append((prep, jobs))
            planning = split_planning
        # GPU motion supervision can overlap queued CPU patch sampling.
        lm, stats = timer.call('motion_loss', motion_loss, output, merged, provider.device, patch_resolution,
            materialize_stats=not bundle, distributed=distributed, gpu=True)
        audits = []
        for prep, job in planning:
            tick = time.perf_counter()
            if horizons:
                mapped = [future.result() for future in job]
                plans = [plan for plan, _ in mapped]; seconds = sum(s for _, s in mapped)
            else: plans, seconds = job.result()
            candidate_wait_seconds += time.perf_counter()-tick; candidate_worker_seconds += seconds
            tick = time.perf_counter()
            selected = (draw_online_column_indices if horizons else select_online_columns)(
                prep, joint.columns.config, provider.pcfg.grid, rng, candidates=plans)
            selection_seconds += time.perf_counter()-tick
            candidate_columns+=sum(len(plan) for _,plan,_ in plans)
            audits.extend(plan for _,plan,_ in plans)
            if column_feature_sampler is not None:
                jobs = [submit_features(pack_gpu_window, prep, selected)]
                feature_jobs += len(jobs); index_jobs += int(bool(selected))
                pending.append((prep, selected, jobs, None))
            elif horizons:
                # One immutable membership index per window, no repeated
                # inversions/tables per horizon and no large process copies.
                index_job = submit_features(history_index, prep, selected) if selected else None
                jobs = [submit_features(sample_horizon, prep, draw, index_job) for draw in selected]
                feature_jobs += len(jobs); index_jobs += int(index_job is not None)
                pending.append((prep, selected, jobs, index_job))
            else:
                jobs = [pool.submit(sample_window, prep, selected)] if optimize_cpu else [pool.submit(sample, prep, item) for item in selected]
                pending.append((prep, selected, jobs, None))
        for prep, selected, jobs, index_job in pending:
            tick = time.perf_counter(); mapped = [job.result() for job in jobs]
            feature_wait_seconds += time.perf_counter()-tick
            if column_feature_sampler is not None:
                selected, packed, seconds, materialized = mapped[0]
                worker_seconds += seconds; materialize_seconds += materialized
                arrays, extra = timer.call('gpu_history_features', column_feature_sampler.sample,
                    prep, selected, provider.pcfg.grid, joint.columns.config, pose_motion, packed=packed, gpu=True)
                for k, v in extra.items(): gpu_stats[k] = gpu_stats.get(k, 0)+v
            elif horizons:
                if index_job is not None: index_worker_seconds += index_job.result()[1]
                selected = [row for row, _, _, _ in mapped]
                arrays = [a for _, a, _, _ in mapped]
                worker_seconds += sum(seconds for _, _, seconds, _ in mapped)
                materialize_seconds += sum(seconds for _, _, _, seconds in mapped)
            else:
                worker_seconds += sum(seconds for _, seconds in mapped)
                arrays = mapped[0][0] if optimize_cpu else [a for a, _ in mapped]
            b = timer.call('assemble_transfer', assemble_online_columns, prep, joint.columns, selected, arrays, provider.device)
            if b is not None:
                batches.append(b)
                for features in arrays:
                    supervision_parts.append({k: features[k] for k in ('kind','legal','target','weight')})
        # Audit AFTER worker-owned materialization, not at the draw-only stage.
        for plan in audits:
            if hasattr(plan, 'audit'):
                audit=plan.audit(); compact_columns+=audit['population']; compact_bytes+=audit['compact_bytes']
                materialized_columns+=audit['materialized_rows']
            else: materialized_columns+=len(plan)
    lc = lm.new_zeros(()); column_stats = {}; link_grad = None; sampled = 0
    if batches:
        batch = timer.call('concatenate_columns', lambda: {k: torch.cat([b[k] for b in batches]) for k in batches[0]})
        sampled = len(batch['kind'])
        with torch.autocast(device_type=provider.device.type, dtype=torch.bfloat16, enabled=provider.device.type == 'cuda'):
            g, r = timer.call('column_forward', joint.columns,
                **{k: batch[k] for k in (*FEATURE_KEYS, 'source_features')}, gpu=True)
        from real_motion.local_supervision_fastpath import enabled, column_indices
        host_indices = None
        if bundle and enabled():
            host_indices = column_indices(**{k: np.concatenate([v[k] for v in supervision_parts])
                for k in ('kind','legal','target','weight')}, device=provider.device)
        lc, column_stats = timer.call('column_loss', column_loss, joint.columns, g, r,
            batch['kind'], batch['legal'], batch['target'], batch['weight'], materialize_stats=not bundle,
            distributed=distributed, supervision_indices=host_indices, gpu=True)
        if probe and len(output['future_transport_queries']) and output['future_transport_queries'].requires_grad:
            grad = torch.autograd.grad(lc, output['future_transport_queries'], retain_graph=True, allow_unused=True)[0]
            link_grad = float(grad.float().norm()) if grad is not None else 0.
    elif distributed is not None:
        lc, column_stats = distributed.columns({}, lm.new_zeros(2), lm.new_zeros(()))
    loss = lm+lc
    updated = loss.requires_grad and (bool(batches) or bool(merged['supervised_source'].any()))
    mn = cn = 0.
    if distributed is not None:
        if updated: timer.call('backward', loss.backward, gpu=True)
        updated = timer.call('gradient_sync', distributed.synchronize_gradients, joint, gpu=True)
    if updated:
        if distributed is None: timer.call('backward', loss.backward, gpu=True)
        mn = timer.call('clip_motion', torch.nn.utils.clip_grad_norm_, joint.transport.parameters(), 5., error_if_nonfinite=True, gpu=True)
        cn = timer.call('clip_columns', torch.nn.utils.clip_grad_norm_, joint.columns.parameters(), 1., error_if_nonfinite=True, gpu=True)
        timer.call('optimizer', optimizer.step, gpu=True)
    control_loss = None
    if control is not None:
        if control_optimizer is None: raise ValueError('control optimizer required')
        control.train(); control_optimizer.zero_grad(set_to_none=True)
        if continuation is None: set_lr(control_optimizer, update-1, schedule_steps)
        else:
            from tools.real_motion.joint_training_extension import set_extension_lr
            set_extension_lr(control_optimizer, update-1, {**continuation, 'start_learning_rates': continuation['start_learning_rates'][:1]})
        values = runtime._gpu_inputs(merged, provider.device)
        with torch.autocast(device_type=provider.device.type, dtype=torch.bfloat16, enabled=provider.device.type == 'cuda'):
            co = control(values['features'], values['tube'], values['kta'], values['frame_motion'], values['source_mask'])
        control_loss, _ = motion_loss(co, merged, provider.device, patch_resolution)
        if control_loss.requires_grad and bool(merged['supervised_source'].any()):
            control_loss.backward(); torch.nn.utils.clip_grad_norm_(control.parameters(), 5., error_if_nonfinite=True)
            control_optimizer.step()
    scalars={'loss':loss.detach(),'motion_loss':lm.detach(),'column_loss':lc.detach(),
        'paired_control_motion_loss':control_loss.detach() if control_loss is not None else None,
        'grad_norm':mn,'column_grad_norm':cn,**stats,**column_stats}
    if bundle: scalars=timer.call('diagnostics_readback',materialize_scalar_stats,scalars)
    else: scalars={k:float(v) if isinstance(v,torch.Tensor) else v for k,v in scalars.items()}
    timings = timer.finish()
    return {**timings,**scalars,**gpu_stats,'optimizer_updated': bool(updated),
        'column_feature_backend': 'gpu' if column_feature_sampler is not None else 'cpu',
        'source_query_gradient_norm': link_grad, 'gradient_probe': probe, 'sampled_columns': sampled,
        'windows': len(rows), 'sources': sum(sizes), 'prepare_main_seconds': prep_seconds,
        'online_sampling_seconds': candidate_wait_seconds+selection_seconds+feature_wait_seconds,
        'online_candidate_wait_seconds': candidate_wait_seconds,
        'online_candidate_worker_seconds_sum': candidate_worker_seconds,
        'online_selection_seconds': selection_seconds, 'online_feature_wait_seconds': feature_wait_seconds,
        'online_worker_seconds_sum': worker_seconds, 'online_sampling_workers': workers,
        'cpu_task_granularity': 'horizon' if horizons else 'window',
        'online_candidate_horizon_jobs': candidate_jobs, 'online_feature_horizon_jobs': feature_jobs,
        'online_history_index_jobs': index_jobs,
        'online_history_index_worker_seconds_sum': index_worker_seconds,
        'online_materialize_worker_seconds_sum': materialize_seconds,
        'online_candidate_pool_workers': getattr(pool, 'candidate_workers', workers),
        'online_feature_pool_workers': getattr(pool, 'feature_workers', workers),
        'online_shared_worker_pool': getattr(pool, 'shared_worker_pool', False),
        'causal_geometry_cache_hits': sum(bool(raw.get('_causal_geometry_cache_hit')) for _, raw in rows if raw is not None),
        'causal_geometry_worker_seconds_sum': sum(float(raw.get('_causal_geometry_seconds', 0.)) for _, raw in rows if raw is not None),
        'cpu_pipeline_optimized': optimize_cpu,
        'cpu_kernels_optimized': optimize_cpu and optimize_kernels,
        'integer_cpu_backend': backend_name(),
        'cpu_bundle_optimized': bundle,
        'full_candidate_columns':candidate_columns,'compact_candidate_columns':compact_columns,
        'materialized_candidate_columns':materialized_columns,'compact_descriptor_bytes':compact_bytes,
        'parallel_warm_preparations': sum(job is not None for _, _, _, job in preparation),
        'peak_memory_mib': torch.cuda.max_memory_allocated(provider.device)/2**20 if provider.device.type == 'cuda' else None,
        }
