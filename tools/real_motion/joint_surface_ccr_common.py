"""Live one-stage CCR supervision; fixed history cache, NO frozen outputs."""
from collections import defaultdict
from dataclasses import replace
import time
import numpy as np
import torch
from real_motion.canonical_causal_repair import repair_targets
from real_motion.canonical_repair_context import (
    sample_compact_causal_points, map_sampled_compact_canonical,
    sample_causal_points, map_sampled_canonical,
)
from tools.real_motion import ccr_screen_common as ccr
from tools.real_motion.joint_column_common import motion_loss, set_lr
from tools.real_motion.joint_column_full_common import pack_records, MOTION_KEYS, LABEL_KEYS


def prepare_live(provider, record, raw, output):
    """Only hard geometry is detached. The learned CCR receives ORIGINAL output."""
    diagnostic = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
    cached = raw.get('_column_causal_preparation')
    with torch.no_grad():
        if cached is not None and not getattr(provider, 'columns_checked', False):
            del raw['_column_causal_preparation']
            try:
                prep = provider.prepare_columns(None, record, include_gt=True,
                                                raw_window=raw, outputs=diagnostic)
            finally:
                raw['_column_causal_preparation'] = cached
        else:
            prep = provider.prepare_columns(None, record, include_gt=True,
                                            raw_window=raw, outputs=diagnostic)
    prep.outputs = output
    return prep


def sampled_rows(provider, prep, rng):
    causal = prep.raw.get('_column_causal_preparation') or {}
    compact = causal.get('_ccr_compact_support')
    if compact is not None:
        ids, importance = sample_compact_causal_points(compact, rng, per_role=provider.ccr_samples_per_role)
        def materialize():
            return map_sampled_compact_canonical(compact, ids, prep, provider.pcfg.grid,
                kernels=ccr.execution_kernels(provider), static_conflicts=causal.get('_ccr_compact_conflicts'))
        population = len(compact)
    else:
        evidence, _ = provider.ccr_cache.get(prep, provider.pcfg.grid)
        conflicts = provider.ccr_cache.static_conflicts(evidence, prep, provider.pcfg.grid)
        ids, importance = sample_causal_points(evidence, rng, per_role=provider.ccr_samples_per_role)
        def materialize():
            return map_sampled_canonical(evidence, ids, prep, provider.pcfg.grid, conflicts,
                                         kernels=ccr.execution_kernels(provider))
        population = len(evidence)
    def task():
        sample, plan = materialize()
        sample, plan = provider.ccr_augment_sample(sample, plan, prep)
        y, valid = repair_targets(sample, plan, prep.raw['future_gt_occ'])
        return sample, plan, y, importance[:, None, None] * valid
    # Sampling RNG is consumed on the main thread BEFORE dispatch, in record
    # order. CPU jobs access neither model, RNG, nor CUDA/autograd tensors.
    return task, population, len(ids)


def train_batch(joint, optimizer, provider, rows, rng, update, schedule_steps, *, pool=None, probe=False):
    if not rows or not all(p.requires_grad for p in joint.parameters()):
        raise RuntimeError('clean joint training requires nonempty batch and ALL parameters trainable')
    device = provider.device; start = time.perf_counter(); stages = defaultdict(float)
    joint.train(); optimizer.zero_grad(set_to_none=True); set_lr(optimizer, update-1, schedule_steps)
    records = [r for r, _ in rows]; sizes = [len(r['features']) for r in records]
    tick = time.perf_counter()
    merged_record = pack_records(records, (*MOTION_KEYS, *LABEL_KEYS))
    output = joint.motion(merged_record, device)
    lm, motion_stats = motion_loss(output, merged_record, device, materialize_stats=False)
    stages['motion_forward_loss'] += time.perf_counter()-tick
    local = [] ; cursor = 0
    for n in sizes:
        local.append({k: v[cursor:cursor+n] for k, v in output.items() if isinstance(v, torch.Tensor)})
        cursor += n
    pending = []; total = sampled = 0
    for (record, raw), out in zip(rows, local):
        tick = time.perf_counter(); prep = prepare_live(provider, record, raw, out)
        stages['live_renderer'] += time.perf_counter()-tick; tick = time.perf_counter()
        task, population, count = sampled_rows(provider, prep, rng)
        pending.append(pool.submit(task) if pool is not None else task())
        total += population; sampled += count
        stages['causal_draw_dispatch'] += time.perf_counter()-tick
    tick = time.perf_counter()
    packed = [p.result() if hasattr(p, 'result') else p for p in pending]
    stages['sample_descriptor_phase_wait'] += time.perf_counter()-tick
    fields = list(zip(*packed)); tick = time.perf_counter()
    losses = ccr._batched_add_only_losses(joint.columns, fields[0], fields[1], output, sizes,
                                        fields[2], fields[3], device)
    lc = sum(losses)/len(rows); loss = lm + lc
    if not torch.isfinite(loss): raise RuntimeError('nonfinite clean joint loss; previous checkpoint preserved')
    query_norm = None
    if probe and output['future_transport_queries'].requires_grad:
        gradient = torch.autograd.grad(lc, output['future_transport_queries'], retain_graph=True, allow_unused=True)[0]
        query_norm = 0. if gradient is None else float(gradient.detach().float().norm())
    stages['repair_forward_loss'] += time.perf_counter()-tick; tick = time.perf_counter()
    loss.backward()
    mn = torch.nn.utils.clip_grad_norm_(joint.transport.parameters(), 5., error_if_nonfinite=True)
    cn = torch.nn.utils.clip_grad_norm_(joint.columns.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    stats = dict(loss=float(loss.detach()), motion_loss=float(lm.detach()), repair_loss=float(lc.detach()),
                 motion_grad_norm=float(mn), repair_grad_norm=float(cn), source_query_gradient_norm=query_norm,
                 windows=len(rows), sources=sum(sizes), sampled_points=sampled, canonical_points=total,
                 learning_rates=[g['lr'] for g in optimizer.param_groups], transport_frozen=False,
                 GT_only=True, KD=False, optimizer_updated=True)
    stats.update({k: float(v) for k, v in motion_stats.items()})
    stages['backward_clip_optimizer'] += time.perf_counter()-tick
    optimizer.zero_grad(set_to_none=True)
    stats.update(seconds=time.perf_counter()-start, stages_seconds=dict(stages))
    return stats


@torch.no_grad()
def fit_train_prior(provider, source, records, joint, *, progress=None, stop_event=None):
    """Once, full unsampled TRAIN action counts at initialization; no DEV fit."""
    from real_motion.column_runtime_pipeline import prefetch_raw_columns
    joint.eval(); counts = np.zeros((2, 2, 2), np.int64)
    for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, source, records), 1):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('TRAIN prior interrupted')
        prep = prepare_live(provider, record, raw, joint.motion(record, provider.device))
        # Counts need neither plane descriptors nor phases. Preserve full
        # support/legality; avoid evaluating learned or geometry feature heads.
        plain = ccr._build_inputs_base(provider, prep)
        plan = provider.ccr_execution.map(plain, prep, provider.pcfg.grid)
        if wi <= 3:
            # Fail closed on the real population before any optimizer update.
            # Validation is outside throughput/FPS timing, not a permanent
            # duplicate-projection cost in the normal path.
            from real_motion.canonical_causal_repair import map_canonical_evidence
            from real_motion.surface_canonical_repair import augment_projection
            ref = map_canonical_evidence(plain,prep,provider.pcfg.grid,kernels=ccr.execution_kernels(provider))
            for key in ('flat','base','fallback','legal','context'):
                if not np.array_equal(getattr(ref,key),getattr(plan,key)):
                    raise RuntimeError('live fused projection parity failed: '+key)
            enriched=provider.ccr_augment_evidence(plain,prep)
            expected=augment_projection(enriched,ref,prep.state['current_pose'],prep.state['world_to_future'],provider.pcfg.grid)
            actual=augment_projection(enriched,plan,prep.state['current_pose'],prep.state['world_to_future'],provider.pcfg.grid)
            if not np.array_equal(expected.context,actual.context):
                raise RuntimeError('live fused phase parity failed; use reference execution')
            print(f'JOINT_SURFACE_LIVE_PROJECTION_PARITY {wi}/3 PASS',flush=True)
        y, valid = repair_targets(plain, plan, raw['future_gt_occ'])
        for role in range(2):
            for action in range(2):
                mask = valid[..., action] & ((plain.actor >= 0) == bool(role))[:, None]
                positive = int(y[..., action][mask].sum())
                counts[role, action] += (int(mask.sum())-positive, positive)
        if progress: progress(dict(event='TRAIN_prior', window=wi, windows=len(records)))
        if wi == 1 or wi % 32 == 0: print(f'JOINT_SURFACE_PRIOR {wi}/{len(records)} TRAIN_only', flush=True)
    weights = np.sqrt(counts[..., 0]/np.maximum(counts[..., 1], 1)).clip(1, 32).astype(np.float32)
    joint.columns.positive_weight.copy_(torch.as_tensor(weights, device=provider.device))
    return dict(population='TRAIN-only unsampled full legal support at RANDOM initialization',
                windows=len(records), counts=counts.tolist(), positive_weights=weights.tolist(),
                probability_correction='none', thresholds=[.5, None])
