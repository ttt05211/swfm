"""Synchronous two-rank Local training with unchanged GLOBAL batch/loss means.

The live motion->hard CPU geometry->linked columns path is not exposed as a
single DDP.forward. Reduce explicit GLOBAL-normalized gradient
sums before clipping/Adam instead. Not a per-rank loss-mean average.
"""
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import os
import random

import numpy as np
import torch
import torch.distributed as dist


class DistributedTraining:
    def __init__(self, enabled=False, *, backend=None, init_method=None):
        self.world_size = int(os.environ.get('WORLD_SIZE', '1')) if enabled else 1
        self.rank = int(os.environ.get('RANK', '0')) if enabled else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0')) if enabled else 0
        self.active = self.world_size > 1
        self.device = torch.device('cpu')
        if self.active:
            if self.world_size != 2: raise RuntimeError('this audited Local backend requires exactly two ranks')
            if backend is None:
                if not torch.cuda.is_available(): raise RuntimeError('dual-card training requires CUDA')
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device('cuda', self.local_rank); backend = 'nccl'
            dist.init_process_group(backend=backend, init_method=init_method, timeout=timedelta(minutes=30),
                                    rank=self.rank, world_size=self.world_size)
        elif not enabled and int(os.environ.get('WORLD_SIZE', '1')) > 1:
            raise RuntimeError('torchrun requires explicit --distributed; do not duplicate a single-card trainer')

    @property
    def primary(self): return self.rank == 0

    def close(self):
        if self.active and dist.is_initialized(): dist.destroy_process_group()

    def barrier(self):
        if self.active: dist.barrier()

    def broadcast(self, value):
        if not self.active: return value
        items = [value if self.primary else None]
        dist.broadcast_object_list(items, src=0)
        return items[0]

    def gather(self, value):
        if not self.active: return [value]
        result = [None]*self.world_size; dist.all_gather_object(result, value)
        return result

    def sum(self, value):
        result = value.detach().clone()
        if self.active: dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def stop(self, local):
        flag = torch.tensor(int(local), dtype=torch.int32, device=self.device)
        if self.active: dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def identity(self):
        return {'world_size': self.world_size, 'global_batch_preserved': True,
                'partition': 'global_plan_window_round_robin_no_padding_no_drop_v1',
                'gradient_reduction': 'global_denominator_sum_before_global_clip_and_AdamW_v1',
                'rng': 'saved_per_rank_original_rank0_rank1_jumped_stream_v1'}

    def motion(self, stats, record, zero):
        # Counts are exactly those of the established V18 four loss terms.
        get = lambda key: torch.as_tensor(record[key], device=self.device)
        sup = get('supervised_source').bool(); valid = get('se2_target_valid').bool() & sup[:, None]
        yaw = valid & get('yaw_label_valid').bool() & get('yaw_enabled').bool()[:, None]
        mask = get('target_source_mask_tube')[:, -1]
        usable = valid & (mask.flatten(1).sum(1) > 0)[:, None] if len(mask) else valid
        counts = torch.stack((valid.sum()*2, sup.sum()*6, yaw.sum(), usable.sum())).float()
        return self.motion_terms(stats, counts, zero)

    def motion_terms(self, stats, counts, zero):
        global_counts = self.sum(counts)
        names = ('translation_smooth_l1', 'existence_bce', 'yaw_periodic_loss', 'se2_shape_loss')
        values = {name: torch.as_tensor(stats.get(name, 0.), device=zero.device)*counts[i]/global_counts[i].clamp_min(1)
                  for i, name in enumerate(names)}
        return sum(values[name]*w for name, w in zip(names, (1., 1., 19., .25))), values

    def columns(self, terms, denominators, zero):
        global_denominators = self.sum(denominators)
        count = (global_denominators > 0).sum().clamp_min(1)
        names = ('generation_bce', 'refine_action_ce')
        values = {name: terms.get(i, zero)*denominators[i]/global_denominators[i].clamp_min(1e-20)
                  for i, name in enumerate(names)}
        return sum(values.values())/count, values

    def synchronize_gradients(self, model):
        """SUM contributions already normalized globally; preserve unused None."""
        parameters = list(model.parameters())
        flags = torch.tensor([p.grad is not None for p in parameters], device=self.device, dtype=torch.int32)
        if self.active: dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        used = flags.cpu().tolist()
        selected = [p for p, take in zip(parameters, used) if take]
        # Fused one-buffer reduction avoids thousands of tiny collectives.
        if selected:
            packed = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in selected])
            if self.active: dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            offset = 0
            for p in selected:
                p.grad = packed[offset:offset+p.numel()].view_as(p).clone(); offset += p.numel()
        return bool(selected)

    def rng_state(self, rng):
        return {'sampling': rng.bit_generator.state, 'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state(self.device).cpu() if self.device.type == 'cuda' else None,
                'python': random.getstate(), 'numpy': np.random.get_state()}

    def restore_rng(self, states, rng):
        if len(states) != self.world_size: raise RuntimeError('distributed RNG rank count changed')
        value = states[self.rank]; rng.bit_generator.state = value['sampling']; torch.set_rng_state(value['torch'])
        random.setstate(value['python']); np.random.set_state(value['numpy'])
        if self.device.type == 'cuda': torch.cuda.set_rng_state(value['cuda'], self.device)


def distributed_empty_batch(joint, optimizer, context, device, *, probe=False):
    """A one-window global batch leaves one rank empty: no padding/GT duplication."""
    optimizer.zero_grad(set_to_none=True)
    zero = torch.zeros((), device=device)
    lm, motion = context.motion_terms({}, torch.zeros(4, device=device), zero)
    lc, columns = context.columns({}, torch.zeros(2, device=device), zero)
    updated = context.synchronize_gradients(joint)
    mn = cn = 0.
    if updated:
        mn = torch.nn.utils.clip_grad_norm_(joint.transport.parameters(), 5., error_if_nonfinite=True)
        cn = torch.nn.utils.clip_grad_norm_(joint.columns.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    return {'loss': float(lm+lc), 'motion_loss': float(lm), 'column_loss': float(lc),
            **{k: float(v) for k, v in {**motion, **columns}.items()},
            'grad_norm': float(mn), 'column_grad_norm': float(cn), 'optimizer_updated': updated,
            'windows': 0, 'sources': 0, 'sampled_columns': 0,
            'source_query_gradient_norm': 0., 'gradient_probe': probe,
            'prepare_main_seconds': 0., 'online_sampling_seconds': 0.}


def merge_rank_stats(rows):
    """Weighted-loss contributions SUM; timings MAX; population counters SUM."""
    result = dict(rows[0])
    sum_keys = {'loss', 'motion_loss', 'column_loss', 'translation_smooth_l1', 'existence_bce',
        'yaw_periodic_loss', 'se2_shape_loss', 'generation_bce', 'refine_action_ce',
        'windows', 'sources', 'sampled_columns', 'full_candidate_columns', 'causal_geometry_cache_hits'}
    for key in sum_keys:
        result[key] = sum(row.get(key, 0.) for row in rows)
    for key in set().union(*(row.keys() for row in rows)):
        if key in ('host_stage_seconds', 'cuda_stream_stage_seconds'):
            stages = [row.get(key, {}) for row in rows]
            result[key] = {name: max(stage.get(name, 0.) for stage in stages)
                           for name in set().union(*(stage.keys() for stage in stages))}
        elif key.startswith('gpu_feature_') or key.endswith('_worker_seconds_sum'):
            result[key] = sum(row.get(key, 0.) for row in rows)
        elif key.endswith('_seconds') or key in ('peak_memory_mib', 'source_query_gradient_norm'):
            values = [row[key] for row in rows if row.get(key) is not None]
            result[key] = max(values) if values else None
    result['distributed_world_size'] = len(rows)
    result['rank_windows'] = [row['windows'] for row in rows]
    result['rank_sources'] = [row['sources'] for row in rows]
    return result


def prefetch_distributed_batches(provider, source, records, groups, context, *, io_workers=2):
    """One bounded local look-ahead; empty shards retain the GLOBAL step slot."""
    def load(group):
        local = [records[i] for i in group[context.rank::context.world_size]]
        raw = list(readers.map(lambda row: provider.load_raw_columns(source, row, include_gt=True), local))
        return list(zip(local, raw))
    iterator = iter(groups)
    with ThreadPoolExecutor(max_workers=io_workers) as readers, ThreadPoolExecutor(max_workers=1) as loader:
        first = next(iterator, None)
        if first is None: return
        pending = loader.submit(load, first)
        try:
            while pending is not None:
                rows = pending.result(); group = next(iterator, None)
                pending = loader.submit(load, group) if group is not None else None
                yield rows
        finally:
            if pending is not None: pending.cancel()
