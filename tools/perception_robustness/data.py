"""Clean cache reuse; changed histories rebuild every source geometry view."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import numpy as np
import torch
from tools.final_semantic_evaluation.predictor import Predictor
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.scene_ego_head.integration import rebuild_pred_backgrounds
from .augmentation import corrupt


def historical_row(provider, record, historical, ego_history):
    """Existing history-only adapter, CPU preparation, NO future GT reads."""
    adapter = SimpleNamespace(provider=provider, history=ResidualHistoryInputs(provider, ram_mib=0))
    rec, raw, _, history = Predictor.prepare(adapter, record, historical)
    return rec, raw, (ego_history, None, None), history


def augment_row(provider, row, epoch, seed):
    historical, audit = corrupt(row[1], row[0], epoch, seed, provider.pcfg.grid)
    audit['original_sources'] = len(row[0]['source_class_id'])
    if audit['clean_geometry_reusable']:
        raw = dict(row[1], _perception_augmentation=audit)
        return row[0], raw, row[2], row[3]
    tick = time.perf_counter()
    rec, raw, inputs, history = historical_row(provider, row[0], historical, row[2][0])
    # Labels and GT ego are supervised outputs/explicit GT condition only;
    # source identities, KTA, registrations, source crops and memory are NEW.
    raw.update(future_poses=row[1]['future_poses'], future_gt_occ=row[1]['future_gt_occ'],
               _perception_augmentation=audit, _robust_background_pending=True,
               _joint_prepare_timing=row[1].get('_joint_prepare_timing', {}),
               _joint_worker_timing=row[1].get('_joint_worker_timing', {}))
    audit['geometry_prepare_seconds'] = time.perf_counter()-tick
    return rec, raw, inputs, history


def finish_backgrounds(provider, row):
    if not row[1].get('_robust_background_pending'): return row
    rec, raw, inputs, history = row
    raw = dict(raw)
    causal = dict(raw['_column_causal_preparation'])
    state = dict(causal['prepared_state'])
    poses = torch.as_tensor(np.asarray(raw['future_poses']), device=provider.device, dtype=torch.float64)
    state['column_backgrounds'] = rebuild_pred_backgrounds(raw, state, poses, provider)
    causal['prepared_state'] = state
    raw['_column_causal_preparation'] = causal
    raw['_robust_background_pending'] = False
    return rec, raw, inputs, history


def prefetch_adapter(base, augmentation, seed, initial_epoch=0):
    epoch_cursor = initial_epoch
    def batches(provider, source, groups, descriptors, *, workers=4, depth=2):
        nonlocal epoch_cursor
        training = source is provider.ccr_history_cache_source
        epoch = epoch_cursor
        if training: epoch_cursor += 1
        iterator = base(provider, source, groups, descriptors, workers=workers, depth=depth)
        try:
            with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
                for rows in iterator:
                    if augmentation == 'robust' and training:
                        tick = time.perf_counter()
                        futures = [pool.submit(augment_row, provider, row, epoch, seed) for row in rows]
                        rows = [f.result() for f in futures]
                        raw = dict(rows[0][1]); timing = dict(raw.get('_joint_prepare_timing', {}))
                        timing['augmentation_wait'] = time.perf_counter()-tick
                        raw['_joint_prepare_timing'] = timing
                        rows[0] = (rows[0][0], raw, rows[0][2], rows[0][3])
                    yield rows
        finally:
            close = getattr(iterator, 'close', None)
            if close: close()
    return batches
