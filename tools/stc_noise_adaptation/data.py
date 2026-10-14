"""Phase-aware known convention and complete history geometry reconstruction."""
from concurrent.futures import ThreadPoolExecutor
import time
from tools.history_validity.data import policy_raw
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.perception_robustness.data import historical_row, finish_backgrounds
from .augmentation import corrupt


def history_inputs(eval_keys):
    evaluation = set(map(tuple, eval_keys))
    class DenseTrainHistory(ResidualHistoryInputs):
        def get(self, record, raw):
            if (str(record['scene_name']), str(record['t0_token'])) not in evaluation:
                raw = policy_raw(raw, 'dense_valid')
            return super().get(record, raw)
    return DenseTrainHistory


def augment_row(provider, row, model, seed):
    historical = {k: row[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
    registrations = row[1]['_column_causal_preparation']['registrations']
    tick = time.perf_counter()
    historical, audit = corrupt(historical, row[0], provider.pcfg.grid, model, seed,
                                registrations=registrations)
    audit['simulation_seconds'] = time.perf_counter()-tick
    audit['original_sources'] = len(row[0]['source_class_id'])
    if audit['clean_geometry_reusable']:
        return row[0], dict(row[1], _empirical_noise=audit), row[2], row[3]
    tick = time.perf_counter()
    rec, raw, inputs, history = historical_row(provider, row[0], historical, row[2][0])
    raw.update(future_poses=row[1]['future_poses'], future_gt_occ=row[1]['future_gt_occ'],
        _empirical_noise=audit, _robust_background_pending=True,
        _joint_prepare_timing=row[1].get('_joint_prepare_timing', {}),
        _joint_worker_timing=row[1].get('_joint_worker_timing', {}))
    audit['geometry_prepare_seconds'] = time.perf_counter()-tick
    return rec, raw, inputs, history


def prefetch_adapter(base, arm, model, seed):
    def batches(provider, source, groups, descriptors, *, workers=4, depth=2):
        training = source is provider.ccr_history_cache_source
        iterator = base(provider, source, groups, descriptors, workers=workers, depth=depth)
        try:
            with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
                for rows in iterator:
                    if training:
                        rows = [(rec, policy_raw(raw, 'dense_valid'), inputs, hist)
                                for rec, raw, inputs, hist in rows]
                        if arm == 'empirical':
                            tick = time.perf_counter()
                            futures = [pool.submit(augment_row, provider, row, model, seed) for row in rows]
                            rows = [f.result() for f in futures]
                            raw = dict(rows[0][1]); timing = dict(raw.get('_joint_prepare_timing', {}))
                            timing['simulation_and_rebuild_wait'] = time.perf_counter()-tick
                            raw['_joint_prepare_timing'] = timing
                            rows[0] = rows[0][0], raw, rows[0][2], rows[0][3]
                    yield rows
        finally:
            close = getattr(iterator, 'close', None)
            if close: close()
    return batches
