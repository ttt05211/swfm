"""Frozen four-history joint model, two causal six-frame blocks. No GT inputs."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from real_motion.geometry import relative_transform, warp_mask
from real_motion.local_history_contract import four_frame_motion_inputs
from real_motion.causal_column_sampling import ColumnHistoryIndex
from real_motion.column_inference_pipeline import inference_gpu
from real_motion.v21_source_induction import select_scene_balanced_round_robin
from tools.real_motion import causal_column_common as columns
from tools.real_motion import eval_p0_f9_v18_zero_shot_long_rollout as legacy
from tools.real_motion.joint_column_full_common import prepare_causal_evidence

PROTOCOL = 'p0_f9_joint_history4_frozen_block_rollout_6s_v1'
OBSERVATION_PROTOCOL = 'initial_real_history_observation_union_forward_warp_no_future_sensor_v1'
THRESHOLDS = (.5, .5, None)
REPORT_HORIZONS = legacy.REPORT_HORIZONS


def select_long_population(records, windows, parent_keys, population):
    """Intersect explicit identities, then freeze scene-balanced long-dev64.

    dev512 means the LONG-ELIGIBLE subset of the original dev512, not 512 new
    records. No cache prefix, scene crossing, or short-future fallback.
    """
    if population not in ('dev64', 'dev512', 'all'):
        raise ValueError('invalid long population')
    by = {(str(r['scene_name']), str(r['t0_token'])): r for r in records}
    if len(by) != len(records): raise RuntimeError('duplicate cache identities')
    parent = tuple((str(s), str(t)) for s, t in parent_keys)
    if len(parent) != len(set(parent)): raise RuntimeError('duplicate parent identities')
    if set(parent)-set(by): raise RuntimeError('parent identity missing from dev cache')
    long = {}
    for w in windows:
        key = (str(w.scene_name), str(w.t0_token))
        if key in long: raise RuntimeError('duplicate long-window identity')
        if len(w.history_tokens) != 4 or len(w.future_tokens) != 12:
            raise RuntimeError('long window requires four history and twelve future frames')
        long[key] = w
        if key not in by: continue
        r = by[key]
        if (tuple(map(str, r['history_tokens'][-4:])) != tuple(map(str, w.history_tokens))
                or tuple(map(str, r['future_tokens'])) != tuple(map(str, w.future_tokens[:6]))
                or str(w.history_tokens[-1]) != str(w.t0_token)):
            raise RuntimeError('long/cache token identity or order mismatch')
    requested = tuple(by) if population == 'all' else parent
    eligible = tuple(k for k in requested if k in long)
    if not eligible: raise RuntimeError('no complete four-history/twelve-future windows')
    if population == 'dev64':
        if len(eligible) < 64: raise RuntimeError('fewer than 64 long-eligible dev512 windows')
        chosen = select_scene_balanced_round_robin(eligible, 64)
    else: chosen = eligible
    audit = dict(population=population, requested_parent_windows=len(requested),
        eligible_windows=len(eligible), excluded_short_future_keys=[list(k) for k in requested if k not in long],
        selected_keys=[list(k) for k in chosen], selected_windows=len(chosen),
        scenes=len({s for s, _ in chosen}), selection='parent-order intersection; scene-balanced round-robin for dev64')
    return [(long[k], by[k]) for k in chosen], audit


def validate_timestamps(nusc, window, tolerance_s=.06):
    tokens = tuple(window.history_tokens)+tuple(window.future_tokens)
    if len(tokens) != 16 or len(set(tokens)) != 16:
        raise RuntimeError('nonunique/incomplete four-history plus twelve-future sequence')
    samples = [nusc.get('sample', str(t)) for t in tokens]
    if len({s['scene_token'] for s in samples}) != 1:
        raise RuntimeError('long window crosses a scene boundary')
    for a, b, token in zip(samples, samples[1:], tokens[1:]):
        if str(a['next']) != str(token): raise RuntimeError('noncontiguous sample sequence')
    relative = (np.asarray([s['timestamp'] for s in samples], np.float64)-samples[3]['timestamp'])/1e6
    expected = (np.arange(16)-3)*.5
    if not np.allclose(relative, expected, rtol=0, atol=tolerance_s):
        raise RuntimeError('long sample timestamps do not match the frozen 2Hz horizons')


def inherited_observation_masks(raw, predicted_poses, grid, workers=1):
    """Only initial REAL sensor evidence; predictions never become observations.

    A forward-warp union preserves known support and marks genuinely new space
    unobserved. The flag denotes inherited evidence, NOT future LiDAR visibility.
    No semantics, source adapter, future masks, or annotation are accepted.
    """
    masks, poses = raw['history_observed'], raw['history_poses']
    if len(masks) != 4 or len(poses) != 4 or len(predicted_poses) != 4:
        raise ValueError('observation inheritance requires four real/four predicted poses')
    def one(target):
        out = np.zeros(grid.shape_hwd, bool)
        for mask, pose in zip(masks, poses):
            out |= warp_mask(mask, relative_transform(pose, target), grid=grid)
        return out
    with ThreadPoolExecutor(max_workers=max(1, min(4, workers))) as pool:
        return np.stack(list(pool.map(one, predicted_poses)))


def build_four_history_state(history_occ, history_poses, future_poses, pcfg, strong, device):
    """Reuse the proven V18 ABI with TWO EMPTY slots, not extra observations.

    The transport encoder slices to last4; columns see ONLY the actual four.
    Empty slots cannot affect causal association or the last-two KTA velocity.
    """
    if len(history_occ) != 4 or len(history_poses) != 4 or len(future_poses) != 6:
        raise ValueError('requires four observations and six future ego poses')
    history = [np.asarray(x, np.uint8) for x in history_occ]
    if any(x.shape != tuple(pcfg.grid.shape_hwd) or not np.isin(x, np.arange(18)).all() for x in history):
        raise ValueError('invalid predicted semantic grid')
    empty = np.full(pcfg.grid.shape_hwd, pcfg.free_label, np.uint8)
    state = legacy._build_block_state([empty, empty, *history],
        [history_poses[0], history_poses[0], *history_poses], future_poses, pcfg, strong, device)
    rec = state['rec']
    # Compute directly to avoid subtraction-induced float32 ULPs.
    rec['source_centroid_xy_t0_m'] = torch.from_numpy(legacy._kta_tensors(
        state['current'], state['velocities'], history_poses[-1], pcfg.frame_dt_s)[0])
    return state


def assert_four_inputs_equal(reference, rebuilt):
    def view(rec):
        result = four_frame_motion_inputs(rec['features'], rec['local_semantic_tube'],
            rec['frame_motion_features'], rec['target_source_mask_tube'])
        return (*result, rec['kta_displacement_xy_m'], rec['anchors_xy_t0_m'], rec['source_class_id'])
    for i, (a, b) in enumerate(zip(view(reference), view(rebuilt))):
        legacy._assert_tensor_close(f'four-history input {i}', a, b)


def synthetic_preparation(first_predictions, first_raw, future_poses, window, provider):
    """Whitelist-only predicted-history preparation: no NuScenes source access."""
    if len(first_predictions) != 6 or len(future_poses) != 12:
        raise ValueError('rollout requires six first predictions/twelve future ego poses')
    history = [np.asarray(x, np.uint8).copy() for x in first_predictions[-4:]]
    poses = list(future_poses[2:6])
    state = build_four_history_state(history, poses, future_poses[6:],
        provider.pcfg, provider.strong, provider.device)
    record = state['rec']
    record.update(scene_name=str(window.scene_name), t0_token=str(window.future_tokens[5]),
        history_tokens=tuple(map(str, window.future_tokens[:6])),
        future_tokens=tuple(map(str, window.future_tokens[6:])), sample_id='predicted-rollout-block2')
    raw = dict(history_occ=np.stack(history), history_poses=poses, future_poses=list(future_poses[6:]),
        history_observed=inherited_observation_masks(first_raw, poses, provider.pcfg.grid, provider.workers),
        future_gt_occ=None)
    evidence = prepare_causal_evidence(raw, provider.pcfg, provider.strong, provider.workers,
        state=state, column_config=provider.joint.columns.config)
    evidence['prepared_state'] = state
    raw['_column_causal_preparation'] = evidence
    # source=None makes any accidental real future data load fail immediately.
    prep = provider.prepare_columns(None, record, include_gt=False, raw_window=raw)
    if len(prep.raw['history_occ']) != 4 or prep.raw.get('future_gt_occ') is not None:
        raise RuntimeError('synthetic history budget/causality violation')
    return prep


def predict_joint_block(prep, model, grid, device, batch_size=256, feature_backend='cpu', candidate_pool=None):
    """All six horizons, same complete candidates, probabilities and compositor."""
    if prep.raw.get('future_gt_occ') is not None: raise RuntimeError('GT supplied to forecast')
    model.column_sampling_workers = 4
    predictions, edits = [], dict(added=0, removed=0, changed=0, candidate_columns=0)
    futures = ([candidate_pool.submit(columns.candidate_plan, prep, h, grid, model.config) for h in range(6)]
        if candidate_pool is not None else None)
    # Shared only inside this block: no model-dependent state persists into the
    # second block. GPU option must actually create a sampler, not silently CPU.
    index = ColumnHistoryIndex(prep, grid)
    sampler, resident = inference_gpu(device, feature_backend, prep, grid, model.config)
    with resident:
      for h in range(6):
        plan = futures[h].result() if futures else columns.candidate_plan(prep, h, grid, model.config)
        probability = columns.predict_probabilities(model, prep, h, plan, grid, device, batch_size,
            feature_backend=feature_backend, history_index=index, gpu_sampler=sampler,
            verify_features=feature_backend == 'gpu')
        actions = columns.actions_from_probabilities(plan, probability, THRESHOLDS)
        ids, before, after = columns.compose_sparse(plan, actions)
        prediction = prep.baseline[h].copy()
        prediction.reshape(-1)[ids] = after
        edits['candidate_columns'] += len(plan)
        edits['changed'] += int(np.count_nonzero(before != after))
        edits['added'] += int(np.count_nonzero((before == 17)&(after != 17)))
        edits['removed'] += int(np.count_nonzero((before != 17)&(after == 17)))
        predictions.append(prediction)
    return predictions, edits


def assert_dense_equal(reference, actual):
    if len(reference) != 6 or len(actual) != 6: raise RuntimeError('six-frame exactness required')
    for h, (a, b) in enumerate(zip(reference, actual)):
        if not np.array_equal(a, b):
            raise RuntimeError(f'first-block joint dense exactness failed horizon={h}; changed={np.count_nonzero(a != b)}')


def update_metrics(raw, hi, prediction, gt, moving, free_label):
    """Same frozen counts, one confusion histogram instead of 17 grid scans."""
    counts = columns.Metrics.counts(prediction, gt, moving, free_label)
    for key, value in zip(('occ_inter', 'occ_union', 'sem_inter', 'sem_union', 'mov_inter', 'mov_union'), counts):
        raw[key][hi] += value


def finalize_metrics(raw):
    """Dataset-accumulated per-horizon counts; empty support is NA, never zero."""
    def mean(x, axis=None):
        x = np.asarray(x)
        valid = np.isfinite(x); count = valid.sum(axis=axis)
        out = np.full(np.shape(count), np.nan)
        np.divide(np.where(valid, x, 0).sum(axis=axis), count, out=out, where=count > 0)
        return out
    occ = legacy._safe_iou(raw['occ_inter'], raw['occ_union'])
    sem = mean(legacy._safe_iou(raw['sem_inter'], raw['sem_union']), 1)
    macro = mean(legacy._safe_iou(raw['mov_inter'], raw['mov_union']), 1)
    micro = legacy._safe_iou(raw['mov_inter'].sum(1), raw['mov_union'].sum(1))
    values = dict(IoU=occ, mIoU=sem, MovingMacro=macro, MovingMicro=micro)
    return dict(per_horizon={str(h): {k: float(v[i]) for k, v in values.items()}
                for i, h in enumerate(REPORT_HORIZONS)},
        average_1s_2s_3s={k: float(mean(v[:3])) for k, v in values.items()},
        average_4s_5s_6s={k: float(mean(v[3:])) for k, v in values.items()})


def validate_resume_state(saved, contract, total):
    from real_motion.v21_source_induction import stable_json_fingerprint
    if saved.get('contract_fingerprint') != stable_json_fingerprint(contract):
        raise RuntimeError('long evaluation resume contract changed')
    cursor = saved.get('completed_windows')
    if not isinstance(cursor, int) or not 0 <= cursor <= total:
        raise RuntimeError('invalid completed-window resume cursor')
    raw = saved.get('raw_counts', {})
    expected = legacy._new_raw()
    if set(raw) != set(expected): raise RuntimeError('resume raw counts incomplete')
    restored = {}
    for k, template in expected.items():
        x = np.asarray(raw[k])
        if x.shape != template.shape or x.dtype.kind not in 'iu' or np.any(x < 0):
            raise RuntimeError('invalid resume metric counts')
        restored[k] = x.astype(np.int64)
    for prefix in ('occ', 'sem', 'mov'):
        if np.any(restored[prefix+'_inter'] > restored[prefix+'_union']):
            raise RuntimeError('invalid intersection/union counts')
    if cursor and not saved.get('first_block_exactness_passed'):
        raise RuntimeError('resume lacks first-block exactness gate')
    return cursor, restored
