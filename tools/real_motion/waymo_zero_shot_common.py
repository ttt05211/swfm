"""Whole-window, integer-only recovery for frozen Waymo zero-shot evaluation."""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from real_motion.waymo_i2world import WaymoMetrics, fingerprint

BRANCHES = ('transport', 'joint')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def restore(saved, contract, *, voxel_count):
    value = dict(saved); digest = value.pop('fingerprint', None)
    if digest != fingerprint(value) or value.get('contract_fingerprint') != fingerprint(contract):
        raise RuntimeError('Waymo resume fingerprint/contract changed')
    completed = value.get('completed_windows')
    if type(completed) is not int or not 0 <= completed <= contract['windows']:
        raise RuntimeError('invalid Waymo completed-window cursor')
    if set(value.get('counts', {})) != set(BRANCHES):
        raise RuntimeError('incomplete Waymo branches')
    for counts in value['counts'].values():
        arr = np.asarray(counts)
        if (arr.shape != (3, 18, 18) or arr.dtype.kind not in 'ui' or (arr < 0).any()
                or not (arr.sum((1, 2)) == completed * voxel_count).all()):
            raise RuntimeError('invalid Waymo integer metric state')
    for key in ('stage_seconds', 'edits'):
        if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in value[key].values()):
            raise RuntimeError('invalid saved Waymo timing/edit state')
    if value['edits'].get('removed', 0) or (completed and not value.get('exactness_passed')):
        raise RuntimeError('ADD-only/exactness recovery contract violated')
    return value


def evaluate_windows(source, selected, predict, contract, *, saved=None, save=None,
                     progress=None, stop_event=None, checkpoint_every=8):
    """predict(record, raw, verify=...) must finish SIX frames before GT reads.

    Only complete two-branch windows enter persisted counts. Reverify the live
    implementation after every process restart; replay at most the unsaved tail.
    """
    if checkpoint_every < 1 or contract['windows'] != len(selected):
        raise ValueError('invalid Waymo population/checkpoint interval')
    state = dict(completed_windows=0, counts={k: WaymoMetrics().counts.tolist() for k in BRANCHES},
                 exactness_passed=False, stage_seconds={}, edits={}) if saved is None else restore(
                     saved, contract, voxel_count=int(np.prod(source.shape)))
    metrics = {k: WaymoMetrics(v, state['completed_windows']) for k, v in state['counts'].items()}
    stages = defaultdict(float, state['stage_seconds']); edits = defaultdict(int, state['edits'])
    live_checked = False

    def persist():
        state.update(counts={k: v.counts.tolist() for k, v in metrics.items()},
                     stage_seconds=dict(stages), edits=dict(edits), contract_fingerprint=fingerprint(contract))
        result = dict(state); result['fingerprint'] = fingerprint(result)
        if save:
            save(result)
        return result

    persist()
    try:
        for window in selected[state['completed_windows']:]:
            if stop_event is not None and stop_event.is_set():
                break
            tick = time.perf_counter(); mark = tick
            record, raw = source.prediction_inputs(window)
            timing = {'history_io': time.perf_counter()-mark}; mark = time.perf_counter()
            transport, joint, changed, model_stages = predict(record, raw, verify=not live_checked)
            # Fail before reading any future label if a forecast is incomplete.
            for forecasts in (transport, joint):
                if len(forecasts) != 6 or any(np.asarray(v).shape != source.shape for v in forecasts):
                    raise RuntimeError('Waymo prediction must complete SIX dense grids before GT access')
            live_checked = True
            timing['prediction_and_first_exactness'] = time.perf_counter()-mark; mark = time.perf_counter()
            targets = source.metric_targets(window)
            timing['future_target_io'] = time.perf_counter()-mark; mark = time.perf_counter()
            # Validate BOTH deltas before committing either accumulator.
            increments = {k: WaymoMetrics() for k in BRANCHES}
            increments['transport'].add(transport, targets); increments['joint'].add(joint, targets)
            if changed.get('removed', 0):
                raise RuntimeError('Waymo ADD-only forecast removed occupied geometry')
            for key in BRANCHES:
                metrics[key].counts += increments[key].counts; metrics[key].windows += 1
            timing['integer_metrics'] = time.perf_counter()-mark
            timing['window'] = time.perf_counter()-tick
            for k, v in timing.items():
                stages[k] += v
            for k, v in model_stages.items():
                stages['model.'+k] += v
            for k, v in changed.items():
                edits[k] += int(v)
            state['completed_windows'] += 1; state['exactness_passed'] = True
            if progress:
                progress(dict(window=state['completed_windows'], windows=len(selected), anchor=window.anchor,
                              t0=record['t0_token'], seconds=timing['window'], stages=timing,
                              model_stages=model_stages, edits=changed))
            if state['completed_windows'] % checkpoint_every == 0:
                persist()
            del transport, joint, targets
    finally:
        persist()  # failed windows are not counted; source weights are never written
    return dict(status='complete' if state['completed_windows'] == len(selected) else 'stopped',
                completed_windows=state['completed_windows'], reports={k: v.report() for k, v in metrics.items()},
                stages=dict(stages), edits=dict(edits), exactness_passed=state['exactness_passed'])
