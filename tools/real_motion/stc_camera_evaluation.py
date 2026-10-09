"""Atomic four-setting windows; integer recovery, no partial-setting commits."""
from collections import defaultdict
import time

import numpy as np

from real_motion.stc_camera_protocol import SETTINGS
from real_motion.waymo_i2world import WaymoMetrics, fingerprint


def restore(saved, contract, voxel_count):
    value = dict(saved); digest = value.pop('fingerprint', None)
    if digest != fingerprint(value) or value.get('contract_fingerprint') != fingerprint(contract):
        raise RuntimeError('STC resume contract/fingerprint changed')
    completed = value.get('completed_windows')
    if type(completed) is not int or not 0 <= completed <= contract['windows'] or set(value['counts']) != set(SETTINGS):
        raise ValueError('invalid STC recovery cursor/settings')
    for counts in value['counts'].values():
        a = np.asarray(counts)
        if (a.dtype.kind not in 'ui' or a.shape != (3, 18, 18) or (a < 0).any()
                or not np.all(a.sum((1, 2)) == completed * voxel_count)):
            raise ValueError('invalid STC integer counts')
    if completed and set(value.get('verified_settings', [])) != set(SETTINGS):
        raise ValueError('missing STC live exactness gate')
    if any(not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in value['stage_seconds'].values()):
        raise ValueError('invalid timing state')
    return value


def evaluate(source, selected, predict, contract, *, saved=None, save=None,
             progress=None, stop_event=None, checkpoint_every=8):
    if checkpoint_every < 1 or len(selected) != contract['windows']:
        raise ValueError('invalid evaluation population/checkpoint interval')
    state = (dict(completed_windows=0, counts={k: WaymoMetrics().counts.tolist() for k in SETTINGS},
                  stage_seconds={}, verified_settings=[]) if saved is None else
             restore(saved, contract, int(np.prod(source.shape))))
    meters = {k: WaymoMetrics(v, state['completed_windows']) for k, v in state['counts'].items()}
    stages = defaultdict(float, state['stage_seconds']); checked = set()
    def persist():
        state.update(counts={k: v.counts.tolist() for k, v in meters.items()}, stage_seconds=dict(stages),
                     contract_fingerprint=fingerprint(contract))
        snapshot = dict(state); snapshot['fingerprint'] = fingerprint(snapshot)
        if save:
            save(snapshot)
        return snapshot
    persist()
    try:
        for w in selected[state['completed_windows']:]:
            if stop_event is not None and stop_event.is_set():
                break
            start = time.perf_counter(); predictions = {}; local = {}
            for setting in SETTINGS:
                tick = time.perf_counter()
                rec, raw = source.prediction_inputs(w, setting)
                dense, details = predict(rec, raw, verify=setting not in checked)
                if len(dense) != 6 or any(np.asarray(a).shape != source.shape for a in dense):
                    raise RuntimeError('ALL four settings must finish SIX predictions before future GT access')
                checked.add(setting); predictions[setting] = dense
                local[setting] = time.perf_counter() - tick
                for name, value in details.items():
                    local[setting + '.' + name] = value
            # All four settings completed before target semantics are loaded.
            targets = source.metric_targets(w)
            delta = {k: WaymoMetrics() for k in SETTINGS}
            for k in SETTINGS:
                delta[k].add(predictions[k], targets)
            for k in SETTINGS:
                meters[k].counts += delta[k].counts; meters[k].windows += 1
            local['four_setting_window'] = time.perf_counter() - start
            for k, v in local.items():
                stages[k] += v
            state['completed_windows'] += 1; state['verified_settings'] = list(SETTINGS)
            if progress:
                progress(dict(window=state['completed_windows'], windows=len(selected), scene=w.scene, t0=w.t0,
                              seconds=local['four_setting_window'], stages=local))
            if state['completed_windows'] % checkpoint_every == 0:
                persist()
    finally:
        persist()
    return dict(status='complete' if state['completed_windows'] == len(selected) else 'stopped',
                completed_windows=state['completed_windows'], reports={k: v.report() for k, v in meters.items()},
                stage_seconds=dict(stages), verified_settings=state['verified_settings'])
