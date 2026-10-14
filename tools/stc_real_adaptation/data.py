"""Real frame reads and pure-history geometry reuse; no simulated corruption."""
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
import numpy as np
from real_motion.causal_geometry_cache import array_bytes
from tools.final_semantic_evaluation.sources import window
from tools.stc_noise_execution.pipeline import finish_backgrounds  # Strong-only helper, no augmentation call


def validate_chain(nusc, record):
    w = window(record); tokens = w.history_tokens+w.future_tokens
    if len(tokens) != 10 or len(set(tokens)) != 10 or tokens[3] != w.t0_token:
        raise ValueError('exact unique FOUR history -> SIX future sample tokens required')
    samples = [nusc.get('sample', t) for t in tokens]
    if len({s['scene_token'] for s in samples}) != 1:
        raise ValueError('history/target window crosses a scene boundary')
    actual_scene = nusc.get('scene', samples[0]['scene_token'])['name']
    if actual_scene != w.scene_name: raise ValueError('record scene differs from sample scene')
    for left, right, token, previous in zip(samples, samples[1:], tokens[1:], tokens[:-1]):
        dt = (right['timestamp']-left['timestamp'])/1e6
        if left['next'] != token or right['prev'] != previous or not .25 <= dt <= .75:
            raise ValueError('non-contiguous or wrong-cadence history/target window')
    return dict(key=[w.scene_name, w.t0_token],
        actual_future_times_s=[(s['timestamp']-samples[3]['timestamp'])/1e6 for s in samples[4:]])


def schedule(records, seed, steps=500):
    from tools.real_motion.height_field_screen_common import epoch_groups
    if not records: raise ValueError('empty real-STC adaptation population')
    groups, positions, cycle = [], [], 0
    while len(groups) < steps:
        epoch = epoch_groups(records, seed, cycle, 4, 128)
        for index, group in enumerate(epoch):
            if len(groups) == steps: break
            groups.append(group); positions.append(dict(cycle=cycle, batch=index+1, cycle_batches=len(epoch)))
        cycle += 1
    return groups, positions


class HistoryRows:
    """CPU-owned bounded LRU of actual observations + deterministic geometry only."""
    def __init__(self, inputs, predictor, ram_mib=512):
        self.inputs, self.predictor = inputs, predictor
        self.limit = ram_mib*2**20; self.cache = OrderedDict(); self.bytes = 0
        self.lock = Lock(); self.io_lock = Lock(); self.hits = self.misses = 0

    def prepare(self, record, kind='stc'):
        key = (kind, str(record['scene_name']), str(record['t0_token']))
        with self.lock:
            if key in self.cache:
                self.hits += 1; self.cache.move_to_end(key); return self.cache[key][0]
        # NuscInputs owns an OrderedDict frame LRU; serialize short reads, not geometry work.
        with self.io_lock: _, raw, times = self.inputs.history(window(record), kind)
        row = self.predictor.prepare(record, raw, times=times)
        size = array_bytes(row)
        with self.lock:
            self.misses += 1
            if size <= self.limit:
                while self.cache and self.bytes+size > self.limit:
                    _, (_, old) = self.cache.popitem(last=False); self.bytes -= old
                if key in self.cache:
                    _, old = self.cache.pop(key); self.bytes -= old
                self.cache[key] = row, size; self.bytes += size
        return row

    def conditioned(self, record, kind='stc', target=False):
        rec, raw, inp, hist = self.prepare(record, kind)
        with self.io_lock:
            raw = dict(raw, future_poses=self.inputs.poses(window(record)), _robust_background_pending=True)
            # Labels are NEVER supplied to Predictor.prepare or the source matcher.
            if target: raw['future_gt_occ'] = np.stack(self.inputs.targets(window(record), range(6)))
        return rec, raw, inp, hist

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, ram_mib=self.bytes/2**20,
                    limit_mib=self.limit/2**20, learned_features_cached=False, persistent_writes=False)


def batches(reader, groups, event, workers=2):
    """One pending B4 CPU group; never CUDA/model forward in worker threads."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(); iterator = iter(groups)
        def submit():
            group = next(iterator, None)
            if group is None: return False
            pending.append([pool.submit(reader.conditioned, r, target=True) for r in group]); return True
        submit()
        try:
            while pending and not event.is_set():
                tick = time.perf_counter(); futures = pending.popleft()
                rows = [f.result() for f in futures]; wait = time.perf_counter()-tick
                submit(); yield rows, wait
        finally:
            for group in pending:
                for f in group: f.cancel()


def ready(provider, row):
    return finish_backgrounds(provider, row)
