"""Deterministic 2 STC + 2 clean GT per update; whole-scene holdout."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from tools.joint_occ_experiments import run as legacy
from tools.stc_real_adaptation.data import HistoryRows

STEPS = 500


def selection(val_records, gt_records, eval_keys, seed):
    legacy.record_keys(val_records); legacy.record_keys(gt_records)
    if len(eval_keys) != 64 or len({tuple(k) for k in eval_keys}) != 64:
        raise ValueError('exact previous unique remaining64 required')
    dev = legacy.align_records(val_records, list(map(tuple, eval_keys)))
    holdout = {r['scene_name'] for r in dev}
    val_scenes = {r['scene_name'] for r in val_records}
    if val_scenes & {r['scene_name'] for r in gt_records}:
        raise RuntimeError('original TRAIN and VAL scenes overlap')
    stc = [r for r in val_records if r['scene_name'] not in holdout]
    if not stc or not holdout or len(holdout) == len(val_scenes):
        raise RuntimeError('empty training or holdout scene population')
    # 1000 distinct clean TRAIN windows, scene-balanced; no clean VAL replay.
    shuffled = [gt_records[int(i)] for i in np.random.default_rng(seed).permutation(len(gt_records))]
    keys = legacy.select_scene_balanced_round_robin(legacy.record_keys(shuffled), STEPS*2)
    if len(keys) != STEPS*2: raise RuntimeError('at least 1000 unique original TRAIN windows required')
    clean = legacy.align_records(gt_records, keys)
    return stc, clean, dev


def schedule(stc, clean, seed):
    if not stc or len(clean) != STEPS*2: raise ValueError('mixed500 population differs')
    # Draw each STC candidate once before repeating, over a fixed local RNG stream.
    stream, cycle = [], 0
    while len(stream) < STEPS*2:
        rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 7919]))
        stream.extend(stc[int(i)] for i in rng.permutation(len(stc))); cycle += 1
    clean = [clean[int(i)] for i in np.random.default_rng(np.random.SeedSequence([seed, 104729])).permutation(len(clean))]
    groups = [[('stc', stream[2*i]), ('occ', clean[2*i]), ('stc', stream[2*i+1]), ('occ', clean[2*i+1])]
              for i in range(STEPS)]
    return groups


class MixedRows:
    def __init__(self, val_inputs, train_inputs, predictor, gt_scenes):
        self.inputs, self.predictor = val_inputs, predictor
        self.gt_scenes = set(gt_scenes)
        self.val = HistoryRows(val_inputs, predictor, ram_mib=256)
        self.train = HistoryRows(train_inputs, predictor, ram_mib=256)

    def conditioned(self, record, kind='stc', target=False):
        if record['scene_name'] in self.gt_scenes:
            if kind != 'occ': raise ValueError('original TRAIN provides clean GT only, not released STC')
            reader = self.train
        else: reader = self.val
        return reader.conditioned(record, kind, target)

    def stats(self): return dict(val=self.val.stats(), clean_TRAIN=self.train.stats())


def batches(reader, groups, event):
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = deque(); iterator = iter(groups)
        def submit():
            group = next(iterator, None)
            if group is None: return
            pending.append([pool.submit(reader.conditioned, r, kind, True) for kind, r in group])
        submit()
        try:
            while pending and not event.is_set():
                tick = time.perf_counter(); futures = pending.popleft()
                rows = [f.result() for f in futures]; wait = time.perf_counter()-tick
                submit(); yield rows, wait
        finally:
            for group in pending:
                for f in group: f.cancel()
