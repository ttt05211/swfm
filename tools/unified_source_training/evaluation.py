"""Fixed DEV64, each model uses its own trained track convention on OCC and STC."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from tools.final_semantic_evaluation import metrics
from tools.final_semantic_evaluation.predictor import Predictor
from tools.scene_ego_head.diagnostics.engine import external_forward
from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance
from tools.shared_history_experiments.full_gt import engine as gt
from .data import live_variant

PROTOCOL = 'unified_track_trained_occ_stc_fixed_dev64_v1'
ROUTES = tuple(k+'_'+r for k in ('occ_gt', 'stc_gt') for r in ('transport', 'semantic_refine'))


def evaluator(stc_root, arm, *, inventory):
    expected = {p: (size, timestamp) for p, size, timestamp in inventory}
    @torch.no_grad()
    def evaluate(models, provider, source, batches, *, saved=None, save=None, stop_event=None, progress=None, **kwargs):
        if set(models) != {'gt'}: raise ValueError('single GT-body arm required')
        body = models['gt']; body.eval()
        counts = {n: metrics.empty(3) for n in ROUTES}; keys = []
        if saved is not None:
            if saved.get('protocol') != PROTOCOL or saved.get('arm') != arm or set(saved['counts']) != set(ROUTES):
                raise RuntimeError('saved track evaluation differs')
            keys = copy.deepcopy(saved['keys'])
            if saved['windows'] != len(keys) or len(keys) != len({tuple(k) for k in keys}):
                raise RuntimeError('invalid evaluation cursor')
            for name, value in saved['counts'].items():
                for field in ('confusion', 'moving'):
                    a = np.asarray(value[field])
                    if a.shape != (3, 18, 18) or a.dtype.kind not in 'iu' or np.any(a < 0):
                        raise RuntimeError('invalid integer evaluation counts')
                    counts[name][field] = a.astype(np.int64)
                if (np.any(counts[name]['confusion'].sum((1, 2)) != len(keys)*np.prod(provider.pcfg.grid.shape_hwd))
                        or np.any(counts[name]['moving'] > counts[name]['confusion'])):
                    raise RuntimeError('incomplete evaluation window counts')
        def snapshot():
            return dict(protocol=PROTOCOL, arm=arm, windows=len(keys), keys=copy.deepcopy(keys),
                counts={n: {k: a.tolist() for k, a in c.items()} for n, c in counts.items()})
        predictor = Predictor(body, provider.pcfg, provider.device, min(provider.workers, 4))
        tick = time.perf_counter(); offset = 0
        try:
            with deterministic_entrance():
                for rows in batches:
                    for old in rows:
                        key = [str(old[0]['scene_name']), str(old[0]['t0_token'])]; offset += 1
                        if offset <= len(keys):
                            if keys[offset-1] != key: raise RuntimeError('evaluation prefix changed')
                            continue
                        if stop_event is not None and stop_event.is_set(): raise InterruptedError('saved whole DEV window')
                        raw = old[1]; labels = raw['history_occ']; camera = []
                        for token in old[0]['history_tokens'][-4:]:
                            path = Path(stc_root)/key[0]/str(token)/'labels.npz'; st = path.stat()
                            if expected.get(str(path.resolve())) != (st.st_size, st.st_mtime_ns):
                                raise RuntimeError('STC file changed')
                            with np.load(path, allow_pickle=False) as z: a = np.asarray(z['semantics'])
                            if a.shape != labels.shape[1:] or a.dtype.kind not in 'iu' or np.any((a < 0) | (a > 17)):
                                raise ValueError('invalid full STC history')
                            camera.append(a.astype(np.uint8, copy=True))
                        times = np.asarray([source.nusc.get('sample', str(t))['timestamp']/1e6
                            for t in old[0]['history_tokens'][-4:]])
                        predictions = {}
                        for kind in ('occ_gt', 'stc_gt'):
                            historical = dict(history_occ=labels if kind == 'occ_gt' else np.stack(camera),
                                history_observed=raw['history_observed'] if kind == 'occ_gt' else np.ones_like(camera, bool),
                                history_poses=raw['history_poses'])
                            row = predictor.prepare(old[0], historical, times=times)
                            row = live_variant(row, predictor.provider, predictor.history, times, arm)
                            # Default rebuild_pred_backgrounds intentionally keeps original
                            # two-frame Strong for BOTH models; no new Strong algorithm.
                            fw = external_forward(body, predictor.provider, row, raw['future_poses'])
                            for name, dense in predictor.dense(fw).items(): predictions[kind+'_'+name] = dense
                            del fw
                        # Legacy loader may pre-read target arrays, but no target field
                        # is passed into either historical preparation/forecast above.
                        target = raw['future_gt_occ']; w = gt.window_from_record(old[0])
                        support = gt.gt_moving_support_sequence(source.nusc, w.t0_token, w.future_tokens,
                            tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
                        moving = gt.moving_support_masks(support, target.shape[1:])
                        delta = {n: metrics.empty(3) for n in ROUTES}
                        for n in ROUTES:
                            for i, h in enumerate((1, 3, 5)): metrics.add(delta[n], i, predictions[n][h], target[h], moving[h])
                        for n in ROUTES:
                            for k in ('confusion', 'moving'): counts[n][k] += delta[n][k]
                        keys.append(key)
                        if len(keys) % 8 == 0 and save: save(snapshot())
                        if len(keys) % 16 == 0:
                            print(f'TRACK_TRAIN_DEV64 arm={arm} {len(keys)}/64', flush=True)
                            if progress: progress(dict(event='track_dev64', arm=arm, windows=len(keys)))
        finally:
            predictor.close()
            if save: save(snapshot())
        if offset < len(keys) or len(keys) != 64: raise RuntimeError('exact complete DEV64 required')
        reports = {n: metrics.report(c, (1., 2., 3.)) for n, c in counts.items()}
        flat = {n: dict(**v['average'], per_horizon=v['per_horizon']) for n, v in reports.items()}
        return dict(protocol=PROTOCOL, track_arm=arm, windows=64, variants={'gt': flat}, trajectory={},
            seconds=time.perf_counter()-tick, scoring='full grid, GT ego conditioned; STC all-known, NO mask/alignment')
    return evaluate
