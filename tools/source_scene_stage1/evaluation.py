"""Single-GPU OCC/STC, SAME dense-valid policy, complete-window integer resume."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_source_track.adapter import SharedTrackPredictor, forward_conditioned
from tools.perception_robustness.evaluation import ROUTES
from .history import StageHistory

PROTOCOL = 'source_scene_stage1_same_dense_OCC_STC_remaining64_v1'


def restore(saved):
    metrics = {name: gt.Metrics() for name in ROUTES}; keys = []
    if saved:
        if saved.get('protocol') != PROTOCOL or saved['windows'] != len(saved['keys']) or set(saved['metrics']) != set(metrics):
            raise RuntimeError('evaluation state differs')
        keys = copy.deepcopy(saved['keys'])
        if len(set(map(tuple, keys))) != len(keys): raise RuntimeError('duplicate eval keys')
        for name, fields in saved['metrics'].items():
            if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric fields differ')
            for field, values in fields.items():
                counts = np.asarray(values); target = getattr(metrics[name], field)
                if counts.dtype.kind not in 'iu' or counts.shape != target.shape or np.any(counts < 0):
                    raise RuntimeError('invalid integer counts')
                target[:] = counts
    return metrics, keys


@torch.no_grad()
def evaluate(model, provider, source, rows, *, arm, stc_root, saved=None, save=None, stop_event=None):
    model.eval(); metrics, keys = restore(saved); offset = 0
    history = StageHistory(provider, arm=arm, ram_mib=0)
    adapter = SimpleNamespace(provider=provider, history=history)
    def snapshot():
        return dict(protocol=PROTOCOL, windows=len(keys), keys=copy.deepcopy(keys),
            metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n, m in metrics.items()})
    try:
        for group in rows:
            for row in group:
                key = [str(row[0]['scene_name']), str(row[0]['t0_token'])]; offset += 1
                if offset <= len(keys):
                    if keys[offset-1] != key: raise RuntimeError('evaluation prefix changed')
                    continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('complete eval window saved')
                frames = []
                for token in row[0]['history_tokens'][-4:]:
                    with np.load(Path(stc_root)/key[0]/str(token)/'labels.npz', allow_pickle=False) as payload:
                        values = payload['semantics']
                        if values.shape != row[1]['history_occ'].shape[1:] or values.dtype.kind not in 'iu' or np.any((values < 0) | (values > 17)):
                            raise ValueError('invalid released STC history')
                        frames.append(np.array(values, np.uint8, copy=True))
                historical = dict(history_occ=np.stack(frames), history_observed=np.ones_like(frames, dtype=bool),
                                  history_poses=row[1]['history_poses'])
                stc = SharedTrackPredictor.prepare(adapter, row[0], historical, times=row[1]['_shared_source_track'].times)
                # Same original ego-history time bytes; future conditions are
                # supplied separately to ALL future geometry/background paths.
                stc = stc[0], stc[1], (row[2][0], None, None), stc[3]
                predictions = {}
                for kind, item in (('occ_gt', row), ('stc_gt', stc)):
                    fw = gt.forward_window(model, provider, item) if kind == 'occ_gt' else forward_conditioned(
                        model, provider, item, row[1]['future_poses'])
                    for name, value in (('transport', fw['prior']), ('semantic_refine', fw['logits'])):
                        predictions[kind+'_'+name] = value.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                    del fw
                target = row[1]['future_gt_occ']; w = gt.window_from_record(row[0])
                support = gt.gt_moving_support_sequence(source.nusc, w.t0_token, w.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
                moving = gt.moving_support_masks(support, target.shape[1:])
                delta = {name: gt.Metrics() for name in ROUTES}
                for name in ROUTES:
                    for i, h in enumerate((1, 3, 5)): delta[name].update(i, predictions[name][h], target[h], moving[h])
                for name in ROUTES:
                    for field, value in vars(delta[name]).items(): getattr(metrics[name], field)[:] += value
                keys.append(key)
                if save and len(keys) % 8 == 0: save(snapshot())
                if len(keys) % 16 == 0: print(f'SCENE_READER_EVAL {arm} {len(keys)}/64', flush=True)
    finally:
        if save: save(snapshot())
    if offset < len(keys): raise RuntimeError('saved eval cursor exceeds population')
    return dict(windows=len(keys), keys=keys, variants={n: m.compute() for n, m in metrics.items()},
                known='OCC and STC dense_valid', scoring='full grid; GT future ego; not an independent test')
