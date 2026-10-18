"""Same remaining64 OCC/STC; stage2 integer states cannot resume stage1."""
import copy
from pathlib import Path
import numpy as np
import torch
from tools.shared_history_experiments.full_gt import engine as gt
from tools.source_scene_stage1.evaluation import restore as old_counts, PROTOCOL as COUNTS
from tools.perception_robustness.evaluation import ROUTES
from .adapter import DirectPredictor, forward_window
from .history import DirectHistory

PROTOCOL = 'stage2_direct_history_same_dense_remaining64_v1'


@torch.no_grad()
def evaluate(model, provider, source, rows, *, stc_root, saved=None, save=None, stop_event=None):
    if saved and saved.get('protocol') != PROTOCOL: raise RuntimeError('not stage2 evaluation state')
    metrics, keys = old_counts(saved['counts'] if saved else None)
    model.eval(); offset = 0
    # No extra model/GPU context: only the stateless prepare method is used.
    from types import SimpleNamespace
    adapter = SimpleNamespace(provider=provider, history=DirectHistory(provider, ram_mib=0))
    def snapshot():
        return dict(protocol=PROTOCOL, counts=dict(protocol=COUNTS, windows=len(keys), keys=copy.deepcopy(keys),
            metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n, m in metrics.items()}))
    try:
        for group in rows:
            for row in group:
                key = [str(row[0]['scene_name']), str(row[0]['t0_token'])]; offset += 1
                if offset <= len(keys):
                    if keys[offset-1] != key: raise RuntimeError('evaluation prefix changed')
                    continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('completed eval window saved')
                frames = []
                for token in row[0]['history_tokens'][-4:]:
                    with np.load(Path(stc_root)/key[0]/str(token)/'labels.npz', allow_pickle=False) as payload:
                        values = payload['semantics']
                        if values.shape != row[1]['history_occ'].shape[1:] or values.dtype.kind not in 'iu' or np.any((values < 0)|(values > 17)):
                            raise ValueError('invalid released STC history')
                        frames.append(np.array(values, np.uint8, copy=True))
                historical = dict(history_occ=np.stack(frames), history_observed=np.ones_like(frames, dtype=bool),
                                  history_poses=row[1]['history_poses'])
                stc = DirectPredictor.prepare(adapter, row[0], historical, times=row[1]['_shared_source_track'].times)
                stc = stc[0], stc[1], (row[2][0], None, None), stc[3]
                predictions = {}
                for kind, item in (('occ_gt', row), ('stc_gt', stc)):
                    fw = forward_window(model, provider, item, poses=row[1]['future_poses'], fresh=True)
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
                if len(keys) % 16 == 0: print(f'SCENE_STAGE2_EVAL {len(keys)}/64', flush=True)
    finally:
        if save: save(snapshot())
    if offset < len(keys): raise RuntimeError('evaluation cursor exceeds population')
    return dict(windows=len(keys), keys=keys, variants={n: m.compute() for n, m in metrics.items()},
                known='OCC and STC dense_valid', scoring='full grid; GT ego; development not independent test')
