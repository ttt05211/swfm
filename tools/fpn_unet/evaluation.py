"""One final same64 OCC/STC evaluation, four routes, no extra counterfactuals."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tools.shared_history_experiments.full_gt import engine as gt
from tools.source_scene_stage2.adapter import DirectPredictor
from tools.source_scene_stage2.history import DirectHistory
from .adapter import forward_window

PROTOCOL = 'FPN_UNet_same64_dense_valid_GT_ego_full_grid_v1'
KINDS = ('occ_gt', 'stc_gt')
NAMES = tuple(k+'_'+r for k in KINDS for r in ('transport', 'semantic_refine'))


def restore(saved):
    metrics = {n: gt.Metrics() for n in NAMES}; keys = []
    if saved:
        if saved.get('protocol') != PROTOCOL or saved['windows'] != len(saved['keys']) or set(saved['metrics']) != set(metrics):
            raise RuntimeError('not this combined candidate evaluation')
        keys = copy.deepcopy(saved['keys'])
        if len(set(map(tuple, keys))) != len(keys): raise RuntimeError('duplicate completed windows')
        for name, fields in saved['metrics'].items():
            if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric fields differ')
            for field, values in fields.items():
                array = np.asarray(values); dest = getattr(metrics[name], field)
                if array.shape != dest.shape or array.dtype.kind not in 'iu' or np.any(array < 0):
                    raise RuntimeError('invalid integer metric counts')
                dest[:] = array
    return metrics, keys


@torch.no_grad()
def evaluate(model, provider, source, rows, *, stc_root, saved=None, save=None, stop_event=None):
    metrics, keys = restore(saved); model.eval(); offset = 0
    adapter = SimpleNamespace(provider=provider, history=DirectHistory(provider, ram_mib=0))
    def snapshot():
        return dict(protocol=PROTOCOL, windows=len(keys), keys=copy.deepcopy(keys),
            metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n, m in metrics.items()})
    try:
        for group in rows:
            for row in group:
                key = [str(row[0]['scene_name']), str(row[0]['t0_token'])]; offset += 1
                if offset <= len(keys):
                    if keys[offset-1] != key: raise RuntimeError('evaluation population differs')
                    continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('complete window saved')
                frames = []
                for token in row[0]['history_tokens'][-4:]:
                    with np.load(Path(stc_root)/key[0]/str(token)/'labels.npz', allow_pickle=False) as z:
                        sem = z['semantics']
                        if sem.shape != row[1]['history_occ'].shape[1:] or sem.dtype.kind not in 'iu' or np.any((sem < 0)|(sem > 17)):
                            raise ValueError('invalid released STC semantics')
                        frames.append(np.array(sem, np.uint8, copy=True))
                historical = dict(history_occ=np.stack(frames), history_observed=np.ones_like(frames, bool),
                    history_poses=row[1]['history_poses'])
                stc = DirectPredictor.prepare(adapter, row[0], historical, times=row[1]['_shared_source_track'].times)
                stc = stc[0], stc[1], (row[2][0], None, None), stc[3]
                predictions = {}
                for kind, item in (('occ_gt', row), ('stc_gt', stc)):
                    fw = forward_window(model, provider, item, poses=row[1]['future_poses'], fresh=True)
                    for route, tensor in (('transport', fw['prior']), ('semantic_refine', fw['logits'])):
                        predictions[kind+'_'+route] = tensor.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                    del fw
                # Both modalities forecast ALL SIX futures before labels/support.
                target = row[1]['future_gt_occ']; w = gt.window_from_record(row[0])
                support = gt.gt_moving_support_sequence(source.nusc, w.t0_token, w.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
                moving = gt.moving_support_masks(support, target.shape[1:])
                for name in NAMES:
                    for i, h in enumerate((1, 3, 5)): metrics[name].update(i, predictions[name][h], target[h], moving[h])
                keys.append(key)
                if save and len(keys) % 8 == 0: save(snapshot())
                if len(keys) % 16 == 0: print(f'FPN_UNET_EVAL {len(keys)}/64', flush=True)
    finally:
        if save: save(snapshot())
    if offset < len(keys): raise RuntimeError('evaluation cursor exceeds population')
    return dict(windows=len(keys), keys=keys, variants={n: m.compute() for n, m in metrics.items()},
        scoring='same remaining64; 1/2/3s mean; full grid; dense-valid OCC/STC; GT ego; NOT FPS')
