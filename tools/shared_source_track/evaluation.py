"""Explicit shared-track OCC/STC evaluation, no imported-module rebinding."""
import copy
from pathlib import Path
import numpy as np
import torch
from tools.perception_robustness.evaluation import ROUTES
from tools.shared_history_experiments.full_gt import engine as gt
from .adapter import historical_row, forward_conditioned

PROTOCOL = 'shared_pinned_track_occ_stc_GTego_remaining64_v1'


def evaluator(stc_root, *, inventory=None):
    expected = {str(Path(p).resolve()): (size, stamp) for p, size, stamp in inventory or []}
    @torch.no_grad()
    def evaluate(models, provider, source, batches, *, saved=None, save=None, stop_event=None, **kwargs):
        if set(models) != {'gt'}: raise ValueError('GT body only')
        model = models['gt']; model.eval()
        metrics = {name: gt.Metrics() for name in ROUTES}; keys = []
        if saved:
            if saved['protocol'] != PROTOCOL or saved['windows'] != len(saved['keys']) or set(saved['metrics']) != set(metrics):
                raise RuntimeError('shared-track evaluation resume differs')
            keys = copy.deepcopy(saved['keys'])
            for name, fields in saved['metrics'].items():
                if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric fields differ')
                for field, value in fields.items():
                    array = np.asarray(value)
                    if array.dtype.kind not in 'iu' or array.shape != getattr(metrics[name], field).shape or np.any(array < 0):
                        raise RuntimeError('invalid saved integer counts')
                    setattr(metrics[name], field, array.astype(np.int64))
        def snapshot():
            return dict(protocol=PROTOCOL, windows=len(keys), keys=copy.deepcopy(keys),
                metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n, m in metrics.items()})
        offset = 0
        try:
            for rows in batches:
                for row in rows:
                    key = [str(row[0]['scene_name']), str(row[0]['t0_token'])]; offset += 1
                    if offset <= len(keys):
                        if keys[offset-1] != key: raise RuntimeError('evaluation prefix differs')
                        continue
                    if stop_event is not None and stop_event.is_set(): raise InterruptedError('saved complete window')
                    frames = []
                    for token in row[0]['history_tokens'][-4:]:
                        path = (Path(stc_root)/key[0]/str(token)/'labels.npz').resolve()
                        if expected and expected.get(str(path)) != (path.stat().st_size, path.stat().st_mtime_ns):
                            raise RuntimeError('STC history changed')
                        with np.load(path, allow_pickle=False) as payload:
                            values = payload['semantics']
                            if values.shape != row[1]['history_occ'].shape[1:] or values.dtype.kind not in 'iu' or np.any((values < 0) | (values > 17)):
                                raise ValueError('invalid released STC semantics')
                            frames.append(np.array(values, np.uint8, copy=True))
                    historical = dict(history_occ=np.stack(frames), history_observed=np.ones_like(frames, dtype=bool),
                                      history_poses=row[1]['history_poses'])
                    stc = historical_row(provider, row[0], historical, row[2][0],
                                         times=row[1]['_shared_source_track'].times)
                    predictions = {}
                    for kind, item in (('occ_gt', row), ('stc_gt', stc)):
                        fw = (gt.forward_window(model, provider, item) if kind == 'occ_gt' else
                              forward_conditioned(model, provider, item, row[1]['future_poses']))
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
                    if len(keys) % 8 == 0 and save: save(snapshot())
                    if len(keys) % 16 == 0: print(f'SHARED_TRACK_EVAL local_windows={len(keys)}', flush=True)
        finally:
            if save: save(snapshot())
        if offset < len(keys): raise RuntimeError('saved evaluation cursor exceeds population')
        return dict(windows=len(keys), variants={'gt': {n: m.compute() for n, m in metrics.items()}})
    return evaluate
