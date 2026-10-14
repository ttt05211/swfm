"""Original GTepoch2 versus adapted model, same remaining64 and real inputs."""
import copy
import numpy as np
import torch
from tools.shared_history_experiments.full_gt import engine as gt
from tools.final_semantic_evaluation.sources import window
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from .data import ready

PROTOCOL = 'real_STC_adaptation_original_vs_adapt_remaining64_v1'
ROUTES = tuple(f'{m}_{r}' for m in ('occ_gt', 'stc_gt') for r in ('transport', 'semantic_refine'))


@torch.no_grad()
def evaluate(models, reader, records, *, saved=None, save=None, event=None):
    for m in models.values(): m.eval()
    if set(models) != {'original', 'adapted'}: raise ValueError('original/adapted pair required')
    metrics = {arm: {route: Metrics() for route in ROUTES} for arm in models}
    keys = []
    if saved:
        if saved['protocol'] != PROTOCOL or set(saved['metrics']) != set(models): raise RuntimeError('eval protocol differs')
        keys = copy.deepcopy(saved['keys'])
        if len(keys) > len(records) or len({tuple(k) for k in keys}) != len(keys): raise RuntimeError('invalid eval cursor')
        for arm, routes in saved['metrics'].items():
            if set(routes) != set(ROUTES): raise RuntimeError('eval routes differ')
            for route, fields in routes.items():
                if set(fields) != set(vars(metrics[arm][route])): raise RuntimeError('metric schema differs')
                for name, value in fields.items():
                    a = np.asarray(value); dest = getattr(metrics[arm][route], name)
                    if a.shape != dest.shape or a.dtype.kind not in 'iu' or np.any(a < 0): raise RuntimeError('invalid integer counts')
                    dest[:] = a
    def snapshot():
        return dict(protocol=PROTOCOL, keys=copy.deepcopy(keys), metrics={arm: {r: {k: v.tolist() for k, v in vars(m).items()}
            for r, m in routes.items()} for arm, routes in metrics.items()})
    done = len(keys)
    try:
        for index, rec in enumerate(records):
            key = [str(rec['scene_name']), str(rec['t0_token'])]
            if index < done:
                if keys[index] != key: raise RuntimeError('evaluation prefix changed')
                continue
            if event is not None and event.is_set(): break
            predictions = {arm: {} for arm in models}
            for kind, name in (('occ', 'occ_gt'), ('stc', 'stc_gt')):
                row = ready(reader.predictor.provider, reader.conditioned(rec, kind))
                for arm, m in models.items():
                    fw = gt.forward_window(m, reader.predictor.provider, row)
                    for route, tensor in (('transport', fw['prior']), ('semantic_refine', fw['logits'])):
                        predictions[arm][name+'_'+route] = tensor.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                    del fw
            target = reader.inputs.targets(window(rec), (1, 3, 5))
            p = reader.predictor.provider
            moving = reader.inputs.moving(window(rec), (1, 3, 5), (1., 2., 3.), p.pcfg.grid, p.workers)
            delta = {arm: {r: Metrics() for r in ROUTES} for arm in models}
            for arm in models:
                for route in ROUTES:
                    for i, h in enumerate((1, 3, 5)): delta[arm][route].update(i, predictions[arm][route][h], target[i], moving[i])
            for arm in models:
                for route in ROUTES:
                    for name, value in vars(delta[arm][route]).items(): getattr(metrics[arm][route], name)[:] += value
            keys.append(key)
            if save: save(snapshot())
            if len(keys) % 8 == 0: print(f'REAL_STC_EVAL {len(keys)}/{len(records)}', flush=True)
    finally:
        if save: save(snapshot())
    return dict(windows=len(keys), keys=keys,
                metrics={arm: {r: m.compute() for r, m in routes.items()} for arm, routes in metrics.items()})
