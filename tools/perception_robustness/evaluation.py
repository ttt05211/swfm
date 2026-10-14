"""Only real OCC/STC + GT ego, same fixed DEV64; no counterfactual diagnostics."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.shared_history_experiments.full_gt import engine as gt
from tools.scene_ego_head.diagnostics.engine import external_forward
from .data import historical_row


PROTOCOL = 'robust_history_occ_stc_gt_fixed_dev64_v1'
ROUTES = ('occ_gt_transport', 'occ_gt_semantic_refine', 'stc_gt_transport', 'stc_gt_semantic_refine')


def evaluator(stc_root, *, inventory=None):
    expected = {p: (size, timestamp) for p, size, timestamp in inventory or []}
    @torch.no_grad()
    def evaluate(models, provider, source, batches, *, saved=None, save=None, stop_event=None, progress=None, **kwargs):
        if set(models) != {'gt'}: raise ValueError('GT-body experiment only')
        model = models['gt']; model.eval()
        metrics = {name: Metrics() for name in ROUTES}; keys = []
        if saved:
            if saved.get('protocol') != PROTOCOL or set(saved['metrics']) != set(metrics):
                raise RuntimeError('robustness evaluation state differs')
            keys = copy.deepcopy(saved['keys'])
            if saved['windows'] != len(keys): raise RuntimeError('invalid evaluation cursor')
            for name, fields in saved['metrics'].items():
                if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric schema differs')
                for field, value in fields.items():
                    array = np.asarray(value)
                    if (array.dtype.kind not in 'iu' or np.any(array < 0)
                            or array.shape != getattr(metrics[name], field).shape):
                        raise RuntimeError('invalid integer metric counts')
                    setattr(metrics[name], field, array.astype(np.int64))
        def snapshot():
            return dict(protocol=PROTOCOL, windows=len(keys), keys=copy.deepcopy(keys),
                metrics={name: {k: v.tolist() for k, v in vars(m).items()} for name, m in metrics.items()})
        tick, offset = time.perf_counter(), 0
        try:
            for rows in batches:
                for row in rows:
                    key = [str(row[0]['scene_name']), str(row[0]['t0_token'])]
                    offset += 1
                    if offset <= len(keys):
                        if keys[offset-1] != key: raise RuntimeError('evaluation prefix changed')
                        continue
                    if stop_event is not None and stop_event.is_set(): raise InterruptedError('saved complete DEV64 window')
                    stc_frames = []
                    for token in row[0]['history_tokens'][-4:]:
                        path = Path(stc_root)/key[0]/str(token)/'labels.npz'
                        if expected:
                            st = path.stat()
                            if expected.get(str(path.resolve())) != (st.st_size, st.st_mtime_ns):
                                raise RuntimeError('STC history changed during evaluation')
                        with np.load(path, allow_pickle=False) as z:
                            # Only semantics; deliberately never reads GT/sensor masks.
                            values = np.asarray(z['semantics'])
                            if (values.shape != row[1]['history_occ'].shape[1:] or values.dtype.kind not in 'iu'
                                    or np.any(values > 17) or np.any(values < 0)):
                                raise ValueError('invalid STC semantic grid: '+str(path))
                            stc_frames.append(np.array(values, dtype=np.uint8, copy=True))
                    historical = dict(history_occ=np.stack(stc_frames),
                        history_observed=np.ones_like(stc_frames, dtype=bool), history_poses=row[1]['history_poses'])
                    stc_row = historical_row(provider, row[0], historical, row[2][0])
                    predictions = {}
                    for name, fw in (('occ_gt', gt.forward_window(model, provider, row)),
                                     ('stc_gt', external_forward(model, provider, stc_row, row[1]['future_poses']))):
                        for route, values in (('transport', fw['prior']), ('semantic_refine', fw['logits'])):
                            predictions[name+'_'+route] = values.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                        del fw
                    # Future GT occupancy / Moving are used ONLY after both predictions.
                    target = row[1]['future_gt_occ']; w = gt.window_from_record(row[0])
                    support = gt.gt_moving_support_sequence(source.nusc, w.t0_token, w.future_tokens,
                        tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
                    moving = gt.moving_support_masks(support, target.shape[1:])
                    # Window-local counts prevent half-window publication on errors.
                    delta = {name: Metrics() for name in ROUTES}
                    for name in ROUTES:
                        for i, h in enumerate((1, 3, 5)): delta[name].update(i, predictions[name][h], target[h], moving[h])
                    for name in ROUTES:
                        for field, value in vars(delta[name]).items(): getattr(metrics[name], field)[:] += value
                    keys.append(key)
                    if len(keys) % 8 == 0 and save: save(snapshot())
                    if len(keys) % 16 == 0:
                        print(f'ROBUSTNESS_DEV64 {len(keys)}/64', flush=True)
                        if progress: progress(dict(event='robustness_dev64', windows=len(keys)))
        finally:
            if save: save(snapshot())
        if offset < len(keys): raise RuntimeError('evaluation cursor exceeds population')
        return dict(windows=len(keys), variants={'gt': {n: m.compute() for n, m in metrics.items()}},
            trajectory={}, seconds=time.perf_counter()-tick, main_readout='semantic_refine',
            scoring='same fixed DEV64, full grid, GT future ego; STC full semantics / all-known',
            no_counterfactuals=True)
    return evaluate
