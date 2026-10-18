"""Same64 learned/KTA readouts; all six forecasts precede target statistics."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tools.shared_history_experiments.full_gt import engine as gt
from tools.source_scene_stage2.adapter import DirectPredictor
from tools.source_scene_stage2.history import DirectHistory
from tools.source_motion_audit.statistics import box_motion_rows, source_summary
from tools.joint_occ_experiments.soft_geometry import source_targets
from .adapter import forward_window, render_future

PROTOCOL = 'stage2_kta_query_same64_learned_and_KTA_v1'
KINDS = ('occ_gt', 'stc_gt')
NAMES = tuple(k+'/'+m+'/'+r for k in KINDS for m in ('learned', 'KTA_future') for r in ('transport', 'semantic_refine'))


def restore(saved):
    metrics = {name: gt.Metrics() for name in NAMES}; keys = []; sources = {k: [] for k in KINDS}
    if saved:
        if saved.get('protocol') != PROTOCOL or saved['windows'] != len(saved['keys']) or set(saved['metrics']) != set(metrics):
            raise RuntimeError('not the same KTA-query evaluation')
        keys = copy.deepcopy(saved['keys']); sources = copy.deepcopy(saved['sources'])
        for name, fields in saved['metrics'].items():
            if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric fields differ')
            for f, v in fields.items():
                a = np.asarray(v); dest = getattr(metrics[name], f)
                if a.shape != dest.shape or a.dtype.kind not in 'iu' or np.any(a < 0): raise RuntimeError('invalid integer metric')
                dest[:] = a
    return metrics, keys, sources


@torch.no_grad()
def evaluate(model, provider, source, rows, *, stc_root, saved=None, save=None, stop_event=None):
    metrics, keys, sources = restore(saved); model.eval(); offset = 0
    adapter = SimpleNamespace(provider=provider, history=DirectHistory(provider, ram_mib=0))
    def snapshot():
        return dict(protocol=PROTOCOL, windows=len(keys), keys=copy.deepcopy(keys), sources=copy.deepcopy(sources),
            metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n, m in metrics.items()})
    try:
        for group in rows:
            for row in group:
                key = [str(row[0]['scene_name']), str(row[0]['t0_token'])]; offset += 1
                if offset <= len(keys):
                    if keys[offset-1] != key: raise RuntimeError('evaluation population changed')
                    continue
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('complete evaluation window saved')
                frames = []
                for token in row[0]['history_tokens'][-4:]:
                    with np.load(Path(stc_root)/key[0]/str(token)/'labels.npz', allow_pickle=False) as z:
                        sem = z['semantics']
                        if sem.shape != row[1]['history_occ'].shape[1:] or sem.dtype.kind not in 'iu' or np.any((sem < 0)|(sem > 17)):
                            raise ValueError('invalid released STC semantics')
                        frames.append(np.array(sem, np.uint8, copy=True))
                historical = dict(history_occ=np.stack(frames), history_observed=np.ones_like(frames, bool), history_poses=row[1]['history_poses'])
                stc = DirectPredictor.prepare(adapter, row[0], historical, times=row[1]['_shared_source_track'].times)
                stc = stc[0], stc[1], (row[2][0], None, None), stc[3]
                predictions = {}; diagnostics = {}; items = dict(occ_gt=row, stc_gt=stc)
                for kind, item in items.items():
                    fw = forward_window(model, provider, item, poses=row[1]['future_poses'], fresh=True)
                    for mode in ('learned', 'KTA_future'):
                        alt = fw if mode == 'learned' else render_future(model, provider, item, fw, mode)
                        for name, value in (('transport', alt['prior']), ('semantic_refine', alt['logits'])):
                            predictions[kind+'/'+mode+'/'+name] = value.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                        del alt
                    record = item[0]; state = fw['state']; out = fw['output']
                    from tools.source_motion_audit.engine import intervention
                    kta = source_targets(intervention(out, 'KTA_future'), record, state['current_pose'], state['source_z_t0'])
                    center = np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1, 3)
                    diagnostics[kind] = dict(residual_xy_m=out['residual_xy_m'].float().cpu().numpy(),
                        effective_future_yaw_rad=fw['yaws'].float().cpu().numpy(),
                        learned_world_displacement=fw['targets'].cpu().numpy()-center[:, None],
                        KTA_world_displacement=kta.cpu().numpy()-center[:, None])
                    del fw
                target = row[1]['future_gt_occ']; w = gt.window_from_record(row[0])
                support = gt.gt_moving_support_sequence(source.nusc, w.t0_token, w.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
                moving = gt.moving_support_masks(support, target.shape[1:])
                delta = {n: gt.Metrics() for n in NAMES}
                for name in NAMES:
                    for i, h in enumerate((1, 3, 5)): delta[name].update(i, predictions[name][h], target[h], moving[h])
                local_sources = {k: box_motion_rows(items[k], diagnostics[k], source.nusc, provider.pcfg.grid) for k in KINDS}
                for name in NAMES:
                    for f, v in vars(delta[name]).items(): getattr(metrics[name], f)[:] += v
                for kind in KINDS: sources[kind].extend(local_sources[kind])
                keys.append(key)
                if save and len(keys) % 8 == 0: save(snapshot())
                if len(keys) % 16 == 0: print(f'KTA_QUERY_EVAL {len(keys)}/64', flush=True)
    finally:
        if save: save(snapshot())
    if offset < len(keys): raise RuntimeError('evaluation cursor exceeds population')
    reports = lambda mode: {k+'_'+r: metrics[k+'/'+mode+'/'+r].compute() for k in KINDS for r in ('transport', 'semantic_refine')}
    return dict(windows=len(keys), keys=keys, variants=reports('learned'), kta_variants=reports('KTA_future'),
        source_motion={k: source_summary(v) for k, v in sources.items()}, known='OCC and STC dense_valid',
        scoring='full grid; GT ego; fixed development; not FPS')
