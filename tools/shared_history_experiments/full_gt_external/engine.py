"""External poses only; same full-grid semantic/occupancy counts, no Moving reads."""
import copy
import time
import numpy as np
import torch
from tools.scene_ego_head.diagnostics.engine import external_forward
from tools.shared_history_experiments.full_gt.engine import Metrics

PROTOCOL = 'full_gt_external_all_val_readonly_v1'
READOUTS = ('transport', 'semantic_refine')


def report(meter):
    occupancy = meter.iou(meter.oi, meter.ou)
    semantic = meter.iou(meter.si, meter.su)
    means = np.nanmean(semantic, axis=1)
    return dict(IoU=float(np.nanmean(occupancy)), mIoU=float(np.nanmean(means)),
        per_horizon={str(h): dict(IoU=float(occupancy[i]), mIoU=float(means[i]),
            semantic_per_class={str(c): float(v) for c, v in enumerate(semantic[i])})
            for i, h in enumerate((1., 2., 3.))})


@torch.no_grad()
def evaluate(models, provider, source, batches, *, planner, saved=None, save=None,
             stop_event=None, progress=None, remove_threshold=None):
    if set(models) != {'gt'}: raise ValueError('one frozen GT-trained body required')
    model = models['gt']; model.eval()
    meters = {n: Metrics() for n in READOUTS}
    keys = planner.metadata['selected_keys']
    state = dict(protocol=PROTOCOL, population=planner.metadata, keys=[])
    if saved:
        if saved.get('protocol') != PROTOCOL or saved.get('population') != planner.metadata:
            raise RuntimeError('external evaluation population/planner differs')
        state = copy.deepcopy(saved)
        if state['keys'] != keys[:len(state['keys'])] or len(state['keys']) > len(keys):
            raise RuntimeError('invalid external evaluation prefix')
        if set(state['counts']) != set(READOUTS): raise RuntimeError('readouts differ')
        for n, meter in meters.items():
            if set(state['counts'][n]) != set(vars(meter)): raise RuntimeError('metric fields differ')
            for k, old in vars(meter).items():
                v = np.asarray(state['counts'][n][k])
                if v.shape != old.shape or v.dtype.kind not in 'iu' or (v < 0).any():
                    raise RuntimeError('invalid integer metric state')
                setattr(meter, k, v.astype(np.int64))
    def persist():
        state['counts'] = {n: {k: v.tolist() for k, v in vars(m).items()} for n, m in meters.items()}
        if save: save(copy.deepcopy(state))
    offset = 0; tick = time.perf_counter()
    try:
        for rows in batches:
            for row in rows:
                key = [str(row[0]['scene_name']), str(row[0]['t0_token'])]
                if offset >= len(keys) or key != keys[offset]: raise RuntimeError('full val order differs')
                offset += 1
                if offset <= len(state['keys']): continue
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError('saved complete external val windows')
                poses = planner.poses(row[0], row[1]['history_poses'][-1])
                fw = external_forward(model, provider, row, poses)
                predicted = {n: v.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                    for n, v in (('transport', fw['prior']), ('semantic_refine', fw['logits']))}
                del fw
                # Labels are used only after all six external-conditioned predictions.
                labels = row[1]['future_gt_occ']; delta = {n: Metrics() for n in READOUTS}
                for n in READOUTS:
                    for i, h in enumerate((1, 3, 5)):
                        delta[n].update(i, predicted[n][h], labels[h], np.zeros(labels[h].shape, bool))
                for n in READOUTS:
                    for k in vars(meters[n]): getattr(meters[n], k)[:] += getattr(delta[n], k)
                state['keys'].append(key)
                if offset % 16 == 0:
                    persist()
                    print(f'FULL_VAL_EXTERNAL {offset}/{len(keys)}', flush=True)
                    if progress: progress(dict(event='full_val_external', windows=offset))
    finally: persist()
    if offset != len(keys): raise RuntimeError('incomplete full-val stream')
    return dict(windows=len(state['keys']), scenes=len({k[0] for k in keys}), population=planner.metadata,
        variants={'external': {n: report(m) for n, m in meters.items()}}, trajectory={},
        seconds=time.perf_counter()-tick, scoring='full grid; no masks/alignment; standard zero-retaining mIoU',
        Moving_evaluated=False, GT_reference_run=False, internal_ego_run=False)
