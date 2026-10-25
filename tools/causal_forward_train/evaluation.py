"""ONE forward per modality: raw + fixed FREE -1; labels read afterwards."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tools.shared_history_experiments.full_gt import engine as gt
from tools.source_scene_stage2.history import DirectHistory
from .data import ForwardPredictor,forward_window
from .control import NAMES as RAW_NAMES
from .calibration import predict,delta,CALIBRATION

PROTOCOL = 'AdaLN_forward_singleGPU_same64_raw_and_fixed_FREEminus1_H6_v2'
NAMES = RAW_NAMES+tuple(k+'_semantic_refine_free_minus1' for k in ('occ_gt','stc_gt'))


def restore(saved):
    metrics = {n:gt.Metrics() for n in NAMES};keys = []
    if saved:
        if (saved.get('protocol') != PROTOCOL or saved.get('calibration') != CALIBRATION
                or saved['windows'] != len(saved['keys']) or set(saved['metrics']) != set(metrics)):
            raise RuntimeError('only same raw+FREE-1 complete-window evaluation can resume')
        keys = copy.deepcopy(saved['keys'])
        if len(set(map(tuple,keys))) != len(keys):raise RuntimeError('duplicate eval keys')
        for name,fields in saved['metrics'].items():
            if set(fields) != set(vars(metrics[name])):raise RuntimeError('metric fields differ')
            for field,values in fields.items():
                array = np.asarray(values);dest = getattr(metrics[name],field)
                if array.shape != dest.shape or array.dtype.kind not in 'iu' or np.any(array < 0):
                    raise RuntimeError('invalid integer eval counts')
                dest[:] = array
    return metrics,keys


@torch.no_grad()
def evaluate(model,provider,source,rows,*,stc_root,saved=None,save=None,stop_event=None):
    metrics,keys = restore(saved);model.eval();offset = 0
    adapter = SimpleNamespace(provider=provider,history=DirectHistory(provider,ram_mib=0))
    def snapshot():
        return dict(protocol=PROTOCOL,calibration=copy.deepcopy(CALIBRATION),windows=len(keys),keys=copy.deepcopy(keys),
            metrics={n:{k:v.tolist() for k,v in vars(m).items()} for n,m in metrics.items()})
    try:
        for group in rows:
            for row in group:
                key = [str(row[0]['scene_name']),str(row[0]['t0_token'])];offset += 1
                if offset <= len(keys):
                    if keys[offset-1] != key:raise RuntimeError('evaluation population differs')
                    continue
                if stop_event is not None and stop_event.is_set():raise InterruptedError('complete window saved')
                frames = []
                for token in row[0]['history_tokens'][-4:]:
                    with np.load(Path(stc_root)/key[0]/str(token)/'labels.npz',allow_pickle=False) as z:
                        sem = z['semantics']
                        if (sem.shape != row[1]['history_occ'].shape[1:] or sem.dtype.kind not in 'iu'
                                or np.any((sem < 0)|(sem > 17))):raise ValueError('invalid STC semantics')
                        frames.append(np.array(sem,np.uint8,copy=True))
                historical = dict(history_occ=np.stack(frames),history_observed=np.ones_like(frames,bool),
                    history_poses=row[1]['history_poses'])
                stc = ForwardPredictor.prepare(adapter,row[0],historical,times=row[1]['_shared_source_track'].times)
                stc = stc[0],stc[1],(row[2][0],None,None),stc[3]
                predictions = {}
                for kind,item in (('occ_gt',row),('stc_gt',stc)):
                    fw = forward_window(model,provider,item,poses=row[1]['future_poses'],fresh=True)
                    predictions[kind+'_transport'] = fw['prior'].argmax(1).reshape(6,*fw['shape']).byte().cpu().numpy()
                    for name,value in predict(fw['logits'],fw['shape']).items():predictions[kind+'_'+name] = value
                    del fw
                # Both modalities, ALL SIX readouts and BOTH calibration routes
                # are fixed BEFORE consulting future occupancy/moving labels.
                target = row[1]['future_gt_occ'];w = gt.window_from_record(row[0])
                support = gt.gt_moving_support_sequence(source.nusc,w.t0_token,w.future_tokens,
                    tuple(.5*(h+1) for h in range(6)),grid=provider.pcfg.grid,workers=provider.workers)
                moving = gt.moving_support_masks(support,target.shape[1:])
                for name in NAMES:
                    for i,h in enumerate((1,3,5)):metrics[name].update(i,predictions[name][h],target[h],moving[h])
                keys.append(key)
                if save and len(keys)%8 == 0:save(snapshot())
                if len(keys)%16 == 0:print(f'FORWARD_RAW_FREE_EVAL {len(keys)}/64',flush=True)
    finally:
        if save:save(snapshot())
    if offset < len(keys):raise RuntimeError('eval cursor exceeds population')
    variants = {n:m.compute() for n,m in metrics.items()}
    calibration_delta = {}
    for kind in ('occ_gt','stc_gt'):
        a,b = variants[kind+'_semantic_refine'],variants[kind+'_semantic_refine_free_minus1']
        calibration_delta[kind] = dict(average=delta(b,a),per_horizon={
            h:delta(b['per_horizon'][h],a['per_horizon'][h]) for h in a['per_horizon']})
    return dict(windows=len(keys),keys=keys,variants=variants,calibration=copy.deepcopy(CALIBRATION),
        free_minus1_minus_raw=calibration_delta,
        scoring='same64/full grid/dense-valid/GT ego; fixed inherited labeled calibration, no fitting; NOT FPS')
