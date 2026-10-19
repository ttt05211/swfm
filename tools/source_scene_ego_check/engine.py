"""Paired learned ego versus ZERO EMBEDDING OUTPUT; renderer poses never change."""
from contextlib import contextmanager
import copy
import numpy as np
import torch
from tools.shared_history_experiments.model import encode_window
from tools.source_scene_ego_conditioned.model import relative_ego_features
from tools.source_scene_ego_conditioned.adapter import render_future
from tools.shared_source_track.adapter import backgrounds
from tools.final_semantic_evaluation import metrics

PROTOCOL = 'frozen_ego_embedding_output_zero_same64_v1'
KINDS = ('occ', 'stc')
MODES = ('ego_on', 'ego_off')
READOUTS = ('transport', 'semantic_refine')
ROUTES = tuple(k+'/'+m+'/'+r for k in KINDS for m in MODES for r in READOUTS)


@contextmanager
def zero_ego_projection(model):
    """Scoped hook intercepts ONLY projection output; never changes a weight/input pose."""
    called = []
    def zero(_module, _args, value):
        called.append(True)
        return torch.zeros_like(value)
    handle = model.transport.ego_future_proj.register_forward_hook(zero)
    try: yield called
    finally: handle.remove()


def maxdiff(a, b):
    return float((a.float()-b.float()).abs().max()) if a.numel() else 0.


@torch.no_grad()
def paired_forecasts(predictor, row, poses, *, repeat=False):
    model, provider = predictor.body, predictor.provider
    rec, raw, ego, history = row; device = provider.device; state = history[0]['state']
    poses = torch.as_tensor(np.asarray(poses), device=device).double()
    ego_features = relative_ego_features(torch.as_tensor(np.asarray(state['current_pose']), device=device).double(), poses)
    # Shared HISTORICAL computation, SAME fresh background and SAME future poses for BOTH branches.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(model, raw, history, ego[0], device)
        lifted = model.columns.lift_history(field)
        kta = torch.as_tensor(np.asarray(rec['kta_displacement_xy_m']), device=device).float()
        active_embedding = model.transport.ego_future_proj((ego_features/ego_features.new_tensor([20.,20.,1.,1.])).to(field.dtype))
    state = dict(state, column_backgrounds=backgrounds(raw, state, poses, provider, raw['_shared_source_track']))
    common = dict(field=field, lifted=lifted, state=state, poses=poses, ego_features=ego_features)
    def predict(zero=False):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            if zero:
                with zero_ego_projection(model) as called:
                    out = model.transport(field, reads, times, None, kta, ego_features)
                if len(called) != (1 if len(kta) else 0): raise RuntimeError('ego output interception did not occur exactly once')
            else: out = model.transport(field, reads, times, None, kta, ego_features)
        return render_future(model, provider, row, dict(common, output=out))
    on = predict(); on_dense = predictor.dense(on)
    repeated = None
    if repeat:
        again = predict(); again_dense = predictor.dense(again)
        repeated = {r:int(np.count_nonzero(again_dense[r] != on_dense[r])) for r in READOUTS}
        repeat_logits_max = maxdiff(again['logits'], on['logits']); del again
    off = predict(True); off_dense = predictor.dense(off)
    if not torch.equal(on['poses'], off['poses']): raise RuntimeError('renderer trajectory changed')
    differences = dict(
        embedding_norm=active_embedding.float().norm(dim=-1).cpu().tolist(),
        query_delta_norm=(on['output']['future_transport_queries'].float()-off['output']['future_transport_queries'].float()).norm(dim=-1).flatten().cpu().tolist(),
        source_XY_delta_m=(on['targets'][...,:2]-off['targets'][...,:2]).norm(dim=-1).flatten().cpu().tolist(),
        source_yaw_delta_deg=((on['yaws']-off['yaws']).abs()*180/np.pi).flatten().cpu().tolist(),
        history_context_max_abs=maxdiff(on['output']['history_source_context'],off['output']['history_source_context']),
        history_yaw_max_abs=maxdiff(on['output']['history_yaw_to_t0_rad'],off['output']['history_yaw_to_t0_rad']),
        changed_voxels={r:int(np.count_nonzero(on_dense[r] != off_dense[r])) for r in READOUTS},
        source_count=len(kta), repeated_on_changed_voxels=repeated,
        repeated_on_logits_max_abs=repeat_logits_max if repeat else None,
        same_future_poses=True, shared_fresh_background=True)
    # These histories do not consume ego; record unexpected drift explicitly, do NOT waive a FAIL.
    differences['history_byte_equal'] = differences['history_context_max_abs']==0 and differences['history_yaw_max_abs']==0
    return dict(ego_on=on_dense,ego_off=off_dense), differences


def restore(saved, contract, keys, shape):
    value = dict(protocol=PROTOCOL,contract=contract,keys=[],counts={r:metrics.empty(3) for r in ROUTES},
        diagnostics={k:[] for k in KINDS})
    if saved is None: return value
    if (saved.get('protocol') != PROTOCOL or saved.get('contract') != contract or
            saved.get('keys') != keys[:len(saved['keys'])] or len(saved['keys'])>len(keys) or
            set(saved['counts']) != set(ROUTES) or set(saved['diagnostics']) != set(KINDS)):
        raise RuntimeError('only SAME complete-window audit prefix can resume')
    value = copy.deepcopy(saved); n=len(value['keys'])
    for kind in KINDS:
        if len(value['diagnostics'][kind])!=n: raise RuntimeError('diagnostic/window cursor mismatch')
    for r,c in value['counts'].items():
        if set(c)!= {'confusion','moving'}: raise RuntimeError('count fields differ')
        for k,a in c.items():
            a=np.asarray(a)
            if a.shape!=(3,18,18) or a.dtype.kind not in 'iu' or np.any(a<0): raise RuntimeError('invalid integer counts')
            c[k]=a.astype(np.int64)
        if np.any(c['confusion'].sum((1,2)) != n*int(np.prod(shape))) or np.any(c['moving']>c['confusion']):
            raise RuntimeError('count totals differ from complete-window cursor')
    return value


def distribution(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(n=0)


def report(state):
    scores={r:metrics.report(c,(1.,2.,3.)) for r,c in state['counts'].items()}
    diagnostics={}
    for kind,rows in state['diagnostics'].items():
        fields=('embedding_norm','query_delta_norm','source_XY_delta_m','source_yaw_delta_deg')
        d={f:distribution([x for row in rows for x in row[f]]) for f in fields}
        d.update(history_byte_fail_windows=sum(not r['history_byte_equal'] for r in rows),
            history_context_max_abs=max((r['history_context_max_abs'] for r in rows),default=0.),
            history_yaw_max_abs=max((r['history_yaw_max_abs'] for r in rows),default=0.),
            changed_voxels={r:sum(v['changed_voxels'][r] for v in rows) for r in READOUTS},
            repeated_on_windows=sum(r['repeated_on_changed_voxels'] is not None for r in rows),
            repeated_on_byte_fail_windows=sum(any(r['repeated_on_changed_voxels'].values()) for r in rows if r['repeated_on_changed_voxels'] is not None),
            repeated_on_max_changed_voxels=max((max(r['repeated_on_changed_voxels'].values()) for r in rows if r['repeated_on_changed_voxels'] is not None),default=0))
        diagnostics[kind]=d
    return dict(windows=len(state['keys']),reports=scores,diagnostics=diagnostics)


@torch.no_grad()
def evaluate(predictor, inputs, jobs, keys, contract, *, saved=None,save=None,stop_event=None):
    from tools.history_validity.data import policy_raw
    state=restore(saved,contract,keys,predictor.provider.pcfg.grid.shape_hwd)
    def persist():
        if save:
            v=copy.deepcopy(state)
            v['counts']={r:{k:a.tolist() for k,a in c.items()} for r,c in state['counts'].items()}
            save(v)
    try:
        for index,job in enumerate(jobs):
            if index<len(state['keys']):continue
            if stop_event is not None and stop_event.is_set():raise InterruptedError('saved complete paired window')
            predictions={}; diagnostics={}; poses=inputs.poses(job)
            for kind in KINDS:
                rec,raw,times=inputs.history(job,kind)
                row=predictor.prepare(rec,policy_raw(raw,'dense_valid'),times=times)
                paired,diagnostic=paired_forecasts(predictor,row,poses,repeat=index<8)
                predictions.update({kind+'/'+m+'/'+r:p for m,v in paired.items() for r,p in v.items()})
                diagnostics[kind]=diagnostic
            # BOTH modalities/modes/ALL six outputs before ANY future occupancy/Moving read.
            target=inputs.targets(job,(1,3,5))
            moving=inputs.moving(job,(1,3,5),(1.,2.,3.),predictor.provider.pcfg.grid,predictor.provider.workers)
            delta={r:metrics.empty(3) for r in ROUTES}
            for r,p in predictions.items():
                for i,h in enumerate((1,3,5)):metrics.add(delta[r],i,p[h],target[i],moving[i])
            for r,c in delta.items():
                for k,a in c.items():state['counts'][r][k]+=a
            for kind in KINDS:state['diagnostics'][kind].append(diagnostics[kind])
            state['keys'].append(keys[index])
            if len(state['keys'])%8==0:persist()
            if len(state['keys'])%8==0:print(f'EGO_BRANCH_CHECK {len(state["keys"])}/{len(jobs)}',flush=True)
    finally:persist()
    if state['keys']!=keys:raise RuntimeError('incomplete population')
    return report(state)
