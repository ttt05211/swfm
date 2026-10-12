"""Original network/readout/FP64 operations; staged inputs and ordered overlays."""
from contextlib import contextmanager
import sys
import time
import numpy as np
import torch
from tools.joint_occ_experiments.fast_autograd import compose_sparse
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.shared_history_experiments.refine import engine
from .staging import Constants, Inverses, pack_row, Scene, memory

BACKENDS=('reference','staged_inputs','staged_sync','staged_compose')


def overlay_values(local,alpha,mask):
    value=local*(1-alpha[None])
    return torch.where(mask[:,None],value+alpha[None],value)


class OrderedOverlay(torch.autograd.Function):
    """Only forward elementwise FP32 math compiled; original reverse VJP order."""
    @staticmethod
    def forward(ctx,background,operations,values,masks,*alphas):
        result=background.clone(); previous=[]
        for (_,ids,cls,_),alpha in zip(operations,alphas):
            local=result[:,ids]; previous.append(local)
            result[:,ids]=values(local,alpha,masks[cls])
        ctx.operations=operations;ctx.save_for_backward(*alphas,*previous)
        return result
    @staticmethod
    def backward(ctx,upstream):
        count=len(ctx.operations);alphas,previous=ctx.saved_tensors[:count],ctx.saved_tensors[count:]
        gradient=upstream.clone(); derivatives=[None]*count
        for i in range(count-1,-1,-1):
            _,ids,cls,_=ctx.operations[i];local=gradient[:,ids]
            derivatives[i]=-(local*previous[i]).sum(0)+local[cls]
            gradient[:,ids]=local*(1-alphas[i][None])
        return (gradient,None,None,None,*derivatives)


class Pipeline:
    def __init__(self,backend):
        self.backend=backend;self.constants=Constants();self.values=None
    def compose(self,background,operations,alphas,c):
        if self.backend!='staged_compose': return compose_sparse(background,operations,alphas)
        if not operations:return background
        if any(op[0]!='overlay' for op in operations):raise ValueError('source overlay only; no repair action rules')
        if self.values is None:
            if background.device.type=='cuda':
                if sys.platform=='win32':raise RuntimeError('compiled overlay requires Linux/Triton; not validated on Windows')
                self.values=torch.compile(overlay_values,dynamic=True,fullgraph=True)
            else:self.values=overlay_values  # CPU is mathematical fallback, NOT compiled-speed evidence
        masks={cls:c['class_masks'][cls] for cls in {op[2] for op in operations}}
        return OrderedOverlay.apply(background,tuple(operations),self.values,masks,*alphas)


def forward(model,provider,row,pipeline):
    record,raw,ego,history=row;state={**history[0]['state']}
    state['column_backgrounds']=raw['_column_causal_preparation']['prepared_state']['column_backgrounds']
    if len(state['current'])!=len(record['source_class_id']) or not np.array_equal(
            record['source_class_id'],[int(s['class_id']) for s in state['current']]):raise RuntimeError('source identity differs')
    current=np.asarray(state['current_pose'])
    xyz=np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1,3)
    native=(xyz-current[:3,3]) @ current[:3,:3]
    if not np.allclose(native[:,:2],record['source_centroid_xy_t0_m'],atol=2e-4,rtol=0):raise RuntimeError('source centroid differs')
    if np.asarray(raw['future_poses']).shape!=(6,4,4) or not np.isfinite(raw['future_poses']).all():
        raise ValueError('six finite GT future ego poses required')
    inputs,lengths=pack_row(row,provider.device);c=pipeline.constants.get(provider.pcfg.grid,provider.device)
    inverse=Inverses(pipeline.backend in ('staged_sync','staged_compose'))
    with torch.autocast(device_type=provider.device.type,dtype=torch.bfloat16,enabled=provider.device.type=='cuda'):
        field=model.transport.history_encoder(inputs['semantics'],inputs['observed'])
        reads={k:inputs['reads/'+k] for k in history[0]['shared_reads']}
        output=model.transport(field,reads,inputs['ego_history'][0,:,-1],inputs['classes'],inputs['kta'])
        prediction=model.ego(inputs['ego_history'],inputs['commands'])['se2'][0]
        lifted=model.columns.lift_history(field)
    poses=inputs['future_poses']
    targets=source_targets(output,{**record,'anchors_xy_t0_m':inputs['anchors']},inputs['current_pose'],inputs['source_z'])
    yaws=effective_yaw(output,[int(s['class_id']) for s in state['current']])
    tick=time.perf_counter();scene=Scene(inputs,lengths,state,provider.pcfg.grid,c,inverse,pipeline.compose)
    prior=scene.render(poses,targets,yaws);prior_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    evidence=memory(model.columns,lifted,output,history,raw,targets,yaws,poses,provider.pcfg.grid,inputs,c,inverse)
    memory_seconds=time.perf_counter()-tick
    tick=time.perf_counter();logits=model.columns(evidence,prior,range(6),scene.shape)
    inverse.validate()  # fail BEFORE loss/backward/Adam; safety checks are NOT removed
    return dict(output=output,ego=prediction,ego_target=ego[2],state=state,field=field,
        poses=poses,targets=targets,yaws=yaws,prior=prior,memory=evidence,logits=logits,shape=scene.shape,
        _staging_owner=inputs,stages=dict(transport_prior=prior_seconds,
        aligned_feature_memory=memory_seconds,semantic_refiner=time.perf_counter()-tick))


@contextmanager
def execution(backend,cache=None):
    if backend not in BACKENDS:raise ValueError('unknown staged backend')
    if backend=='reference':yield;return
    pipeline=cache.rows.get('pipeline') if cache is not None else None
    if pipeline is None:
        pipeline=Pipeline(backend)
        if cache is not None:cache.rows['pipeline']=pipeline
    original=engine.forward_window
    try:
        engine.forward_window=lambda model,provider,row:forward(model,provider,row,pipeline)
        yield
    finally:engine.forward_window=original
