"""Evaluate ONLY the new frozen model; reuse validated original/joint SAME64."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import threading
import numpy as np
import torch
from real_motion.runtime_config import load_runtime_config,make_prepare_config
from tools.coverage_projection.run import ROOT,sha,write_json
from tools.coverage_projection.model import from_spec
from tools.coverage_projection.adapter import forward_window
from tools.coverage_projection.eval_inputs import prepare_evaluation_row
from tools.coverage_projection.evaluate import evaluation_plan,counts_from_json
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.fpn_unet_eval.common import DenseInputs
from tools.final_semantic_evaluation import metrics
from tools.final_semantic_evaluation.sources import window
from .control import reference,validate_completed


def copy_once(source,target):
    if target.exists():
        if sha(source) != sha(target):raise ValueError('existing visual control differs: '+str(target))
    else:shutil.copyfile(source,target)


def run(a,stop):
    device=torch.device(a.device);torch.set_num_threads(1)
    path=a.out_dir/'epoch_0001.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
    c=validate_completed(saved);source=reference(c['joint_dir']);joint=source['contract']
    for name,key in (('checkpoint_sha256','joint_checkpoint_sha256'),('contract_sha256','joint_contract_sha256'),
            ('evaluation_sha256','joint_evaluation_sha256')):
        if source[name] != c[key]:raise ValueError('matched joint evidence changed: '+name)
    if c['eval_keys'] != joint['eval_keys'] or c['model'] != joint['model']:raise ValueError('model/data population differs')
    from tools.joint_occ_experiments import run as base
    _,records=base.load_cache(a.dev_cache);selected=evaluation_plan(records,c['eval_keys'])
    original=source['original_contract']
    if sha(a.dev_cache) != original['data']['dev_cache'] or sha(a.dev_info) != original['data']['dev_info']:
        raise ValueError('VAL metadata differs from the recorded controls')
    if sha(a.config) != c['train_data']['config']:raise ValueError('geometry config differs')
    pcfg=make_prepare_config(load_runtime_config(a.config,c['overrides']))
    inputs=DenseInputs(a.dataroot,a.dev_info,stc_root=a.stc_root,cache_mib=128)
    inventory={kind:inputs.preflight([window(r) for r in selected],kind) for kind in ('occ','stc')}
    if inventory != source['evaluation']['receipt']['inventory']:raise ValueError('SAME64 raw files changed')
    receipt=dict(candidate=sha(path),joint=c['joint_checkpoint_sha256'],control=c['control_sha256'],
        joint_evaluation=c['joint_evaluation_sha256'],keys=c['eval_keys'],inventory=inventory,
        full_grid=True,known='dense_valid',horizons=[1.,2.,3.],indices=[1,3,5],
        FREE=False,fixed_age=False,calibration=False,projection_supervision=False,training_updates=0,
        controls='original and joint reused from completed compatible reports; new model freshly evaluated')
    state_path=a.out_dir/'evaluation_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(receipt=receipt,progress={},counts={})
    if state['receipt'] != receipt:raise ValueError('evaluation resume source/data changed')
    model=from_spec(c['model'],candidate=True);model.load_state_dict(saved['models']['gt'],strict=True);del saved
    model.to(device).eval().requires_grad_(False)
    predictor=RecoveryPredictor(model,pcfg,device,workers=a.cpu_workers)
    visual=a.out_dir/'visuals';visual.mkdir(exist_ok=True)
    try:
        for kind in ('occ','stc'):
            done=state['progress'].get(kind,0)
            if type(done) is not int or not 0 <= done <= 64:raise ValueError('invalid complete-window prefix')
            counts=counts_from_json(state['counts'][kind]) if kind in state['counts'] else {
                route:metrics.empty(3) for route in ('transport','semantic_refine')}
            for i in range(done,64):
                if stop.is_set():raise InterruptedError('evaluation saved; restart eval_only.sh')
                row,poses,_=prepare_evaluation_row(inputs,predictor,selected[i],kind=kind)
                with torch.no_grad():
                    fw=forward_window(model,predictor.provider,row,poses=poses)
                    predictions={route:fw[key].argmax(1).reshape(6,*fw['shape']).cpu().numpy().astype(np.uint8)
                        for route,key in (('transport','prior'),('semantic_refine','logits'))}
                    coverage=(fw['prior_coverage'].reshape(6,*fw['shape'])*255).round().byte().cpu().numpy()
                del fw
                target=np.stack(inputs.targets(window(selected[i]),(1,3,5)))  # six outputs precede future labels
                for route,pred in predictions.items():
                    for h,index in enumerate((1,3,5)):metrics.add(counts[route],h,pred[index],target[h])
                if i<8:
                    prefix=f'{kind}_{i+1:02d}'
                    arrays=dict(prediction=predictions['semantic_refine'][[1,3,5]],transport=predictions['transport'][[1,3,5]],
                        gt=target,t0=np.asarray(row[1]['history_occ'][-1],np.uint8),future_poses=np.asarray(poses)[[1,3,5]],
                        prior_coverage=coverage[[1,3,5]])
                    for old_name,new_name in (('control','control'),('candidate','joint')):
                        origin=source['folder']/'visuals'/f'{prefix}_{old_name}.npz'
                        with np.load(origin,allow_pickle=False) as old:
                            if any(not np.array_equal(old[k],arrays[k]) for k in ('gt','t0','future_poses')):
                                raise ValueError('paired visual GT/input/poses differ')
                        copy_once(origin,visual/f'{prefix}_{new_name}.npz')
                    np.savez_compressed(visual/f'{prefix}_candidate.npz',**arrays)
                state['progress'][kind]=i+1
                state['counts'][kind]={r:{k:v.tolist() for k,v in fields.items()} for r,fields in counts.items()}
                write_json(state_path,state);print(f'PRIOR_ONLY_EVAL {kind} {i+1}/64',flush=True)
    finally:predictor.close()
    reports={}
    for kind in ('occ','stc'):
        reports['control/'+kind]=source['evaluation']['reports']['control/'+kind]
        reports['with_projection/'+kind]=source['evaluation']['reports']['candidate/'+kind]
        reports['candidate/'+kind]={r:metrics.report(v,(1.,2.,3.),moving=False)
            for r,v in counts_from_json(state['counts'][kind]).items()}
    lines=['===== MATCHED COVERAGE / PROJECTION OFF / SAME64 =====','status=complete; windows=64; ONE eval GPU.',
        'Original and projection-ON scores reused; ONLY the new projection-OFF model freshly evaluated.',
        'RANDOM TRAIN4086x1/1031 updates/GLOBAL B4/source128; SAME coverage/model/seed/order/Adam/cosine.',
        'No trained initialization, FREE/age/calibration/STC training/extra loss. Full-grid; GT future ego.',
        'model input route mIoU IoU']
    for model_name,prefix in (('original','control'),('projection_ON','with_projection'),('projection_OFF','candidate')):
        for kind in ('occ','stc'):
            for route,m in reports[prefix+'/'+kind].items():
                v=m['average'];lines.append(f'{model_name:15s} {kind:3s} {route:18s} {v["mIoU"]:.6f} {v["IoU"]:.6f}')
    delta={}
    for baseline in ('control','with_projection'):
        delta[baseline]={kind:{route:{k:reports['candidate/'+kind][route]['average'][k]-reports[baseline+'/'+kind][route]['average'][k]
            for k in ('mIoU','IoU')} for route in ('transport','semantic_refine')} for kind in ('occ','stc')}
        lines.append('OFF_minus_'+baseline+'='+json.dumps(delta[baseline]))
    lines+=['OFF vs ON isolates projection supervision; original vs OFF also changes coverage/Prior fusion.',
        'Final epoch only; repeatedly inspected development64, NOT independent/full-VAL/FPS. No automatic promotion.']
    cases=[dict(index=i+1,scene=r['scene_name'],t0=r['t0_token']) for i,r in enumerate(selected[:8])]
    write_json(a.out_dir/'evaluation.json',dict(status='complete',receipt=receipt,reports=reports,delta=delta,visual_cases=cases))
    write_json(visual/'manifest.json',dict(cases=cases,horizons=[1,2,3],palette='OccWorld/COME',selection='same fixed first8; no outcome selection',
        grid=dict(origin=[pcfg.grid.x_min,pcfg.grid.y_min,pcfg.grid.z_min],step=list(pcfg.grid.voxel_size),shape=list(pcfg.grid.shape_hwd))))
    (a.out_dir/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out-dir','dataroot','dev-info','dev-cache','stc-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/real_motion_occfm.yaml')
    p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
    lock=a.out_dir/'eval_lock';lock.mkdir(exist_ok=True)
    with evaluation_lock(lock):run(a,stop)


if __name__=='__main__':main()
