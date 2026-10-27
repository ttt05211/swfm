"""ONE GPU / epoch2 vs existing mean3 / fixed256 excluding original64 scenes."""
import argparse
import copy
import json
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.causal_forward_average.run import ForwardMeanPredictor
from tools.causal_forward_train.data import updated_record
from tools.fpn_unet_eval.common import DenseInputs
from tools.final_semantic_evaluation.sources import window
from tools.final_semantic_evaluation import metrics
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .artifacts import select_mean,load_pair,sha

PROTOCOL='frozen_forward_epoch2_vs_existing_mean3_scene_disjoint_other_VAL256_v1'
MODELS=('epoch2','mean3');KINDS=('occ','stc');HORIZONS=(1.,2.,3.)
NAMES=tuple('/'.join((m,k,r)) for m in MODELS for k in KINDS for r in ('transport','semantic_refine'))


def select_records(records,c):
    keys=base.record_keys(records);old=[tuple(map(str,k)) for k in c['eval_keys']]
    if (len(keys)!=4369 or len(set(keys))!=4369 or len({k[0] for k in keys})!=150
            or len(old)!=64 or len(set(old))!=64 or not set(old)<=set(keys)):
        raise RuntimeError('complete VAL4369/150 and exact original64 required')
    if {k[0] for k in keys}&{k[0] for k in c['train_keys']}:raise RuntimeError('TRAIN/VAL scene overlap')
    excluded={k[0] for k in old};remaining=sorted(k for k in keys if k[0] not in excluded)
    chosen=base.select_scene_balanced_round_robin(remaining,256)
    if len(chosen)!=256 or len(set(chosen))!=256:raise RuntimeError('256 other-scene windows required; never backfill from original64 scenes')
    return base.align_records(records,chosen),dict(parent_windows=4369,parent_scenes=150,windows=256,
        scenes=len({k[0] for k in chosen}),excluded_scenes=sorted(excluded),original64=copy.deepcopy(c['eval_keys']),
        selection='sorted(scene,t0) -> scene-balanced round-robin256, EXCLUDING ALL original64 scenes',
        independent_test=False,planner_trimmed=False)


def restore(saved,contract,voxels):
    counts={n:metrics.empty(3) for n in NAMES};keys=[]
    if saved is None:return counts,keys
    keys=copy.deepcopy(saved['keys'])
    if (saved['contract']!=contract or keys!=contract['keys'][:len(keys)] or len(keys)>len(contract['keys'])
            or set(saved['counts'])!=set(NAMES)):raise RuntimeError('same pair/population/code/data complete-window prefix required')
    for name,fields in saved['counts'].items():
        if set(fields)!=set(counts[name]):raise RuntimeError('count fields differ')
        for field,value in fields.items():
            a=np.asarray(value)
            if a.shape!=(3,18,18) or a.dtype.kind not in 'iu' or np.any(a<0):raise RuntimeError('invalid integer counts')
            counts[name][field][:]=a
        if (np.any(counts[name]['confusion'].sum((1,2))!=len(keys)*voxels)
                or np.any(counts[name]['moving']>counts[name]['confusion'])):raise RuntimeError('incomplete window/Moving counts')
    return counts,keys


def difference(reports):
    result={}
    def delta(a,b):return {k:None if a[k] is None or b[k] is None else a[k]-b[k]
        for k in ('mIoU','IoU','MovingMicro','MovingMacro')}
    for kind in KINDS:
        for route in ('transport','semantic_refine'):
            a,b=(reports[m+'/'+kind+'/'+route] for m in ('mean3','epoch2'))
            result[kind+'/'+route]=dict(average=delta(a['average'],b['average']),
                per_horizon={h:delta(a['per_horizon'][h],b['per_horizon'][h]) for h in a['per_horizon']})
    return result


@torch.no_grad()
def evaluate(predictors,inputs,records,contract,*,saved=None,save=None,event=None):
    p=predictors['epoch2'];grid=p.provider.pcfg.grid;event=event or threading.Event()
    counts,keys=restore(saved,contract,int(np.prod(grid.shape_hwd)))
    def snapshot():return dict(contract=contract,keys=copy.deepcopy(keys),counts={
        n:{k:v.tolist() for k,v in fields.items()} for n,fields in counts.items()})
    try:
        for record in records[len(keys):]:
            if event.is_set():raise InterruptedError('complete paired window saved')
            job=window(record);predictions={};poses=inputs.poses(job)
            for kind in KINDS:
                _,raw,times=inputs.history(job,kind);row=p.prepare(record,raw,times=times)
                if kind=='occ':updated_record(record,row[1])
                # Geometry-only four-history association/reads are identical.
                # Each model computes its OWN field/motion/fresh future geometry/Memory/Refiner.
                for name in MODELS:
                    fw=predictors[name].forward(row,poses=poses)
                    dense=predictors[name].dense(fw)
                    if set(dense)!={'transport','semantic_refine'} or any(v.shape!=(6,*grid.shape_hwd) for v in dense.values()):
                        raise RuntimeError('complete SIX-frame readouts required before labels')
                    predictions.update({name+'/'+kind+'/'+r:v for r,v in dense.items()});del fw
                del row,raw
            target=inputs.targets(job,(1,3,5))
            moving=inputs.moving(job,(1,3,5),HORIZONS,grid,p.provider.workers)
            local={n:metrics.empty(3) for n in NAMES}
            for name in NAMES:
                for h,at in enumerate((1,3,5)):metrics.add(local[name],h,predictions[name][at],target[h],moving[h])
            for name in NAMES:
                for field in counts[name]:counts[name][field]+=local[name][field]
            keys.append([str(record['scene_name']),str(record['t0_token'])])
            if save:save(snapshot())
            if len(keys)%8==0:print(f'COMPARE256 {len(keys)}/256; BOTH models / BOTH inputs complete; ZERO training',flush=True)
    finally:
        if save:save(snapshot())
    reports={n:metrics.report(v,HORIZONS) for n,v in counts.items()}
    return dict(keys=keys,reports=reports,mean3_minus_epoch2=difference(reports))


def summary(value):
    lines=['===== CURRENT FORWARD AdaLN / EPOCH2 vs EXISTING MEAN3 / FIXED256 =====','status='+value['status'],
        f'windows={len(value["keys"])}/256; ONE GPU / ONE process; scientific_updates=0.',
        'Same fixed256 OTHER VAL-scene windows; ALL original64 scenes excluded; SAME models/no new fusion.',
        'GT future ego/dense-valid/full-grid/1,2,3s/Moving; current forward AdaLN/FPN/U-Net; FREE and fixed age OFF.',
        'Shared history geometry ONLY; per-model field/motion/Strong/Transport/Memory/Refiner recomputed independently.',
        'model input_route                         mIoU       IoU MovingMicro']
    fmt=lambda v:'NA' if v is None else f'{v:.6f}'
    for name,report in value['reports'].items():
        m=report['average'];lines.append(f'{name:38s} '+' '.join(f'{fmt(m[k]):>10s}' for k in ('mIoU','IoU','MovingMicro')))
    for name,row in value.get('mean3_minus_epoch2',{}).items():lines.append(name+' mean3_minus_epoch2='+json.dumps(row['average']))
    pop=value['contract']['population']
    lines+=['population='+json.dumps(dict(windows=pop['windows'],scenes=pop['scenes'],parent_windows=pop['parent_windows'],
        excluded_original64_scene_count=len(pop['excluded_scenes']),selection=pop['selection'])),
        'Detailed per-horizon scores/deltas are retained in results.json; ordered keys in contract.json.',
        'Previously inspected VAL / DEV-selected mean; NOT a pristine independent test or standard full-VAL/FPS.',
        'Source checkpoints/Adam/RNG/mean3/reports/history caches READ ONLY; no automatic selection or promotion.']
    return '\n'.join(lines)+'\n'


def main(argv=None,event=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mean-dir');p.add_argument('--full-dir');p.add_argument('--out-dir',required=True);p.add_argument('--resume',action='store_true')
    for name in ('config','dev-cache','dev-info','dataroot','stc-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-workers',type=int,default=4)
    a=p.parse_args(argv);event=event or threading.Event();root=Path(__file__).resolve().parents[2];out=Path(a.out_dir).resolve()
    if a.device.startswith('cuda') and (torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()):raise RuntimeError('ONE BF16 GPU required')
    if not 1<=a.cpu_workers<=4:raise ValueError('workers1..4 required')
    torch.set_num_threads(1);torch.manual_seed(21)
    if a.resume and not a.mean_dir:a.mean_dir=json.loads((out/'contract.json').read_text())['sources']['mean_dir']
    mean_dir=select_mean(a.mean_dir,root/'outputs/p0_f9_shared_history')
    bodies,c,sources=load_pair(mean_dir,a.full_dir)
    for folder in (Path(sources['full_dir']),mean_dir):
        if out==folder or folder in out.parents:raise RuntimeError('NEW output outside BOTH source runs required')
    for name in ('config','dev_cache','dev_info'):
        if sha(getattr(a,name))!=c['data'][name]:raise RuntimeError('source input differs: '+name)
    _,all_records=base.load_cache(a.dev_cache);records,population=select_records(all_records,c)
    inputs=DenseInputs(a.dataroot,a.dev_info,stc_root=a.stc_root,cache_mib=256);jobs=[window(r) for r in records]
    files=[*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'run.sh',*[root/n for n in (
        'tools/causal_forward_average/run.py','tools/causal_forward_average/weights.py','tools/causal_forward_full/model.py',
        'tools/causal_forward_recovery/adapter.py','tools/causal_forward_train/data.py','tools/fpn_unet_eval/common.py',
        'tools/final_semantic_evaluation/sources.py','tools/final_semantic_evaluation/metrics.py')]]
    contract=json.loads(json.dumps(dict(protocol=PROTOCOL,sources=sources,source_contract=c,
        keys=[list(k) for k in base.record_keys(records)],population=population,FREE_bias=0.,history_age=False,
        inventory={kind:inputs.preflight(jobs,kind) for kind in KINDS},
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device,workers=a.cpu_workers),
        implementation={f.relative_to(root).as_posix():sha(f) for f in files})))
    lock=out.parent/'.forward_compare256_locks'/base.stable_json_fingerprint(str(out));lock.mkdir(parents=True,exist_ok=True)
    predictors={};fingerprints={};start=time.perf_counter()
    try:
        with evaluation_lock(lock):
            if not a.resume:out.mkdir(parents=True,exist_ok=False)
            elif json.loads((out/'contract.json').read_text())!=contract:raise RuntimeError('resume same frozen pair/population/code/data only')
            base.write_json(out/'contract.json',contract)
            pcfg=base.make_prepare_config(base.load_runtime_config(a.config,c['overrides']))
            for name,body in bodies.items():
                body.to(a.device).eval().requires_grad_(False);fingerprints[name]=tensor_state_fingerprint(body.state_dict())
                predictors[name]=ForwardMeanPredictor(body,pcfg,a.device,a.cpu_workers)
            saved=json.loads((out/'state.json').read_text()) if a.resume and (out/'state.json').exists() else None
            value=dict(status='running',contract=contract,keys=saved['keys'] if saved else [],reports={},scientific_updates=0)
            def store(state):
                base.write_json(out/'state.json',state)
                base.write_json(out/'progress.json',dict(windows=len(state['keys']),total=256,seconds=time.perf_counter()-start))
            try:
                value.update(evaluate(predictors,inputs,records,contract,saved=saved,save=store,event=event))
                if value['keys']!=contract['keys']:raise RuntimeError('incomplete paired256')
                value['status']='complete'
            except InterruptedError:value['status']='stopped_at_complete_paired_window';return 130
            except Exception as error:value.update(status='failed',error=repr(error));raise
            finally:
                if (out/'state.json').exists():value['keys']=json.loads((out/'state.json').read_text())['keys']
                base.write_json(out/'results.json',value);(out/'summary.txt').write_text(summary(value),encoding='utf-8');print(summary(value),flush=True)
    finally:
        for predictor in predictors.values():predictor.close()
        for name,before in fingerprints.items():
            if tensor_state_fingerprint(bodies[name].state_dict())!=before:raise RuntimeError('frozen model changed: '+name)
        for item in [*sources['checkpoints'].values(),*sources['source_checkpoints'],sources['full_report']]:
            if sha(item['path'])!=item['sha256']:raise RuntimeError('source file changed: '+item['path'])
    return 0


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
