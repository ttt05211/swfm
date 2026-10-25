"""ONE completed C checkpoint / SAME64 / OCC+STC / fixed memory-age ONLY."""
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
from tools.causal_forward_recovery.run import load_source,sha,reference
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.causal_forward_train.data import updated_record
from tools.fpn_unet_eval.common import DenseInputs
from tools.fpn_unet_stc_age.processing import candidate,HALF_LIFE_SECONDS
from tools.final_semantic_evaluation.sources import window
from tools.final_semantic_evaluation import metrics
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint

PROTOCOL = 'frozen_forward_trained_AdaLN_same64_OCC_STC_memory_age_off_on_raw_v1'
HORIZONS = (1.,2.,3.)
NAMES = tuple(f'{kind}/{route}' for kind in ('occ','stc') for route in
    ('transport','age_off/semantic_refine','age_on/semantic_refine'))


def restore(saved,contract,voxels):
    counts = {n:metrics.empty(3) for n in NAMES};keys = []
    if saved is None:return counts,keys
    keys = copy.deepcopy(saved['keys'])
    if (saved['contract'] != contract or keys != contract['keys'][:len(keys)]
            or len(keys)>len(contract['keys']) or set(saved['counts']) != set(NAMES)):
        raise RuntimeError('same source/code/data/ordered64 complete-window prefix required')
    if set(saved['weights'])!={str(i) for i in range(len(keys))}:raise RuntimeError('age evidence/window prefix differs')
    for value in saved['weights'].values():
        if set(value)!={'occ','stc'}:raise RuntimeError('age evidence modalities differ')
        for weights in value.values():
            a = np.asarray(weights)
            if a.shape!=(4,) or not np.isfinite(a).all() or np.any((a<=0)|(a>1)) or a[-1]!=1:
                raise RuntimeError('invalid saved four age weights')
    for name,fields in saved['counts'].items():
        if set(fields) != set(counts[name]):raise RuntimeError('metric count fields differ')
        for field,value in fields.items():
            a = np.asarray(value)
            if a.shape!=(3,18,18) or a.dtype.kind not in 'iu' or np.any(a<0):raise RuntimeError('invalid integer counts')
            counts[name][field][:] = a
        if (np.any(counts[name]['confusion'].sum((1,2)) != len(keys)*voxels)
                or np.any(counts[name]['moving']>counts[name]['confusion'])):
            raise RuntimeError('incomplete full-grid/Moving counts')
    return counts,keys


def change(new,old):
    return {k:None if new[k] is None or old[k] is None else new[k]-old[k]
        for k in ('mIoU','IoU','MovingMicro','MovingMacro')}


@torch.no_grad()
def evaluate(body,predictor,inputs,records,contract,*,saved=None,save=None,event=None):
    voxels = int(np.prod(predictor.provider.pcfg.grid.shape_hwd))
    counts,keys = restore(saved,contract,voxels);event = event or threading.Event();body.eval()
    weights_seen = {} if saved is None else copy.deepcopy(saved['weights'])
    def snapshot():
        return dict(contract=contract,keys=copy.deepcopy(keys),weights=copy.deepcopy(weights_seen),counts={
            n:{k:v.tolist() for k,v in fields.items()} for n,fields in counts.items()})
    try:
        for record in records[len(keys):]:
            if event.is_set():raise InterruptedError('stopped after complete window')
            job = window(record);predictions = {};local_weights = {}
            for kind in ('occ','stc'):
                _,raw,times = inputs.history(job,kind)
                row = predictor.prepare(record,raw,times=times)
                if kind=='occ':updated_record(record,row[1])
                fw = predictor.infer(row,inputs.poses(job))
                # The existing fixed-age implementation recomputes ONLY aligned
                # Memory and Refiner from this SAME field/motion/Strong/prior.
                aged,weights = candidate(body,predictor.provider,row,fw)
                if weights.shape!=(4,) or not np.isfinite(weights).all():raise RuntimeError('invalid four age weights')
                local_weights[kind] = weights.tolist()
                for name,value in (('transport',fw['prior']),('age_off/semantic_refine',fw['logits']),
                                   ('age_on/semantic_refine',aged)):
                    if value.shape[:2]!=(6,18) or not bool(torch.isfinite(value).all()):
                        raise RuntimeError('SIX finite full-class predictions required')
                    predictions[kind+'/'+name] = value.argmax(1).reshape(6,*fw['shape']).byte().cpu().numpy()
                del fw,aged,row,raw
            # BOTH modalities / BOTH six-frame readouts precede future labels.
            target = inputs.targets(job,(1,3,5))
            moving = inputs.moving(job,(1,3,5),HORIZONS,predictor.provider.pcfg.grid,predictor.provider.workers)
            local = {n:metrics.empty(3) for n in NAMES}
            for name in NAMES:
                for h,index in enumerate((1,3,5)):metrics.add(local[name],h,predictions[name][index],target[h],moving[h])
            # Commit only after EVERY route/horizon has valid counts.
            for name in NAMES:
                for field in counts[name]:counts[name][field] += local[name][field]
            keys.append([str(record['scene_name']),str(record['t0_token'])])
            weights_seen[str(len(keys)-1)] = local_weights
            if save:save(snapshot())
            if len(keys)%8==0:print(f'FORWARD_AGE {len(keys)}/64 weights={local_weights["stc"]}; ZERO training',flush=True)
    finally:
        if save:save(snapshot())
    reports = {n:metrics.report(v,HORIZONS) for n,v in counts.items()}
    deltas = {}
    for kind in ('occ','stc'):
        off,on = (reports[kind+'/'+mode+'/semantic_refine'] for mode in ('age_off','age_on'))
        deltas[kind] = dict(average=change(on['average'],off['average']),per_horizon={
            h:change(on['per_horizon'][h],off['per_horizon'][h]) for h in off['per_horizon']})
    return dict(keys=keys,reports=reports,age_on_minus_off=deltas,weights=weights_seen)


def summary(result):
    lines = ['===== FROZEN FORWARD-TRAINED AdaLN / HISTORY AGE OFF-ON =====',
        'status='+result['status'],f'windows={len(result["keys"])}/64; scientific_updates=0; ONE GPU / ONE model process.',
        'Original C epoch1/update='+str(result['contract']['source']['updates'])+'; SAME checkpoint/keys/GT future ego/dense-valid.',
        f'Fixed half_life_s={HALF_LIFE_SECONDS}; actual historical timestamps; nominal old->new=[1/8,1/4,1/2,1].',
        'OCC AND STC each evaluated off/on. ONLY static+dynamic Memory fusion weights change, numerator AND denominator.',
        'Shared association/KTA/Encoder/source motion/history yaw/Strong/Transport Prior. All SIX futures before labels.',
        'FREE bias OFF; no training, new fitting, norm adaptation, masks, threshold search or automatic promotion.',
        'input route                         mIoU       IoU MovingMicro']
    fmt = lambda v:'NA' if v is None else f'{v:.6f}'
    for name,report in result['reports'].items():
        row = report['average'];lines.append(f'{name:33s} '+' '.join(f'{fmt(row[k]):>10s}' for k in ('mIoU','IoU','MovingMicro')))
    for kind,value in result.get('age_on_minus_off',{}).items():
        lines.append(kind+' age_on_minus_off='+json.dumps(value))
    lines.append('Repeated development64; NOT independent/full-VAL proof or FPS. Source weights/Adam/RNG/caches unchanged.')
    return '\n'.join(lines)+'\n'


def main(argv=None,event=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','out-dir','config','dataroot','dev-info','dev-cache','stc-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    a = p.parse_args(argv);event = event or threading.Event();out = Path(a.out_dir).resolve();source = Path(a.checkpoint).resolve()
    if out==source.parent or source.parent in out.parents:raise RuntimeError('output outside source C_train required')
    if a.device.startswith('cuda') and (torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('ONE BF16 GPU required')
    if not 1<=a.cpu_workers<=4:raise ValueError('workers1..4 required')
    torch.set_num_threads(1);torch.manual_seed(21)
    body,c,receipt = load_source(source,a.device)  # All completed-C/old source SHA checks retained.
    for name in ('config','dev_info','dev_cache'):
        if sha(getattr(a,name))!=c['data'][name]:raise RuntimeError('source input differs: '+name)
    cp,_ = reference(Path(c['model']['experiment']['control_results']).parent)
    if sha(cp)!=c['model']['experiment']['control_results_sha256']:raise RuntimeError('control report changed')
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config,c['overrides']))
    _,records = base.load_cache(a.dev_cache);records = base.align_records(records,c['eval_keys'])
    if len(records)!=64 or len(set(map(tuple,c['eval_keys'])))!=64:raise RuntimeError('exact same unique64 required')
    inputs = DenseInputs(a.dataroot,a.dev_info,stc_root=a.stc_root,cache_mib=256);jobs = [window(r) for r in records]
    inventory = set()
    for job in jobs:
        for token in job.history_tokens:
            path = inputs.path('stc',job.scene_name,token).resolve();size,mtime = inputs.stat(path)
            inventory.add((str(path),size,mtime))
    if [list(row) for row in sorted(inventory)]!=c['STC_inventory']:raise RuntimeError('source STC history inventory changed')
    root = Path(__file__).resolve().parents[2]
    files = [*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'run.sh',
        root/'tools/causal_forward_recovery/adapter.py',root/'tools/causal_forward_recovery/run.py',
        root/'tools/fpn_unet_stc_age/processing.py',root/'tools/fpn_unet_eval/common.py',
        root/'tools/final_semantic_evaluation/sources.py',root/'tools/final_semantic_evaluation/metrics.py']
    contract = json.loads(json.dumps(dict(protocol=PROTOCOL,source=receipt,source_contract=c,keys=c['eval_keys'],
        half_life_s=HALF_LIFE_SECONDS,modalities=['occ','stc'],FREE_bias=0.,
        data={kind:inputs.preflight(jobs,kind) for kind in ('occ','stc')},
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device,workers=a.cpu_workers),
        implementation={f.relative_to(root).as_posix():sha(f) for f in files})))
    lock = out.parent/'.forward_age_locks'/base.stable_json_fingerprint(str(out));lock.mkdir(parents=True,exist_ok=True)
    predictor = RecoveryPredictor(body,pcfg,a.device,workers=a.cpu_workers)
    before = tensor_state_fingerprint(body.state_dict());start = time.perf_counter()
    try:
        with evaluation_lock(lock):
            if not a.resume:out.mkdir(parents=True,exist_ok=False)
            elif not out.is_dir():raise RuntimeError('resume original age output only')
            if a.resume and json.loads((out/'contract.json').read_text())!=contract:raise RuntimeError('resume source/code/data changed')
            base.write_json(out/'contract.json',contract)
            saved = json.loads((out/'state.json').read_text()) if a.resume and (out/'state.json').exists() else None
            def store(value):
                base.write_json(out/'state.json',value)
                base.write_json(out/'progress.json',dict(windows=len(value['keys']),total=64,
                    scientific_updates=0,seconds=time.perf_counter()-start))
            result = dict(status='running',contract=contract,keys=saved['keys'] if saved else [],reports={})
            try:
                result.update(evaluate(body,predictor,inputs,records,contract,saved=saved,save=store,event=event))
                if result['keys']!=contract['keys']:raise RuntimeError('incomplete same64')
                result['status']='complete'
            except InterruptedError:
                result['status']='stopped_at_complete_window';return 130
            except Exception as error:
                result.update(status='failed',error=repr(error));raise
            finally:
                state_path = out/'state.json'
                if state_path.exists():result['keys']=json.loads(state_path.read_text())['keys']
                base.write_json(out/'results.json',result);(out/'summary.txt').write_text(summary(result),encoding='utf-8')
                print(summary(result),flush=True)
    finally:
        predictor.close()
        if tensor_state_fingerprint(body.state_dict())!=before or sha(source)!=receipt['sha256']:
            raise RuntimeError('source checkpoint/frozen model changed')
    return 0


if __name__=='__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
