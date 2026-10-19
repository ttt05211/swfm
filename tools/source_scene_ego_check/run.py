"""Single-GPU frozen new candidate: learned ego projection versus zero projection OUTPUT."""
import argparse
import json
from pathlib import Path
import signal
import threading
import torch
from tools.joint_occ_experiments import run as base
from tools.source_scene_ego_conditioned.run import PROTOCOL as TRAIN_PROTOCOL
from tools.source_scene_ego_conditioned.model import checkpoint_model, VARIANT
from tools.source_scene_stage2.adapter import DirectPredictor
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .engine import PROTOCOL,evaluate


def select_source(root,explicit=None):
    if explicit:return Path(explicit).resolve()
    candidates=[]
    for p in Path(root).glob('scene_ego_query_1ep_*/results.json'):
        v=json.loads(p.read_text(encoding='utf-8'))
        if v.get('status')=='complete' and v.get('contract',{}).get('protocol')==TRAIN_PROTOCOL:candidates.append(p.parent.resolve())
    if len(candidates)!=1:raise RuntimeError('set EGO_QUERY_DIR to unique COMPLETED three-change pilot; candidates='+str(candidates))
    return candidates[0]


def load_source(folder):
    report_path=Path(folder)/'results.json';sha=base.sha256(report_path)
    v=json.loads(report_path.read_text(encoding='utf-8'));c=v['contract'];keys=c['eval_keys']
    if (v.get('status')!='complete' or c.get('protocol')!=TRAIN_PROTOCOL or c.get('epochs')!=1 or
            c['model'].get('variant')!=VARIANT or c.get('known')!='TRAIN_OCC_STC_dense_valid' or
            c.get('source_policy')!='shared_pinned' or c.get('registration')!='none' or
            c.get('total_updates')!=1031 or c.get('epochs_updates')!=[1031] or v['updates']!=c['total_updates'] or
            len(c['train_keys'])!=4086 or len(set(map(tuple,c['train_keys'])))!=4086 or len(keys)!=64 or
            len(set(map(tuple,keys)))!=64 or v['evaluations']['1']['windows']!=64 or
            v['evaluations']['1']['keys']!=keys):
        raise RuntimeError('complete correct new RANDOM TRAIN20%x1 candidate / same remaining64 required')
    path=Path(folder)/'epoch_0001.pt';checkpoint_sha=base.sha256(path)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved.get('protocol')!=TRAIN_PROTOCOL or saved.get('contract')!=c or saved.get('epoch')!=1 or
            saved.get('batch')!=0 or saved.get('updates')!=c['total_updates'] or set(saved['models'])!={'gt'}):
        raise RuntimeError('checkpoint lineage/cursor differs')
    if base.sha256(path)!=checkpoint_sha or base.sha256(report_path)!=sha:raise RuntimeError('source changed while reading')
    header={k:saved[k] for k in ('protocol','contract','models')} # NO Adam/RNG retained or stepped.
    return c,header,dict(path=str(path.resolve()),sha256=checkpoint_sha,report_sha256=sha),v['evaluations']['1']['variants']


def summary(result):
    lines=['===== FROZEN EGO QUERY CONTRIBUTION / SAME64 =====','status='+result['status'],
        'ONE GPU / ONE frozen checkpoint; scientific_updates=0. No training/optimizer/head selection.',
        'ego_off = projection OUTPUT zero, NOT input pose zero. SAME correct future poses in ALL renderers.',
        'Same historical field/reads/lift, fresh background and KTA per paired window; dense-valid OCC/STC.',
        'First fixed8 windows/modality additionally repeat ego_on; byte FAIL is reported, never waived.',
        'input mode     route                    mIoU       IoU MovingMicro']
    audit=result.get('evaluation')
    if audit:
        lines.append('windows='+str(audit['windows']))
        for name,r in audit['reports'].items():
            k,m,route=name.split('/');v=r['average']
            fmt=lambda x:'NA' if x is None else f'{x:.6f}'
            lines.append(f'{k:5s} {m:8s} {route:20s} '+' '.join(fmt(v[x]) for x in ('mIoU','IoU','MovingMicro')))
        for kind in ('occ','stc'):
            for route in ('transport','semantic_refine'):
                a=audit['reports'][kind+'/ego_on/'+route]['average'];b=audit['reports'][kind+'/ego_off/'+route]['average']
                lines.append(f'{kind}/{route} ego_on_minus_off='+json.dumps({k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in ('mIoU','IoU','MovingMicro')}))
                original=result['contract']['recorded_variants'][('occ_gt' if kind=='occ' else 'stc_gt')+'_'+route]
                lines.append(f'{kind}/{route} fresh_on_minus_recorded='+json.dumps({k:a[k]-original[k] if a[k] is not None else None for k in ('mIoU','IoU','MovingMicro')}))
            lines.append(kind+' response='+json.dumps(audit['diagnostics'][kind]))
    lines+=['Check measures reliance on the learned ego branch at THIS checkpoint, NOT a retrained no-ego ablation.',
        'GT future ego conditioned; fixed repeatedly inspected development64, NOT Pred/full-VAL/FPS or statistical proof.',
        'Shared-prepare numerical execution, NOT bit-identical to cached training evaluation. Source weights/Adam/RNG/caches read only.']
    return '\n'.join(lines)+'\n'


def main(argv=None,event=None):
    event=event or threading.Event();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir');p.add_argument('--out-dir',required=True);p.add_argument('--resume',action='store_true')
    for n in ('config','dev-cache','dev-info','dataroot','stc-root'):p.add_argument('--'+n,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-workers',type=int,default=4)
    a=p.parse_args(argv);repo=Path(__file__).resolve().parents[2];out=Path(a.out_dir).resolve()
    if not 1<=a.cpu_workers<=8:raise ValueError('bounded CPU workers required')
    if a.device.startswith('cuda') and (torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()):raise RuntimeError('ONE BF16 GPU required')
    torch.set_num_threads(1)
    if a.resume and not a.source_dir:a.source_dir=json.loads((out/'contract.json').read_text(encoding='utf-8'))['source_dir']
    source=select_source(repo/'outputs/p0_f9_shared_history',a.source_dir)
    if source==out or source in out.parents:raise RuntimeError('independent NEW audit directory outside source required')
    c,saved,receipt,recorded=load_source(source)
    for name in ('config','dev_cache','dev_info'):
        if base.sha256(getattr(a,name))!=c['data'][name]:raise RuntimeError('source metadata differs: '+name)
    for name,sha in c['implementation'].items():
        if base.sha256(repo/name)!=sha:raise RuntimeError('trained implementation changed: '+name)
    _,records=base.load_cache(a.dev_cache);keys=c['eval_keys'];jobs=[window(r) for r in base.align_records(records,keys)]
    inputs=NuscInputs(a.dataroot,a.dev_info,stc_root=a.stc_root,cache_mib=256)
    inventory={k:inputs.preflight(jobs,k) for k in ('occ','stc')}
    for path,size,stamp in c['STC_inventory']:
        stat=Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(size,stamp):raise RuntimeError('source STC input changed')
    files=list(Path(__file__).parent.glob('*.py'))+[repo/n for n in ('tools/final_semantic_evaluation/sources.py',
        'tools/final_semantic_evaluation/metrics.py','tools/final_semantic_evaluation/predictor.py')]
    contract=dict(protocol=PROTOCOL,source_dir=str(source),source=receipt,source_contract=c,keys=keys,recorded_variants=recorded,
        intervention='ONLY ego_future_proj OUTPUT zero; SAME pose inputs/geometry',repeat_first_windows=8,
        inventory=inventory,execution=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device,cpu_workers=a.cpu_workers),
        implementation={f.relative_to(repo).as_posix():base.sha256(f) for f in sorted(files)})
    contract=json.loads(json.dumps(contract))
    lock=out.parent/'.ego_check_locks'/base.stable_json_fingerprint(str(out));lock.mkdir(parents=True,exist_ok=True)
    with evaluation_lock(lock):
        if a.resume:
            if json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:raise RuntimeError('same unchanged audit contract required')
        else:out.mkdir(parents=True,exist_ok=False);base.write_json(out/'contract.json',contract)
        body=checkpoint_model(saved,a.device);del saved
        fingerprint=tensor_state_fingerprint(body.state_dict())
        predictor=DirectPredictor(body,base.make_prepare_config(base.load_runtime_config(a.config,c['overrides'])),a.device,a.cpu_workers)
        result=dict(status='running',contract=contract,scientific_updates=0)
        def publish():
            base.write_json(out/'results.json',result);(out/'summary.txt').write_text(summary(result),encoding='utf-8')
        try:
            publish();statefile=out/'state.json'
            state=json.loads(statefile.read_text(encoding='utf-8')) if a.resume and statefile.exists() else None
            result['evaluation']=evaluate(predictor,inputs,jobs,keys,contract,saved=state,
                save=lambda v:base.write_json(statefile,v),stop_event=event)
            result['status']='complete'
        except InterruptedError:result['status']='stopped_at_complete_window';return 2
        except Exception as error:result.update(status='failed',error=repr(error));raise
        finally:
            predictor.close()
            if (tensor_state_fingerprint(body.state_dict())!=fingerprint or any(p.grad is not None for p in body.parameters()) or
                    body.transport.ego_future_proj._forward_hooks):raise RuntimeError('frozen weights/grad/hook state changed')
            if base.sha256(receipt['path'])!=receipt['sha256'] or base.sha256(source/'results.json')!=receipt['report_sha256']:
                raise RuntimeError('original checkpoint/report changed')
            publish();print(summary(result),flush=True)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    raise SystemExit(main(event=event) or 0)
