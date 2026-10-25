"""Frozen epoch1 -> same64 raw/FREE-1; ZERO training/optimizer/checkpoint writes."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.causal_forward_train import model as trained_model
from tools.causal_forward_train.run import PROTOCOL as TRAIN_PROTOCOL
from tools.causal_forward_train.control import sha,reference
from tools.causal_forward_train.calibration import CALIBRATION,predict,delta
from tools.causal_forward_train.evaluation import NAMES,gt
from tools.causal_forward_train.data import updated_record
from tools.causal_forward_train.report import combine
from tools.fpn_unet_eval.common import DenseInputs
from tools.final_semantic_evaluation.sources import window
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .adapter import RecoveryPredictor

PROTOCOL = 'frozen_completed_forward_epoch1_history_metadata_recovery_same64_v1'


def validate_checkpoint(saved):
    c = saved['contract']
    if (saved.get('protocol') != TRAIN_PROTOCOL or c.get('protocol') != TRAIN_PROTOCOL
            or saved.get('epoch') != 1 or saved.get('batch') != 0 or saved.get('updates') != c.get('total_updates')
            or not isinstance(saved.get('updates'),int) or saved['updates'] <= 0 or c.get('epochs') != 1
            or c.get('world_size') != 1 or c.get('source_policy') != trained_model.SOURCE_POLICY
            or c.get('known') != 'TRAIN_OCC_STC_dense_valid' or c.get('registration') != 'none'
            or c.get('objective') != dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.)
            or len(c.get('train_keys',[])) != 4086 or len(set(map(tuple,c['train_keys']))) != 4086
            or len(c.get('eval_keys',[])) != 64 or len(set(map(tuple,c['eval_keys']))) != 64
            or set(saved.get('models',{})) != {'gt'}):
        raise RuntimeError('completed epoch1/batch0/all-updates checkpoint required; NEVER resume/retrain a partial source')
    return c


def load_source(path,device):
    path = Path(path).resolve();digest = sha(path)
    saved = torch.load(path,map_location='cpu',weights_only=False);c = copy.deepcopy(validate_checkpoint(saved))
    root = Path(__file__).resolve().parents[2]
    for name,expected in c['implementation'].items():
        file = (root/name).resolve()
        if root not in file.parents or sha(file) != expected:raise RuntimeError('trained source changed: '+name)
    body = trained_model.checkpoint_model(saved,device).requires_grad_(False)
    if sum(p.numel() for p in body.parameters()) != c['parameters']:raise RuntimeError('source parameter count differs')
    if sha(path) != digest:raise RuntimeError('source checkpoint changed during loading')
    return body,c,dict(checkpoint=str(path),sha256=digest,epoch=1,updates=saved['updates'],scientific_updates=0)


def restore(saved,contract):
    metrics = {n:gt.Metrics() for n in NAMES};keys = []
    if saved is None:return metrics,keys
    if (saved['contract'] != contract or saved['keys'] != contract['keys'][:len(saved['keys'])]
            or len(saved['keys']) > len(contract['keys']) or set(saved['metrics']) != set(metrics)):
        raise RuntimeError('recovery same source/code/data/ordered64 complete-window prefix required')
    keys = copy.deepcopy(saved['keys'])
    for name,fields in saved['metrics'].items():
        if set(fields) != set(vars(metrics[name])):raise RuntimeError('metric fields differ')
        for field,raw in fields.items():
            a = np.asarray(raw);dest = getattr(metrics[name],field)
            if a.shape != dest.shape or a.dtype.kind not in 'iu' or np.any(a < 0):raise RuntimeError('invalid integer counts')
            dest[:] = a
    return metrics,keys


@torch.no_grad()
def evaluate(body,predictor,inputs,records,contract,*,saved=None,save=None,event=None):
    metrics,keys = restore(saved,contract);event = event or threading.Event();body.eval()
    def snapshot():
        return dict(contract=contract,keys=copy.deepcopy(keys),metrics={
            n:{k:v.tolist() for k,v in vars(m).items()} for n,m in metrics.items()})
    try:
        for record in records[len(keys):]:
            if event.is_set():raise InterruptedError('stopped after complete window')
            job = window(record);predictions = {}
            for kind in ('occ','stc'):
                _,raw,times = inputs.history(job,kind)
                row = predictor.prepare(record,raw,times=times)
                if kind=='occ':updated_record(record,row[1])  # Preserve cached t0 actor identity/order/count.
                fw = predictor.infer(row,inputs.poses(job))
                predictions[kind+'_gt_transport'] = fw['prior'].argmax(1).reshape(6,*fw['shape']).byte().cpu().numpy()
                for name,value in predict(fw['logits'],fw['shape']).items():predictions[kind+'_gt_'+name] = value
                del fw,row,raw
            # Both modalities / ALL SIX futures / BOTH readouts precede labels.
            target = inputs.targets(job,(1,3,5))
            moving = inputs.moving(job,(1,3,5),(1.,2.,3.),predictor.provider.pcfg.grid,predictor.provider.workers)
            for name in NAMES:
                for h,index in enumerate((1,3,5)):metrics[name].update(h,predictions[name][index],target[h],moving[h])
            keys.append([str(record['scene_name']),str(record['t0_token'])])
            if save:save(snapshot())
            if len(keys)%8 == 0:print(f'FORWARD_RECOVERY_EVAL {len(keys)}/64; optimizer_steps=0',flush=True)
    finally:
        if save:save(snapshot())
    variants = {n:m.compute() for n,m in metrics.items()}
    changes = {}
    for kind in ('occ_gt','stc_gt'):
        a,b = variants[kind+'_semantic_refine'],variants[kind+'_semantic_refine_free_minus1']
        changes[kind] = dict(average=delta(b,a),per_horizon={h:delta(b['per_horizon'][h],a['per_horizon'][h]) for h in a['per_horizon']})
    return dict(windows=len(keys),keys=keys,variants=variants,calibration=copy.deepcopy(CALIBRATION),
        free_minus1_minus_raw=changes,scoring='same64/full-grid/GT ego/dense-valid; evaluation-only metadata recovery; NOT FPS')


def copy_evidence(source,destination):
    source,destination = Path(source),Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination):raise RuntimeError('existing evidence differs: '+str(destination))
    else:shutil.copyfile(source,destination)


def finish(workflow,out,c,evaluation,receipt):
    # Read-only original evidence, independent new result view. NEVER publish a
    # corrected status into the original C_train directory or rewrite weights.
    for folder in ('frozen_B','pair_audit'):
        copy_evidence(Path(workflow)/folder/'results.json',Path(out)/folder/'results.json')
        path = Path(workflow)/folder/'summary.txt'
        if path.is_file():copy_evidence(path,Path(out)/folder/'summary.txt')
    view = dict(status='complete',contract=c,epoch=1,updates=receipt['updates'],evaluations={'1':evaluation},
        scientific_updates=0,recovery=receipt)
    (Path(out)/'C_train').mkdir(parents=True,exist_ok=True)
    base.write_json(Path(out)/'C_train/results.json',view)
    text = combine(Path(c['model']['experiment']['control_results']).parent,out)
    note = ('\n===== READ-ONLY EVALUATION RECOVERY =====\n'
        f'Source epoch1/update={receipt["updates"]}; ZERO training/optimizer steps; source Adam/RNG/checkpoints unchanged.\n'
        'Legacy token metadata normalized to last4; actual four inputs/six futures unchanged.\n'
        'Original C results untouched; this is a new frozen evaluation, not a continued training run.\n')
    (Path(out)/'summary.txt').write_text(text+note,encoding='utf-8')
    joined = json.loads((Path(out)/'results.json').read_text(encoding='utf-8'))
    joined.update(recovery=receipt,scientific_updates=0)
    base.write_json(Path(out)/'results.json',joined)
    base.write_json(Path(out)/'recovery_receipt.json',receipt)
    return text+note


def main(argv=None,event=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow-dir','out-dir','dataroot','dev-info','dev-cache','stc-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--config',default='configs/real_motion_occfm.yaml');p.add_argument('--checkpoint')
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    a = p.parse_args(argv);event = event or threading.Event();workflow = Path(a.workflow_dir).resolve();out = Path(a.out_dir).resolve()
    if out == workflow or workflow/'C_train' == out or workflow/'C_train' in out.parents:
        raise RuntimeError('NEW output outside source C_train required')
    if a.device.startswith('cuda') and (torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('ONE BF16 GPU required')
    if not 1 <= a.cpu_workers <= 4:raise ValueError('workers1..4 required')
    torch.set_num_threads(1);torch.manual_seed(21)
    checkpoint = Path(a.checkpoint or workflow/'C_train/epoch_0001.pt').resolve()
    body,c,receipt = load_source(checkpoint,a.device)
    for name in ('config','dev_info','dev_cache'):
        if sha(getattr(a,name)) != c['data'][name]:raise RuntimeError('source input differs: '+name)
    cp,_ = reference(Path(c['model']['experiment']['control_results']).parent)
    if sha(cp) != c['model']['experiment']['control_results_sha256']:raise RuntimeError('control report changed')
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config,c['overrides']))
    _,records = base.load_cache(a.dev_cache);records = base.align_records(records,c['eval_keys'])
    if len(records)!=64:raise RuntimeError('exact same64 required')
    inputs = DenseInputs(a.dataroot,a.dev_info,stc_root=a.stc_root,cache_mib=256)
    jobs = [window(r) for r in records]
    inventory = set()
    for job in jobs:
        for token in job.history_tokens:
            path = inputs.path('stc',job.scene_name,token).resolve();size,mtime = inputs.stat(path)
            inventory.add((str(path),size,mtime))
    if [list(row) for row in sorted(inventory)] != c['STC_inventory']:
        raise RuntimeError('source STC history inventory changed')
    root = Path(__file__).resolve().parents[2]
    contract = json.loads(json.dumps(dict(protocol=PROTOCOL,source=receipt,source_contract=c,keys=c['eval_keys'],
        data={kind:inputs.preflight(jobs,kind) for kind in ('occ','stc')},calibration=CALIBRATION,
        repair='metadata suffix4 ONLY; actual input arrays/poses, trajectory and six futures unchanged',
        metadata_history_token_counts=sorted({len(r['history_tokens']) for r in records}),
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device,workers=a.cpu_workers),
        implementation={f.relative_to(root).as_posix():sha(f) for f in Path(__file__).parent.iterdir() if f.suffix in ('.py','.sh')})))
    from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
    lock = out.parent/'.forward_recovery_locks'/base.stable_json_fingerprint(str(out));lock.mkdir(parents=True,exist_ok=True)
    predictor = RecoveryPredictor(body,pcfg,a.device,workers=a.cpu_workers)
    before = tensor_state_fingerprint(body.state_dict());start = time.perf_counter()
    try:
        with evaluation_lock(lock):
            if not a.resume:out.mkdir(parents=True,exist_ok=False)
            elif not out.is_dir():raise RuntimeError('resume original recovery output only')
            saved = json.loads((out/'state.json').read_text()) if a.resume and (out/'state.json').exists() else None
            existing = out/'contract.json'
            if a.resume and existing.exists() and json.loads(existing.read_text()) != contract:
                raise RuntimeError('recovery source/code/data changed; original output preserved')
            base.write_json(out/'contract.json',contract)
            def store(value):
                base.write_json(out/'state.json',value)
                base.write_json(out/'progress.json',dict(windows=len(value['keys']),total=64,scientific_updates=0,seconds=time.perf_counter()-start))
            print('FROZEN_RECOVERY source_update='+str(receipt['updates'])+' metadata_tokens='+
                str(contract['metadata_history_token_counts'])+' actual_histories=4; NO TRAINING',flush=True)
            value = evaluate(body,predictor,inputs,records,contract,saved=saved,save=store,event=event)
            if value['windows']!=64 or value['keys']!=c['eval_keys']:raise RuntimeError('incomplete same64')
            base.write_json(out/'evaluation.json',value)
            print(finish(workflow,out,c,value,receipt),flush=True)
    except InterruptedError:
        return 130
    finally:
        predictor.close()
        if tensor_state_fingerprint(body.state_dict()) != before:raise RuntimeError('frozen model changed')
        if sha(checkpoint) != receipt['sha256']:raise RuntimeError('source checkpoint changed')
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
