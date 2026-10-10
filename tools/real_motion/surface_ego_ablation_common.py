"""Paired ego-head experiment; immutable legacy feature bank and frozen WM."""
from copy import deepcopy
from pathlib import Path
import json
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from real_motion.ego_trajectory_head import EgoHeadConfig, HistoryEgoTrajectoryHead, ego_supervision, PROTOCOL as HEAD_PROTOCOL
from tools.real_motion.ego_trajectory_common import (
    FEATURE_FIELDS, atomic_save, digest_file, fingerprint, implementation_fingerprint,
    row_fingerprint, stack_features,
)
from tools.real_motion.joint_surface_checkpoint_selection import load_evaluation_model, AVERAGE_EPOCHS

PROTOCOL = 'surface_frozen_ego_paired_budget_geometry_v1'
NEW_PYTHON_FILES = (
    'tools/real_motion/surface_ego_ablation_common.py',
    'tools/real_motion/train_p0_f9_surface_ego_ablation.py',
    'tools/real_motion/eval_p0_f9_surface_ego_ablation.py',
)
HEAD_NAMES = ('prior', 'old320', 'A_original', 'B_geometry')
TRAIN_NAMES = HEAD_NAMES[2:]


def unchanged_legacy_implementation(root, training):
    """Exclude ONLY these added experiment files, never changed feature equations.

    The first screen hashed every Python file, including unrelated entrypoints.
    Reconstruct that exact manifest to permit bank reuse without weakening gates.
    """
    root = Path(root)
    files = [*sorted((root / 'real_motion').rglob('*.py')),
             *sorted((root / 'real_motion/native').glob('*.cpp')),
             *sorted((root / 'tools/real_motion').glob('*.py'))]
    manifest = {p.relative_to(root).as_posix(): digest_file(p) for p in files
                if p.relative_to(root).as_posix() not in NEW_PYTHON_FILES}
    legacy = fingerprint(manifest)
    if legacy != training['feature_geometry_implementation']:
        raise RuntimeError('legacy feature/geometry code changed; cannot reuse this bank')
    for name, sha in training['implementation'].items():
        if digest_file(root / name) != sha:
            raise RuntimeError('original head/feature implementation changed: ' + name)
    return dict(legacy_implementation=legacy, current_implementation=implementation_fingerprint(root),
                only_added_files_excluded=list(NEW_PYTHON_FILES))


def load_source_bank(source_dir, root):
    source_dir = Path(source_dir).resolve()
    contract_path, checkpoint = source_dir / 'training.json', source_dir / 'head_last.pt'
    original_sha = {str(p): digest_file(p) for p in (contract_path, checkpoint)}
    training = json.loads(contract_path.read_text(encoding='utf-8'))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved.get('protocol') != HEAD_PROTOCOL or saved.get('contract') != training
            or saved.get('config') != training['head_config']):
        raise RuntimeError('source ego checkpoint/contract mismatch')
    schedule = training['schedule']; count = len(training['keys'])
    expected = math.ceil(count / schedule['batch_size']) * schedule['epochs']
    if (count < 1 or saved['epoch'] != schedule['epochs'] or saved['cursor'] != 0
            or saved['updates'] != expected or saved['order']):
        raise RuntimeError('completed original head and complete bank required')
    if len({tuple(k) for k in training['keys']}) != count:
        raise RuntimeError('duplicate source TRAIN identity')
    if digest_file(training['frozen_checkpoint']) != training['frozen_sha256']:
        raise RuntimeError('source frozen WM/CCR weights changed')
    compatibility = unchanged_legacy_implementation(root, training)
    rows = []; receipt = []
    for i, key in enumerate(training['keys']):
        path = source_dir / 'bank' / f'{i:06d}.pt'; sha = digest_file(path)
        row = torch.load(path, map_location='cpu', weights_only=False)
        if (row.get('key') != key or row.get('contract_fingerprint') != fingerprint(training)
                or row.get('content_fingerprint') != row_fingerprint(row)
                or set(row['features']) != set(FEATURE_FIELDS)):
            raise RuntimeError('legacy feature shard identity/content mismatch: ' + str(path))
        if digest_file(path) != sha:
            raise RuntimeError('source bank changed while reading')
        rows.append(row); receipt.append(dict(path=str(path),sha256=sha,content=row['content_fingerprint']))
    audit = dict(source_dir=str(source_dir), original_files=original_sha,
                 frozen_checkpoint=training['frozen_checkpoint'], frozen_sha256=training['frozen_sha256'],
                 original_updates=saved['updates'], original_training=training,
                 bank_receipt=receipt, **compatibility)
    verify_sources(audit)
    return rows, saved, audit


def verify_sources(audit):
    for path, sha in audit['original_files'].items():
        if digest_file(path) != sha: raise RuntimeError('original source changed: ' + path)
    if digest_file(audit['frozen_checkpoint']) != audit['frozen_sha256']:
        raise RuntimeError('frozen WM/CCR checkpoint changed')
    for row in audit['bank_receipt']:
        if digest_file(row['path']) != row['sha256']:
            raise RuntimeError('read-only source bank changed: ' + row['path'])


def recover_original_initialization(saved, frozen_path, *, loader=load_evaluation_model):
    """Reconstruct old seed + frozen-model construction, and check its RNG witness.

    The original dropout-free trainer consumes no Torch RNG after initialization.
    Fail closed if this receipt cannot establish the same initializer as old320.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(saved['contract']['schedule']['seed'])
        meta, frozen = loader(frozen_path, device='cpu')
        if meta.get('source_epochs') != list(AVERAGE_EPOCHS) or not meta.get('averaging'):
            raise RuntimeError('original frozen mean5/6/8/12/14 required')
        head = HistoryEgoTrajectoryHead(EgoHeadConfig(**saved['config']))
        if not torch.equal(torch.get_rng_state(), saved['torch_rng']):
            raise RuntimeError('original initializer RNG witness differs; do not claim budget-only comparison')
        del frozen
    return head


def historical_prior(bank):
    e = bank['ego_history']; times = torch.arange(1,7,device=e.device,dtype=e.dtype) * .5
    xy = e[:,-1,5:7,None].transpose(1,2) * 10. * times[None,:,None]
    yaw = e[:,-1,8,None] * math.pi * times[None]
    return torch.cat((xy,yaw[...,None]),-1)


def geometry_loss(output, target, *, radius_m=10.):
    """Single Smooth-L1 over FIVE fixed SE(2)-transformed reference points.

    Center and symmetric +/-10m axis points; no GT boxes/shapes/occupancy used.
    XY and angular displacement are both supervised in metres, all six horizons.
    """
    if not math.isfinite(radius_m) or radius_m <= 0: raise ValueError('positive finite fixed radius required')
    pred = output['se2']; target = torch.as_tensor(target,device=pred.device,dtype=pred.dtype)
    if pred.ndim != 3 or pred.shape[1:] != (6,3) or pred.shape != target.shape or not torch.isfinite(target).all():
        raise ValueError('finite absolute six-horizon XY/yaw targets required')
    points = pred.new_tensor([[0,0],[radius_m,0],[-radius_m,0],[0,radius_m],[0,-radius_m]])
    def transform(a):
        cs, sn = a[...,2].cos()[...,None], a[...,2].sin()[...,None]
        x = cs * points[:,0] - sn * points[:,1] + a[...,0,None]
        y = sn * points[:,0] + cs * points[:,1] + a[...,1,None]
        return torch.stack((x,y),-1)
    return F.smooth_l1_loss(transform(pred),transform(target))


def trajectory_report(pred, target, commands):
    p, t, c = np.asarray(pred), np.asarray(target), np.asarray(commands)
    if p.shape != t.shape or p.shape[1:] != (6,3) or c.shape != p.shape[:2] or not len(p):
        raise ValueError('complete nonempty six-horizon trajectory population required')
    if not np.isfinite(p).all() or not np.isfinite(t).all() or not np.isin(c,[0,1,2]).all():
        raise ValueError('finite trajectories/valid commands required')
    xy = np.linalg.norm(p[...,:2]-t[...,:2],axis=-1)
    d = p[...,2]-t[...,2]; yaw = np.abs(np.arctan2(np.sin(d),np.cos(d))) * 180 / np.pi
    return dict(windows=len(p), ADE_m=float(xy.mean()), FDE_3s_m=float(xy[:,-1].mean()),
        xy_mean_m=xy.mean(0).tolist(), xy_median_m=np.median(xy,0).tolist(), xy_p90_m=np.quantile(xy,.9,axis=0).tolist(),
        yaw_mean_deg=yaw.mean(0).tolist(), yaw_median_deg=np.median(yaw,0).tolist(), yaw_p90_deg=np.quantile(yaw,.9,axis=0).tolist(),
        command_counts_by_horizon=np.stack([(c==k).sum(0) for k in range(3)],axis=1).tolist(),
        command_order=['right','left','straight'])


@torch.no_grad()
def bank_reports(models, bank, targets, commands, *, batch_size=64):
    predictions = {k:[] for k in models}; predictions['prior'] = []
    for start in range(0,len(targets),batch_size):
        sl = slice(start,start+batch_size); features = {k:v[sl] for k,v in bank.items()}
        predictions['prior'].append(historical_prior(features).cpu().numpy())
        for name, model in models.items():
            model.eval(); predictions[name].append(model(features,commands[sl])['se2'].cpu().numpy())
    t, c = targets.cpu().numpy(), commands.cpu().numpy()
    return {k:trajectory_report(np.concatenate(v),t,c) for k,v in predictions.items()}


def paired_train(initial, rows, old_head, contract, out, *, max_updates=2000, batch_size=64,
                 lr=3e-4, min_lr=3e-6, seed=21, radius_m=10., checkpoint_every=32,
                 monitor_every=200, resume=False, stop_event=None):
    """One atomic lockstep update for A/B; same initializer, order and schedule."""
    if (min(max_updates,batch_size,checkpoint_every,monitor_every)<1 or not 0<min_lr<=lr
            or not all(math.isfinite(v) for v in (lr,min_lr,radius_m)) or radius_m<=0):
        raise ValueError('invalid paired training configuration')
    out = Path(out); path = out/'pair_last.pt'; device = next(initial.parameters()).device
    models = {name:deepcopy(initial) for name in TRAIN_NAMES}
    opts = {name:torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01) for name,model in models.items()}
    rng = np.random.default_rng(seed); update=epoch=cursor=0; order=[]; monitors=[]
    sums = {k:dict(loss=0.,xy=0.,yaw=0.,windows=0) for k in TRAIN_NAMES}
    bank = stack_features([r['features'] for r in rows],device)
    targets = torch.as_tensor(np.stack([r['target'] for r in rows]),device=device,dtype=torch.float32)
    commands = torch.as_tensor(np.stack([r['commands'] for r in rows]),device=device,dtype=torch.long)
    initial_sha = tensor_state_fingerprint(initial.state_dict()); tick=time.perf_counter()
    if resume:
        if not path.is_file(): raise RuntimeError('paired resume checkpoint missing')
        saved = torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('protocol') != PROTOCOL or saved.get('contract') != contract or saved.get('initial_weights') != initial_sha:
            raise RuntimeError('paired resume contract/initializer changed')
        update,epoch,cursor,order = [saved[k] for k in ('updates','epoch','cursor','order')]
        if (type(update)!=int or type(epoch)!=int or type(cursor)!=int or not 0<=update<=max_updates or not 0<=cursor<len(rows)
                or (order and sorted(order)!=list(range(len(rows)))) or (cursor and not order) or (not cursor and order)):
            raise RuntimeError('corrupt paired training cursor')
        # Epoch/cursor must correspond exactly to the count of completed batches.
        per_epoch = math.ceil(len(rows)/batch_size)
        if epoch != update//per_epoch or cursor != min((update%per_epoch)*batch_size,len(rows)):
            raise RuntimeError('paired update/epoch/cursor mismatch')
        for name in TRAIN_NAMES:
            models[name].load_state_dict(saved['models'][name],strict=True)
            opts[name].load_state_dict(saved['optimizers'][name])
        rng.bit_generator.state=saved['sampling_rng'];torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng']);monitors=saved['monitors'];sums=saved['epoch_sums']
        if (set(sums)!=set(TRAIN_NAMES) or any(v['windows']!=cursor or
                any(not math.isfinite(v[k]) or v[k]<0 for k in ('loss','xy','yaw')) for v in sums.values())):
            raise RuntimeError('paired epoch-mean prefix is corrupt')
        if saved['cuda_rng']:
            if len(saved['cuda_rng'])!=torch.cuda.device_count():raise RuntimeError('CUDA device count changed')
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if update==max_updates:
            # An eval-only resume must not rewrite the already frozen paired
            # checkpoint, otherwise its SHA invalidates the saved eval prefix.
            return dict(status='complete',updates=update,completed_epochs=epoch,cursor=cursor,
                checkpoint=str(path),seconds_this_invocation=time.perf_counter()-tick,initial_weights=initial_sha,
                TRAIN_reports=monitors[-1]['reports'],last_TRAIN_monitor_update=monitors[-1]['update'])
    def persist():
        atomic_save(path,dict(protocol=PROTOCOL,contract=contract,config=initial.config.__dict__,initial_weights=initial_sha,
            models={k:m.state_dict() for k,m in models.items()},optimizers={k:o.state_dict() for k,o in opts.items()},
            updates=update,epoch=epoch,cursor=cursor,order=order,sampling_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            python_rng=random.getstate(),monitors=monitors,epoch_sums=sums))
    def monitor():
        reports = bank_reports({'old320':old_head,**models},bank,targets,commands,batch_size=batch_size)
        row = dict(event='TRAIN_full_bank',update=update,reports=reports)
        monitors.append(row);append_log(out,row)
        print('EGO_AB_TRAIN '+json.dumps({k:dict(ADE_m=v['ADE_m'],FDE_3s_m=v['FDE_3s_m']) for k,v in reports.items()}),flush=True)
    if not monitors: monitor()
    persist()
    while update<max_updates:
        if stop_event is not None and stop_event.is_set():break
        if not order:order=rng.permutation(len(rows)).tolist()
        ids=torch.as_tensor(order[cursor:cursor+batch_size],device=device)
        features={k:v[ids] for k,v in bank.items()}; target=targets[ids]; command=commands[ids]
        rate=min_lr+(lr-min_lr)*.5*(1+math.cos(math.pi*update/max(1,max_updates-1)))
        log=dict(event='paired_update',update=update+1,epoch=epoch+1,lr=rate,windows=len(ids))
        for name in TRAIN_NAMES:
            model,opt=models[name],opts[name];model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=rate
            output=model(features,command);original,components=ego_supervision(output,target,yaw_weight=model.config.yaw_weight)
            loss=original if name=='A_original' else geometry_loss(output,target,radius_m=radius_m)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite paired loss; no automatic retry')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            values=dict(loss=float(loss.detach()),xy=float(components['xy']),yaw=float(components['yaw']),grad_norm=float(grad))
            log[name]=values
            for k in ('loss','xy','yaw'):sums[name][k]+=values[k]*len(ids)
            sums[name]['windows']+=len(ids)
        update+=1;cursor+=len(ids)
        if cursor==len(rows):
            epoch+=1;cursor=0;order=[]
            means={name:{k:values[k]/values['windows'] for k in ('loss','xy','yaw')} for name,values in sums.items()}
            append_log(out,dict(event='paired_epoch_mean',epoch=epoch,update=update,windows=len(rows),means=means))
            sums={k:dict(loss=0.,xy=0.,yaw=0.,windows=0) for k in TRAIN_NAMES}
        if update%16==0:append_log(out,log)
        if update%monitor_every==0 or update==max_updates:monitor()
        if update%checkpoint_every==0:persist()
        if update%100==0:print(f'EGO_AB update={update}/{max_updates} epoch={epoch} lr={rate:.8f}',flush=True)
    persist()
    return dict(status='complete' if update==max_updates else 'stopped',updates=update,completed_epochs=epoch,
        cursor=cursor,checkpoint=str(path),seconds_this_invocation=time.perf_counter()-tick,
        initial_weights=initial_sha,TRAIN_reports=monitors[-1]['reports'],last_TRAIN_monitor_update=monitors[-1]['update'])


def tensor_state_fingerprint(state):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def append_log(out, value):
    with (Path(out)/'progress.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n')
