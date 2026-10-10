#!/usr/bin/env python3
"""Paired THREE-epoch control heads; reuse the exact existing bank and split."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import asdict
import argparse
import json
import math
import os
import random
import signal
import threading
import time
import numpy as np
import torch
from real_motion.ego_trajectory_head import EgoHeadConfig, HistoryEgoTrajectoryHead
from tools.ego_experiments.ego_kinematic import KinematicEgoHead, KinematicPrior, PROTOCOL
from tools.ego_experiments import train_surface_ego_three as three
from tools.ego_experiments import screen_surface_ego_partial as pilot
from tools.ego_experiments.train_surface_ego_full import pack, clone_state
from tools.real_motion.ego_trajectory_common import atomic_save, digest_file
from tools.real_motion.surface_ego_ablation_common import bank_reports, geometry_loss, tensor_state_fingerprint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.waymo_zero_shot_common import write_json

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('control_history', 'control_scene')
FILES = ('tools/ego_experiments/ego_kinematic.py', 'tools/ego_experiments/train_ego_kinematic.py',
         'tools/ego_experiments/eval_ego_kinematic.py', 'tools/ego_experiments/run_ego_kinematic.sh')


def code_receipt(root=ROOT):
    return {p:digest_file(Path(root)/p) for p in FILES}


def check_execution(c, device):
    execution = dict(device=str(torch.device(device)), torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, WM_precision='bf16' if str(device).startswith('cuda') else 'fp32')
    env = {k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')}
    if execution != c['execution'] or env != c['runtime_environment']:
        raise RuntimeError('original bank runtime required; use wrapper')


@torch.no_grad()
def report(models, packed, indices):
    bank, target, nav = packed; ids = torch.tensor(indices, device=target.device)
    return bank_reports(models, {k:v[ids] for k,v in bank.items()}, target[ids], nav[ids], batch_size=64)


def train(models, legacy, rows, c, out, *, resume=False, stop_event=None, checkpoint_every=32):
    """Lockstep order/supervision/cosine; save BOTH Adam states at safe boundaries."""
    out = Path(out); device = next(legacy.parameters()).device; packed = pack(rows, device)
    ids = c['source_training']['split']['train_indices']; held = c['source_training']['split']['holdout_indices']
    if (not ids or not held or set(ids)&set(held) or sorted(ids+held) != list(range(len(rows))) or
            {rows[i]['key'][0] for i in ids}&{rows[i]['key'][0] for i in held}):
        raise ValueError('unchanged scene-disjoint TRAIN split required')
    init = {k:tensor_state_fingerprint(h.state_dict()) for k,h in models.items()}
    opts = {k:torch.optim.AdamW(h.parameters(), lr=3e-4, weight_decay=.01) for k,h in models.items()}
    rng = np.random.default_rng(c['seed']); order = []; epoch = cursor = update = 0; history = []
    baseline = None; status = 'running'; path = out/'last.pt'
    if resume:
        s = torch.load(path, map_location='cpu', weights_only=False)
        if s['contract'] != c or s['protocol'] != PROTOCOL or s['initial'] != init:
            raise RuntimeError('paired control resume contract/initializer changed')
        epoch, cursor, update, order, history, baseline, status = [s[k] for k in
            ('epoch','cursor','update','order','history','baseline','status')]
        per_epoch = math.ceil(len(ids)/64)
        if (type(epoch) != int or type(cursor) != int or type(update) != int or not 0 <= epoch <= 3 or
                not 0 <= cursor < len(ids) or update != epoch*per_epoch+math.ceil(cursor/64) or
                (cursor and cursor%64) or (order and sorted(order) != sorted(ids)) or
                (cursor and not order) or len(history) != epoch or
                status not in ('running','stopped','complete') or
                (status == 'complete' and (epoch != 3 or cursor or order))):
            raise RuntimeError('invalid control-head recovery cursor')
        for k,h in models.items(): h.load_state_dict(s['models'][k]); opts[k].load_state_dict(s['optimizers'][k])
        rng.bit_generator.state = s['rng']; torch.set_rng_state(s['torch_rng']); random.setstate(s['python_rng'])
        if s['cuda_rng']:
            if len(s['cuda_rng']) != torch.cuda.device_count(): raise RuntimeError('CUDA count changed')
            torch.cuda.set_rng_state_all(s['cuda_rng'])
    else:
        baseline = report(dict(old_epoch3=legacy, kinematic_prior=KinematicPrior(legacy.config).to(device), **models), packed, held)
    def persist(status):
        atomic_save(path, dict(protocol=PROTOCOL, contract=c, initial=init, epoch=epoch, cursor=cursor,
            update=update, order=order, history=history, baseline=baseline, status=status,
            models={k:clone_state(h) for k,h in models.items()}, optimizers={k:o.state_dict() for k,o in opts.items()},
            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
    steps = 3*math.ceil(len(ids)/64)
    if status != 'complete':
        persist('running')
        while epoch < 3:
            if not order: order = rng.permutation(ids).tolist()
            while cursor < len(ids):
                if stop_event is not None and stop_event.is_set():
                    persist('stopped'); return dict(status='stopped', update=update, epochs=epoch)
                ix = torch.tensor(order[cursor:cursor+64], device=device)
                bank, target, nav = packed; chunk = {k:v[ix] for k,v in bank.items()}
                lr = 3e-6+(3e-4-3e-6)*.5*(1+math.cos(math.pi*update/max(1,steps-1)))
                losses = {}
                for k,h in models.items():
                    h.train(); opt = opts[k]; opt.zero_grad(set_to_none=True)
                    for g in opt.param_groups: g['lr'] = lr
                    loss = geometry_loss(h(chunk,nav[ix]),target[ix],radius_m=10.)
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite control-head loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(h.parameters(),5.,error_if_nonfinite=True)
                    opt.step(); losses[k] = float(loss.detach())
                cursor += len(ix); update += 1
                # Normalize final partial batch to the NEXT epoch boundary before
                # persisting, so recovery never has a half-committed epoch report.
                if cursor < len(ids) and update%checkpoint_every == 0: persist('running')
                if update%32 == 0: print('EGO_CONTROL '+json.dumps(dict(epoch=epoch+1,update=update,lr=lr,loss=losses)),flush=True)
            held_report = report(dict(old_epoch3=legacy, kinematic_prior=KinematicPrior(legacy.config).to(device), **models), packed, held)
            train_report = report(models, packed, ids)
            epoch += 1; cursor = 0; order = []
            history.append(dict(epoch=epoch, train=train_report, holdout=held_report))
            persist('running'); write_json(out/'epoch_history.json', history)
        persist('complete')
    weights = {k:clone_state(h) for k,h in models.items()}
    export = dict(protocol=PROTOCOL+'_final3',contract=c,config=asdict(legacy.config),models=weights,
        state_fingerprints={k:tensor_state_fingerprint(v) for k,v in weights.items()}, epochs=3,updates=update,evaluation_only=True)
    dest = out/'heads_epoch3.pt'
    if dest.exists():
        old = torch.load(dest,map_location='cpu',weights_only=False)
        if old['contract'] != c or old['state_fingerprints'] != export['state_fingerprints']:
            raise RuntimeError('existing derived export differs; never overwrite')
        for k,v in old['models'].items():
            if tensor_state_fingerprint(v) != export['state_fingerprints'][k]: raise RuntimeError('derived weights changed')
    else: atomic_save(dest,export)
    result = dict(status='complete', epochs=3, updates=update, history=history, baseline=baseline,
        parameters={k:sum(p.numel() for p in h.parameters()) for k,h in models.items()},
        head_checkpoint=str(dest), selection='fixed epoch3, BOTH arms reported; no dev/holdout model selection')
    write_json(out/'training_summary.json',result)
    return result


def fit(source_dir, out, *, device='cuda', resume=False, stop_event=None):
    source_dir, out = Path(source_dir).resolve(), Path(out).resolve()
    if out == source_dir or out.is_relative_to(source_dir) or source_dir.is_relative_to(out):
        raise ValueError('independent NEW output required')
    if out.exists() and not resume: raise FileExistsError('NEW output or --resume SAME directory')
    if resume and not (out/'training.json').is_file(): raise RuntimeError('missing recovery contract')
    torch.set_num_threads(1); tick = time.perf_counter()
    with evaluation_lock(source_dir):
        saved = torch.load(source_dir/'head_epoch3.pt',map_location='cpu',weights_only=False)
        original = three.validate_selected(saved); check_execution(original,device)
        if json.loads((source_dir/'training.json').read_text(encoding='utf-8')) != original:
            raise RuntimeError('source epoch3 export/contract mismatch')
        parent = Path(original['partial_screen']['parent'])
        if out == parent or out.is_relative_to(parent) or parent.is_relative_to(out): raise ValueError('output overlaps bank')
        with evaluation_lock(parent):
            pc = json.loads((parent/'training.json').read_text(encoding='utf-8'))
            pilot.verify_parent(parent,pc,original['partial_screen']['parent_contract_sha256'])
            try:
                rows, _ = pilot.read_prefix(parent,pc,len(original['keys']),
                    expected=original['partial_screen']['shards'],stop_event=stop_event)
            except InterruptedError: return 130
            c = dict(protocol=PROTOCOL,source_dir=str(source_dir),source_head_sha256=digest_file(source_dir/'head_epoch3.pt'),
                source_contract_sha256=digest_file(source_dir/'training.json'),source_training=original,
                seed=21,epochs=3,batch_size=64,lr=3e-4,min_lr=3e-6,radius_m=10.,
                implementation=code_receipt(),arms=list(ARMS),selection='fixed final3 BOTH, no automatic promotion')
            if resume:
                if json.loads((out/'training.json').read_text(encoding='utf-8')) != c: raise RuntimeError('control source/code changed')
            else: out.mkdir(parents=True);write_json(out/'training.json',c)
            with evaluation_lock(out):
                cfg = EgoHeadConfig(**saved['config']); torch.manual_seed(21)
                models = {k:KinematicEgoHead(cfg,scene=k=='control_scene').to(device) for k in ARMS}
                legacy = HistoryEgoTrajectoryHead(cfg).to(device).eval().requires_grad_(False)
                legacy.load_state_dict(saved['state_dict'])
                loaded = time.perf_counter()-tick; tick = time.perf_counter()
                result = train(models,legacy,rows,c,out,resume=resume and (out/'last.pt').is_file(),stop_event=stop_event)
                if digest_file(source_dir/'head_epoch3.pt') != c['source_head_sha256']: raise RuntimeError('source weights changed')
                pilot.verify_receipt(parent,original['partial_screen']['shards'])
                if result['status'] == 'complete':
                    result['seconds'] = dict(read_existing=loaded,paired_training_and_reports=time.perf_counter()-tick)
                    write_json(out/'training_summary.json',result)
                    lines = ['===== EGO KINEMATIC / THREE EPOCHS, NO NEW FEATURES =====',
                        f'windows={len(rows)} fit={len(original["split"]["train_indices"])} holdout={len(original["split"]["holdout_indices"])}',
                        f'updates_each={result["updates"]}; parameters={result["parameters"]}',
                        'Same existing bank/split/R10m supervision/3-epoch cosine; WM/CCR unchanged.',
                        'TRAIN scene-holdout, NOT DEV: arm ADE / FDE3s / yaw3s']
                    for k,r in result['history'][-1]['holdout'].items():
                        lines.append(f'{k:20} {r["ADE_m"]:.4f} {r["FDE_3s_m"]:.4f} {r["yaw_mean_deg"][-1]:.4f}')
                    lines += [result['selection'],'seconds='+json.dumps(result['seconds'])]
                    text = '\n'.join(lines)+'\n';(out/'summary.txt').write_text(text,encoding='utf-8');print(text,flush=True)
    return 0 if result['status'] == 'complete' else 130


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:event.set())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    a = p.parse_args();sys.exit(fit(a.source_dir,a.out_dir,device=a.device,resume=a.resume,stop_event=event))
