#!/usr/bin/env python3
"""Three ego-head epochs using exactly the completed pilot's existing feature bank."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from copy import deepcopy
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

from tools.ego_experiments import train_surface_ego_full as full
from tools.ego_experiments import screen_surface_ego_partial as pilot
from tools.ego_experiments import eval_surface_ego_full as old_dev
from tools.real_motion.ego_trajectory_common import atomic_save, digest_file, implementation_fingerprint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint, recover_original_initialization
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.waymo_zero_shot_common import write_json

ROOT = full.ROOT
PROTOCOL = 'surface_frozen_ego_existing_prefix_three_epochs_v1'
SELECTED_PROTOCOL = PROTOCOL+'_selected_head'
IMPLEMENTATION = ('tools/ego_experiments/train_surface_ego_three.py',
    'tools/ego_experiments/eval_surface_ego_three.py',
    'tools/real_motion/run_p0_f9_surface_ego_three.sh')
pack, report, clone_state, geometry_loss = full.pack, full.report, full.clone_state, full.geometry_loss


def validate_selected(saved, *, root=None):
    root = root or ROOT
    c = saved['contract']
    if (saved.get('protocol') != SELECTED_PROTOCOL or saved.get('evaluation_only') is not True or
            saved.get('training_completed') is not True or saved.get('config') != c['head_config'] or
            not saved['selected_epoch'] == saved['completed_epochs'] == c['schedule']['epochs'] == 3 or
            tensor_state_fingerprint(saved['state_dict']) != saved['state_fingerprint']):
        raise RuntimeError('completed final-epoch3 ego export required')
    if implementation_fingerprint(root) != c['feature_geometry_implementation']:
        raise RuntimeError('historical feature implementation changed')
    if ({p:digest_file(Path(root)/p) for p in full.IMPLEMENTATION} != c['implementation'] or
            {p:digest_file(Path(root)/p) for p in IMPLEMENTATION} != c['three_epoch']['implementation']):
        raise RuntimeError('training/evaluation implementation changed')
    if digest_file(c['frozen_checkpoint']) != c['frozen_sha256']:
        raise RuntimeError('frozen WM/CCR changed')
    return c


def train(head, rows, contract, out, *, epochs=3, batch_size=64, seed=21,
          lr=3e-4, min_lr=3e-6, checkpoint_every=32, resume=False, stop_event=None):
    if (epochs != 3 or min(batch_size, checkpoint_every) < 1 or not 0 < min_lr <= lr or
            not all(math.isfinite(v) for v in (lr, min_lr))): raise ValueError('invalid full-head schedule')
    if contract['schedule'] != dict(epochs=epochs, batch_size=batch_size, lr=lr,
            min_lr=min_lr, seed=seed, radius_m=10.):
        raise RuntimeError('three-epoch schedule differs from recovery contract')
    out = Path(out); path = out/'last.pt'; device = next(head.parameters()).device
    ids = contract['split']['train_indices']; held = contract['split']['holdout_indices']
    if (not ids or not held or set(ids) & set(held) or sorted(ids+held) != list(range(len(rows))) or
            {rows[i]['key'][0] for i in ids} & {rows[i]['key'][0] for i in held}):
        raise ValueError('complete scene-disjoint TRAIN split required')
    initial_sha = tensor_state_fingerprint(head.state_dict()); packed = pack(rows, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
    rng = np.random.default_rng(seed); epoch = cursor = updates = 0
    order = []; history = []; epoch_sum = 0.; epoch_count = 0; baseline = None; status = 'running'
    steps = epochs*math.ceil(len(ids)/batch_size); saved = None
    if resume:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved['protocol'] != PROTOCOL or saved['contract'] != contract or
                saved['initial_weights'] != initial_sha or saved['config'] != asdict(head.config)):
            raise RuntimeError('full-head resume contract/initializer changed')
        epoch, cursor, updates, epoch_sum, epoch_count, order, history, baseline, status = [saved[k] for k in
            ('epoch', 'cursor', 'updates', 'epoch_sum', 'epoch_count', 'order', 'history', 'baseline', 'status')]
        per_epoch = math.ceil(len(ids)/batch_size)
        if (type(epoch) != int or type(cursor) != int or type(updates) != int or not 0 <= epoch <= epochs or
                not 0 <= cursor <= len(ids) or updates != epoch*per_epoch+math.ceil(cursor/batch_size) or
                (cursor and cursor != min(math.ceil(cursor/batch_size)*batch_size, len(ids))) or
                epoch_count != cursor or not math.isfinite(epoch_sum) or epoch_sum < 0 or
                (order and sorted(order) != sorted(ids)) or (cursor and not order) or
                len(history) != epoch or not math.isfinite(baseline['geometry_loss']) or
                status not in ('running', 'stopped', 'complete') or
                (status == 'complete' and (epoch != epochs or cursor or order))):
            raise RuntimeError('invalid full-head recovery cursor/history')
        head.load_state_dict(saved['state_dict'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['sampling_rng']; torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng'])
        if saved['cuda_rng']:
            if len(saved['cuda_rng']) != torch.cuda.device_count(): raise RuntimeError('CUDA count changed')
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        baseline = report(head, packed, held, batch_size=batch_size)
    def persist(boundary_status):
        atomic_save(path, dict(protocol=PROTOCOL, contract=contract, config=asdict(head.config),
            initial_weights=initial_sha, state_dict=clone_state(head), optimizer=optimizer.state_dict(),
            epoch=epoch, cursor=cursor, updates=updates, epoch_sum=epoch_sum, epoch_count=epoch_count,
            order=list(order), history=list(history), baseline=baseline, status=boundary_status,
            sampling_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], python_rng=random.getstate()))
    if status != 'complete':
        persist('running')
        while epoch < epochs:
            if not order: order = rng.permutation(ids).tolist()
            while cursor < len(order):
                if stop_event is not None and stop_event.is_set():
                    persist('stopped'); return dict(status='stopped', updates=updates, completed_epochs=epoch)
                tick = time.perf_counter(); chosen = order[cursor:cursor+batch_size]
                rate = min_lr+(lr-min_lr)*.5*(1+math.cos(math.pi*updates/max(1, steps-1)))
                for group in optimizer.param_groups: group['lr'] = rate
                ix = torch.tensor(chosen, device=device)
                bank, targets, commands = packed; chunk = {k: v.index_select(0, ix) for k, v in bank.items()}
                head.train(); optimizer.zero_grad(set_to_none=True)
                loss = geometry_loss(head(chunk, commands.index_select(0, ix)), targets.index_select(0, ix), radius_m=10.)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss; no automatic retry')
                loss.backward(); grad = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); updates += 1; cursor += len(chosen)
                epoch_sum += float(loss.detach())*len(chosen); epoch_count += len(chosen)
                if updates % checkpoint_every == 0: persist('running')
                if updates % 32 == 0:
                    line = dict(event='ego_full_train', epoch=epoch+1, update=updates, lr=rate,
                        loss=float(loss.detach()), grad_norm=float(grad), seconds=time.perf_counter()-tick)
                    with (out/'progress.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(line)+'\n')
                    print('EGO_FULL_TRAIN '+json.dumps(line), flush=True)
            # No holdout/dev selection: always retain the requested final epoch.
            validation = report(head, packed, held, batch_size=batch_size)
            train_report = report(head, packed, ids, batch_size=batch_size)
            epoch += 1
            history.append(dict(epoch=epoch, updates=updates, train_mean_loss=epoch_sum/epoch_count,
                train=train_report, holdout=validation))
            epoch_sum = 0.; epoch_count = cursor = 0; order = []
            persist('running'); write_json(out/'epoch_history.json', history)
            print(f'EGO_FULL_EPOCH {epoch}/{epochs} train_FDE={train_report["learned"]["FDE_3s_m"]:.4f} '
                f'holdout_FDE={validation["learned"]["FDE_3s_m"]:.4f} '
                f'holdout_R10={validation["geometry_loss"]:.6f}', flush=True)
        persist('complete')
    # A derived, evaluation-only export, never a resume checkpoint. Do not rewrite
    # an already exported file: its SHA may be pinned by an interrupted dev eval.
    state = clone_state(head); final_report = history[-1]['holdout']
    exported = out/'head_epoch3.pt'; export = dict(protocol=SELECTED_PROTOCOL, contract=contract,
        config=asdict(head.config), state_dict=state, selected_epoch=epoch,
        updates=updates, holdout=final_report, baseline=baseline,
        completed_epochs=epoch, training_completed=True, evaluation_only=True,
        state_fingerprint=tensor_state_fingerprint(state))
    if exported.exists():
        old = torch.load(exported, map_location='cpu', weights_only=False)
        if (old.get('contract') != contract or old.get('state_fingerprint') != export['state_fingerprint'] or
                old.get('protocol') != SELECTED_PROTOCOL or old.get('config') != asdict(head.config) or
                old.get('selected_epoch') != epoch or old.get('completed_epochs') != epoch or
                old.get('updates') != updates or old.get('training_completed') is not True or
                old.get('evaluation_only') is not True or
                tensor_state_fingerprint(old['state_dict']) != export['state_fingerprint']):
            raise RuntimeError('existing selected export changed; do not overwrite')
    else: atomic_save(exported, export)
    write_json(out/'epoch_history.json', history)
    return dict(status='complete', updates=updates, completed_epochs=epoch, evaluated_epoch=epoch,
        holdout=final_report, baseline=baseline, last_epoch=history[-1], selected_head=str(exported),
        route='three_epoch_fixed_final_no_automatic_extension_or_promotion')


def fit(source_dir, out, *, resume=False, device='cuda', stop_event=None):
    source_dir, out = Path(source_dir).resolve(), Path(out).resolve()
    if out == source_dir or out.is_relative_to(source_dir) or source_dir.is_relative_to(out):
        raise ValueError('independent NEW output required')
    if out.exists() and not resume: raise FileExistsError('NEW output or --resume SAME directory required')
    if resume and not (out/'training.json').is_file(): raise ValueError('training recovery contract missing')
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    with evaluation_lock(source_dir):
        started = time.perf_counter()
        source_sha = digest_file(source_dir/'training.json')
        original = json.loads((source_dir/'training.json').read_text(encoding='utf-8'))
        if original.get('partial_screen',{}).get('protocol') != pilot.PROTOCOL:
            raise ValueError('completed one-epoch partial pilot required')
        old_export = torch.load(source_dir/'head_epoch1.pt',map_location='cpu',weights_only=False)
        if old_dev.validate_selected(old_export,root=ROOT) != original:
            raise RuntimeError('source pilot export/contract mismatch')
        if {p:digest_file(ROOT/p) for p in pilot.FILES} != original['partial_screen']['implementation']:
            raise RuntimeError('original pilot implementation changed')
        parent = Path(original['partial_screen']['parent'])
        if out == parent or out.is_relative_to(parent) or parent.is_relative_to(out):
            raise ValueError('output must not overlap original feature bank')
        execution = dict(device=str(torch.device(device)),torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,WM_precision='bf16' if device.startswith('cuda') else 'fp32')
        if (execution != original['execution'] or
                {k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')} != original['runtime_environment']):
            raise RuntimeError('original feature execution required; use wrapper')
        with evaluation_lock(parent):
            parent_contract = json.loads((parent/'training.json').read_text(encoding='utf-8'))
            pilot.verify_parent(parent,parent_contract,original['partial_screen']['parent_contract_sha256'])
            try:
                rows, _ = pilot.read_prefix(parent,parent_contract,len(original['keys']),
                    expected=original['partial_screen']['shards'],stop_event=stop_event)
            except InterruptedError:
                print('Stopped during read-only bank load; source unchanged.'); return 130
            contract = deepcopy(original)
            contract['schedule']['epochs'] = 3
            contract['selection'] = 'fixed final epoch3; same scene-sorted pilot population and split'
            contract['three_epoch'] = dict(protocol=PROTOCOL,source_dir=str(source_dir),
                source_contract_sha256=source_sha,initializer='original random initializer; NOT epoch1 weights',
                implementation={p:digest_file(ROOT/p) for p in IMPLEMENTATION})
            if resume:
                if json.loads((out/'training.json').read_text(encoding='utf-8')) != contract:
                    raise RuntimeError('three-epoch recovery source/schedule changed')
            else:
                out.mkdir(parents=True);write_json(out/'training.json',contract)
            with evaluation_lock(out):
                saved = torch.load(Path(original['source']['source_dir'])/'head_last.pt',
                    map_location='cpu',weights_only=False)
                head = recover_original_initialization(saved,original['frozen_checkpoint']).to(device)
                print(f'EGO_THREE: cached={len(rows)} fit={len(contract["split"]["train_indices"])} '
                    f'holdout={len(contract["split"]["holdout_indices"])}; NO extraction; RANDOM init -> 3 epochs.',flush=True)
                loaded_seconds = time.perf_counter()-started;tick = time.perf_counter()
                kw = {k:contract['schedule'][k] for k in ('epochs','batch_size','lr','min_lr','seed')}
                result = train(head,rows,contract,out,resume=resume and (out/'last.pt').is_file(),
                    stop_event=stop_event,**kw)
                result['stage_seconds'] = dict(read_existing_bank_and_initializer=loaded_seconds,
                    ego_training_and_reports=time.perf_counter()-tick)
                if digest_file(source_dir/'training.json') != source_sha: raise RuntimeError('source pilot changed')
                pilot.verify_receipt(parent,original['partial_screen']['shards'])
                if result['status'] == 'complete':
                    validate_selected(torch.load(out/'head_epoch3.pt',map_location='cpu',weights_only=False))
                write_json(out/'training_summary.json',result)
                inventory = dict(reused=sum(r['origin']=='original_read_only_bank' for r in rows),
                    fresh=sum(r['origin']=='fresh_history' for r in rows))
                text = full.training_text(result,contract,inventory).replace(
                    'ONE epoch cosine','THREE epoch whole-cycle cosine')
                text += 'Existing pilot population/split only; no new extraction; fixed epoch3, no dev selection.\n'
                (out/'summary.txt').write_text(text,encoding='utf-8');print(text,flush=True)
    return 0 if result['status'] == 'complete' else 130


def main(stop_event=None,argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    a = p.parse_args(argv)
    return fit(a.source_dir,a.out_dir,resume=a.resume,device=a.device,stop_event=stop_event)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
