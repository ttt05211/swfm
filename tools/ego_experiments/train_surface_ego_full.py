#!/usr/bin/env python3
"""One expanded ego-head epoch; frozen WM, scene-held-out TRAIN diagnostics."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import argparse
import gc
import hashlib
import json
import math
import os
import random
import signal
import threading
import time

import numpy as np
import torch

from real_motion.ego_navigation import navigation_commands, relative_se2, validate_window_identity
from real_motion.ego_trajectory_head import EgoHeadConfig
from real_motion.nuscenes_adapter import NuScenesWindowSource
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from tools.real_motion.ego_trajectory_common import (
    atomic_save, digest_file, fingerprint, row_fingerprint,
    stack_features, extract_history_features, implementation_fingerprint,
)
from tools.real_motion.surface_ego_ablation_common import (
    load_source_bank, verify_sources, recover_original_initialization,
    geometry_loss, historical_prior, trajectory_report, tensor_state_fingerprint,
)
from tools.real_motion.joint_surface_checkpoint_selection import load_evaluation_model
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.waymo_zero_shot_common import write_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'surface_frozen_ego_full_one_epoch_geometry_v1'
SELECTED_PROTOCOL = PROTOCOL+'_selected_head'
IMPLEMENTATION = ('tools/ego_experiments/train_surface_ego_full.py',
    'tools/ego_experiments/eval_surface_ego_full.py',
    'tools/real_motion/run_p0_f9_surface_ego_full.sh')


def scene_split(keys, *, fraction=.1, seed=21):
    if not 0 < fraction < .5 or len(set(map(tuple, keys))) != len(keys):
        raise ValueError('unique keys and bounded scene holdout fraction required')
    scenes = sorted({k[0] for k in keys})
    if len(scenes) < 2:
        raise ValueError('at least two TRAIN scenes required')
    ordered = sorted(scenes, key=lambda s: (hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest(), s))
    held = set(ordered[:max(1, math.ceil(len(scenes)*fraction))])
    train = [i for i, k in enumerate(keys) if k[0] not in held]
    validation = [i for i, k in enumerate(keys) if k[0] in held]
    if not train or not validation:
        raise ValueError('nonempty scene-disjoint fit/holdout required')
    return dict(train_indices=train, holdout_indices=validation, holdout_scenes=sorted(held),
        train_scenes=sorted(set(scenes)-held), fraction=fraction, seed=seed,
        rule='SHA256(seed:scene); entire scenes; no target/model/error selection')


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class HistoryReader:
    """Bounded raw-history LRU and ordered prefetch; never opens future NPZs."""
    def __init__(self, source, *, ram_mib=512):
        self.source = source; self.limit = int(ram_mib*1024**2)
        self.frames = OrderedDict(); self.bytes = 0; self.lock = threading.Lock()

    def frame(self, scene, token):
        p = self.source._label_path(scene, token); st = p.stat()
        key = (str(p), st.st_size, st.st_mtime_ns)
        with self.lock:
            if key in self.frames:
                self.frames.move_to_end(key); return self.frames[key]
        a, m = self.source.load_occ3d(scene, token)
        a = np.array(a, copy=True); m = np.array(m, copy=True)
        a.setflags(write=False); m.setflags(write=False); size = a.nbytes+m.nbytes
        with self.lock:
            if key in self.frames: return self.frames[key]
            if size <= self.limit:
                while self.frames and self.bytes+size > self.limit:
                    _, old = self.frames.popitem(last=False); self.bytes -= sum(x.nbytes for x in old)
                self.frames[key] = (a, m); self.bytes += size
        return a, m

    def __call__(self, rec):
        loaded = [self.frame(rec['scene_name'], t) for t in rec['history_tokens']]
        return dict(history_occ=np.stack([a for a, _ in loaded]),
            history_observed=np.stack([m for _, m in loaded]),
            history_poses=[self.source.pose(t) for t in rec['history_tokens']]), [
                self.source.nusc.get('sample', t)['timestamp']/1e6 for t in rec['history_tokens']]


def build_bank(out, records, contract, original_rows, source, materialize, *, stop_event=None,
               workers=4, reader=None, progress=print):
    """Each new shard is atomic; same keys/content/code required on recovery."""
    if not 1 <= workers <= 8: raise ValueError('bounded positive workers required')
    out = Path(out); directory = out/'bank'; directory.mkdir(exist_ok=True)
    old = {tuple(r['key']): r for r in original_rows}; sha = fingerprint(contract)
    reader = reader or HistoryReader(source)
    expected = [(r['scene_name'], r['t0_token']) for r in records]
    missing = []; receipt = []; reused = fresh = 0
    for i, key in enumerate(expected):
        path = directory/f'{i:06d}.pt'
        if path.exists():
            row = torch.load(path, map_location='cpu', weights_only=False)
            if (row['key'] != list(key) or row['contract_fingerprint'] != sha or
                    row['content_fingerprint'] != row_fingerprint(row) or
                    row['origin'] not in ('original_read_only_bank', 'fresh_history')):
                raise RuntimeError('new bank shard identity/content changed: '+str(path))
            reused += row['origin'] == 'original_read_only_bank'; fresh += row['origin'] == 'fresh_history'
        else:
            missing.append(i)
    # Only missing, non-reusable raw histories enter the bounded CPU pipeline.
    jobs = [i for i in missing if expected[i] not in old]; next_job = 0; pending = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def fill():
            nonlocal next_job
            while next_job < len(jobs) and len(pending) < workers:
                i = jobs[next_job]; next_job += 1
                pending[i] = pool.submit(reader, records[i])
        fill()
        for i in missing:
            if stop_event is not None and stop_event.is_set(): break
            key = expected[i]; r = records[i]; tick = time.perf_counter()
            if key in old:
                src = old[key]
                row = dict(features=src['features'], target=src['target'], commands=src['commands'],
                           origin='original_read_only_bank'); reused += 1
            else:
                raw, timestamps = pending.pop(i).result(); fill()
                features = materialize(r, raw, timestamps)
                # Future supervision is built AFTER historical features; never future occupancy.
                target = relative_se2(raw['history_poses'][-1], [source.pose(t) for t in r['future_tokens']])
                commands = navigation_commands(source.nusc, r['future_tokens'])
                row = dict(features=features, target=target, commands=commands, origin='fresh_history'); fresh += 1
            row.update(key=list(key), contract_fingerprint=sha); row['content_fingerprint'] = row_fingerprint(row)
            atomic_save(directory/f'{i:06d}.pt', row)
            if progress and (i+1) % 64 == 0:
                progress('EGO_FULL_BANK '+json.dumps(dict(window=i+1, total=len(records),
                    reused=reused, fresh=fresh, last_seconds=time.perf_counter()-tick)), flush=True)
    for i in range(len(records)):
        p = directory/f'{i:06d}.pt'
        if p.exists(): receipt.append(dict(index=i, sha256=digest_file(p)))
    inventory = dict(status='complete' if len(receipt) == len(records) else 'stopped',
        windows=len(receipt), total=len(records), reused=reused, fresh=fresh,
        disk_mib=sum((directory/f'{r["index"]:06d}.pt').stat().st_size for r in receipt)/1024**2,
        contract_fingerprint=sha, shards=receipt)
    write_json(out/'bank_inventory.json', inventory)
    return inventory


def load_bank(out, contract, inventory, *, max_mib=4096):
    sha = fingerprint(contract)
    if (inventory['status'] != 'complete' or inventory['contract_fingerprint'] != sha or
            inventory['windows'] != len(contract['keys']) or inventory['total'] != len(contract['keys']) or
            len(inventory['shards']) != len(contract['keys'])):
        raise RuntimeError('complete same-contract full bank required')
    rows = []; size = 0
    for i, entry in enumerate(inventory['shards']):
        p = Path(out)/'bank'/f'{i:06d}.pt'
        if entry['index'] != i or digest_file(p) != entry['sha256']: raise RuntimeError('bank inventory changed')
        r = torch.load(p, map_location='cpu', weights_only=False)
        if (r['key'] != contract['keys'][i] or r['content_fingerprint'] != row_fingerprint(r) or
                r['contract_fingerprint'] != sha):
            raise RuntimeError('full bank shard changed')
        size += sum(v.numel()*v.element_size() for v in r['features'].values())
        if size > max_mib*1024**2: raise RuntimeError('bounded feature-bank memory exceeded')
        rows.append(r)
    return rows


def pack(rows, device):
    return (stack_features([r['features'] for r in rows], device),
        torch.as_tensor(np.stack([r['target'] for r in rows]), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack([r['commands'] for r in rows]), dtype=torch.long, device=device))


@torch.no_grad()
def report(head, packed, indices, *, batch_size=64):
    bank, targets, commands = packed; result = []; prior = []; loss_sum = 0.
    head.eval()
    for start in range(0, len(indices), batch_size):
        ids = torch.tensor(indices[start:start+batch_size], device=targets.device)
        chunk = {k: v.index_select(0, ids) for k, v in bank.items()}
        pred = head(chunk, commands.index_select(0, ids))['se2']
        target = targets.index_select(0, ids)
        loss_sum += float(geometry_loss({'se2': pred}, target, radius_m=10.))*len(ids)
        result.append(pred.cpu().numpy()); prior.append(historical_prior(chunk).cpu().numpy())
    selected = np.asarray(indices)
    t = targets.index_select(0, torch.tensor(selected, device=targets.device)).cpu().numpy()
    c = commands.index_select(0, torch.tensor(selected, device=targets.device)).cpu().numpy()
    return dict(geometry_loss=loss_sum/len(indices), learned=trajectory_report(np.concatenate(result), t, c),
                prior=trajectory_report(np.concatenate(prior), t, c))


def train(head, rows, contract, out, *, epochs=1, batch_size=64, seed=21,
          lr=3e-4, min_lr=3e-6, checkpoint_every=32, resume=False, stop_event=None):
    if (epochs != 1 or min(batch_size, checkpoint_every) < 1 or not 0 < min_lr <= lr or
            not all(math.isfinite(v) for v in (lr, min_lr))): raise ValueError('invalid full-head schedule')
    if contract['schedule'] != dict(epochs=epochs, batch_size=batch_size, lr=lr,
            min_lr=min_lr, seed=seed, radius_m=10.):
        raise RuntimeError('one-epoch schedule differs from recovery contract')
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
    exported = out/'head_epoch1.pt'; export = dict(protocol=SELECTED_PROTOCOL, contract=contract,
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
        route='one_epoch_diagnostic_no_automatic_extension_or_promotion')


def training_text(result, contract, inventory):
    lines = ['===== EXPANDED TRAIN / FROZEN WM EGO HEAD =====',
        f'status={result["status"]}; complete_epochs={result["completed_epochs"]}; updates={result["updates"]}',
        f'population={len(contract["keys"])}; fit={len(contract["split"]["train_indices"])}; '
        f'holdout={len(contract["split"]["holdout_indices"])}; scene-disjoint inside TRAIN only.',
        f'bank reused={inventory["reused"]}; fresh={inventory["fresh"]}; WM/CCR never optimized.',
        'Same original head initializer; fixed R10m geometry; ONE epoch cosine; final epoch only.',
        'Holdout is unseen by ego-head optimizer, NOT unseen by the frozen WM trained on full TRAIN.',
        'GT-derived destination navigation-conditioned; NOT navigation-free.']
    if 'stage_seconds' in result: lines.append('stage_seconds='+json.dumps(result['stage_seconds']))
    if 'disk_mib' in inventory: lines.append(f'feature_bank_mib={inventory["disk_mib"]:.2f}')
    if result['status'] == 'complete':
        lines += [f'evaluated_epoch={result["evaluated_epoch"]}; route={result["route"]}',
            'TRAIN-HOLDOUT diagnostic: geometry='+str(result['holdout']['geometry_loss']),
            'head      ADE_m    FDE_3s_m    yaw_3s_deg']
        for name in ('prior', 'learned'):
            r = result['holdout'][name]
            lines.append(f'{name:8} {r["ADE_m"]:.6f} {r["FDE_3s_m"]:.6f} {r["yaw_mean_deg"][-1]:.6f}')
        lines.append('selected_head='+result['selected_head'])
    return '\n'.join(lines)+'\n'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source-dir', 'dataroot', 'train-cache', 'train-info', 'out-dir'): p.add_argument('--'+key, required=True)
    p.add_argument('--config', default=str(ROOT/'configs/real_motion_occfm.yaml'))
    p.add_argument('--epochs', type=int, choices=(1,), default=1); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--holdout-fraction', type=float, default=.1)
    p.add_argument('--cpu-workers', type=int, default=4); p.add_argument('--device', default='cuda')
    p.add_argument('--expected-windows', type=int, default=20430); p.add_argument('--resume', action='store_true')
    return p


def main(stop_event=None, argv=None):
    started = time.perf_counter()
    a = parser().parse_args(argv); out = Path(a.out_dir).resolve(); origin = Path(a.source_dir).resolve()
    if min(a.batch_size, a.expected_windows) < 1 or not 1 <= a.cpu_workers <= 8:
        raise ValueError('bounded positive full-head config required')
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in
        (origin, Path(a.dataroot).resolve(), Path(a.train_cache).resolve(), Path(a.train_info).resolve())):
        raise ValueError('new output outside existing inputs required')
    if out.exists() and not a.resume: raise FileExistsError('NEW full-head output required, or --resume SAME directory')
    if a.resume and not (out/'training.json').is_file(): raise ValueError('full-head recovery contract missing')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    original_rows, original_head, source_audit = load_source_bank(origin, ROOT)
    old = source_audit['original_training']; schedule = old['schedule']; cfg = EgoHeadConfig(**old['head_config'])
    execution = dict(device=str(torch.device(a.device)), torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, WM_precision='bf16' if a.device.startswith('cuda') else 'fp32')
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith('SWFM_')}
    if execution != old['feature_execution'] or env != old['runtime_environment']:
        raise RuntimeError('original feature execution/environment required; use wrapper')
    for path, sha in ((a.train_cache, old['train_cache_sha256']), (a.train_info, old['train_info_sha256']),
                      (a.config, old['runtime_config_sha256'])):
        if digest_file(path) != sha: raise RuntimeError('original TRAIN/config source changed: '+str(path))
    header, cached = load_cache(a.train_cache)
    records = sorted([dict(scene_name=str(r['scene_name']), t0_token=str(r['t0_token']),
        history_tokens=tuple(r['history_tokens'][-4:]), future_tokens=tuple(r['future_tokens'])) for r in cached],
        key=lambda r: (r['scene_name'], r['t0_token']))
    del header, cached; gc.collect()
    keys = [[r['scene_name'], r['t0_token']] for r in records]
    if len(records) != a.expected_windows: raise ValueError('full TRAIN population mismatch; no truncation allowed')
    split = scene_split(keys, fraction=a.holdout_fraction, seed=schedule['seed'])
    if not {tuple(r['key']) for r in original_rows}.issubset(map(tuple, keys)):
        raise RuntimeError('original bank is not a subset of this TRAIN population')
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.train_info)
    if any(k[0] not in source.allowed_scenes for k in keys): raise ValueError('non-TRAIN scene in full population')
    for rec in records: validate_window_identity(source.nusc, rec)
    metadata = Path(a.dataroot)/'v1.0-trainval'
    hashes = {p.name: digest_file(p) for p in sorted(metadata.glob('*.json'))}
    if hashes != old['metadata_hashes']: raise RuntimeError('original nuScenes metadata changed')
    paths = sorted({source._label_path(r['scene_name'], t) for r in records for t in r['history_tokens']})
    stat_fp = fingerprint([[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths])
    contract = dict(protocol=PROTOCOL, keys=keys, split=split, source=source_audit,
        head_config=asdict(cfg), frozen_checkpoint=source_audit['frozen_checkpoint'], frozen_sha256=source_audit['frozen_sha256'],
        schedule=dict(epochs=a.epochs, batch_size=a.batch_size,
            lr=schedule['lr'], min_lr=schedule['min_lr'], seed=schedule['seed'], radius_m=10.),
        runtime_config_sha256=old['runtime_config_sha256'], execution=execution, runtime_environment=env,
        feature_geometry_implementation=implementation_fingerprint(ROOT),
        implementation={p: digest_file(ROOT/p) for p in IMPLEMENTATION},
        historical_files_fingerprint=stat_fp, metadata_hashes=hashes,
        selection='final epoch1 only; TRAIN holdout diagnostic, NOT checkpoint selection',
        frozen_WM_has_seen_holdout_scenes=True, future_occupancy_inputs=False,
        command_condition='GT-derived destination-row; NOT navigation-free', cpu_workers=a.cpu_workers)
    if a.resume:
        if json.loads((out/'training.json').read_text(encoding='utf-8')) != contract:
            raise RuntimeError('full-head population/source/config/schedule/code changed')
    else: out.mkdir(parents=True); write_json(out/'training.json', contract)
    with evaluation_lock(out):
        stage_seconds = dict(source_checks_and_population=time.perf_counter()-started)
        tick = time.perf_counter()
        initial = recover_original_initialization(original_head, source_audit['frozen_checkpoint']).to(a.device)
        stage_seconds['original_initialization'] = time.perf_counter()-tick
        tick = time.perf_counter()
        print(f'EGO_ONE_EPOCH: population={len(keys)} fit={len(split["train_indices"])} '
            f'holdout={len(split["holdout_indices"])} old_bank={len(original_rows)}; '
            'first extract/reuse historical features, then train ONE epoch; no WM/CCR updates.', flush=True)
        need = any(not (out/'bank'/f'{i:06d}.pt').is_file() for i in range(len(records)))
        predictor = None; joint = None
        try:
            if need:
                _, joint = load_evaluation_model(source_audit['frozen_checkpoint'], device=a.device)
                frozen_before = tensor_state_fingerprint(joint.state_dict())
                predictor = Predictor(joint, make_prepare_config(load_runtime_config(a.config)), a.device,
                    workers=a.cpu_workers, graphs=False, geometry_mib=512)
            def extract(rec, raw, times):
                return extract_history_features(predictor.provider, rec, raw, cfg, timestamps_s=times)
            inventory = build_bank(out, records, contract, original_rows, source, extract,
                workers=a.cpu_workers, stop_event=stop_event)
            if joint is not None and tensor_state_fingerprint(joint.state_dict()) != frozen_before:
                raise RuntimeError('frozen WM/CCR mutated')
        finally:
            if predictor: predictor.close()
        del predictor, joint, original_rows, source; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        verify_sources(source_audit)
        if inventory['status'] != 'complete':
            print('BANK stopped safely; resume SAME directory.'); return 130
        rows = load_bank(out, contract, inventory)
        stage_seconds['historical_feature_bank'] = time.perf_counter()-tick
        tick = time.perf_counter()
        result = train(initial, rows, contract, out, epochs=a.epochs, batch_size=a.batch_size,
            lr=schedule['lr'], min_lr=schedule['min_lr'], seed=schedule['seed'], resume=a.resume and (out/'last.pt').is_file(),
            stop_event=stop_event)
        stage_seconds['ego_training_and_reports'] = time.perf_counter()-tick
        verify_sources(source_audit)
        for path, sha in ((a.train_cache, old['train_cache_sha256']), (a.train_info, old['train_info_sha256']),
                          (a.config, old['runtime_config_sha256'])):
            if digest_file(path) != sha: raise RuntimeError('TRAIN/config source changed during run')
        for name, sha in hashes.items():
            if digest_file(metadata/name) != sha: raise RuntimeError('metadata changed during training')
        if fingerprint([[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths]) != stat_fp:
            raise RuntimeError('historical source files changed during run')
        result['seconds_this_invocation'] = time.perf_counter()-started
        result['stage_seconds'] = stage_seconds
        write_json(out/'training_summary.json', result)
        summary = training_text(result, contract, inventory)
        (out/'summary.txt').write_text(summary, encoding='utf-8'); print(summary, flush=True)
    return 0 if result['status'] == 'complete' else 130


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
