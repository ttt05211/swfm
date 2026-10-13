"""Compact frozen HISTORY tokens, never future-conditioned features."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from real_motion.ego_navigation import ego_history_features, relative_se2, navigation_commands
from tools.ego_experiments.train_surface_ego_full import HistoryReader
from tools.scene_ego_head.model import HistoricalScene, prepare_scene
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.joint_training_recovery import save_resume_checkpoint

FIELDS = ('features', 'geometry', 'valid', 'history', 'commands', 'target')


@torch.no_grad()
def materialize(body, raw, times, grid, *, pool_side):
    device = next(body.parameters()).device
    e = ego_history_features(raw['history_poses'], times)
    semantics = torch.as_tensor(np.asarray(raw['history_occ']), device=device)
    observed = torch.as_tensor(np.asarray(raw['history_observed']), device=device)
    known = observed.bool() & (semantics >= 0) & (semantics < 18)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field = body.transport.history_encoder(semantics, observed)
    scene = prepare_scene(field, observed=known, history_poses=raw['history_poses'],
        ego_history=e, grid=grid, pool_side=pool_side)
    return dict(features=scene.features[0].cpu(), geometry=scene.geometry[0].cpu(),
                valid=scene.valid[0].cpu(), history=torch.from_numpy(e))


def pack(bank, ids, device):
    ids = torch.as_tensor(ids, dtype=torch.long)
    values = {k: bank[k].index_select(0, ids).to(device) for k in FIELDS}
    scene = HistoricalScene(values['features'], values['geometry'], values['valid'])
    return values['history'], values['commands'], scene, values['target']


def build(body, records, source, grid, directory, contract, *, pool_side=8,
          workers=2, stop_event=None, reader=None, chunk_size=64):
    directory = Path(directory); directory.mkdir(exist_ok=True)
    fingerprint = legacy.stable_json_fingerprint(contract)
    keys = [[r['scene_name'], r['t0_token']] for r in records]
    manifest_path = directory/'manifest.json'
    manifest = legacy.json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        contract=fingerprint, keys=keys, chunks=[], complete=False)
    if manifest['contract'] != fingerprint or manifest['keys'] != keys:
        raise RuntimeError('frozen bank source/population changed')
    reader = reader or HistoryReader(source, ram_mib=256)
    chunks = manifest['chunks']
    for i, entry in enumerate(chunks):
        path = directory/f'{i:04d}.pt'
        if entry['index'] != i or legacy.sha256(path) != entry['sha256']:
            raise RuntimeError('frozen bank receipt changed')
    def read(record):
        # Records can retain legacy six histories; the head sees EXACTLY last four.
        r = {**record, 'history_tokens': tuple(record['history_tokens'])[-4:]}
        return reader(r)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(len(chunks)*chunk_size, len(records), chunk_size):
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('bank prefix retained')
            part = records[start:start+chunk_size]
            # CPU workers read history only; CUDA remains on the main thread.
            raw_rows = pool.map(read, part)
            rows = []
            for record, (raw, times) in zip(part, raw_rows):
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError('complete bank chunks retained')
                values = materialize(body, raw, times, grid, pool_side=pool_side)
                # Future poses are SUPERVISION only, constructed after history tokens.
                values['target'] = torch.from_numpy(relative_se2(raw['history_poses'][-1],
                    [source.pose(t) for t in record['future_tokens']]))
                values['commands'] = torch.from_numpy(navigation_commands(source.nusc, record['future_tokens']))
                rows.append(values)
            path = directory/f'{len(chunks):04d}.pt'
            payload = dict(contract=fingerprint, keys=keys[start:start+len(rows)],
                values={k: torch.stack([r[k] for r in rows]) for k in FIELDS})
            if path.exists():
                # An interrupted publication before its manifest may leave ONE orphan.
                old = torch.load(path, map_location='cpu', weights_only=False)
                if (old['contract'] != fingerprint or old['keys'] != payload['keys'] or
                        any(not torch.equal(old['values'][k], payload['values'][k]) for k in FIELDS)):
                    raise RuntimeError('uncommitted bank chunk differs; do not overwrite')
            else: save_resume_checkpoint(path, payload)
            chunks.append(dict(index=len(chunks), windows=len(rows), sha256=legacy.sha256(path)))
            manifest['complete'] = start+len(rows) == len(records)
            legacy.write_json(manifest_path, manifest)
            print(f'FROZEN_EGO_BANK {start+len(rows)}/{len(records)} history_tokens_only', flush=True)
    values = {k: [] for k in FIELDS}; used = 0; offset = 0
    for entry in chunks:
        payload = torch.load(directory/f'{entry["index"]:04d}.pt', map_location='cpu', weights_only=False)
        if payload['contract'] != fingerprint or payload['keys'] != keys[offset:offset+entry['windows']]:
            raise RuntimeError('bank identity/order differs')
        for k in FIELDS:
            v = payload['values'][k]; used += v.numel()*v.element_size(); values[k].append(v)
        offset += entry['windows']
    if offset != len(records) or not manifest['complete']: raise RuntimeError('incomplete bank')
    if used > 3072*2**20: raise RuntimeError('compact token bank exceeds 3GiB; no dense F caching')
    bank = {k: torch.cat(v) for k, v in values.items()}
    for k, v in bank.items():
        if len(v) != len(records) or k != 'valid' and not bool(torch.isfinite(v).all()):
            raise RuntimeError('invalid bank tensors')
    manifest['tensor_mib'] = used/2**20; legacy.write_json(manifest_path, manifest)
    return bank, manifest
