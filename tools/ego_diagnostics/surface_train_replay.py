#!/usr/bin/env python3
"""Read-only TRAIN-bank/live replay; deliberately outside legacy hashed folders."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import os
import signal
import threading
import time

import numpy as np
import torch

from real_motion.ego_trajectory_head import EgoHeadConfig, HistoryEgoTrajectoryHead
from real_motion.ego_navigation import relative_se2, navigation_commands, validate_window_identity
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.stc_camera_protocol import load_catalog, semantics
from tools.real_motion.ego_trajectory_common import (
    FEATURE_FIELDS, digest_file, fingerprint, stack_features, extract_history_features,
)
from tools.real_motion.surface_ego_ablation_common import (
    PROTOCOL as PAIR_PROTOCOL, TRAIN_NAMES, historical_prior, bank_reports,
    load_source_bank, verify_sources, trajectory_report,
)
from tools.real_motion.joint_surface_checkpoint_selection import load_evaluation_model
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.waymo_zero_shot_common import write_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'surface_frozen_ego_train_bank_live_replay_v1'
TOLERANCES = dict(feature_atol=1e-6, feature_rtol=1e-5, xy_m=1e-4, yaw_rad=1e-5)
METADATA = ('scene', 'sample', 'sample_data', 'ego_pose', 'sensor', 'calibrated_sensor')
IMPLEMENTATION = ('tools/ego_diagnostics/surface_train_replay.py',
                  'tools/real_motion/run_p0_f9_surface_ego_train_replay.sh')


def select_indices(rows, count=64):
    """Deterministic three destination-command strata, scene-balanced within each.

    Selection never reads a model/error. This is a diagnostic subset, not a
    representative replacement for the original full TRAIN1024 population.
    """
    if not 1 <= count <= len(rows):
        raise ValueError('positive bounded replay population required')
    for row in rows:
        cmd = np.asarray(row['commands'])
        if cmd.shape != (6,) or not np.isin(cmd, (0, 1, 2)).all():
            raise ValueError('six valid destination commands required')
    groups = []
    for command in range(3):
        scenes = {}
        for i, row in enumerate(rows):
            if int(row['commands'][-1]) == command:
                scenes.setdefault(str(row['key'][0]), []).append(i)
        queue = []
        for level in range(max((len(v) for v in scenes.values()), default=0)):
            queue.extend(v[level] for v in scenes.values() if level < len(v))
        groups.append(queue)
    chosen = []
    while len(chosen) < count:
        for group in groups:
            if group and len(chosen) < count:
                chosen.append(group.pop(0))
    return chosen


def array_difference(a, b, *, atol=0., rtol=0.):
    def array(v):
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    a, b = array(a), array(b)
    base = dict(shape_equal=a.shape == b.shape, dtype_equal=a.dtype == b.dtype,
                dtype_a=str(a.dtype), dtype_b=str(b.dtype))
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(base, exact=False, numerical=False, max_abs=None, mean_abs=None,
                    different_elements=None, outside_tolerance=None)
    d = np.abs(a.astype(np.float64)-b.astype(np.float64))
    exact = base['dtype_equal'] and a.tobytes() == b.tobytes()
    # Integer masks/commands always require exact values and dtype.
    close = (d <= atol + rtol*np.abs(a)) if a.dtype.kind == b.dtype.kind == 'f' else (a == b)
    return dict(base, exact=exact, numerical=bool(base['dtype_equal'] and close.all()),
                max_abs=float(d.max()) if d.size else 0., mean_abs=float(d.mean()) if d.size else 0.,
                different_elements=int((a != b).sum()), outside_tolerance=int((~close).sum()))


def feature_difference(a, b):
    if set(a) != set(FEATURE_FIELDS) or set(b) != set(FEATURE_FIELDS):
        raise ValueError('exact seven history-only fields required')
    return {k: array_difference(a[k], b[k], atol=TOLERANCES['feature_atol'],
                                rtol=TOLERANCES['feature_rtol']) for k in FEATURE_FIELDS}


def trajectory_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.shape[-2:] != (6, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(pass_gate=False, max_xy_m=None, max_yaw_deg=None)
    xy = np.linalg.norm(a[..., :2]-b[..., :2], axis=-1)
    yaw = np.abs(np.arctan2(np.sin(a[..., 2]-b[..., 2]), np.cos(a[..., 2]-b[..., 2])))
    return dict(pass_gate=bool((xy <= TOLERANCES['xy_m']).all() and
                              (yaw <= TOLERANCES['yaw_rad']).all()),
                max_xy_m=float(xy.max()), max_yaw_deg=float(yaw.max()*180/np.pi))


@torch.no_grad()
def predictions(models, features, commands, *, batch_size=64, training_mode=False):
    device = next(models['old320'].parameters()).device
    bank = stack_features(features, device); commands = np.asarray(commands)
    result = {k: [] for k in ('prior', *models)}
    flags = {k: m.training for k, m in models.items()}
    if training_mode and any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or
            isinstance(module, torch.nn.Dropout) and module.p > 0
            for model in models.values() for module in model.modules()):
        raise RuntimeError('mode replay requires original dropout-free, BatchNorm-free head')
    try:
        for model in models.values():
            model.train(training_mode)
        for start in range(0, len(features), batch_size):
            sl = slice(start, start+batch_size); chunk = {k: v[sl] for k, v in bank.items()}
            result['prior'].append(historical_prior(chunk).cpu().numpy())
            for name, model in models.items():
                result[name].append(model(chunk, commands[sl])['se2'].cpu().numpy())
    finally:
        for name, model in models.items():
            model.train(flags[name])
    return {k: np.concatenate(v) for k, v in result.items()}


def report_difference(a, b):
    if set(a) != set(b):
        raise RuntimeError('original TRAIN report head names differ')
    fields = ('xy_mean_m', 'xy_median_m', 'xy_p90_m', 'ADE_m', 'FDE_3s_m',
              'yaw_mean_deg', 'yaw_median_deg', 'yaw_p90_deg')
    return {name: dict(windows_equal=a[name]['windows'] == b[name]['windows'],
        commands_equal=a[name]['command_counts_by_horizon'] == b[name]['command_counts_by_horizon'],
        metrics={k: array_difference(np.asarray(a[name][k]), np.asarray(b[name][k]),
                    atol=1e-4 if not k.startswith('yaw') else TOLERANCES['yaw_rad']*180/np.pi)
                 for k in fields}) for name in a}


def state_digest(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str((str(a.dtype), tuple(a.shape))).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class TrainMaterializer:
    def __init__(self, source, catalog, records, provider, shape):
        self.source, self.catalog, self.records, self.provider, self.shape = source, catalog, records, provider, shape

    def __call__(self, index, config):
        rec = self.records[index]; validate_window_identity(self.source.nusc, rec)
        loaded = [self.source.load_occ3d(rec['scene_name'], t) for t in rec['history_tokens']]
        poses = [self.source.pose(t) for t in rec['history_tokens']]
        times = [self.source.nusc.get('sample', t)['timestamp']/1e6 for t in rec['history_tokens']]
        train_raw = dict(history_occ=np.stack([a for a, _ in loaded]),
            history_observed=np.stack([b for _, b in loaded]), history_poses=poses)
        train_features = extract_history_features(self.provider, rec, train_raw, config, timestamps_s=times)
        # Independently reproduce the actual dev OCC reader's uint8/mask/pose/time
        # convention on THESE SAME TRAIN tokens. No validation planner is needed:
        # the feature extractor physically whitelists away ALL future geometry.
        loaded_eval = []
        for token in rec['history_tokens']:
            with np.load(self.source._label_path(rec['scene_name'], token), allow_pickle=False) as z:
                loaded_eval.append((semantics(z['semantics'], self.shape), np.array(z['mask_lidar'], bool)))
        eval_raw = dict(history_occ=np.stack([a for a, _ in loaded_eval]),
            history_observed=np.stack([b for _, b in loaded_eval]),
            history_poses=[self.catalog[t].pose for t in rec['history_tokens']])
        eval_times = [self.catalog[t].timestamp/1e6 for t in rec['history_tokens']]
        eval_features = extract_history_features(self.provider, rec, eval_raw, config, timestamps_s=eval_times)
        inputs = {k: array_difference(train_raw[k], eval_raw[k]) for k in train_raw}
        inputs['timestamps_s'] = array_difference(times, eval_times)
        # Rebuild supervision ONLY AFTER both historical extractions finish.
        target_train = relative_se2(poses[-1], [self.source.pose(t) for t in rec['future_tokens']])
        target_eval = relative_se2(eval_raw['history_poses'][-1], [self.catalog[t].pose for t in rec['future_tokens']])
        return dict(train_features=train_features, eval_features=eval_features, inputs=inputs,
            target_train=target_train, target_eval=target_eval,
            commands=navigation_commands(self.source.nusc, rec['future_tokens']))


def derive_records(nusc, rows):
    records = []
    for row in rows:
        scene, t0 = row['key']; hist = [t0]; future = []; token = t0
        for _ in range(3):
            hist.insert(0, nusc.get('sample', hist[0])['prev'])
        for _ in range(6):
            token = nusc.get('sample', token)['next']; future.append(token)
        rec = dict(scene_name=scene, t0_token=t0, history_tokens=tuple(hist), future_tokens=tuple(future))
        validate_window_identity(nusc, rec); records.append(rec)
    return records


def historical_provenance(source, records, indices, original):
    base = Path(source.dataroot)/'v1.0-trainval'
    receipt = {}
    for name in METADATA:
        p = base/(name+'.json'); sha = digest_file(p)
        if sha != original['metadata_hashes'].get(p.name):
            raise RuntimeError('source-bank metadata changed: '+str(p))
        receipt[str(p)] = sha
    paths = sorted({source._label_path(r['scene_name'], t) for r in records for t in r['history_tokens']})
    stats = [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
    if fingerprint(stats) != original['historical_files_fingerprint']:
        raise RuntimeError('original TRAIN historical-file population/stat changed')
    for i in indices:
        r = records[i]
        for t in r['history_tokens']:
            p = source._label_path(r['scene_name'], t)
            if str(p) not in receipt:
                receipt[str(p)] = digest_file(p)
    return receipt


def verify_receipt(receipt):
    for path, sha in receipt.items():
        if digest_file(path) != sha:
            raise RuntimeError('read-only replay source changed: '+path)


@torch.no_grad()
def audit(rows, indices, models, materialize, recorded_train, contract, *, saved=None, save=None,
          stop_event=None, progress=None):
    if not indices or len(set(indices)) != len(indices) or any(not 0 <= i < len(rows) for i in indices):
        raise ValueError('unique original-bank indices required')
    full_bank = stack_features([r['features'] for r in rows], next(models['old320'].parameters()).device)
    full_targets = torch.tensor(np.stack([r['target'] for r in rows]), device=next(models['old320'].parameters()).device)
    full_cmds = torch.tensor(np.stack([r['commands'] for r in rows]), device=full_targets.device)
    full_train = bank_reports(models, full_bank, full_targets, full_cmds)
    reproduction = report_difference(full_train, recorded_train)
    chosen = [rows[i] for i in indices]; features = [r['features'] for r in chosen]
    commands = np.stack([r['commands'] for r in chosen]); targets = np.stack([r['target'] for r in chosen])
    batch = predictions(models, features, commands)
    single = predictions(models, features, commands, batch_size=1)
    train_mode = predictions(models, features, commands, training_mode=True)
    batching = {k: trajectory_difference(batch[k], single[k]) for k in batch}
    mode = {k: trajectory_difference(batch[k], train_mode[k]) for k in batch}
    cursor = 0; audits = []; elapsed = 0.
    if saved is not None:
        checked = dict(saved); sha = checked.pop('fingerprint', None)
        if sha != fingerprint(checked) or saved['contract_fingerprint'] != fingerprint(contract):
            raise RuntimeError('TRAIN replay state/contract fingerprint changed')
        cursor = saved['completed_windows']; audits = saved['window_audits']; elapsed = saved['seconds']
        if type(cursor) != int or not 0 <= cursor <= len(indices) or len(audits) != cursor or any(
                row['index'] != indices[i] or row['key'] != rows[indices[i]]['key'] for i, row in enumerate(audits)):
            raise RuntimeError('TRAIN replay complete-window prefix invalid')
    def persist(status):
        state = dict(status=status, completed_windows=cursor, contract_fingerprint=fingerprint(contract),
                     window_audits=list(audits), seconds=elapsed)
        state['fingerprint'] = fingerprint(state)
        if save:
            save(state)
        return state
    persist('running')
    for offset in range(cursor, len(indices)):
        if stop_event is not None and stop_event.is_set():
            break
        tick = time.perf_counter(); index = indices[offset]; old = rows[index]
        live = materialize(index, models['old320'].config)
        differences = dict(cache_vs_train=feature_difference(old['features'], live['train_features']),
            cache_vs_eval=feature_difference(old['features'], live['eval_features']),
            train_vs_eval=feature_difference(live['train_features'], live['eval_features']))
        labels = dict(cache_vs_train=array_difference(old['target'], live['target_train']),
                      cache_vs_eval=array_difference(old['target'], live['target_eval']))
        cmd = array_difference(old['commands'], live['commands'])
        outputs = {}; effects = {}
        for route, fields in (('live_train', live['train_features']), ('live_eval', live['eval_features'])):
            structural = all(fields[k].shape == old['features'][k].shape and torch.isfinite(fields[k]).all() for k in FEATURE_FIELDS)
            valid_cmd = np.asarray(live['commands']).shape == (6,) and np.isin(live['commands'], [0, 1, 2]).all()
            if structural and valid_cmd:
                p = predictions(models, [fields], np.asarray(live['commands'])[None], batch_size=1)
                outputs[route] = {k: v[0].tolist() for k, v in p.items()}
                effects[route] = {k: trajectory_difference(single[k][offset], v[0]) for k, v in p.items()}
            else:
                outputs[route] = None; effects[route] = {'invalid_live_feature_shape_or_command': True}
        # A failure before here never publishes half a TRAIN window.
        audits.append(dict(index=index, key=old['key'], features=differences, labels=labels, commands=cmd,
            inputs=live['inputs'], predictions=outputs, trajectory_effect=effects))
        cursor += 1; elapsed += time.perf_counter()-tick
        if cursor % 8 == 0:
            persist('running')
        if progress:
            progress(dict(window=cursor, total=len(indices), seconds=time.perf_counter()-tick))
    state = persist('complete' if cursor == len(indices) else 'stopped')
    subset = {'cached': {k: trajectory_report(p[:cursor], targets[:cursor], commands[:cursor])
                         for k, p in single.items()} if cursor else {}}
    for route in ('live_train', 'live_eval'):
        valid = [i for i, r in enumerate(audits) if r['predictions'][route] is not None]
        subset[route] = {k: trajectory_report(np.asarray([audits[i]['predictions'][route][k] for i in valid]),
                            targets[valid], commands[valid]) for k in single} if valid else {}
    gates = dict(full_TRAIN_report_reproduced=all(r['windows_equal'] and r['commands_equal'] and
        all(v['numerical'] for v in r['metrics'].values()) for r in reproduction.values()),
        head_batch64_vs_single=all(v['pass_gate'] for v in batching.values()),
        head_train_mode_vs_eval=all(v['pass_gate'] for v in mode.values()),
        cache_vs_live_features=all(v['numerical'] for r in audits for pair in r['features'].values() for v in pair.values()),
        labels_and_commands=all(r['commands']['exact'] and all(v['exact'] for v in r['labels'].values()) for r in audits),
        train_vs_eval_history_reader=all(v['numerical'] for r in audits for v in r['inputs'].values()),
        live_head_outputs=all(r['predictions'][route] is not None and
            all(v['pass_gate'] for v in r['trajectory_effect'][route].values())
            for r in audits for route in ('live_train', 'live_eval')),
        complete=cursor == len(indices))
    passed = all(gates.values())
    route = ('interfaces_consistent_on_checked_TRAIN_subset_generalization_gap_remains' if passed else
             'stopped_not_a_completed_audit' if not gates['complete'] else 'interface_discrepancy_do_not_expand_training')
    return dict(protocol=PROTOCOL, state=state, gates=gates, pass_gate=passed, route=route,
                full_TRAIN=full_train, full_TRAIN_reproduction=reproduction,
                batch64_vs_single=batching, train_mode_vs_eval=mode, subset=subset, contract=contract)


def summary_text(result, dev=None):
    lines = ['===== FROZEN EGO TRAIN CACHE / LIVE INTERFACE AUDIT =====',
        f'status={result["state"]["status"]}; windows={result["state"]["completed_windows"]}',
        'Original TRAIN subset, command/scene-balanced; NOT dev and NOT representative full TRAIN metrics.',
        'No training/optimizer steps, dense forecasting, future occupancy reads, bank rewrites or corrections.',
        'gates='+json.dumps(result['gates']), 'route='+result['route'],
        'Declared numerical budgets='+json.dumps(TOLERANCES),
        'Feature bytes and numerical agreement both reported; a numerical pass is NOT byte identity.']
    for pair in ('cache_vs_train', 'cache_vs_eval', 'train_vs_eval'):
        lines += ['', '===== '+pair+' / FEATURE DIFFERENCES =====']
        for key in FEATURE_FIELDS:
            vals = [r['features'][pair][key] for r in result['state']['window_audits']]
            if not vals:
                continue
            max_abs = max((v['max_abs'] for v in vals if v['max_abs'] is not None), default=None)
            lines.append(f'{key:18} exact={sum(v["exact"] for v in vals)}/{len(vals)} '
                f'numerical={sum(v["numerical"] for v in vals)}/{len(vals)} max_abs={max_abs}')
    for key in ('batch64_vs_single', 'train_mode_vs_eval'):
        lines += ['', key+'='+json.dumps(result[key])]
    lines += ['', '===== SAME TRAIN SUBSET / ORIGINAL LABELS =====', 'route        head             ADE_m   FDE_3s_m   yaw_3s_deg']
    for route, heads in result['subset'].items():
        for name, r in heads.items():
            lines.append(f'{route:12} {name:14} {r["ADE_m"]:.6f} {r["FDE_3s_m"]:.6f} {r["yaw_mean_deg"][-1]:.6f}')
    lines += ['', '===== FULL TRAIN1024 REPRODUCTION / EXISTING DEV64 =====',
              'head             TRAIN_3s_XY_m  DEV_OCC_3s_XY_m  TRAIN_3s_yaw_deg  DEV_OCC_3s_yaw_deg']
    for name, r in result['full_TRAIN'].items():
        d = dev['trajectory'].get('occ_'+name) if dev else None
        lines.append(f'{name:14} {r["FDE_3s_m"]:.6f} '+(f'{d["FDE_3s_m"]:.6f}' if d else 'NA')+
            f' {r["yaw_mean_deg"][-1]:.6f} '+(f'{d["yaw_mean_deg"][-1]:.6f}' if d else 'NA'))
    failed = [dict(index=r['index'], key=r['key']) for r in result['state']['window_audits'] if
        not r['commands']['exact'] or any(not v['numerical'] for p in r['features'].values() for v in p.values()) or
        any(not v['exact'] for v in r['labels'].values()) or
        any(not v['numerical'] for v in r['inputs'].values()) or
        any(r['predictions'][route] is None or
            any(not v['pass_gate'] for v in r['trajectory_effect'][route].values())
            for route in ('live_train', 'live_eval'))]
    lines += ['first_mismatch_windows='+json.dumps(failed[:5]), f'live_replay_seconds={result["state"]["seconds"]:.2f}',
              'If all gates pass, interfaces are consistent on checked TRAIN windows, NOT proof all DEV shifts are excluded.',
              'No automatic retry, larger training, head selection, deployment or changes to main metrics.']
    return '\n'.join(lines)+'\n'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('pair-dir', 'dataroot', 'out-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--dev-report'); p.add_argument('--config', default=str(ROOT/'configs/real_motion_occfm.yaml'))
    p.add_argument('--windows', type=int, default=64); p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--device', default='cuda'); p.add_argument('--resume', action='store_true')
    return p


def main(stop_event=None, argv=None):
    a = parser().parse_args(argv); pairdir = Path(a.pair_dir).resolve(); out = Path(a.out_dir).resolve()
    if not 1 <= a.windows <= 128 or not 1 <= a.cpu_workers <= 8:
        raise ValueError('bounded replay windows/workers required')
    pairpath = pairdir/'pair_last.pt'; pair_sha = digest_file(pairpath)
    pair_contract_path = pairdir/'contract.json'; pair_contract_sha = digest_file(pair_contract_path)
    pair = torch.load(pairpath, map_location='cpu', weights_only=False); training = pair['contract']
    if pair.get('protocol') != PAIR_PROTOCOL or pair['updates'] != training['schedule']['max_updates'] or pair['config'] != training['head_config']:
        raise RuntimeError('completed original A/B checkpoint required')
    if not pair.get('monitors') or pair['monitors'][-1]['update'] != pair['updates']:
        raise RuntimeError('original final-update TRAIN report required')
    if training != json.loads(pair_contract_path.read_text(encoding='utf-8')):
        raise RuntimeError('paired checkpoint/contract mismatch')
    source_dir = training['source']['source_dir']
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in
           (pairdir, Path(source_dir).resolve(), Path(a.dataroot).resolve())):
        raise ValueError('new independent replay output required')
    if out.exists() and not a.resume:
        raise FileExistsError('NEW output required, or --resume SAME replay directory')
    if a.resume and not (out/'state.json').is_file():
        raise ValueError('missing replay recovery state')
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    rows, old, audit_source = load_source_bank(source_dir, ROOT)
    if audit_source != training['source']:
        raise RuntimeError('original feature/geometry implementation or sources changed')
    original = audit_source['original_training']
    execution = dict(device=str(torch.device(a.device)), torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                     WM_precision='bf16' if a.device.startswith('cuda') else 'fp32')
    if execution != original['feature_execution']:
        raise RuntimeError('use original Torch/device/precision for bank replay')
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith('SWFM_')}
    if env != original['runtime_environment'] or digest_file(a.config) != original['runtime_config_sha256']:
        raise RuntimeError('original SWFM environment/runtime geometry required; use the wrapper')
    dev_path = Path(a.dev_report) if a.dev_report else Path(str(pairdir)+'_eval_dev64')/'evaluation.json'
    dev = None; dev_sha = None
    if dev_path.exists():
        dev_sha = digest_file(dev_path); dev = json.loads(dev_path.read_text(encoding='utf-8'))
        if (dev['state']['status'] != 'complete' or dev['contract']['training'] != training or
                dev['contract']['pair_checkpoint_sha256'] != pair_sha):
            raise RuntimeError('existing dev report belongs to a different/uncompleted paired experiment')
    elif a.dev_report:
        raise FileNotFoundError(dev_path)
    models = {k: HistoryEgoTrajectoryHead(EgoHeadConfig(**pair['config'])).to(a.device).eval().requires_grad_(False)
              for k in ('old320', *TRAIN_NAMES)}
    models['old320'].load_state_dict(old['state_dict'], strict=True)
    for name in TRAIN_NAMES:
        models[name].load_state_dict(pair['models'][name], strict=True)
    _, joint = load_evaluation_model(audit_source['frozen_checkpoint'], device=a.device)
    before = {k: state_digest(m) for k, m in dict(WM_CCR=joint, **models).items()}
    indices = select_indices(rows, a.windows); source = NuScenesWindowSource(a.dataroot)
    records = derive_records(source.nusc, rows)
    receipt = historical_provenance(source, records, indices, original)
    catalog, _ = load_catalog(a.dataroot, {records[i]['scene_name'] for i in indices})
    contract = dict(protocol=PROTOCOL, pair_sha256=pair_sha, pair_contract_sha256=pair_contract_sha, training=training,
        indices=indices, keys=[rows[i]['key'] for i in indices], input_receipt=receipt,
        selection='deterministic destination3s_cmd_balanced_scene_round_robin; no model/error selection',
        execution=execution, runtime_environment=env, tolerances=TOLERANCES,
        diagnostic_implementation={p: digest_file(ROOT/p) for p in IMPLEMENTATION},
        dev_report_sha256=dev_sha, windows=a.windows, cpu_workers=a.cpu_workers,
        feature_geometry_ram_mib=0, future_occupancy_read=False)
    previous = None
    if a.resume:
        if json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract:
            raise RuntimeError('TRAIN replay contract changed')
        previous = json.loads((out/'state.json').read_text(encoding='utf-8'))
    else:
        out.mkdir(parents=True); write_json(out/'contract.json', contract)
    predictor = Predictor(joint, make_prepare_config(load_runtime_config(a.config)), a.device,
                          workers=a.cpu_workers, graphs=False, geometry_mib=0)
    try:
        with evaluation_lock(out):
            result = audit(rows, indices, models, TrainMaterializer(source, catalog, records, predictor.provider,
                predictor.provider.pcfg.grid.shape_hwd), pair['monitors'][-1]['reports'], contract,
                saved=previous, save=lambda v: write_json(out/'state.json', v), stop_event=stop_event,
                progress=lambda v: print('EGO_TRAIN_REPLAY '+json.dumps(v), flush=True) if v['window'] % 8 == 0 else None)
            verify_sources(audit_source); verify_receipt(receipt)
            if (digest_file(pairpath) != pair_sha or digest_file(pair_contract_path) != pair_contract_sha or
                    dev_sha and digest_file(dev_path) != dev_sha):
                raise RuntimeError('original head/dev artifacts changed during read-only replay')
            after = {k: state_digest(m) for k, m in dict(WM_CCR=joint, **models).items()}
            if before != after:
                raise RuntimeError('frozen model state changed during replay')
            result['frozen_weights_unchanged'] = True
            write_json(out/'audit.json', result)
            summary = summary_text(result, dev); (out/'summary.txt').write_text(summary, encoding='utf-8')
            print(summary, flush=True); print('结果：'+str(out/'summary.txt'), flush=True)
    finally:
        predictor.close()
    return 0 if result['state']['status'] == 'complete' else 130


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
