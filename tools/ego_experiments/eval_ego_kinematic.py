#!/usr/bin/env python3
"""One concentrated real-data check of prior/old3/two control heads + planner."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import signal
import threading
import time
import numpy as np
import torch
from real_motion.ego_trajectory_head import EgoHeadConfig, HistoryEgoTrajectoryHead
from real_motion.ego_navigation import navigation_commands, relative_se2, poses_from_se2, replace_future_geometry
from real_motion.stc_camera_protocol import STCFourSettingSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.waymo_i2world import WaymoMetrics
from tools.ego_experiments.ego_kinematic import KinematicEgoHead, KinematicPrior, PROTOCOL
from tools.ego_experiments.train_ego_kinematic import ARMS, ROOT, code_receipt, check_execution
from tools.ego_experiments.train_surface_ego_three import validate_selected
from tools.ego_experiments.eval_surface_ego_three_population import select_dev512
from tools.real_motion.ego_trajectory_common import digest_file, fingerprint, extract_history_features, stack_features
from tools.real_motion.surface_ego_ablation_common import trajectory_report, tensor_state_fingerprint
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.joint_surface_checkpoint_selection import load_evaluation_model
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.waymo_zero_shot_common import write_json


@torch.no_grad()
def evaluate(source, windows, predictor, models, commands_for, contract, *, saved=None, save=None,
             stop_event=None, progress=None, feature_extractor=extract_history_features):
    routes = (*models, 'external', 'gt'); arms = tuple(f'{m}_{r}' for m in contract['modalities'] for r in routes)
    counts = {k:WaymoMetrics() for k in arms}; cursor = 0; errors = {}; elapsed = 0.
    if saved is not None:
        check = dict(saved); sha = check.pop('fingerprint',None)
        if sha != fingerprint(check) or saved['contract_fingerprint'] != fingerprint(contract):
            raise RuntimeError('control eval recovery contract/fingerprint changed')
        cursor = saved['completed_windows']
        if type(cursor) != int or not 0 <= cursor <= len(windows) or set(saved['counts']) != set(arms):
            raise RuntimeError('invalid control eval cursor/arms')
        for k,a in saved['counts'].items():
            a = np.asarray(a)
            if a.dtype.kind not in 'ui' or a.shape != (3,18,18) or (a<0).any() or not (a.sum((1,2)) == cursor*np.prod(source.shape)).all():
                raise RuntimeError('invalid complete-window metric prefix')
            counts[k] = WaymoMetrics(a,cursor)
        errors = saved['trajectory']; elapsed = saved['seconds']
        if any(len(v) != cursor for v in errors.values()) or (cursor and len(errors) != len(contract['modalities'])*(len(models)+1)):
            raise RuntimeError('invalid trajectory prefix')
    device = next(models['old_epoch3'].parameters()).device; checked = set()
    for h in models.values(): h.eval()
    def persist(status):
        s = dict(status=status,completed_windows=cursor,contract_fingerprint=fingerprint(contract),
            counts={k:v.counts.tolist() for k,v in counts.items()},trajectory=errors,seconds=elapsed)
        s['fingerprint'] = fingerprint(s)
        if save: save(s)
        return s
    persist('running')
    for w in windows[cursor:]:
        if stop_event is not None and stop_event.is_set(): break
        tick = time.perf_counter(); nav = commands_for(w); dense = {}; local_errors = {}
        for modality in contract['modalities']:
            rec, raw = source.prediction_inputs(w,modality+'_pred')
            times = [source.catalog[t].timestamp/1e6 for t in w.history]
            features = feature_extractor(predictor.provider,rec,raw,models['old_epoch3'].config,timestamps_s=times)
            bank = stack_features([features],device)
            trajectories = {k:h(bank,np.asarray(nav)[None])['se2'][0].cpu().numpy() for k,h in models.items()}
            raws = {k:replace_future_geometry(raw,poses_from_se2(raw['history_poses'][-1],p)) for k,p in trajectories.items()}
            raws['external'] = raw
            # No future GT pose has entered a candidate's inference or choices.
            _, gt = source.prediction_inputs(w,modality+'_gt'); raws['gt'] = gt
            for r in routes:
                key = f'{modality}_{r}'
                _, prediction, _, _ = predictor.full(rec,raws[r],verify=key not in checked)
                dense[key] = prediction; checked.add(key)
            target = relative_se2(raw['history_poses'][-1],gt['future_poses'])
            trajectories['external'] = relative_se2(raw['history_poses'][-1],raw['future_poses'])
            for k,p in trajectories.items(): local_errors[f'{modality}_{k}'] = dict(pred=p.tolist(),target=target.tolist(),commands=np.asarray(nav).tolist())
        target = source.metric_targets(w)  # AFTER ALL predictions, including STC.
        deltas = {k:WaymoMetrics() for k in arms}
        for k in arms: deltas[k].add(dense[k],target)
        for k in arms: counts[k].counts += deltas[k].counts; counts[k].windows += 1
        for k,r in local_errors.items(): errors.setdefault(k,[]).append(r)
        cursor += 1; elapsed += time.perf_counter()-tick
        if cursor%8 == 0: persist('running')
        if progress: progress(dict(window=cursor,total=len(windows),seconds=time.perf_counter()-tick))
    state = persist('complete' if cursor == len(windows) else 'stopped')
    trajectory = {k:trajectory_report(np.asarray([r['pred'] for r in rows]),np.asarray([r['target'] for r in rows]),
        np.asarray([r['commands'] for r in rows])) for k,rows in errors.items() if rows}
    return dict(state=state,reports={k:v.report() for k,v in counts.items()},trajectory=trajectory,contract=contract)


def summary_text(result, training_dir):
    lines = [(Path(training_dir)/'summary.txt').read_text(encoding='utf-8'),
        '===== FIXED REAL DEV / KINEMATIC EGO SCREEN =====',
        f'status={result["state"]["status"]}; windows={result["state"]["completed_windows"]}',
        'GT-derived navigation-conditioned. No alignment/masks/threshold/epoch tuning.',
        'setting                 mIoU_avg IoU_avg  mIoU@1/2/3s']
    for k,r in result['reports'].items():
        v = r['average']; hs = '/'.join(f'{x["standard_mIoU"]:.3f}' for x in r['horizons'].values())
        lines.append(f'{k:24} {v["standard_mIoU"]:.6f} {v["IoU"]:.6f} {hs}')
    lines += ['','trajectory              ADE_m FDE3s_m yaw3s_deg']
    for k,r in result['trajectory'].items(): lines.append(f'{k:24} {r["ADE_m"]:.6f} {r["FDE_3s_m"]:.6f} {r["yaw_mean_deg"][-1]:.6f}')
    lines += [f'seconds={result["state"]["seconds"]:.2f}; quality eval, NOT FPS.',
        'Both final heads/prior/old3 reported. No automatic retry, extension, method selection or promotion.']
    return '\n'.join(lines)+'\n'


def main(stop_event=None, argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training-dir','dataroot','stc-root','plan-cache','population-manifest','out-dir'): p.add_argument('--'+name,required=True)
    p.add_argument('--config',default=str(ROOT/'configs/real_motion_occfm.yaml'))
    p.add_argument('--population',choices=('dev64','dev512'),default='dev64')
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    a = p.parse_args(argv); training_dir = Path(a.training_dir).resolve(); out = Path(a.out_dir).resolve()
    if not 1 <= a.cpu_workers <= 8: raise ValueError('bounded workers required')
    if any(out == d or out.is_relative_to(d) or d.is_relative_to(out) for d in map(lambda x:Path(x).resolve(),
        (a.dataroot,a.stc_root,a.plan_cache,training_dir))): raise ValueError('independent new evaluation output required')
    if out.exists() and not a.resume: raise FileExistsError('new output or resume SAME eval')
    torch.set_num_threads(1); path = training_dir/'heads_epoch3.pt'; head_sha = digest_file(path)
    saved = torch.load(path,map_location='cpu',weights_only=False); c = saved['contract']; original = c['source_training']
    if (saved.get('protocol') != PROTOCOL+'_final3' or saved['epochs'] != 3 or not saved['evaluation_only'] or
            saved['config'] != original['head_config'] or set(saved['models']) != set(ARMS) or
            c['implementation'] != code_receipt() or c != json.loads((training_dir/'training.json').read_text(encoding='utf-8'))):
        raise RuntimeError('complete same-code final control heads required')
    check_execution(original,a.device)
    if digest_file(a.config) != original['runtime_config_sha256']: raise RuntimeError('geometry config changed')
    old_path = Path(c['source_dir'])/'head_epoch3.pt'
    if digest_file(old_path) != c['source_head_sha256']: raise RuntimeError('source legacy head changed')
    old = torch.load(old_path,map_location='cpu',weights_only=False);validate_selected(old)
    cfg = EgoHeadConfig(**saved['config'])
    models = {k:KinematicEgoHead(cfg,scene=k=='control_scene').to(a.device) for k in ARMS}
    for k,h in models.items():
        if tensor_state_fingerprint(saved['models'][k]) != saved['state_fingerprints'][k]: raise RuntimeError('control weights changed')
        h.load_state_dict(saved['models'][k]);h.eval().requires_grad_(False)
    legacy = HistoryEgoTrajectoryHead(cfg).to(a.device).eval().requires_grad_(False);legacy.load_state_dict(old['state_dict'])
    models = dict(old_epoch3=legacy,kinematic_prior=KinematicPrior(cfg).to(a.device),**models)
    _, joint = load_evaluation_model(original['frozen_checkpoint'],device=a.device);wm_before = tensor_state_fingerprint(joint.state_dict())
    source = STCFourSettingSource.from_files(a.dataroot,a.stc_root,a.plan_cache,cache_mib=512)
    manifest, _, _ = load_manifest(a.population_manifest)
    if a.population == 'dev512': windows,population = select_dev512(source,manifest)
    else: windows,population = source.select('dev64',manifest['parent_keys'])
    if {w.scene for w in windows}&{k[0] for k in original['keys']}: raise RuntimeError('TRAIN/dev scene overlap')
    inventory = source.preflight(windows); nusc = NuScenesWindowSource(a.dataroot).nusc
    contract = dict(protocol=PROTOCOL+'_dev',training=c,head_sha256=head_sha,windows=len(windows),population=population,
        inventory=inventory,modalities=['occ','stc'],routes=list(models)+['external','gt'],thresholds=[.5,None],
        GT_derived_navigation_condition=True,no_GT_pose_alignment=True,cpu_workers=a.cpu_workers,device=a.device)
    if a.resume:
        if json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract: raise RuntimeError('evaluation contract changed')
        previous = json.loads((out/'state.json').read_text(encoding='utf-8'))
    else: out.mkdir(parents=True);write_json(out/'contract.json',contract);previous = None
    predictor = Predictor(joint,make_prepare_config(load_runtime_config(a.config)),a.device,workers=a.cpu_workers,geometry_mib=512)
    try:
        with evaluation_lock(out):
            result = evaluate(source,windows,predictor,models,lambda w:navigation_commands(nusc,w.future),contract,
                saved=previous,save=lambda s:write_json(out/'state.json',s),stop_event=stop_event,
                progress=lambda r:print('EGO_CONTROL_DEV '+json.dumps(r),flush=True) if r['window']%8 == 0 else None)
            if (digest_file(path) != head_sha or digest_file(old_path) != c['source_head_sha256'] or
                    tensor_state_fingerprint(joint.state_dict()) != wm_before): raise RuntimeError('read-only weights changed')
            write_json(out/'evaluation.json',result)
            text = summary_text(result,training_dir);(out/'summary.txt').write_text(text,encoding='utf-8');print(text,flush=True)
    finally: predictor.close()
    return 0 if result['state']['status'] == 'complete' else 130


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event))
