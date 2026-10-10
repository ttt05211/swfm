#!/usr/bin/env python3
"""Evaluate existing epoch3 weights on the frozen dev512 planner-covered intersection."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import os
import signal
import threading
import torch

from real_motion.ego_trajectory_head import EgoHeadConfig, HistoryEgoTrajectoryHead
from real_motion.ego_navigation import navigation_commands
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.stc_camera_protocol import STCFourSettingSource
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from tools.real_motion.ego_trajectory_common import digest_file, implementation_fingerprint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.real_motion.eval_p0_f9_surface_ego_head import evaluate
from tools.real_motion.joint_surface_checkpoint_selection import load_evaluation_model
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.waymo_zero_shot_common import write_json
from tools.ego_experiments.train_surface_ego_three import ROOT, SELECTED_PROTOCOL, validate_selected



def select_dev512(source, manifest):
    """Freeze requested identities BEFORE planner filtering; never top up missing rows."""
    from real_motion.v21_source_induction import select_scene_balanced_round_robin
    parent = [tuple(k) for k in manifest['parent_keys']]
    selected = [tuple(k) for k in manifest['selected_keys']]
    if len(selected) == 512:
        requested = selected
        rule = 'existing_manifest_selected512'
    elif len(parent) >= 512:
        requested = parent if len(parent) == 512 else select_scene_balanced_round_robin(parent,512)
        rule = 'parent512' if len(parent) == 512 else 'frozen_parent_scene_balanced512_before_planner_filter'
    else:
        raise RuntimeError(f'manifest has only {len(parent)} parent windows; supply a dev512 parent manifest')
    windows, audit = source.select('dev512',requested)
    audit.update(parent_windows=len(parent),requested_windows=512,
        requested_keys=[list(k) for k in requested],selection_rule=rule,
        actual_planner_covered_windows=len(windows),no_top_up_or_padding=True,
        population_label='dev512' if len(windows)==512 else 'dev512_planner_covered_intersection')
    return windows, audit


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('training-dir', 'dataroot', 'stc-root', 'plan-cache', 'population-manifest', 'out-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--config', default=str(ROOT/'configs/real_motion_occfm.yaml'))
    p.add_argument('--cpu-workers', type=int, default=4); p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true'); return p


def main(stop_event=None, argv=None):
    a = parser().parse_args(argv); out = Path(a.out_dir).resolve(); training_dir = Path(a.training_dir).resolve()
    head_path = training_dir/'head_epoch3.pt'; head_sha = digest_file(head_path)
    if not 1 <= a.cpu_workers <= 8: raise ValueError('bounded workers required')
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in
        (training_dir, Path(a.dataroot).resolve(), Path(a.stc_root).resolve(), Path(a.plan_cache).resolve())):
        raise ValueError('new independent dev output required')
    if out.exists() and not a.resume: raise FileExistsError('NEW output required or --resume SAME dev directory')
    if a.resume and not (out/'state.json').is_file(): raise ValueError('dev recovery state missing')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    saved = torch.load(head_path, map_location='cpu', weights_only=False); training = validate_selected(saved)
    execution = dict(device=str(torch.device(a.device)), torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, WM_precision='bf16' if a.device.startswith('cuda') else 'fp32')
    if execution != training['execution']: raise RuntimeError('original feature execution required')
    if (digest_file(a.config) != training['runtime_config_sha256'] or
            {k: v for k, v in sorted(os.environ.items()) if k.startswith('SWFM_')} != training['runtime_environment']):
        raise RuntimeError('original runtime geometry/environment required')
    head = HistoryEgoTrajectoryHead(EgoHeadConfig(**saved['config'])).to(a.device).eval().requires_grad_(False)
    head.load_state_dict(saved['state_dict'], strict=True)
    _, joint = load_evaluation_model(training['frozen_checkpoint'], device=a.device)
    frozen_before = tensor_state_fingerprint(joint.state_dict())
    source = STCFourSettingSource.from_files(a.dataroot, a.stc_root, a.plan_cache, cache_mib=512)
    manifest, _, _ = load_manifest(a.population_manifest)
    windows, population = select_dev512(source,manifest)
    population_summary = {k:population[k] for k in ('parent_windows','requested_windows',
        'actual_planner_covered_windows','selection_rule','population_label','no_top_up_or_padding')}
    population_summary['missing_planner_windows'] = len(population['missing_parent_keys'])
    print('EGO_DEV_POPULATION '+json.dumps(population_summary),flush=True)
    inventory = source.preflight(windows)
    # Existing research dev, not an independent test; no checkpoint selection here.
    if {w.scene for w in windows} & {k[0] for k in training['keys']}:
        raise RuntimeError('dev population overlaps full TRAIN scenes')
    nusc = NuScenesWindowSource(a.dataroot).nusc
    contract = dict(protocol=SELECTED_PROTOCOL+'_dev512_populationfix', training=training, head_sha256=head_sha,
        evaluator_sha256=digest_file(__file__),
        selected_epoch=saved['selected_epoch'], update=saved['updates'], population=population, inventory=inventory,
        windows=len(windows), modalities=['occ', 'stc'], thresholds=[.5, None],
        GT_derived_navigation_condition=True, future_GT_pose_alignment=False,
        future_GT_labels='metrics only after all six routes', device=a.device,
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda, cpu_workers=a.cpu_workers)
    previous = None
    if a.resume:
        if json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract:
            raise RuntimeError('full-head dev contract changed')
        previous = json.loads((out/'state.json').read_text(encoding='utf-8'))
    else: out.mkdir(parents=True); write_json(out/'contract.json', contract)
    predictor = Predictor(joint, make_prepare_config(load_runtime_config(a.config)), a.device,
        workers=a.cpu_workers, geometry_mib=512)
    try:
        with evaluation_lock(out):
            result = evaluate(source, windows, predictor, head, lambda w: navigation_commands(nusc, w.future), contract,
                saved=previous, save=lambda s: write_json(out/'state.json', s), stop_event=stop_event,
                progress=lambda r: print('EGO_FULL_DEV '+json.dumps(r), flush=True) if r['window'] % 8 == 0 else None)
            validate_selected(saved)
            if (digest_file(head_path) != head_sha or tensor_state_fingerprint(joint.state_dict()) != frozen_before or
                    tensor_state_fingerprint(head.state_dict()) != saved['state_fingerprint']):
                raise RuntimeError('read-only head/WM source changed during dev evaluation')
            write_json(out/'evaluation.json', result)
            lines = [(training_dir/'summary.txt').read_text(encoding='utf-8'),
                '===== FIXED DEV512 REQUEST / FINAL EPOCH3 EGO =====',
                'population='+json.dumps(population_summary),
                f'status={result["state"]["status"]}; windows={result["state"]["completed_windows"]}',
                'No dev selection/retraining, alignment, future visibility or threshold tuning.',
                'setting        horizon       IoU      mIoU']
            for name, r in result['reports'].items():
                for h, row in [*r['horizons'].items(), ('Avg', r['average'])]:
                    fmt = lambda x: 'NA' if x is None else f'{x:.6f}'
                    lines.append(f'{name:14} {h:>6} {fmt(row["IoU"]):>10} {fmt(row["standard_mIoU"]):>10}')
            lines += ['trajectory_errors='+json.dumps(result['trajectory']),
                f'seconds={result["state"]["seconds"]:.2f}; quality evaluation, NOT FPS.',
                'Research dev, not independent test; no automatic promotion or larger evaluation.']
            text = '\n'.join(lines)+'\n'; (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)
    finally: predictor.close()
    return 0 if result['state']['status'] == 'complete' else 130


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
