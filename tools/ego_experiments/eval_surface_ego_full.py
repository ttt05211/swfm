#!/usr/bin/env python3
"""One fixed dev64 evaluation of the expanded, final epoch1 ego head."""
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
from tools.ego_experiments.train_surface_ego_full import ROOT, SELECTED_PROTOCOL, IMPLEMENTATION


def validate_selected(saved, *, root=ROOT):
    c = saved['contract']
    if (saved.get('protocol') != SELECTED_PROTOCOL or saved.get('evaluation_only') is not True or
            saved.get('training_completed') is not True or saved.get('config') != c['head_config'] or
            not saved['selected_epoch'] == saved['completed_epochs'] == c['schedule']['epochs'] == 1 or
            tensor_state_fingerprint(saved['state_dict']) != saved['state_fingerprint']):
        raise RuntimeError('completed final-epoch1 ego export required, not last.pt/legacy head')
    if implementation_fingerprint(root) != c['feature_geometry_implementation']:
        raise RuntimeError('historical feature/geometry implementation changed')
    if {p: digest_file(Path(root)/p) for p in IMPLEMENTATION} != c['implementation']:
        raise RuntimeError('expanded training/evaluation implementation changed')
    if digest_file(c['frozen_checkpoint']) != c['frozen_sha256']:
        raise RuntimeError('frozen WM/CCR checkpoint changed')
    return c


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('training-dir', 'dataroot', 'stc-root', 'plan-cache', 'population-manifest', 'out-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--config', default=str(ROOT/'configs/real_motion_occfm.yaml'))
    p.add_argument('--cpu-workers', type=int, default=4); p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true'); return p


def main(stop_event=None, argv=None):
    a = parser().parse_args(argv); out = Path(a.out_dir).resolve(); training_dir = Path(a.training_dir).resolve()
    head_path = training_dir/'head_epoch1.pt'; head_sha = digest_file(head_path)
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
    windows, population = source.select('dev64', manifest['parent_keys']); inventory = source.preflight(windows)
    # Existing research dev, not an independent test; no checkpoint selection here.
    if {w.scene for w in windows} & {k[0] for k in training['keys']}:
        raise RuntimeError('dev population overlaps full TRAIN scenes')
    nusc = NuScenesWindowSource(a.dataroot).nusc
    contract = dict(protocol=SELECTED_PROTOCOL+'_dev64', training=training, head_sha256=head_sha,
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
                '===== FIXED DEV64 / EXPANDED FINAL EPOCH1 EGO =====',
                f'status={result["state"]["status"]}; windows={result["state"]["completed_windows"]}',
                'No dev selection/retraining, alignment, future visibility or threshold tuning.',
                'setting        horizon       IoU      mIoU']
            for name, r in result['reports'].items():
                for h, row in [*r['horizons'].items(), ('Avg', r['average'])]:
                    fmt = lambda x: 'NA' if x is None else f'{x:.6f}'
                    lines.append(f'{name:14} {h:>6} {fmt(row["IoU"]):>10} {fmt(row["standard_mIoU"]):>10}')
            lines += ['trajectory_errors='+json.dumps(result['trajectory']),
                f'seconds={result["state"]["seconds"]:.2f}; quality evaluation, NOT FPS.',
                'Research dev, not independent test; no automatic promotion or full dev evaluation.']
            text = '\n'.join(lines)+'\n'; (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)
    finally: predictor.close()
    return 0 if result['state']['status'] == 'complete' else 130


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
