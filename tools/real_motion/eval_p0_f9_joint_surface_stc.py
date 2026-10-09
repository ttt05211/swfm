#!/usr/bin/env python3
"""Frozen Surface CCR: STC/GT occupancy x external Pred/GT ego, same population."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from contextlib import nullcontext
import json
import os
import signal
import threading
import time

import numpy as np
import torch

from real_motion.stc_camera_protocol import PROTOCOL, SETTINGS, SHAPE, STCFourSettingSource, UPSTREAM
from real_motion.waymo_i2world import file_sha256, fingerprint
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution
from tools.real_motion.eval_p0_f9_joint_surface_waymo import (
    WaymoSurfaceProvider, IMPLEMENTATION_FILES as BASE_IMPLEMENTATION)
from tools.real_motion.joint_surface_long_rollout_common import SurfaceBlockExecution, verify_first_block
from tools.real_motion.joint_surface_checkpoint_selection import AVERAGE_EPOCHS, AVERAGE_NAME, load_evaluation_model
from tools.real_motion.eval_p0_f9_joint_surface_mean_full import find_frozen_bundle
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest
from tools.real_motion.stc_camera_evaluation import evaluate
from tools.real_motion.waymo_zero_shot_common import write_json

IMPLEMENTATION_FILES = tuple(dict.fromkeys((*BASE_IMPLEMENTATION,
    'real_motion/stc_camera_protocol.py', 'tools/real_motion/stc_camera_evaluation.py',
    'tools/real_motion/eval_p0_f9_joint_surface_stc.py',
    'real_motion/waymo_i2world.py', 'real_motion/geometry.py', 'real_motion/prepared.py',
    'real_motion/surface_canonical_repair.py', 'real_motion/local_history_contract.py',
    'tools/real_motion/joint_surface_checkpoint_selection.py', 'real_motion/runtime_config.py')))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataroot', required=True); p.add_argument('--stc-root', required=True)
    p.add_argument('--plan-cache', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--population', choices=('dev64', 'dev512', 'all'), default='dev64')
    p.add_argument('--population-manifest'); p.add_argument('--audit-only', action='store_true')
    p.add_argument('--checkpoint'); p.add_argument('--runs-root'); p.add_argument('--run-dir')
    p.add_argument('--source-bundle-dir'); p.add_argument('--resume', action='store_true')
    p.add_argument('--frame-cache-mib', type=int, default=512)
    p.add_argument('--cpu-workers', type=int, default=4); p.add_argument('--checkpoint-every', type=int, default=8)
    p.add_argument('--execution', choices=('numpy', 'native_parallel'), default='native_parallel')
    p.add_argument('--no-graphs', action='store_true'); p.add_argument('--parallel-majority', action='store_true')
    p.add_argument('--config', default=str(Path(__file__).resolve().parents[2] / 'configs/real_motion_occfm.yaml'))
    return p


def summary(result):
    c = result['contract']; lines = ['===== FROZEN SURFACE CCR / STC FOUR SETTINGS =====',
        'status=' + result['status'], f'windows={result["completed_windows"]}/{c["windows"]}; population={c["population"]["population"]}',
        'Mean5/6/8/12/14; FOUR histories -> SIX futures; ADD raw0.5 / REMOVEoff.',
        'No training/calibration/threshold selection; Camera frontend=STCOcc-Res, NOT BEVStereo.',
        'Camera histories: full released semantics; NO GT lidar/camera visibility inputs.',
        'Scoring: full grid, no camera/lidar mask; future labels read after all FOUR predictions.',
        'Pred ego: external BEV-Planner+yaw from exact v4 cache, used in ALL future geometry.',
        'No post-hoc GT pose alignment; cached predicted z/tilt from t0, not future GT.',
        'Same planner-covered population for all rows; not claimed identical to literature paper population.',
        'setting       horizon        IoU       mIoU    I2_code_mIoU']
    def fmt(v):
        return 'NA' if v is None else f'{v:.6f}'
    for k in SETTINGS:
        r = result['reports'][k]
        for h, scores in [*r['horizons'].items(), ('Avg', r['average'])]:
            lines.append(f'{k:13s} {h:>7s} {fmt(scores["IoU"]):>11s} '
                         +f'{fmt(scores["standard_mIoU"]):>11s} {fmt(scores["i2world_mIoU"]):>15s}')
    lines += ['mIoU=standard valid-union semantic classes, retains zero scores; I2 code excludes exact-zero class IoUs.',
        'Parent keys absent from legacy planner cache: ' + str(len(c['population']['missing_parent_keys'])),
        'verified_settings=' + str(result['verified_settings']),
        'stage_seconds=' + json.dumps(result['stage_seconds']),
        'Quality evaluation wall time is NOT raw-camera end-to-end FPS.',
        'All old checkpoints/caches unchanged. Resume SAME output directory; no automatic promotion.']
    return '\n'.join(lines) + '\n'


def main(stop_event=None, argv=None):
    p = parser(); a = p.parse_args(argv)
    if not 1 <= a.cpu_workers <= 8 or a.checkpoint_every < 1 or not 0 <= a.frame_cache_mib <= 4096:
        p.error('invalid CPU/RAM/checkpoint bounds')
    if a.audit_only and a.resume:
        p.error('audit-only is not an evaluation resume')
    out = Path(a.out_dir).resolve()
    roots = [Path(x).resolve() for x in (a.dataroot, a.stc_root, a.plan_cache)]
    if any(out == root or out.is_relative_to(root) for root in roots) or any(
            (d / 'training.json').is_file() for d in (out, *out.parents)):
        p.error('output cannot be inside original data/cache/training directories')
    if a.resume:
        if not (out / 'contract.json').is_file() or not (out / 'state.json').is_file():
            p.error('resume requires SAME existing STC evaluation output')
    elif out.exists():
        p.error('new output required; never replace old experiments')
    source = STCFourSettingSource.from_files(a.dataroot, a.stc_root, a.plan_cache,
                                             cache_mib=a.frame_cache_mib)
    parent = None; pop_sha = None
    if a.population != 'all':
        if not a.population_manifest:
            p.error('dev64/dev512 requires the existing frozen population manifest')
        manifest, _, _ = load_manifest(a.population_manifest)
        parent = manifest['parent_keys']; pop_sha = file_sha256(a.population_manifest)
    selected, population = source.select(a.population, parent)
    print('STC_SHARED_POPULATION ' + json.dumps(population | {'selected_keys': 'saved in contract.json'}, ensure_ascii=False), flush=True)
    inventory = source.preflight(selected)
    if a.audit_only:
        # Input-only check; NEVER load future targets or the model here.
        for setting in SETTINGS:
            source.prediction_inputs(selected[0], setting)
        out.mkdir(parents=True)
        value = dict(status='audit_only', population=population, source=source.metadata,
                     inventory=inventory, future_targets_loaded=False, model_loaded=False)
        write_json(out / 'audit.json', value); print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        return 0
    if not torch.cuda.is_available():
        p.error('CUDA required for dataset evaluation; --audit-only is CPU-only')
    cfg = load_runtime_config(a.config); pcfg = make_prepare_config(cfg)
    if (tuple(pcfg.grid.shape_hwd) != SHAPE or pcfg.future_frames != 6 or pcfg.frame_dt_s != .5 or pcfg.free_label != 17
            or not np.allclose(pcfg.grid.voxel_size, (.4,) * 3, atol=1e-12, rtol=0)
            or not np.allclose((pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min), (-40, -40, -1), atol=1e-12, rtol=0)):
        p.error('unchanged Occ3D grid/six-future/2Hz predictor required')
    if a.checkpoint:
        checkpoint = Path(a.checkpoint).resolve()
    else:
        if not a.runs_root or not a.run_dir:
            p.error('set --checkpoint OR --runs-root/--run-dir to find frozen mean')
        bundle = find_frozen_bundle(a.runs_root, a.run_dir, a.source_bundle_dir)
        checkpoint = Path(bundle['candidates'][AVERAGE_NAME]['path']).resolve()
    if out.is_relative_to(checkpoint.parent):
        p.error('output cannot be inside source mean directory')
    ck_sha = file_sha256(checkpoint)
    saved, joint = load_evaluation_model(checkpoint, device='cuda', z_bins=16)
    if (saved.get('source_epochs') != list(AVERAGE_EPOCHS) or not saved.get('averaging')
            or joint.transport.config.history_frames != 4 or file_sha256(checkpoint) != ck_sha):
        raise RuntimeError('unchanged frozen 5/6/8/12/14 FOUR-history mean required')
    root = Path(__file__).resolve().parents[2]
    contract = dict(protocol=PROTOCOL, upstream=UPSTREAM, windows=len(selected), population=population,
        source=source.metadata, inventory=inventory, population_manifest_sha256=pop_sha,
        dataroot=str(source.root), stc_root=str(source.stc_root), plan_cache=str(source.plan_cache),
        checkpoint=str(checkpoint), checkpoint_sha256=ck_sha, source_epochs=list(AVERAGE_EPOCHS),
        thresholds=[.5, None], config_sha256=file_sha256(a.config),
        implementation={f: file_sha256(root / f) for f in IMPLEMENTATION_FILES},
        runtime_environment={k: v for k, v in sorted(os.environ.items()) if k.startswith('SWFM_')},
        torch_version=str(torch.__version__), cpu_workers=a.cpu_workers, frame_cache_mib=a.frame_cache_mib,
        execution=a.execution, graphs=not a.no_graphs, parallel_majority=a.parallel_majority)
    if not a.resume:
        out.mkdir(parents=True)
    with evaluation_lock(out):
        if a.resume:
            previous = json.loads((out / 'contract.json').read_text())
            if fingerprint(previous) != fingerprint(contract):
                raise RuntimeError('STC resume data/population/model/config/execution/implementation changed')
            if (out / 'evaluation.json').is_file():
                complete = json.loads((out / 'evaluation.json').read_text())
                if complete.get('status') == 'complete':
                    print('Already complete: ' + str(out / 'summary.txt')); return 0
            resume = json.loads((out / 'state.json').read_text())
        else:
            write_json(out / 'contract.json', contract); resume = None
        torch.set_num_threads(1)
        provider = WaymoSurfaceProvider(joint, pcfg, 'cuda', a.cpu_workers)
        execution = SurfaceBlockExecution(provider, mode=a.execution, workers=a.cpu_workers,
                                           query_workers=a.cpu_workers, graphs=not a.no_graphs)
        majority = ParallelNativeMajority(workers=4) if a.parallel_majority else None
        @torch.no_grad()
        def predict(record, raw, *, verify):
            tick = time.perf_counter()
            scope = strong_majority_execution(majority) if majority is not None else nullcontext()
            with scope:
                prep = provider.prepare_columns(None, record, include_gt=False, raw_window=raw)
            stage = dict(history_and_transport_prepare=time.perf_counter() - tick)
            dense, _, details, probability = execution.predict(prep); stage.update(details)
            if verify:
                verify_first_block(provider, prep.state['rec'], prep, dense, probability, execution)
            return dense, stage
        start = time.perf_counter()
        try:
            with (out / 'progress.jsonl').open('a', encoding='utf-8') as handle:
                def progress(row):
                    handle.write(json.dumps(row, allow_nan=False) + '\n'); handle.flush()
                    if row['window'] == 1 or row['window'] % 16 == 0 or row['window'] == row['windows']:
                        print(f'STC_FOUR_SETTINGS {row["window"]}/{row["windows"]} seconds={row["seconds"]:.3f}', flush=True)
                result = evaluate(source, selected, predict, contract, saved=resume,
                    save=lambda s: write_json(out / 'state.json', s), progress=progress,
                    stop_event=stop_event, checkpoint_every=a.checkpoint_every)
        finally:
            execution.close()
            if majority is not None:
                majority.close()
        if file_sha256(checkpoint) != ck_sha:
            raise RuntimeError('source mean changed during evaluation')
        result.update(contract=contract, elapsed_seconds_this_invocation=time.perf_counter() - start)
        write_json(out / 'evaluation.json', result)
        (out / 'summary.txt').write_text(summary(result), encoding='utf-8')
        print(summary(result), flush=True); print('RESULT: ' + str(out / 'summary.txt'), flush=True)
    return 0


if __name__ == '__main__':
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    sys.exit(main(stopped))
