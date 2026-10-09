#!/usr/bin/env python3
"""Frozen Clean Joint Surface mean -> I2-World Occ3D-Waymo 2Hz zero-shot."""
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

from real_motion.waymo_i2world import (WaymoI2WorldSource, PROTOCOL, UPSTREAM_URL,
                                      SHAPE, file_sha256, fingerprint)
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution
from tools.real_motion.joint_surface_checkpoint_selection import AVERAGE_EPOCHS, AVERAGE_NAME, load_evaluation_model
from tools.real_motion.eval_p0_f9_joint_surface_mean_full import find_frozen_bundle
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_surface_long_rollout_common import (SurfaceRolloutProvider, SurfaceBlockExecution,
                                                               verify_first_block)
from tools.real_motion.waymo_zero_shot_common import evaluate_windows, write_json

IMPLEMENTATION_FILES = (
    'real_motion/waymo_i2world.py', 'tools/real_motion/waymo_zero_shot_common.py',
    'tools/real_motion/eval_p0_f9_joint_surface_waymo.py',
    'tools/real_motion/joint_surface_long_rollout_common.py',
    'tools/real_motion/joint_long_rollout_common.py',
    'tools/real_motion/eval_p0_f9_v18_zero_shot_long_rollout.py',
    'tools/real_motion/causal_column_common.py',
    'tools/real_motion/benchmark_p0_f9_v18_runtime.py',
    'real_motion/joint_surface_ccr.py', 'real_motion/surface_canonical_repair.py',
    'real_motion/surface_ccr_execution.py', 'real_motion/surface_projection_execution.py',
    'real_motion/canonical_causal_repair.py', 'real_motion/strong_majority_execution.py',
    'real_motion/runtime_fastpath.py', 'real_motion/native/column_cpu.cpp',
    'real_motion/strong_w2det.py', 'real_motion/strong_warp_execution.py',
    'real_motion/canonical_repair_execution.py', 'real_motion/canonical_repair_context.py',
    'real_motion/native_column_cpu.py', 'real_motion/local_st_world_model_v18_se2.py',
    'real_motion/local_st_world_model_v17.py', 'real_motion/rigid_transport.py',
    'tools/real_motion/ccr_screen_common.py', 'tools/real_motion/joint_column_common.py',
)


class WaymoSurfaceProvider(SurfaceRolloutProvider):
    """Same predictor as Surface rollout, without irrelevant E14/nuscenes files."""
    def __init__(self, joint, pcfg, device, workers):
        self.joint = joint; self.model = joint.transport; self.reference = self.control = None
        self.reference_enabled = False; self.latents_checked = True; self.columns_checked = False
        self.dim = joint.transport.config.d_model
        self.pcfg, self.device, self.workers = pcfg, torch.device(device), workers
        self.strong = StrongW2DetConfig(free_label=17)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--waymo-root', required=True)
    p.add_argument('--info-file'); p.add_argument('--pose-file')
    p.add_argument('--checkpoint', help='already frozen weights-only 5/6/8/12/14 mean')
    p.add_argument('--runs-root'); p.add_argument('--run-dir'); p.add_argument('--source-bundle-dir')
    p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    p.add_argument('--audit-only', action='store_true', help='no model/CUDA; metadata, files and first-history encoding audit')
    p.add_argument('--raw-free-label', type=int, choices=(23, 15), default=23)
    p.add_argument('--expected-scenes', type=int, default=202)
    p.add_argument('--max-windows', type=int, default=0, help='explicit DIAGNOSTIC prefix, 0=all anchors')
    p.add_argument('--frame-cache-mib', type=int, default=256)
    p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--checkpoint-every', type=int, default=8)
    p.add_argument('--execution', choices=('numpy', 'native_parallel'), default='native_parallel')
    p.add_argument('--no-graphs', action='store_true')
    p.add_argument('--parallel-majority', action='store_true', help='existing exact 4-worker Strong majority; no forecast cache')
    p.add_argument('--config', default=str(Path(__file__).resolve().parents[2]/'configs/real_motion_occfm.yaml'))
    return p


def resolve_checkpoint(a):
    if a.checkpoint:
        return Path(a.checkpoint).resolve()
    if not a.runs_root or not a.run_dir:
        raise ValueError('set --checkpoint OR --runs-root and --run-dir for the already frozen mean')
    bundle = find_frozen_bundle(a.runs_root, a.run_dir, a.source_bundle_dir)
    return Path(bundle['candidates'][AVERAGE_NAME]['path']).resolve()


def text_report(result):
    lines = ['===== SURFACE CCR / I2-WORLD WAYMO 2HZ ZERO-SHOT =====',
             'status='+result['status'], f"windows={result['completed_windows']}/{result['contract']['windows']}",
             'Frozen nuScenes mean; NO Waymo training/calibration/threshold search.',
             '4 history frames including t0 -> 6 futures; report future indices 1/3/5 = nominal 1/2/3s.',
             'Scene boundaries repeat valid frames/targets, as in official code.',
             'branch/horizon             IoU   I2_mIoU standard_mIoU']
    def fmt(value):
        return 'NA' if value is None else f'{value:.6f}'
    for name, report in result['reports'].items():
        for h, row in [('average', report['average']), *report['horizons'].items()]:
            lines.append(f'{name+"/"+h:24s} {fmt(row["IoU"]):>10} {fmt(row["i2world_mIoU"]):>10} '
                         +f'{fmt(row["standard_mIoU"]):>13}')
    lines += ['I2_mIoU excludes exactly-zero class IoU (upstream behavior); standard includes valid zero classes.',
              'Raw labels: '+str(result['contract']['data']['label_encoding']),
              'Metrics use no lidar/camera visibility mask. Moving metrics are NOT measured for this protocol.',
              'Future ego poses are supplied conditioning; future occupancy is read only AFTER prediction.',
              'Wall evaluation timings include preparation/GT/metrics, NOT formal Dense Forecast FPS.',
              'Weights and all nuScenes caches unchanged. No automatic promotion.',
              'details=waymo_validation.json; resume integer counts in state.json']
    return '\n'.join(lines)+'\n'


def main(stop_event=None, argv=None):
    p = parser(); a = p.parse_args(argv)
    if (not 1 <= a.cpu_workers <= 8 or not 0 <= a.frame_cache_mib <= 4096
            or a.checkpoint_every < 1 or a.max_windows < 0 or a.expected_scenes < 0):
        p.error('invalid bounded execution/population settings')
    if a.audit_only and a.resume:
        p.error('audit-only is not an evaluation resume')
    out = Path(a.out_dir).resolve(); data_root = Path(a.waymo_root).resolve()
    if out.is_relative_to(data_root) or any((d/'training.json').is_file() for d in (out, *out.parents)):
        p.error('output cannot be inside data/original training directories')
    if a.resume:
        if not (out/'contract.json').is_file() or not (out/'state.json').is_file():
            p.error('resume requires the SAME existing Waymo evaluation output')
    elif out.exists():
        p.error('new output required; never overwrite an existing experiment')
    source = WaymoI2WorldSource.from_files(data_root, info_file=a.info_file, pose_file=a.pose_file,
                                         raw_free_label=a.raw_free_label, cache_mib=a.frame_cache_mib)
    if a.expected_scenes and source.metadata['scenes'] != a.expected_scenes:
        p.error(f"Waymo scene count {source.metadata['scenes']} != {a.expected_scenes}; check official metadata")
    if a.max_windows > len(source.windows):
        p.error('--max-windows exceeds available anchors; do not invent samples')
    selected = source.windows[:a.max_windows or len(source.windows)]
    inventory = source.preflight(selected)
    if a.audit_only:
        source.prediction_inputs(selected[0])  # history encoding only, not a future GT audit
        out.mkdir(parents=True)
        audit = dict(status='audit_only', data=source.metadata, inventory=inventory,
                     manifest_fingerprint=source.manifest_fingerprint, windows=len(selected),
                     future_GT_loaded=False, model_loaded=False, upstream=UPSTREAM_URL)
        write_json(out/'audit.json', audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        return 0
    if not torch.cuda.is_available():
        p.error('real CUDA required for evaluation; --audit-only does not require CUDA')
    cfg = load_runtime_config(a.config); pcfg = make_prepare_config(cfg)
    if (tuple(pcfg.grid.shape_hwd) != SHAPE or not np.allclose(pcfg.grid.voxel_size, (.4,)*3, rtol=0, atol=1e-12)
            or not np.allclose((pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min), (-40, -40, -1), rtol=0, atol=1e-12)
            or pcfg.future_frames != 6 or pcfg.free_label != 17 or pcfg.frame_dt_s != .5):
        p.error('I2-World Waymo requires unchanged 200x200x16 XYZ / 0.4m / 2Hz predictor')
    checkpoint = resolve_checkpoint(a); ck_sha = file_sha256(checkpoint)
    if out.is_relative_to(checkpoint.parent):
        p.error('output cannot be inside the source mean directory')
    saved, joint = load_evaluation_model(checkpoint, device='cuda', z_bins=16)
    if (saved.get('source_epochs') != list(AVERAGE_EPOCHS) or not saved.get('averaging')
            or joint.transport.config.history_frames != 4 or file_sha256(checkpoint) != ck_sha):
        raise RuntimeError('frozen 5/6/8/12/14 FOUR-history mean required; weights changed or wrong recipe')
    root = Path(__file__).resolve().parents[2]
    contract = dict(protocol=PROTOCOL, upstream=UPSTREAM_URL, windows=len(selected),
        population='all_official_2hz_anchors' if not a.max_windows else 'DIAGNOSTIC_PREFIX_NOT_FULL',
        checkpoint=str(checkpoint), checkpoint_sha256=ck_sha, source_epochs=list(AVERAGE_EPOCHS),
        data=source.metadata, data_root=str(data_root), inventory=inventory,
        manifest_fingerprint=source.manifest_fingerprint, thresholds=[.5, None],
        frame_cache_mib=a.frame_cache_mib, cpu_workers=a.cpu_workers, execution=a.execution,
        graphs=not a.no_graphs, parallel_majority=a.parallel_majority,
        runtime_environment={k: v for k, v in sorted(os.environ.items()) if k.startswith('SWFM_')},
        torch_version=str(torch.__version__),
        config_sha256=file_sha256(a.config), implementation={f: file_sha256(root/f) for f in IMPLEMENTATION_FILES},
        input_budget_note='ours FOUR total including t0; upstream temporal tokenizer uses previous-cache/current semantics, not identical architecture',
        future_ego_poses='metadata conditioning, not predicted', Waymo_adaptation='label mapping only; no learned adaptation')
    if not a.resume:
        out.mkdir(parents=True)
    with evaluation_lock(out):
        if a.resume:
            previous = json.loads((out/'contract.json').read_text(encoding='utf-8'))
            if fingerprint(previous) != fingerprint(contract):
                raise RuntimeError('Waymo resume population/weights/data/config/execution/implementation changed')
            if (out/'waymo_validation.json').is_file():
                complete = json.loads((out/'waymo_validation.json').read_text(encoding='utf-8'))
                if complete.get('status') == 'complete':
                    print('Waymo evaluation already complete; original result unchanged: '+str(out/'summary.txt'), flush=True)
                    return 0
            resume = json.loads((out/'state.json').read_text(encoding='utf-8'))
        else:
            write_json(out/'contract.json', contract); resume = None
        torch.set_num_threads(1)
        provider = WaymoSurfaceProvider(joint, pcfg, 'cuda', a.cpu_workers)
        execution = SurfaceBlockExecution(provider, mode=a.execution, workers=a.cpu_workers,
                                           query_workers=a.cpu_workers, graphs=not a.no_graphs)
        majority = ParallelNativeMajority(workers=4) if a.parallel_majority else None
        started = time.perf_counter()
        @torch.no_grad()
        def predict(record, raw, *, verify):
            tick = time.perf_counter()
            scope = strong_majority_execution(majority) if majority is not None else nullcontext()
            with scope:
                prep = provider.prepare_columns(None, record, include_gt=False, raw_window=raw)
            stage = {'history_and_transport_prepare': time.perf_counter()-tick}
            dense, edits, readout, probability = execution.predict(prep)
            stage.update(readout)
            if verify:
                # Rebuild independently OUTSIDE the optional majority scope.
                # All four motion inputs, transport, probabilities and six
                # dense repaired frames must match the existing predictor.
                verify_first_block(provider, prep.state['rec'], prep, dense, probability, execution)
            return prep.baseline, dense, edits, stage
        with (out/'progress.jsonl').open('a', encoding='utf-8') as handle:
            def progress(row):
                handle.write(json.dumps(row, allow_nan=False)+'\n'); handle.flush()
                if row['window'] % 32 == 0 or row['window'] == len(selected):
                    print(f"WAYMO_ZERO_SHOT {row['window']}/{row['windows']} seconds={row['seconds']:.3f}", flush=True)
            print(f'WAYMO I2-WORLD 2HZ: anchors={len(selected)} scenes={source.metadata["scenes"]} '
                  +f'raw_free={a.raw_free_label}; checkpoint={checkpoint}', flush=True)
            try:
                result = evaluate_windows(source, selected, predict, contract, saved=resume,
                    save=lambda state: write_json(out/'state.json', state), progress=progress,
                    stop_event=stop_event, checkpoint_every=a.checkpoint_every)
            finally:
                execution.close()
                if majority is not None:
                    majority.close()
        if file_sha256(checkpoint) != ck_sha:
            raise RuntimeError('source frozen mean changed during evaluation')
        result.update(contract=contract, elapsed_seconds_this_invocation=time.perf_counter()-started,
                      frame_io=dict(reads=source.io_reads, hits=source.cache_hits, RAM_bytes=source.cache_bytes))
        write_json(out/'waymo_validation.json', result)
        (out/'summary.txt').write_text(text_report(result), encoding='utf-8')
        print(text_report(result), flush=True)
        print('RESULT: '+str(out/'summary.txt'), flush=True)
        return 0


if __name__ == '__main__':
    event = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *args: event.set())
    sys.exit(main(event))
