#!/usr/bin/env python3
"""Formal six-dense-frame FPS of the already frozen clean-joint mean, no scoring."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import signal
import threading
import numpy as np
import torch

from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.eval_p0_f9_joint_surface_mean_full import find_frozen_bundle
from tools.real_motion.compare_p0_f9_joint_surface_checkpoints import verify_sources, VAL_WINDOWS
from tools.real_motion.joint_surface_checkpoint_selection import (
    AVERAGE_NAME, load_evaluation_model, training_implementation, weight_fingerprint,
)
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json
from tools.real_motion.surface_ccr_validation_common import paired_speed


def summary(speed):
    lines = ['===== CLEAN JOINT FROZEN MEAN / FORMAL DENSE FORECAST FPS =====',
        f'windows={speed["fps_windows"]} scenes={speed["scenes"]} repeats={speed["repeats"]} actual_cuda={speed["actual_cuda"]}',
        'epoch5/6/8/12/14 mean; weighted ADD raw0.5 / REMOVEoff; no weights/threshold changes.',
        'FPS=6/mean_six_frame_latency, NOT mean per-window FPS.']
    for mode, seconds in speed['six_frame_mean_seconds'].items():
        lines.append(f'{mode}: six_ms={1000*seconds:.3f} FPS={6/seconds:.3f} '
                     f'P90_ms={speed["p90_six_ms"][mode]:.3f}')
    if speed.get('strong_warp_comparison'):
        original = speed['six_frame_mean_seconds']['surface_fused_graph']
        buffered = speed['six_frame_mean_seconds']['surface_fused_buffered_graph']
        lines += [f'SAME-WINDOW whole-forecast speedup={original/buffered:.4f} (Strong scheduling ONLY); no automatic backend promotion.',
                  'buffered_host_stage_ms=' + json.dumps(speed['stages_mean_ms']['surface_fused_buffered_graph'])]
    lines += [f'probability + SIX dense byte parity: {speed["probability_and_six_dense_parity_windows"]}/{speed["fps_windows"]}',
        'fixed_execution=' + speed['selected_execution'],
        'graph_execution=' + json.dumps(speed['graph_execution']),
        'host_stage_ms_NOT_GPU_utilization=' + json.dumps(speed['stages_mean_ms'][speed['selected_execution']]),
        'peak_memory_mib=' + json.dumps(speed['memory_mib']),
        f'Excluded history_prepare_ms={1000*speed["history_prepare_seconds_per_window"]:.3f} '
        f'optimized_surface_descriptor_ms={1000*speed["optimized_surface_descriptor_prepare_seconds_per_window"]:.3f}',
        'boundary: ' + speed['boundary'], 'excludes: ' + speed['excludes'],
        'Population differs from old20: do not claim a paired cross-run speedup.',
        'No training, quality evaluation/selection, forecast-cache writes or deployment promotion.']
    return '\n'.join(lines) + '\n'


def main(argv=None, stop_event=None):
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for name in ('run-dir', 'runs-root', 'out-dir', 'base-checkpoint', 'dev-cache', 'dev-info', 'dataroot'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--source-bundle-dir')
    p.add_argument('--windows', type=int, default=256)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--compare-strong-warp', action='store_true',
                   help='paired original/fused_graph vs buffered Strong/fused_graph; no automatic promotion')
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-workers', type=int, default=10)
    a = p.parse_args(argv); out = Path(a.out_dir).resolve()
    if out.exists(): p.error('new output required; never overwrite an experiment')
    if not 1 <= a.windows <= VAL_WINDOWS or not 1 <= a.repeats <= 10 or not 1 <= a.cpu_workers <= 16:
        p.error('invalid windows/repeats/CPU budget')
    if any((directory/'training.json').is_file() for directory in (out, *out.parents)):
        p.error('FPS output must be outside training directories')
    bundle = find_frozen_bundle(a.runs_root, a.run_dir, a.source_bundle_dir)
    if out.is_relative_to(Path(bundle['source_comparison_directory'])):
        p.error('FPS output must be outside the original comparison')
    contract = bundle['audit']['contract']; root = Path(__file__).resolve().parents[2]
    for name in ('dev_cache', 'dev_info', 'base_checkpoint'):
        if sha256(getattr(a, name)) != contract['data'][name]:
            raise RuntimeError('original data/reference changed: ' + name)
    if sha256(a.base_checkpoint) != CLEAN_SHA256 or str(Path(a.dataroot).resolve()) != contract['dataroot']:
        raise RuntimeError('original renderer/dataroot required')
    cfg = load_runtime_config(a.config, a.override)
    if stable_json_fingerprint(cfg) != contract['runtime_config_fingerprint']:
        raise RuntimeError('original runtime config required')
    if training_implementation(root) != contract['implementation'] or str(torch.__version__) != contract['torch_version']:
        raise RuntimeError('original model implementation/Torch environment required')
    device = require_cuda(a.device); torch.set_num_threads(1); pcfg = make_prepare_config(cfg)
    saved, joint = load_evaluation_model(bundle['candidates'][AVERAGE_NAME]['path'],
                                         device=device, z_bins=int(pcfg.grid.shape_hwd[2]))
    if stable_json_fingerprint(saved['training_contract']) != stable_json_fingerprint(contract):
        raise RuntimeError('mean training contract changed')
    _, records = load_cache(a.dev_cache); record_keys(records)
    if len(records) != VAL_WINDOWS: raise RuntimeError('complete original VAL4369 required for FPS sampling')
    # Randomize within each scene before the existing scene round-robin selector;
    # neither latency, predictions nor GT determine which windows enter the sample.
    order = np.random.default_rng(a.seed).permutation(len(records))
    records = [records[i] for i in order]
    provider = PilotProvider(a.base_checkpoint, CLEAN_SHA256, pcfg, device, a.cpu_workers, joint, None)
    execution = CanonicalCpuExecution('native_parallel', min(4, a.cpu_workers))
    provider.ccr_execution = execution
    source = CachedColumnSource(NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info, verbose=False), 1024)
    out.mkdir(parents=True); write_json(out/'source_bundle.json', bundle)
    print(f'FROZEN MEAN: {bundle["candidates"][AVERAGE_NAME]["path"]}', flush=True)
    print(f'FPS ONLY: {a.windows} scene-balanced random windows x {a.repeats}; fresh Strong/live SIX outputs.', flush=True)
    try:
        with (out/'progress.jsonl').open('x', encoding='utf-8') as log:
            def progress(row):
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            speed = paired_speed(provider, source, records, joint, joint.columns, None,
                stop_event=stop_event, windows=a.windows, repeats=a.repeats, stress_windows=0,
                surface_only=True, progress=progress, compare_strong_warp=a.compare_strong_warp)
        verify_sources(bundle)
        if weight_fingerprint(joint.state_dict()) != saved['weight_fingerprint']:
            raise RuntimeError('in-memory mean weights changed during read-only FPS')
        speed.update(protocol='p0_f9_joint_surface_frozen_mean_dense_fps_v1', status='complete',
            scenes=len({row['key'][0] for row in speed['population']}), seed=a.seed,
            sampling='seeded within-scene random + scene round-robin; no stress/GT/latency selection',
            population_fingerprint=stable_json_fingerprint(speed['population']),
            weights_sha256=bundle['candidates'][AVERAGE_NAME]['sha256'],
            source_bundle_fingerprint=bundle['fingerprint'], graph_capture_inside_timing=False,
            persistent_forecast_cache=False,
            execution_implementation={name: sha256(root/name) for name in (
                'real_motion/strong_warp_execution.py', 'real_motion/runtime_fastpath.py',
                'real_motion/final_dataflow.py', 'tools/real_motion/surface_ccr_validation_common.py')})
        write_json(out/'speed.json', speed)
        text = summary(speed); (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)
    except InterruptedError:
        write_json(out/'status.json', dict(status='interrupted', note='no checkpoint/cache changes; rerun in a NEW output directory'))
        return 130
    finally:
        execution.close()
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: event.set())
    sys.exit(main(stop_event=event))
