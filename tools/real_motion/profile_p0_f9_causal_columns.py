#!/usr/bin/env python3
"""Read-only same-window profiling; no training, gate selection or report promotion."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import cProfile
import pstats
import json
import numpy as np
import torch
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from real_motion.nuscenes_adapter import NuScenesWindowSource
from tools.real_motion import causal_column_common as common
from tools.real_motion.train_p0_f9_causal_columns import load_columns
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import align_records, sha256


def check_reference(provider, source, record, model):
    prep = provider.prepare_columns(source, record, include_gt=True)
    checked = 0
    for h in common.REPORT:
        plan = common.candidate_plan(prep, h, provider.pcfg.grid, model.config)
        cache = common.ColumnFeatureSampler(prep, h, plan, provider.pcfg.grid, model.config,
            common.pose_motion, workers=provider.workers)
        selected = []
        for actor in np.unique(plan.actor):
            indices = np.flatnonzero(plan.actor == actor)
            selected.extend(indices[np.linspace(0, len(indices)-1, min(8, len(indices)), dtype=int)])
        small = plan.subset(np.unique(selected).astype(np.int64))
        old = common.sample_column_features(prep, h, small, provider.pcfg.grid, model.config)
        new = cache.sample(small, common.sample_column_features)
        if any(not np.array_equal(v, new[k]) for k, v in old.items()):
            raise RuntimeError(f'real-data reference feature mismatch at horizon {h}; do not use optimized evaluation')
        checked += len(small)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--window', type=int, default=220, help='1-based original ordered dev window')
    parser.add_argument('--cpu-workers', type=int, default=8)
    parser.add_argument('--dataroot', default='/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes')
    args = parser.parse_args()
    if min(args.window, args.cpu_workers) < 1: parser.error('positive window/workers required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA/BF16 required')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]; checkpoint = Path(args.model_dir)/'candidate.pt'
    digest = sha256(checkpoint)
    cfg = load_runtime_config(root/'configs/real_motion_occfm.yaml', [])
    device = torch.device('cuda')
    ck, model = load_columns(checkpoint, device, base_sha=CLEAN_SHA256,
        config_sha=stable_json_fingerprint(cfg), allow_diagnostic=True)
    if ck['checkpoint_role'] != 'calibrated_candidate': raise RuntimeError('candidate.pt required')
    if args.window > len(ck['dev_keys']): parser.error('window exceeds original population')
    info = Path(args.dataroot)/'nuscenes_infos_val_temporal_v3_scene.pkl'
    if sha256(info) != ck['info_fingerprints']['dev']: raise RuntimeError('dev info differs from original run')
    provider = common.FrozenColumns(root/'outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt',
        CLEAN_SHA256, make_prepare_config(cfg), device, args.cpu_workers)
    source = NuScenesWindowSource(args.dataroot, info_pkl=str(info), verbose=False)
    _, records = load_cache(root/'data/p0_f9_v18_se2_val_all_4369.pt')
    record = align_records(records, [tuple(ck['dev_keys'][args.window-1])])[0]; del records
    checked = check_reference(provider, source, record, model)
    print(f'REFERENCE FEATURE EXACTNESS: {checked} actor-stratified queries / all six history frames PASS', flush=True)
    profile = cProfile.Profile()
    def progress(row):
        if row['window'] == 1: profile.enable()
        else:
            profile.disable()
            print('SECOND_WINDOW_TIMING:', json.dumps(row, ensure_ascii=False), flush=True)
    common.evaluate_columns(provider, source, [record, record], model, tuple(ck['thresholds']), progress=progress)
    pstats.Stats(profile).strip_dirs().sort_stats('cumulative').print_stats(25)
    if sha256(checkpoint) != digest: raise RuntimeError('checkpoint changed during read-only profile')
    print('Profiling only; original checkpoint/thresholds unchanged; no formal validation or deployment promotion.')


if __name__ == '__main__': main()
