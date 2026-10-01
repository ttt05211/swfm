#!/usr/bin/env python3
"""Bounded one-stage random-init screen, live supervision, paired scratch V18 control."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import copy
import json
import subprocess
import time
import numpy as np
import torch
from real_motion.joint_causal_columns import JointCausalColumns, PROTOCOL, CONTRACT, LINK_PROTOCOL
from real_motion.causal_column_completion import ColumnConfig
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.joint_column_common import JointColumnProvider, count_proposals, weights_from_counts, train_window
from tools.real_motion.causal_column_common import calibrate_columns, evaluate_columns, safe_metrics
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.v18_source_interaction_common import select_population, validate_records
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256, validate_clean_e14_checkpoint
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, atomic_checkpoint, finite_json
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP


def load_joint(path, device, *, reference_sha, config_sha, allow_diagnostic=False):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if (ck.get('protocol') != PROTOCOL or ck.get('training_contract') != CONTRACT or ck.get('source_link') != LINK_PROTOCOL
            or ck.get('reference_checkpoint_sha256') != reference_sha or ck.get('runtime_config_fingerprint') != config_sha
            or ck.get('checkpoint_role') not in ('resume_last', 'calibrated_candidate')):
        raise RuntimeError('joint checkpoint/reference/config contract mismatch')
    if not allow_diagnostic and (ck['checkpoint_role'] != 'calibrated_candidate' or ck.get('mode') != 'screen'
                                 or not ck.get('screen_pass') or ck.get('successful_updates', 0) <= 0):
        raise RuntimeError('failed/smoke/last joint candidate cannot be deployed')
    model = JointCausalColumns(config_from_mapping_v17(ck['model_configs']['motion']),
                              ColumnConfig(**ck['model_configs']['columns'])).to(device)
    model.load_state_dict(ck['state_dict'], strict=True); model.eval()
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()): raise RuntimeError('nonfinite joint state')
    weights = ck['TRAIN_weights']
    if (not np.isclose(float(model.columns.generation_pos_weight), weights['generation_pos_weight'], rtol=0, atol=1e-6)
            or not np.allclose(model.columns.refine_class_weights.cpu().numpy(), weights['refine_class_weights'], rtol=0, atol=1e-6)):
        raise RuntimeError('persisted TRAIN correction mismatch')
    if ck['checkpoint_role'] == 'calibrated_candidate':
        gates = ck.get('thresholds')
        if (gates is None or len(gates) != 3 or any(t is not None and (not np.isfinite(t) or not .5 <= t <= 1) for t in gates)):
            raise RuntimeError('invalid final TRAIN thresholds')
        if not allow_diagnostic and not any(t is not None for t in gates): raise RuntimeError('disabled identity candidate')
    return ck, model


def summary_text(summary):
    lines = ['===== ONE-STAGE JOINT TRANSPORT + CAUSAL COLUMNS =====', f'protocol: {PROTOCOL}',
        f"mode: {summary['mode']}", f"successful_updates: {summary['successful_updates']}",
        f"train_windows: {summary['train_windows']}", f"training_window_passes: {summary['training_window_passes']}",
        f"sampled_columns: {summary['sampled_columns']}", 'E14 weights NOT used for initialization; geometry stop-gradient only.',
        f"gradient_link_observed: {summary['gradient_link_observed']}", f"thresholds_from_TRAIN_only: {summary['thresholds']}"]
    for population in ('dev64', 'all'):
        if population not in summary['evaluation']: continue
        row = summary['evaluation'][population]
        lines += [f"\n===== {population}: {row['windows']} windows / {row['scenes']} scenes =====",
            f"joint_transport_only_mIoU: {row['baseline']['mIoU']:.6f}"]
        for name, item in row['variants'].items():
            d, q = item['delta_vs_v18_pp'], item['quality']
            lines += [f"{name}: vs_joint_transport dMiOU={d['mIoU']:+.6f} dIoU={d['IoU']:+.6f} "
                f"dMovingMicro={d['MovingMicro']:+.6f} added={q.get('added',0)} removed={q.get('removed',0)} "
                f"semantic_precision={q['addition_semantic_precision']} scenes={item['scene_delta']}"]
        for name, d in row.get('joint_vs_reference_pp', {}).items():
            lines += [f"joint vs {name}: dMiOU={d['mIoU']:+.6f} dIoU={d['IoU']:+.6f} dMovingMicro={d['MovingMicro']:+.6f}"]
        lines += [f"screen_gate: {summary['gates'].get(population)}"]
    lines += [f"route: {summary['route']}", f"elapsed_seconds: {summary['elapsed_seconds']:.2f}",
        f"stage_seconds: {summary['stage_seconds']}",
        f"candidate_checkpoint: {summary['candidate_checkpoint']}",
        'This is a one-window-pass from-scratch screen, NOT a converged full-data method.',
        'No full4369, dev-selected best checkpoint, automatic retry or deployment promotion.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_config_args(parser)
    for name in ('train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint', 'dataroot', 'train-info', 'dev-info', 'out-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mode', choices=('smoke', 'screen'), default='screen')
    parser.add_argument('--device', default='cuda'); parser.add_argument('--cpu-workers', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--frame-cache-mib', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--resume', help='matching joint last.pt into a NEW output directory')
    args = parser.parse_args(); started = time.perf_counter(); out = Path(args.out_dir)
    if out.exists(): parser.error('NEW output directory required')
    for name in ('config', 'train_cache', 'dev_cache', 'population_manifest', 'base_checkpoint', 'train_info', 'dev_info'):
        if not str(getattr(args, name) or '').strip() or not Path(getattr(args, name)).is_file(): parser.error(f'missing {name}')
    if not Path(args.dataroot).is_dir() or min(args.cpu_workers, args.eval_batch_size) < 1 or args.frame_cache_mib < 0:
        parser.error('invalid paths/budgets')
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): raise RuntimeError('CUDA/BF16 required')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)
    cfg = load_runtime_config(args.config, args.override); pcfg = make_prepare_config(cfg); config_sha = stable_json_fingerprint(cfg)
    manifest, dev64, _ = load_manifest(args.population_manifest)
    if len(dev64) != 64 or len(manifest['parent_keys']) != 512 or manifest['selected_key_fingerprint'] != DEV64_FP:
        raise RuntimeError('frozen scene-balanced dev64/dev512 manifest required')
    meta, train_all = load_cache(args.train_cache); train_keys_all = record_keys(train_all)
    if args.mode == 'screen' and len(train_keys_all) != 20430: raise RuntimeError('screen requires full20430 input cache')
    count = 1024 if args.mode == 'screen' else min(4, len(train_all)//2)
    train_keys, cal_keys = select_population(train_keys_all, {str(s) for s, _ in manifest['parent_keys']},
        fraction=count/len(train_all)+1e-12, calibration_scenes=32 if args.mode == 'screen' else 1, seed=args.seed)
    if args.mode == 'smoke': cal_keys = cal_keys[:2]
    records, calibration = align_records(train_all, train_keys), align_records(train_all, cal_keys); del train_all
    if args.mode == 'screen' and (len(records) != 1024 or len(calibration) != 64): raise RuntimeError('TRAIN1024/calibration64 required')
    validate_records(records)
    _, dev_all = load_cache(args.dev_cache); record_keys(dev_all)
    dev_keys = tuple(tuple(k) for k in manifest['parent_keys']) if args.mode == 'screen' else tuple(dev64[:2])
    dev = align_records(dev_all, dev_keys); del dev_all
    if {s for s, _ in (*train_keys, *cal_keys)} & {s for s, _ in dev_keys}: raise RuntimeError('train/dev scene leakage')
    base = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    base_sha = validate_clean_e14_checkpoint(base, args.base_checkpoint, CLEAN_SHA256)
    if not np.isclose(base.get('yaw_weight', 19.), 19.): raise RuntimeError('reference yaw loss contract mismatch')
    motion_config = config_from_mapping_v17(base['model_config']); del base
    joint = JointCausalColumns(motion_config, ColumnConfig(z_bins=int(pcfg.grid.shape_hwd[2]))).to(device)
    control = copy.deepcopy(joint.transport).to(device)
    if any(not torch.equal(v, control.state_dict()[k]) for k, v in joint.transport.state_dict().items()):
        raise RuntimeError('paired initialization mismatch')
    provider = JointColumnProvider(args.base_checkpoint, base_sha, pcfg, device, args.cpu_workers, joint, control)
    train_source = CachedColumnSource(NuScenesWindowSource(args.dataroot, info_pkl=args.train_info, verbose=False), args.frame_cache_mib)
    dev_source = CachedColumnSource(NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False), args.frame_cache_mib)
    target = 1024 if args.mode == 'screen' else 2
    identity = {'protocol': PROTOCOL, 'training_contract': CONTRACT, 'source_link': LINK_PROTOCOL,
        'reference_checkpoint_sha256': base_sha, 'runtime_config_fingerprint': config_sha,
        'model_configs': joint.configs(), 'mode': args.mode, 'seed': args.seed, 'target_updates': target,
        'train_keys': train_keys, 'calibration_keys': cal_keys, 'dev_keys': dev_keys,
        'dev_manifest_fingerprint': manifest['manifest_fingerprint'],
        'info_fingerprints': {'train': sha256(args.train_info), 'dev': sha256(args.dev_info)},
        'patch_resolution_m': float(meta.get('patch_resolution_m', .8))}
    # A single joint optimizer; the matched control is outside the method.
    optimizer = torch.optim.AdamW([
        {'params': joint.transport.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4, 'weight_decay': 1e-4},
        {'params': joint.columns.parameters(), 'lr': 3e-4, 'initial_lr': 3e-4, 'weight_decay': .01}])
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=5e-4, weight_decay=1e-4)
    control_optimizer.param_groups[0]['initial_lr'] = 5e-4
    rng = np.random.default_rng(args.seed+1); order = np.random.default_rng(args.seed+2).permutation(len(records))
    start = 0; successful_total = 0; sampled_total = 0; link_observed = False; weights = None
    if args.resume:
        ck, restored = load_joint(args.resume, device, reference_sha=base_sha, config_sha=config_sha, allow_diagnostic=True)
        if ck['checkpoint_role'] != 'resume_last' or any(stable_json_fingerprint(ck.get(k)) != stable_json_fingerprint(v) for k, v in identity.items()):
            raise RuntimeError('resume population/model/training contract mismatch')
        joint.load_state_dict(restored.state_dict()); del restored
        control.load_state_dict(ck['control_state_dict']); optimizer.load_state_dict(ck['optimizer']); control_optimizer.load_state_dict(ck['control_optimizer'])
        rng.bit_generator.state = ck['sampling_rng_state']; torch.set_rng_state(ck['torch_rng_state'])
        if device.type == 'cuda': torch.cuda.set_rng_state_all(ck['cuda_rng_states'])
        start, weights = int(ck['executed_windows']), ck['TRAIN_weights']
        successful_total = int(ck['successful_updates'])
        sampled_total, link_observed = ck['sampled_columns'], ck['gradient_link_observed']
        if not 0 <= start <= target: raise RuntimeError('invalid update counter')
    out.mkdir(parents=True)
    write_json(out/'execution_contract.json', {**identity, 'arguments': vars(args)})
    stage_seconds = {}; audit_started = time.perf_counter()
    with (out/'progress.jsonl').open('x', encoding='utf-8') as handle:
        def progress(row):
            handle.write(json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False)+'\n'); handle.flush()
        if weights is None:
            joint.eval(); counts = {'generation': np.zeros(2, np.float64), 'refine': np.zeros(3, np.float64)}
            for wi, (record, raw) in enumerate(prefetch_raw_columns(provider, train_source, records), 1):
                tick = time.perf_counter()
                prep = provider.prepare_columns(train_source, record, include_gt=True, raw_window=raw)
                count_proposals(prep, pcfg.grid, joint.columns.config, counts)
                progress({'event': 'TRAIN_prior_audit', 'window': wi, 'windows': len(records), 'seconds': time.perf_counter()-tick})
                if wi == 1 or wi % 32 == 0 or wi == len(records): print(f'prior_audit={wi}/{len(records)} seconds={time.perf_counter()-tick:.3f}', flush=True)
            weights = weights_from_counts(counts)
            joint.columns.generation_pos_weight.fill_(weights['generation_pos_weight'])
            joint.columns.refine_class_weights.copy_(torch.tensor(weights['refine_class_weights'], device=device))
        write_json(out/'TRAIN_prior_counts.json', weights)
        stage_seconds['TRAIN_prior_audit'] = time.perf_counter()-audit_started
        def payload(update, role):
            return {**identity, 'checkpoint_role': role, 'screen_pass': False,
                'successful_updates': successful_total, 'executed_windows': update,
                'TRAIN_weights': weights, 'sampled_columns': sampled_total, 'gradient_link_observed': link_observed,
                'state_dict': {k: v.detach().cpu().clone() for k, v in joint.state_dict().items()},
                'control_state_dict': {k: v.detach().cpu().clone() for k, v in control.state_dict().items()}}
        def save_last(update):
            atomic_checkpoint(out/'last.pt', {**payload(update, 'resume_last'), 'optimizer': optimizer.state_dict(),
                'control_optimizer': control_optimizer.state_dict(), 'sampling_rng_state': rng.bit_generator.state,
                'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        save_last(start)
        scheduled = [records[int(order[(u-1) % len(records)])] for u in range(start+1, target+1)]
        training_started = previous_end = time.perf_counter(); monitor_seconds = 0.
        for update, (record, raw) in enumerate(prefetch_raw_columns(provider, train_source, scheduled), start+1):
            tick = time.perf_counter()
            input_wait = tick-previous_end
            stats = train_window(joint, control, optimizer, control_optimizer, provider, train_source, record, raw, rng,
                update, target, probe=update <= 2 or update % 16 == 0, patch_resolution=identity['patch_resolution_m'])
            sampled_total += stats['sampled_columns']; link_observed |= (stats['source_query_gradient_norm'] or 0.) > 0
            successful_total += int(stats['optimizer_updated'])
            compute_seconds = time.perf_counter()-tick
            progress({'event': 'train', 'update': update, **stats, 'seconds': compute_seconds+input_wait,
                'compute_seconds': compute_seconds, 'input_wait_seconds': input_wait})
            if update == 1 or update % 16 == 0 or update == target:
                print(f"update={update}/{target} motion={stats['motion_loss']:.6f} column={stats['column_loss']:.6f} "
                    f"control={stats['paired_control_motion_loss']:.6f} columns={stats['sampled_columns']} "
                    f"source_grad={stats['source_query_gradient_norm']} seconds={compute_seconds+input_wait:.3f} "
                    f"input_wait={input_wait:.3f}", flush=True)
            if update % 128 == 0 or update == target: save_last(update)
            # Fixed probes only; never choose best/stop/recalibrate on dev.
            if args.mode == 'screen' and update in (256, 512):
                monitor_started = time.perf_counter()
                joint.eval(); control.eval(); provider.reference_enabled = True
                report = evaluate_columns(provider, dev_source, align_records(dev, dev64), joint.columns, (.5, .5, None),
                    progress=progress, batch_size=args.eval_batch_size, diagnostic_thresholds=None)
                write_json(out/f'monitor_{update:04d}.json', report); provider.reference_enabled = False
                monitor_seconds += time.perf_counter()-monitor_started
            previous_end = time.perf_counter()
        stage_seconds['training_including_paired_control'] = time.perf_counter()-training_started-monitor_seconds
        stage_seconds['dev64_fixed_monitors'] = monitor_seconds
        joint.eval(); control.eval()
        calibration_started = time.perf_counter()
        gates, calibration_report = calibrate_columns(provider, train_source, calibration, joint.columns,
            progress=progress, batch_size=args.eval_batch_size)
        write_json(out/'TRAIN_calibration.json', calibration_report)
        stage_seconds['TRAIN64_calibration'] = time.perf_counter()-calibration_started
        frozen_candidate = {**payload(target, 'calibrated_candidate'), 'thresholds': gates, 'calibration': calibration_report}
        atomic_checkpoint(out/'candidate.pt', frozen_candidate)
        ck, persisted = load_joint(out/'candidate.pt', device, reference_sha=base_sha, config_sha=config_sha, allow_diagnostic=True)
        if any(not torch.equal(v.cpu(), persisted.state_dict()[k].cpu()) for k, v in joint.state_dict().items()):
            raise RuntimeError('joint checkpoint serialization changed weights')
        provider.joint, provider.model = persisted, persisted.transport; provider.reference_enabled = True
        evaluation_started = time.perf_counter()
        evaluation = evaluate_columns(provider, dev_source, dev, persisted.columns, tuple(gates), progress=progress,
            batch_size=args.eval_batch_size, dev64_keys=dev64 if args.mode == 'screen' else None)
        stage_seconds['final_dev512_shared'] = time.perf_counter()-evaluation_started
        checks = {}
        for population in ('dev64', 'all'):
            if population not in evaluation: continue
            row = evaluation[population]; joint_metrics = row['variants']['joint']['metrics']
            checks[population] = {'column_branches_gate': row['gate']['pass'], 'gradient_link_observed': link_observed,
                'joint_nonnegative_vs_frozen_E14': safe_metrics(joint_metrics, row['reference_metrics']['frozen_E14']),
                'joint_nonnegative_vs_paired_control': safe_metrics(joint_metrics, row['reference_metrics']['paired_scratch_V18_only'])}
            checks[population]['pass'] = all(checks[population].values())
        passed = args.mode == 'screen' and all(v['pass'] for v in checks.values())
        ck['screen_pass'] = passed; atomic_checkpoint(out/'candidate.pt', ck)
        try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError): commit = 'unavailable'
        summary = {**identity, 'git_commit': commit, 'successful_updates': successful_total, 'executed_windows': target,
            'skipped_empty_windows': target-successful_total, 'train_windows': len(records),
            'training_window_passes': target/len(records), 'sampled_columns': sampled_total, 'gradient_link_observed': link_observed,
            'thresholds': gates, 'evaluation': evaluation, 'gates': checks, 'screen_pass': passed,
            'frame_cache': {'limit_mib': args.frame_cache_mib, 'TRAIN_hits': train_source.hits, 'dev_hits': dev_source.hits},
            'elapsed_seconds': time.perf_counter()-started, 'stage_seconds': stage_seconds,
            'candidate_checkpoint': str((out/'candidate.pt').resolve()),
            'checkpoint_sha256': sha256(out/'candidate.pt'), 'route': 'smoke_only_not_effectiveness_evidence' if args.mode == 'smoke' else
                'joint_screen_passed_requires_explicit_next_step' if passed else 'joint_screen_not_passed_no_automatic_retry'}
        write_json(out/'summary.json', summary); (out/'summary.txt').write_text(summary_text(summary), encoding='utf-8')
        print(summary_text(summary), flush=True)


if __name__ == '__main__': main()
