#!/usr/bin/env python3
"""Explicit dev512 threshold tuning; immutable fixed checkpoint, no training."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import time
import torch

from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.joint_causal_columns import FULL_PROTOCOLS
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.eval_p0_f9_joint_causal_columns import evaluation_keys
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256
from tools.real_motion.joint_column_full_common import EvaluationJointColumnProvider
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.joint_threshold_sweep import sweep, LEVELS, FIXED, PROTOCOL
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP

CALIBRATION_WINDOWS = 512


def summary_text(result):
    report = result['report']; selected = report['selected']; fixed = report['fixed']; rows = report['candidates']
    lines = ['===== DEV512 THRESHOLD CALIBRATION (NOT INDEPENDENT TEST) =====',
        f"epoch={result['epoch']} update={result['update']} windows={report['windows']} scenes={report['scenes']}",
        'ONE probability pass; 64 joint settings + 20 branch settings; no training/checkpoint edits.',
        'grid=(0.5,0.75,0.95,off); order=generation_ADD/refine_ADD/refine_REMOVE',
        f"fixed={list(FIXED)} mIoU={fixed['metrics']['mIoU']:.6f} MovingMicro={fixed['metrics']['MovingMicro']:.6f}",
        '===== TOP 10 BY mIoU (guard failures are diagnostic only) =====']
    for row in rows[:10]:
        d = row['delta_vs_fixed_pp']; m = row['metrics']
        lines.append(f"{row['thresholds']} mIoU={m['mIoU']:.6f} dMiOU={d['mIoU']:+.6f} "
            f"dMoving={d['MovingMicro']:+.6f} safe={row['eligible']} "
            f"add={row['quality'].get('added',0)} remove={row['quality'].get('removed',0)}")
    lines += ['===== GUARDED SELECTION =====', 'selected_thresholds='+str(selected['thresholds']),
        'selection_rule='+report['rule'], 'route='+report['route'],
        f"selected_mIoU={selected['metrics']['mIoU']:.6f} dMiOU_vs_fixed={selected['delta_vs_fixed_pp']['mIoU']:+.6f}",
        f"selected_MovingMicro={selected['metrics']['MovingMicro']:.6f} dMoving_vs_fixed={selected['delta_vs_fixed_pp']['MovingMicro']:+.6f}"]
    for h, delta in selected['delta_vs_fixed_pp']['per_horizon'].items():
        lines.append(f"{h}s dMiOU={delta['mIoU']:+.6f} dMoving={delta['MovingMicro']:+.6f}")
    lines += ['selected_quality='+str(selected['quality']),
        'selected_scenes='+str({k:v for k,v in selected['scene_delta_vs_fixed'].items() if k != 'by_scene'}),
        'performance='+json.dumps(result['performance'], ensure_ascii=False),
        'Checkpoint already selected on dev512; these are tuning scores, not independent generalization evidence.',
        'The concurrently running fixed-threshold full evaluation is NOT modified or reinterpreted. No automatic full re-evaluation/promotion.']
    return '\n'.join(lines)+'\n'


def main(stop_event=None):
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for key in ('checkpoint', 'dev-cache', 'population-manifest', 'base-checkpoint', 'dataroot', 'dev-info', 'out-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-workers', type=int, default=8)
    p.add_argument('--resume', action='store_true'); p.add_argument('--checkpoint-every', type=int, default=8)
    a = p.parse_args(); out = Path(a.out_dir).resolve()
    for key in ('config', 'checkpoint', 'dev_cache', 'population_manifest', 'base_checkpoint', 'dev_info'):
        if not Path(getattr(a, key)).is_file(): p.error('missing input: '+key)
    if not Path(a.dataroot).is_dir() or not 1 <= a.cpu_workers <= 16 or a.checkpoint_every < 1: p.error('invalid paths/budgets')
    if a.resume:
        if not (out/'threshold_state.json').is_file(): p.error('resume needs existing threshold_state.json')
        if (out/'calibration.json').is_file(): p.error('calibration already complete')
    else:
        if out.exists(): p.error('NEW output required; never overwrite full evaluation or checkpoints')
        out.mkdir(parents=True)
    with evaluation_lock(out): return execute(a, out, stop_event)


def execute(a, out, stop_event):
    torch.set_num_threads(1); device = torch.device(a.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): raise RuntimeError('CUDA/BF16 required')
    from real_motion.native_column_cpu import backend_name, prepare_native
    if backend_name() == 'native': prepare_native()
    snapshot = out/'checkpoint_snapshot.pt'
    digest = sha256(snapshot) if a.resume else snapshot_checkpoint(a.checkpoint, snapshot)
    if sha256(a.checkpoint) != digest: raise RuntimeError('source checkpoint/snapshot changed')
    cfg = load_runtime_config(a.config, a.override); config_fp = stable_json_fingerprint(cfg)
    ck, joint = load_joint(snapshot, device, reference_sha=CLEAN_SHA256, config_sha=config_fp, allow_diagnostic=True)
    if (ck['protocol'] not in FULL_PROTOCOLS or ck['model_configs']['motion']['history_frames'] != 4
            or not ck.get('prior_completed', True) or ck['cursor_batch'] != 0):
        raise RuntimeError('complete strict-four-history Local checkpoint required')
    if sha256(a.base_checkpoint) != CLEAN_SHA256: raise RuntimeError('frozen E14 changed')
    if sha256(a.dev_info) != ck['info_fingerprints']['dev'] or sha256(a.dev_cache) != ck['cache_fingerprints']['dev']:
        raise RuntimeError('dev info/cache provenance differs from training')
    manifest, dev64, _ = load_manifest(a.population_manifest)
    if (manifest['selected_key_fingerprint'] != DEV64_FP or manifest['manifest_fingerprint'] != ck['dev_manifest_fingerprint']
            or tuple(map(tuple, manifest['parent_keys'])) != tuple(map(tuple, ck['dev_keys']))):
        raise RuntimeError('frozen selection population changed')
    _, all_records = load_cache(a.dev_cache); full_keys = record_keys(all_records)
    chosen = evaluation_keys('dev512', dev64, manifest['parent_keys'], full_keys, ck['train_keys'])
    if len(chosen) != CALIBRATION_WINDOWS or len(set(chosen)) != CALIBRATION_WINDOWS:
        raise RuntimeError('512 unique frozen calibration windows required')
    records = align_records(all_records, chosen); del all_records
    execution = dict(protocol=PROTOCOL, checkpoint_sha256=digest, runtime_config_fingerprint=config_fp,
        cache_fingerprint=ck['cache_fingerprints']['dev'], info_fingerprint=ck['info_fingerprints']['dev'],
        manifest_fingerprint=manifest['manifest_fingerprint'], population_key_fingerprint=stable_json_fingerprint(chosen),
        population='dev512_ALREADY_used_for_checkpoint_selection', levels=list(LEVELS), fixed_thresholds=list(FIXED),
        inference_batch=256, feature_backend='cpu', cpu_workers=a.cpu_workers,
        safety='aggregate_and_report_horizon_MovingMicro_nonnegative_vs_fixed')
    saved = None; cursor = 0; elapsed_prefix = 0.
    if a.resume:
        saved = json.loads((out/'threshold_state.json').read_text(encoding='utf-8')); fingerprint = saved.pop('fingerprint')
        if stable_json_fingerprint(saved) != fingerprint or saved['execution'] != execution: raise RuntimeError('threshold resume provenance/state changed')
        cursor = saved['completed_windows']; elapsed_prefix = saved['elapsed_seconds']
        if type(cursor) is not int or not 0 <= cursor <= len(records): raise RuntimeError('invalid threshold resume cursor')
    provider = EvaluationJointColumnProvider(a.base_checkpoint, CLEAN_SHA256, make_prepare_config(cfg), device, a.cpu_workers, joint, None)
    provider.reference_enabled = True
    source = CachedColumnSource(NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info, verbose=False), 256)
    epoch, update = ck['cursor_epoch'], ck['attempted_updates']; del ck
    tick = time.perf_counter()
    def save(wi, state, totals):
        payload = dict(execution=execution, completed_windows=wi, state=state,
            elapsed_seconds=elapsed_prefix+time.perf_counter()-tick)
        payload['fingerprint'] = stable_json_fingerprint(payload); write_json(out/'threshold_state.json', payload)
    with (out/'progress.jsonl').open('a' if a.resume else 'x', encoding='utf-8') as log:
        def progress(row):
            log.write(json.dumps(finite_json(row), ensure_ascii=False)+'\n'); log.flush()
            if row['window'] == 1 or row['window'] % 8 == 0 or row['window'] == row['windows']:
                print(f"threshold_calibration={row['window']}/{row['windows']} seconds={row['seconds']:.3f}", flush=True)
        try:
            report, performance = sweep(provider, source, records, joint.columns, progress=progress, stop_event=stop_event,
                start_window=cursor, saved_state=saved['state'] if saved else None, save_state=save,
                checkpoint_every=a.checkpoint_every)
        except InterruptedError:
            write_json(out/'threshold_status.json', dict(status='interrupted', source_checkpoint_unchanged=True))
            print('Threshold calibration stopped safely; same command + --resume. Full evaluation unchanged.', flush=True); return 130
    if sha256(snapshot) != digest or sha256(a.checkpoint) != digest: raise RuntimeError('immutable calibration checkpoint changed')
    if report['windows'] != len(records): raise RuntimeError('incomplete calibration population')
    performance.update(seconds_this_invocation=time.perf_counter()-tick, accumulated_seconds=elapsed_prefix+time.perf_counter()-tick,
        reused_prefix_windows=cursor, actual_cuda=device.type == 'cuda',
        timing_scope='stage seconds and probability horizons refer to this invocation only')
    result = finite_json(dict(status='complete', execution=execution, epoch=epoch, update=update, report=report, performance=performance))
    result['artifact_fingerprint'] = stable_json_fingerprint(result)
    write_json(out/'calibration.json', result); (out/'summary.txt').write_text(summary_text(result), encoding='utf-8')
    write_json(out/'threshold_status.json', dict(status='complete', no_deployment=True))
    print(summary_text(result), flush=True); return 0


if __name__ == '__main__':
    import signal
    from threading import Event
    event = Event()
    signal.signal(signal.SIGINT, lambda *_: event.set()); signal.signal(signal.SIGTERM, lambda *_: event.set())
    sys.exit(main(event))
