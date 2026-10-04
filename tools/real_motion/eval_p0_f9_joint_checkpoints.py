#!/usr/bin/env python3
"""Audit all ancestral epochs; optional resumable one-pass dev512 comparison."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import copy
from contextlib import contextmanager
import json
import os
import time
import torch

from real_motion.joint_causal_columns import FULL_PROTOCOLS
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.joint_checkpoint_selection import audit_runs, audit_text, identity, weight_fingerprint
from tools.real_motion.joint_checkpoint_evaluation import evaluate_group, rank_reports
from tools.real_motion.manage_p0_f9_joint_training import model_directory, status
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.joint_column_full_common import EvaluationJointColumnProvider
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.eval_p0_f9_joint_causal_columns import evaluation_keys
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256, align_records, load_manifest
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP


@contextmanager
def evaluation_lock(out):
    """Kernel-owned lease: released even on SIGKILL; never unlink a lock inode."""
    with (out/'evaluation.lock').open('a+b') as handle:
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('another evaluator owns this output; do not concurrently resume it') from error
        try:
            if os.fstat(handle.fileno()).st_size == 0: handle.write(b'0'); handle.flush()
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt': msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def batch_text(result):
    lines = ['===== SHARED LOCAL CHECKPOINT EVALUATION =====',
        f"population={result['population']}; windows={result['windows']}; thresholds=(0.5,0.5,REMOVE-off)",
        'epoch     mIoU      IoU MovingMicro dMoving_vs_E14',
        'Selection population, not independent test; no automatic promotion.']
    for r in result['ranking']['by_mIoU']:
        dm = r['delta_MovingMicro_vs_E14']
        lines.append(f"{r['checkpoint']:>10} {r['mIoU']:9.6f} {r['IoU']:9.6f} {r['MovingMicro']:11.6f} "
            +('NA' if dm is None else f'{dm:+.6f}'))
    lines += ['moving_safe_best='+str(result['ranking']['moving_safe_best']),
        'overall_mIoU_best='+result['ranking']['overall_mIoU_best'],
        'moving_safe_snapshot='+str(result['checkpoints'].get(result['ranking']['moving_safe_best'], {}).get('snapshot')),
        'overall_mIoU_snapshot='+result['checkpoints'][result['ranking']['overall_mIoU_best']]['snapshot'],
        'performance='+json.dumps(result['performance'], ensure_ascii=False),
        'Original checkpoints/optimizer/RNG are unchanged; directory cleanup is NOT performed.']
    return '\n'.join(lines)+'\n'


def main(stop_event=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True); p.add_argument('--runs-root', required=True)
    p.add_argument('--out-dir', required=True); p.add_argument('--shortlist-size', type=int, default=8)
    p.add_argument('--audit-only', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--population', choices=('dev64', 'dev512'), default='dev512')
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-workers', type=int, default=8)
    p.add_argument('--checkpoint-every', type=int, default=8)
    p.add_argument('--reference-inference', action='store_true', help='disable new CPU prefetch/deferred checks')
    a = p.parse_args(); out = Path(a.out_dir).resolve(); directory = model_directory(a.run_dir)
    if not 4 <= a.shortlist_size <= 12 or not 1 <= a.cpu_workers <= 16 or a.checkpoint_every < 1:
        p.error('shortlist=4..12 and positive bounded worker/checkpoint budgets required')
    if (directory/'runtime_status.json').is_file() and status(directory)['matching_trainer_running']:
        p.error('wait for completed training or safely stop before this GPU evaluation')
    if a.resume:
        if a.audit_only or not (out/'evaluation_state.json').is_file(): p.error('resume requires existing evaluation state')
        if (out/'comparison.json').is_file(): p.error('comparison already complete; no re-evaluation needed')
        audit = json.loads((out/'selection.json').read_text(encoding='utf-8'))
        if audit['run_directory'] != str(Path(a.run_dir).resolve()): p.error('resume run directory changed')
    else:
        if out.exists(): p.error('NEW output directory required; use --resume explicitly')
        audit = audit_runs(a.run_dir, a.runs_root, a.shortlist_size)
        out.mkdir(parents=True); write_json(out/'selection.json', audit)
        (out/'selection.txt').write_text(audit_text(audit), encoding='utf-8')
        print(audit_text(audit), flush=True)
    check = dict(audit); fingerprint = check.pop('selection_fingerprint')
    if stable_json_fingerprint(check) != fingerprint: raise RuntimeError('selection artifact fingerprint mismatch')
    if a.audit_only: return 0
    with evaluation_lock(out):
        return evaluate(a, out, directory, audit, fingerprint, stop_event)


def evaluate(a, out, directory, audit, fingerprint, stop_event):
    torch.set_num_threads(1); device = torch.device(a.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('CUDA/BF16 required')
    from real_motion.native_column_cpu import backend_name, prepare_native
    if backend_name() == 'native': prepare_native()
    contract = json.loads((directory/'execution_contract.json').read_text(encoding='utf-8'))
    if identity(contract) != audit['scientific_identity']: raise RuntimeError('anchor scientific identity changed')
    args = dict(contract['arguments']); cwd = Path(contract.get('launch_cwd', Path.cwd()))
    for key in ('config', 'dev_cache', 'dev_info', 'base_checkpoint', 'dataroot', 'population_manifest'):
        args[key] = str((cwd/Path(args[key])).resolve())
    cfg = load_runtime_config(args['config'], args.get('override', [])); pcfg = make_prepare_config(cfg)
    config_sha = stable_json_fingerprint(cfg)
    if sha256(args['base_checkpoint']) != CLEAN_SHA256: raise RuntimeError('reference E14 changed')
    for key, expected in (('dev_info', contract['info_fingerprints']['dev']), ('dev_cache', contract['cache_fingerprints']['dev'])):
        if sha256(args[key]) != expected: raise RuntimeError('evaluation provenance changed: '+key)
    manifest, dev64, _ = load_manifest(args['population_manifest'])
    if (manifest['selected_key_fingerprint'] != DEV64_FP or manifest['manifest_fingerprint'] != contract['dev_manifest_fingerprint']
            or tuple(map(tuple, manifest['parent_keys'])) != tuple(map(tuple, contract['dev_keys']))):
        raise RuntimeError('frozen dev population identity/order changed')
    _, records = load_cache(args['dev_cache']); keys = record_keys(records)
    chosen = evaluation_keys(a.population, dev64, manifest['parent_keys'], keys, contract['train_keys'])
    records = align_records(records, chosen)
    jobs = {}; snapshots = {}; first = None
    for row in audit['selected']:
        name = f"epoch_{row['epoch']:04d}"; selected = row['checkpoint']; snapshot = out/(name+'.pt')
        if not a.resume:
            digest = snapshot_checkpoint(selected['path'], snapshot)
        else: digest = sha256(snapshot)
        if digest != selected['sha256']: raise RuntimeError('selected checkpoint changed: '+name)
        ck, joint = load_joint(snapshot, device, reference_sha=CLEAN_SHA256, config_sha=config_sha, allow_diagnostic=True)
        if (ck['protocol'] not in FULL_PROTOCOLS or identity(ck) != audit['scientific_identity']
                or ck['cursor_epoch'] != row['epoch'] or ck['cursor_batch'] != 0
                or ck['attempted_updates'] != row['update'] or weight_fingerprint(ck['state_dict']) != selected['weight_fingerprint']):
            raise RuntimeError('selected checkpoint identity/weights changed: '+name)
        if first is None:
            provider = EvaluationJointColumnProvider(args['base_checkpoint'], CLEAN_SHA256, pcfg, device, a.cpu_workers, joint, None)
            first = provider
        else:
            provider = copy.copy(first); provider.joint, provider.model = joint, joint.transport
            provider.columns_checked = False
        provider.reference_enabled = True; jobs[name] = (provider, joint.columns)
        snapshots[name] = digest
        del ck
    execution = dict(selection_fingerprint=fingerprint, snapshots=snapshots, population=a.population,
        population_key_fingerprint=stable_json_fingerprint(chosen), thresholds=[.5, .5, None],
        inference_batch=256, feature_backend='cpu', optimized_inference=not a.reference_inference,
        metric_protocol=cfg.get('METRIC'), cpu_workers=a.cpu_workers)
    saved = None; cursor = 0; accumulated_seconds = 0.
    if a.resume:
        saved = json.loads((out/'evaluation_state.json').read_text(encoding='utf-8'))
        checksum = saved.pop('fingerprint')
        if stable_json_fingerprint(saved) != checksum or saved['execution'] != execution:
            raise RuntimeError('evaluation resume identity/state changed')
        cursor = saved['completed_windows']; accumulated_seconds = saved['elapsed_seconds']
        if type(cursor) is not int or not 0 <= cursor <= len(records): raise RuntimeError('invalid evaluation cursor')
        for state in saved['states'].values():
            if state['counts_windows'].get('all', 0) != cursor: raise RuntimeError('inconsistent all-model saved cursor')
    tick = time.perf_counter()
    def save(wi, states, totals):
        payload = dict(execution=execution, completed_windows=wi, states=states,
            elapsed_seconds=accumulated_seconds+time.perf_counter()-tick)
        payload['fingerprint'] = stable_json_fingerprint(payload)
        write_json(out/'evaluation_state.json', payload)
    with (out/'progress.jsonl').open('a' if a.resume else 'x', encoding='utf-8') as log:
        def progress(row):
            log.write(json.dumps(finite_json(row), ensure_ascii=False)+'\n'); log.flush()
            if row['event'] == 'group_window_complete':
                print(f"shared_eval={row['window']}/{row['windows']} checkpoints={row['models']} all-model boundary", flush=True)
        try:
            source = CachedColumnSource(NuScenesWindowSource(args['dataroot'], info_pkl=args['dev_info'], verbose=False), 256)
            reports, performance = evaluate_group(jobs, source, records, progress=progress, stop_event=stop_event,
                start_window=cursor, saved_states=saved['states'] if saved else None, save_state=save,
                checkpoint_every=a.checkpoint_every, optimized=not a.reference_inference,
                dev64_keys=dev64 if a.population != 'dev64' else None)
        except InterruptedError:
            write_json(out/'evaluation_status.json', dict(status='interrupted', resume_command='same command plus --resume'))
            print('Evaluation interrupted; resume saved all-model boundary with --resume. Training files unchanged.', flush=True)
            return 130
    performance.update(elapsed_seconds_this_invocation=time.perf_counter()-tick,
        accumulated_evaluation_seconds=accumulated_seconds+time.perf_counter()-tick,
        reused_prefix_windows=cursor, no_persistent_geometry_writes=True, actual_cuda=device.type == 'cuda')
    for name, digest in snapshots.items():
        if sha256(out/(name+'.pt')) != digest: raise RuntimeError('immutable evaluation snapshot changed: '+name)
    result = dict(status='complete', population=a.population, windows=len(records), execution=execution,
        reports=reports, ranking=rank_reports(reports), performance=performance,
        checkpoints={f"epoch_{row['epoch']:04d}": dict(epoch=row['epoch'], update=row['update'],
            source=row['checkpoint']['path'], snapshot=str(out/f"epoch_{row['epoch']:04d}.pt"))
            for row in audit['selected']})
    write_json(out/'comparison.json', result); (out/'summary.txt').write_text(batch_text(result), encoding='utf-8')
    write_json(out/'evaluation_status.json', dict(status='complete'))
    print(batch_text(result), flush=True); return 0


if __name__ == '__main__':
    import signal
    from threading import Event
    event = Event()
    signal.signal(signal.SIGINT, lambda *_: event.set()); signal.signal(signal.SIGTERM, lambda *_: event.set())
    sys.exit(main(event))
