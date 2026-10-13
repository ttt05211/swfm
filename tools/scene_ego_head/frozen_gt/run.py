"""Completed GT body -> frozen historical scene -> ego-head-only TRAIN -> internal DEV512."""
import itertools
import json
from pathlib import Path
import signal
import sys
import threading
import time
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt import run as gt_run, engine as gt_engine
from tools.shared_history_experiments.refiner_batch_readout.probe import validate_source
from tools.scene_ego_head.model import SceneEgoConfig, SceneEgoHead
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from . import bank, training, engine, reuse


def validate_completed_source(saved):
    c = validate_source(saved)
    if saved['epoch'] != 2 or saved.get('batch') != 0 or saved['updates'] != c['total_updates']:
        raise RuntimeError('GT body must FINISH epoch2 first; no interim weights accepted')
    return c


def summarize(result, head_result=None):
    lines = ['===== TWO-STAGE FROZEN GT / SCENE EGO HEAD =====', 'status='+result['status'],
        f"source_GT_epoch={result.get('epoch')}; source_update={result.get('updates')}",
        'Encoder/STWM/Refiner frozen; ONLY new scene ego head trained with R10 supervision.',
        'Two-stage training, NOT independent RANDOM Pred joint training. GT pretraining cost must be counted.',
        'GT-derived navigation-conditioned; NOT navigation-free; future GT poses are labels ONLY for the head.',
        'Predicted ego drives ALL inference geometry/background/memory. No external planner or GT reference score.']
    if head_result:
        lines += [f"head_epochs={head_result['complete_epochs']}; head_updates={head_result['updates']}; "
            f"TRAIN_windows={head_result['train_windows']}", 'TRAIN IN-SAMPLE: route ADE / FDE3s / yaw3s']
        for n, r in head_result['report'].items():
            lines.append(f"{n} {r['ADE_m']:.6f} {r['FDE_3s_m']:.6f} {r['yaw_mean_deg'][-1]:.6f}")
    r = result.get('evaluation')
    if r:
        lines += [f"{result.get('actual_eval_population', 'DEV512')} windows={r['windows']}; no planner coverage trimming", 'route mIoU IoU MovingMicro']
        for name, m in r['variants']['internal'].items():
            lines.append(f"internal_{name} {m['mIoU']:.6f} {m['IoU']:.6f} {m['MovingMicro']:.6f}")
            for h, v in m['per_horizon'].items():
                lines.append(f"  {h}s mIoU={v['mIoU']:.6f} IoU={v['IoU']:.6f} MovingMicro={v['MovingMicro']:.6f}")
        lines.append('DEV trajectory head/prior='+json.dumps(r['trajectory']))
    if result.get('error'): lines.append('error='+result['error'])
    if result.get('stage_seconds'): lines.append('stage_seconds='+json.dumps(result['stage_seconds']))
    lines.append('Original GT checkpoints/trainer/caches unchanged. Fixed final head epoch; no automatic promotion or retry.')
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event(); p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=1., eval_population='dev512')
    p.add_argument('--head-epochs', type=int, choices=range(1, 11), default=3)
    p.add_argument('--head-batch', type=int, default=64)
    p.add_argument('--head-seed', type=int, default=21)
    p.add_argument('--head-lr', type=float, default=3e-4)
    p.add_argument('--head-pool-side', type=int, default=8)
    p.add_argument('--resume-stage', action='store_true')
    p.add_argument('--reuse-stage-dir', help='completed legacy stage; import only history tokens READ ONLY')
    p.add_argument('--final-population', choices=('dev512', 'all'), default='dev512')
    a = p.parse_args(argv)
    if (not a.resume or a.max_updates or a.evaluate_only or a.mode != 'gt' or a.epochs != 2 or a.train_fraction != 1.
            or a.eval_population != 'dev512' or a.window_batch_size != 4 or a.source_budget != 128
            or not 1 <= a.head_batch <= 128 or not 0 < a.head_lr <= .01
            or not 1 <= a.head_pool_side <= 16 or not 0 <= a.head_seed < 2**31):
        raise ValueError('completed full-GT source / bounded head-only settings required')
    origin, out = Path(a.resume).resolve(), Path(a.out_dir).resolve()
    if out == origin.parent or origin.parent in out.parents: raise ValueError('NEW stage directory OUTSIDE GT train')
    if a.reuse_stage_dir:
        reused = Path(a.reuse_stage_dir).resolve()
        if out == reused or reused in out.parents: raise ValueError('NEW stage OUTSIDE reused history stage')
    if not a.resume_stage:
        validate_completed_source(torch.load(origin, map_location='cpu', weights_only=False))
    lock = out.parent/'.frozen_scene_ego_locks'/legacy.stable_json_fingerprint(str(out))
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume_stage:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(origin, out/'frozen/body.pt')
        frozen = out/'frozen/body.pt'; saved = torch.load(frozen, map_location='cpu', weights_only=False)
        source_contract = validate_completed_source(saved)
        fields = vars(a).copy()
        for n in ('resume_stage', 'evaluate_only'): fields.pop(n)
        root = Path(__file__).resolve().parents[3]
        dependencies = list(Path(__file__).parent.glob('*.py')) + [root/f for f in (
            'tools/scene_ego_head/model.py', 'tools/scene_ego_head/integration.py',
            'real_motion/ego_navigation.py', 'tools/ego_experiments/ego_kinematic.py',
            'tools/ego_experiments/train_surface_ego_full.py')]
        identity = dict(protocol=training.PROTOCOL, args=fields, source_sha256=legacy.sha256(frozen),
            implementation={str(f.relative_to(root)): legacy.sha256(f) for f in dependencies})
        contract_path = out/'stage_contract.json'
        if a.resume_stage and json.loads(contract_path.read_text()) != identity:
            raise RuntimeError('frozen head stage source/settings/implementation changed')
        legacy.write_json(contract_path, identity)
        print(f'FROZEN_BODY {origin}; GT epoch2 update={saved["updates"]}; body will NEVER be optimized', flush=True)
        names = ('load_cache', 'prefetched_batches', 'summarize', 'save_resume_checkpoint')
        original = {n: getattr(legacy, n) for n in names}
        original_eval, original_train, original_inputs = gt_engine.evaluate, gt_engine.train_arm, gt_engine.history_inputs
        captured = []; val_records = []; holder = {}
        def load(path):
            value = original['load_cache'](path)
            if Path(path).resolve() == Path(a.train_cache).resolve(): captured[:] = [value[1]]
            if Path(path).resolve() == Path(a.dev_cache).resolve(): val_records[:] = [value[1]]
            return value
        def batches(provider, source, groups, descriptors, **kwargs):
            if a.final_population == 'all':
                if not val_records: raise RuntimeError('full VAL cache missing')
                keys = legacy.record_keys(val_records[0])
                if len(keys) != 4369: raise RuntimeError('complete VAL4369 required; no planner trimming')
                if set(k[0] for k in keys) & set(k[0] for k in source_contract['train_keys']):
                    raise RuntimeError('TRAIN/VAL scenes overlap')
                groups = [[r] for r in val_records[0]]
                holder['val_keys'] = [list(k) for k in keys]
                legacy.write_json(out/'population.json', dict(population='all_val', keys=holder['val_keys'],
                    windows=len(keys), scenes=len(set(k[0] for k in keys)), planner_trimmed=False))
                print(f'INTERNAL_ALL_VAL windows={len(keys)}; no planner trimming', flush=True)
            return original['prefetched_batches'](provider, source, groups, descriptors, **kwargs)
        def forbidden(*args, **kwargs): raise RuntimeError('frozen body cannot train/save body checkpoints')
        def evaluate(models, provider, dev_source, rows, **kwargs):
            body = models['gt']; body.eval().requires_grad_(False)
            before = tensor_state_fingerprint(body.state_dict())
            torch.manual_seed(a.head_seed)
            # Explicit independent initializer, not source RNG's next draw.
            head = SceneEgoHead(SceneEgoConfig(feature_width=body.transport.history_encoder.width,
                pool_side=a.head_pool_side)).to(provider.device)
            contract = dict(stage=identity, head=head.specification(), source_train_keys=source_contract['train_keys'],
                source_runtime=source_contract['execution'], schedule=dict(epochs=a.head_epochs, batch_size=a.head_batch,
                    seed=a.head_seed, lr=a.head_lr, floor=.01, supervision='fixed_R10_geometry_ONLY'),
                body_state_sha256=before, encoder_precision='BF16' if provider.device.type == 'cuda' else 'FP32',
                head_precision='FP32', features='pooled pure history F; no future-conditioned memory',
                bank_chunk=64, final_epoch_only=True)
            iterator = iter(rows); first = next(iterator, None)
            if first is None: raise RuntimeError('empty DEV population')
            try:
                audit_path = out/'same_pose_entrance_audit_v2.json'
                if audit_path.exists(): audit = json.loads(audit_path.read_text())
                else:
                    audit = engine.entrance_audit(body, provider, first[0]); legacy.write_json(audit_path, audit)
                if not audit['pass_gate']: raise RuntimeError('same-pose entrance mismatch; do not train ego head yet')
                if not captured: raise RuntimeError('TRAIN metadata missing')
                records = legacy.align_records(captured[0], [tuple(k) for k in source_contract['train_keys']])
                tick = time.perf_counter()
                if a.reuse_stage_dir:
                    if Path(a.reuse_stage_dir).resolve() == out: raise ValueError('reused stage must be separate')
                    features, inventory = reuse.load_readonly(a.reuse_stage_dir, contract, stop_event=event)
                    legacy.write_json(out/'history_reuse_receipt.json', inventory)
                    print(f'REUSED_HISTORY windows={inventory["windows"]}; fresh=0; old head/Adam NOT imported', flush=True)
                else:
                    features, inventory = bank.build(body, records, provider.ccr_history_cache_source, provider.pcfg.grid,
                        out/'history_tokens', contract, pool_side=a.head_pool_side, workers=a.prefetch_workers, stop_event=event)
                holder['stage_seconds'] = dict(history_tokens_build_or_reuse=time.perf_counter()-tick)
                print(f'COMPACT_HISTORY_BANK {inventory["tensor_mib"]:.1f}MiB; source-bound frozen features', flush=True)
                tick = time.perf_counter()
                with torch.enable_grad():
                    head_result = training.train(head, features, out, contract, stop_event=event, resume=a.resume_stage)
                holder['stage_seconds']['head_training_and_TRAIN_report'] = time.perf_counter()-tick
                holder['training'] = head_result; del features, records
                report = engine.evaluate(body, head, provider, dev_source, itertools.chain([first], iterator), **kwargs)
                expected = 4369 if a.final_population == 'all' else 512
                if report['windows'] != expected: raise RuntimeError('exact complete evaluation population required')
                if a.final_population == 'all' and report['population_keys'] != holder['val_keys']:
                    raise RuntimeError('full VAL keys/order differ')
                return report
            finally:
                if tensor_state_fingerprint(body.state_dict()) != before:
                    raise RuntimeError('frozen body weights changed')
                legacy.write_json(out/'body_freeze_audit.json', dict(body_state_sha256=before,
                    unchanged=True, body_gradient_parameters=sum(p.grad is not None for p in body.parameters()),
                    trainable_body_parameters=sum(p.requires_grad for p in body.parameters())))
        inner = vars(a).copy()
        for k in tuple(inner):
            if k.startswith('head_') or k in ('resume_stage', 'reuse_stage_dir', 'final_population'): inner.pop(k)
        inner.update(resume=str(frozen), evaluate_only=True)
        args = []
        for k, v in inner.items():
            if k == 'override' or v is None: continue
            if isinstance(v, bool):
                if v: args.append('--'+k.replace('_', '-'))
            else: args += ['--'+k.replace('_', '-'), str(v)]
        for v in a.override or []: args += ['--override', v]
        try:
            legacy.load_cache, legacy.prefetched_batches, legacy.save_resume_checkpoint = load, batches, forbidden
            legacy.summarize = lambda r: summarize(r, holder.get('training'))
            gt_engine.evaluate, gt_engine.train_arm, gt_engine.history_inputs = evaluate, forbidden, engine.metadata
            status = gt_run.main(event, args)
        finally:
            for k, v in original.items(): setattr(legacy, k, v)
            gt_engine.evaluate, gt_engine.train_arm, gt_engine.history_inputs = original_eval, original_train, original_inputs
            captured.clear(); val_records.clear()
        if legacy.sha256(frozen) != identity['source_sha256']: raise RuntimeError('source snapshot changed')
        result_path = out/'results.json'; result = json.loads(result_path.read_text())
        if 'performance' in result: result['source_GT_training_performance'] = result.pop('performance')
        result.update(training_protocol=training.PROTOCOL, head_training=holder.get('training'),
                      frozen_body_source=str(origin), two_stage_not_joint_RANDOM=True,
                      stage_seconds=holder.get('stage_seconds', {}))
        result['actual_eval_population'] = 'ALL_VAL4369' if a.final_population == 'all' else 'DEV512'
        if result.get('evaluation'): result['status'] = 'complete'
        legacy.write_json(result_path, result)
        text = summarize(result, holder.get('training')); (out/'summary.txt').write_text(text, encoding='utf-8')
        print(text, flush=True); return status


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
