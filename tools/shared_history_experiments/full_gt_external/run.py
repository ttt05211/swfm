"""Frozen full-GT epoch checkpoint, ALL planner-covered VAL, external route only."""
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt import run as training, engine as gt_engine
from tools.shared_history_experiments.refiner_batch_readout.probe import validate_source
from tools.scene_ego_head.experiment.planner import PlannerCache
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from . import engine


def summarize(result):
    lines = ['===== FROZEN FULL GT / ALL VAL / EXTERNAL PLANNER ONLY =====',
        'status='+result['status'], f"epoch={result.get('epoch')}; update={result.get('updates')}; scientific_updates=0",
        'GT-trained body; external BEV-Planner+yaw drives ALL future geometry; no internal ego/GT reference run.',
        'No training, alignment, masks, threshold tuning or backend switch. Moving not evaluated.']
    r = result.get('evaluation')
    if r:
        lines += [f"VAL requested={r['population']['requested']}; covered={r['windows']}; "
            f"missing={len(r['population']['missing_keys'])}; scenes={r['scenes']}",
            'route                    horizon       IoU      mIoU']
        for name, m in r['variants']['external'].items():
            for h, v in m['per_horizon'].items():
                lines.append(f"external_{name:16s} {h:>4s}s {v['IoU']:9.6f} {v['mIoU']:9.6f}")
            lines.append(f"external_{name:16s}   Avg {m['IoU']:9.6f} {m['mIoU']:9.6f}")
    if result.get('error'): lines.append('error='+result['error'])
    lines.append('Original checkpoints/optimizer/RNG/history caches unchanged. Quality evaluation, NOT FPS.')
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event()
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=1., eval_population='dev512')
    p.add_argument('--plan-cache', required=True)
    p.add_argument('--resume-evaluation', action='store_true')
    a = p.parse_args(argv)
    if (not a.resume or a.max_updates or a.mode != 'gt' or a.epochs != 2 or a.train_fraction != 1.
            or a.eval_population != 'dev512' or a.window_batch_size != 4 or a.source_budget != 128):
        raise ValueError('frozen full-GT source only; original B4/source128 contracts required')
    source, out = Path(a.resume).resolve(), Path(a.out_dir).resolve()
    if out == source.parent or source.parent in out.parents: raise ValueError('evaluation OUTSIDE training directory')
    lock = out.parent/'.full_gt_external_locks'/legacy.stable_json_fingerprint(str(out))
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume_evaluation:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(source, out/'frozen/source.pt')
        snapshot = out/'frozen/source.pt'
        saved = torch.load(snapshot, map_location='cpu', weights_only=False); validate_source(saved)
        fields = vars(a).copy()
        for k in ('resume_evaluation', 'evaluate_only'): fields.pop(k)
        identity = dict(protocol=engine.PROTOCOL, args=fields, source_sha256=legacy.sha256(snapshot),
            implementation={f.name: legacy.sha256(f) for f in Path(__file__).parent.glob('*.py')})
        path = out/'evaluation_contract.json'
        if a.resume_evaluation and json.loads(path.read_text()) != identity:
            raise RuntimeError('evaluation source/settings/code changed')
        legacy.write_json(path, identity)
        print(f'FROZEN_SOURCE {source}; epoch={saved["epoch"]}; update={saved["updates"]}', flush=True)
        planner = PlannerCache(a.plan_cache); captured = []
        names = ('load_cache', 'prefetched_batches', 'summarize', 'save_resume_checkpoint')
        original = {n: getattr(legacy, n) for n in names}
        original_eval, original_train = gt_engine.evaluate, gt_engine.train_arm
        def load(path):
            value = original['load_cache'](path)
            if Path(path).resolve() == Path(a.dev_cache).resolve(): captured[:] = [value[1]]
            return value
        def batches(provider, dev_source, ignored, descriptors, **kwargs):
            if not captured: raise RuntimeError('VAL cache not loaded')
            records = captured[0]; requested = legacy.record_keys(records)
            if len(requested) != 4369: raise RuntimeError('full VAL4369 required')
            planner.prepare(requested)
            selected = legacy.align_records(records, [tuple(k) for k in planner.metadata['selected_keys']])
            if set(k[0] for k in requested) & set(k[0] for k in saved['contract']['train_keys']):
                raise RuntimeError('TRAIN/VAL scene overlap')
            legacy.write_json(out/'population.json', planner.metadata)
            print(f'VAL requested=4369 covered={len(selected)} missing={len(planner.metadata["missing_keys"])}', flush=True)
            return original['prefetched_batches'](provider, dev_source, [[r] for r in selected], descriptors, **kwargs)
        def evaluate(models, provider, dev_source, rows, **kwargs):
            return engine.evaluate(models, provider, dev_source, rows, planner=planner, **kwargs)
        def forbidden(*args, **kwargs): raise RuntimeError('external evaluation cannot train/save training checkpoints')
        inner = vars(a).copy()
        for k in ('plan_cache', 'resume_evaluation'): inner.pop(k)
        inner.update(resume=str(snapshot), evaluate_only=True)
        args = []
        for k, v in inner.items():
            if k == 'override' or v is None: continue
            if isinstance(v, bool):
                if v: args.append('--'+k.replace('_', '-'))
            else: args += ['--'+k.replace('_', '-'), str(v)]
        for v in a.override or []: args += ['--override', v]
        try:
            legacy.load_cache, legacy.prefetched_batches = load, batches
            legacy.summarize, legacy.save_resume_checkpoint = summarize, forbidden
            gt_engine.evaluate, gt_engine.train_arm = evaluate, forbidden
            status = training.main(event, args)
        finally:
            for k, v in original.items(): setattr(legacy, k, v)
            gt_engine.evaluate, gt_engine.train_arm = original_eval, original_train
            captured.clear()
        if legacy.sha256(snapshot) != identity['source_sha256']: raise RuntimeError('frozen checkpoint changed')
        result_path = out/'results.json'; result = json.loads(result_path.read_text())
        # Preserve the source DEV512 contract used for strict checkpoint loading;
        # distinguish it from the ACTUAL all-val evaluation population explicitly.
        result['evaluation_protocol'] = engine.PROTOCOL
        result['actual_eval_population'] = 'all_val_planner_covered'
        result['source_checkpoint'] = str(source); result['source_sha256'] = identity['source_sha256']
        if result.get('evaluation'): result['status'] = 'complete'
        legacy.write_json(result_path, result)
        text = summarize(result); (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)
        return status


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
