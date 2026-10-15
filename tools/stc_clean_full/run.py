"""Clean TRAIN GT, dense-valid, FULL unfreeze500; same previous remaining64."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy, data as joint_data
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.loss_controls.run import capped_groups
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.perception_robustness.evaluation import evaluator
from tools.scene_ego_head.frozen_gt.run import validate_completed_source
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.stc_noise_adaptation.data import history_inputs, prefetch_adapter
from tools.stc_noise_full.run import FullBody, references

PROTOCOL = 'GTepoch2_FULL_unfreeze_CLEAN_dense_valid500_v1'


class CleanFullBody(FullBody):
    def configs(self):
        value = super().configs()
        value['adaptation'].update(protocol=PROTOCOL, augmentation='clean', synthetic_noise=False)
        return value


def summary(result, refs):
    lines = ['===== ORIGINAL GT EPOCH2 / CLEAN DENSE-VALID / FULL UNFREEZE500 =====',
        'status='+result['status'], f'updates={result.get("updates", 0)}/500; complete_dataset_epochs=0',
        'ORIGINAL GTepoch2 initialization; fresh Adam/RNG. No adapted weights imported.',
        'SharedEncoder/STWM/Refiner ALL trainable; ONLY full6 CE+Lovasz, H6/B4/source128 grouping.',
        'Body LR1e-5 / Refiner LR1e-4; same first500 groups/seed/order as previous clean Refiner500.',
        'CLEAN original TRAIN GT ONLY; dense-valid TRAIN, OCC eval original lidar-known / STC dense-valid.',
        'No noise fitting parameters, corruption, real STC training, ego/motion/existence losses or coefficient changes.',
        'GT future ego explicitly conditioned; c=1 unchanged; full-grid metrics.',
        'Exact previous remaining64, NOT old DEV64/512 or a pristine independent test.']
    if 'performance' in result: lines += ['PERFORMANCE '+json.dumps(result['performance'])]
    if result.get('evaluation'):
        metrics = result['evaluation']['variants']['gt']
        lines += ['arm                route                        mIoU       IoU MovingMicro']
        rows = [(n, refs[n]['evaluation']['variants']['gt']) for n in ('clean', 'empirical')]
        for arm, values in rows+[('full_clean', metrics)]:
            for name, m in values.items():
                lines += [f'{arm:18s} {name:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}']
        for ref in ('clean', 'empirical'):
            delta = {name: {k: m[k]-refs[ref]['evaluation']['variants']['gt'][name][k]
                            for k in ('mIoU', 'IoU', 'MovingMicro')} for name, m in metrics.items()}
            lines += ['full_clean_minus_'+ref+'='+json.dumps(delta)]
        lines += ['vs clean tests trainable scope; vs empirical ALSO changes corruption.',
                  'This does NOT isolate known versus additional optimization; no automatic promotion.']
    if result.get('error'): lines += ['error='+result['error']]
    return '\n'.join(lines)+'\nPrevious runs/checkpoints/caches unchanged. No extension. NOT FPS.\n'


def main(event=None, argv=None):
    event = event or threading.Event()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-workflow', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--resume-experiment', action='store_true'); p.add_argument('--max-updates', type=int, default=0)
    a = p.parse_args(argv)
    if not 0 <= a.max_updates <= 500: raise ValueError('fixed500 budget')
    parent, workflow, refs = references(a.reference_workflow)
    out = Path(a.out_dir).resolve()
    if out == parent or parent in out.parents: raise ValueError('new output OUTSIDE previous paired run')
    fields = workflow['args'].copy(); prior = refs['clean']['contract']
    source_path = parent/'clean/frozen/GTepoch2.pt'; digest = legacy.sha256(source_path)
    if digest != prior['source_GT_sha256']: raise RuntimeError('ORIGINAL GTepoch2 snapshot differs')
    source = torch.load(source_path, map_location='cpu', weights_only=False); validate_completed_source(source)
    keys = prior['eval_keys']; inventory = workflow['STC_inventory']
    if len(keys) != 64 or len({tuple(k) for k in keys}) != 64: raise RuntimeError('exact previous unique remaining64 required')
    for path, size, mtime in inventory:
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime): raise RuntimeError('previous STC evaluation data changed')
    for k in ('train_cache', 'dev_cache', 'train_info', 'dev_info', 'base_checkpoint'):
        if legacy.sha256(fields[k]) != source['contract']['data'][k]: raise RuntimeError('original GT metadata differs: '+k)
    identity = dict(protocol=PROTOCOL, source_GT_sha256=digest, eval_keys=keys,
        reference={str(x.resolve()): legacy.sha256(x) for x in
                   (parent/'workflow.json', parent/'clean/results.json', parent/'empirical/results.json')})
    lock = out.parent/'.clean_full_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if out.exists():
            if not a.resume_experiment or json.loads((out/'reference.json').read_text(encoding='utf-8')) != identity:
                raise RuntimeError('NEW directory or SAME clean-full run with --resume-experiment')
        else:
            if a.resume_experiment: raise RuntimeError('missing experiment to resume')
            out.mkdir(); (out/'frozen').mkdir(); snapshot_checkpoint(source_path, out/'frozen/GTepoch2.pt')
            legacy.write_json(out/'reference.json', identity)
        if legacy.sha256(out/'frozen/GTepoch2.pt') != digest: raise RuntimeError('source snapshot changed')
        names = ('JointOccEgo', 'HistoryInputs', 'BothActionsExecution', 'PROTOCOL', 'summarize',
            'train_arm', 'evaluate', 'epoch_groups', 'load_manifest', 'prefetched_batches', 'write_json',
            'restore_checkpoint', 'stable_json_fingerprint')
        old = {n: getattr(legacy, n) for n in names}; old_ego = joint_data.ego_inputs
        models, initial = [], {}
        def construct(config):
            model = CleanFullBody(config, source, digest); models.append(model)
            initial.update(Encoder_STWM_sha256=tensor_state_fingerprint(model.transport.state_dict()),
                           Refiner_sha256=tensor_state_fingerprint(model.columns.state_dict()))
            old['write_json'](out/'initialization_audit.json', dict(initial, source_GT_sha256=digest,
                imported_adapted_weights=False, imported_Adam=False, imported_RNG=False, fully_trainable=True))
            return model
        def mark(c):
            for field in ('seed', 'train_keys', 'eval_keys', 'window_batch', 'source_budget', 'total_updates', 'data', 'schedule', 'lr'):
                if field in prior and c[field] != prior[field]: raise RuntimeError('same clean sampling/schedule required: '+field)
            c.update(initialization='ORIGINAL_GTepoch2_fresh_Adam_RNG', source_GT_sha256=digest,
                augmentation='clean', synthetic_noise=False, fully_unfrozen=True, training_known='dense_valid',
                objective={'CE': 1., 'Lovasz': 1., 'motion': 0., 'existence': 0., 'ego': 0.},
                eval_selection='SAME_prior_remaining64_NOT_pristine_test', thresholds=None, weights=[0., 0., 1.],
                complete_dataset_epochs=0)
        def write(path, value):
            if Path(path).name == 'contract.json': mark(value)
            return old['write_json'](path, value)
        def restore(saved, models, optimizers, contract):
            mark(contract); return old['restore_checkpoint'](saved, models, optimizers, contract)
        def manifest(path):
            original, _, third = old['load_manifest'](path); return original, list(map(tuple, keys)), third
        eval_base = evaluator(fields['stc_root'], inventory=inventory)
        def evaluate(*xs, **kw):
            r = eval_base(*xs, **kw)
            if r['windows'] != 64: raise RuntimeError('exact previous remaining64 required')
            r.update(evaluation_keys=keys, scoring='same remaining64; OCC lidar-known / STC dense-valid; full-grid; GT ego')
            return r
        root = Path(__file__).resolve().parents[2]; dependencies = list(Path(__file__).parent.glob('*.py'))
        for sub in ('tools/stc_noise_adaptation', 'tools/stc_noise_full', 'tools/shared_history_experiments/refine',
                    'tools/shared_history_experiments/full_gt', 'tools/perception_robustness'):
            dependencies += list((root/sub).glob('*.py'))
        dependencies += [root/x for x in ('tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
            'tools/history_validity/data.py', 'tools/final_semantic_evaluation/predictor.py',
            'tools/scene_ego_head/integration.py', 'tools/scene_ego_head/diagnostics/engine.py')]
        def fingerprint(value):
            if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
                value = dict(base=value, prototype={str(x.relative_to(root)): legacy.sha256(x) for x in dependencies})
            return old['stable_json_fingerprint'](value)
        args = []
        fields.update(mode='gt', epochs=1, train_fraction=.2, eval_population='dev64',
                      motion_lr=1e-5, repair_lr=1e-4, ego_lr=1e-4)
        for key, value in fields.items():
            if key in ('gt_checkpoint', 'noise_model', 'stc_root', 'resume', 'evaluate_only', 'out_dir', 'max_updates', 'augmentation'): continue
            if key == 'override':
                for v in value or []: args += ['--override', v]
            elif isinstance(value, bool):
                if value: args += ['--'+key.replace('_', '-')]
            elif value is not None: args += ['--'+key.replace('_', '-'), str(value)]
        args += ['--out-dir', str(out/'train')]
        if a.max_updates: args += ['--max-updates', str(a.max_updates)]
        if a.resume_experiment: args += ['--resume', str(out/'train/last.pt')]
        try:
            legacy.JointOccEgo, legacy.PROTOCOL = construct, PROTOCOL
            legacy.HistoryInputs, legacy.BothActionsExecution = history_inputs(keys), NoLegacyHeadGraph
            legacy.train_arm, legacy.evaluate, legacy.summarize = gt.train_arm, evaluate, lambda r: summary(r, refs)
            legacy.epoch_groups = lambda *xs: capped_groups(old['epoch_groups'], *xs)
            legacy.load_manifest, legacy.write_json, legacy.restore_checkpoint = manifest, write, restore
            legacy.stable_json_fingerprint = fingerprint
            # CLEAN path: no simulator or fitted probabilities are passed or invoked.
            legacy.prefetched_batches = prefetch_adapter(old['prefetched_batches'], 'clean', None, fields['seed'])
            joint_data.ego_inputs = gt.history_inputs
            print('CLEAN FULL500: ORIGINAL GTepoch2; TRAIN GT only; dense-valid; no noise; same remaining64.', flush=True)
            print('Generic RANDOM/epoch1 text is not this warm initialization or a complete dataset epoch.', flush=True)
            status = legacy.main(event, args)
            result = json.loads((out/'train/results.json').read_text(encoding='utf-8'))
            old['write_json'](out/'results.json', result); (out/'summary.txt').write_text(summary(result, refs), encoding='utf-8')
            return status
        finally:
            joint_data.ego_inputs = old_ego
            for n, v in old.items(): setattr(legacy, n, v)
            if models:
                m = models[0]
                legacy.write_json(out/'trainable_audit.json', dict(fully_trainable=all(p.requires_grad for p in m.parameters()),
                    Encoder_STWM_changed=tensor_state_fingerprint(m.transport.state_dict()) != initial['Encoder_STWM_sha256'],
                    Refiner_changed=tensor_state_fingerprint(m.columns.state_dict()) != initial['Refiner_sha256']))


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
