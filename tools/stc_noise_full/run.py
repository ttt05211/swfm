"""Unfreeze SharedEncoder/STWM/Refiner, fresh Adam, empirical HISTORY x500."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as legacy, data as joint_data
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.loss_controls.run import capped_groups
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.perception_robustness.evaluation import evaluator
from tools.scene_ego_head.frozen_gt.run import validate_completed_source
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.stc_noise_adaptation.augmentation import load_model, RECIPE
from tools.stc_noise_adaptation.data import history_inputs
from tools.stc_noise_adaptation.run import selection, PROTOCOL as FROZEN_PROTOCOL
from tools.stc_noise_execution.pipeline import prefetch_adapter, noise_execution, CheckedStrong

PROTOCOL = 'GTepoch2_FULL_unfreeze_empirical_history_adapt500_v1'


class FullBody(GTBody):
    def __init__(self, config, source, digest):
        c = source['contract']['model']; shared, repair = c['shared_history'], c['repair']
        super().__init__(config, feature_width=shared['width'], z_bins=shared['z_bins'],
            memory_width=repair['memory_width'], width=repair['width'], chunk_size=repair['chunk_size'])
        self.load_state_dict(source['models']['gt'], strict=True)
        self.columns.epsilon = repair['epsilon']; self.source_sha256 = digest
        self.requires_grad_(True)

    def configs(self):
        return {**super().configs(), 'adaptation': dict(protocol=PROTOCOL, fixed_updates=500,
            source_sha256=self.source_sha256, frozen=[], trainable=['SharedEncoder', 'STWM', 'SemanticRefiner'],
            initialization='ORIGINAL_GTepoch2_fresh_Adam_RNG', prior_coefficient=1., training_known='dense_valid')}


def references(directory):
    root = Path(directory).resolve(); workflow = json.loads((root/'workflow.json').read_text(encoding='utf-8'))
    reports = {}
    for arm in ('clean', 'empirical'):
        r = json.loads((root/arm/'results.json').read_text(encoding='utf-8')); c = r['contract']
        if (r['status'] != 'complete' or r['updates'] != 500 or c['protocol'] != FROZEN_PROTOCOL
                or c['augmentation'] != arm or r['evaluation']['windows'] != 64
                or c['eval_keys'] != r['evaluation']['evaluation_keys']): raise RuntimeError('completed original paired500 required')
        reports[arm] = r
    for k in reports['clean']['contract']:
        if k != 'augmentation' and reports['clean']['contract'][k] != reports['empirical']['contract'][k]:
            raise RuntimeError('original clean/empirical are not matched: '+k)
    return root, workflow, reports


def summary(result, refs):
    lines = ['===== ORIGINAL GT EPOCH2 / FULL UNFREEZE / EMPIRICAL500 =====', 'status='+result['status'],
        f'updates={result.get("updates", 0)}/500; complete_dataset_epochs=0',
        'ORIGINAL GTepoch2 weights, not previous adapted Refiner; fresh Adam/RNG.',
        'SharedEncoder/STWM/Refiner ALL trainable; ONLY full6 CE+Lovasz; no ego/motion/existence auxiliary.',
        'Body LR1e-5 / Refiner LR1e-4; same first500 groups/seed/order/B4/H6/noise recipe/GT ego.',
        'TRAIN dense-valid; 50% empirical HISTORY corruption; c=1 unchanged.',
        'Exact same remaining64 as prior paired run; full-grid OCC+GT / STC+GT; NOT pristine test.',
        'VAL-calibrated augmentation, NOT target-domain-information-free zero-shot.']
    if 'performance' in result: lines += ['PERFORMANCE '+json.dumps(result['performance'])]
    if result.get('evaluation'):
        metrics = result['evaluation']['variants']['gt']
        lines += ['arm                route                        mIoU       IoU MovingMicro']
        rows = [(n, refs[n]['evaluation']['variants']['gt']) for n in ('clean', 'empirical')]+[('full_empirical', metrics)]
        for arm, values in rows:
            for name, m in values.items():
                lines += [f'{arm:18s} {name:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}']
        for ref in ('clean', 'empirical'):
            delta = {name: {k: m[k]-refs[ref]['evaluation']['variants']['gt'][name][k]
                            for k in ('mIoU', 'IoU', 'MovingMicro')} for name, m in metrics.items()}
            lines += ['full_minus_'+ref+'='+json.dumps(delta)]
        lines += ['vs empirical tests trainable scope; vs clean ALSO changes corruption, NOT a full-unfreeze clean control.']
    if result.get('error'): lines += ['error='+result['error']]
    return '\n'.join(lines)+'\nNo automatic retry, larger training or promotion. NOT FPS.\n'


def main(event=None, argv=None):
    event = event or threading.Event()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-workflow', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--resume-experiment', action='store_true')
    p.add_argument('--max-updates', type=int, default=0)
    a = p.parse_args(argv)
    if not 0 <= a.max_updates <= 500: raise ValueError('fixed 500-update budget')
    parent, workflow, refs = references(a.reference_workflow)
    out = Path(a.out_dir).resolve()
    if out == parent or parent in out.parents: raise ValueError('new experiment OUTSIDE old paired directory')
    fields = workflow['args'].copy()
    source_path = parent/'empirical/frozen/GTepoch2.pt'
    digest = legacy.sha256(source_path)
    if digest != refs['empirical']['contract']['source_GT_sha256']: raise RuntimeError('ORIGINAL GT snapshot differs')
    source = torch.load(source_path, map_location='cpu', weights_only=False); validate_completed_source(source)
    noise, binding = load_model(parent/'profile/noise_model.json')
    runtime = argparse.Namespace(**fields)
    keys, inventory = selection(runtime, noise)
    if (keys != refs['empirical']['contract']['eval_keys'] or inventory != workflow['STC_inventory']
            or binding != refs['empirical']['contract']['noise_identity'] or binding['source_GT_sha256'] != digest):
        raise RuntimeError('same remaining64/profile/STC/GT identity required; no population substitution')
    for k in ('train_cache', 'dev_cache', 'train_info', 'dev_info', 'base_checkpoint'):
        if legacy.sha256(fields[k]) != source['contract']['data'][k]: raise RuntimeError('original GT metadata differs: '+k)
    identity = dict(protocol=PROTOCOL, reference={str(x.resolve()): legacy.sha256(x) for x in
        (parent/'workflow.json', parent/'clean/results.json', parent/'empirical/results.json')},
        source_GT_sha256=digest, noise_identity=binding, eval_keys=keys)
    lock = out.parent/'.full_noise_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if out.exists():
            if not a.resume_experiment or json.loads((out/'reference.json').read_text(encoding='utf-8')) != identity:
                raise RuntimeError('NEW directory or SAME full-unfreeze experiment with --resume-experiment')
        else:
            if a.resume_experiment: raise RuntimeError('missing experiment to resume')
            out.mkdir(); (out/'frozen').mkdir()
            snapshot_checkpoint(source_path, out/'frozen/GTepoch2.pt')
            for name in ('noise_model.json', 'evaluation.json'): shutil.copy2(parent/'profile'/name, out/'frozen'/name)
            legacy.write_json(out/'reference.json', identity)
        if legacy.sha256(out/'frozen/GTepoch2.pt') != digest or load_model(out/'frozen/noise_model.json')[1] != binding:
            raise RuntimeError('frozen source artifacts changed')
        names = ('parser', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution', 'PROTOCOL', 'summarize',
            'train_arm', 'evaluate', 'epoch_groups', 'load_manifest', 'prefetched_batches', 'write_json',
            'restore_checkpoint', 'stable_json_fingerprint')
        old = {n: getattr(legacy, n) for n in names}; old_ego = joint_data.ego_inputs
        models = []; initial = {}
        def construct(config):
            model = FullBody(config, source, digest); models.append(model)
            initial.update(Encoder_STWM_sha256=tensor_state_fingerprint(model.transport.state_dict()),
                           Refiner_sha256=tensor_state_fingerprint(model.columns.state_dict()))
            legacy.write_json(out/'initialization_audit.json', dict(initial, source_GT_sha256=digest,
                imported_adapted_weights=False, imported_Adam=False, imported_RNG=False, fully_trainable=True))
            return model
        def mark(c):
            for field in ('seed', 'train_keys', 'eval_keys', 'window_batch', 'source_budget', 'total_updates', 'data', 'schedule', 'lr'):
                previous = refs['empirical']['contract']
                if field in previous and c[field] != previous[field]:
                    raise RuntimeError('same prior sampling/budget/optimizer schedule required: '+field)
            c.update(initialization='ORIGINAL_GTepoch2_fresh_Adam_RNG', source_GT_sha256=digest,
                augmentation='empirical', noise_identity=binding, recipe=RECIPE, fully_unfrozen=True,
                eval_selection='SAME_prior_remaining64_NOT_pristine_test', training_known='dense_valid',
                objective={'CE': 1., 'Lovasz': 1., 'motion': 0., 'existence': 0., 'ego': 0.},
                thresholds=None, weights=[0., 0., 1.], complete_dataset_epochs=0)
        def write(path, value):
            if Path(path).name == 'contract.json': mark(value)
            return old['write_json'](path, value)
        def restore(saved, models, optimizers, contract):
            mark(contract); return old['restore_checkpoint'](saved, models, optimizers, contract)
        def manifest(path):
            original, _, third = old['load_manifest'](path); return original, list(map(tuple, keys)), third
        def record(audit): legacy.write_json(out/'Strong_reuse_audit.json', audit)
        finish = CheckedStrong(record)
        def train(model, optimizer, provider, rows, mode, **kw):
            tick = time.perf_counter(); prepared = [finish(provider, row) for row in rows]; sec = time.perf_counter()-tick
            stats = gt.train_arm(model, optimizer, provider, prepared, mode, **kw)
            stats['seconds'] += sec; stats['stages_seconds']['fresh_corrupted_Strong'] = sec
            audits = [r[1].get('_empirical_noise', {}) for r in rows]
            stats['augmentation'] = dict(selected_windows=sum(bool(v.get('selected')) for v in audits),
                changed_voxels=sum(sum(v.get('changed_voxels', [])) for v in audits),
                CPU_simulation_seconds_sum=sum(v.get('simulation_seconds', 0.) for v in audits),
                CPU_geometry_seconds_sum=sum(v.get('geometry_prepare_seconds', 0.) for v in audits))
            return stats
        eval_base = evaluator(fields['stc_root'], inventory=inventory)
        def evaluate(*xs, **kw):
            r = eval_base(*xs, **kw)
            if r['windows'] != 64: raise RuntimeError('exact prior remaining64 required')
            r.update(evaluation_keys=keys, scoring='same prior remaining64; full-grid; GT ego; OCC lidar-known/STC dense-valid')
            return r
        root = Path(__file__).resolve().parents[2]
        dependencies = list(Path(__file__).parent.glob('*.py'))
        for subdir in ('tools/stc_noise_adaptation', 'tools/stc_noise_execution', 'tools/shared_history_experiments/refine',
                       'tools/shared_history_experiments/full_gt', 'tools/perception_robustness'):
            dependencies.extend((root/subdir).glob('*.py'))
        dependencies.extend(root/x for x in ('tools/shared_history_experiments/model.py',
            'tools/shared_history_experiments/history.py', 'tools/history_validity/data.py',
            'tools/final_semantic_evaluation/predictor.py', 'tools/scene_ego_head/integration.py',
            'tools/scene_ego_head/diagnostics/engine.py', 'tools/stc_noise_profile/statistics.py',
            'tools/stc_noise_profile/data.py'))
        def fingerprint(value):
            if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
                value = dict(base=value, prototype={str(x.relative_to(root)): legacy.sha256(x) for x in dependencies})
            return old['stable_json_fingerprint'](value)
        args = []
        fields.update(mode='gt', epochs=1, train_fraction=.2, eval_population='dev64',
                      motion_lr=1e-5, repair_lr=1e-4, ego_lr=1e-4)
        for key, value in fields.items():
            if key in ('gt_checkpoint', 'noise_model', 'stc_root', 'resume', 'evaluate_only'): continue
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
            legacy.train_arm, legacy.evaluate, legacy.summarize = train, evaluate, lambda r: summary(r, refs)
            legacy.epoch_groups = lambda *xs: capped_groups(old['epoch_groups'], *xs)
            legacy.load_manifest, legacy.write_json, legacy.restore_checkpoint = manifest, write, restore
            legacy.stable_json_fingerprint = fingerprint
            legacy.prefetched_batches = prefetch_adapter(old['prefetched_batches'], 'empirical', noise, fields['seed'], stop_event=event)
            joint_data.ego_inputs = gt.history_inputs
            print('FULL UNFREEZE: ORIGINAL GTepoch2 -> fresh Adam x500 / same empirical HISTORY / same remaining64.', flush=True)
            print('Generic driver RANDOM/epoch1 text is not this initialization or a complete dataset epoch.', flush=True)
            with noise_execution(): status = legacy.main(event, args)
            result = json.loads((out/'train/results.json').read_text(encoding='utf-8'))
            legacy.write_json(out/'results.json', result); (out/'summary.txt').write_text(summary(result, refs), encoding='utf-8')
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
