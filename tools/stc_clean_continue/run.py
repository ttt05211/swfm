"""Preserve clean-full500 weights/Adam/RNG; next500 groups; same remaining64."""
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
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.perception_robustness.evaluation import evaluator
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.stc_noise_adaptation.data import history_inputs, prefetch_adapter
from tools.stc_clean_full import run as previous

PROTOCOL = 'CLEAN_full500_preserve_Adam_RNG_next500_cosine_restart_v1'


def select_source(directory, search_root):
    if directory: return Path(directory).resolve()
    candidates = []
    for root in Path(search_root).glob('stc_clean_full500_*'):
        path = root/'results.json'
        if path.is_file():
            report = json.loads(path.read_text(encoding='utf-8'))
            if (report.get('status') == 'complete' and report.get('updates') == 500
                    and report.get('contract', {}).get('protocol') == previous.PROTOCOL):
                candidates.append(root.resolve())
    if len(candidates) != 1:
        raise RuntimeError(f'need ONE completed clean-full500, found {len(candidates)}; set CLEAN_CONT_SOURCE explicitly')
    return candidates[0]


def next_groups(original, records, seed, epoch, batch_size, source_budget):
    if epoch != 0: raise ValueError('single next500 stage only')
    groups = original(records, seed, 0, batch_size, source_budget)
    if len(groups) < 1000: raise RuntimeError('at least1000 original TRAIN20% groups required; never repeat first500')
    return groups[500:1000]


def import_training_state(saved, models, optimizers):
    if set(models) != {'gt'} or set(saved['models']) != {'gt'}: raise RuntimeError('GT-only source required')
    models['gt'].load_state_dict(saved['models']['gt'], strict=True)
    optimizers['gt'].load_state_dict(saved['optimizers']['gt'])
    legacy.restore_rng(saved['rng'])
    # Keep Adam step/moments/RNG. gt.train_arm resets LR only, via the explicit
    # new-stage cosine with initial_lr retained from the completed500 optimizer.
    return 0, 0, 0, dict(training=[], gradient_links={}, epochs=[])


class ContinueBody(previous.CleanFullBody):
    def configs(self):
        c = super().configs()
        c['adaptation'].update(protocol=PROTOCOL, fixed_updates=500,
            initialization='completed_CLEAN_full500_weights_Adam_RNG', cumulative_updates=1000)
        return c


def summary(result, source_report):
    n = result.get('updates', 0)
    lines = ['===== CLEAN DENSE-VALID / CONTINUE500 -> CUMULATIVE1000 =====', 'status='+result['status'],
        f'additional_updates={n}/500; cumulative_adaptation_updates={500+n}; NOT a dataset epoch.',
        'Start from COMPLETED clean-full500; weights/Adam steps/moments/RNG imported, not original GTepoch2 again.',
        'Next original TRAIN groups501--1000; first500 NOT repeated; SAME dense-valid/H6/B4/source128/full6 CE+Lovasz.',
        'SharedEncoder/STWM/Refiner ALL trainable; no pollution/STC training/ego/motion/existence auxiliary.',
        'NEW500-step cosine RESTART: body1e-5/refiner1e-4, floor0.01; NOT one planned1000-step cosine.',
        'Same previous remaining64/full-grid/GT ego; OCC lidar-known / STC dense-valid. NOT pristine test.']
    if 'performance' in result: lines += ['PERFORMANCE '+json.dumps(result['performance'])]
    if result.get('evaluation'):
        before = source_report['evaluation']['variants']['gt']; after = result['evaluation']['variants']['gt']
        lines += ['cumulative_updates route                        mIoU       IoU MovingMicro']
        for step, values in ((500, before), (500+n, after)):
            for name, m in values.items():
                lines += [f'{step:18d} {name:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}']
        lines += ['additional500_minus_source500='+json.dumps({name: {k: m[k]-before[name][k]
                  for k in ('mIoU', 'IoU', 'MovingMicro')} for name, m in after.items()})]
    if result.get('error'): lines += ['error='+result['error']]
    return '\n'.join(lines)+'\nSource500/checkpoints/caches READ ONLY; no auto-extension/promotion. NOT FPS.\n'


def main(event=None, argv=None):
    event = event or threading.Event(); p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True)
    p.add_argument('--resume-experiment', action='store_true'); p.add_argument('--max-updates', type=int, default=0)
    a = p.parse_args(argv)
    if not 0 <= a.max_updates <= 500: raise ValueError('only500 additional updates allowed')
    repo = Path(__file__).resolve().parents[2]
    parent = select_source(a.source_dir, repo/'outputs/p0_f9_shared_history')
    source_path = parent/'train/last.pt'; report_path = parent/'results.json'
    source_report = json.loads(report_path.read_text(encoding='utf-8'))
    saved = torch.load(source_path, map_location='cpu', weights_only=False); c = saved['contract']
    if (source_report['status'] != 'complete' or saved['protocol'] != previous.PROTOCOL or c != source_report['contract']
            or saved['updates'] != 500 or source_report['updates'] != 500 or saved['epoch'] != 1 or saved['batch'] != 0
            or c['total_updates'] != 500 or not c['fully_unfrozen'] or c['augmentation'] != 'clean'
            or c['training_known'] != 'dense_valid' or source_report['evaluation']['windows'] != 64
            or source_report['evaluation']['evaluation_keys'] != c['eval_keys']
            or c['objective'] != dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.)):
        raise RuntimeError('completed CLEAN full-unfreeze500 checkpoint + matched same64 report required')
    reference = json.loads((parent/'reference.json').read_text(encoding='utf-8'))
    workflows = [Path(path) for path in reference['reference'] if Path(path).name == 'workflow.json']
    if len(workflows) != 1: raise RuntimeError('one original paired workflow provenance required')
    for path, digest in reference['reference'].items():
        if legacy.sha256(path) != digest: raise RuntimeError('original paired reference changed')
    workflow = json.loads(workflows[0].read_text(encoding='utf-8')); fields = workflow['args'].copy()
    keys = c['eval_keys']; inventory = workflow['STC_inventory']
    if len(keys) != 64 or len({tuple(k) for k in keys}) != 64: raise RuntimeError('same unique remaining64 required')
    for path, size, timestamp in inventory:
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, timestamp): raise RuntimeError('STC evaluation data changed')
    for key, digest in c['data'].items():
        if legacy.sha256(fields[key]) != digest: raise RuntimeError('TRAIN/VAL metadata changed: '+key)
    out = Path(a.out_dir).resolve()
    if out == parent or parent in out.parents or out == workflows[0].parent or workflows[0].parent in out.parents:
        raise ValueError('new output OUTSIDE previous source and paired experiments required')
    source_sha = legacy.sha256(source_path)
    identity = dict(protocol=PROTOCOL, source_checkpoint=str(source_path), source_sha256=source_sha,
        source_report_sha256=legacy.sha256(report_path), source_reference_sha256=legacy.sha256(parent/'reference.json'))
    lock = out.parent/'.clean_continue_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if out.exists():
            if not a.resume_experiment or json.loads((out/'reference.json').read_text(encoding='utf-8')) != identity:
                raise RuntimeError('NEW output or SAME continuation with --resume-experiment required')
        else:
            if a.resume_experiment: raise RuntimeError('missing continuation to resume')
            out.mkdir(); legacy.write_json(out/'reference.json', identity)
        names = ('JointOccEgo', 'HistoryInputs', 'BothActionsExecution', 'PROTOCOL', 'summarize',
            'train_arm', 'evaluate', 'epoch_groups', 'load_manifest', 'prefetched_batches', 'write_json',
            'restore_checkpoint', 'stable_json_fingerprint')
        old = {n: getattr(legacy, n) for n in names}; old_ego = joint_data.ego_inputs
        def construct(config): return ContinueBody(config, saved, source_sha)
        def mark(contract):
            for key in ('seed', 'train_keys', 'eval_keys', 'window_batch', 'source_budget', 'data', 'lr', 'execution', 'training_backend'):
                if contract[key] != c[key]: raise RuntimeError('source500 data/execution differs: '+key)
            contract.update(initialization='completed_clean_full500_weights_Adam_RNG', source500=identity,
                source_GT_sha256=c['source_GT_sha256'], source_completed_updates=500, cumulative_budget=1000,
                augmentation='clean', synthetic_noise=False, fully_unfrozen=True, training_known='dense_valid',
                objective=c['objective'], thresholds=None, weights=[0., 0., 1.],
                schedule='ADDITIONAL500_cosine_RESTART_floor0.01_Adam_preserved', complete_dataset_epochs=0)
        def write(path, value):
            if Path(path).name == 'contract.json': mark(value)
            if Path(path).name == 'results.json':
                value.update(additional_updates=value['updates'], cumulative_adaptation_updates=500+value['updates'])
            return old['write_json'](path, value)
        def restore(checkpoint, models, optimizers, contract):
            mark(contract)
            if a.resume_experiment: return old['restore_checkpoint'](checkpoint, models, optimizers, contract)
            if checkpoint['protocol'] != previous.PROTOCOL or checkpoint['contract'] != c:
                raise RuntimeError('bootstrap must import exact source500 checkpoint')
            state = import_training_state(checkpoint, models, optimizers)
            old['write_json'](out/'import_audit.json', dict(source_update=500, imported_Adam_steps_moments=True,
                imported_RNG=True, source_model_sha256=tensor_state_fingerprint(checkpoint['models']['gt']),
                stage_initial_model_sha256=tensor_state_fingerprint(models['gt'].state_dict()),
                next_source_group=501, stage_LR_restarts=True))
            return state
        def manifest(path):
            original, _, third = old['load_manifest'](path); return original, list(map(tuple, keys)), third
        eval_base = evaluator(fields['stc_root'], inventory=inventory)
        def evaluate(*xs, **kw):
            r = eval_base(*xs, **kw)
            if r['windows'] != 64: raise RuntimeError('exact remaining64 required')
            r['evaluation_keys'] = keys; return r
        files = list(Path(__file__).parent.glob('*.py'))
        for sub in ('tools/stc_clean_full', 'tools/stc_noise_full', 'tools/stc_noise_adaptation',
                    'tools/perception_robustness', 'tools/shared_history_experiments/refine', 'tools/shared_history_experiments/full_gt'):
            files += list((repo/sub).glob('*.py'))
        files += [repo/x for x in ('tools/history_validity/data.py', 'tools/shared_history_experiments/model.py',
                  'tools/shared_history_experiments/history.py', 'tools/scene_ego_head/diagnostics/engine.py')]
        def fingerprint(value):
            if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
                value = dict(base=value, prototype={str(f.relative_to(repo)): legacy.sha256(f) for f in files})
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
        args += ['--out-dir', str(out/'train'), '--resume', str(out/'train/last.pt' if a.resume_experiment else source_path)]
        if a.max_updates: args += ['--max-updates', str(a.max_updates)]
        try:
            legacy.JointOccEgo, legacy.PROTOCOL = construct, PROTOCOL
            legacy.HistoryInputs, legacy.BothActionsExecution = history_inputs(keys), NoLegacyHeadGraph
            legacy.train_arm, legacy.evaluate, legacy.summarize = gt.train_arm, evaluate, lambda r: summary(r, source_report)
            legacy.epoch_groups = lambda *xs: next_groups(old['epoch_groups'], *xs)
            legacy.load_manifest, legacy.write_json, legacy.restore_checkpoint = manifest, write, restore
            legacy.stable_json_fingerprint = fingerprint
            legacy.prefetched_batches = prefetch_adapter(old['prefetched_batches'], 'clean', None, fields['seed'])
            joint_data.ego_inputs = gt.history_inputs
            print(f'CLEAN CONTINUE source={parent}; next500 groups; weights/Adam/RNG retained; new500 cosine.', flush=True)
            print('Printed updates are ADDITIONAL stage updates0--500; cumulative adaptation500--1000.', flush=True)
            status = legacy.main(event, args)
            result = json.loads((out/'train/results.json').read_text(encoding='utf-8'))
            old['write_json'](out/'results.json', result); (out/'summary.txt').write_text(summary(result, source_report), encoding='utf-8')
            return status
        finally:
            joint_data.ego_inputs = old_ego
            for key, value in old.items(): setattr(legacy, key, value)
            if legacy.sha256(source_path) != source_sha: raise RuntimeError('source500 checkpoint modified')


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
