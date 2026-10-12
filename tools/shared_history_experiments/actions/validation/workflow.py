"""Train ONLY REMOVE; DEV512 after both epochs; reuse old ADD-only weights."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock

BASE_PROTOCOL = 'p0_f9_shared_history_gt_soft_addonly_20percent_v1_shared'
CHECKS = ('train_keys', 'eval_keys', 'epochs', 'seed', 'window_batch', 'source_budget',
          'total_updates', 'lr', 'weights', 'schedule', 'data', 'cache_namespace', 'execution')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def launch_child(command, event):
    """Forward a parent-only SIGTERM/stop to the child's safe boundary."""
    process = subprocess.Popen(command)
    forwarded = False
    while True:
        try:
            return SimpleResult(process.wait(timeout=1.))
        except subprocess.TimeoutExpired:
            if event.is_set() and not forwarded:
                try: process.send_signal(signal.SIGINT)
                except ProcessLookupError: pass
                forwarded = True


class SimpleResult:
    def __init__(self, returncode): self.returncode = returncode


def report_complete(path, epoch):
    if not Path(path).is_file(): return False
    report = load_json(path)
    if report.get('status') not in ('complete', 'interim_read_only_evaluation'): return False
    if report['epoch'] != epoch or report['evaluation']['windows'] != 512:
        raise RuntimeError('epoch/DEV512 report mismatch')
    return True


def find_baseline(explicit, parent):
    candidates = [Path(explicit).resolve()] if explicit else sorted(Path(parent).glob('shared_20pct_2ep_*'))
    found = []
    for directory in candidates:
        path = directory/'results.json'
        if path.is_file():
            result = load_json(path)
            if (result.get('contract', {}).get('protocol') == BASE_PROTOCOL and report_complete(path, 2)
                    and (directory/'epoch_0001.pt').is_file()): found.append(directory)
    if len(found) != 1:
        raise RuntimeError(f'exactly ONE completed old Shared ADD-only baseline required; found {len(found)}; set ADD_ONLY_DIR')
    return found[0]


def build_comparison(root, baseline):
    root, baseline = Path(root), Path(baseline)
    comparisons, lines = [], ['===== REMOVE VALIDATION / TRAIN20% x2 / FIXED DEV512 =====',
        'ONLY ADD+REMOVE newly trained; previous ADD-only baseline reused, never retrained.',
        'Old baseline H2 vs new H6; head/input/initialization differ: NOT a strict single-variable ablation.',
        'Epoch1/epoch2 both reported; no best-epoch/route selection.']
    for epoch in (1, 2):
        new_path = root/'add_remove'/(f'eval_epoch{epoch}/results.json' if epoch == 1 else 'results.json')
        old_path = root/'baseline_epoch1/results.json' if epoch == 1 else baseline/'results.json'
        if not all(report_complete(path, epoch) for path in (new_path, old_path)): continue
        old, new = load_json(old_path), load_json(new_path)
        if old['contract'].get('protocol') != BASE_PROTOCOL:
            raise RuntimeError('wrong ADD-only baseline protocol')
        if any(old['contract'][k] != new['contract'][k] for k in CHECKS):
            raise RuntimeError('baseline population/budget/objective contracts differ')
        if old['updates'] != new['updates']: raise RuntimeError('epoch update counts differ')
        a, b = old['evaluation']['variants']['gt'], new['evaluation']['variants']['gt']
        delta = lambda x, y: {k: x[k]-y[k] for k in ('mIoU', 'IoU', 'MovingMicro')}
        rows = {'previous_ADDonly': a['soft_add_only'], 'new_transport': b['soft_transport'],
                'new_REMOVEoff': b['soft_add_only'], 'new_ADD_REMOVE': b['soft_add_remove']}
        comparisons.append(dict(epoch=epoch, updates=new['updates'], rows=rows,
            vs_previous_ADDonly=delta(b['soft_add_remove'], a['soft_add_only']),
            same_model_REMOVE_effect=delta(b['soft_add_remove'], b['soft_add_only']),
            source_results=dict(baseline=str(old_path), new=str(new_path))))
        lines += [f'epoch={epoch} update={new["updates"]}', 'route                    mIoU       IoU MovingMicro']
        for name, row in rows.items():
            lines.append(f'{name:23s} {row["mIoU"]:9.6f} {row["IoU"]:9.6f} {row["MovingMicro"]:9.6f}')
        for key in ('vs_previous_ADDonly', 'same_model_REMOVE_effect'):
            lines.append(key+' '+json.dumps(comparisons[-1][key]))
    result = dict(status='complete' if len(comparisons) == 2 else 'partial', epochs=comparisons,
        baseline_directory=str(baseline), new_training_arms=['add_remove'], strict_single_variable_ablation=False)
    legacy.write_json(root/'comparison.json', result)
    (root/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main(stop_event=None, argv=None, launch=None):
    event = stop_event or threading.Event()
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(epochs=2, mode='gt', eval_population='dev512')
    p.add_argument('--resume-workflow', action='store_true')
    p.add_argument('--add-only-dir', help='completed original Shared ADD-only run; otherwise auto-detect unique run')
    a = p.parse_args(argv)
    if (a.epochs != 2 or a.mode != 'gt' or a.eval_population != 'dev512' or a.evaluate_only
            or a.resume or a.max_updates or a.train_fraction != .2 or a.window_batch_size != 4
            or a.source_budget != 128 or a.seed != 20261011 or a.training_backend != 'fused'):
        raise ValueError('fixed GT / 20% x2 / DEV512 / H6 / batch4-source128; resume via --resume-workflow')
    root = Path(a.out_dir).resolve()
    if a.resume_workflow and (root/'workflow.json').is_file():
        previous = load_json(root/'workflow.json')
        baseline = find_baseline(a.add_only_dir or previous['baseline_directory'], root.parent)
    else: baseline = find_baseline(a.add_only_dir, root.parent)
    fields = vars(a).copy()
    for name in ('out_dir', 'resume_workflow', 'add_only_dir'): fields.pop(name)
    identity = dict(protocol='REMOVE_epoch_workflow_v1', args=fields, baseline_directory=str(baseline),
        baseline_inputs={name: legacy.sha256(baseline/name) for name in ('results.json', 'epoch_0001.pt')},
        inputs={k: legacy.sha256(getattr(a, k)) for k in ('config', 'train_cache', 'dev_cache',
            'population_manifest', 'base_checkpoint', 'train_info', 'dev_info')},
        implementation={f.name: legacy.sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))})
    if root.exists():
        if not a.resume_workflow or load_json(root/'workflow.json') != identity:
            raise RuntimeError('use NEW output, or SAME unchanged workflow with --resume-workflow')
    else:
        if a.resume_workflow: raise RuntimeError('missing workflow to resume')
        root.mkdir(parents=True); legacy.write_json(root/'workflow.json', identity)
    _, records = legacy.load_cache(a.train_cache)
    if len(records) != 20430: raise RuntimeError('TRAIN20430 required')
    keys = legacy.select_scene_balanced_round_robin(legacy.record_keys(records), 4086)
    chosen = legacy.align_records(records, keys)
    first_boundary = len(legacy.epoch_groups(chosen, a.seed, 0, 4, 128))
    total_updates = first_boundary + len(legacy.epoch_groups(chosen, a.seed, 1, 4, 128))
    old_contract = load_json(baseline/'results.json')['contract']
    manifest, _, _ = legacy.load_manifest(a.population_manifest)
    expected = dict(train_keys=[list(k) for k in keys], eval_keys=[list(k) for k in manifest['parent_keys']],
        epochs=2, seed=a.seed, window_batch=4, source_budget=128, total_updates=total_updates,
        lr=[a.motion_lr, a.repair_lr, a.ego_lr], weights=[a.object_weight, a.ego_weight, a.lovasz_weight],
        data={k: identity['inputs'][k] for k in ('train_cache', 'dev_cache', 'train_info', 'dev_info', 'base_checkpoint')})
    if any(old_contract[k] != value for k, value in expected.items()):
        raise RuntimeError('old ADD-only baseline differs from requested population/budget/objective; no new training started')
    del records, chosen
    base = []
    for name, value in fields.items():
        if name in ('resume', 'evaluate_only', 'max_updates', 'override') or value is None: continue
        if isinstance(value, bool):
            if value: base.append('--'+name.replace('_', '-'))
        else: base += ['--'+name.replace('_', '-'), str(value)]
    for value in (a.override or []): base += ['--override', value]

    def child(directory, *, resume=None, limit=None, evaluate=False, old=False):
        module = 'tools.shared_history_experiments.run' if old else 'tools.shared_history_experiments.actions.run'
        args = [sys.executable, '-u', '-m', module, *base, '--out-dir', str(directory)]
        if old:
            if not evaluate: raise RuntimeError('old ADD-only training is forbidden in this workflow')
            args += ['--variant', 'shared']
        if resume: args += ['--resume', str(resume)]
        if limit: args += ['--max-updates', str(limit)]
        if evaluate: args += ['--evaluate-only']
        completed = launch(args, check=False) if launch else launch_child(args, event)
        if completed.returncode != 0: raise RuntimeError(f'child failed: {completed.returncode}; saved checkpoint retained')

    lock = root.parent/'.shared_pilot_locks'/('workflow_'+legacy.stable_json_fingerprint(str(root)))
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        directory = root/'add_remove'; last = directory/'last.pt'
        saved = torch.load(last, map_location='cpu', weights_only=False) if last.is_file() else None
        if not event.is_set() and (saved is None or saved['updates'] < first_boundary):
            child(directory, resume=last if saved else None, limit=first_boundary)
        if not event.is_set():
            saved = torch.load(last, map_location='cpu', weights_only=False)
            if saved['epoch'] < 1: raise RuntimeError('training stopped before epoch1; resume same workflow')
            eval1 = directory/'eval_epoch1'
            if not report_complete(eval1/'results.json', 1):
                child(eval1, resume=directory/'epoch_0001.pt', evaluate=True)
            if not event.is_set() and report_complete(eval1/'results.json', 1):
                old_eval = root/'baseline_epoch1'
                if not report_complete(old_eval/'results.json', 1):
                    child(old_eval, resume=baseline/'epoch_0001.pt', evaluate=True, old=True)
                if not event.is_set() and report_complete(old_eval/'results.json', 1):
                    build_comparison(root, baseline)
                    print((root/'summary.txt').read_text(encoding='utf-8'), flush=True)
                    if not report_complete(directory/'results.json', 2):
                        child(directory, resume=last)
        result = build_comparison(root, baseline)
        print((root/'summary.txt').read_text(encoding='utf-8'), flush=True)
    return 0 if result['status'] == 'complete' or event.is_set() else 1


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
