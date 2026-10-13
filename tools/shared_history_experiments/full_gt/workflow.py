"""Fresh TRAIN20430 x2; fixed DEV512 after each epoch; resumable workflow."""
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.shared_history_experiments.actions.validation.workflow import launch_child
from .model import PROTOCOL


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def complete(path, epoch):
    if not Path(path).is_file(): return False
    result = read(path)
    if result.get('status') not in ('complete', 'interim_read_only_evaluation'): return False
    if (result.get('epoch') != epoch or result.get('contract', {}).get('protocol') != PROTOCOL
            or result.get('evaluation', {}).get('windows') != 512):
        raise RuntimeError('wrong Refiner epoch/protocol/DEV512 report')
    return True


def summary(root):
    rows = []
    for epoch, path in ((1, root/'train/eval_epoch1/results.json'), (2, root/'train/results.json')):
        if complete(path, epoch):
            result = read(path)
            rows.append(dict(epoch=epoch, update=result['updates'],
                metrics=result['evaluation']['variants']['gt'], result=str(path)))
    lines = ['===== SEMANTIC REFINER / TRAIN20430 x2 / FIXED DEV512 =====',
        'Fresh RANDOM; no ego head or old weights/Adam imported; original H6.',
        'Both epochs reported; fixed final epoch2, no best-epoch selection; full6 CE+Lovasz ONLY.',
        'epoch route                 mIoU       IoU MovingMicro']
    for row in rows:
        for name, values in row['metrics'].items():
            lines.append(f'{row["epoch"]:5d} {name:20s} {values["mIoU"]:9.6f} '
                f'{values["IoU"]:9.6f} {values["MovingMicro"]:9.6f}')
    legacy.write_json(root/'comparison.json', dict(status='complete' if len(rows) == 2 else 'partial', epochs=rows))
    (root/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return len(rows) == 2


def main(event=None, argv=None, launch=None):
    event = event or threading.Event()
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=1., eval_population='dev512')
    p.add_argument('--resume-workflow', action='store_true')
    a = p.parse_args(argv)
    if (a.mode != 'gt' or a.epochs != 2 or a.train_fraction != 1. or a.eval_population != 'dev512'
            or a.resume or a.evaluate_only or a.max_updates or a.window_batch_size != 4
            or a.source_budget != 128 or a.seed != 20261011 or a.training_backend != 'fused'):
        raise ValueError('fixed GT/TRAIN20430 x2/full6/batch4-source128; resume via --resume-workflow')
    if (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.):
        raise ValueError('CE/Lovasz weight1; auxiliary compatibility arguments cannot change objective')
    root = Path(a.out_dir).resolve()
    fields = vars(a).copy()
    for name in ('out_dir', 'resume_workflow'): fields.pop(name)
    code_root = Path(__file__).resolve().parents[3]
    files = sorted(Path(__file__).resolve().parent.glob('*.py'))
    identity = dict(protocol='full_gt_ce_lovasz_epoch_workflow_v1', args=fields,
        inputs={k: legacy.sha256(getattr(a, k)) for k in ('config', 'train_cache', 'dev_cache',
            'population_manifest', 'base_checkpoint', 'train_info', 'dev_info')},
        implementation={str(f.relative_to(code_root)): legacy.sha256(f) for f in files})
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent/'.shared_pilot_locks'/('refiner_workflow_'+legacy.stable_json_fingerprint(str(root)))
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if root.exists():
            if not a.resume_workflow or read(root/'workflow.json') != identity:
                raise RuntimeError('use NEW output, or SAME unchanged workflow with --resume-workflow')
        else:
            if a.resume_workflow: raise RuntimeError('missing workflow to resume')
            root.mkdir(); legacy.write_json(root/'workflow.json', identity)
        base = []
        for key, value in fields.items():
            if key in ('resume', 'evaluate_only', 'max_updates', 'override') or value is None: continue
            if isinstance(value, bool):
                if value: base += ['--'+key.replace('_', '-')]
            else: base += ['--'+key.replace('_', '-'), str(value)]
        for value in a.override or []: base += ['--override', value]
        def child(module, directory, extra=()):
            command = [sys.executable, '-u', '-m', module, *base, '--out-dir', str(directory), *extra]
            result = launch(command, check=False) if launch else launch_child(command, event)
            if result.returncode and not event.is_set():
                raise RuntimeError(f'child failed ({result.returncode}); no automatic retry; resume checkpoint retained')
        if event.is_set(): return 0
        _, records = legacy.load_cache(a.train_cache)
        if len(records) != 20430: raise RuntimeError('TRAIN20430 required')
        keys = legacy.select_scene_balanced_round_robin(legacy.record_keys(records), 20430)
        boundary = len(legacy.epoch_groups(legacy.align_records(records, keys), a.seed, 0, 4, 128))
        del records
        directory = root/'train'; last = directory/'last.pt'
        saved = torch.load(last, map_location='cpu', weights_only=False) if last.is_file() else None
        if saved and saved.get('protocol') != PROTOCOL: raise RuntimeError('wrong training checkpoint')
        module = 'tools.shared_history_experiments.full_gt.run'
        if not saved or saved['updates'] < boundary:
            extra = ['--max-updates', str(boundary)]
            if saved: extra += ['--resume', str(last)]
            child(module, directory, extra)
        if event.is_set(): return 0
        if not (directory/'epoch_0001.pt').is_file(): raise RuntimeError('missing exact epoch1 snapshot')
        if not complete(directory/'eval_epoch1/results.json', 1):
            child(module, directory/'eval_epoch1', ['--resume', str(directory/'epoch_0001.pt'), '--evaluate-only'])
        if event.is_set(): summary(root); return 0
        summary(root)
        if not complete(directory/'results.json', 2): child(module, directory, ['--resume', str(last)])
        done = summary(root)
        return 0 if done or event.is_set() else 1


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
