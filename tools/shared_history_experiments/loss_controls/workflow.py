"""Three independent matched 500-update arms, one fixed DEV512 each."""
import json
from pathlib import Path
import signal
import sys
import threading
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.shared_history_experiments.actions.validation.workflow import launch_child
from .run import ARMS, STEPS, protocol


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def complete(path, arm):
    if not Path(path).is_file(): return False
    result = read(path)
    if result.get('status') != 'complete': return False
    if (result.get('updates') != STEPS or result.get('contract', {}).get('protocol') != protocol(arm)
            or result.get('evaluation', {}).get('windows') != 512):
        raise RuntimeError('wrong loss arm/budget/DEV512 receipt')
    return True


def summarize(root):
    rows = []
    lines = ['===== MATCHED RANDOM LOSS500 / FIXED DEV512 =====',
        'Same RANDOM initializer/Adam/order/batch4-source128/500-step cosine; no old weights.',
        'CE+Lovasz+ego retained. 500 updates is a partial TRAIN20% pass, NOT a full epoch.',
        'arm            route                 mIoU       IoU MovingMicro']
    for arm in ARMS:
        path = root/arm/'results.json'
        if not complete(path, arm): continue
        result = read(path)
        metrics = result['evaluation']['variants']['gt']
        rows.append(dict(arm=arm, updates=STEPS, weights=list(ARMS[arm]), metrics=metrics,
                         result=str(path), ego_errors=result['evaluation'].get('trajectory', {})))
        for name, m in metrics.items():
            lines.append(f'{arm:14s} {name:20s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:9.6f}')
    if rows and rows[0]['arm'] == 'full':
        baseline = rows[0]['metrics']['semantic_refine']
        for row in rows[1:]:
            delta = {k: row['metrics']['semantic_refine'][k]-baseline[k] for k in ('mIoU', 'IoU', 'MovingMicro')}
            lines.append(row['arm']+' minus full: '+json.dumps(delta))
    lines += ['No dev-best selection, retry, full training or automatic loss removal.',
              'This short screen cannot prove a loss unnecessary after convergence.']
    legacy.write_json(root/'comparison.json', dict(status='complete' if len(rows) == len(ARMS) else 'partial', arms=rows))
    (root/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return len(rows) == len(ARMS)


def main(event=None, argv=None, launch=None):
    event = event or threading.Event()
    p = legacy.parser(); p.set_defaults(mode='gt', epochs=1, eval_population='dev512')
    p.add_argument('--resume-workflow', action='store_true')
    a = p.parse_args(argv)
    if (a.mode != 'gt' or a.epochs != 1 or a.train_fraction != .2 or a.eval_population != 'dev512'
            or a.resume or a.evaluate_only or a.max_updates or a.window_batch_size != 4
            or a.source_budget != 128 or a.seed != 20261011 or a.training_backend != 'fused'
            or (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.)):
        raise ValueError('fixed three-arm loss500 controls; resume via --resume-workflow')
    root = Path(a.out_dir).resolve()
    fields = vars(a).copy()
    for name in ('out_dir', 'resume_workflow'): fields.pop(name)
    code_root = Path(__file__).resolve().parents[3]
    files = sorted(Path(__file__).parent.glob('*.py')) + sorted((Path(__file__).parent.parent/'refine').glob('*.py'))
    identity = dict(protocol='shared_refiner_loss500_workflow_v1', args=fields,
        inputs={k: legacy.sha256(getattr(a, k)) for k in ('config', 'train_cache', 'dev_cache',
            'population_manifest', 'base_checkpoint', 'train_info', 'dev_info')},
        implementation={str(f.relative_to(code_root)): legacy.sha256(f) for f in files})
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent/'.shared_pilot_locks'/('loss500_'+legacy.stable_json_fingerprint(str(root)))
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if root.exists():
            if not a.resume_workflow or read(root/'workflow.json') != identity:
                raise RuntimeError('use NEW output or SAME unchanged --resume-workflow')
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
        for arm in ARMS:
            if event.is_set(): break
            directory = root/arm
            if complete(directory/'results.json', arm): continue
            command = [sys.executable, '-u', '-m', 'tools.shared_history_experiments.loss_controls.run',
                *base, '--out-dir', str(directory), '--loss-arm', arm]
            if (directory/'last.pt').is_file(): command += ['--resume', str(directory/'last.pt')]
            child = launch(command, check=False) if launch else launch_child(command, event)
            summarize(root)
            if event.is_set(): return 0
            if child.returncode or not complete(directory/'results.json', arm):
                raise RuntimeError('child failed/incomplete; checkpoint retained, no automatic retry')
        return 0 if summarize(root) or event.is_set() else 1


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
