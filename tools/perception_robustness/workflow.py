"""Two independent matched RANDOM arms; run sequentially; fixed final DEV64."""
import json
from pathlib import Path
import signal
import sys
import threading
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.shared_history_experiments.actions.validation.workflow import launch_child
from .augmentation import RECIPE
from .run import PROTOCOL, stc_inventory


def parser():
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=.2, eval_population='dev64')
    p.add_argument('--stc-root', required=True)
    p.add_argument('--resume-workflow', action='store_true')
    return p


def comparison(root):
    rows = {}
    for arm in ('clean', 'robust'):
        path = root/arm/'results.json'
        if not path.is_file(): continue
        result = json.loads(path.read_text(encoding='utf-8'))
        if result.get('status') != 'complete': continue
        c = result['contract']; report = result['evaluation']
        if (c['protocol'] != PROTOCOL or result['epoch'] != 2 or report['windows'] != 64
                or result['updates'] != c['total_updates']): raise RuntimeError('wrong completed arm')
        rows[arm] = dict(contract=c, metrics=report['variants']['gt'], updates=result['updates'])
    if len(rows) == 2:
        left, right = rows['clean']['contract'], rows['robust']['contract']
        # Only augmentation/implementation fingerprint may differ. Initialization,
        # population, grouping, loss, clock and optimizer schedules MUST match.
        for field in left:
            if field != 'implementation' and left[field] != right[field]:
                raise RuntimeError('unmatched A/B contract: '+field)
    lines = ['===== MATCHED PERCEPTION ROBUSTNESS / TRAIN20% x2 =====',
        'Clean vs structured HISTORY corruption; same RANDOM seed/body/order/loss/cosine.',
        'GT future ego conditioned; no ego-head/planner adaptation. Fixed final epoch2.',
        'Real STC only in held-out evaluation; full grid, all-known STC; no GT masks/alignment.',
        'arm    input    route               mIoU       IoU MovingMicro    Refiner_dMiOU']
    for arm, result in rows.items():
        for modality in ('occ_gt', 'stc_gt'):
            base = result['metrics'][modality+'_transport']
            for readout in ('transport', 'semantic_refine'):
                m = result['metrics'][modality+'_'+readout]
                lines.append(f'{arm:6s} {modality:8s} {readout:16s} {m["mIoU"]:9.6f} '
                    f'{m["IoU"]:9.6f} {m["MovingMicro"]:11.6f} {m["mIoU"]-base["mIoU"]:+.6f}')
    delta = {}
    if len(rows) == 2:
        for name in rows['clean']['metrics']:
            delta[name] = {k: rows['robust']['metrics'][name][k]-rows['clean']['metrics'][name][k]
                           for k in ('mIoU', 'IoU', 'MovingMicro')}
        lines += ['robust_minus_clean='+json.dumps(delta),
                  'All results reported; no automatic selection, retry, full training or deployment.']
    value = dict(status='complete' if len(rows) == 2 else 'partial', arms=rows, robust_minus_clean=delta,
                 no_automatic_promotion=True, recipe=RECIPE.specification())
    legacy.write_json(root/'comparison.json', value)
    (root/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return len(rows) == 2


def main(event=None, argv=None, launch=None):
    event = event or threading.Event(); a = parser().parse_args(argv)
    if (a.mode != 'gt' or a.epochs != 2 or a.train_fraction != .2 or a.eval_population != 'dev64'
            or a.window_batch_size != 4 or a.source_budget != 128 or a.seed != 20261011
            or a.resume or a.evaluate_only or a.training_backend != 'fused'):
        raise ValueError('fixed TRAIN20% x2 GT A/B; resume via --resume-workflow')
    root = Path(a.out_dir).resolve(); fields = vars(a).copy()
    for name in ('out_dir', 'resume_workflow', 'max_updates'): fields.pop(name)
    code = Path(__file__).resolve().parents[2]
    identity = dict(protocol='perception_robustness_matched_workflow_v1', args=fields,
        recipe=RECIPE.specification(),
        STC_inventory=stc_inventory(a),
        data={k: legacy.sha256(getattr(a, k)) for k in ('config', 'train_cache', 'dev_cache',
            'population_manifest', 'base_checkpoint', 'train_info', 'dev_info')},
        implementation={str(f.relative_to(code)): legacy.sha256(f) for f in Path(__file__).parent.glob('*.py')})
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent/'.robustness_locks'/('workflow_'+legacy.stable_json_fingerprint(str(root)))
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if root.exists():
            if not a.resume_workflow or json.loads((root/'workflow.json').read_text(encoding='utf-8')) != identity:
                raise RuntimeError('NEW output, or SAME unchanged workflow with --resume-workflow')
        else:
            if a.resume_workflow: raise RuntimeError('missing workflow to resume')
            root.mkdir(); legacy.write_json(root/'workflow.json', identity)
        base = []
        for key, value in fields.items():
            if key in ('resume', 'evaluate_only', 'override') or value is None: continue
            if isinstance(value, bool):
                if value: base += ['--'+key.replace('_', '-')]
            else: base += ['--'+key.replace('_', '-'), str(value)]
        for value in a.override or []: base += ['--override', value]
        for arm in ('clean', 'robust'):
            if event.is_set(): comparison(root); return 0
            for path, size, timestamp in identity['STC_inventory']:
                stat = Path(path).stat()
                if (stat.st_size, stat.st_mtime_ns) != (size, timestamp):
                    raise RuntimeError('STC data changed between matched arms')
            directory = root/arm; result_path = directory/'results.json'
            if result_path.is_file() and json.loads(result_path.read_text(encoding='utf-8')).get('status') == 'complete':
                comparison(root); continue
            args = [sys.executable, '-u', '-m', 'tools.perception_robustness.run', *base,
                    '--augmentation', arm, '--out-dir', str(directory)]
            if a.max_updates: args += ['--max-updates', str(a.max_updates)]
            if (directory/'last.pt').is_file(): args += ['--resume', str(directory/'last.pt')]
            result = launch(args, check=False) if launch else launch_child(args, event)
            if result.returncode and not event.is_set(): raise RuntimeError('child failed; prior checkpoint preserved; no automatic retry')
            done = comparison(root)
            if event.is_set(): return 0
            if not result_path.is_file() or json.loads(result_path.read_text(encoding='utf-8')).get('status') != 'complete':
                if a.max_updates: continue  # Smoke BOTH arms, retaining each original two-epoch schedule.
                return 0  # other safe boundary: resume SAME workflow.
        return 0 if comparison(root) or a.max_updates else 1


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
