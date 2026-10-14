"""Sequential single-variable RANDOM known control, fixed final DEV64."""
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.shared_history_experiments.actions.validation.workflow import launch_child
from .data import POLICIES
from .evaluation import ROUTES
from .run import PROTOCOL, stc_inventory, validate, dependency_files


def parser():
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=1, train_fraction=.2, eval_population='dev64')
    p.add_argument('--stc-root', required=True)
    p.add_argument('--resume-workflow', action='store_true')
    return p


def completed(directory):
    path = directory/'results.json'
    if not path.is_file(): return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'complete': return None
    c = result['contract']; report = result['evaluation']
    if (c['protocol'] != PROTOCOL or result['epoch'] != 1 or c['epochs'] != 1 or report['windows'] != 64
            or result['updates'] != c['total_updates'] or report['history_known_policy'] != directory.name
            or set(report['variants']['gt']) != set(ROUTES)):
        raise RuntimeError('wrong completed known-policy arm')
    saved = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
    if (saved['contract'] != c or saved['updates'] != result['updates'] or saved['epoch'] != 1
            or saved['batch'] != 0 or saved['history_known_policy'] != directory.name):
        raise RuntimeError('completed report/checkpoint mismatch')
    return dict(contract=c, metrics=report['variants']['gt'], updates=result['updates'],
                random_initializer=saved['random_initializer'], checkpoint_sha256=legacy.sha256(directory/'last.pt'))


def comparison(root):
    rows = {p: result for p in POLICIES if (result := completed(root/p)) is not None}
    if len(rows) == 2:
        left, right = (rows[p] for p in POLICIES)
        if left['random_initializer'] != right['random_initializer']:
            raise RuntimeError('RANDOM initializers differ')
        if set(left['contract']) != set(right['contract']): raise RuntimeError('contract schema differs')
        for field in left['contract']:
            if field != 'implementation' and left['contract'][field] != right['contract'][field]:
                raise RuntimeError('unmatched known A/B contract: '+field)
    lines = ['===== MATCHED HISTORY KNOWN / TRAIN20% x1 / DEV64 =====',
        'Same RANDOM body/initializer/order/H6/full6 CE+Lovasz/cosine; no ego head or noise.',
        'Original GT lidar-known vs dense-valid TRAIN AND OCC eval. STC dense-valid in BOTH arms.',
        'Geometry/source/Strong construction unchanged; learned history coverage rebuilt.',
        'Full-grid metrics; no GT mask supplied to STC, alignment, prior gating or threshold tuning.',
        'policy       input   route                mIoU       IoU MovingMicro Refiner_dMiOU']
    for policy, result in rows.items():
        for modality in ('occ_gt', 'stc_gt'):
            base = result['metrics'][modality+'_transport']
            for readout in ('transport', 'semantic_refine'):
                m = result['metrics'][modality+'_'+readout]
                lines.append(f'{policy:12s} {modality:7s} {readout:17s} {m["mIoU"]:9.6f} '
                    f'{m["IoU"]:9.6f} {m["MovingMicro"]:11.6f} {m["mIoU"]-base["mIoU"]:+.6f}')
    delta = {}
    if len(rows) == 2:
        for name in ROUTES:
            delta[name] = {k: rows['dense_valid']['metrics'][name][k]-rows['lidar_known']['metrics'][name][k]
                           for k in ('mIoU', 'IoU', 'MovingMicro')}
        lines.append('dense_valid_minus_lidar_known='+json.dumps(delta))
    lines.append('Both fixed final epoch1 reported; NO automatic selection, expansion or promotion. NOT FPS.')
    legacy.write_json(root/'comparison.json', dict(status='complete' if len(rows) == 2 else 'partial',
        arms=rows, dense_valid_minus_lidar_known=delta, no_automatic_promotion=True))
    (root/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return len(rows) == 2


def main(event=None, argv=None, launch=None):
    event = event or threading.Event(); a = parser().parse_args(argv); validate(a)
    if a.resume or a.evaluate_only: raise ValueError('use --resume-workflow and SAME output')
    root = Path(a.out_dir).resolve(); fields = vars(a).copy()
    for name in ('out_dir', 'resume_workflow', 'max_updates'): fields.pop(name)
    code = Path(__file__).resolve().parents[2]
    identity = dict(protocol='history_known_matched_one_epoch_workflow_v1', args=fields,
        STC_inventory=stc_inventory(a),
        data={k: legacy.sha256(getattr(a, k)) for k in ('config', 'train_cache', 'dev_cache',
            'population_manifest', 'base_checkpoint', 'train_info', 'dev_info')},
        implementation={str(f.relative_to(code)): legacy.sha256(f) for f in dependency_files()})
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent/'.known_locks'/('workflow_'+legacy.stable_json_fingerprint(str(root)))
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
        for policy in POLICIES:
            if event.is_set(): comparison(root); return 0
            for path, size, timestamp in identity['STC_inventory']:
                st = Path(path).stat()
                if (st.st_size, st.st_mtime_ns) != (size, timestamp): raise RuntimeError('STC files changed')
            directory = root/policy
            if completed(directory): comparison(root); continue
            args = [sys.executable, '-u', '-m', 'tools.history_validity.run', *base,
                    '--known-policy', policy, '--out-dir', str(directory)]
            if a.max_updates: args += ['--max-updates', str(a.max_updates)]
            if (directory/'last.pt').is_file(): args += ['--resume', str(directory/'last.pt')]
            result = launch(args, check=False) if launch else launch_child(args, event)
            if result.returncode and not event.is_set(): raise RuntimeError('child failed; no automatic retry')
            comparison(root)
            if event.is_set(): return 0
            if not completed(directory) and not a.max_updates: return 0
        return 0 if comparison(root) or a.max_updates else 1


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
