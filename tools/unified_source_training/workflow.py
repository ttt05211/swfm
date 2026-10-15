"""Two independent matched RANDOM arms, one fixed epoch each, final DEV64."""
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.shared_history_experiments.actions.validation.workflow import launch_child
from .data import ARMS
from .evaluation import ROUTES
from .train import PROTOCOL, stc_inventory, validate, dependency_files


def parser():
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=1, train_fraction=.2, eval_population='dev64')
    p.add_argument('--stc-root', required=True); p.add_argument('--resume-workflow', action='store_true')
    return p


def completed(directory):
    path = directory/'results.json'
    if not path.is_file(): return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'complete': return None
    c, report = result['contract'], result['evaluation']
    if (c['protocol'] != PROTOCOL or c['epochs'] != 1 or result['epoch'] != 1 or report['windows'] != 64
            or report['track_arm'] != directory.name or result['updates'] != c['total_updates']
            or set(report['variants']['gt']) != set(ROUTES)):
        raise RuntimeError('wrong completed track arm')
    saved = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
    if (saved['contract'] != c or saved['epoch'] != 1 or saved['batch'] != 0
            or saved['updates'] != result['updates'] or saved['track_arm'] != directory.name):
        raise RuntimeError('completed report/checkpoint differs')
    return dict(contract=c, metrics=report['variants']['gt'], updates=result['updates'],
        random_initializer=saved['random_initializer'], checkpoint_sha256=legacy.sha256(directory/'last.pt'))


def comparison(root):
    rows = {arm: value for arm in ARMS if (value := completed(root/arm)) is not None}
    if len(rows) == 2:
        a, b = (rows[k] for k in ARMS)
        if a['random_initializer'] != b['random_initializer']: raise RuntimeError('RANDOM initializers differ')
        if set(a['contract']) != set(b['contract']): raise RuntimeError('arm contract schemas differ')
        for field in a['contract']:
            if field != 'implementation' and a['contract'][field] != b['contract'][field]:
                raise RuntimeError('unmatched arm contract: '+field)
    lines = ['===== MATCHED RANDOM TRACK/KTA / TRAIN20% x1 EACH =====',
        'Same RANDOM initializer/body/order/batch4-source128/H6/full6 CE+Lovasz/cosine.',
        'baseline vs unified track + multi-frame KTA, used consistently in TRAIN and EVAL.',
        'Original Strong/CLEAR background algorithm fixed; no ego head or auxiliary losses.',
        'Same fixed DEV64 OCC+GT/STC+GT; full grid, STC all-known, no mask/alignment.',
        'arm input route mIoU IoU MovingMicro Refiner_dMiOU']
    for arm, value in rows.items():
        for kind in ('occ_gt', 'stc_gt'):
            base = value['metrics'][kind+'_transport']
            for route in ('transport', 'semantic_refine'):
                m = value['metrics'][kind+'_'+route]
                lines.append(f'{arm:12s} {kind:7s} {route:16s} {m["mIoU"]:.6f} {m["IoU"]:.6f} '
                    f'{m["MovingMicro"]:.6f} {m["mIoU"]-base["mIoU"]:+.6f}')
    delta = {}
    if len(rows) == 2:
        delta = {n: {k: rows['unified_fit']['metrics'][n][k]-rows['baseline']['metrics'][n][k]
                     for k in ('mIoU', 'IoU', 'MovingMicro')} for n in ROUTES}
        lines.append('unified_fit_minus_baseline='+json.dumps(delta))
    lines.append('Fixed final epoch1 only; no automatic selection, retry, expansion or promotion. Quality time NOT FPS.')
    legacy.write_json(root/'comparison.json', dict(status='complete' if len(rows) == 2 else 'partial',
        arms=rows, unified_fit_minus_baseline=delta, no_automatic_promotion=True))
    (root/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8'); print('\n'.join(lines), flush=True)
    return len(rows) == 2


def main(event=None, argv=None, launch=None):
    event = event or threading.Event(); a = parser().parse_args(argv); validate(a)
    if a.resume or a.evaluate_only: raise ValueError('use SAME workflow and --resume-workflow')
    root = Path(a.out_dir).resolve(); fields = vars(a).copy()
    for k in ('out_dir', 'resume_workflow', 'max_updates'): fields.pop(k)
    code = Path(__file__).resolve().parents[2]
    identity = dict(protocol='matched_track_KTA_one_epoch_workflow_v1', args=fields, STC_inventory=stc_inventory(a),
        data={k: legacy.sha256(getattr(a, k)) for k in ('config', 'train_cache', 'dev_cache', 'population_manifest',
            'base_checkpoint', 'train_info', 'dev_info')},
        implementation={str(f.relative_to(code)): legacy.sha256(f) for f in dependency_files()})
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent/'.track_train_locks'/('workflow_'+legacy.stable_json_fingerprint(str(root)))
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if root.exists():
            if not a.resume_workflow or json.loads((root/'workflow.json').read_text(encoding='utf-8')) != identity:
                raise RuntimeError('NEW directory, or SAME unchanged workflow with --resume-workflow')
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
        for arm in ARMS:
            if event.is_set(): comparison(root); return 0
            for path, size, timestamp in identity['STC_inventory']:
                st = Path(path).stat()
                if (st.st_size, st.st_mtime_ns) != (size, timestamp): raise RuntimeError('STC inventory changed')
            directory = root/arm
            if completed(directory): comparison(root); continue
            args = [sys.executable, '-u', '-m', 'tools.unified_source_training.train', *base,
                    '--track-arm', arm, '--out-dir', str(directory)]
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
