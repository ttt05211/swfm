"""Two matched warm-start arms, fixed 500 steps, fixed remaining-scene64."""
import json
from pathlib import Path
import shutil
import signal
import sys
import threading
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.shared_history_experiments.actions.validation.workflow import launch_child
from .augmentation import load_model, RECIPE
from .run import PROTOCOL, selection, validate_args, parser as arm_parser


def parser():
    p = arm_parser()
    p.set_defaults(augmentation='clean')
    for action in p._actions:
        if action.dest == 'augmentation': action.required = False
    p.add_argument('--resume-workflow', action='store_true')
    return p


def comparison(root):
    rows = {}
    for arm in ('clean', 'empirical'):
        path = root/arm/'results.json'
        if not path.is_file(): continue
        r = json.loads(path.read_text(encoding='utf-8'))
        if r.get('status') != 'complete': continue
        c = r['contract']; report = r['evaluation']
        if (c['protocol'] != PROTOCOL or r['updates'] != 500 or c['total_updates'] != 500
                or report['windows'] != 64 or c['augmentation'] != arm
                or report['evaluation_keys'] != c['eval_keys']): raise RuntimeError('invalid completed adaptation')
        rows[arm] = dict(contract=c, metrics=report['variants']['gt'], updates=r['updates'])
    if len(rows) == 2:
        for field in rows['clean']['contract']:
            if field != 'augmentation' and rows['clean']['contract'][field] != rows['empirical']['contract'][field]:
                raise RuntimeError('unmatched clean/noise contract: '+field)
        audits = [json.loads((root/a/'initialization_audit.json').read_text(encoding='utf-8')) for a in rows]
        if audits[0] != audits[1]: raise RuntimeError('different warm initializers')
    lines = ['===== MATCHED VAL-CALIBRATED HISTORY / REFINER500 =====',
        'Both GTepoch2 initialized; frozen Encoder/STWM; fresh Adam; ONLY full6 CE+Lovasz.',
        'Both TRAIN dense-valid; clean vs 50% empirical historical noise; c=1 unchanged.',
        'SAME fixed remaining64, NOT old DEV64/512, NOT a pristine independent test.',
        'VAL fitting information used for augmentation; NOT target-domain-information-free zero-shot.',
        'arm       input   route                  mIoU       IoU MovingMicro Refiner_dMiOU']
    for arm, r in rows.items():
        for modality in ('occ_gt', 'stc_gt'):
            base = r['metrics'][modality+'_transport']
            for route in ('transport', 'semantic_refine'):
                m = r['metrics'][modality+'_'+route]
                lines.append(f'{arm:9s} {modality:7s} {route:18s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} '
                             f'{m["MovingMicro"]:11.6f} {m["mIoU"]-base["mIoU"]:+.6f}')
    delta = {}
    if len(rows) == 2:
        for name in rows['clean']['metrics']:
            delta[name] = {k: rows['empirical']['metrics'][name][k]-rows['clean']['metrics'][name][k]
                          for k in ('mIoU', 'IoU', 'MovingMicro')}
        lines += ['empirical_minus_clean='+json.dumps(delta)]
    lines += ['No automatic promotion, retry, coefficient/threshold search or larger training. NOT FPS.']
    legacy.write_json(root/'comparison.json', dict(status='complete' if len(rows) == 2 else 'partial',
        arms=rows, empirical_minus_clean=delta, no_automatic_promotion=True, recipe=RECIPE))
    (root/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return len(rows) == 2


def main(event=None, argv=None, launch=None):
    event = event or threading.Event(); a = parser().parse_args(argv); validate_args(a)
    if a.resume or a.augmentation != 'clean': raise ValueError('paired workflow; use --resume-workflow only')
    noise, binding = load_model(a.noise_model)
    keys, inventory = selection(a, noise)
    if legacy.sha256(a.gt_checkpoint) != binding['source_GT_sha256']: raise RuntimeError('GT/profile mismatch')
    fields = vars(a).copy()
    for name in ('out_dir', 'resume_workflow', 'max_updates', 'augmentation'): fields.pop(name)
    root = Path(a.out_dir).resolve(); code = Path(__file__).resolve().parents[2]
    identity = dict(protocol=PROTOCOL, args=fields, recipe=RECIPE, noise_identity=binding,
        eval_keys=keys, STC_inventory=inventory,
        data={k: legacy.sha256(getattr(a, k)) for k in ('config', 'train_cache', 'dev_cache',
            'population_manifest', 'base_checkpoint', 'train_info', 'dev_info')},
        implementation={str(f.relative_to(code)): legacy.sha256(f) for f in Path(__file__).parent.glob('*.py')})
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent/'.empirical_locks'/('workflow_'+legacy.stable_json_fingerprint(str(root)))
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if root.exists():
            if not a.resume_workflow or json.loads((root/'workflow.json').read_text(encoding='utf-8')) != identity:
                raise RuntimeError('NEW output, or SAME unchanged workflow with --resume-workflow')
        else:
            if a.resume_workflow: raise RuntimeError('missing workflow to resume')
            source = Path(a.gt_checkpoint).resolve()
            if root == source.parent or source.parent in root.parents: raise ValueError('output outside source GT directory')
            root.mkdir(); (root/'profile').mkdir()
            for name in ('noise_model.json', 'evaluation.json'):
                shutil.copy2(Path(a.noise_model).with_name(name), root/'profile'/name)
            copied, copied_binding = load_model(root/'profile/noise_model.json')
            if copied_binding != binding: raise RuntimeError('profile changed during snapshot')
            legacy.write_json(root/'workflow.json', identity)
        # Revalidate immutable profile snapshot even when both arms finished.
        if load_model(root/'profile/noise_model.json')[1] != binding: raise RuntimeError('profile snapshot changed')
        base = []
        for key, value in fields.items():
            if key in ('resume', 'evaluate_only', 'override') or value is None: continue
            if key == 'noise_model': value = str(root/'profile/noise_model.json')
            if isinstance(value, bool):
                if value: base += ['--'+key.replace('_', '-')]
            else: base += ['--'+key.replace('_', '-'), str(value)]
        for value in a.override or []: base += ['--override', value]
        for arm in ('clean', 'empirical'):
            if event.is_set(): comparison(root); return 0
            for path, size, timestamp in inventory:
                st = Path(path).stat()
                if (st.st_size, st.st_mtime_ns) != (size, timestamp): raise RuntimeError('STC histories changed between arms')
            directory = root/arm; report = directory/'results.json'
            if report.is_file() and json.loads(report.read_text(encoding='utf-8')).get('status') == 'complete':
                comparison(root); continue
            args = [sys.executable, '-u', '-m', 'tools.stc_noise_adaptation.run', *base,
                    '--augmentation', arm, '--out-dir', str(directory)]
            if a.max_updates: args += ['--max-updates', str(a.max_updates)]
            if (directory/'train/last.pt').is_file(): args += ['--resume', str(directory/'train/last.pt')]
            result = launch(args, check=False) if launch else launch_child(args, event)
            if result.returncode and not event.is_set(): raise RuntimeError('arm failed; checkpoint preserved; no automatic retry')
            comparison(root)
            if event.is_set(): return 0
            if not report.is_file() or json.loads(report.read_text(encoding='utf-8')).get('status') != 'complete':
                if a.max_updates: continue
                return 0
        return 0 if comparison(root) or a.max_updates else 1


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
