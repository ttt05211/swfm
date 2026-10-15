"""Two explicitly requested fixed four-checkpoint recipes; single-card evaluation only."""
import argparse
import csv
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as base
from tools.final_semantic_evaluation.engine import evaluate
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.history_validity.data import policy_raw
from tools.shared_source_track.adapter import SharedTrackPredictor
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .weights import (PROTOCOL, RECIPES, METRICS, model_from_spec, select_directories,
                      load_sources, historical_report, averaged_payload)


class DenseModalInputs:
    def __init__(self, inputs, kind): self.inputs, self.kind = inputs, kind
    def history(self, job, _kind='occ'):
        record, raw, times = self.inputs.history(job, self.kind)
        return record, policy_raw(raw, 'dense_valid'), times
    def poses(self, job): return self.inputs.poses(job)
    def targets(self, job, indices): return self.inputs.targets(job, indices)
    def moving(self, *args): return self.inputs.moving(*args)


def fmt(value): return 'NA' if value is None else f'{value:.6f}'


def comparison_text(result):
    steps = result['contract']['source_steps']
    lines = ['===== FIXED FOUR-CHECKPOINT PARAMETER MEAN / SAME REMAINING64 =====',
        'status='+result['status'], 'Equal weights 1/4 each:'+','.join(str(s)+('(epoch2)' if s == 10292 else '')
            for s in steps)+'; single-model inference.',
        'No training/Adam averaging/threshold search/automatic selection; not guaranteed to improve.',
        'MATCHED: OCC AND STC dense-valid; GT future ego; full-grid; 1/2/3s average.',
        'Existing complete compatible metrics reused; missing/wrong-known routes evaluated with fresh shared-track inference.']
    for table, rows in (('HISTORICAL AS RECORDED (known explicitly labeled; NOT all directly comparable)', result['historical']),
                        ('MATCHED DENSE-VALID COMPARISON', result['matched'])):
        lines += ['', table, '| model | OCC known | route | OCC mIoU | OCC IoU | OCC Moving | STC mIoU | STC IoU | STC Moving |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for name, row in rows.items():
            for route in ('semantic_refine', 'transport'):
                values = [row.get('metrics', {}).get(kind+'_'+route, {}).get(k)
                          for kind in ('occ_gt', 'stc_gt') for k in METRICS]
                lines.append('| '+ ' | '.join([name, row.get('known', {}).get('occ', 'missing'), route]+list(map(fmt, values)))+' |')
    lines += ['', 'Original checkpoints/reports/caches read only. Fresh vs cached inference is NOT a byte-identity claim.',
              'DEV64 has been inspected repeatedly; this is development, NOT independent test or FPS.']
    return '\n'.join(lines)+'\n'


def publish(out, result):
    base.write_json(out/'results.json', result)
    text = comparison_text(result)
    (out/'summary.txt').write_text(text, encoding='utf-8')
    (out/'comparison.md').write_text(text, encoding='utf-8')
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f); writer.writerow(['table', 'checkpoint', 'OCC_known', 'STC_known', 'route', *METRICS, 'origin'])
        for table in ('historical', 'matched'):
            for name, row in result[table].items():
                for route, metric in row.get('metrics', {}).items():
                    writer.writerow([table, name, row['known']['occ'], row['known']['stc'], route,
                        *[metric.get(k) for k in METRICS], row.get('origins', {}).get(route, row.get('path', ''))])


def main(event=None, argv=None):
    event = event or threading.Event(); p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epoch3-dir'); p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True)
    p.add_argument('--recipe', choices=tuple(RECIPES), default='recent')
    p.add_argument('--resume', action='store_true'); p.add_argument('--config', required=True)
    p.add_argument('--dataroot', required=True); p.add_argument('--dev-info', required=True)
    p.add_argument('--dev-cache', required=True); p.add_argument('--stc-root', required=True)
    p.add_argument('--cpu-workers', type=int, default=4); p.add_argument('--device', default='cuda')
    a = p.parse_args(argv); repo = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    steps = RECIPES[a.recipe]
    if not 1 <= a.cpu_workers <= 8: raise ValueError('bounded CPU workers required')
    if a.device.startswith('cuda') and torch.cuda.device_count() != 1: raise RuntimeError('expose exactly ONE GPU')
    torch.set_num_threads(1); torch.manual_seed(21)
    if a.resume and not a.epoch3_dir:
        a.epoch3_dir = json.loads((out/'contract.json').read_text(encoding='utf-8'))['epoch3_dir']
    stage, parent, stage_c = select_directories(a.epoch3_dir, a.source_dir, repo/'outputs/p0_f9_shared_history')
    if any(out == old or old in out.parents for old in (stage, parent)):
        raise ValueError('independent output outside BOTH training runs required')
    sources = load_sources(stage, parent, stage_c, steps=steps)
    original = stage_c['source_contract']; keys = stage_c['eval_keys']
    for name in ('config', 'dev_cache', 'dev_info'):
        if base.sha256(getattr(a, name)) != original['data'][name]: raise RuntimeError('original metadata differs: '+name)
    for c in (original, stage_c):
        for name, sha in c['implementation'].items():
            if base.sha256(repo/name) != sha: raise RuntimeError('model/runtime implementation changed: '+name)
    _, records = base.load_cache(a.dev_cache); chosen = base.align_records(records, keys)
    jobs = [window(r) for r in chosen]
    inputs = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
    inventory = {kind: inputs.preflight(jobs, kind) for kind in ('occ', 'stc')}
    for path, size, stamp in original['STC_inventory']:
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, stamp): raise RuntimeError('original STC history changed: '+path)
    historical = {str(s['step']): historical_report(s) for s in sources}
    # Continuation already re-evaluated epoch2 with the correct dense policy.
    # Prefer that record, instead of re-running epoch2 or reusing its old lidar-known score.
    source2 = next(s for s in sources if s['step'] == 10292)
    dense_epoch2 = historical_report(dict(source2, path=str(stage/'source_epoch2.pt'), contract=stage_c, evaluations={}))
    if dense_epoch2 is not None: historical['10292'] = dense_epoch2
    body = model_from_spec(stage_c['model'])
    args = vars(a).copy(); args.pop('resume'); args.update(epoch3_dir=str(stage), source_dir=str(parent), out_dir=str(out))
    files = sorted(Path(__file__).parent.glob('*.py'))
    contract = dict(protocol=PROTOCOL, args=args, epoch3_dir=str(stage), source_steps=list(steps), keys=keys,
        sources=[{k: s[k] for k in ('step', 'path', 'sha256')} for s in sources], coefficients=[.25]*4,
        source_contract=original, continuation_contract=stage_c,
        known=dict(occ='dense_valid', stc='dense_valid'), historical=historical, inventory=inventory,
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
        implementation={str(f.relative_to(repo)): base.sha256(f) for f in files})
    contract = json.loads(json.dumps(contract))
    lock = out.parent/'.dense_mean_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if a.resume:
            if json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract:
                raise RuntimeError('same unchanged averaging/evaluation run required')
        else:
            out.mkdir(parents=True, exist_ok=False)
            base.write_json(out/'contract.json', contract)
            save_resume_checkpoint(out/'mean4.pt', averaged_payload(sources, body, contract))
        mean = torch.load(out/'mean4.pt', map_location='cpu', weights_only=False)
        if mean.get('protocol') != PROTOCOL or mean['contract'] != contract or mean.get('resume_allowed') is not False:
            raise RuntimeError('averaged evaluation-only artifact differs')
        result = dict(status='running', scientific_updates=0, contract=contract,
            historical={str(s): historical[str(s)] if historical[str(s)] else
                        dict(known=dict(occ='missing', stc='missing'), metrics={}) for s in steps}, matched={})
        if a.resume: result = json.loads((out/'results.json').read_text(encoding='utf-8'))
        result['status'] = 'running'; publish(out, result)
        pcfg = base.make_prepare_config(base.load_runtime_config(a.config, original.get('overrides')))
        candidates = [(str(s['step']), s['state']) for s in sources]+[('mean4', mean['models']['gt'])]
        print(f'DENSE_MEAN4 source={parent}; epoch3={stage}; same64/OCC+STC; ONE GPU; output={out}', flush=True)
        try:
            for name, state in candidates:
                row = result['matched'].setdefault(name, dict(known=dict(occ='dense_valid', stc='dense_valid'), metrics={}, origins={}))
                archived = historical.get(name)
                for kind, prefix in (('occ', 'occ_gt'), ('stc', 'stc_gt')):
                    routes = [prefix+'_'+r for r in ('transport', 'semantic_refine')]
                    if all(r in row['metrics'] for r in routes): continue
                    if archived and archived['known'][kind] == 'dense_valid':
                        for r in routes:
                            row['metrics'][r] = archived['metrics'][r]; row['origins'][r] = 'reused: '+archived['path']
                        print(f'REUSE {name}/{kind} complete matching dense-valid64', flush=True)
                    else:
                        if event.is_set(): raise InterruptedError('saved at complete evaluation window')
                        body.cpu(); body.load_state_dict(state, strict=True); body.to(a.device).eval().requires_grad_(False)
                        predictor = SharedTrackPredictor(body, pcfg, a.device, workers=a.cpu_workers)
                        before = tensor_state_fingerprint(body.state_dict()); state_path = out/f'{name}_{kind}_state.json'
                        old = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else None
                        print(f'EVALUATE {name}/{kind}: missing or wrong-known report; dense-valid64', flush=True)
                        try:
                            with torch.no_grad():
                                report = evaluate(predictor, DenseModalInputs(inputs, kind), jobs, 'gt', keys,
                                    dict(base=contract, model=name, modality=kind), saved=old,
                                    save=lambda value: base.write_json(state_path, value), stop_event=event)
                            for r in routes:
                                m = report['reports']['gt/'+r[len(prefix)+1:]]
                                row['metrics'][r] = {**m['average'], 'per_horizon': m['per_horizon']}
                                row['origins'][r] = 'fresh single-card dense-valid'
                        finally:
                            predictor.close()
                            if tensor_state_fingerprint(body.state_dict()) != before: raise RuntimeError('frozen model changed')
                    publish(out, result)
            result['status'] = 'complete'
            return 0
        except InterruptedError:
            result['status'] = 'stopped_at_complete_window'; return 130
        except Exception as error:
            result.update(status='failed_originals_unchanged', error=repr(error)); raise
        finally:
            for s in sources:
                if base.sha256(s['path']) != s['sha256']: raise RuntimeError('original source checkpoint changed')
            publish(out, result); print(comparison_text(result), flush=True)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
