"""Frozen fixed mean3, fresh no-ICP/KTA-query inference on the SAME remaining64."""
import argparse
import json
from pathlib import Path
import signal
import threading
import torch
from tools.joint_occ_experiments import run as base
from tools.source_scene_integrated_full.model import checkpoint_model
from tools.source_scene_stage2.adapter import DirectPredictor
from tools.source_scene_kta_query.adapter import forward_window
from tools.final_semantic_evaluation.engine import evaluate
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.history_validity.data import policy_raw
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .weights import PROTOCOL, STEPS, METRICS, select_source, load_sources, mean_payload


class QueryPredictor(DirectPredictor):
    @torch.no_grad()
    def forward(self, row, *, poses):
        return forward_window(self.body, self.provider, row, poses=poses, fresh=True)


class DenseInputs:
    def __init__(self, source, kind): self.source, self.kind = source, kind
    def history(self, job, _kind='occ'):
        rec, raw, times = self.source.history(job, self.kind)
        return rec, policy_raw(raw, 'dense_valid'), times
    def poses(self, job): return self.source.poses(job)
    def targets(self, job, indices): return self.source.targets(job, indices)
    def moving(self, *args): return self.source.moving(*args)


def summary(v):
    lines = ['===== LATEST INTEGRATED 2+2 / FIXED MEAN3 / SAME REMAINING64 =====',
        'status='+v['status'], 'Equal 1/3: 8000,10000,10292(epoch2); ONE GPU / one model process.',
        'KTA query+skip / no ICP / direct history yaw memory / AdaGN; no old model routes.',
        'OCC AND STC dense-valid; GT future ego; full-grid 1/2/3s average.',
        'Original three controls READ from same-run completed reports; ONLY mean3 freshly evaluated.',
        'No training, Adam averaging, threshold search, automatic selection or promotion.',
        'model    route                         mIoU       IoU MovingMicro']
    for name, scores in v['comparison'].items():
        for route, m in scores.items():
            lines.append(f'{name:8s} {route:28s} '+ ' '.join(f'{m[k]:.6f}' for k in METRICS))
    mean = v['comparison'].get('mean3', {})
    for step in ('8000', '10292'):
        for route, m in mean.items():
            old = v['comparison'][step][route]
            lines.append(f'mean3_minus_{step} {route}: '+json.dumps({k:m[k]-old[k] for k in METRICS}))
    return '\n'.join(lines+['Original weights/Adam/RNG/history caches unchanged. Fresh vs cached NOT byte-identity.',
        'Repeatedly inspected development64, NOT full-VAL proof, independent test or FPS.'])+'\n'


def main(argv=None, event=None):
    event = event or threading.Event(); p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    for n in ('config', 'dev-cache', 'dev-info', 'dataroot', 'stc-root'): p.add_argument('--'+n, required=True)
    p.add_argument('--cpu-workers', type=int, default=4); p.add_argument('--device', default='cuda')
    a = p.parse_args(argv); repo = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    if not 1 <= a.cpu_workers <= 8: raise ValueError('bounded CPU workers required')
    if a.device.startswith('cuda') and (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('exactly ONE BF16 CUDA GPU required')
    torch.set_num_threads(1)
    if a.resume and not a.source_dir:
        a.source_dir = json.loads((out/'contract.json').read_text(encoding='utf-8'))['source_dir']
    source = select_source(a.source_dir, repo/'outputs/p0_f9_shared_history')
    if out == source or source in out.parents: raise RuntimeError('new independent output outside source run required')
    c, sources, controls, header, report_sha = load_sources(source)
    for name in ('config', 'dev_cache', 'dev_info'):
        if base.sha256(getattr(a, name)) != c['data'][name]: raise RuntimeError('source data differs: '+name)
    for name, sha in c['implementation'].items():
        if base.sha256(repo/name) != sha: raise RuntimeError('source implementation changed: '+name)
    _, records = base.load_cache(a.dev_cache); keys = c['eval_keys']
    jobs = [window(r) for r in base.align_records(records, keys)]
    inputs = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
    inventory = {k: inputs.preflight(jobs, k) for k in ('occ', 'stc')}
    for path, size, stamp in c['STC_inventory']:
        st = Path(path).stat()
        if (st.st_size, st.st_mtime_ns) != (size, stamp): raise RuntimeError('source STC history changed')
    body = checkpoint_model(header, 'cpu')
    execution_files = sorted(Path(__file__).parent.glob('*.py'))+[repo/n for n in (
        'tools/final_semantic_evaluation/engine.py', 'tools/final_semantic_evaluation/sources.py',
        'tools/final_semantic_evaluation/metrics.py', 'tools/final_semantic_evaluation/predictor.py',
        'tools/real_motion/joint_surface_checkpoint_selection.py')]
    contract = dict(protocol=PROTOCOL, source_dir=str(source), source_contract=c, keys=keys,
        sources=[{k:s[k] for k in ('step', 'path', 'sha256')} for s in sources], source_report_sha256=report_sha,
        coefficients=[1/3]*3, controls=controls, inventory=inventory,
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, cpu_workers=a.cpu_workers, device=a.device),
        implementation={str(f.relative_to(repo)):base.sha256(f) for f in execution_files})
    contract = json.loads(json.dumps(contract))
    lock = out.parent/'.integrated_mean_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if a.resume:
            if json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract:
                raise RuntimeError('same unchanged averaging/evaluation contract required')
            receipt = json.loads((out/'mean_receipt.json').read_text(encoding='utf-8'))
            if base.sha256(out/'mean3.pt') != receipt['sha256']: raise RuntimeError('mean artifact changed')
        else:
            out.mkdir(parents=True, exist_ok=False); base.write_json(out/'contract.json', contract)
            save_resume_checkpoint(out/'mean3.pt', mean_payload(sources, body, contract))
            base.write_json(out/'mean_receipt.json', dict(sha256=base.sha256(out/'mean3.pt')))
        averaged = torch.load(out/'mean3.pt', map_location='cpu', weights_only=False)
        if (averaged.get('protocol') != PROTOCOL or averaged['contract'] != contract or
                averaged.get('resume_allowed') is not False or averaged.get('source_updates') != list(STEPS) or
                averaged.get('coefficients') != [1/3]*3 or set(averaged['models']) != {'gt'}):
            raise RuntimeError('not this fixed evaluation-only mean3')
        body.load_state_dict(averaged['models']['gt'], strict=True)
        expected = tensor_state_fingerprint(body.state_dict()); del sources, averaged, header
        body = body.to(a.device).eval().requires_grad_(False)
        pcfg = base.make_prepare_config(base.load_runtime_config(a.config, c['overrides']))
        predictor = QueryPredictor(body, pcfg, a.device, a.cpu_workers)
        result = dict(status='running', contract=contract, comparison=dict(controls), evaluation={}, scientific_updates=0)
        if a.resume and (out/'results.json').is_file():
            result = json.loads((out/'results.json').read_text(encoding='utf-8'))
            if result['contract'] != contract: raise RuntimeError('result provenance differs')
        def publish():
            base.write_json(out/'results.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
            (out/'comparison.md').write_text(summary(result), encoding='utf-8')
        try:
            publish()
            for kind, prefix in (('occ', 'occ_gt'), ('stc', 'stc_gt')):
                if kind in result['evaluation']: continue
                statefile = out/(kind+'_state.json')
                saved = json.loads(statefile.read_text(encoding='utf-8')) if statefile.exists() else None
                print(f'MEAN3 {kind} {len(saved["keys"]) if saved else 0}/64', flush=True)
                report = evaluate(predictor, DenseInputs(inputs, kind), jobs, 'gt', keys, contract,
                    saved=saved, save=lambda value: base.write_json(statefile, value), stop_event=event)
                result['evaluation'][kind] = report
                row = result['comparison'].setdefault('mean3', {})
                for route in ('transport', 'semantic_refine'):
                    m = report['reports']['gt/'+route]
                    row[prefix+'_'+route] = dict(m['average'], per_horizon=m['per_horizon'])
                publish()
            result['status'] = 'complete'
        except InterruptedError:
            result['status'] = 'stopped_at_complete_window'; return 2
        except Exception as error:
            result.update(status='failed', error=repr(error)); raise
        finally:
            predictor.close()
            if tensor_state_fingerprint(body.state_dict()) != expected: raise RuntimeError('frozen model mutated')
            for item in contract['sources']:
                if base.sha256(item['path']) != item['sha256']: raise RuntimeError('source checkpoint changed')
            if base.sha256(source/'results.json') != report_sha: raise RuntimeError('source report changed')
            publish(); print(summary(result), flush=True)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    raise SystemExit(main(event=event) or 0)
