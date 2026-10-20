"""Frozen NEW ego-query/pure-memory/cross-first full epoch2; ALL4369 only."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as base
from .common import select_source, load_frozen, DenseModalInputs, EgoPredictor
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.final_semantic_evaluation.engine import evaluate
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint

PROTOCOL = 'frozen_NEW_ego_query_pure_memory_cross_first_epoch2_ALL4369_v1'


def select_jobs(records):
    keys = base.record_keys(records)
    if len(keys) != 4369 or len(set(keys)) != 4369 or len({k[0] for k in keys}) != 150:
        raise RuntimeError('complete unique ALL_VAL4369 / 150 scenes required; no population trimming')
    return [window(r) for r in records], [list(k) for k in keys]


def summary(result):
    c = result['contract']
    lines = ['===== FROZEN NEW EGO QUERY / PURE MEMORY / CROSS-FIRST / ALL VAL4369 =====', 'status='+result['status'],
        'Fixed epoch2/update10292; no parameter mean, third-epoch weights, training or optimizer steps.',
        'Latest2+2/KTA query+skip/shared-pinned/NO ICP/pure-history Memory/cross-first/AdaGN.',
        'FOUR histories -> SIX futures; SAME GT future ego conditions Decoder AND renderer; NO Pred/head/navigation.',
        'OCC AND STC histories dense-valid; full-grid scoring, no visibility mask/alignment/threshold tuning.',
        f'windows={c["windows"]}; shard={c.get("shard", [0, 1])}; parent=ALL4369/150; no planner trimming. NOT FPS.',
        'route                         horizon       IoU      mIoU MovingMicro']
    for name, report in result['reports'].items():
        for h, row in [*report['per_horizon'].items(), ('Avg', report['average'])]:
            values = ['NA' if row.get(k) is None else f'{row[k]:10.6f}' for k in ('IoU', 'mIoU', 'MovingMicro')]
            lines.append(f'{name:29s} {h:>5s} '+ ' '.join(values))
    return '\n'.join(lines)+'\nOriginal checkpoint/Adam/RNG/caches read only. No automatic selection or extension.\n'


def main(event=None, argv=None):
    event = event or threading.Event(); p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    p.add_argument('--config', required=True); p.add_argument('--dataroot', required=True)
    p.add_argument('--dev-info', required=True); p.add_argument('--dev-cache', required=True); p.add_argument('--stc-root', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--shard-index', type=int, default=0); p.add_argument('--shard-count', type=int, default=1)
    a = p.parse_args(argv); repo = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    if not 1 <= a.cpu_workers <= 8: raise ValueError('bounded CPU workers required')
    if a.shard_count not in (1, 2) or not 0 <= a.shard_index < a.shard_count: raise ValueError('invalid single-card shard')
    if a.device.startswith('cuda') and torch.cuda.device_count() != 1: raise RuntimeError('expose exactly ONE GPU')
    torch.set_num_threads(1); torch.manual_seed(21)
    if a.resume and not a.source_dir:
        a.source_dir = str(Path(json.loads((out/'contract.json').read_text(encoding='utf-8'))['source']).parent)
    source = select_source(a.source_dir, repo/'outputs/p0_f9_shared_history')
    if out == source.parent or source.parent in out.parents: raise ValueError('independent new output outside training run required')
    source_sha = base.sha256(source)
    lock = out.parent/'.ego_all_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); snapshot_checkpoint(source, out/'epoch2_frozen.pt')
        frozen = out/'epoch2_frozen.pt'
        if base.sha256(frozen) != source_sha: raise RuntimeError('source checkpoint differs from fixed snapshot')
        c, body = load_frozen(frozen, a.device)
        for name in ('config', 'dev_info', 'dev_cache'):
            if base.sha256(getattr(a, name)) != c['data'][name]: raise RuntimeError('original metadata differs: '+name)
        for name, expected in c['implementation'].items():
            if base.sha256(repo/name) != expected: raise RuntimeError('source implementation changed: '+name)
        _, records = base.load_cache(a.dev_cache); jobs, keys = select_jobs(records)
        if {k[0] for k in keys} & {k[0] for k in c['train_keys']}: raise RuntimeError('TRAIN/VAL scene overlap')
        jobs, keys = jobs[a.shard_index::a.shard_count], keys[a.shard_index::a.shard_count]
        inputs = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
        inventory = {kind: inputs.preflight(jobs, kind) for kind in ('occ', 'stc')}
        files = [*sorted(Path(__file__).parent.glob('*.py')),
            *[repo/n for n in ('tools/final_semantic_evaluation/engine.py','tools/final_semantic_evaluation/metrics.py',
                              'tools/final_semantic_evaluation/sources.py','tools/final_semantic_evaluation/predictor.py')]]
        contract = dict(protocol=PROTOCOL, source=str(source), source_sha256=source_sha, source_contract=c,
            keys=keys, windows=len(keys), scenes=len({k[0] for k in keys}), parent_windows=4369,
            shard=[a.shard_index, a.shard_count], known=dict(occ='dense_valid', stc='dense_valid'),
            inventory=inventory, runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=a.device,
                cpu_workers=a.cpu_workers, dataroot=str(Path(a.dataroot).resolve()), stc_root=str(Path(a.stc_root).resolve())),
            implementation={f.relative_to(repo).as_posix(): base.sha256(f) for f in files})
        cp = out/'contract.json'
        if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract: raise RuntimeError('same unchanged evaluation required')
        base.write_json(cp, contract)
        pcfg = base.make_prepare_config(base.load_runtime_config(a.config, c.get('overrides')))
        predictor = EgoPredictor(body, pcfg, a.device, workers=a.cpu_workers)
        before = tensor_state_fingerprint(body.state_dict())
        result = dict(status='running', contract=contract, scientific_updates=0, reports={})
        if a.resume: result = json.loads((out/'evaluation.json').read_text(encoding='utf-8')); result['status'] = 'running'
        def publish():
            base.write_json(out/'evaluation.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
        publish()
        print(f'EGO_ALL source={source}; fixed epoch2/10292; shard={a.shard_index}/{a.shard_count}; '
              f'windows={len(jobs)}; ONE visible GPU; output={out}', flush=True)
        try:
            with torch.no_grad():
                for kind, prefix in (('occ', 'occ_gt'), ('stc', 'stc_gt')):
                    path = out/f'{kind}_state.json'; state = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
                    print(f'EGO_ALL modality={kind}; dense-valid; shard_windows={len(jobs)}', flush=True)
                    value = evaluate(predictor, DenseModalInputs(inputs, kind), jobs, 'gt', keys,
                        dict(base=contract, modality=kind), saved=state,
                        save=lambda state, path=path: base.write_json(path, state), stop_event=event)
                    result['reports'].update({prefix+'/'+name.split('/', 1)[1]: m for name, m in value['reports'].items()})
                    publish()
            result['status'] = 'complete'; return 0
        except InterruptedError:
            result['status'] = 'stopped_at_complete_window'; return 130
        except Exception as error:
            result.update(status='failed_original_unchanged', error=repr(error)); raise
        finally:
            predictor.close()
            if tensor_state_fingerprint(body.state_dict()) != before or base.sha256(source) != source_sha:
                raise RuntimeError('read-only source/model changed')
            publish(); print(summary(result), flush=True)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
