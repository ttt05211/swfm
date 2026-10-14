"""Real STC four histories -> GT six futures; 18 VAL scenes, fixed500, no simulated noise."""
import argparse
import copy
import json
from pathlib import Path
import random
import signal
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt import engine as gt
from tools.final_semantic_evaluation.predictor import load_body, Predictor
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.real_motion.joint_training_recovery import snapshot_checkpoint, save_resume_checkpoint, preserve_training_rng
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .data import HistoryRows, validate_chain, schedule, batches, ready
from .evaluation import evaluate

PROTOCOL = 'GTepoch2_real_STC_VAL18_supervised_adaptation500_v1'


def selection(records, split, eval_keys):
    fit, remaining = set(split['fit_scenes']), set(split['remaining_scenes'])
    if len(fit) != 18 or len(remaining) != 132 or fit & remaining:
        raise RuntimeError('original fixed 18/132 disjoint scene list required')
    train = [r for r in records if r['scene_name'] in fit]
    evaluation = legacy.align_records(records, list(map(tuple, eval_keys)))
    if {r['scene_name'] for r in train} != fit or len(evaluation) != 64 or not {r['scene_name'] for r in evaluation} <= remaining:
        raise RuntimeError('real 18-scene training / exact prior remaining64 required')
    if len(set(legacy.record_keys(train))) != len(train): raise RuntimeError('duplicate adaptation windows')
    return train, evaluation


def checkpoint(model, adam, contract, update, reports):
    return dict(protocol=PROTOCOL, contract=contract, update=update, model=model.state_dict(),
                adam=adam.state_dict(), rng=legacy.rng_state(), reports=reports)


def summary(result):
    lines = ['===== REAL STC18 / ORIGINAL GT EPOCH2 / FULL ADAPT500 =====', 'status='+result['status'],
        f'updates={result.get("update", 0)}/500; completed_dataset_passes={result.get("complete_passes", 0)}',
        f'TRAIN real released STC histories: {result["contract"]["train_windows"]} windows / 18 VAL scenes.',
        'No simulated noise / extra augmentation / internal ego head. GT future ego explicitly conditioned.',
        'Original GTepoch2 -> fresh Adam/RNG; full Encoder/STWM/Refiner unfrozen.',
        'Body LR1e-5 / Refiner LR1e-4; B4 nominal GT-source128 grouping / H6 CE+Lovasz ONLY; c=1.',
        'Supervised target-domain VAL adaptation, NOT zero-shot or standard full-VAL score.',
        'Exact same remaining64; scenes disjoint from adaptation, but previously inspected, NOT pristine test.']
    if result.get('performance'): lines += ['PERFORMANCE '+json.dumps(result['performance'])]
    if result.get('evaluation'):
        e = result['evaluation']; lines += [f'evaluated_windows={e["windows"]}/64', 'model     route                         mIoU       IoU MovingMicro']
        for arm, metrics in e['metrics'].items():
            for name, m in metrics.items(): lines += [f'{arm:9s} {name:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}']
        for name in e['metrics']['original']:
            lines += [name+'_adapted_minus_original='+json.dumps({k: e['metrics']['adapted'][name][k]-e['metrics']['original'][name][k]
                for k in ('mIoU', 'IoU', 'MovingMicro')})]
    return '\n'.join(lines)+'\nOriginal checkpoints/caches unchanged. No automatic retry/expansion/promotion. NOT FPS.\n'


def main(event=None, argv=None):
    event = event or threading.Event()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--max-updates', type=int, default=0)
    a = p.parse_args(argv)
    if not 0 <= a.max_updates <= 500: raise ValueError('fixed500; max-updates only stops within the same schedule')
    parent = Path(a.reference_dir).resolve(); out = Path(a.out_dir).resolve()
    if out == parent or parent in out.parents: raise ValueError('independent output outside previous runs required')
    workflow = json.loads((parent/'workflow.json').read_text(encoding='utf-8'))
    reference = json.loads((parent/'empirical/results.json').read_text(encoding='utf-8'))
    profile = json.loads((parent/'profile/evaluation.json').read_text(encoding='utf-8'))
    # ONLY scene names are consumed. No fitted probabilities/component/noise model read.
    split = profile['split']; fields = argparse.Namespace(**workflow['args'])
    source_path = parent/'empirical/frozen/GTepoch2.pt'; digest = legacy.sha256(source_path)
    if reference['status'] != 'complete' or reference['updates'] != 500 or digest != reference['contract']['source_GT_sha256']:
        raise RuntimeError('completed paired reference and ORIGINAL GTepoch2 snapshot required')
    device = legacy.require_cuda(fields.device); torch.set_num_threads(1)
    _, records = legacy.load_cache(fields.dev_cache)
    if len(records) != 4369: raise RuntimeError('original VAL4369 metadata required')
    train, dev = selection(records, split, reference['contract']['eval_keys'])
    inputs = NuscInputs(fields.dataroot, fields.dev_info, stc_root=fields.stc_root, cache_mib=fields.frame_cache_mib)
    chain = [validate_chain(inputs.source.nusc, r) for r in train+dev]
    inventory = inputs.preflight([window(r) for r in train+dev], 'stc')
    inventory += inputs.preflight([window(r) for r in dev], 'occ')
    groups, positions = schedule(train, fields.seed)
    root = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob('*.py'))
    for directory in ('tools/shared_history_experiments/full_gt', 'tools/shared_history_experiments/refine'):
        files += list((root/directory).glob('*.py'))
    files += [root/x for x in ('tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
        'tools/final_semantic_evaluation/predictor.py', 'tools/final_semantic_evaluation/sources.py',
        'tools/stc_noise_execution/pipeline.py', 'tools/scene_ego_head/integration.py')]
    contract = dict(protocol=PROTOCOL, source_GT_sha256=digest, seed=fields.seed, updates=500,
        train_windows=len(train), train_keys=[list(k) for k in legacy.record_keys(train)],
        eval_keys=[list(k) for k in legacy.record_keys(dev)], fit_scenes=sorted(split['fit_scenes']),
        remaining_scenes=sorted(split['remaining_scenes']), chain_audit=chain, inventory=inventory,
        batch4_nominal_source_budget=128, actual_sources_never_truncated=True,
        groups_sha256=legacy.stable_json_fingerprint([[list(k) for k in legacy.record_keys(g)] for g in groups]),
        objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.), lr=[1e-5, 1e-4],
        schedule='fixed500_cosine_floor_0.01', synthetic_noise=False, extra_augmentation=False,
        data={k: legacy.sha256(getattr(fields, k)) for k in ('config', 'dev_cache', 'dev_info')},
        reference={str(x): legacy.sha256(x) for x in (parent/'workflow.json', parent/'empirical/results.json', parent/'profile/evaluation.json')},
        implementation={str(x.relative_to(root)): legacy.sha256(x) for x in files},
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, cpu_workers=2, geometry_ram_mib=512))
    for k in ('config', 'dev_cache', 'dev_info'):
        if contract['data'][k] != workflow['data'][k]: raise RuntimeError('original metadata/config changed: '+k)
    lock = out.parent/'.real_stc_locks'/legacy.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if out.exists():
            if not a.resume or json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract:
                raise RuntimeError('NEW output or SAME unchanged real-STC run with --resume')
        else:
            if a.resume: raise RuntimeError('missing run to resume')
            out.mkdir(); legacy.write_json(out/'contract.json', contract); snapshot_checkpoint(source_path, out/'GTepoch2.pt')
        if legacy.sha256(out/'GTepoch2.pt') != digest: raise RuntimeError('GT source snapshot changed')
        torch.manual_seed(fields.seed); torch.cuda.manual_seed_all(fields.seed); np.random.seed(fields.seed); random.seed(fields.seed)
        _, original = load_body(out/'GTepoch2.pt', device)
        model = copy.deepcopy(original).train().requires_grad_(True)
        pcfg = legacy.make_prepare_config(legacy.load_runtime_config(fields.config, fields.override))
        predictor = Predictor(model, pcfg, device, workers=min(fields.cpu_workers, 4))
        reader = HistoryRows(inputs, predictor)
        adam = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=1e-5, initial_lr=1e-5, weight_decay=1e-4),
                                 dict(params=model.columns.parameters(), lr=1e-4, initial_lr=1e-4, weight_decay=.01)])
        update, reports = 0, []
        initial = tensor_state_fingerprint(original.state_dict())
        if a.resume and (out/'last.pt').is_file():
            saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            if saved['protocol'] != PROTOCOL or saved['contract'] != contract or not 0 <= saved['update'] <= 500:
                raise RuntimeError('real-STC checkpoint contract/cursor differs')
            model.load_state_dict(saved['model'], strict=True); adam.load_state_dict(saved['adam'])
            legacy.restore_rng(saved['rng']); update, reports = saved['update'], saved['reports']
            if len(reports) != update: raise RuntimeError('training report/update cursor differs')
        elif a.resume: raise RuntimeError('missing real-STC last.pt')
        else: legacy.write_json(out/'initialization.json', dict(original_GT_sha256=digest, body_state_sha256=initial,
            same_initial_weights=initial == tensor_state_fingerprint(model.state_dict()), fresh_Adam_RNG=True))
        result = dict(status='running', contract=contract, update=update); healthy = True
        def save(): save_resume_checkpoint(out/'last.pt', checkpoint(model, adam, contract, update, reports))
        try:
            iterator = batches(reader, groups[update:] if not a.max_updates or update < a.max_updates else [], event)
            try:
                for rows, wait in iterator:
                    if event.is_set(): break
                    healthy = False; tick = time.perf_counter()
                    prepared = [ready(predictor.provider, row) for row in rows]
                    stats = gt.train_arm(model, adam, predictor.provider, prepared, 'gt', update=update+1,
                        total_updates=500, probe=update == 0)
                    update += 1; healthy = True
                    item = dict(update=update, input_wait=wait, wall_seconds=time.perf_counter()-tick,
                        position=positions[update-1], **stats)
                    reports.append(item)
                    with (out/'progress.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(item)+'\n')
                    if update == 1 or update % 16 == 0:
                        print(f'REAL_STC_ADAPT {update}/500 CE={stats["CE"]:.4f} seconds/window={(item["wall_seconds"]+wait)/len(rows):.4f}', flush=True)
                    if update % 32 == 0: save()
                    if a.max_updates and update >= a.max_updates: break
            finally: iterator.close()
            save()
            if update == 500 and not event.is_set():
                state_path = out/'eval_state.json'
                saved_eval = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else None
                with preserve_training_rng():
                    result['evaluation'] = evaluate({'original': original, 'adapted': model}, reader, dev,
                        saved=saved_eval, save=lambda v: legacy.write_json(state_path, v), event=event)
                result['status'] = 'complete' if result['evaluation']['windows'] == 64 else 'stopped_at_complete_eval_window'
            else: result['status'] = 'stopped_at_complete_update'
        except Exception as error:
            result.update(status='failed_previous_checkpoint_preserved', error=repr(error))
            if healthy: save()
            raise
        finally:
            result['update'] = update
            result['complete_passes'] = sum(r['position']['batch'] == r['position']['cycle_batches'] for r in reports)
            latest = reports[-64:]; n = sum(r['windows'] for r in latest)
            result['performance'] = dict(windows=n, seconds_per_window_including_input=
                sum(r['wall_seconds']+r['input_wait'] for r in latest)/max(n, 1),
                host_stages_seconds_per_window={k: sum(r['stages_seconds'].get(k, 0.) for r in latest)/max(n, 1)
                    for k in {k for r in latest for k in r['stages_seconds']}}, NOT_FPS=True)
            result['geometry_RAM_cache'] = reader.stats()
            result['original_weights_unchanged'] = tensor_state_fingerprint(original.state_dict()) == initial
            predictor.close()
            legacy.write_json(out/'results.json', result); text = summary(result)
            (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)
        if not result['original_weights_unchanged']: raise RuntimeError('original model changed')
        if legacy.sha256(source_path) != digest: raise RuntimeError('source GT file changed during run')
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
