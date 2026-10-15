"""GTepoch2 warm-start; expanded real STC + clean TRAIN GT, mixed500, same remaining64."""
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
from tools.stc_real_adaptation.data import validate_chain, ready
from tools.stc_real_adaptation.evaluation import evaluate
from .data import STEPS, selection, schedule, MixedRows, batches

PROTOCOL = 'GTepoch2_expanded_real_STC_clean_TRAIN_mix500_v1'


def checkpoint(model, adam, contract, update, reports):
    return dict(protocol=PROTOCOL, contract=contract, update=update, model=model.state_dict(),
                adam=adam.state_dict(), rng=legacy.rng_state(), reports=reports)


def summary(result):
    c = result['contract']; population = c['population']
    lines = ['===== EXPANDED REAL STC + CLEAN GT / MIXED500 =====', 'status='+result['status'],
        f'updates={result.get("update", 0)}/500; B4 = TWO real STC + TWO clean original TRAIN GT each step.',
        f'STC candidates: {population["STC_windows"]} windows / {population["STC_scenes"]} scenes; '
        f'actually scheduled distinct STC windows={population["used_STC_windows"]} / scenes={population["used_STC_scenes"]}.',
        '500 updates: 1000 STC exposures + 1000 DISTINCT clean TRAIN GT windows; NOT a dataset epoch.',
        'ALL original remaining64 evaluation scenes excluded from BOTH training modalities.',
        'Original GTepoch2 initialization, fresh Adam/RNG; NOT previous STC-only500 weights.',
        'Encoder/STWM/Refiner all unfrozen; LR1e-5/1e-4; full6 CE+Lovasz only, equal per-window weighting.',
        'No simulated noise/extra augmentation/ego head/new source-track branch; c=1; GT future ego conditioned.',
        'Fixed B4, ALL actual sources retained (no source128 truncation).',
        'Supervised VAL target-domain adaptation; NOT zero-shot or pristine/standard full-VAL test.']
    if result.get('performance'): lines += ['PERFORMANCE '+json.dumps(result['performance'])]
    if result.get('evaluation'):
        e = result['evaluation']; lines += [f'evaluated_windows={e["windows"]}/64', 'model     route                         mIoU       IoU MovingMicro']
        for arm, metrics in e['metrics'].items():
            for name, m in metrics.items(): lines += [f'{arm:9s} {name:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}']
        for name in e['metrics']['original']:
            lines += [name+'_mixed_minus_original='+json.dumps({k: e['metrics']['adapted'][name][k]-e['metrics']['original'][name][k]
                for k in ('mIoU', 'IoU', 'MovingMicro')})]
    return '\n'.join(lines)+'\nOriginal runs/checkpoints/caches unchanged. No automatic extension/promotion. NOT FPS.\n'


def main(event=None, argv=None):
    event = event or threading.Event()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--max-updates', type=int, default=0)
    a = p.parse_args(argv)
    if not 0 <= a.max_updates <= STEPS: raise ValueError('fixed mixed500 budget')
    parent, out = Path(a.reference_dir).resolve(), Path(a.out_dir).resolve()
    if out == parent or parent in out.parents: raise ValueError('independent new output required')
    workflow = json.loads((parent/'workflow.json').read_text(encoding='utf-8'))
    reference = json.loads((parent/'empirical/results.json').read_text(encoding='utf-8'))
    fields = argparse.Namespace(**workflow['args'])
    source_path = parent/'empirical/frozen/GTepoch2.pt'; digest = legacy.sha256(source_path)
    if reference['status'] != 'complete' or digest != reference['contract']['source_GT_sha256']:
        raise RuntimeError('original GTepoch2 source identity mismatch')
    device = legacy.require_cuda(fields.device); torch.set_num_threads(1)
    data_hash = {k: legacy.sha256(getattr(fields, k)) for k in ('config', 'dev_cache', 'dev_info', 'train_cache', 'train_info')}
    if any(data_hash[k] != workflow['data'][k] for k in data_hash): raise RuntimeError('original dataset/config changed')
    _, val_records = legacy.load_cache(fields.dev_cache); _, gt_records = legacy.load_cache(fields.train_cache)
    if (len(val_records), len(gt_records)) != (4369, 20430): raise RuntimeError('original VAL4369 / TRAIN20430 required')
    stc, clean, dev = selection(val_records, gt_records, reference['contract']['eval_keys'], fields.seed)
    groups = schedule(stc, clean, fields.seed)
    val_inputs = NuscInputs(fields.dataroot, fields.dev_info, stc_root=fields.stc_root, cache_mib=256)
    train_inputs = NuscInputs(fields.dataroot, fields.train_info, cache_mib=256)
    used = {tuple(legacy.record_keys([r])[0]): r for g in groups for kind, r in g if kind == 'stc'}
    holdout = sorted({r['scene_name'] for r in dev})
    chain = dict(stc=[validate_chain(val_inputs.source.nusc, r) for r in stc],
        clean_TRAIN=[validate_chain(train_inputs.source.nusc, r) for r in clean],
        evaluation=[validate_chain(val_inputs.source.nusc, r) for r in dev])
    inventory = dict(STC=val_inputs.preflight([window(r) for r in stc+dev], 'stc'),
        eval_OCC=val_inputs.preflight([window(r) for r in dev], 'occ'),
        clean_TRAIN=train_inputs.preflight([window(r) for r in clean], 'occ'))
    root = Path(__file__).resolve().parents[2]; files = list(Path(__file__).parent.glob('*.py'))
    for directory in ('tools/stc_real_adaptation', 'tools/shared_history_experiments/full_gt', 'tools/shared_history_experiments/refine'):
        files += list((root/directory).glob('*.py'))
    files += [root/x for x in ('tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
        'tools/final_semantic_evaluation/predictor.py', 'tools/final_semantic_evaluation/sources.py',
        'tools/stc_noise_execution/pipeline.py', 'tools/scene_ego_head/integration.py')]
    contract = dict(protocol=PROTOCOL, source_GT_sha256=digest, seed=fields.seed, updates=STEPS,
        population=dict(STC_windows=len(stc), STC_scenes=len({r['scene_name'] for r in stc}),
            used_STC_windows=len(used), used_STC_scenes=len({r['scene_name'] for r in used.values()}),
            GT_candidate_windows=len(gt_records), GT_unique_windows=len(clean)),
        train_STC_keys=[list(k) for k in legacy.record_keys(stc)], clean_TRAIN_keys=[list(k) for k in legacy.record_keys(clean)],
        eval_keys=[list(k) for k in legacy.record_keys(dev)], heldout_scenes=holdout, chain_audit=chain, inventory=inventory,
        groups_sha256=legacy.stable_json_fingerprint([[[kind, r['scene_name'], r['t0_token']] for kind, r in g] for g in groups]),
        batch=dict(windows=4, STC=2, GT=2, source_truncation=False), objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.),
        lr=[1e-5, 1e-4], schedule='fixed500_cosine_floor_0.01', synthetic_noise=False, extra_augmentation=False,
        data=data_hash, reference={str(x): legacy.sha256(x) for x in (parent/'workflow.json', parent/'empirical/results.json')},
        implementation={str(x.relative_to(root)): legacy.sha256(x) for x in files},
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, cpu_workers=2, total_geometry_ram_mib=512))
    lock = out.parent/'.mixed_stc_locks'/legacy.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if out.exists():
            if not a.resume or json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract:
                raise RuntimeError('NEW output or SAME unchanged mixed500 with --resume')
        else:
            if a.resume: raise RuntimeError('missing mixed experiment to resume')
            out.mkdir(); legacy.write_json(out/'contract.json', contract); snapshot_checkpoint(source_path, out/'GTepoch2.pt')
        if legacy.sha256(out/'GTepoch2.pt') != digest: raise RuntimeError('source snapshot changed')
        torch.manual_seed(fields.seed); torch.cuda.manual_seed_all(fields.seed); np.random.seed(fields.seed); random.seed(fields.seed)
        _, original = load_body(out/'GTepoch2.pt', device)
        model = copy.deepcopy(original).train().requires_grad_(True)
        pcfg = legacy.make_prepare_config(legacy.load_runtime_config(fields.config, fields.override))
        predictor = Predictor(model, pcfg, device, workers=min(fields.cpu_workers, 4))
        reader = MixedRows(val_inputs, train_inputs, predictor, {r['scene_name'] for r in gt_records})
        adam = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=1e-5, initial_lr=1e-5, weight_decay=1e-4),
                                 dict(params=model.columns.parameters(), lr=1e-4, initial_lr=1e-4, weight_decay=.01)])
        update, reports = 0, []; initial = tensor_state_fingerprint(original.state_dict())
        if a.resume:
            saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            if saved['protocol'] != PROTOCOL or saved['contract'] != contract or not 0 <= saved['update'] <= STEPS:
                raise RuntimeError('mixed checkpoint contract/cursor differs')
            model.load_state_dict(saved['model'], strict=True); adam.load_state_dict(saved['adam'])
            legacy.restore_rng(saved['rng']); update, reports = saved['update'], saved['reports']
            if len(reports) != update: raise RuntimeError('mixed reports/cursor differs')
        else: legacy.write_json(out/'initialization.json', dict(original_GT_sha256=digest, same_initial_weights=
            initial == tensor_state_fingerprint(model.state_dict()), fresh_Adam_RNG=True, imported_STC_only_weights=False))
        result = dict(status='running', contract=contract, update=update); healthy = True
        def save(): save_resume_checkpoint(out/'last.pt', checkpoint(model, adam, contract, update, reports))
        print('MIXED500: '+json.dumps(contract['population'])+' / B4=STC,GT,STC,GT; no source branch fusion.', flush=True)
        try:
            iterator = batches(reader, groups[update:] if not a.max_updates or update < a.max_updates else [], event)
            try:
                for rows, wait in iterator:
                    if event.is_set(): break
                    healthy = False; tick = time.perf_counter()
                    prepared = [ready(predictor.provider, row) for row in rows]
                    stats = gt.train_arm(model, adam, predictor.provider, prepared, 'gt', update=update+1,
                        total_updates=STEPS, probe=update == 0)
                    update += 1; healthy = True
                    item = dict(update=update, input_wait=wait, wall_seconds=time.perf_counter()-tick, STC_windows=2, GT_windows=2, **stats)
                    reports.append(item)
                    with (out/'progress.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(item)+'\n')
                    if update == 1 or update % 16 == 0:
                        print(f'MIXED_STC_GT {update}/500 CE={stats["CE"]:.4f} seconds/window={(item["wall_seconds"]+wait)/4:.4f}', flush=True)
                    if update % 32 == 0: save()
                    if a.max_updates and update >= a.max_updates: break
            finally: iterator.close()
            save()
            if update == STEPS and not event.is_set():
                state_path = out/'eval_state.json'; saved_eval = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else None
                with preserve_training_rng():
                    result['evaluation'] = evaluate({'original': original, 'adapted': model}, reader, dev, saved=saved_eval,
                        save=lambda v: legacy.write_json(state_path, v), event=event)
                result['status'] = 'complete' if result['evaluation']['windows'] == 64 else 'stopped_at_complete_eval_window'
            else: result['status'] = 'stopped_at_complete_update'
        except Exception as error:
            result.update(status='failed_previous_checkpoint_preserved', error=repr(error))
            if healthy: save()
            raise
        finally:
            result['update'] = update; latest = reports[-64:]; n = sum(r['windows'] for r in latest)
            result['performance'] = dict(windows=n, seconds_per_window_including_input=
                sum(r['wall_seconds']+r['input_wait'] for r in latest)/max(n, 1), NOT_FPS=True)
            result['history_RAM_cache'] = reader.stats()
            result['original_weights_unchanged'] = tensor_state_fingerprint(original.state_dict()) == initial
            predictor.close(); legacy.write_json(out/'results.json', result)
            text = summary(result); (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)
        if not result['original_weights_unchanged'] or legacy.sha256(source_path) != digest: raise RuntimeError('original source changed')
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
