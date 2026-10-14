"""No training: scene-disjoint-from-adaptation TRAIN64, clean and empirical histories."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.history_validity.data import policy_raw
from tools.stc_noise_adaptation import data as noise_data
from tools.stc_noise_adaptation.augmentation import load_model
from tools.stc_noise_full.run import references, PROTOCOL as FULL_PROTOCOL
from tools.stc_noise_execution.pipeline import noise_execution, CheckedStrong
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock

PROTOCOL = 'read_only_empirical_adaptation_scene_holdout64_v1'
ARMS = ('clean', 'empirical', 'full_empirical')
INPUTS = ('clean_dense', 'empirical_mix')
ROUTES = ('transport', 'semantic_refine')


def select_holdout(records, contract, count=64):
    """Reconstruct consumed optimizer groups, then exclude WHOLE scenes."""
    training = legacy.align_records(records, list(map(tuple, contract['train_keys'])))
    groups = legacy.epoch_groups(training, contract['seed'], 0,
        contract['window_batch'], contract['source_budget'])
    if contract['total_updates'] != 500 or len(groups) < 500:
        raise RuntimeError('original 500-group population required')
    used = [[str(r['scene_name']), str(r['t0_token'])] for g in groups[:500] for r in g]
    excluded = {k[0] for k in used}
    eligible = [r for r in records if str(r['scene_name']) not in excluded]
    if len(eligible) < count:
        raise RuntimeError('fewer than 64 adaptation-unseen TRAIN windows; no window-only fallback')
    keys = legacy.select_scene_balanced_round_robin(legacy.record_keys(eligible), count)
    if len(eligible) < count or len(keys) != count or len(set(keys)) != count:
        raise RuntimeError('fewer than 64 adaptation-unseen TRAIN windows; no window-only fallback')
    selected = legacy.align_records(records, keys)
    return selected, dict(keys=[list(k) for k in keys], excluded_adaptation_scenes=sorted(excluded),
        consumed_windows=len(used), consumed_keys_sha256=legacy.stable_json_fingerprint(used),
        eligible_windows=len(eligible), eligible_scenes=len({r['scene_name'] for r in eligible}),
        selected_scenes=len({r['scene_name'] for r in selected}),
        body_pretraining_saw_these_TRAIN_scenes=True)


def confusion(pred, target):
    if pred.shape != target.shape or len(pred) != 6:
        raise ValueError('six matching full-grid prediction/label frames required')
    result = []
    for h in (1, 3, 5):
        p, g = pred[h].ravel(), target[h].ravel()
        valid = (g >= 0) & (g < 18)
        if np.any((p < 0) | (p >= 18)): raise ValueError('prediction outside 18 classes')
        result.append(np.bincount(18*g[valid].astype(np.int64)+p[valid], minlength=324).reshape(18, 18))
    return np.stack(result)


def scores(counts):
    c = np.asarray(counts)
    if c.shape != (3, 18, 18) or c.dtype.kind not in 'iu' or np.any(c < 0):
        raise ValueError('nonnegative integer horizon confusion required')
    per = {}
    for i, h in enumerate(('1.0', '2.0', '3.0')):
        matrix = c[i]; d = np.diag(matrix); union = matrix.sum(0)+matrix.sum(1)-d
        good = union[:17] > 0
        sem = 100*d[:17][good]/union[:17][good]
        binary_union = matrix.sum()-matrix[17, 17]
        per[h] = dict(mIoU=float(sem.mean()) if len(sem) else None,
            IoU=float(100*matrix[:17, :17].sum()/binary_union) if binary_union else None)
    return {k: float(np.mean([v[k] for v in per.values()]))
            if all(v[k] is not None for v in per.values()) else None for k in ('mIoU', 'IoU')} | {'per_horizon': per}


@torch.no_grad()
def evaluate(models, provider, rows, noise, seed, *, saved=None, save=None, event=None):
    """Same simulated history realization for all models; complete-window publication."""
    state = dict(protocol=PROTOCOL, rows=[]) if saved is None else copy.deepcopy(saved)
    if state.get('protocol') != PROTOCOL: raise RuntimeError('evaluation state protocol differs')
    if len(state['rows']) > 64 or len({tuple(r['key']) for r in state['rows']}) != len(state['rows']):
        raise RuntimeError('invalid saved population')
    for saved_row in state['rows']:
        if set(saved_row['counts']) != set(ARMS): raise RuntimeError('saved model rows differ')
        for inputs in saved_row['counts'].values():
            if set(inputs) != set(INPUTS): raise RuntimeError('saved input rows differ')
            for routes in inputs.values():
                if set(routes) != set(ROUTES): raise RuntimeError('saved readouts differ')
                for counts in routes.values(): scores(counts)  # validate integer schema on resume
    for model in models.values(): model.eval()
    done = len(state['rows']); offset = 0
    strong = CheckedStrong(lambda audit: state.update(Strong_reuse_audit=audit))
    try:
        with noise_execution():
            for original in rows:
                key = [str(original[0]['scene_name']), str(original[0]['t0_token'])]
                if offset < done:
                    if state['rows'][offset]['key'] != key: raise RuntimeError('saved evaluation prefix differs')
                    offset += 1; continue
                if event is not None and event.is_set(): break
                # No learned features shared between models. Pure geometric rows may be shared.
                synthetic = strong(provider, noise_data.augment_row(provider, original, noise, seed))
                audit = synthetic[1].get('_empirical_noise', {})
                predictions = {}
                for arm, model in models.items():
                    predictions[arm] = {}
                    for modality, row in zip(INPUTS, (original, synthetic)):
                        fw = gt.forward_window(model, provider, row)
                        predictions[arm][modality] = {
                            route: fw[name].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                            for route, name in (('transport', 'prior'), ('semantic_refine', 'logits'))}
                        del fw
                # Neither simulator nor scene forward consumes future occupancy.
                target = np.asarray(original[1]['future_gt_occ'])
                counts = {arm: {inp: {route: confusion(p, target).tolist() for route, p in values.items()}
                    for inp, values in inputs.items()} for arm, inputs in predictions.items()}
                state['rows'].append(dict(key=key, audit=audit, counts=counts)); offset += 1
                if save: save(state)
                if offset % 8 == 0: print(f'SYNTHETIC_HOLDOUT {offset}/64', flush=True)
    finally:
        close = getattr(rows, 'close', None)
        if close: close()
        if save: save(state)
    if offset < done: raise RuntimeError('evaluation cursor exceeds population')
    return state


def report(state):
    result = {}
    for subset in ('all', 'corrupted_only'):
        rows = [r for r in state['rows'] if subset == 'all' or r['audit'].get('selected')]
        metrics = {arm: {inp: {route: scores(np.sum([r['counts'][arm][inp][route] for r in rows], axis=0,
            dtype=np.int64)) for route in ROUTES} for inp in INPUTS} for arm in ARMS} if rows else {}
        result[subset] = dict(windows=len(rows), metrics=metrics)
    return result


def summary(result):
    lines = ['===== FROZEN THREE ARMS / SYNTHETIC TRAIN HOLDOUT64 =====', 'status='+result['status'],
        'No training/optimizer. Entire evaluation scenes excluded from ALL first500 adaptation groups.',
        'GT body pretraining DID use these TRAIN scenes; NOT an independent test.',
        'Same dense-valid clean / 50%-mixture empirical history realization for all three models.',
        'Full6 forecasts, full-grid standard IoU/mIoU at 1/2/3s. Moving NOT evaluated.',
        'VAL-calibrated noise, unchanged c=1. No coefficient/threshold/model selection.']
    for subset, group in result['subsets'].items():
        lines += [f'===== {subset}: windows={group["windows"]} =====', 'arm              input            mIoU       IoU Refiner_dMiOU']
        for arm, inputs in group['metrics'].items():
            for inp, routes in inputs.items():
                base, m = routes['transport'], routes['semantic_refine']
                lines += [f'{arm:16s} {inp:15s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["mIoU"]-base["mIoU"]:+.6f}']
        if group['metrics']:
            for arm in ('empirical', 'full_empirical'):
                delta = {inp: {k: group['metrics'][arm][inp]['semantic_refine'][k]-
                    group['metrics']['clean'][inp]['semantic_refine'][k] for k in ('mIoU', 'IoU')} for inp in INPUTS}
                lines += [arm+'_minus_clean='+json.dumps(delta)]
    if result.get('existing_real_remaining64'):
        real = result['existing_real_remaining64']
        lines += ['===== EXISTING REAL STC / SAME PRIOR remaining64 (NOT rerun) =====']
        for arm, m in real.items(): lines += [f'{arm:16s} mIoU={m["mIoU"]:.6f} IoU={m["IoU"]:.6f}']
        for arm in ('empirical', 'full_empirical'):
            lines += [arm+'_minus_clean='+json.dumps({k: real[arm][k]-real['clean'][k] for k in ('mIoU', 'IoU')})]
        lines += ['Compare ARM DELTAS across domains, not absolute scores across different populations.']
    lines += ['full_empirical vs clean changes BOTH trainable scope and corruption; no full-unfreeze clean control.']
    return '\n'.join(lines)+'\nSynthetic vs real gap is a diagnosis, NOT proof of a unique cause or a new STC score. NOT FPS.\n'


def choose_full(parent, explicit=None):
    if explicit: return Path(explicit).resolve()
    candidates = []
    for path in parent.parent.glob('stc_full_unfreeze500_*/results.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        ref = path.parent/'reference.json'
        if (value.get('status') == 'complete' and value.get('updates') == 500
                and value.get('contract', {}).get('protocol') == FULL_PROTOCOL and ref.is_file()
                and json.loads(ref.read_text(encoding='utf-8'))['reference'].get(str(parent/'workflow.json')) == legacy.sha256(parent/'workflow.json')):
            candidates.append(path.parent)
    if len(candidates) != 1: raise RuntimeError('set FULL_NOISE_DIR: need one matching completed full-unfreeze run')
    return candidates[0]


def main(argv=None, event=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', required=True); p.add_argument('--full-dir')
    p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv); event = event or threading.Event()
    parent, workflow, refs = references(a.reference_dir); full = choose_full(parent, a.full_dir)
    full_report = json.loads((full/'results.json').read_text(encoding='utf-8'))
    if full_report['status'] != 'complete' or full_report['updates'] != 500 or full_report['contract']['protocol'] != FULL_PROTOCOL:
        raise RuntimeError('completed full-unfreeze500 required')
    for name in ('seed', 'train_keys', 'window_batch', 'source_budget', 'total_updates', 'data', 'source_GT_sha256', 'noise_identity'):
        if full_report['contract'][name] != refs['empirical']['contract'][name]: raise RuntimeError('unmatched full run: '+name)
    fields = argparse.Namespace(**workflow['args']); device = legacy.require_cuda(fields.device)
    torch.set_num_threads(1)
    checkpoints = dict(clean=parent/'clean/train/last.pt', empirical=parent/'empirical/train/last.pt', full_empirical=full/'train/last.pt')
    original_path = parent/'empirical/frozen/GTepoch2.pt'
    if legacy.sha256(original_path) != refs['empirical']['contract']['source_GT_sha256']:
        raise RuntimeError('original GT snapshot changed')
    raw_source = torch.load(original_path, map_location='cpu', weights_only=False)
    noise, binding = load_model(parent/'profile/noise_model.json')
    if binding != refs['empirical']['contract']['noise_identity']: raise RuntimeError('profile binding changed')
    for name in ('config', 'train_cache', 'dev_cache', 'population_manifest', 'base_checkpoint', 'train_info', 'dev_info'):
        if legacy.sha256(getattr(fields, name)) != workflow['data'][name]: raise RuntimeError('source metadata changed: '+name)
    _, records = legacy.load_cache(fields.train_cache)
    selected, population = select_holdout(records, refs['empirical']['contract'])
    paths = [parent/'workflow.json', parent/'clean/results.json', parent/'empirical/results.json', full/'results.json',
             original_path, parent/'profile/noise_model.json', parent/'profile/evaluation.json', *checkpoints.values()]
    root = Path(__file__).resolve().parents[2]
    code_dirs = ('tools/stc_noise_holdout', 'tools/stc_noise_adaptation', 'tools/stc_noise_execution',
                 'tools/shared_history_experiments/refine', 'tools/shared_history_experiments/full_gt')
    identity = dict(protocol=PROTOCOL, sources={str(x): legacy.sha256(x) for x in paths}, population=population,
        implementation={str(x.relative_to(root)): legacy.sha256(x) for d in code_dirs for x in (root/d).glob('*.py')},
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda), noise_identity=binding)
    out = Path(a.out_dir).resolve()
    if any(out == source_dir or source_dir in out.parents for source_dir in (parent, full)):
        raise ValueError('independent output outside source runs required')
    lock = out.parent/'.synthetic_eval_locks'/legacy.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if out.exists():
            if not a.resume or json.loads((out/'contract.json').read_text(encoding='utf-8')) != identity:
                raise RuntimeError('NEW output or SAME unchanged --resume evaluation')
        else:
            if a.resume: raise RuntimeError('missing evaluation to resume')
            out.mkdir(); legacy.write_json(out/'contract.json', identity)
        base = torch.load(fields.base_checkpoint, map_location='cpu', weights_only=False)
        cfg = replace(legacy.config_from_mapping_v17(base['model_config']), history_frames=4)
        c = raw_source['contract']['model']; s, r = c['shared_history'], c['repair']; models = {}
        for arm, path in checkpoints.items():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            reference = full_report if arm == 'full_empirical' else refs[arm]
            if saved['updates'] != 500 or saved['contract'] != reference['contract'] or set(saved['models']) != {'gt'}:
                raise RuntimeError('checkpoint/result identity mismatch: '+arm)
            m = GTBody(cfg, feature_width=s['width'], z_bins=s['z_bins'], memory_width=r['memory_width'],
                       width=r['width'], chunk_size=r['chunk_size'])
            m.load_state_dict(saved['models']['gt'], strict=True); m.columns.epsilon = r['epsilon']
            models[arm] = m.to(device).eval().requires_grad_(False)
        pcfg = legacy.make_prepare_config(legacy.load_runtime_config(fields.config, fields.override))
        provider = legacy.PilotProvider(fields.base_checkpoint, legacy.CLEAN_SHA256, pcfg, device, fields.cpu_workers, models['clean'], None)
        provider.raw_io_workers = 1
        from real_motion.canonical_repair_execution import CanonicalCpuExecution
        from real_motion.surface_projection_execution import SurfaceMapExecution
        from real_motion.strong_majority_execution import ParallelNativeMajority
        provider.ccr_execution = SurfaceMapExecution(CanonicalCpuExecution('native_parallel', min(4, fields.cpu_workers)))
        provider.strong_majority = ParallelNativeMajority(min(4, fields.cpu_workers))
        source = legacy.CachedColumnSource(legacy.NuScenesWindowSource(fields.dataroot, info_pkl=fields.train_info), fields.frame_cache_mib, copy_on_insert=False)
        provider.ccr_history_cache_source = source
        legacy.attach_caches(provider, fields, root)
        descriptors = ResidualHistoryInputs(provider, ram_mib=fields.descriptor_ram_mib)
        before = {name: tensor_state_fingerprint(m.state_dict()) for name, m in models.items()}
        def rows():
            for rec in selected:
                raw = policy_raw(provider.load_raw_columns(source, rec, include_gt=True), 'dense_valid')
                yield rec, raw, gt.history_inputs(source, rec, raw), descriptors.get(rec, raw)
        state_path = out/'state.json'
        state = json.loads(state_path.read_text(encoding='utf-8')) if a.resume and state_path.is_file() else None
        tick = time.perf_counter()
        print(f'SYNTHETIC_HOLDOUT: 64 windows / {population["selected_scenes"]} adaptation-unseen TRAIN scenes; no training.', flush=True)
        try:
            state = evaluate(models, provider, rows(), noise, fields.seed, saved=state,
                save=lambda v: legacy.write_json(state_path, v), event=event)
            result = dict(status='complete' if len(state['rows']) == 64 else 'interrupted',
                contract=identity, subsets=report(state), seconds=time.perf_counter()-tick,
                model_weights_unchanged=before == {name: tensor_state_fingerprint(m.state_dict()) for name, m in models.items()})
            result['existing_real_remaining64'] = {arm: {k: reference['evaluation']['variants']['gt']['stc_gt_semantic_refine'][k]
                for k in ('mIoU', 'IoU')} for arm, reference in
                (('clean', refs['clean']), ('empirical', refs['empirical']), ('full_empirical', full_report))}
            if not result['model_weights_unchanged']: raise RuntimeError('read-only evaluation changed model weights')
            for path, digest in identity['sources'].items():
                if legacy.sha256(path) != digest: raise RuntimeError('source artifact changed during evaluation')
            legacy.write_json(out/'results.json', result); text = summary(result)
            (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)
        finally:
            provider.ccr_history_cache.close(); provider.ccr_val_history_cache.close()
            provider.strong_majority.close(); provider.ccr_execution.close()
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event=event) or 0)
