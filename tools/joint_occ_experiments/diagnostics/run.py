"""One read-only TRAIN16 + DEV64 WM/CCR diagnosis; no optimization."""
import hashlib
import json
from pathlib import Path
import signal
import threading
import time
import numpy as np
import torch
from .. import run as legacy
from ..model import JointOccEgo, PROTOCOL
from ..data import HistoryInputs, attach_caches, prefetched_batches
from ..execution import BothActionsExecution
from ..performance.probe import legacy_fingerprint
from ..horizons.probe import execution_fingerprint
from .audit import ROUTES, SCORE_EDGES, gradient_batch, evaluation_window, merge_counts
from tools.real_motion.joint_training_recovery import snapshot_checkpoint, preserve_training_rng
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from real_motion.nuscenes_adapter import gt_moving_support_sequence

AUDIT_PROTOCOL = 'joint_occ_frozen_loss_action_surrogate_audit_v1'


def discover_checkpoint(directory):
    """Only completed three-epoch GT pilots; no filename/mtime-only guessing."""
    eligible = []
    for path in sorted(Path(directory).glob('*/last.pt')):
        if not path.parent.name.startswith('gt3_'): continue
        saved = torch.load(path, map_location='cpu', weights_only=False)
        c = saved.get('contract', {})
        if (saved.get('protocol') == PROTOCOL and c.get('modes') == ['gt'] and c.get('epochs') == 3
                and saved.get('epoch') == 3 and saved.get('batch') == 0
                and saved.get('updates') == c.get('total_updates')):
            eligible.append((int(saved['updates']), path.stat().st_mtime_ns, str(path)))
        del saved
    if not eligible:
        raise RuntimeError('No completed GT-only three-epoch last.pt found; set DIAG_CHECKPOINT explicitly.')
    return Path(max(eligible)[2])


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def metric_state(metrics):
    return {name: {key: value.tolist() for key, value in vars(m).items()} for name, m in metrics.items()}


def restore_metrics(values):
    metrics = {name: Metrics() for name in ROUTES}
    if values:
        if set(values) != set(metrics): raise RuntimeError('diagnostic routes differ')
        for name, fields in values.items():
            if set(fields) != set(vars(metrics[name])): raise RuntimeError('metric fields differ')
            for key, value in fields.items():
                value = np.asarray(value, np.int64)
                if value.shape != getattr(metrics[name], key).shape or np.any(value < 0):
                    raise RuntimeError('invalid saved diagnostic counts')
                setattr(metrics[name], key, value)
    return metrics


def summarize(result):
    lines = ['===== FROZEN WM / CCR LOSS-ACTION-SURROGATE AUDIT =====',
        f"status={result['status']}; source_update={result['source_updates']}; source_epochs={result['source_epochs']}",
        f"source={result['source_checkpoint']}",
        f"TRAIN gradient windows={sum(r['windows'] for r in result['gradients'])}; DEV windows={result['dev_windows']}",
        'No optimizer/clip/updates, threshold search, oracle deployment, new bank or old cache writes.']
    lines.append('===== WM weighted gradients BEFORE clip: mean batch cosine / norms =====')
    groups = sorted({key for r in result['gradients'] for key in r['groups']})
    for group in groups:
        rows = [r['groups'][group] for r in result['gradients'] if group in r['groups']]
        valid = [r['cosine'] for r in rows if r['cosine'] is not None]
        cosine = f'{np.mean(valid):+.4f}' if valid else 'undefined_zero_gradient'
        lines.append(f"{group}: cosine={cosine} negative_batches={sum(v < 0 for v in valid)}/{len(valid)} "
                     f"occ_norm={np.mean([r['occupancy_norm'] for r in rows]):.6g} "
                     f"motion_norm={np.mean([r['motion_norm'] for r in rows]):.6g}")
    lines.append('Local gradient agreement/conflict is NOT a causal loss-utility ablation or proof of convergence.')
    if result['dev_windows']:
        metrics = result['metrics']
        lines.append('===== SAME DEV POPULATION / hard versus continuous soft =====')
        for name in ROUTES:
            row = metrics[name]
            lines.append(f"{name:18s} mIoU={row['mIoU']:.6f} IoU={row['IoU']:.6f} MovingMicro={row['MovingMicro']:.6f}")
        for kind in ('hard', 'soft'):
            base, added, removed = (metrics[kind+'_'+name] for name in ('transport', 'add_only', 'add_remove'))
            lines.append(f"{kind} ADD_increment dMiOU={added['mIoU']-base['mIoU']:+.6f}; "
                         f"REMOVE_increment dMiOU={removed['mIoU']-added['mIoU']:+.6f}")
        for name in ('transport', 'add_only', 'add_remove'):
            lines.append(f"soft_minus_hard {name}: dMiOU="
                         f"{metrics['soft_'+name]['mIoU']-metrics['hard_'+name]['mIoU']:+.6f}")
        lines.append('Soft readout uses training continuous context; hard uses deployed context/thresholds. No route selection.')
        lines.append('===== UNIQUE VOXEL edits / report horizons 1/2/3s combined =====')
        for key in ('add_quality', 'remove_incremental_quality'):
            lines.append(key+' '+json.dumps(result['funnel'][key]))
        lines.append(f"ADD wrong new voxels not undoable by same-pass REMOVE={result['funnel']['add_errors_not_undoable_by_remove']}; "
                     f"actual REMOVE touches new ADD voxels={result['funnel']['remove_changes_to_new_add_voxels']}")
        lines.append('===== ACTION FUNNEL / row histograms, not threshold optimization =====')
        lines.append('score_edges='+json.dumps(SCORE_EDGES.tolist()))
        for name, groups in result['funnel']['actions'].items():
            for role, row in groups.items(): lines.append(name+'/'+role+' '+json.dumps(row))
        base, final = metrics['hard_transport'], metrics['hard_add_remove']
        changes = {}
        for cls in range(17):
            values = [final['per_horizon'][str(h)]['semantic_per_class'][str(cls)]
                      -base['per_horizon'][str(h)]['semantic_per_class'][str(cls)] for h in (1., 2., 3.)]
            values = [v for v in values if np.isfinite(v)]
            if values: changes[str(cls)] = float(np.mean(values))
        lines.append('lowest_class_dIoU_pp='+json.dumps(sorted(changes.items(), key=lambda r: r[1])[:5]))
        lines.append('soft_context_vs_hard_action_scores='+json.dumps(result['head_gap']))
    lines += ['Source weights/Adam/RNG/checkpoints are READ ONLY. Gradients use TRAIN only; DEV is evaluation only.',
              'Diagnostic wall time is NOT FPS; no automatic retraining, loss-weight tuning or model promotion.']
    return '\n'.join(lines)+'\n'


def main(argv=None, stop_event=None):
    p = legacy.parser()
    p.description = __doc__
    p.add_argument('--checkpoint-root', default='outputs/p0_f9_joint_occ_ego')
    p.add_argument('--train-windows', type=int, default=16)
    p.add_argument('--resume-audit', action='store_true')
    a = p.parse_args(argv)
    if a.mode != 'gt' or a.training_backend != 'fused' or not 4 <= a.train_windows <= 64 or a.eval_population != 'dev64':
        raise ValueError('GT-only, TRAIN4..64, fixed DEV64 required')
    if (not 1 <= a.cpu_workers <= 20 or not 1 <= a.prefetch_workers <= 8
            or not 0 <= a.frame_cache_mib <= 8192
            or not 0 <= a.geometry_ram_mib <= 4096 or not 0 <= a.descriptor_ram_mib <= 4096):
        raise ValueError('bounded CPU/RAM budgets required')
    device = legacy.require_cuda(a.device)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    out = Path(a.out_dir).resolve()
    if a.resume_audit:
        prior = json.loads((out/'contract.json').read_text(encoding='utf-8'))
        source_checkpoint = Path(prior['source_checkpoint'])
        saved = torch.load(out/'source_snapshot.pt', map_location='cpu', weights_only=False)
        digest = legacy.sha256(out/'source_snapshot.pt')
    else:
        source_checkpoint = Path(a.resume).resolve() if a.resume else discover_checkpoint(a.checkpoint_root).resolve()
        if out.exists() or source_checkpoint.parent == out or source_checkpoint.parent in out.parents:
            raise RuntimeError('NEW diagnostic directory OUTSIDE source run required')
        out.mkdir(parents=True)
        digest = snapshot_checkpoint(source_checkpoint, out/'source_snapshot.pt')
        saved = torch.load(out/'source_snapshot.pt', map_location='cpu', weights_only=False)
    c = saved['contract']
    if (saved.get('protocol') != PROTOCOL or c.get('modes') != ['gt'] or c.get('training_backend') != 'fused'
            or c.get('implementation') not in (legacy_fingerprint(root), execution_fingerprint(root))):
        raise RuntimeError('current fused/sequential-h2 GT-only checkpoint required')
    print('DIAG_SOURCE '+json.dumps(dict(checkpoint=str(source_checkpoint), update=saved['updates'],
          completed_epochs=saved['epoch'], train_gradient_windows=a.train_windows, dev_windows=64,
          optimizer_steps=0, snapshot_sha256=digest)), flush=True)
    if c['execution']['torch_version'] != str(torch.__version__) or c['execution']['cuda_version'] != torch.version.cuda:
        raise RuntimeError('use original checkpoint Torch/CUDA environment')
    for key, expected in c['data'].items():
        if legacy.sha256(getattr(a, key)) != expected: raise RuntimeError('original source data differs: '+key)
    config = legacy.config_from_mapping_v17(c['model']['motion'])
    model = JointOccEgo(config, width=c['model']['repair']['width']).to(device)
    if model.configs() != c['model']: raise RuntimeError('model architecture differs')
    model.load_state_dict(saved['models']['gt'], strict=True)
    initial_model_digest = model_digest(model)
    pcfg = legacy.make_prepare_config(legacy.load_runtime_config(a.config, a.override))
    provider = legacy.PilotProvider(a.base_checkpoint, legacy.CLEAN_SHA256, pcfg, device, a.cpu_workers, model, None)
    provider.raw_io_workers = 1
    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    from real_motion.surface_projection_execution import SurfaceMapExecution
    from real_motion.strong_majority_execution import ParallelNativeMajority
    provider.ccr_execution = SurfaceMapExecution(CanonicalCpuExecution('native_parallel', min(4, a.cpu_workers)))
    provider.strong_majority = ParallelNativeMajority(min(4, a.cpu_workers))
    train_source = legacy.CachedColumnSource(legacy.NuScenesWindowSource(a.dataroot, info_pkl=a.train_info),
                                            a.frame_cache_mib, copy_on_insert=False)
    dev_source = legacy.CachedColumnSource(legacy.NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info),
                                          a.frame_cache_mib, copy_on_insert=False)
    provider.ccr_history_cache_source, provider.ccr_val_history_cache_source = train_source, dev_source
    try:
        caches = attach_caches(provider, a, root)
        if {k: caches[k]['namespace'] for k in ('train', 'val')} != c['cache_namespace']:
            raise RuntimeError('original read-only history namespaces required')
        _, records = legacy.load_cache(a.train_cache)
        keys = legacy.select_scene_balanced_round_robin([tuple(k) for k in c['train_keys']], a.train_windows)
        train = legacy.align_records(records, keys); del records
        _, records = legacy.load_cache(a.dev_cache)
        _, keys64, _ = legacy.load_manifest(a.population_manifest)
        keys64 = [tuple(k) for k in keys64]
        if len(keys64) != 64 or not set(keys64) <= {tuple(k) for k in c['eval_keys']}:
            raise RuntimeError('original fixed DEV64 must be contained in source evaluation population')
        dev = legacy.align_records(records, keys64); del records
        if {r['scene_name'] for r in train} & {r['scene_name'] for r in dev}: raise RuntimeError('TRAIN/DEV scene overlap')
        contract = dict(protocol=AUDIT_PROTOCOL, source_checkpoint=str(source_checkpoint), source_sha256=digest,
            model_digest=initial_model_digest, train_keys=[[r['scene_name'], r['t0_token']] for r in train],
            dev_keys=keys64, thresholds=c['thresholds'], weights=c['weights'], seed=c['seed'],
            implementation=legacy.stable_json_fingerprint({p.name: legacy.sha256(p) for p in Path(__file__).parent.glob('*.py')}),
            source_implementation=c['implementation'], data=c['data'], cache_namespace=c['cache_namespace'])
        if a.resume_audit:
            if prior != json.loads(json.dumps(contract)): raise RuntimeError('diagnostic contract changed; use NEW output')
            state = json.loads((out/'state.json').read_text(encoding='utf-8'))
        else:
            legacy.write_json(out/'contract.json', contract)
            state = dict(gradients=[], dev_windows=0, metric_counts=None, funnel={}, head_gap=dict(max_abs=0., sum_abs=0., elements=0))
        descriptors = HistoryInputs(provider, ram_mib=a.descriptor_ram_mib)
        metrics = restore_metrics(state['metric_counts'])
        start = time.perf_counter()
        def save():
            state['metric_counts'] = metric_state(metrics)
            legacy.write_json(out/'state.json', state)
        def progress(event, **values):
            with (out/'progress.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(event=event, **values))+'\n')
            print(event+' '+json.dumps(values), flush=True)
        groups = legacy.epoch_groups(train, c['seed'], 0, c['window_batch'], c['source_budget'])
        if len(state['gradients']) > len(groups) or not 0 <= state['dev_windows'] <= len(dev):
            raise RuntimeError('invalid diagnostic cursor')
        halted = False
        for index, rows in enumerate(prefetched_batches(provider, train_source, groups, descriptors, workers=a.prefetch_workers)):
            if index < len(state['gradients']): continue
            if stop_event is not None and stop_event.is_set(): halted = True; break
            with preserve_training_rng():
                torch.manual_seed(c['seed']+index)
                row = gradient_batch(model, provider, rows, object_weight=c['weights'][0], lovasz_weight=c['weights'][2])
            state['gradients'].append(row); save()
            progress('WM_GRADIENT_AUDIT', batch=index+1, windows=row['windows'], groups=row['groups'])
        if not halted:
            model.eval().requires_grad_(False)
            sessions = {id(model.columns): BothActionsExecution(model.columns, device, graphs=True,
                                                                capture_full_chunks_only=True)}
            provider.paired_eval_sessions = sessions
            try:
                for rows in prefetched_batches(provider, dev_source, [[r] for r in dev[state['dev_windows']:]],
                                               descriptors, workers=a.prefetch_workers):
                    if stop_event is not None and stop_event.is_set(): halted = True; break
                    row = rows[0]
                    predictions, funnels, head_gap = evaluation_window(model, provider, row, remove_threshold=c['thresholds'][1])
                    window = window_from_record(row[0])
                    support = gt_moving_support_sequence(dev_source.nusc, window.t0_token, window.future_tokens,
                        tuple(.5*(h+1) for h in range(6)), grid=pcfg.grid, workers=provider.workers)
                    moving = moving_support_masks(support, row[1]['future_gt_occ'].shape[1:])
                    for name, values in predictions.items():
                        for i, h in enumerate((1, 3, 5)): metrics[name].update(i, values[i], row[1]['future_gt_occ'][h], moving[h])
                    for funnel in funnels: merge_counts(state['funnel'], funnel)
                    state['head_gap']['max_abs'] = max(state['head_gap']['max_abs'], head_gap['max_abs'])
                    for key in ('sum_abs', 'elements'): state['head_gap'][key] += head_gap[key]
                    state['dev_windows'] += 1
                    if state['dev_windows'] % 8 == 0: save(); progress('CCR_AUDIT', windows=state['dev_windows'], seconds=time.perf_counter()-start)
                    del predictions, funnels
            finally:
                for session in sessions.values(): session.close()
        save()
        if model_digest(model) != initial_model_digest: raise RuntimeError('read-only diagnosis modified model state')
        result = dict(status='stopped' if halted else 'complete', source_checkpoint=str(source_checkpoint),
            source_updates=saved['updates'], source_epochs=saved['epoch'], source_sha256=digest,
            contract=contract, **{k: state[k] for k in ('gradients', 'dev_windows', 'funnel', 'head_gap')},
            metrics={name: m.compute() for name, m in metrics.items()} if state['dev_windows'] else {},
            seconds_this_invocation=time.perf_counter()-start, scientific_updates=0, source_read_only=True,
            route='diagnostic_only_no_automatic_retraining')
        legacy.write_json(out/'evaluation.json', result)
        (out/'summary.txt').write_text(summarize(result), encoding='utf-8')
        print(summarize(result), flush=True)
        print('RESULT: '+str(out/'summary.txt'), flush=True)
    finally:
        for name in ('ccr_history_cache', 'ccr_val_history_cache'):
            cache = getattr(provider, name, None)
            if cache is not None: cache.close()
        provider.ccr_execution.close(); provider.strong_majority.close()
    return 0


if __name__ == '__main__':
    stopped = threading.Event()
    def stop(*_):
        stopped.set()
        print('Stop requested; save after complete diagnostic batch/window. No training updates.', flush=True)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    main(stop_event=stopped)
