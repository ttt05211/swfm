"""Read-only completed500 snapshots; no source Adam/RNG or training writes."""
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.shared_history_experiments.refine.model import ResidualJoint
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.shared_history_experiments.loss_controls.run import capped_groups
from tools.shared_history_experiments.loss_minimal.run import PROTOCOL as GT_PROTOCOL
from tools.scene_ego_head.integration import SceneEgoRefiner
from tools.scene_ego_head.experiment.internal_only.run import PROTOCOL as EGO_PROTOCOL
from tools.scene_ego_head.experiment.planner import PlannerCache
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint, preserve_training_rng
from . import engine

PROTOCOL = 'readonly_scene_ego_checks_and_gt500_external_v1'


def validate_source(saved, mode):
    expected = EGO_PROTOCOL if mode == 'pred' else GT_PROTOCOL
    c = saved.get('contract', {})
    if (saved.get('protocol') != expected or c.get('protocol') != expected or saved.get('updates') != 500
            or saved.get('epoch') != 1 or saved.get('batch') != 0 or c.get('total_updates') != 500
            or c.get('modes') != [mode] or len(c.get('train_keys', [])) != 4086
            or len(c.get('eval_keys', [])) != 512 or c.get('window_batch') != 4 or c.get('source_budget') != 128):
        raise RuntimeError('completed matched RANDOM500 source ONLY; no partial/old/full-loss checkpoint')


def build_model(config, spec, scene):
    h, r = spec['shared_history'], spec['repair']
    kwargs = dict(feature_width=h['width'], z_bins=h['z_bins'], memory_width=r['memory_width'],
                  width=r['width'], chunk_size=r['chunk_size'])
    if scene:
        e = spec['ego']; kwargs.update(ego_width=e['width'], ego_heads=e['heads'], pool_side=e['pool_side'])
    model = (SceneEgoRefiner if scene else ResidualJoint)(config, **kwargs)
    actual = model.configs(); source = {k: v for k, v in spec.items() if k not in ('experiment', 'loss_control')}
    if actual != source: raise RuntimeError('source architecture or numerical readout differs from current model')
    return model


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def locate_gt500(root):
    candidates = sorted((root/'outputs/p0_f9_shared_history').glob('loss500_minimal_*/last.pt'),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        try: validate_source(saved, 'gt')
        except RuntimeError: continue
        return path
    raise RuntimeError('no completed loss500_minimal checkpoint found; set GT500_CKPT explicitly')


def summarize(result):
    task = result['contract']['model']['diagnostic']['task']
    lines = ['===== FROZEN EGO CHECKS / GT500 EXTERNAL INFERENCE =====', 'status='+result['status'],
        f'task={task}; scientific_updates=0; no optimizer steps; source Adam/RNG/checkpoints unchanged.',
        'GT-derived navigation declared; no pose alignment, threshold/head selection or new training.']
    report = result.get('evaluation')
    if report and task == 'ego_checks':
        lines += [f"DEV windows={report['windows']}; trajectory ONLY, no dense forecasting.", 'route ADE_m FDE3s_m yaw3s_deg']
        for name, values in report['trajectory'].items():
            lines.append(f"{name} {values['ADE_m']:.6f} {values['FDE_3s_m']:.6f} {values['yaw_mean_deg'][-1]:.6f}")
        lines += ['head_minus_prior='+json.dumps(report['head_minus_prior']), 'trajectory='+json.dumps(report['trajectory']),
            f"TRAIN gradient windows={len(report['gradient_rows'])}; fixed first trained batches; NOT full TRAIN metrics."]
        for group in ('pose_R10', 'shared_F', 'ego_parameters'):
            values = [r['gradients'][group] for r in report['gradient_rows']]
            defined = [r['cosine'] for r in values if r['cosine'] is not None]
            lines.append(group+' '+json.dumps(dict(
                mean_occupancy_norm=float(np.mean([r['occupancy_norm'] for r in values])) if values else None,
                mean_ego_norm=float(np.mean([r['ego_norm'] for r in values])) if values else None,
                mean_cosine=float(np.mean(defined)) if defined else None,
                negative_cosine=sum(v < 0 for v in defined), defined_windows=len(defined))))
        lines.append('Pose gradient yaw scaled by R10; directions are diagnostics, not proof of harmful joint training.')
    elif report:
        p = report['population']
        lines += [f"requested={p['requested']}; SAME covered windows={report['windows']}; missing={len(p['missing_keys'])}",
            'GT500 body stays GT-trained; external poses drive ALL future geometry, no internal head added.',
            'Pred500 internal is separately trained; same-population comparison, NOT head-only ablation.']
        for route, values in report['variants'].items():
            for name, m in values.items():
                lines.append(f"{route}_{name} mIoU={m['mIoU']:.6f} IoU={m['IoU']:.6f} MovingMicro={m['MovingMicro']:.6f}")
                for h, row in m['per_horizon'].items():
                    lines.append(f"  {h}s mIoU={row['mIoU']:.6f} IoU={row['IoU']:.6f} MovingMicro={row['MovingMicro']:.6f}")
        lines += ['external_minus_internal='+json.dumps(report['external_minus_internal']),
                  'trajectory='+json.dumps(report['trajectory'])]
    if 'error' in result: lines.append('error='+result['error'])
    lines.append('No GT-ego reference forecast, deployment promotion, training extension or FPS claim.')
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event()
    p = legacy.parser(); p.description = __doc__
    p.add_argument('--task', choices=('ego_checks', 'gt_external'), required=True)
    p.add_argument('--ego-checkpoint', required=True); p.add_argument('--plan-cache')
    p.add_argument('--diagnostic-resume', action='store_true')
    a = p.parse_args(argv)
    if (not a.evaluate_only or not a.resume or a.epochs != 1 or a.eval_population != 'dev512'
            or a.max_updates or a.train_fraction != .2 or a.window_batch_size != 4 or a.source_budget != 128):
        raise ValueError('read-only completed500/full DEV512 and original data/batch required')
    mode = 'pred' if a.task == 'ego_checks' else 'gt'
    if a.mode != mode: raise ValueError('source mode differs from diagnostic task')
    if a.task == 'gt_external' and not a.plan_cache: raise ValueError('external task requires exact v4 planner cache')
    root = Path(__file__).resolve().parents[3]; out = Path(a.out_dir).resolve()
    requested_source = locate_gt500(root) if a.resume == 'AUTO_GT500' and not a.diagnostic_resume else Path(a.resume).resolve()
    if not a.diagnostic_resume and (out == requested_source.parent or requested_source.parent in out.parents):
        raise ValueError('NEW diagnostic directory OUTSIDE source training directory required')
    lock = out.parent/'.ego_diagnostic_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if a.diagnostic_resume:
            old = json.loads((out/'diagnostic_contract.json').read_text())
            source = out/'frozen/source.pt'; companion_path = out/'frozen/internal.pt'
            source_origin = old['source_origin']; companion_origin = old.get('companion_origin')
        else:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            source = out/'frozen/source.pt'; snapshot_checkpoint(requested_source, source)
            source_origin = str(requested_source)
            companion_path = out/'frozen/internal.pt'; companion_origin = None
            if a.task == 'gt_external':
                companion_origin = str(Path(a.ego_checkpoint).resolve()); snapshot_checkpoint(companion_origin, companion_path)
        frozen = torch.load(source, map_location='cpu', weights_only=False); validate_source(frozen, mode)
        companion = torch.load(companion_path, map_location='cpu', weights_only=False) if a.task == 'gt_external' else None
        if companion:
            validate_source(companion, 'pred')
            for key in ('data', 'train_keys', 'eval_keys', 'cache_namespace', 'seed', 'weights', 'execution'):
                if companion['contract'][key] != frozen['contract'][key]: raise RuntimeError('GT500 and Pred500 population/data/budget differ: '+key)
        fields = vars(a).copy()
        for key in ('diagnostic_resume', 'resume', 'ego_checkpoint'): fields.pop(key)
        identity = dict(protocol=PROTOCOL, args=fields, source_origin=source_origin, companion_origin=companion_origin,
            source_sha256=legacy.sha256(source), companion_sha256=legacy.sha256(companion_path) if companion else None,
            implementation={str(path.relative_to(root)): legacy.sha256(path) for path in list(Path(__file__).parent.glob('*.py'))+
                [root/'tools/scene_ego_head/integration.py', root/'tools/scene_ego_head/model.py',
                 root/'tools/shared_history_experiments/refine/model.py', root/'tools/shared_history_experiments/refine/geometry.py',
                 root/'tools/shared_history_experiments/refine/engine.py', root/'tools/scene_ego_head/experiment/planner.py']},
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda))
        identity = json.loads(json.dumps(identity))
        if a.diagnostic_resume and old != identity: raise RuntimeError('diagnostic snapshot/settings/implementation changed')
        legacy.write_json(out/'diagnostic_contract.json', identity)
        print(f'FROZEN_SOURCE {source_origin}; task={a.task}; scientific_updates=0', flush=True)
        planner = PlannerCache(a.plan_cache) if a.task == 'gt_external' else None
        if planner:
            planner.prepare(frozen['contract']['eval_keys'])
            print(f'PLANNER requested=512 covered={planner.metadata["covered"]}; same intersection for BOTH models', flush=True)
        names = ('parser', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution', 'PROTOCOL', 'restore_checkpoint',
                 'train_arm', 'evaluate', 'epoch_groups', 'summarize', 'save_resume_checkpoint')
        original = {n: getattr(legacy, n) for n in names}
        had_module_print = 'print' in vars(legacy)
        original_print = getattr(legacy, 'print', print)
        def diagnostic_print(*values, **kwargs):
            if values and isinstance(values[0], str):
                if values[0].startswith('JOINT_OCC_RESUME'):
                    values = ('READ_ONLY_LOADED frozen weights at update500; source Adam/RNG NOT restored or updated.', *values[1:])
                elif values[0].startswith('JOINT_OCC RANDOM'):
                    values = ('READ_ONLY_DATA_READY; no training loop or optimizer step will run.', *values[1:])
            original_print(*values, **kwargs)
        holder = {}; source_contract = frozen['contract']
        def factory(config):
            model = build_model(config, source_contract['model'], mode == 'pred')
            holder['main'] = model
            architecture = model.configs()
            model.configs = lambda: dict(frozen_architecture=architecture, diagnostic=dict(task=a.task, source_sha256=identity['source_sha256']))
            return model
        def restore(saved, models, optimizers, constructed):
            validate_source(saved, mode)
            for key in ('modes', 'data', 'train_keys', 'eval_keys', 'cache_namespace', 'seed', 'weights', 'execution',
                        'window_batch', 'source_budget', 'total_updates', 'lr', 'schedule'):
                if constructed[key] != source_contract[key]: raise RuntimeError('source/runtime/input mismatch: '+key)
            model = models[mode]; model.load_state_dict(saved['models'][mode], strict=True)
            holder['digest'] = model_digest(model)
            if companion:
                other = build_model(model.transport.config, companion['contract']['model'], True).to(next(model.parameters()).device)
                other.load_state_dict(companion['models']['pred'], strict=True); other.eval().requires_grad_(False)
                holder['companion'] = other; holder['companion_digest'] = model_digest(other)
            # Adam/RNG are intentionally NOT loaded or updated by this diagnostic.
            return 1, 0, 500, dict(training=[], gradient_links={}, epochs=[])
        def forbidden(*args, **kwargs): raise RuntimeError('read-only diagnostic cannot train/save a training checkpoint')
        def evaluate(models, provider, source_adapter, batches, **kwargs):
            if a.task == 'gt_external':
                return engine.matched_external(models['gt'], holder['companion'], provider, source_adapter, batches, planner, **kwargs)
            gradient_path = out/'train_gradient_state.json'
            gradients = json.loads(gradient_path.read_text()) if gradient_path.exists() else dict(source_sha256=identity['source_sha256'], rows=[])
            if gradients['source_sha256'] != identity['source_sha256']: raise RuntimeError('gradient source changed')
            _, all_train = legacy.load_cache(a.train_cache)
            records = legacy.align_records(all_train, [tuple(k) for k in source_contract['train_keys']]); del all_train
            selected = original['epoch_groups'](records, a.seed, 0, 4, 128)[:4]
            expected = [[r['scene_name'], r['t0_token']] for group in selected for r in group]
            if [r['key'] for r in gradients['rows']] != expected[:len(gradients['rows'])]: raise RuntimeError('TRAIN gradient prefix differs')
            it = legacy.prefetched_batches(provider, provider.ccr_history_cache_source, selected,
                ResidualHistoryInputs(provider, ram_mib=a.descriptor_ram_mib), workers=a.prefetch_workers)
            offset = 0
            try:
                for rows in it:
                    for row in rows:
                        offset += 1
                        if offset <= len(gradients['rows']): continue
                        if event.is_set(): raise InterruptedError('saved complete TRAIN gradient windows')
                        gradients['rows'].append(engine.gradient_window(models['pred'], provider, row))
                        legacy.write_json(gradient_path, gradients)
                        print(f'EGO_GRADIENT_CHECK {offset}/{len(expected)}', flush=True)
            finally: it.close()
            kwargs.pop('remove_threshold', None)
            report = engine.ego_checks(models['pred'], provider, batches, **kwargs)
            report['gradient_rows'] = gradients['rows']; return report
        inner = []
        for key, value in vars(a).items():
            if key in ('override', 'diagnostic_resume') or value is None: continue
            if key == 'resume': value = str(source)
            if isinstance(value, bool):
                if value: inner.append('--'+key.replace('_', '-'))
            else: inner += ['--'+key.replace('_', '-'), str(value)]
        for value in a.override or []: inner += ['--override', value]
        try:
            legacy.parser = lambda: p
            legacy.print = diagnostic_print
            legacy.JointOccEgo, legacy.HistoryInputs, legacy.BothActionsExecution = factory, ResidualHistoryInputs, NoLegacyHeadGraph
            legacy.PROTOCOL, legacy.restore_checkpoint, legacy.train_arm = PROTOCOL, restore, forbidden
            legacy.evaluate, legacy.summarize, legacy.save_resume_checkpoint = evaluate, summarize, forbidden
            legacy.epoch_groups = lambda *args: capped_groups(original['epoch_groups'], *args)
            with preserve_training_rng(): status = legacy.main(event, inner)
            if 'main' in holder and model_digest(holder['main']) != holder['digest']: raise RuntimeError('frozen main weights changed')
            if companion and model_digest(holder['companion']) != holder['companion_digest']: raise RuntimeError('frozen companion weights changed')
            if legacy.sha256(source) != identity['source_sha256']: raise RuntimeError('source snapshot changed')
            if companion and legacy.sha256(companion_path) != identity['companion_sha256']: raise RuntimeError('companion snapshot changed')
            return status
        finally:
            for n, value in original.items(): setattr(legacy, n, value)
            if had_module_print: legacy.print = original_print
            else: del legacy.print


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
