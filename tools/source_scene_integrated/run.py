"""Isolated joint Scene-Temporal encoder + AdaGN / matched KTA-query control."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import random
import signal
import sys
import threading
import time
import numpy as np
import torch
import torch.distributed as dist
from tools.joint_occ_experiments import run as base
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.dense_valid_ddp.engine import train_step
from tools.source_scene_stage1.run import plan
from tools.source_scene_stage1.history import batches
from tools.source_scene_stage2.adapter import DirectProvider
from tools.source_scene_kta_query.adapter import forward_window
from tools.source_scene_stage2.history import DirectHistory
from .model import make_model
from tools.source_scene_kta_query.evaluation import evaluate
from .report import reference, compare, MATCH_FIELDS

PROTOCOL = 'scene_temporal_AdaGN_joint_RANDOM_TRAIN20x1_v1'


def parser():
    p = base.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=1, train_fraction=.2, seed=20261011, cpu_workers=4,
        prefetch_workers=2, frame_cache_mib=512, geometry_ram_mib=128, descriptor_ram_mib=128,
        eval_population='dev64')
    p.add_argument('--reference-dir', required=True); p.add_argument('--eval-reference', required=True)
    p.add_argument('--stc-root', required=True)
    return p


def validate_resume(saved, contract, groups):
    if saved.get('protocol') != PROTOCOL or saved['contract'] != contract: raise RuntimeError('not SAME integrated contract')
    epoch, batch, updates = (int(saved[k]) for k in ('epoch', 'batch', 'updates'))
    if ((epoch, batch, updates) != (1, 0, len(groups[0])) and
            not (epoch == 0 and 0 <= batch < len(groups[0]) and updates == batch)):
        raise RuntimeError('invalid complete-update cursor')
    if len(saved['rank_rng']) != contract['world_size']: raise RuntimeError('rank count differs')
    return epoch, batch, updates


def summary(result):
    c = result['contract']; lines = ['===== SCENE-TEMPORAL + ADAGN / TRAIN20% x1 =====', 'status='+result['status'],
        f'RANDOM TRAIN4086 x1; updates={result["updates"]}/{c["total_updates"]}; epoch={result["epoch"]}; parameters={c["parameters"]}',
        'No trained weights imported; semantically shared FRESH modules and post-init RNG matched to KTA-query control.',
        'TWO interleaved PreNorm scene/temporal/FFN blocks; one multi-scale attention per block; initial coarse GAP.',
        'No ICP computed/consumed. Reader validity=track_valid; learned relative history yaw, t0 gauge fixed zero.',
        'History feature points move DIRECTLY from their own source center to predicted future center.',
        'Refiner residual-branch AdaGN, affine-free GN and zero condition projections. KTA query/decoder/geometry unchanged; ONLY full6 CE+Lovasz.',
        'GT future ego / dense-valid TRAIN+OCC+STC; same remaining64; no automatic promotion or extension.']
    for e in result['evaluations'].values():
        for route, m in e['variants'].items(): lines.append(f'{route:28s} {m["mIoU"]:.6f} {m["IoU"]:.6f} {m["MovingMicro"]:.6f}')
    return '\n'.join(lines)+'\n'


def launch(a, event, ctx):
    if (a.mode != 'gt' or a.epochs != 1 or a.train_fraction != .2 or a.seed != 20261011 or
        a.window_batch_size != 4 or a.source_budget != 128 or a.training_backend != 'fused' or
        a.evaluate_only or a.max_updates < 0 or a.checkpoint_every < 1 or a.lovasz_weight != 1. or
        a.eval_population != 'dev64' or not 1 <= a.cpu_workers <= 20 or not 1 <= a.prefetch_workers <= 8):
        raise ValueError('fixed RANDOM TRAIN20% x1 / B4-source128 / H6 CE+Lovasz required')
    for name in ('frame_cache_mib', 'geometry_ram_mib', 'descriptor_ram_mib'):
        if not 0 <= getattr(a, name) <= 4096: raise ValueError('bounded CPU RAM required')
    root = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve(); device = ctx.device
    if device.type != 'cuda' or not torch.cuda.is_bf16_supported(): raise RuntimeError('original BF16 CUDA required')
    control = reference(a.reference_dir); torch.set_num_threads(1)
    torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, a.override))
    if pcfg.frame_dt_s != .5: raise ValueError('original .5s pair convention required')
    architecture = torch.load(a.base_checkpoint, map_location='cpu', weights_only=False)['model_config']
    cfg = replace(base.config_from_mapping_v17(architecture), history_frames=4)
    _, all_train = base.load_cache(a.train_cache); _, all_dev = base.load_cache(a.dev_cache)
    keys, evaluation, groups = plan(all_train, all_dev, a.eval_reference, a.seed, 1); total = len(groups[0])
    model = make_model(cfg).to(device)
    if ctx.active:
        for value in model.state_dict().values(): dist.broadcast(value, src=0)
    torch.manual_seed(a.seed+ctx.rank); torch.cuda.manual_seed(a.seed+ctx.rank)
    random.seed(a.seed+ctx.rank); np.random.seed(a.seed+ctx.rank); rng = np.random.default_rng(a.seed+ctx.rank)
    optimizer = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=a.motion_lr, initial_lr=a.motion_lr, weight_decay=1e-4),
        dict(params=model.columns.parameters(), lr=a.repair_lr, initial_lr=a.repair_lr, weight_decay=.01)])
    provider = DirectProvider(a.base_checkpoint, base.CLEAN_SHA256, pcfg, device, a.cpu_workers, model, None)
    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    from real_motion.surface_projection_execution import SurfaceMapExecution
    from real_motion.strong_majority_execution import ParallelNativeMajority
    provider.raw_io_workers = 1
    provider.ccr_execution = SurfaceMapExecution(CanonicalCpuExecution('native_parallel', min(4, a.cpu_workers)))
    provider.strong_majority = ParallelNativeMajority(min(4, a.cpu_workers))
    train_source = base.CachedColumnSource(base.NuScenesWindowSource(a.dataroot, info_pkl=a.train_info), a.frame_cache_mib, copy_on_insert=False)
    dev_source = base.CachedColumnSource(base.NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info), a.frame_cache_mib, copy_on_insert=False)
    provider.ccr_history_cache_source, provider.ccr_val_history_cache_source = train_source, dev_source
    cache_report = base.attach_caches(provider, a, root)
    folders = ('tools/source_scene_integrated', 'tools/source_scene_kta_query', 'tools/source_scene_stage2', 'tools/source_scene_stage1', 'tools/shared_source_track',
               'tools/shared_history_experiments/full_gt', 'tools/shared_history_experiments/refine')
    files = [p for folder in folders for p in (root/folder).glob('*.py')]
    files += [root/'tools/source_motion_audit/engine.py', root/'tools/source_motion_audit/statistics.py']
    for name in control['contract']['implementation']:
        if base.sha256(root/name) != control['contract']['implementation'][name]:
            raise RuntimeError('KTA-query control implementation changed: '+name)
        files.append(root/name)
    implementation = {str(p.relative_to(root)): base.sha256(p) for p in files}
    inventory = []
    for record in evaluation:
        for token in record['history_tokens'][-4:]:
            path = (Path(a.stc_root)/record['scene_name']/str(token)/'labels.npz').resolve(); st = path.stat()
            inventory.append((str(path), st.st_size, st.st_mtime_ns))
    contract = dict(protocol=PROTOCOL, arm='integrated_encoder_AdaGN', model=model.configs(), world_size=ctx.world_size,
        parameters=sum(p.numel() for p in model.parameters()), source_policy='shared_pinned', registration='none',
        train_keys=[list(k) for k in keys], eval_keys=base.record_keys(evaluation), epochs=1, total_updates=total,
        epochs_updates=[total], seed=a.seed, objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.),
        learning_rates=[a.motion_lr, a.repair_lr], schedule='whole_1_epoch_cosine_floor0.01',
        known='TRAIN_OCC_STC_dense_valid', precision='original_BF16_geometry_FP64_H6',
        data={n: base.sha256(getattr(a, n)) for n in ('config', 'train_cache', 'dev_cache', 'base_checkpoint', 'train_info', 'dev_info')},
        overrides=a.override, STC_inventory=sorted(set(inventory)), implementation=implementation,
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, cpu_workers=a.cpu_workers, prefetch_workers=a.prefetch_workers),
        reference_results_sha256=base.sha256(Path(a.reference_dir)/'results.json'))
    contract = json.loads(json.dumps(contract))
    for name in MATCH_FIELDS:
        if contract[name] != control['contract'][name]: raise RuntimeError('unmatched KTA-query control: '+name)
    old_spec = control['contract']['model']
    for name in ('motion', 'reader', 'shared_history'):
        if contract['model'][name] != old_spec[name]: raise RuntimeError('unmatched shared architecture: '+name)
    for name in ('memory_width', 'width', 'chunk_size', 'epsilon'):
        if contract['model']['repair'][name] != old_spec['repair'][name]: raise RuntimeError('unmatched repair shape: '+name)
    cache_report['direct_identity'] = provider.attach_tracks(out/'identity_history', rank=ctx.rank, implementation=implementation)
    history = DirectHistory(provider, ram_mib=a.descriptor_ram_mib)
    epoch = batch = updates = 0; evaluations = {}
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        epoch, batch, updates = validate_resume(saved, contract, groups)
        model.load_state_dict(saved['models']['gt'], strict=True); optimizer.load_state_dict(saved['optimizers']['gt'])
        ctx.restore_rng(saved['rank_rng'], rng); evaluations = saved['evaluations']; del saved
    result = dict(status='running', contract=contract, epoch=epoch, updates=updates, evaluations=evaluations)
    def publish():
        if ctx.primary:
            base.write_json(out/'results.json', result); (out/'summary.txt').write_text(summary(result), encoding='utf-8')
            if result['status'] == 'complete': (out/'comparison.txt').write_text(compare(result, control), encoding='utf-8')
    def save(extra=None):
        states = ctx.gather(ctx.rng_state(rng))
        if ctx.primary:
            payload = dict(protocol=PROTOCOL, contract=contract, epoch=epoch, batch=batch, updates=updates,
                models={'gt': model.state_dict()}, optimizers={'gt': optimizer.state_dict()}, rank_rng=states, evaluations=evaluations)
            save_resume_checkpoint(out/'last.pt', payload)
            if extra: save_resume_checkpoint(out/extra, payload)
            publish()
        ctx.barrier()
    def evaluate_final():
        if epoch != 1 or '1' in evaluations: return True
        states = ctx.gather(ctx.rng_state(rng)); message = None
        if ctx.primary:
            iterator = None
            try:
                path = out/'eval_epoch_0001_state.json'; saved = json.loads(path.read_text()) if path.exists() else None
                if saved and (saved['contract'] != contract or saved['updates'] != updates): raise RuntimeError('eval provenance differs')
                def store(value): base.write_json(path, dict(contract=contract, updates=updates, state=value))
                iterator = batches(provider, dev_source, [[r] for r in evaluation], history, workers=a.prefetch_workers)
                report = evaluate(model, provider, dev_source, iterator, stc_root=a.stc_root,
                    saved=saved['state'] if saved else None, save=store, stop_event=event)
                if report['windows'] != 64 or report['keys'] != contract['eval_keys']: raise RuntimeError('eval population differs')
                base.write_json(out/'eval_epoch_0001.json', dict(contract=contract, updates=updates, evaluation=report))
                message = dict(status='complete', report=report)
            except InterruptedError: message = dict(status='stopped')
            except Exception as error: message = dict(status='error', error=repr(error))
            finally:
                if iterator is not None: iterator.close()
        message = ctx.broadcast(message); ctx.restore_rng(states, rng)
        if message['status'] == 'error': raise RuntimeError(message['error'])
        if message['status'] == 'stopped': return False
        evaluations['1'] = message['report']; result['evaluations'] = evaluations
        return True
    try:
        if ctx.primary:
            base.write_json(out/'contract.json', contract); base.write_json(out/'cache_audit.json', cache_report)
            print(f'SCENE_INTEGRATED RANDOM {updates}/{total}; world={ctx.world_size}; no ICP; KTA query + joint encoder/AdaGN; NEW={out}', flush=True)
        if not evaluate_final(): event.set()
        iterator = batches(provider, train_source, [g[ctx.rank::ctx.world_size] for g in groups[0][batch:]], history, workers=a.prefetch_workers)
        try:
            for group in groups[0][batch:] if epoch == 0 else []:
                if ctx.stop(event.is_set()) or a.max_updates and updates >= a.max_updates: break
                tick = time.perf_counter(); rows = next(iterator)
                stats = train_step(model, optimizer, provider, rows, ctx, global_windows=len(group),
                    update=updates+1, total_updates=total, forward=forward_window)
                ranks = ctx.gather(dict(stats, pipeline_seconds=time.perf_counter()-tick))
                updates += 1; batch += 1
                if batch == total: epoch, batch = 1, 0
                result.update(epoch=epoch, updates=updates)
                if ctx.primary:
                    wall = max(r['pipeline_seconds'] for r in ranks)/len(group)
                    item = dict(event='scene_integrated_train', update=updates, epoch=epoch, windows=len(group), seconds_per_window=wall, **stats)
                    with (out/'progress.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(item)+'\n')
                    if updates == 1 or updates % 16 == 0: print(f'SCENE_INTEGRATED {updates}/{total} CE={stats["CE"]:.4f} seconds/window={wall:.4f}', flush=True)
                if epoch or updates % a.checkpoint_every == 0: save('epoch_0001.pt' if epoch else None)
        finally: iterator.close()
        if not ctx.stop(event.is_set()) and not evaluate_final(): event.set()
        save(); result['status'] = 'complete' if epoch == 1 and '1' in evaluations else 'stopped_at_complete_update_or_eval_window'
        publish()
        if ctx.primary:
            print(summary(result), flush=True)
            if result['status'] == 'complete': print(compare(result, control), flush=True)
    finally:
        provider.close_tracks(); provider.ccr_history_cache.close(); provider.ccr_val_history_cache.close()
        provider.ccr_execution.close(); provider.strong_majority.close()


def main(argv=None, event=None):
    a = parser().parse_args(argv); event = event or threading.Event(); ctx = DistributedTraining(True)
    if not ctx.active: ctx.device = torch.device(a.device)
    out = Path(a.out_dir).resolve()
    try:
        with ExitStack() as stack:
            if ctx.primary:
                lock = out.parent/'.scene_integrated_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
                stack.enter_context(evaluation_lock(lock))
                if a.resume:
                    if Path(a.resume).resolve() != out/'last.pt': raise RuntimeError('resume SAME new output last.pt only')
                    header = torch.load(a.resume, map_location='cpu', weights_only=False)
                    if header.get('protocol') != PROTOCOL: raise RuntimeError('only this integrated run can resume')
                    del header
                elif out.exists(): raise RuntimeError('NEW output required; old results never overwritten')
                out.mkdir(parents=True, exist_ok=True)
            ctx.barrier(); return launch(a, event, ctx)
    finally: ctx.close()


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event=event) or 0)
