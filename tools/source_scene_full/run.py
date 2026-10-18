"""Stage2 architecture, RANDOM FULL TRAIN20430 x2; fixed H6/dense-valid."""
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
from tools.dense_valid_ddp.run import eval_keys
from tools.source_scene_stage1.history import batches
from tools.source_scene_stage2.adapter import DirectProvider, forward_window
from tools.source_scene_stage2.history import DirectHistory
from tools.source_scene_stage2.model import make_model
from tools.source_scene_stage2.evaluation import evaluate


PROTOCOL = 'scene_stage2_RANDOM_fullTRAIN20430x2_dense_valid_H6_v1'


def plan(all_train, all_dev, path, seed):
    if len(all_train) != 20430 or len(all_dev) != 4369: raise ValueError('TRAIN20430 / VAL4369 required')
    train_keys = base.select_scene_balanced_round_robin(base.record_keys(all_train), len(all_train))
    train = base.align_records(all_train, train_keys)
    evaluation = base.align_records(all_dev, eval_keys(path))
    if {r['scene_name'] for r in train} & {r['scene_name'] for r in evaluation}: raise ValueError('TRAIN/DEV scene overlap')
    return train_keys, evaluation, [base.epoch_groups(train, seed, e, 4, 128) for e in range(2)]


def evaluation_due(update, batch):
    return update > 0 and (update % 2000 == 0 or batch == 0)


def checkpoint_names(update, epoch, batch):
    names = []
    if update > 0 and update % 2000 == 0: names.append(f'step_{update:07d}.pt')
    if update > 0 and batch == 0: names.append(f'epoch_{epoch:04d}.pt')
    return names


def parser():
    p = base.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=1., checkpoint_every=2000, seed=20261011, cpu_workers=4,
        prefetch_workers=2, frame_cache_mib=512, geometry_ram_mib=128, descriptor_ram_mib=128,
        eval_population='dev64')
    p.add_argument('--eval-reference', required=True, help='same remaining64 keys ONLY; no scores or weights imported')
    p.add_argument('--stc-root', required=True)
    return p


def validate_resume(saved, contract, groups):
    if saved.get('protocol') != PROTOCOL or saved['contract'] != contract: raise RuntimeError('not SAME stage2 contract')
    epoch, batch, updates = (int(saved[k]) for k in ('epoch', 'batch', 'updates'))
    if (not 0 <= epoch <= len(groups) or epoch == len(groups) and batch != 0 or
            epoch < len(groups) and not 0 <= batch < len(groups[epoch]) or
            updates != sum(map(len, groups[:epoch]))+batch):
        raise RuntimeError('invalid complete-update cursor')
    if len(saved['rank_rng']) != contract['world_size']: raise RuntimeError('rank count differs')
    return epoch, batch, updates


def summary(result):
    c = result['contract']; lines = ['===== STAGE2 / RANDOM FULL TRAIN x2 =====', 'status='+result['status'],
        f'RANDOM TRAIN20430 x2; updates={result["updates"]}/{c["total_updates"]}; epoch={result["epoch"]}; parameters={c["parameters"]}',
        'No stage1 trained weights imported; common RANDOM Encoder/Reader/Temporal/Decoder/Refiner initialization.',
        'KTA only residual skip, NOT query input. Same pinned identities and original two-frame KTA.',
        'No ICP computed/consumed. Reader validity=track_valid; learned relative history yaw, t0 gauge fixed zero.',
        'History feature points move DIRECTLY from their own source center to predicted future center.',
        'Same Geometry Transport/Refiner; ONLY full6 CE+Lovasz, no ego/motion/existence/alignment loss.',
        'GT future ego / dense-valid TRAIN+OCC+STC; same remaining64; no automatic promotion or extension.',
        'GLOBAL B4/source128; whole TWO-epoch cosine. Save/eval every2000 GLOBAL updates AND epoch ends.',
        'Training may use two GPUs; every OCC/STC evaluation uses rank0 ONE GPU only. NOT FPS.']
    for update, e in sorted(result['evaluations'].items(), key=lambda v: int(v[0])):
        lines.append(f'===== update={update}; windows={e["windows"]} =====')
        for route, m in e['variants'].items(): lines.append(f'{route:28s} {m["mIoU"]:.6f} {m["IoU"]:.6f} {m["MovingMicro"]:.6f}')
    return '\n'.join(lines)+'\n'


def launch(a, event, ctx):
    if (a.mode != 'gt' or a.epochs != 2 or a.train_fraction != 1. or a.seed != 20261011 or
        a.window_batch_size != 4 or a.source_budget != 128 or a.training_backend != 'fused' or
        a.evaluate_only or a.max_updates < 0 or a.checkpoint_every != 2000 or a.lovasz_weight != 1. or
        a.eval_population != 'dev64' or not 1 <= a.cpu_workers <= 20 or not 1 <= a.prefetch_workers <= 8):
        raise ValueError('fixed RANDOM TRAIN20430 x2 / B4-source128 / H6 / checkpoint2000 required')
    for name in ('frame_cache_mib', 'geometry_ram_mib', 'descriptor_ram_mib'):
        if not 0 <= getattr(a, name) <= 4096: raise ValueError('bounded CPU RAM required')
    root = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve(); device = ctx.device
    if device.type != 'cuda' or not torch.cuda.is_bf16_supported(): raise RuntimeError('original BF16 CUDA required')
    torch.set_num_threads(1)
    torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, a.override))
    if pcfg.frame_dt_s != .5: raise ValueError('original .5s pair convention required')
    architecture = torch.load(a.base_checkpoint, map_location='cpu', weights_only=False)['model_config']
    cfg = replace(base.config_from_mapping_v17(architecture), history_frames=4)
    _, all_train = base.load_cache(a.train_cache); _, all_dev = base.load_cache(a.dev_cache)
    keys, evaluation, groups = plan(all_train, all_dev, a.eval_reference, a.seed); total = sum(map(len, groups))
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
    folders = ('tools/source_scene_full', 'tools/source_scene_stage2', 'tools/source_scene_stage1', 'tools/shared_source_track',
               'tools/shared_history_experiments/full_gt', 'tools/shared_history_experiments/refine')
    files = [p for folder in folders for p in (root/folder).glob('*.py')]
    files += [root/name for name in (
        'tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
        'tools/joint_occ_experiments/data.py', 'tools/history_validity/data.py',
        'tools/dense_valid_ddp/engine.py', 'tools/dense_valid_ddp/run.py',
        'tools/real_motion/joint_training_distributed.py', 'tools/joint_occ_experiments/soft_geometry.py',
        'tools/joint_occ_experiments/fast_autograd.py', 'tools/joint_occ_experiments/batching/geometry.py',
        'real_motion/shared_source_track.py', 'real_motion/source_evidence_audit.py',
        'real_motion/strong_w2det.py', 'real_motion/runtime_fastpath.py',
        'tools/real_motion/benchmark_p0_f9_v18_runtime.py', 'real_motion/local_st_world_model.py')]
    implementation = {str(p.relative_to(root)): base.sha256(p) for p in files}
    inventory = []
    for record in evaluation:
        for token in record['history_tokens'][-4:]:
            path = (Path(a.stc_root)/record['scene_name']/str(token)/'labels.npz').resolve(); st = path.stat()
            inventory.append((str(path), st.st_size, st.st_mtime_ns))
    contract = dict(protocol=PROTOCOL, arm='direct', model=model.configs(), world_size=ctx.world_size,
        parameters=sum(p.numel() for p in model.parameters()), source_policy='shared_pinned', registration='none',
        train_keys=[list(k) for k in keys], eval_keys=base.record_keys(evaluation), epochs=2, total_updates=total,
        epochs_updates=list(map(len, groups)), seed=a.seed, objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.),
        learning_rates=[a.motion_lr, a.repair_lr], schedule='whole_2_epoch_cosine_floor0.01',
        known='TRAIN_OCC_STC_dense_valid', precision='original_BF16_geometry_FP64_H6',
        data={n: base.sha256(getattr(a, n)) for n in ('config', 'train_cache', 'dev_cache', 'base_checkpoint', 'train_info', 'dev_info')},
        overrides=a.override, STC_inventory=sorted(set(inventory)), implementation=implementation,
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, cpu_workers=a.cpu_workers, prefetch_workers=a.prefetch_workers),
        eval_reference_sha256=base.sha256(a.eval_reference), checkpoint_every=2000,
        initialization='RANDOM; old checkpoint architecture ONLY; no stage1/stage2/full-GT trained weights')
    contract = json.loads(json.dumps(contract))
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

    def save(extra=()):
        states = ctx.gather(ctx.rng_state(rng))
        if ctx.primary:
            payload = dict(protocol=PROTOCOL, contract=contract, epoch=epoch, batch=batch, updates=updates,
                models={'gt': model.state_dict()}, optimizers={'gt': optimizer.state_dict()}, rank_rng=states, evaluations=evaluations)
            save_resume_checkpoint(out/'last.pt', payload)
            for name in extra: save_resume_checkpoint(out/name, payload)
            publish()
        ctx.barrier()
    def evaluate_current():
        key = str(updates)
        if not evaluation_due(updates, batch) or key in evaluations: return True
        states = ctx.gather(ctx.rng_state(rng)); message = None
        if ctx.primary:
            iterator = None
            try:
                path = out/f'eval_step_{updates:07d}_state.json'; saved = json.loads(path.read_text()) if path.exists() else None
                if saved and (saved['contract'] != contract or saved['updates'] != updates): raise RuntimeError('eval provenance differs')
                def store(value): base.write_json(path, dict(contract=contract, updates=updates, state=value))
                iterator = batches(provider, dev_source, [[r] for r in evaluation], history, workers=a.prefetch_workers)
                report = evaluate(model, provider, dev_source, iterator, stc_root=a.stc_root,
                    saved=saved['state'] if saved else None, save=store, stop_event=event)
                if report['windows'] != 64 or report['keys'] != contract['eval_keys']: raise RuntimeError('eval population differs')
                base.write_json(out/f'eval_step_{updates:07d}.json', dict(contract=contract, updates=updates, evaluation=report))
                message = dict(status='complete', report=report)
            except InterruptedError: message = dict(status='stopped')
            except Exception as error: message = dict(status='error', error=repr(error))
            finally:
                if iterator is not None: iterator.close()
        message = ctx.broadcast(message); ctx.restore_rng(states, rng)
        if message['status'] == 'error': raise RuntimeError(message['error'])
        if message['status'] == 'stopped': return False
        evaluations[key] = message['report']; result['evaluations'] = evaluations
        return True
    try:
        if ctx.primary:
            base.write_json(out/'contract.json', contract); base.write_json(out/'cache_audit.json', cache_report)
            print(f'SCENE_FULL RANDOM {updates}/{total}; world={ctx.world_size}; no ICP, no KTA query; NEW={out}', flush=True)
        if not evaluate_current(): event.set()
        while epoch < 2 and not ctx.stop(event.is_set()) and not (a.max_updates and updates >= a.max_updates):
            remaining = groups[epoch][batch:]
            iterator = batches(provider, train_source, [g[ctx.rank::ctx.world_size] for g in remaining],
                               history, workers=a.prefetch_workers)
            try:
                for group in remaining:
                    tick = time.perf_counter(); rows = next(iterator)
                    if ctx.stop(event.is_set()): break
                    stats = train_step(model, optimizer, provider, rows, ctx, global_windows=len(group),
                        update=updates+1, total_updates=total, forward=forward_window)
                    ranks = ctx.gather(dict(stats, pipeline_seconds=time.perf_counter()-tick))
                    updates += 1; batch += 1
                    if batch == len(groups[epoch]): epoch += 1; batch = 0
                    result.update(epoch=epoch, updates=updates)
                    if ctx.primary:
                        wall = max(r['pipeline_seconds'] for r in ranks)/len(group)
                        item = dict(event='scene_full_train', update=updates, epoch=epoch, windows=len(group), seconds_per_window=wall, **stats)
                        with (out/'progress.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(item)+'\n')
                        if updates == 1 or updates % 16 == 0: print(f'SCENE_FULL {updates}/{total} epoch={epoch} CE={stats["CE"]:.4f} seconds/window={wall:.4f}', flush=True)
                    extra = checkpoint_names(updates, epoch, batch)
                    if extra: save(extra)
                    if evaluation_due(updates, batch):
                        if not evaluate_current(): event.set()
                        save()
                    if ctx.stop(event.is_set()) or a.max_updates and updates >= a.max_updates or batch == 0: break
            finally: iterator.close()
        save(); result['status'] = 'complete' if epoch == 2 and str(total) in evaluations else 'stopped_at_complete_update_or_eval_window'
        publish()
        if ctx.primary: print(summary(result), flush=True)
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
                lock = out.parent/'.scene_full_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
                stack.enter_context(evaluation_lock(lock))
                if a.resume:
                    if Path(a.resume).resolve() != out/'last.pt': raise RuntimeError('resume SAME new output last.pt only')
                    header = torch.load(a.resume, map_location='cpu', weights_only=False)
                    if header.get('protocol') != PROTOCOL: raise RuntimeError('only SAME full-TRAIN two-epoch checkpoint can resume here')
                    del header
                elif out.exists(): raise RuntimeError('NEW output required; old results never overwritten')
                out.mkdir(parents=True, exist_ok=True)
            ctx.barrier(); return launch(a, event, ctx)
    finally: ctx.close()


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event=event) or 0)
