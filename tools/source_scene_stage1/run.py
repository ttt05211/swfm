"""Isolated RANDOM TRAIN20% x2, one/two training GPUs; single-GPU DEV64."""
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
from tools.shared_source_track.adapter import TrackProvider
from .model import make_model
from .history import StageHistory, batches
from .evaluation import evaluate

PROTOCOL = 'source_scene_reader_stage1_matched_TRAIN20x2_H6_v1'


def parser():
    p = base.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=.2, seed=20261011, cpu_workers=4,
                   prefetch_workers=2, frame_cache_mib=512, geometry_ram_mib=128,
                   descriptor_ram_mib=128, eval_population='dev64')
    p.add_argument('--arm', choices=('baseline', 'reader'), required=True)
    p.add_argument('--eval-reference', required=True, help='existing clean results.json; eval keys ONLY')
    p.add_argument('--stc-root', required=True)
    return p


def plan(all_train, all_dev, reference, seed):
    if len(all_train) != 20430 or len(all_dev) != 4369: raise ValueError('TRAIN20430 / VAL4369 required')
    keys = base.select_scene_balanced_round_robin(base.record_keys(all_train), 4086)
    train = base.align_records(all_train, keys)
    evaluation = base.align_records(all_dev, eval_keys(reference))
    if {r['scene_name'] for r in train} & {r['scene_name'] for r in evaluation}:
        raise ValueError('TRAIN/DEV scene overlap')
    groups = [base.epoch_groups(train, seed, e, 4, 128) for e in range(2)]
    return keys, evaluation, groups


def validate_resume(saved, contract, groups):
    if saved.get('protocol') != PROTOCOL or saved.get('contract') != contract:
        raise RuntimeError('resume contract changed; old checkpoints are never resumed here')
    epoch, batch, updates = (int(saved[k]) for k in ('epoch', 'batch', 'updates'))
    if (not 0 <= epoch <= 2 or epoch == 2 and batch != 0 or
            epoch < 2 and not 0 <= batch < len(groups[epoch]) or
            updates != sum(map(len, groups[:epoch]))+batch):
        raise RuntimeError('invalid completed-update cursor')
    if len(saved['rank_rng']) != contract['world_size']: raise RuntimeError('rank count changed')
    return epoch, batch, updates


def summary(result):
    c = result['contract']
    lines = ['===== MATCHED SOURCE SCENE READER / STAGE1 =====', 'status='+result['status'],
        f'arm={c["arm"]}; updates={result["updates"]}/{c["total_updates"]}; completed_epochs={result["epoch"]}',
        'RANDOM TRAIN4086 x2; latest shared_pinned identities; no old body/norm/calibration weights.',
        'SAME train keys/order/global B4-source128/AdamW/LR/cosine; common fresh Encoder/Refiner initialization.',
        'Only full6 CE+Lovasz. No ego/motion/existence loss or navigation. GT future ego conditioned.',
        'TRAIN, OCC and STC use dense-valid. Registration/KTA/background/memory/Refiner unchanged.',
        'Reader: sparse multi-scale local + explicit coarse global + XY/time position encoding; masked temporal tokens.',
        'KTA Query conditioning retained. No class/velocity/mask numeric feature in reader arm.',
        'Each epoch SAME remaining64 OCC/STC on ONE GPU; both epochs reported, no best selection.',
        f'parameters={c["parameters"]}; training_world_size={c["world_size"]}; NOT an equal-parameter ablation.']
    for epoch, evaluation in sorted(result['evaluations'].items(), key=lambda p: int(p[0])):
        lines += [f'===== epoch={epoch}; windows={evaluation["windows"]} =====',
                  'route                         mIoU       IoU MovingMicro']
        for route, m in evaluation['variants'].items():
            lines.append(f'{route:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}')
    lines += ['Scientific change: reader AND temporal representation/capacity; not a single-input-channel ablation.',
              'Old defaults/checkpoints/caches unchanged. No automatic promotion/extension. Quality time is NOT FPS.']
    return '\n'.join(lines)+'\n'


def launch(a, event, context):
    if (a.mode != 'gt' or a.epochs != 2 or a.train_fraction != .2 or a.seed != 20261011 or
            a.window_batch_size != 4 or a.source_budget != 128 or a.training_backend != 'fused' or
            a.evaluate_only or a.max_updates < 0 or a.checkpoint_every < 1 or a.lovasz_weight != 1. or
            a.eval_population != 'dev64' or
            not 1 <= a.cpu_workers <= 20 or not 1 <= a.prefetch_workers <= 8 or
            min(a.motion_lr, a.repair_lr) <= 0):
        raise ValueError('fixed TRAIN20% x2 / B4-source128 / H6 CE+Lovasz protocol')
    for name in ('frame_cache_mib', 'geometry_ram_mib', 'descriptor_ram_mib'):
        if not 0 <= getattr(a, name) <= 4096: raise ValueError('bounded RAM required')
    root = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    device = context.device
    if device.type != 'cuda' or not torch.cuda.is_bf16_supported(): raise RuntimeError('original CUDA BF16 required')
    torch.set_num_threads(1)
    torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, a.override))
    if pcfg.frame_dt_s != .5: raise ValueError('original nominal .5s KTA/horizon convention required')
    architecture = torch.load(a.base_checkpoint, map_location='cpu', weights_only=False)['model_config']
    cfg = replace(base.config_from_mapping_v17(architecture), history_frames=4)
    _, all_train = base.load_cache(a.train_cache); _, all_dev = base.load_cache(a.dev_cache)
    train_keys, evaluation, groups = plan(all_train, all_dev, a.eval_reference, a.seed)
    total = sum(map(len, groups))
    model = make_model(a.arm, cfg).to(device)
    if context.active:
        for value in model.state_dict().values(): dist.broadcast(value, src=0)
    torch.manual_seed(a.seed+context.rank); torch.cuda.manual_seed(a.seed+context.rank)
    random.seed(a.seed+context.rank); np.random.seed(a.seed+context.rank)
    rng = np.random.default_rng(a.seed+context.rank)
    optimizer = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=a.motion_lr, initial_lr=a.motion_lr, weight_decay=1e-4),
        dict(params=model.columns.parameters(), lr=a.repair_lr, initial_lr=a.repair_lr, weight_decay=.01)])
    provider = TrackProvider(a.base_checkpoint, base.CLEAN_SHA256, pcfg, device, a.cpu_workers, model, None)
    provider.raw_io_workers = 1
    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    from real_motion.surface_projection_execution import SurfaceMapExecution
    from real_motion.strong_majority_execution import ParallelNativeMajority
    provider.ccr_execution = SurfaceMapExecution(CanonicalCpuExecution('native_parallel', min(4, a.cpu_workers)))
    provider.strong_majority = ParallelNativeMajority(min(4, a.cpu_workers))
    train_source = base.CachedColumnSource(base.NuScenesWindowSource(a.dataroot, info_pkl=a.train_info), a.frame_cache_mib, copy_on_insert=False)
    dev_source = base.CachedColumnSource(base.NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info), a.frame_cache_mib, copy_on_insert=False)
    provider.ccr_history_cache_source, provider.ccr_val_history_cache_source = train_source, dev_source
    cache_report = base.attach_caches(provider, a, root)
    source_files = [root/'real_motion/shared_source_track.py', root/'real_motion/source_evidence_audit.py',
                    root/'real_motion/strong_w2det.py', root/'tools/real_motion/benchmark_p0_f9_v18_runtime.py']
    source_implementation = {str(p.relative_to(root)): base.sha256(p) for p in source_files}
    # Writers live ONLY inside this NEW experiment, never in an old cache root.
    cache_report['shared_track'] = provider.attach_tracks(out/'track_history', rank=context.rank, implementation=source_implementation)
    history = StageHistory(provider, arm=a.arm, ram_mib=a.descriptor_ram_mib)
    inventory = []
    for record in evaluation:
        for token in record['history_tokens'][-4:]:
            p = (Path(a.stc_root)/record['scene_name']/str(token)/'labels.npz').resolve()
            st = p.stat(); inventory.append((str(p), st.st_size, st.st_mtime_ns))
    folders = ('tools/source_scene_stage1', 'tools/shared_source_track', 'tools/shared_history_experiments/full_gt',
               'tools/shared_history_experiments/refine')
    files = [p for folder in folders for p in (root/folder).glob('*.py')]+source_files
    files += [root/p for p in ('tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
        'tools/dense_valid_ddp/engine.py', 'tools/joint_occ_experiments/data.py', 'tools/history_validity/data.py',
        'real_motion/local_st_world_model.py', 'tools/joint_occ_experiments/soft_geometry.py',
        'tools/joint_occ_experiments/fast_autograd.py', 'tools/joint_occ_experiments/batching/geometry.py',
        'tools/real_motion/joint_training_distributed.py')]
    contract = dict(protocol=PROTOCOL, arm=a.arm, model=model.configs(), world_size=context.world_size,
        parameters=sum(p.numel() for p in model.parameters()), source_policy='shared_pinned',
        source_implementation=source_implementation, train_keys=[list(k) for k in train_keys],
        eval_keys=base.record_keys(evaluation), epochs=2, total_updates=total, epochs_updates=list(map(len, groups)),
        grouping='original_train20_round_robin_globalB4_source128_no_truncation',
        objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.), seed=a.seed,
        learning_rates=[a.motion_lr, a.repair_lr], schedule='whole_two_epoch_cosine_floor0.01',
        known='TRAIN_OCC_STC_dense_valid', precision='original_BF16_geometry_FP64_H6',
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                       cpu_workers=a.cpu_workers, prefetch_workers=a.prefetch_workers),
        data={n: base.sha256(getattr(a, n)) for n in ('config', 'train_cache', 'dev_cache', 'base_checkpoint', 'train_info', 'dev_info')},
        overrides=a.override, STC_inventory=sorted(set(inventory)),
        implementation={str(p.relative_to(root)): base.sha256(p) for p in files})
    contract = json.loads(json.dumps(contract))
    epoch = batch = updates = 0; evaluations = {}
    if a.resume:
        if Path(a.resume).resolve() != out/'last.pt': raise ValueError('resume SAME new output last.pt only')
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        epoch, batch, updates = validate_resume(saved, contract, groups)
        model.load_state_dict(saved['models']['gt'], strict=True); optimizer.load_state_dict(saved['optimizers']['gt'])
        context.restore_rng(saved['rank_rng'], rng); evaluations = saved['evaluations']; del saved
    result = dict(status='running', contract=contract, epoch=epoch, updates=updates, evaluations=evaluations)
    def publish():
        if context.primary:
            base.write_json(out/'results.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
    def save(extra=None):
        states = context.gather(context.rng_state(rng))
        if context.primary:
            payload = dict(protocol=PROTOCOL, contract=contract, epoch=epoch, batch=batch, updates=updates,
                models={'gt': model.state_dict()}, optimizers={'gt': optimizer.state_dict()}, rank_rng=states,
                evaluations=evaluations)
            save_resume_checkpoint(out/'last.pt', payload)
            if extra: save_resume_checkpoint(out/extra, payload)
            publish()
        context.barrier()
    def evaluate_epoch():
        if batch or epoch == 0 or str(epoch) in evaluations: return True
        states = context.gather(context.rng_state(rng)); message = None
        if context.primary:
            try:
                path = out/f'eval_epoch_{epoch:04d}_state.json'
                saved = json.loads(path.read_text()) if path.exists() else None
                if saved and (saved['contract'] != contract or saved['updates'] != updates):
                    raise RuntimeError('epoch eval provenance differs')
                def store(value): base.write_json(path, dict(contract=contract, updates=updates, state=value))
                iterator = batches(provider, dev_source, [[r] for r in evaluation], history, workers=a.prefetch_workers)
                try:
                    report = evaluate(model, provider, dev_source, iterator, arm=a.arm, stc_root=a.stc_root,
                        saved=saved['state'] if saved else None, save=store, stop_event=event)
                finally: iterator.close()
                if report['keys'] != contract['eval_keys'] or report['windows'] != 64: raise RuntimeError('evaluation population differs')
                base.write_json(out/f'eval_epoch_{epoch:04d}.json', dict(contract=contract, updates=updates, evaluation=report))
                message = dict(status='complete', report=report)
            except InterruptedError: message = dict(status='stopped')
            except Exception as error: message = dict(status='error', error=repr(error))
        message = context.broadcast(message); context.restore_rng(states, rng)
        if message['status'] == 'error': raise RuntimeError(message['error'])
        if message['status'] == 'stopped': return False
        evaluations[str(epoch)] = message['report']; result['evaluations'] = evaluations
        if context.primary: print(summary(result), flush=True)
        return True
    try:
        if context.primary:
            base.write_json(out/'contract.json', contract); base.write_json(out/'cache_audit.json', cache_report)
            print(f'SCENE_STAGE1 {a.arm} RANDOM TRAIN4086 x2; updates={total}; world={context.world_size}; NEW={out}', flush=True)
        if not evaluate_epoch(): event.set()
        while epoch < 2 and not context.stop(event.is_set()) and not (a.max_updates and updates >= a.max_updates):
            global_groups = groups[epoch][batch:]
            iterator = batches(provider, train_source, [g[context.rank::context.world_size] for g in global_groups],
                               history, workers=a.prefetch_workers)
            try:
                for group in global_groups:
                    tick = time.perf_counter(); rows = next(iterator)
                    if context.stop(event.is_set()): break
                    stats = train_step(model, optimizer, provider, rows, context,
                        global_windows=len(group), update=updates+1, total_updates=total)
                    ranks = context.gather(dict(stats, pipeline_seconds=time.perf_counter()-tick))
                    updates += 1; batch += 1
                    if batch == len(groups[epoch]): epoch += 1; batch = 0
                    result.update(epoch=epoch, updates=updates)
                    if context.primary:
                        wall = max(r['pipeline_seconds'] for r in ranks)
                        item = dict(event='scene_stage1_train', arm=a.arm, update=updates, epoch=epoch,
                            windows=len(group), seconds_per_window=wall/len(group), **stats)
                        with (out/'progress.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(item)+'\n')
                        if updates == 1 or updates % 16 == 0:
                            print(f'SCENE_STAGE1 {a.arm} {updates}/{total} epoch={epoch} CE={stats["CE"]:.4f} seconds/window={wall/len(group):.4f}', flush=True)
                    if batch == 0 or updates % a.checkpoint_every == 0:
                        save(f'epoch_{epoch:04d}.pt' if batch == 0 else None)
                    if batch == 0:
                        if not evaluate_epoch(): event.set()
                        save()
                    if context.stop(event.is_set()) or a.max_updates and updates >= a.max_updates or batch == 0: break
            finally: iterator.close()
        save()
        result['status'] = 'complete' if epoch == 2 and '2' in evaluations else 'stopped_at_complete_update_or_eval_window'
        publish()
        if context.primary: print(summary(result), flush=True)
        return 0
    finally:
        provider.close_tracks()
        provider.ccr_history_cache.close(); provider.ccr_val_history_cache.close()
        provider.ccr_execution.close(); provider.strong_majority.close()


def main(argv=None, event=None):
    a = parser().parse_args(argv); event = event or threading.Event()
    context = DistributedTraining(True)
    if not context.active: context.device = torch.device(a.device)
    if context.world_size not in (1, 2): raise RuntimeError('one/two training GPUs only')
    out = Path(a.out_dir).resolve()
    try:
        with ExitStack() as stack:
            if context.primary:
                lock = out.parent/'.scene_stage1_locks'/base.stable_json_fingerprint(str(out))
                lock.mkdir(parents=True, exist_ok=True); stack.enter_context(evaluation_lock(lock))
                if out.exists() and not a.resume: raise RuntimeError('NEW output directory required')
                if a.resume and not (out/'last.pt').is_file(): raise RuntimeError('missing SAME output last.pt')
                if a.resume:
                    if Path(a.resume).resolve() != out/'last.pt': raise ValueError('resume SAME output last.pt only')
                    header = torch.load(a.resume, map_location='cpu', weights_only=False)
                    if (header.get('protocol') != PROTOCOL or header.get('contract', {}).get('arm') != a.arm or
                            header['contract'].get('source_policy') != 'shared_pinned'):
                        raise RuntimeError('not this arm/stage1 checkpoint; no new cache is written into an old run')
                    del header
                out.mkdir(parents=True, exist_ok=True)
            context.barrier()
            return launch(a, event, context)
    finally: context.close()


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event=event))
