"""RANDOM full TRAIN x2, dense-valid, two GPUs, fixed remaining64 every2000."""
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
from tools.joint_occ_experiments import run as base, data
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.perception_robustness.evaluation import evaluator
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .engine import DenseTrainHistory, dense_rows, train_step, merge_evaluation

PROTOCOL = 'RANDOM_fullTRAIN_dense_valid_2epochs_globalB4_twoGPU_H6_v1'


def evaluation_due(update, total):
    return update > 0 and (update % 2000 == 0 or update == total)


def parser():
    p = base.parser()
    p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=1., seed=20261011,
                   cpu_workers=4, prefetch_workers=2, eval_population='dev64')
    p.add_argument('--eval-reference', required=True, help='previous clean/results.json; keys ONLY')
    p.add_argument('--stc-root', required=True)
    p.add_argument('--source-policy', choices=('baseline', 'shared_pinned'), default='shared_pinned')
    p.add_argument('--track-cache', required=True, help='NEW versioned pure-history track cache, not old CCR root')
    return p


def eval_keys(path):
    keys = json.loads(Path(path).read_text(encoding='utf-8'))['contract']['eval_keys']
    if len(keys) != 64 or len(set(map(tuple, keys))) != 64:
        raise ValueError('exact previous remaining64 required')
    return [list(map(str, k)) for k in keys]


def validate_cursor(saved, contract, groups):
    if saved.get('protocol') != PROTOCOL or saved.get('contract') != contract:
        raise RuntimeError('resume contract changed; use SAME directory/data/two-GPU execution')
    epoch, batch, update = (int(saved[k]) for k in ('epoch', 'batch', 'updates'))
    if (not 0 <= epoch <= len(groups) or epoch == len(groups) and batch != 0
            or epoch < len(groups) and not 0 <= batch < len(groups[epoch])
            or update != sum(map(len, groups[:epoch]))+batch):
        raise RuntimeError('invalid completed global-update cursor')
    if len(saved['rank_rng']) != 2: raise RuntimeError('exactly two rank RNG states required')
    return epoch, batch, update


def summary(result):
    c = result['contract']
    lines = ['===== RANDOM DENSE-VALID / FULL TRAIN x2 / TWO GPUs =====',
        'status='+result['status'], f'updates={result["updates"]}/{c["total_updates"]}; completed_epochs={result["epoch"]}',
        'RANDOM initialization; base checkpoint supplies architecture only, NO body/head weights imported.',
        'Full TRAIN20430 x2; GLOBAL B4/source128 grouping unchanged; per-rank shards, no padding/drop.',
        'H6 full-grid CE+Lovasz ONLY; no motion/existence/ego loss/head or navigation.',
        'TRAIN dense-valid; OCC eval original lidar-known; STC dense-valid; GT ego explicitly conditioned.',
        'Source policy='+c['source_policy']+'; original two-frame KTA unchanged; no multiframe fit.',
        'Old geometry caches READ ONLY; new pure-history track namespace; no learned bank or augmentation.',
        'Same previous remaining64, NOT original DEV64 or an independent test. Every2000 GLOBAL updates + final.']
    for update, value in result.get('evaluations', {}).items():
        lines += [f'===== update={update}; windows={value["windows"]} =====',
                  'route                         mIoU       IoU MovingMicro']
        for name, m in value['variants']['gt'].items():
            lines += [f'{name:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}']
    if 'error' in result: lines += ['error='+result['error']]
    return '\n'.join(lines)+'\nTwo GPUs do NOT guarantee 2x speed; no automatic model/epoch selection. NOT FPS.\n'


def launch(a, event, context):
    root, out = Path(__file__).resolve().parents[2], Path(a.out_dir).resolve()
    if (a.mode != 'gt' or a.epochs != 2 or a.train_fraction != 1. or a.seed != 20261011
            or a.window_batch_size != 4 or a.source_budget != 128 or a.training_backend != 'fused'
            or a.lovasz_weight != 1. or a.evaluate_only or a.max_updates < 0
            or not 1 <= a.prefetch_workers <= 8 or not 1 <= a.cpu_workers <= 20
            or a.checkpoint_every < 1 or min(a.motion_lr, a.repair_lr) <= 0):
        raise ValueError('fixed RANDOM TRAIN20430 x2 / GLOBAL B4-source128 / H6 CE+Lovasz protocol')
    for name in ('frame_cache_mib', 'geometry_ram_mib', 'descriptor_ram_mib'):
        if not 0 <= getattr(a, name) <= 4096: raise ValueError('bounded per-rank RAM required: '+name)
    torch.set_num_threads(1)
    device = context.device
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('each selected GPU must support the original BF16 execution')
    torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed)
    random.seed(a.seed); np.random.seed(a.seed)
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, a.override))
    architecture = torch.load(a.base_checkpoint, map_location='cpu', weights_only=False)['model_config']
    cfg = replace(base.config_from_mapping_v17(architecture), history_frames=4)
    _, all_train = base.load_cache(a.train_cache); _, all_dev = base.load_cache(a.dev_cache)
    base.record_keys(all_train); base.record_keys(all_dev)
    if len(all_train) != 20430 or len(all_dev) != 4369: raise ValueError('TRAIN20430 / VAL4369 required')
    keys = eval_keys(a.eval_reference)
    train_keys = base.select_scene_balanced_round_robin(base.record_keys(all_train), len(all_train))
    train = base.align_records(all_train, train_keys); evaluation = base.align_records(all_dev, keys)
    if {r['scene_name'] for r in train} & {r['scene_name'] for r in evaluation}:
        raise ValueError('TRAIN/evaluation scene overlap')
    groups = [base.epoch_groups(train, a.seed, e, 4, 128) for e in range(2)]
    model = GTBody(cfg).to(device)
    # Explicitly start both replicas with the exact same model, including buffers.
    for value in model.state_dict().values(): dist.broadcast(value, src=0)
    # No stochastic stream is shared across ranks after identical initialization.
    torch.manual_seed(a.seed+context.rank); torch.cuda.manual_seed(a.seed+context.rank)
    random.seed(a.seed+context.rank); np.random.seed(a.seed+context.rank)
    sampling_rng = np.random.default_rng(a.seed+context.rank)
    optimizer = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=a.motion_lr, initial_lr=a.motion_lr, weight_decay=1e-4),
        dict(params=model.columns.parameters(), lr=a.repair_lr, initial_lr=a.repair_lr, weight_decay=.01)])
    from tools.shared_source_track.adapter import TrackProvider
    factory = TrackProvider if a.source_policy == 'shared_pinned' else base.PilotProvider
    provider = factory(a.base_checkpoint, base.CLEAN_SHA256, pcfg, device, a.cpu_workers, model, None)
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
    source_files = [root/'real_motion/shared_source_track.py', *sorted((root/'tools/shared_source_track').glob('*.py')),
        *[root/p for p in ('real_motion/source_evidence_audit.py', 'real_motion/strong_w2det.py',
            'real_motion/runtime_fastpath.py', 'tools/real_motion/benchmark_p0_f9_v18_runtime.py',
            'real_motion/native/column_cpu.cpp')]]
    source_implementation = {str(p.relative_to(root)): base.sha256(p) for p in source_files}
    if a.source_policy == 'shared_pinned':
        track_path = Path(a.track_cache).resolve()
        for original in (Path(a.ccr_history_cache).resolve(), Path(a.ccr_val_history_cache).resolve()):
            if track_path == original or original in track_path.parents or track_path in original.parents:
                raise ValueError('new track cache must be separate from original geometry cache roots')
        cache_report['shared_track'] = provider.attach_tracks(track_path, rank=context.rank,
            implementation={k: v for k, v in source_implementation.items()
                            if not k.startswith('tools/shared_source_track/')})
    train_history = DenseTrainHistory(provider, ram_mib=a.descriptor_ram_mib)
    dev_history = ResidualHistoryInputs(provider, ram_mib=a.descriptor_ram_mib)
    # No navigation commands, ego targets or auxiliary object labels are prepared.
    data.ego_inputs = gt.history_inputs
    inventory = []
    for record in evaluation:
        for token in record['history_tokens'][-4:]:
            path = (Path(a.stc_root)/str(record['scene_name'])/str(token)/'labels.npz').resolve()
            st = path.stat(); inventory.append([str(path), st.st_size, st.st_mtime_ns])
    dependencies = [p for folder in ('tools/dense_valid_ddp', 'tools/shared_history_experiments/full_gt',
        'tools/shared_history_experiments/refine', 'tools/perception_robustness') for p in (root/folder).glob('*.py')]
    dependencies += [root/p for p in ('tools/joint_occ_experiments/data.py', 'tools/history_validity/data.py',
        'tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
        'tools/real_motion/joint_training_distributed.py', 'tools/scene_ego_head/diagnostics/engine.py',
        'tools/final_semantic_evaluation/predictor.py')]
    dependencies += source_files
    contract = dict(protocol=PROTOCOL, model=model.configs(), epochs=2, train_keys=[list(k) for k in train_keys],
        eval_keys=keys, eval_every=2000, total_updates=sum(map(len, groups)), global_batch=4, source_budget=128,
        epochs_updates=list(map(len, groups)), initialization='RANDOM_architecture_only_NO_pretrained_weights',
        seed=a.seed, training_known='dense_valid', OCC_eval_known='original_lidar_known', STC_eval_known='dense_valid',
        source_policy=a.source_policy, source_implementation=source_implementation,
        objective={'CE': 1., 'Lovasz': 1., 'motion': 0., 'existence': 0., 'ego': 0.},
        lr=[a.motion_lr, a.repair_lr], schedule='whole_two_epoch_cosine_floor_0.01',
        distributed=dict(world_size=2, partition='window_round_robin_no_padding_drop',
                         reduction='global_window_mean_SUM_gradients_BEFORE_clip_AdamW', rank_rng='saved_separately'),
        data={k: base.sha256(getattr(a, k)) for k in ('config', 'train_cache', 'dev_cache', 'base_checkpoint', 'train_info', 'dev_info')},
        overrides=a.override, cache_namespace={k: cache_report[k]['namespace'] for k in ('train', 'val')},
        STC_inventory=sorted({tuple(r) for r in inventory}),
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, precision='original_BF16_H6'),
        implementation={str(p.relative_to(root)): base.sha256(p) for p in dependencies})
    # JSON-normalize tuples (including file inventory) before checkpoint comparison.
    contract = json.loads(json.dumps(contract))
    epoch = batch = updates = 0; evaluations = {}
    if a.resume:
        if Path(a.resume).resolve() != out/'last.pt': raise ValueError('resume SAME output last.pt only')
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        epoch, batch, updates = validate_cursor(saved, contract, groups)
        model.load_state_dict(saved['models']['gt'], strict=True)
        optimizer.load_state_dict(saved['optimizers']['gt'])
        context.restore_rng(saved['rank_rng'], sampling_rng)
        evaluations = saved['evaluations']
        del saved
    result = dict(status='running', contract=contract, epoch=epoch, updates=updates, evaluations=evaluations)
    def publish():
        if context.primary:
            base.write_json(out/'results.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
    def save(extra=None):
        rank_rng = context.gather(context.rng_state(sampling_rng))
        if context.primary:
            value = dict(protocol=PROTOCOL, contract=contract, epoch=epoch, batch=batch, updates=updates,
                models={'gt': model.state_dict()}, optimizers={'gt': optimizer.state_dict()},
                rank_rng=rank_rng, evaluations=evaluations)
            save_resume_checkpoint(out/'last.pt', value)
            for name in ([extra] if isinstance(extra, str) else extra or []):
                save_resume_checkpoint(out/name, value)
            publish()
        context.barrier()
    def maybe_evaluate():
        if not evaluation_due(updates, contract['total_updates']): return True
        tag = str(updates)
        completed_path = out/f'eval_update_{updates:06d}.json'
        completed = json.loads(completed_path.read_text()) if completed_path.exists() else None
        if completed:
            if completed['contract'] != contract or completed['updates'] != updates:
                raise RuntimeError('periodic evaluation provenance changed')
            evaluations[tag] = completed['evaluation']; return True
        rng = context.gather(context.rng_state(sampling_rng))
        state_path = out/f'eval_update_{updates:06d}_rank{context.rank}_state.json'
        state = json.loads(state_path.read_text()) if state_path.is_file() else None
        if state and (state['updates'] != updates or state['contract'] != contract):
            raise RuntimeError('evaluation resume changed')
        snapshot = {}
        def save_eval(value):
            snapshot.clear(); snapshot.update(value)
            base.write_json(state_path, dict(updates=updates, contract=contract, state=value))
        local = evaluation[context.rank::2]
        batches = data.prefetched_batches(provider, dev_source, [[r] for r in local], dev_history,
                                          workers=a.prefetch_workers)
        interrupted = False
        eval_factory, eval_protocol = evaluator, None
        if a.source_policy == 'shared_pinned':
            from tools.shared_source_track.evaluation import evaluator as eval_factory, PROTOCOL as eval_protocol
        try:
            eval_factory(a.stc_root, inventory=inventory)({'gt': model}, provider, dev_source, batches,
                saved=state['state'] if state else None, save=save_eval, stop_event=event)
        except InterruptedError:
            interrupted = True
        finally:
            batches.close(); context.restore_rng(rng, sampling_rng)
        if context.stop(interrupted or event.is_set()): return False
        states = context.gather(snapshot)
        merged = merge_evaluation(states, keys, protocol=eval_protocol)
        evaluations[tag] = merged
        if context.primary:
            base.write_json(completed_path, dict(updates=updates, contract=contract, evaluation=merged))
            print(summary(dict(result, evaluations={tag: merged})), flush=True)
        context.barrier()
        return True
    started = time.perf_counter()
    try:
        if context.primary:
            base.write_json(out/'contract.json', contract); base.write_json(out/'cache_audit.json', cache_report)
            print(f'DENSE_DDP RANDOM TRAIN20430 x2; global_updates={contract["total_updates"]}; resume={updates}; '
                  f'GLOBAL B4/source128; every2000 fixed remaining64 OCC/STC. Output: {out}', flush=True)
        context.barrier()
        if not maybe_evaluate(): event.set()
        while epoch < 2 and not context.stop(event.is_set()) and not (a.max_updates and updates >= a.max_updates):
            global_groups = groups[epoch][batch:]
            local_groups = [g[context.rank::2] for g in global_groups]
            batches = data.prefetched_batches(provider, train_source, local_groups, train_history,
                                              workers=a.prefetch_workers)
            try:
                for group in global_groups:
                    tick = time.perf_counter(); rows = next(batches)
                    if context.stop(event.is_set()): break
                    stats = train_step(model, optimizer, provider, dense_rows(rows), context,
                        global_windows=len(group), update=updates+1, total_updates=contract['total_updates'])
                    # Cursor advances only after both replicas complete the synchronized Adam update.
                    # Wall time is MAX across ranks, never SUM or divided by rank count.
                    ranks = context.gather(dict(stats, pipeline_seconds=time.perf_counter()-tick))
                    updates += 1; batch += 1
                    if batch == len(groups[epoch]): epoch += 1; batch = 0
                    result.update(epoch=epoch, updates=updates)
                    if context.primary:
                        wall = max(r['pipeline_seconds'] for r in ranks)
                        row = dict(event='dense_ddp_train', update=updates, completed_epochs=epoch,
                            global_windows=len(group), rank_windows=[r['local_windows'] for r in ranks],
                            CE=stats['CE'], Lovasz=stats['Lovasz'], pipeline_seconds=wall,
                            seconds_per_global_window=wall/len(group), ranks=ranks)
                        with (out/'progress.jsonl').open('a', encoding='utf-8') as handle:
                            handle.write(json.dumps(row)+'\n')
                        if updates == 1 or updates % 16 == 0:
                            print(f'DENSE_DDP {updates}/{contract["total_updates"]} epoch={epoch} CE={stats["CE"]:.4f} '
                                  f'seconds/window={wall/len(group):.4f} rank_windows={row["rank_windows"]}', flush=True)
                    due = evaluation_due(updates, contract['total_updates'])
                    if due or updates % a.checkpoint_every == 0 or batch == 0:
                        extra = ([f'update_{updates:06d}.pt'] if due else [])
                        if batch == 0: extra += [f'epoch_{epoch:04d}.pt']
                        save(extra)
                    if due:
                        if not maybe_evaluate(): event.set()
                        save()
                    if context.stop(event.is_set()) or a.max_updates and updates >= a.max_updates or batch == 0: break
            finally: batches.close()
        save()
        result['status'] = 'complete' if epoch == 2 and str(updates) in evaluations else 'stopped_at_complete_update_or_eval_window'
        result['seconds_this_invocation'] = time.perf_counter()-started
        publish()
        if context.primary: print(summary(result), flush=True)
        return 0
    except Exception as error:
        # Never overwrite a last.pt with a partially failed distributed update.
        result.update(status='failed_previous_checkpoint_preserved', error=repr(error)); publish(); raise
    finally:
        if hasattr(provider, 'track_cache'): provider.close_tracks()
        for name in ('ccr_history_cache', 'ccr_val_history_cache'): getattr(provider, name).close()
        provider.ccr_execution.close(); provider.strong_majority.close()


def main(event=None, argv=None):
    a = parser().parse_args(argv); event = event or threading.Event()
    if torch.cuda.device_count() < 2: raise RuntimeError('two visible CUDA GPUs required; set CUDA_VISIBLE_DEVICES=0,1')
    context = DistributedTraining(True)
    if not context.active: raise RuntimeError('launch using torchrun --nproc_per_node=2')
    out = Path(a.out_dir).resolve()
    try:
        with ExitStack() as stack:
            if context.primary:
                lock = out.parent/'.dense_ddp_locks'/base.stable_json_fingerprint(str(out))
                lock.mkdir(parents=True, exist_ok=True); stack.enter_context(evaluation_lock(lock))
                if out.exists() and not a.resume: raise RuntimeError('NEW output directory required')
                if a.resume and not (out/'last.pt').is_file(): raise RuntimeError('missing SAME run last.pt')
                out.mkdir(parents=True, exist_ok=True)
            context.barrier()
            return launch(a, event, context)
    finally: context.close()


if __name__ == '__main__':
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stopped.set())
    sys.exit(main(stopped))
