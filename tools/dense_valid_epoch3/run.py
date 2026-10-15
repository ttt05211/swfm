"""Two-GPU epoch3; preserve completed epoch2 weights/Adam/rank RNG; dense eval."""
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base, data
from tools.dense_valid_ddp import run as original
from tools.dense_valid_ddp.engine import DenseTrainHistory, dense_rows, merge_evaluation
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_source_track.adapter import TrackProvider
from tools.shared_source_track.evaluation import evaluator, PROTOCOL as EVAL_PROTOCOL
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.joint_training_recovery import snapshot_checkpoint, save_resume_checkpoint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .engine import train_step

PROTOCOL = 'dense_pinned_epoch2_to_epoch3_preserve_Adam_RNG_tail_cosine_dense_eval_v1'


def validate_source(saved):
    c = saved['contract']
    if (saved.get('protocol') != original.PROTOCOL or c['protocol'] != original.PROTOCOL
            or saved['epoch'] != 2 or saved['batch'] != 0 or saved['updates'] != c['total_updates']
            or c['epochs'] != 2 or len(c['train_keys']) != 20430 or len(c['eval_keys']) != 64
            or c['training_known'] != 'dense_valid' or c['source_policy'] != 'shared_pinned'
            or len(saved['rank_rng']) != 2 or c['global_batch'] != 4 or c['source_budget'] != 128
            or c['objective'] != dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.)):
        raise RuntimeError('completed original dense-valid/shared-pinned two-epoch DDP checkpoint required')
    return c


def due(update, total): return update > 0 and (update % 2000 == 0 or update == total)


def select_source(directory, root):
    if directory: return Path(directory).resolve()/'epoch_0002.pt'
    paths = list(Path(root).glob('dense_valid_shared_full2_ddp_*/epoch_0002.pt'))
    if not paths: raise RuntimeError('no completed epoch_0002.pt; wait for original two epochs or set DENSE_DDP_DIR')
    return max(paths, key=lambda p: p.stat().st_mtime_ns).resolve()


def summary(r):
    c = r['contract']; lines = ['===== DENSE-VALID / EPOCH2 -> ONE ADDITIONAL EPOCH / TWO GPUs =====',
        'status='+r['status'], f'additional_updates={r["additional_updates"]}/{c["additional_steps"]}; '
        f'cumulative_updates={r["updates"]}/{c["total_updates"]}; completed_epochs={r["epoch"]}',
        'Original epoch2 weights/Adam/rank RNG imported; next data permutation uses original epoch index2.',
        'Same global B4/source128/H6 CE+Lovasz ONLY/shared pinned source; no ego/motion/existence auxiliaries.',
        'TRAIN AND OCC eval dense-valid; STC dense-valid; SAME remaining64/full-grid/GT future ego.',
        'NEW cosine tail from source ACTUAL LR to 0.1x; no high-LR restart; NOT a planned-from-start three-epoch cosine.',
        'LR start='+str(c['start_lrs'])+'; end='+str([v*.1 for v in c['start_lrs']]),
        'Every2000 CUMULATIVE updates + final evaluation; old lidar-known scores NOT used as dense baseline.']
    for update, report in r['evaluations'].items():
        lines += [f'===== cumulative_update={update}; windows={report["windows"]} =====',
                  'route                         mIoU       IoU MovingMicro']
        for name, m in report['variants']['gt'].items():
            lines += [f'{name:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}']
    return '\n'.join(lines)+'\nSource experiment read only; compatible geometry reused; no automatic selection/extension. NOT FPS.\n'


def launch(a, event, ctx, source):
    root, out = Path(__file__).resolve().parents[2], Path(a.out_dir).resolve(); device = ctx.device
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('each selected GPU must support original BF16 execution')
    if ctx.primary and not a.resume: snapshot_checkpoint(source, out/'source_epoch2.pt')
    ctx.barrier()
    frozen = out/'source_epoch2.pt'; source_sha = base.sha256(frozen)
    saved = torch.load(frozen, map_location='cpu', weights_only=False); prior = validate_source(saved)
    if base.sha256(source) != source_sha: raise RuntimeError('source epoch2 snapshot differs')
    for name, expected in prior['data'].items():
        if base.sha256(getattr(a, name)) != expected: raise RuntimeError('source data changed: '+name)
    for name, expected in prior['implementation'].items():
        if base.sha256(root/name) != expected: raise RuntimeError('original dependency changed: '+name)
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, prior.get('overrides')))
    _, all_train = base.load_cache(a.train_cache); _, all_dev = base.load_cache(a.dev_cache)
    train = base.align_records(all_train, prior['train_keys']); evaluation = base.align_records(all_dev, prior['eval_keys'])
    groups = base.epoch_groups(train, prior['seed'], 2, 4, 128)
    spec = prior['model']; sh, rep = spec['shared_history'], spec['repair']
    model = GTBody(base.config_from_mapping_v17(spec['motion']), feature_width=sh['width'], z_bins=sh['z_bins'],
        memory_width=rep['memory_width'], width=rep['width'], chunk_size=rep['chunk_size']).to(device)
    model.columns.epsilon = rep['epsilon']; model.load_state_dict(saved['models']['gt'], strict=True)
    optimizer = torch.optim.AdamW([dict(params=model.transport.parameters()), dict(params=model.columns.parameters())])
    optimizer.load_state_dict(saved['optimizers']['gt'])
    start_lrs = [float(g['lr']) for g in optimizer.param_groups]
    rng = np.random.default_rng(); provider = TrackProvider(a.base_checkpoint, base.CLEAN_SHA256, pcfg, device, a.cpu_workers, model, None)
    provider.raw_io_workers = 1
    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    from real_motion.surface_projection_execution import SurfaceMapExecution
    from real_motion.strong_majority_execution import ParallelNativeMajority
    provider.ccr_execution = SurfaceMapExecution(CanonicalCpuExecution('native_parallel', min(4, a.cpu_workers)))
    provider.strong_majority = ParallelNativeMajority(min(4, a.cpu_workers))
    tr = base.CachedColumnSource(base.NuScenesWindowSource(a.dataroot, info_pkl=a.train_info), a.frame_cache_mib, copy_on_insert=False)
    va = base.CachedColumnSource(base.NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info), a.frame_cache_mib, copy_on_insert=False)
    provider.ccr_history_cache_source, provider.ccr_val_history_cache_source = tr, va
    caches = base.attach_caches(provider, a, root)
    provider.attach_tracks(a.track_cache, rank=ctx.rank,
        implementation={k: v for k, v in prior['source_implementation'].items() if not k.startswith('tools/shared_source_track/')})
    if {k: caches[k]['namespace'] for k in ('train', 'val')} != prior['cache_namespace']:
        raise RuntimeError('original geometry cache namespace changed')
    data.ego_inputs = gt.history_inputs
    train_hist = DenseTrainHistory(provider, ram_mib=a.descriptor_ram_mib)
    dev_hist = DenseTrainHistory(provider, ram_mib=a.descriptor_ram_mib)
    files = sorted(Path(__file__).parent.glob('*.py'))
    contract = dict(protocol=PROTOCOL, source_checkpoint=str(source), source_sha256=source_sha,
        source_contract=prior, model=model.configs(), training_known='dense_valid', OCC_eval_known='dense_valid',
        STC_eval_known='dense_valid', source_updates=saved['updates'], additional_steps=len(groups),
        total_updates=saved['updates']+len(groups), start_lrs=start_lrs, end_factor=.1,
        objective=prior['objective'], train_keys=prior['train_keys'], eval_keys=prior['eval_keys'],
        source_policy=prior['source_policy'], world_size=2, eval_every_cumulative=2000,
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, precision='original_BF16_H6'),
        implementation={str(f.relative_to(root)): base.sha256(f) for f in files})
    contract = json.loads(json.dumps(contract)); local_step = 0; reports = {}
    ctx.restore_rng(saved['rank_rng'], rng)
    if a.resume:
        state = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        if state.get('protocol') != PROTOCOL or state['contract'] != contract: raise RuntimeError('SAME unchanged epoch3 run required')
        local_step = int(state['additional_updates'])
        if (not 0 <= local_step <= len(groups) or state['updates'] != contract['source_updates']+local_step
                or state['epoch'] != (3 if local_step == len(groups) else 2)
                or state['batch'] != (0 if local_step == len(groups) else local_step)):
            raise RuntimeError('invalid additional epoch cursor')
        model.load_state_dict(state['models']['gt']); optimizer.load_state_dict(state['optimizers']['gt'])
        ctx.restore_rng(state['rank_rng'], rng); reports = state['evaluations']; del state
    del saved
    result = dict(status='running', contract=contract, epoch=2 if local_step < len(groups) else 3,
        additional_updates=local_step, updates=contract['source_updates']+local_step, evaluations=reports)
    def publish():
        if ctx.primary:
            base.write_json(out/'results.json', result); (out/'summary.txt').write_text(summary(result), encoding='utf-8')
    def save(extra=()):
        states = ctx.gather(ctx.rng_state(rng))
        if ctx.primary:
            value = dict(protocol=PROTOCOL, contract=contract, epoch=result['epoch'], batch=0 if result['epoch'] == 3 else local_step,
                updates=result['updates'], additional_updates=local_step, models={'gt': model.state_dict()},
                optimizers={'gt': optimizer.state_dict()}, rank_rng=states, evaluations=reports)
            save_resume_checkpoint(out/'last.pt', value)
            for name in extra: save_resume_checkpoint(out/name, value)
            publish()
        ctx.barrier()
    def eval_now():
        update = result['updates']; tag = str(update)
        # Baseline source update is also reported using the CORRECT dense policy.
        complete = out/f'eval_update_{update:06d}.json'
        if complete.is_file():
            value = json.loads(complete.read_text())
            if value['contract'] != contract: raise RuntimeError('evaluation contract changed')
            reports[tag] = value['evaluation']; return True
        previous_rng = ctx.gather(ctx.rng_state(rng)); path = out/f'eval_update_{update:06d}_rank{ctx.rank}_state.json'
        prior_state = json.loads(path.read_text()) if path.exists() else None
        if prior_state and prior_state['contract'] != contract: raise RuntimeError('evaluation state changed')
        snapshot = {}
        def record(state):
            snapshot.clear(); snapshot.update(state)
            base.write_json(path, dict(contract=contract, state=state))
        batches = data.prefetched_batches(provider, va, [[r] for r in evaluation[ctx.rank::2]], dev_hist, workers=a.prefetch_workers)
        def dense_batches():
            for rows in batches: yield dense_rows(rows)
        stopped = False
        try:
            evaluator(a.stc_root, inventory=prior['STC_inventory'])({'gt': model}, provider, va, dense_batches(),
                saved=prior_state['state'] if prior_state else None, save=record, stop_event=event)
        except InterruptedError: stopped = True
        finally: batches.close(); ctx.restore_rng(previous_rng, rng)
        if ctx.stop(stopped or event.is_set()): return False
        value = merge_evaluation(ctx.gather(snapshot), prior['eval_keys'], protocol=EVAL_PROTOCOL)
        value['scoring'] = 'full grid; OCC dense-valid / STC dense-valid; GT future ego'
        reports[tag] = value
        if ctx.primary:
            base.write_json(complete, dict(contract=contract, evaluation=value))
            print(summary(dict(result, evaluations={tag: value})), flush=True)
        ctx.barrier(); return True
    try:
        if ctx.primary:
            base.write_json(out/'contract.json', contract)
            print(f'EPOCH3 source={source}; start={contract["source_updates"]}; extra_steps={len(groups)}; start_lrs={start_lrs}; dense eval.', flush=True)
        # Evaluate source once before learning, and resume any due interrupted evaluation.
        if local_step == 0 or due(result['updates'], contract['total_updates']):
            save()
            if not eval_now(): event.set()
        pending = groups[local_step:]
        batches = data.prefetched_batches(provider, tr, [g[ctx.rank::2] for g in pending], train_hist, workers=a.prefetch_workers)
        try:
            for group in pending:
                tick = time.perf_counter(); rows = next(batches)
                if ctx.stop(event.is_set()) or a.max_updates and local_step >= a.max_updates: break
                stats = train_step(model, optimizer, provider, dense_rows(rows), ctx, global_windows=len(group),
                    update=local_step+1, steps=len(groups), start_lrs=start_lrs)
                ranks = ctx.gather(dict(stats, pipeline_seconds=time.perf_counter()-tick))
                local_step += 1; result.update(additional_updates=local_step, updates=contract['source_updates']+local_step,
                                               epoch=3 if local_step == len(groups) else 2)
                update = result['updates']
                if ctx.primary:
                    wall = max(r['pipeline_seconds'] for r in ranks)
                    with (out/'progress.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(dict(update=update, additional_update=local_step, global_windows=len(group),
                            CE=stats['CE'], Lovasz=stats['Lovasz'], lr=stats['lr'], seconds_per_global_window=wall/len(group)))+'\n')
                    if local_step == 1 or update % 16 == 0:
                        print(f'EPOCH3 {update}/{contract["total_updates"]} additional={local_step}/{len(groups)} '
                              f'CE={stats["CE"]:.4f} seconds/window={wall/len(group):.4f} lr={stats["lr"]}', flush=True)
                evaluate_due = due(update, contract['total_updates'])
                if evaluate_due or local_step % a.checkpoint_every == 0:
                    names = [f'update_{update:06d}.pt'] if evaluate_due else []
                    if local_step == len(groups): names += ['epoch_0003.pt']
                    save(names)
                if evaluate_due:
                    if not eval_now(): event.set()
                    save()
                if ctx.stop(event.is_set()) or a.max_updates and local_step >= a.max_updates: break
        finally: batches.close()
        result['status'] = 'complete' if local_step == len(groups) and str(result['updates']) in reports else 'stopped_at_complete_update_or_eval_window'
        save(); publish()
        if ctx.primary: print(summary(result), flush=True)
    finally:
        provider.close_tracks()
        for name in ('ccr_history_cache', 'ccr_val_history_cache'): getattr(provider, name).close()
        provider.ccr_execution.close(); provider.strong_majority.close()


def main(event=None, argv=None):
    event = event or threading.Event(); p = original.parser(); p.description = __doc__
    p.add_argument('--source-dir'); p.add_argument('--resume-stage', action='store_true')
    a = p.parse_args(argv)
    if a.resume: raise ValueError('use --resume-stage for SAME epoch3 directory, not legacy --resume')
    a.resume = a.resume_stage
    if torch.cuda.device_count() < 2: raise RuntimeError('two visible CUDA GPUs required')
    if not 1 <= a.cpu_workers <= 20 or not 1 <= a.prefetch_workers <= 8 or a.checkpoint_every < 1 or a.max_updates < 0:
        raise ValueError('invalid worker/checkpoint/stop budget')
    for name in ('frame_cache_mib', 'geometry_ram_mib', 'descriptor_ram_mib'):
        if not 0 <= getattr(a, name) <= 4096: raise ValueError('bounded per-rank RAM required: '+name)
    torch.set_num_threads(1); ctx = DistributedTraining(True)
    if not ctx.active: raise RuntimeError('torchrun TWO ranks required')
    root = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    if a.resume and not a.source_dir:
        source = Path(json.loads((out/'contract.json').read_text())['source_checkpoint'])
    else: source = select_source(a.source_dir, root/'outputs/p0_f9_shared_history')
    if source.parent == out or source.parent in out.parents: raise ValueError('NEW independent output outside source run')
    try:
        with ExitStack() as stack:
            if ctx.primary:
                lease = out.parent/'.epoch3_locks'/base.stable_json_fingerprint(str(out)); lease.mkdir(parents=True, exist_ok=True)
                stack.enter_context(evaluation_lock(lease))
                if out.exists() and not a.resume: raise RuntimeError('NEW epoch3 directory required')
                if a.resume and not (out/'last.pt').is_file(): raise RuntimeError('missing SAME epoch3 last.pt')
                out.mkdir(parents=True, exist_ok=True)
            ctx.barrier(); launch(a, event, ctx, source)
    finally: ctx.close()


if __name__ == '__main__':
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stopped.set())
    main(stopped)
