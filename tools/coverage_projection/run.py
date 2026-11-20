"""Cache-free, matched RANDOM TRAIN4086 x1 coverage/projection experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import random
import signal
import threading
import time
import numpy as np
import torch
import torch.distributed as dist
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.final_semantic_evaluation.sources import window
from tools.fpn_unet_eval.common import DenseInputs
from tools.joint_occ_experiments import run as base
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.shared_history_experiments.full_gt.engine import semantic_loss
from tools.source_scene_stage1.model import ReaderConfig
from tools.paper_ablation_train.partition import partition, PARTITION
from .model import make_model, from_spec
from .adapter import forward_window
from .projection import VolumeProjection, PROJECTION
from .control import reference, CONTROL_PROTOCOL

PROTOCOL = 'forward_coverage_projection_RANDOM_TRAIN4086x1_distributed_v2'
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8'); temp.replace(path)


def make_plan(records, c):
    if len(records) != 20430: raise ValueError('original full TRAIN20430 metadata required')
    selected = base.select_scene_balanced_round_robin(base.record_keys(records), 4086)
    if [list(k) for k in selected] != c['train_keys']: raise ValueError('ordered TRAIN20% keys differ')
    rows = base.align_records(records, selected)
    if {r['scene_name'] for r in rows} & {k[0] for k in c['eval_keys']}:
        raise ValueError('TRAIN/DEV scene overlap')
    groups = base.epoch_groups(rows, c['seed'], 0, 4, 128)
    if len(groups) != c['total_updates']: raise ValueError('source128 groups/update count differs')
    return groups


def prepare_row(inputs, predictor, record, *, labels=False, kind='occ'):
    job = window(record)
    rec, raw, times = inputs.history(job, kind)
    row = predictor.prepare(rec, raw, times=times)  # histories ONLY at this boundary
    if 'features' in record and len(row[0]['source_class_id']) != len(record['features']):
        raise ValueError('live t0 component count differs from source128 TRAIN grouping metadata')
    poses = inputs.poses(job)
    target = np.stack(inputs.targets(job, range(6))) if labels else None
    return row, poses, target


@contextmanager
def prefetch(groups, inputs, predictor):
    def load(group): return [prepare_row(inputs, predictor, r, labels=True) for r in group]
    iterator = iter(groups)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='coverage_inputs') as pool:
        group = next(iterator, None)
        pending = pool.submit(load, group) if group is not None else None
        def consume():
            nonlocal pending
            while pending is not None:
                rows = pending.result(); group = next(iterator, None)
                pending = pool.submit(load, group) if group is not None else None
                yield rows
        try: yield consume()
        finally:
            if pending is not None: pending.cancel()


def local_plan(groups, world_size, rank):
    """Whole-window shards, preserving GLOBAL slots for rotating projection h."""
    if world_size not in (1, 2) or not 0 <= rank < world_size: raise ValueError('one or two ranks required')
    local, slots = [], []
    for index, group in enumerate(groups):
        selected = partition(group, world_size, index)[rank]
        position = {tuple(k): i for i, k in enumerate(base.record_keys(group))}
        local.append(selected)
        slots.append([position[tuple(k)] for k in base.record_keys(selected)])
    return local, slots


def train_step(model, optimizer, predictor, rows, projector, *, update, total,
               context=None, global_windows=None, window_slots=None):
    ctx = context or DistributedTraining(False)
    if context is None: ctx.device = predictor.provider.device
    denominator = len(rows) if global_windows is None else global_windows
    slots = list(range(len(rows))) if window_slots is None else window_slots
    if (not 1 <= denominator <= 4 or len(slots) != len(rows) or len(set(slots)) != len(slots)
            or any(not 0 <= s < denominator for s in slots)
            or ctx.active and window_slots is None):
        raise ValueError('explicit unique GLOBAL window slots and GLOBAL B1..B4 required')
    model.train(); optimizer.zero_grad(set_to_none=True)
    factor = .01+.99*.5*(1+np.cos(np.pi*(update-1)/total))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    sums = torch.zeros(5, device=predictor.provider.device)
    for slot, (row, poses, target) in zip(slots, rows):
        fw = forward_window(model, predictor.provider, row, poses=poses)
        loss, ce, lovasz = semantic_loss(fw['logits'], target)
        h = (4*(update-1)+slot) % 6
        projection, semantic, depth = projector.loss(fw['logits'][h], target[h])
        objective = loss.mean()+PROJECTION['weight']*projection
        if not bool(torch.isfinite(objective)): raise RuntimeError('nonfinite loss; no update published')
        (objective/denominator).backward()
        sums += torch.stack((ce.mean().detach(), lovasz.mean().detach(), semantic.detach(),
            depth.detach(), objective.detach()))/denominator
        del fw, loss, ce, lovasz, projection, semantic, depth, objective
    if ctx.stop(not bool(torch.isfinite(sums).all())):
        raise RuntimeError('nonfinite global objective; no optimizer update published')
    ctx.synchronize_gradients(model)  # SUM globally normalized gradients BEFORE clipping/Adam.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    sums = ctx.sum(sums)
    if predictor.provider.device.type == 'cuda': torch.cuda.synchronize(predictor.provider.device)
    return dict(zip(('CE', 'Lovasz', 'projection_semantic', 'projection_depth', 'loss'), sums.tolist()), grad_norm=float(norm))


def validate_cursor(saved, contract):
    if saved.get('protocol') != PROTOCOL or saved.get('contract') != contract:
        raise ValueError('resume SAME implementation/control/data/world-size run only')
    update = saved['updates']
    if type(update) is not int or not 0 <= update <= 1031 or len(saved['rank_rng']) != contract.get('world_size', 1):
        raise ValueError('invalid complete-update cursor')
    return update


def train(a, stop, ctx):
    device = ctx.device
    if device.type != 'cuda' or not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 CUDA GPU required')
    control = reference(a.control_dir); c = control['contract']
    if ctx.primary: print('CONTROL_CHECKPOINT_VERIFIED '+json.dumps(control['source_checkpoint_completion']), flush=True)
    if len(set(map(tuple, c['train_keys']))) != 4086 or len(set(map(tuple, c['eval_keys']))) != 64:
        raise ValueError('duplicate population keys')
    data = dict(config=sha(a.config), train_cache=sha(a.train_cache), train_info=sha(a.train_info))
    if any(c['data'].get(k) != v for k, v in data.items()):
        raise ValueError('config/TRAIN metadata differs from the original matched pilot')
    _, records = base.load_cache(a.train_cache); groups = make_plan(records, c)
    pcfg = make_prepare_config(load_runtime_config(a.config, c.get('overrides', [])))
    if pcfg.frame_dt_s != .5: raise ValueError('original KTA/horizon clock required')
    torch.set_num_threads(1); torch.manual_seed(c['seed']); torch.cuda.manual_seed(c['seed'])
    random.seed(c['seed']); np.random.seed(c['seed'])
    spec = c['model']
    model = make_model(config_from_mapping_v17(spec['motion']), feature_width=spec['shared_history']['width'],
        z_bins=spec['shared_history']['z_bins'], memory_width=spec['repair']['memory_width'],
        reader_config=ReaderConfig(**spec['reader'])).to(device)
    if ctx.active:
        for value in model.state_dict().values(): dist.broadcast(value, src=0)
    # experiment metadata is not an architecture tensor, so compare without it.
    old_spec = {k: v for k, v in spec.items() if k != 'experiment'}
    fresh_spec = {k: v for k, v in __import__('tools.causal_forward_train.model', fromlist=['ForwardBody']).ForwardBody(
        config_from_mapping_v17(spec['motion'])).configs().items() if k != 'experiment'}
    if old_spec != fresh_spec: raise ValueError('reference architecture differs from current control factory')
    # Reset after construction, exactly as in the control; dropout RNG is shared initially.
    rank_seed = c['seed']+ctx.rank
    torch.manual_seed(rank_seed); torch.cuda.manual_seed(rank_seed); random.seed(rank_seed); np.random.seed(rank_seed)
    rng = np.random.default_rng(rank_seed)
    optimizer = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=5e-4, initial_lr=5e-4, weight_decay=1e-4),
        dict(params=model.columns.parameters(), lr=3e-4, initial_lr=3e-4, weight_decay=.01)])
    predictor = RecoveryPredictor(model, pcfg, device, workers=a.cpu_workers)
    inputs = DenseInputs(a.dataroot, a.train_info, cache_mib=128)
    jobs = [window(r) for group in groups for r in group]
    inventory = inputs.preflight(jobs)
    from tools.joint_occ_experiments.soft_geometry import grid_arrays
    origin, step, shape = grid_arrays(pcfg.grid)
    projector = VolumeProjection(origin, step, shape, side=64, step_m=.4)
    implementations = {p.relative_to(ROOT).as_posix(): sha(p)
        for p in sorted((ROOT/'tools/coverage_projection').glob('*.py'))}
    # Record current dependencies too, without silently modifying old contracts.
    for folder in ('causal_forward_train', 'causal_forward_track', 'causal_forward_recovery',
            'fpn_unet', 'fpn_adaln_zero', 'source_scene_ego_conditioned', 'source_scene_stage2',
            'source_scene_stage1', 'source_scene_integrated', 'source_scene_kta_query',
            'shared_source_track', 'shared_history_experiments/refine', 'joint_occ_experiments'):
        for p in (ROOT/'tools'/folder).glob('*.py'): implementations[p.relative_to(ROOT).as_posix()] = sha(p)
    for name in ('tools/paper_ablation_train/partition.py', 'tools/real_motion/joint_training_distributed.py',
            'tools/real_motion/joint_training_recovery.py'):
        implementations[name] = sha(ROOT/name)
    contract = dict(protocol=PROTOCOL, model=model.configs(), control_dir=str(a.control_dir.resolve()),
        control_sha256=sha(a.control_dir/'epoch_0001.pt'), control_results_sha256=sha(a.control_dir/'results.json'),
        control_training_evidence=control['source_checkpoint_completion'],
        train_keys=c['train_keys'], eval_keys=c['eval_keys'], total_updates=1031, epochs=1, seed=c['seed'],
        parameters=sum(p.numel() for p in model.parameters()), train_data=data, TRAIN_inventory=inventory,
        objective=dict(CE=1., Lovasz=1., projection=PROJECTION), learning_rates=c['learning_rates'],
        schedule=c['schedule'], known=c['known'], source_policy=c['source_policy'], registration='none',
        world_size=ctx.world_size, control_world_size=c['world_size'], implementations=implementations,
        distributed=dict(partition=PARTITION, global_batch_preserved=True,
            gradient_reduction='GLOBAL_window_mean_SUM_before_global_clip_and_AdamW',
            projection_horizon='GLOBAL_window_slot_preserved', scientific_single_rank_byte_parity=False),
        group_keys=[base.record_keys(g) for g in groups], raw_live_geometry=True, tensor_disk_cache=False,
        execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, cpu_workers=a.cpu_workers))
    contract = json.loads(json.dumps(contract))
    out = a.out_dir.resolve(); update = 0
    if a.resume:
        saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        update = validate_cursor(saved, contract); model.load_state_dict(saved['models']['gt'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); ctx.restore_rng(saved['rank_rng'], rng); del saved
    elif ctx.primary:
        if out.exists(): raise ValueError('NEW output directory required; old runs are read only')
        out.mkdir(parents=True)
    if ctx.primary: write_json(out/'contract.json', contract)
    ctx.barrier()
    local_groups, slots = local_plan(groups, ctx.world_size, ctx.rank)
    def save():
        states = ctx.gather(ctx.rng_state(rng))
        if ctx.primary:
            payload = dict(protocol=PROTOCOL, contract=contract, updates=update, models={'gt': model.state_dict()},
                optimizer=optimizer.state_dict(), rank_rng=states)
            save_resume_checkpoint(out/'last.pt', payload)
            if update == 1031: save_resume_checkpoint(out/'epoch_0001.pt', payload)
            write_json(out/'progress.json', dict(status='complete' if update == 1031 else 'stopped_or_checkpointed',
                updates=update, total_updates=1031, world_size=ctx.world_size,
                trained_windows=sum(map(len, groups[:update])), updated_utc=time.time()))
        ctx.barrier()
    try:
        if update == 1031 and a.resume:
            # Evaluation may have been interrupted. Do NOT reserialize/replace
            # its frozen source checkpoint, otherwise its byte receipt changes.
            missing = ctx.broadcast(not (out/'epoch_0001.pt').is_file() if ctx.primary else None)
            if missing: save()
            if ctx.primary: print('TRAIN already complete; frozen epoch_0001.pt retained; continue evaluation.', flush=True)
            return
        if not a.resume: save()
        if ctx.primary:
            print(f'COVERAGE_PROJECTION RANDOM {update}/1031; TRAIN4086x1; GPUs={ctx.world_size}; GLOBAL B4; OUT={out}', flush=True)
        with prefetch(local_groups[update:], inputs, predictor) as iterator:
            tick = time.perf_counter()
            for rows in iterator:
                if ctx.stop(stop.is_set()): break
                count = len(groups[update])
                stats = train_step(model, optimizer, predictor, rows, projector, update=update+1, total=1031,
                    context=ctx, global_windows=count, window_slots=slots[update])
                ranks = ctx.gather(dict(seconds=time.perf_counter()-tick, windows=len(rows),
                    sources=sum(len(r[0]['source_class_id']) for r, _, _ in rows)))
                update += 1; wall = max(r['seconds'] for r in ranks)/count
                if ctx.primary:
                    item = dict(update=update, windows=count, seconds_per_window=wall,
                        rank_windows=[r['windows'] for r in ranks], rank_sources=[r['sources'] for r in ranks], **stats)
                    with (out/'progress.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(item)+'\n')
                    write_json(out/'progress.json', dict(status='training', updates=update, total_updates=1031,
                        trained_windows=sum(map(len, groups[:update])), world_size=ctx.world_size,
                        seconds_per_window=wall, updated_utc=time.time(), rank_windows=item['rank_windows']))
                    if update == 1 or update % 16 == 0:
                        print(f'COVERAGE_PROJECTION {update}/1031 CE={stats["CE"]:.4f} Lovasz={stats["Lovasz"]:.4f} '
                            f'proj={stats["projection_semantic"]:.4f}+{stats["projection_depth"]:.4f} '
                            f's/window={wall:.4f} rank_windows={item["rank_windows"]} rank_sources={item["rank_sources"]}', flush=True)
                stopping = ctx.stop(stop.is_set())
                if update % 64 == 0 or stopping: save()
                tick = time.perf_counter()
        save()
    finally: predictor.close()
    if update != 1031: raise InterruptedError('saved complete-update cursor; resume with --resume')
    if ctx.primary: print('TRAIN complete; final epoch only. Next: same64 paired quality + fixed visuals on ONE GPU.', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--dataroot', type=Path, required=True)
    p.add_argument('--train-info', type=Path, required=True)
    p.add_argument('--train-cache', type=Path, required=True, help='small-source metadata only; no geometry cache used')
    p.add_argument('--config', type=Path, default=ROOT/'configs/real_motion_occfm.yaml')
    p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--resume', action='store_true')
    return p


def main():
    a = parser().parse_args(); stop = threading.Event()
    if not 1 <= a.cpu_workers <= 16: raise ValueError('bounded CPU workers required')
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    ctx = DistributedTraining(True)
    if not ctx.active: ctx.device = torch.device(a.device)
    try:
        if ctx.active and ctx.device.type == 'cuda': dist.barrier(device_ids=[ctx.local_rank])
        with ExitStack() as stack:
            if ctx.primary:
                lock = a.out_dir.parent/'.coverage_projection_locks'/hashlib.sha256(str(a.out_dir.resolve()).encode()).hexdigest()
                lock.mkdir(parents=True, exist_ok=True)
                stack.enter_context(evaluation_lock(lock))
            ctx.barrier(); train(a, stop, ctx)
    finally: ctx.close()


if __name__ == '__main__': main()
