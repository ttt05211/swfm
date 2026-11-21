"""One matched TRAIN4086 epoch with a static-context masking augmentation."""
import argparse
from collections import Counter
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

from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.causal_forward_train.model import make_model, ForwardBody
from tools.coverage_projection.control import reference
from tools.coverage_projection.run import ROOT, local_plan, make_plan, prefetch, sha, write_json
from tools.dense_valid_ddp.engine import train_step
from tools.final_semantic_evaluation.sources import window
from tools.fpn_unet_eval.common import DenseInputs
from tools.joint_occ_experiments.run import load_cache
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from tools.source_scene_stage1.model import ReaderConfig

from .adapter import forward_window
from .masking import MASK_RECIPE, depth_for

PROTOCOL = 'forward_refiner_masked_TRAIN4086x1_v1'


def model_from_spec(spec, *, fresh=False):
    factory = make_model if fresh else ForwardBody
    return factory(config_from_mapping_v17(spec['motion']),
                   feature_width=spec['shared_history']['width'], z_bins=spec['shared_history']['z_bins'],
                   memory_width=spec['repair']['memory_width'], reader_config=ReaderConfig(**spec['reader']),
                   experiment=spec.get('experiment'))


def verify_implementation(files):
    for name, digest in files.items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or digest != sha(path):
            raise ValueError(f'checkpoint implementation changed: {name}')


def update_step(model, optimizer, predictor, rows, ctx, *, seed, update, total, windows, force=False):
    def apply(model, provider, row):
        rec, packed, ego, history = row
        raw = {k: v for k, v in packed.items() if k not in ('future_poses', 'future_gt_occ')}
        return forward_window(model, provider, (rec, raw, ego, history), poses=packed['future_poses'],
                              depth=depth_for(rec, seed, force=force))

    # Pass labels separately until the established training-step interface.
    # Neither the cut generator nor the forward path consumes future occupancy.
    packed = [(rec, dict(raw, future_poses=poses, future_gt_occ=target), ego, history)
              for (rec, raw, ego, history), poses, target in rows]
    return train_step(model, optimizer, predictor.provider, packed, ctx, global_windows=windows,
                      update=update, total_updates=total, forward=apply)


def run(a, ctx, stop):
    if ctx.world_size != 2 or ctx.device.type != 'cuda' or not torch.cuda.is_bf16_supported():
        raise RuntimeError('two BF16 CUDA training ranks required')
    ref = reference(a.control_dir)
    original = ref['contract']
    verify_implementation((original['model'].get('experiment') or {}).get('candidate_implementation', {}))
    data = {name: sha(getattr(a, name)) for name in ('config', 'train_cache', 'train_info')}
    if any(original['data'].get(name) != digest for name, digest in data.items()):
        raise ValueError('TRAIN config/index differs from original control')
    groups = make_plan(load_cache(a.train_cache)[1], original)
    local, _ = local_plan(groups, ctx.world_size, ctx.rank)
    total, seed = len(groups), original['seed']
    pcfg = make_prepare_config(load_runtime_config(a.config, original.get('overrides', [])))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = model_from_spec(original['model'], fresh=True).to(ctx.device)
    if model.configs() != original['model'] or sum(p.numel() for p in model.parameters()) != original['parameters']:
        raise ValueError('training model differs from original Refiner')
    for value in model.state_dict().values():
        dist.broadcast(value, src=0)
    rank_seed = seed + ctx.rank
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    np.random.seed(rank_seed)
    random.seed(rank_seed)
    rng = np.random.default_rng(rank_seed)
    motion_lr, refiner_lr = original['learning_rates']
    optimizer = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=motion_lr, initial_lr=motion_lr, weight_decay=1e-4),
        dict(params=model.columns.parameters(), lr=refiner_lr, initial_lr=refiner_lr, weight_decay=.01),
    ])
    inputs = DenseInputs(a.dataroot, a.train_info, cache_mib=128)
    paths = sorted((ROOT/'tools/refiner_masked_completion').glob('*.py'))
    implementations = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    for name in ('tools/fpn_unet/adapter.py', 'tools/fpn_unet/model.py', 'tools/dense_valid_ddp/engine.py',
                 'tools/shared_history_experiments/refine/geometry.py',
                 'tools/source_scene_ego_conditioned/geometry.py', 'tools/coverage_projection/run.py'):
        implementations[name] = sha(ROOT / name)
    receipt = ref['source_checkpoint_completion']
    contract = dict(
        protocol=PROTOCOL, model=original['model'], control=receipt['checkpoint'], control_sha=receipt['sha256'],
        control_world_size=original['world_size'], world_size=ctx.world_size,
        seed=seed, total_updates=total, train_keys=original['train_keys'], eval_keys=original['eval_keys'],
        group_keys=[[[str(row['scene_name']), str(row['t0_token'])] for row in group] for group in groups],
        objective=original['objective'], schedule=original['schedule'], learning_rates=original['learning_rates'],
        overrides=original.get('overrides', []), data=data,
        inventory=inputs.preflight([window(row) for group in groups for row in group]),
        mask=MASK_RECIPE, masked_population=dict(Counter(str(depth_for(row, seed)) for group in groups for row in group)),
        implementations=implementations, torch=str(torch.__version__), cuda=torch.version.cuda,
        cpu_workers=a.cpu_workers,
    )
    contract = json.loads(json.dumps(contract))
    out = a.out_dir.resolve()
    update = 0
    if a.resume:
        saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        if saved['protocol'] != PROTOCOL or saved['contract'] != contract:
            raise ValueError('resume requires the same checkpoint, sources and masking recipe')
        update = saved['updates']
        if not 0 <= update <= total:
            raise ValueError('invalid completed update cursor')
        model.load_state_dict(saved['models']['gt'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        ctx.restore_rng(saved['rank_rng'], rng)
        del saved
    elif ctx.primary:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'contract.json', contract)
    ctx.barrier()

    def save():
        states = ctx.gather(ctx.rng_state(rng))
        if ctx.primary:
            payload = dict(protocol=PROTOCOL, contract=contract, updates=update, models={'gt': model.state_dict()},
                           optimizer=optimizer.state_dict(), rank_rng=states)
            save_resume_checkpoint(out/'last.pt', payload)
            if update == total and not (out/'epoch_0001.pt').exists():
                frozen = {k: payload[k] for k in ('protocol', 'contract', 'updates', 'models')}
                save_resume_checkpoint(out/'epoch_0001.pt', frozen)
            write_json(out/'progress.json', dict(status='complete' if update == total else 'checkpointed',
                                                updates=update, total_updates=total))
        ctx.barrier()

    if update == total:
        if ctx.broadcast(not (out/'epoch_0001.pt').exists() if ctx.primary else None):
            save()
        return
    if not a.resume:
        save()
    predictor = RecoveryPredictor(model, pcfg, ctx.device, workers=a.cpu_workers)
    try:
        with prefetch(local[update:], inputs, predictor) as batches:
            for rows in batches:
                if ctx.stop(stop.is_set()):
                    break
                started = time.perf_counter()
                windows = len(groups[update])
                result = update_step(model, optimizer, predictor, rows, ctx, seed=seed,
                                     update=update+1, total=total, windows=windows)
                masked = ctx.gather(sum(depth_for(row[0][0], seed) is not None for row in rows))
                update += 1
                if ctx.primary:
                    progress = dict(update=update, windows=windows, masked_windows=sum(masked),
                                    update_seconds_per_window=(time.perf_counter()-started)/windows, **result)
                    with (out/'progress.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(progress)+'\n')
                    if update == 1 or update % 16 == 0 or update == total:
                        print(f'MASKED_REFINER {update}/{total} CE={result["CE"]:.4f} '
                              f'Lovasz={result["Lovasz"]:.4f} masked={sum(masked)}/{windows}', flush=True)
                if update % 64 == 0 and update < total:
                    save()
        save()
    finally:
        predictor.close()
    if update < total:
        raise InterruptedError(f'saved update {update}/{total}; resume with the same output directory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-dir', 'control-dir', 'dataroot', 'train-info', 'train-cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/real_motion_occfm.yaml')
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    ctx = DistributedTraining(True)
    try:
        lock = args.out_dir.resolve().parent / '.masked_refiner_locks'
        lock = lock / hashlib.sha256(str(args.out_dir.resolve()).encode()).hexdigest()
        if ctx.primary:
            lock.mkdir(parents=True, exist_ok=True)
            with evaluation_lock(lock):
                ctx.barrier()
                run(args, ctx, stop)
        else:
            ctx.barrier()
            run(args, ctx, stop)
    finally:
        ctx.close()


if __name__ == '__main__':
    main()
