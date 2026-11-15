"""Train and evaluate a direct, ego-conditioned convolutional OCC baseline."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import signal
import threading

import numpy as np
import torch
from torch import nn

from tools.final_semantic_evaluation import metrics
from tools.fpn_unet.model import Residual2D
from tools.fpn_unet_eval.common import DenseInputs
from tools.paper_ablation_eval.population import PreparedInputs, select_jobs
from tools.paper_ablation_train.control import plan, reference, sha
from tools.paper_ablation_train.partition import partition
from tools.paper_ablation_train.run import seed_all
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from tools.real_motion.static_evidence_selector_common import write_json
from tools.shared_history_experiments.full_gt.engine import semantic_loss
from tools.source_scene_ego_conditioned.model import relative_ego_features


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'outputs/p0_f9_shared_history'
DATA = Path('/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes')
PROTOCOL = 'carryworld_direct_bev_conv_full_train1_v1'
HORIZONS = (1., 2., 3.)
INDICES = (1, 3, 5)


class DirectBEV(nn.Module):
    """Encode only t0, then predict each future semantic grid independently."""
    def __init__(self, z_bins=16, width=64):
        super().__init__()
        self.z_bins = z_bins
        self.semantic = nn.Embedding(18, 4)
        self.stem = nn.Sequential(nn.Conv2d(z_bins * 4, width, 4, stride=2, padding=1, bias=False),
                                  nn.GroupNorm(4, width), nn.ReLU(), Residual2D(width), Residual2D(width))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width + 4, width, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(width, 18 * z_bins * 4, 1), nn.PixelShuffle(2))
            for _ in range(6)
        ])

    def forward(self, current, relative_ego):
        x, y, z = current.shape
        if x % 2 or y % 2 or z != self.z_bins or relative_ego.shape != (6, 4):
            raise ValueError('expected one even-XY OCC grid and six relative future poses')
        embedded = self.semantic(current.long()).flatten(2).permute(2, 0, 1)[None]
        field = self.stem(embedded)
        ego = relative_ego.to(field.dtype) / field.new_tensor([40., 40., 1., 1.])
        logits = [head(torch.cat((field, pose.view(1, 4, 1, 1).expand(1, 4, *field.shape[-2:])), 1))
                  for head, pose in zip(self.heads, ego)]
        return torch.cat(logits, dim=0).reshape(6, 18, z, x, y).permute(0, 1, 3, 4, 2).reshape(6, 18, -1)


def predict(model, inputs, scene, token, current_pose, future_poses, device):
    grid, _ = inputs.frame('occ', scene, token)
    current = torch.from_numpy(grid.copy()).to(device)
    ego = relative_ego_features(torch.as_tensor(np.asarray(current_pose), device=device, dtype=torch.float64),
                                torch.as_tensor(np.asarray(future_poses), device=device, dtype=torch.float64))
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        return model(current, ego)


def train_step(model, optimizer, inputs, group, local, ctx):
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(2, device=ctx.device)
    for record in local:
        scene, token, future = (record[k] for k in ('scene_name', 't0_token', 'future_tokens'))
        logits = predict(model, inputs, scene, token, inputs.source.pose(token),
                         [inputs.source.pose(t) for t in future], ctx.device)
        labels = np.stack([inputs.frame('occ', scene, t)[0] for t in future])
        loss, ce, lovasz = semantic_loss(logits, labels)
        (loss.mean() / len(group)).backward()
        totals += torch.stack((ce.detach().mean(), lovasz.detach().mean())) / len(group)
    if ctx.stop(not bool(torch.isfinite(totals).all())):
        raise RuntimeError('nonfinite CE or Lovasz loss')
    ctx.synchronize_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return ctx.sum(totals).tolist()


def train(args, stop):
    ctx = DistributedTraining(True)
    try:
        if not ctx.active or not torch.cuda.is_bf16_supported():
            raise RuntimeError('launch training with torchrun and two BF16-capable GPUs')
        with training_directory(args.out_dir, ctx, args.resume):
            return run_training(args, stop, ctx)
    finally:
        ctx.close()


def run_training(args, stop, ctx):
    source = reference(args.reference, 'c_transport')
    info = args.train_info or args.dataroot / 'nuscenes_infos_train_temporal_v3_scene.pkl'
    data = dict(train_info=sha(info), train_cache=sha(args.train_cache))
    if any(source['data'][key] != value for key, value in data.items()):
        raise ValueError('TRAIN index or metadata differs from the original component ablations')
    _, records = load_cache(args.train_cache)
    keys, groups, budget = plan(records, source, 'c_transport')
    inputs = DenseInputs(args.dataroot, info, cache_mib=128)
    seed_all(source['seed'])
    rng = np.random.default_rng(source['seed'] + ctx.rank)
    model = DirectBEV().to(ctx.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=source['learning_rates'][0], weight_decay=1e-4)
    contract = dict(protocol=PROTOCOL, reference_sha256=sha(args.reference), data=data, budget=budget,
                    parameters=sum(p.numel() for p in model.parameters()), world_size=ctx.world_size,
                    dataroot=str(args.dataroot.resolve()), implementation_sha256=sha(__file__),
                    train_windows=len(keys), input='current GT OCC + six GT relative ego poses')
    out = args.out_dir
    last = out / 'last.pt'
    update = 0
    if args.resume:
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['contract'] != contract or not 0 <= saved['updates'] <= len(groups):
            raise ValueError('resume requires the same baseline, data, and training budget')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        ctx.restore_rng(saved['rank_rng'], rng)
        update = saved['updates']

    def save():
        states = ctx.gather(ctx.rng_state(rng))
        if ctx.primary:
            payload = dict(contract=contract, updates=update, model=model.state_dict(),
                           optimizer=optimizer.state_dict(), rank_rng=states)
            save_resume_checkpoint(last, payload)
            if update == len(groups):
                save_resume_checkpoint(out / 'epoch_0001.pt', dict(contract=contract, updates=update,
                                                                    model=model.state_dict()))
        ctx.barrier()

    if not args.resume:
        save()
    model.train()
    for group in groups[update:]:
        if ctx.stop(stop.is_set()):
            break
        factor = .01 + .99 * .5 * (1 + math.cos(math.pi * update / budget['schedule_reference_updates']))
        optimizer.param_groups[0]['lr'] = source['learning_rates'][0] * factor
        ce, lovasz = train_step(model, optimizer, inputs, group, partition(group, ctx.world_size, update)[ctx.rank], ctx)
        update += 1
        if ctx.primary and (update == 1 or update % 16 == 0 or update == len(groups)):
            print(f'DIRECT_BEV {update}/{len(groups)} CE={ce:.4f} Lovasz={lovasz:.4f}', flush=True)
        if update % 64 == 0 and update < len(groups):
            save()
    save()
    return 0 if update == len(groups) else 130


@contextmanager
def training_directory(out, ctx, resume):
    if ctx.primary:
        out.mkdir(parents=True, exist_ok=resume)
        with evaluation_lock(out):
            ctx.barrier()
            yield
    else:
        ctx.barrier()
        yield


@torch.no_grad()
def evaluate(args):
    if torch.cuda.device_count() != 1:
        raise RuntimeError('expose one GPU for evaluation')
    device = torch.device('cuda:0')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    contract = saved['contract']
    if (contract['protocol'] != PROTOCOL or contract['budget']['run_epochs'] != 1
            or contract['train_windows'] != 20430 or saved['updates'] != contract['budget']['stop_updates']):
        raise ValueError('expected the completed one-epoch direct BEV checkpoint')
    model = DirectBEV().to(device).eval()
    model.load_state_dict(saved['model'])
    content = (args.ablation_dir / 'population.json').read_bytes()
    population = json.loads(content)
    jobs = select_jobs(population, predicted=False)
    if len(jobs) != 4369 or len({job.t0_token for job in jobs}) != 4369 or len({job.scene_name for job in jobs}) != 150:
        raise ValueError('expected the original OCC VAL4369 population')
    inputs = PreparedInputs(population)
    counts = metrics.empty(len(HORIZONS))
    for i, job in enumerate(jobs, start=1):
        logits = predict(model, inputs, job.scene_name, job.t0_token, inputs.gt_poses[job.t0_token],
                         inputs.future_poses(job, predicted=False), device)
        predictions = logits[list(INDICES)].argmax(1).reshape(3, 200, 200, 16).byte().cpu().numpy()
        for h, (prediction, target) in enumerate(zip(predictions, inputs.targets(job, INDICES), strict=True)):
            metrics.add(counts, h, prediction, target)
        if i % 64 == 0 or i == len(jobs):
            print(f'DIRECT_BEV_VAL {i}/{len(jobs)}', flush=True)
    result = metrics.report(counts, HORIZONS, moving=False)
    write_json(args.out, dict(status='complete', windows=len(jobs), checkpoint_sha256=sha(args.checkpoint),
                              population_sha256=hashlib.sha256(content).hexdigest(), report=result,
                              confusion=counts['confusion'].tolist()))
    print('| Horizon | mIoU (%) | IoU (%) |\n|---|---:|---:|')
    for label, row in [*result['per_horizon'].items(), ('Avg', result['average'])]:
        print(f'| {label} | {row["mIoU"]:.3f} | {row["IoU"]:.3f} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--out-dir', type=Path, required=True)
    training.add_argument('--dataroot', type=Path, default=DATA)
    training.add_argument('--reference', type=Path, default=RUNS / 'adaln_forward_full_2ep_20261026_215351/contract.json')
    training.add_argument('--train-cache', type=Path, default=ROOT / 'data/p0_f9_v18_se2_train_full.pt')
    training.add_argument('--train-info', type=Path)
    training.add_argument('--resume', action='store_true')
    evaluation = commands.add_parser('eval')
    evaluation.add_argument('--checkpoint', type=Path, required=True)
    evaluation.add_argument('--ablation-dir', type=Path, required=True)
    evaluation.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'train':
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        return train(args, stop)
    else:
        if args.out.exists():
            parser.error('choose a new evaluation output path')
        evaluate(args)


if __name__ == '__main__':
    raise SystemExit(main())
