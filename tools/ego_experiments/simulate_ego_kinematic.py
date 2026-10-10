#!/usr/bin/env python3
"""Synthetic mechanism check, NOT nuScenes validation or an external-planner test."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import time
import numpy as np
import torch
from real_motion.ego_navigation import ego_history_features
from real_motion.ego_trajectory_head import EgoHeadConfig, HistoryEgoTrajectoryHead
from tools.ego_experiments.ego_kinematic import KinematicEgoHead, KinematicPrior
from tools.real_motion.surface_ego_ablation_common import geometry_loss, historical_prior, trajectory_report

FAMILIES = ('constant', 'accelerate', 'turn', 'brake', 'unobserved_change')


def simulated_bank(count, seed, config=EgoHeadConfig(), *, shifted=False):
    """Independent NumPy fine-step dynamics. Future state NEVER enters features.

    Includes a negative control: future acceleration/turn changes unobservable
    from history. No predictor should be credited with resolving its ambiguity.
    Scene tokens here are independent nuisances, not future target embeddings.
    """
    rng = np.random.default_rng(seed); rows = []; targets = []; commands = []; families = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]; families.append(family)
        speed = rng.uniform(1., 12. if not shifted else 16.)
        a = rng.uniform(-1.5, 1.5) if family == 'accelerate' else 0.
        omega = rng.uniform(-.18, .18) if family == 'turn' else 0.
        if family == 'brake': a = rng.uniform(-2., -.5)
        change_a = rng.uniform(-2., 2.) if family == 'unobserved_change' else a
        change_w = rng.uniform(-.15, .15) if family == 'unobserved_change' else omega
        hist_times = np.array([-1.5, -1., -.5, 0.])
        hist_times[:3] += rng.uniform(-.015, .015, 3)
        # Independent high-resolution midpoint integration, not candidate code.
        clock = np.linspace(-1.6, 3., 461)
        middle = (clock[1:]+clock[:-1])/2
        accel = np.where(middle <= 0, a, change_a)
        rate = np.where(middle <= 0, omega, change_w)
        angle = rate*middle
        speeds = np.maximum(0, speed+accel*middle)
        increments = np.c_[-np.sin(angle), np.cos(angle)]*speeds[:, None]*np.diff(clock)[:, None]
        xy = np.vstack((np.zeros(2), increments.cumsum(0)))
        xy -= np.array([np.interp(0., clock, xy[:, k]) for k in range(2)])
        poses = np.repeat(np.eye(4)[None], 4, axis=0)
        yaw = omega*hist_times
        noise = .005 if not shifted else .02
        poses[:, :2, 3] = np.stack([np.interp(hist_times, clock, xy[:, k]) for k in range(2)], 1)
        poses[:3, :2, 3] += rng.normal(0, noise, (3, 2))
        yaw[:3] += rng.normal(0, .0005 if not shifted else .002, 3)
        poses[:, 0, 0] = np.cos(yaw); poses[:, 1, 1] = np.cos(yaw)
        poses[:, 1, 0] = np.sin(yaw); poses[:, 0, 1] = -np.sin(yaw)
        times = np.arange(1, 7)*.5
        target = np.c_[np.stack([np.interp(times, clock, xy[:, k]) for k in range(2)], 1), change_w*times]
        row = dict(objects=torch.from_numpy(rng.normal(0, .3, (config.object_slots, config.object_dim)).astype('float32')),
            object_geometry=torch.from_numpy(rng.normal(0, .3, (config.object_slots, 8)).astype('float32')),
            object_valid=torch.from_numpy(rng.random(config.object_slots) > .5),
            surfaces=torch.from_numpy(rng.normal(0, .3, (config.surface_side**2, config.surface_dim)).astype('float32')),
            surface_geometry=torch.from_numpy(rng.normal(0, .3, (config.surface_side**2, 3)).astype('float32')),
            surface_valid=torch.from_numpy(rng.random(config.surface_side**2) > .5),
            ego_history=torch.from_numpy(ego_history_features(poses, hist_times)))
        rows.append(row); targets.append(target)
        # Same three-valued, future-derived navigation information for ALL arms.
        commands.append(np.full(6, 0 if target[-1, 0] > 2 else 1 if target[-1, 0] < -2 else 2))
    bank = {k:torch.stack([r[k] for r in rows]) for k in rows[0]}
    return bank, torch.tensor(np.array(targets), dtype=torch.float32), torch.tensor(np.array(commands)), families


@torch.no_grad()
def reports(head, dataset):
    bank, target, commands, families = dataset
    pred = []
    for start in range(0, len(target), 64):
        chunk = {k:v[start:start+64] for k, v in bank.items()}
        p = historical_prior(chunk) if head is None else head.eval()(chunk, commands[start:start+64])['se2']
        pred.append(p.cpu().numpy())
    pred = np.concatenate(pred); t = target.cpu().numpy(); nav = commands.cpu().numpy()
    result = {'all':trajectory_report(pred, t, nav)}
    for family in FAMILIES:
        ids = np.array(families) == family
        result[family] = trajectory_report(pred[ids], t[ids], nav[ids])
    return result


def run(out, *, seeds=(21, 22), train_count=2048, test_count=512, epochs=3):
    out = Path(out)
    if out.exists(): raise FileExistsError('new simulation output required')
    out.mkdir(parents=True); torch.set_num_threads(1); tick = time.perf_counter(); results = []
    c = EgoHeadConfig()
    for seed in seeds:
        train = simulated_bank(train_count, seed, c)
        tests = dict(held_same=simulated_bank(test_count, seed+100, c),
                     held_shift=simulated_bank(test_count, seed+200, c, shifted=True))
        torch.manual_seed(seed)
        heads = dict(legacy=HistoryEgoTrajectoryHead(c),
            control_history=KinematicEgoHead(c, scene=False), control_scene=KinematicEgoHead(c, scene=True))
        initial = {name:reports(head, tests['held_same']) for name, head in heads.items()}
        order_rng = np.random.default_rng(seed); orders = [order_rng.permutation(train_count) for _ in range(epochs)]
        bank, target, nav, _ = train; steps = epochs*math.ceil(train_count/64)
        for name, head in heads.items():
            opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.01); update = 0
            for order in orders:
                for start in range(0, train_count, 64):
                    ids = torch.tensor(order[start:start+64]); h = {k:v[ids] for k, v in bank.items()}
                    lr = 3e-6+(3e-4-3e-6)*.5*(1+math.cos(math.pi*update/max(1,steps-1)))
                    for group in opt.param_groups: group['lr'] = lr
                    head.train(); opt.zero_grad(set_to_none=True)
                    loss = geometry_loss(head(h, nav[ids]), target[ids], radius_m=10.)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                    opt.step(); update += 1
            print(f'SIM seed={seed} arm={name} updates={update}', flush=True)
        arms = dict(cv=None, kinematic_prior=KinematicPrior(c), **heads)
        results.append(dict(seed=seed, initial=initial, reports={name:{kind:reports(head, data)
            for kind, data in tests.items()} for name, head in arms.items()},
            parameters={name:sum(p.numel() for p in h.parameters()) for name, h in heads.items()}))
    value = dict(protocol='synthetic_ego_kinematic_mechanism_v1', train_count=train_count,
        test_count=test_count, epochs=epochs, results=results, seconds=time.perf_counter()-tick,
        scope='Synthetic history-only mechanism test; NOT nuScenes or proof of planner superiority',
        selection='All seeds/families/arms reported. No search or hidden reruns.')
    (out/'simulation.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    lines = ['===== SYNTHETIC EGO MECHANISM CHECK / NOT REAL DATA =====',
        'arm                 same FDE/yaw3s     shifted FDE/yaw3s']
    for arm in results[0]['reports']:
        means = {}
        for kind in tests:
            means[kind] = [np.mean([r['reports'][arm][kind]['all'][key] if key == 'FDE_3s_m' else
                r['reports'][arm][kind]['all'][key][-1] for r in results]) for key in ('FDE_3s_m', 'yaw_mean_deg')]
        lines.append(f'{arm:20} {means["held_same"][0]:.4f}/{means["held_same"][1]:.3f} '
            f'{means["held_shift"][0]:.4f}/{means["held_shift"][1]:.3f}')
    lines += [f'seconds={value["seconds"]:.2f}', value['scope']]
    text = '\n'.join(lines)+'\n'; (out/'summary.txt').write_text(text, encoding='utf-8'); print(text)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out-dir', required=True)
    a = p.parse_args(); run(a.out_dir)
