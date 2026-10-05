#!/usr/bin/env python3
"""CPU/CUDA synthetic architecture screen, NOT nuScenes evidence or method FPS.

One bounded bundle: old 196/36-token cost, four sparse heads, supervised toy
generalization, expressivity ceilings, full-Z/collision safety, forward/backward
cost, and registered-input repair/transport cost. No server assets are required.
"""
from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import contextlib
import copy
import hashlib
from dataclasses import dataclass
import json
import platform
import time
import numpy as np
import torch
from torch.nn import functional as F

from real_motion.causal_column_completion import ColumnConfig, REFINE, CONTEXT_DIM
from real_motion.joint_causal_columns import LinkedColumns
from real_motion.sparse_column_readout import token_probe
from real_motion.sparse_evidence_repair import (CanonicalObservation, EvidenceMemory,
    SparseRepairHead, build_memory, render_add_only, PROTOCOL, FREE, STATIC)


@dataclass
class ToyScene:
    observations: list
    memory: EvidenceMemory
    target: np.ndarray  # [M,6], separate SUPERVISION ONLY
    context: np.ndarray
    future: np.ndarray
    centers: np.ndarray
    future_centers: np.ndarray
    yaw: np.ndarray
    poses: np.ndarray
    grids: dict
    query_keys: np.ndarray
    point_query: np.ndarray
    unseen_positive: int
    nonrigid: bool


def source_context_from_observations(observations, sources):
    """A toy proxy for V18 latents, computed ONLY from occupied past samples.

    No primitive ground-truth size, future motion, annotations or labels. Robust
    observed extent can still be wrong; this is not a real trained V18 latent.
    """
    result = np.zeros((sources,128),np.float32)
    for actor in range(sources):
        rows = [f.occupied[f.occupied[:,0] == actor] for f in observations]
        all_rows = np.concatenate(rows)
        if not len(all_rows):
            continue
        extent = np.quantile(all_rows[:,2:4],.9,axis=0)-np.quantile(all_rows[:,2:4],.1,axis=0)
        result[actor,:6] = (extent[0]/16, extent[1]/16,
            float(all_rows[0,1] == 7), len(rows[-1])/max(1,len(all_rows)),
            float(all_rows[:,4].mean()/16), float(all_rows[:,4].std()/16))
    return result


def make_scene(seed, *, sources=4, static_columns=64, changing=False):
    """Primitive surfaces, partial observations, false association and stale points.

    The canonical alignment here is known synthetic causal motion (optimistic);
    learned registration errors are NOT tested by this speed screen. Non-rigid
    points swap sides after 1.5s: a shared scalar actor gate cannot capture that.
    Dynamic/static candidates both keep ordered Z, no future occupancy inputs.
    """
    rng = np.random.default_rng(seed)
    frames = [([], []) for _ in range(4)]
    true_lookup = {}
    for actor in [*range(sources), STATIC]:
        if actor == STATIC:
            xy = rng.choice(32*32, size=min(static_columns, 32*32), replace=False)
            cells = np.column_stack((xy // 32, xy % 32, rng.choice((0, 1, 3), len(xy), p=(.65, .25, .1))))
            cls = np.where(cells[:, 1] < 16, 11, 13)
            cls[cells[:, 2] == 3] = 15
            is_true = rng.random(len(cells)) > .25
        else:
            # Surface, not solid boxes; random width breaks trivial fixed shape.
            w, l = int(rng.integers(3, 6)), int(rng.integers(2, 5))
            xyz = np.stack(np.meshgrid(np.arange(-w, w+1), np.arange(-l, l+1),
                np.arange(1, 6), indexing='ij'), -1).reshape(-1, 3)
            surface = (np.abs(xyz[:, 0]) == w) | (np.abs(xyz[:, 1]) == l) | (xyz[:, 2] == 5)
            cells = xyz[surface]
            # Disconnected same-class historical false association is intentional.
            noise = np.column_stack((rng.integers(-8, 9, 24), rng.integers(-7, 8, 24), rng.integers(0, 9, 24)))
            cells = np.unique(np.concatenate((cells, noise)), axis=0)
            is_true = ((np.abs(cells[:, 0]) <= w) & (np.abs(cells[:, 1]) <= l)
                & (cells[:, 2] >= 1) & (cells[:, 2] <= 5))
            cls = np.full(len(cells), 7 if changing and actor % 2 else 4)
        rows = np.column_stack((np.full(len(cells), actor), cls, cells)).astype(np.int64)
        observed = rng.random((len(cells), 4)) > .15
        presence = observed & (rng.random((len(cells), 4)) < .78)
        # False/stale correspondence lasts exactly one old frame, never t0.
        presence[~is_true] = False
        bad = np.flatnonzero(~is_true)
        presence[bad, rng.integers(0, 3, len(bad))] = True
        observed |= presence
        # t0 occlusion is severe, both for static evidence and source surfaces.
        presence[:, -1] &= rng.random(len(cells)) > .65
        observed[:, -1] &= ~is_true | presence[:, -1] | (rng.random(len(cells)) > .5)
        observed |= presence
        for t in range(4):
            frames[t][0].append(rows[presence[:, t]])
            frames[t][1].append(rows[observed[:, t]])
        for row, true in zip(rows, is_true):
            y = np.full(6, bool(true))
            # Symmetric per-point changes have incompatible horizon labels.
            # Every candidate STILL comes from genuine past observations.
            if true and actor >= 0 and cls[0] == 7 and row[4] >= 3 and abs(row[2]) >= 2:
                y[:3] = row[2] > 0
                y[3:] = row[2] < 0
            true_lookup[tuple(row)] = y
    observations = [CanonicalObservation(np.concatenate(occ), np.concatenate(vis)) for occ, vis in frames]
    context = source_context_from_observations(observations,sources)
    memory = build_memory(observations)
    target = np.asarray([true_lookup[tuple(k)] for k in memory.keys], bool).reshape(-1, 6)
    # Explicit mass missing from the evidence-only support: NOT passed to heads.
    seen = set(map(tuple, memory.keys))
    unseen_positive = sum(int(y.sum()) for k, y in true_lookup.items() if k not in seen)
    future = np.repeat(context[:, None], 6, axis=1)
    future[:, :, 6] = np.arange(1, 7, dtype=np.float32)[None] / 6
    centers = np.array([[12 + 12*(i % 8), 12 + 12*(i // 8), 0] for i in range(sources)], float).reshape(-1, 3)
    future_centers = np.repeat(centers[None], 6, axis=0)
    future_centers[:, :, 0] += np.arange(1, 7)[:, None] * .7
    future_centers[:, :, 1] -= np.arange(1, 7)[:, None] * .2
    yaw = np.arange(1, 7)[:, None] * np.linspace(-.08, .08, sources)[None]
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[:, 0, 3] = -np.arange(1, 7) * .15
    # Old patch input path: native canonical maps -> 4x7x7x16 per query.
    # Registration/SE3 inverse lookup is deliberately excluded (favors old path).
    grids = {}
    for actor in np.unique(memory.keys[:, 0]):
        lab = np.full((4, 56, 56, 16), 18, np.uint8)
        flags = np.zeros_like(lab)
        for t, frame in enumerate(observations):
            for rows, occupied in ((frame.visible, False), (frame.occupied, True)):
                rows = rows[rows[:, 0] == actor]
                x, y, z = rows[:, 2]+12, rows[:, 3]+12, rows[:, 4]
                lab[t, x, y, z] = rows[:, 1] if occupied else FREE
                flags[t, x, y, z] = 3 if occupied else 1
        grids[int(actor)] = (lab, flags)
    query_keys, point_query = np.unique(memory.keys[:, :4], axis=0, return_inverse=True)
    return ToyScene(observations, memory, target, context, future, centers, future_centers,
        yaw, poses, grids, query_keys, point_query, unseen_positive, changing)


def column_inputs(scene, query_ids, horizons, device='cpu'):
    query_ids, horizons = np.asarray(query_ids), np.asarray(horizons)
    rows = scene.query_keys[query_ids]
    n = len(rows)
    hist = np.empty((n, 4, 7, 7, 16), np.uint8)
    flags = np.empty_like(hist)
    offset = np.arange(-3, 4)
    for actor in np.unique(rows[:, 0]):
        ids = np.flatnonzero(rows[:, 0] == actor)
        lab, bit = scene.grids[int(actor)]
        x = rows[ids, 2]+12; y = rows[ids, 3]+12
        xx = x[:, None, None] + offset[None, :, None]
        yy = y[:, None, None] + offset[None, None, :]
        hist[ids] = lab[:, xx, yy, :].transpose(1, 0, 2, 3, 4)
        flags[ids] = bit[:, xx, yy, :].transpose(1, 0, 2, 3, 4)
    context = np.zeros((n, CONTEXT_DIM), np.float32)
    context[:, :2] = rows[:, 2:4] / 16
    context[:, 2] = (horizons + 1) / 6
    context[:, 3] = rows[:, 0] == STATIC
    live = np.zeros((n, 128), np.float32)
    dyn = rows[:, 0] >= 0
    live[dyn] = scene.future[rows[dyn, 0], horizons[dyn]]
    base = hist[:, -1, 3, 3].copy()
    base[base == 18] = FREE
    arrays = dict(history=hist, flags=flags, base=base, fallback=base.copy(), context=context,
        kind=np.full(n, REFINE, np.int64), classes=rows[:, 1], source_features=live)
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


def column_probabilities(model, scene, *, tokens, device, batch=128):
    p = np.empty((6, len(scene.query_keys), 16), np.float32)
    mode = token_probe(model) if tokens == 36 else contextlib.nullcontext()
    with mode:
        for h in range(6):
            for start in range(0, len(scene.query_keys), batch):
                ids = np.arange(start, min(start+batch, len(scene.query_keys)))
                _, r = model(**column_inputs(scene, ids, np.full(len(ids), h), device))
                p[h, ids] = (r[:, :, 1] - r[:, :, 0]).sigmoid().detach().cpu().numpy()
    return p[:, scene.point_query, scene.memory.keys[:, 4]].T


def supervised_columns(scene, point_ids, horizons, device):
    # Same column/horizon can supervise multiple Z points. The real Local model
    # encodes it once, so do not overstate its training cost by duplicating it.
    queries, inverse = np.unique(np.column_stack((scene.point_query[point_ids], horizons)),
        axis=0, return_inverse=True)
    return column_inputs(scene, queries[:,0], queries[:,1], device), torch.as_tensor(inverse,device=device)


def point_probabilities(model, scene, device):
    p = model(**scene.memory.tensors(device, neighborhood=model.neighborhood),
        source_context=torch.as_tensor(scene.context, device=device),
        future_queries=torch.as_tensor(scene.future, device=device))
    return p.sigmoid().detach().cpu().numpy()


def finish(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measured(fn, device, repeats):
    values = []
    for _ in range(repeats):
        finish(device); start = time.perf_counter()
        fn(); finish(device)
        values.append(time.perf_counter()-start)
    return dict(median_seconds=float(np.median(values)), samples_seconds=values)


def binary_metrics(prediction, target):
    p, y = np.asarray(prediction, bool), np.asarray(target, bool)
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn),
        iou=tp/max(1, tp+fp+fn), f1=2*tp/max(1, 2*tp+fp+fn))


def expressivity(scene):
    eligible = ~scene.memory.presence[:, -1]
    y = scene.target[eligible]
    # A fixed shape must choose the SAME action at all six future horizons.
    shared = np.repeat((y.sum(1) >= 3)[:, None], 6, axis=1)
    return dict(evidence_supported_positive=int(y.sum()), unobserved_positive=scene.unseen_positive,
        evidence_recall_ceiling=float(y.sum()/max(1, y.sum()+scene.unseen_positive)),
        fixed_shape_best_binary_accuracy=float(np.mean(shared == y)),
        fixed_shape_oracle=binary_metrics(shared, y),
        future_point_oracle=binary_metrics(y, y),
        note='oracle is diagnostic, never provided as model input; outside-support positives remain unrecoverable')


def quality_screen(device, *, steps=192, seed=507, batch=32):
    """Identical point minibatches, ordinary BCE and no teacher/threshold search.

    Train/test scenes are disjoint random seeds; old models are trained from
    random initialization too. Toy binary action IoU is NOT semantic mIoU.
    """
    train = [make_scene(seed+i, sources=3, static_columns=48, changing=i % 2 == 0) for i in range(16)]
    test = [make_scene(seed+10000+i, sources=3, static_columns=48, changing=i % 2 == 0) for i in range(8)]
    models = {m: SparseRepairHead(m).to(device) for m in ('once', 'actor_gate', 'cached_future', 'local_consensus')}
    models.update({f'column_{t}': LinkedColumns(ColumnConfig(), 128, history_frames=4).to(device)
        for t in (196, 36)})
    optimizers = {k: torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4) for k, m in models.items()}
    rng = np.random.default_rng(seed)
    losses = {k: [] for k in models}
    for step in range(steps):
        scene = train[int(rng.integers(len(train)))]
        candidates = np.flatnonzero(~scene.memory.presence[:, -1])
        ids = rng.choice(candidates, batch, replace=len(candidates) < batch)
        h = rng.integers(0, 6, batch)
        inp = scene.memory.tensors(device)
        inp = {k: v[torch.as_tensor(ids, device=device)] for k, v in inp.items()}
        source = torch.as_tensor(scene.context, device=device)
        future = torch.as_tensor(scene.future, device=device)
        target = torch.as_tensor(scene.target[ids, h].astype(np.float32), device=device)
        column, query_rows = supervised_columns(scene, ids, h, device)
        for name, model in models.items():
            model.train(); optimizers[name].zero_grad(set_to_none=True)
            if name.startswith('column_'):
                # token_probe forbids training. Override only in this synthetic
                # harness with the identical selected memory mask at readout.
                old_decode = model.decode_history
                if name == 'column_36':
                    def decode(x, invalid, *a, **kw):
                        selected = torch.as_tensor([t*49+x*7+y for t in range(4)
                            for x in (0, 3, 6) for y in (0, 3, 6)], device=device)
                        # Pad back to 196 only for shape validation, then blocks
                        # see the actual 36-token memory in a temporary wrapper.
                        return _decode_sparse(model, x[:, selected], invalid[:, selected], *a, **kw)
                    model.decode_history = decode
                try:
                    _, r = model(**column)
                finally:
                    model.decode_history = old_decode
                output = r[query_rows,
                    torch.as_tensor(scene.memory.keys[ids, 4], device=device), 1] - r[
                    query_rows, torch.as_tensor(scene.memory.keys[ids, 4], device=device), 0]
            else:
                features = inp
                if model.neighborhood:
                    features = {k: v[torch.as_tensor(ids, device=device)] for k,v in
                        scene.memory.tensors(device, neighborhood=True).items()}
                output = model(**features, source_context=source, future_queries=future)[
                    torch.arange(batch, device=device), torch.as_tensor(h, device=device)]
            loss = F.binary_cross_entropy_with_logits(output, target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizers[name].step(); losses[name].append(float(loss.detach().cpu()))
        if (step+1) % 32 == 0 or step == steps-1:
            print(f'TOY_TRAIN {step+1}/{steps} '+json.dumps({k: round(v[-1], 4) for k,v in losses.items()}), flush=True)
    report = {}
    with torch.inference_mode():
        for name, model in models.items():
            model.eval(); predicted, target, by_regime = [], [], {}
            for scene in test:
                prob = (column_probabilities(model, scene, tokens=int(name.split('_')[-1]), device=device)
                    if name.startswith('column_') else point_probabilities(model, scene, device))
                eligible = ~scene.memory.presence[:, -1]
                predicted.append(prob[eligible] >= .5); target.append(scene.target[eligible])
                label = 'changing' if scene.nonrigid else 'rigid'
                by_regime.setdefault(label, [[], []])[0].append(predicted[-1])
                by_regime[label][1].append(target[-1])
            report[name] = dict(binary_metrics(np.concatenate(predicted), np.concatenate(target)),
                by_regime={k: binary_metrics(np.concatenate(p), np.concatenate(y)) for k,(p,y) in by_regime.items()},
                initial_loss=losses[name][0], final_block_loss=float(np.mean(losses[name][-32:])))
    return dict(steps=steps, points_per_update=batch, train_scenes=len(train), test_scenes=len(test),
        train_seeds=[seed, seed+15], test_seeds=[seed+10000, seed+10007],
        thresholds='fixed 0.5, no tuning', results=report,
        warning='same point/action supervision, different architectures; short toy training not converged nuScenes equivalence'), models


def _decode_sparse(model, x, invalid, base, fallback, context, kind, classes, *, query_extra=None):
    """36-token training toy only; retain original coordinates, zero unknown row."""
    empty = invalid.all(1)
    x = x.clone(); invalid = invalid.clone()
    x[:, 0] = torch.where(empty[:, None], 0., x[:, 0]); invalid[:, 0] &= ~empty
    q = (model.query(torch.cat((context.float(), model.semantic(base.long()).flatten(1),
        model.semantic(fallback.long()).flatten(1)), dim=1)) + model.kind(kind.long())
        + model.classes(classes.long())).unsqueeze(1)
    if query_extra is not None:
        q = q + query_extra[:, None].to(q.dtype)
    for block in model.decoder:
        q = block(q, x, invalid)
    return model.norm(q[:, 0])


def training_cost(models, scene, device, repeats):
    """Real forward/BCE/backward/AdamW. Does NOT include V18 motion training.

    All architectures supervise the same 256 point/horizon pairs. Dynamic
    source contexts and six future queries require gradients in every variant.
    """
    rng = np.random.default_rng(770)
    candidates = np.flatnonzero(~scene.memory.presence[:, -1])
    ids = rng.choice(candidates, 256, replace=len(candidates) < 256)
    h = rng.integers(0, 6, len(ids))
    column, query_rows = supervised_columns(scene, ids, h, device)
    y = torch.as_tensor(scene.target[ids, h].astype(np.float32), device=device)
    results = {}
    for name, original in models.items():
        model = copy.deepcopy(original).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        source = torch.tensor(scene.context, device=device, requires_grad=True)
        future = torch.tensor(scene.future, device=device, requires_grad=True)
        def run():
            optimizer.zero_grad(set_to_none=True); source.grad = None; future.grad = None
            if name.startswith('column_'):
                old = model.decode_history
                if name == 'column_36':
                    def decode(x, invalid, *a, **kw):
                        selected = [t*49+x*7+y for t in range(4) for x in (0,3,6) for y in (0,3,6)]
                        return _decode_sparse(model, x[:, selected], invalid[:, selected], *a, **kw)
                    model.decode_history = decode
                try:
                    queries = np.unique(np.column_stack((scene.point_query[ids],h)),axis=0)
                    actors = torch.as_tensor(scene.query_keys[queries[:,0], 0], device=device)
                    zeros = future.new_zeros((1, 6, 128))
                    table = torch.cat((zeros, future), 0)
                    live = table[torch.where(actors >= 0, actors+1, 0), torch.as_tensor(queries[:,1], device=device)]
                    _, r = model(**{**column, 'source_features': live})
                finally:
                    model.decode_history = old
                z = torch.as_tensor(scene.memory.keys[ids, 4], device=device)
                ar = query_rows
                value = r[ar, z, 1] - r[ar, z, 0]
            else:
                point = {k: v[torch.as_tensor(ids, device=device)] for k,v in
                    scene.memory.tensors(device, neighborhood=model.neighborhood).items()}
                value = model(**point, source_context=source, future_queries=future)[
                    torch.arange(len(ids), device=device), torch.as_tensor(h, device=device)]
            F.binary_cross_entropy_with_logits(value, y).backward(); optimizer.step()
        run()  # allocate optimizer state, not timed
        results[name] = measured(run, device, repeats)
        results[name]['live_source_gradient'] = bool((source.grad is not None and source.grad.abs().sum() > 0)
            or (future.grad is not None and future.grad.abs().sum() > 0))
        results[name]['parameters'] = sum(p.numel() for p in model.parameters())
    return results


def speed_screen(models, device, *, repeats=2, seed=3007):
    result = {}
    shapes = [('small', 4, 64), ('mixed', 20, 400), ('static_heavy', 8, 1000)]
    for label, ns, nc in shapes:
        scene = make_scene(seed, sources=ns, static_columns=nc, changing=True)
        # Bounded six dense frames, stress coverage/OOB separately in unit tests.
        shape = (max(64, 24+12*min(ns, 8)), max(64, 24+12*((ns+7)//8)), 16)
        empty = np.full((6, *shape), FREE, np.uint8)
        current = np.repeat(scene.memory.presence[:, -1, None], 6, axis=1).astype(np.float32)
        # Simulated transport baseline carries ONLY t0 points. The repair still
        # cannot edit those points or any unrelated protected occupied voxel.
        current_memory = EvidenceMemory(scene.memory.keys, np.zeros_like(scene.memory.presence),scene.memory.visibility)
        baseline, _ = render_add_only(current_memory,current,empty,scene.centers,scene.future_centers,
            scene.yaw,scene.poses,np.zeros(3),np.ones(3))
        baseline[:, :4, :4, 1] = 15
        fixed = np.repeat((np.arange(len(scene.memory)) % 3 == 0)[:, None], 6, axis=1).astype(np.float32)
        def render(memory):
            return render_add_only(memory, fixed, baseline, scene.centers, scene.future_centers,
                scene.yaw, scene.poses, np.zeros(3), np.ones(3))
        trials = {}
        with torch.inference_mode():
            for name, model in models.items():
                model.eval()
                def inference(memory=None):
                    active = scene if memory is None else ToyScene(**{**vars(scene), 'memory': memory})
                    if name.startswith('column_'):
                        return column_probabilities(model, active, tokens=int(name.split('_')[-1]), device=device)
                    return point_probabilities(model, active, device)
                inference()  # untimed warmup
                trials[name] = dict(head_and_input_pack=measured(inference, device, repeats))
                def pipeline():
                    memory = build_memory(scene.observations)
                    probability = inference(memory)
                    if probability.shape != fixed.shape or not np.isfinite(probability).all():
                        raise RuntimeError('invalid timed inference')
                    render(memory)  # same accepted sparse ADD workload for all
                trials[name]['registered_input_pipeline'] = measured(pipeline, device, repeats)
                print(f'TOY_SPEED {label} {name} head={trials[name]["head_and_input_pack"]["median_seconds"]:.4f}s '
                    f'repair6={trials[name]["registered_input_pipeline"]["median_seconds"]:.4f}s', flush=True)
        build = measured(lambda: build_memory(scene.observations), device, repeats)
        rendering = measured(lambda: render(scene.memory), device, repeats)
        old = trials['column_196']['registered_input_pipeline']['median_seconds']
        for trial in trials.values():
            trial['speedup_vs_column196'] = old/trial['registered_input_pipeline']['median_seconds']
        result[label] = dict(sources=ns, static_input_columns=nc, evidence_voxels=len(scene.memory),
            evidence_missing_t0=int((~scene.memory.presence[:, -1]).sum()),
            old_queries_all_six=len(scene.query_keys)*6, dense_shape=list(shape), memory_build=build,
            common_six_dense_composition=rendering, oracle=expressivity(scene), trials=trials)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=384)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.threads, args.steps, args.repeats) < 1:
        parser.error('threads/steps/repeats must be positive')
    out = Path(args.out_dir)
    if (out/'result.json').exists():
        parser.error('use a NEW output directory; existing result is protected')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(107)
    root = Path(__file__).resolve().parents[2]
    implementation = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in (
        'real_motion/sparse_evidence_repair.py','tools/real_motion/benchmark_sparse_repair_synthetic.py')}
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    start = time.perf_counter()
    quality, models = quality_screen(device, steps=args.steps)
    repeated = SparseRepairHead('repeated_future').to(device)
    repeated.load_state_dict(models['cached_future'].state_dict())
    models['repeated_future'] = repeated
    check = make_scene(50777, changing=True)
    with torch.inference_mode():
        a, b = point_probabilities(models['cached_future'], check, device), point_probabilities(repeated, check, device)
    if not np.allclose(a, b, rtol=1e-5, atol=1e-6):
        raise RuntimeError('cached/repeated point encoding math check failed')
    training = training_cost(models, check, device, args.repeats)
    speed = speed_screen(models, device, repeats=args.repeats)
    result = dict(protocol=PROTOCOL, status='complete', implementation_sha256=implementation,
        environment=dict(device=str(device),
        actual_cuda=device.type == 'cuda', platform=platform.platform(), python=sys.version.split()[0],
        torch=torch.__version__, threads=args.threads, precision='float32'), quality=quality, training_cost=training,
        speed=speed, cached_vs_repeated_probabilities_allclose=True, seconds=time.perf_counter()-start,
        boundary=dict(included='registered causal observations -> memory + repair heads + six SE2 sparse scatters + six dense output copies',
            excluded='raw sensors/I/O, source extraction, association/registration estimation, V18/Strong/KTA, generation, GT metrics',
            baseline_favorable_simplification='old patch sampling in canonical coordinates, no per-voxel SE3 inverse transforms',
            composition='controlled identical accepted ADD workload, no REMOVE; not measured autonomous method FPS'),
        caveats=['CPU speedup does not transfer numerically to L40S',
            'toy repair binary IoU is NOT nuScenes semantic mIoU',
            'different support/expressivity; no guarantee to preserve static/dynamic teacher gains',
            'registration is optimistic; same-class registration errors and unseen points require real-data audit',
            'no live model/weights/thresholds or server artifacts modified'])
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['===== SYNTHETIC REPAIR ARCHITECTURE SCREEN =====',
        f'device={device}; threads={args.threads}; strict 4 history -> 6 future; NOT method FPS',
        'Toy action classification: fixed threshold=.5, disjoint scene seeds, short training',
        *[f'{k}: precision={v["precision"]:.3f} recall={v["recall"]:.3f} binary_IoU={v["iou"]:.3f} '
            f'changing_IoU={v["by_regime"]["changing"]["iou"]:.3f}' for k,v in quality['results'].items()],
        'Forward+BCE+backward+AdamW, same 256 point/horizon pairs; excludes V18 motion:',
        *[f'{k}: {v["median_seconds"]*1000:.2f}ms live_source_gradient={v["live_source_gradient"]}' for k,v in training.items()]]
    for label, r in speed.items():
        lines += [f'--- {label}: evidence={r["evidence_voxels"]} old_queries6={r["old_queries_all_six"]} ---',
            *[f'{k}: head+pack={v["head_and_input_pack"]["median_seconds"]*1000:.2f}ms '
                f'registered_pipeline6={v["registered_input_pipeline"]["median_seconds"]*1000:.2f}ms '
                f'speedup_vs196={v["speedup_vs_column196"]:.2f}x' for k,v in r['trials'].items()],
            f'common memory build={r["memory_build"]["median_seconds"]*1000:.2f}ms '
            f'common six dense scatter={r["common_six_dense_composition"]["median_seconds"]*1000:.2f}ms',
            'fixed-shape oracle='+json.dumps(r['oracle'])]
    lines += ['No nuScenes quality/FPS claim, no automatic deployment or server training.',
        f'elapsed_seconds={result["seconds"]:.2f}']
    (out/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
