"""Separate actual WM loss gradients and diagnose bounded CCR edits."""
from collections import defaultdict
import numpy as np
import torch
from ..engine import forward_window, hard_predictions
from ..model import source_auxiliary
from ..fast_autograd import RenderGradientBridge
from ..batching.geometry import GroupedScene, grouped_loss
from ..soft_geometry import continuous_context

SCORE_EDGES = np.array([0., .1, .25, .5, .75, .9, .95, .99, 1.])
ROUTES = ('hard_transport', 'hard_add_only', 'hard_add_remove',
          'soft_transport', 'soft_add_only', 'soft_add_remove')


def gradient_groups(name):
    result = ['WM_all']
    if name.startswith('decoder.'):
        result.append('decoder')
    elif name.startswith('residual_head.'):
        result.append('XY_head')
    elif name.startswith('yaw_head.'):
        result.append('yaw_head')
    elif name.startswith('existence_head.'):
        result.append('existence_head')
    else:
        result.append('history_and_other_shared')
    return result


def compare_gradients(named, occupancy, motion):
    """Pre-clip, actual weighted gradients; zero norms have undefined cosine."""
    totals = defaultdict(lambda: np.zeros(3, np.float64))
    for (name, parameter), left, right in zip(named, occupancy, motion):
        a = torch.zeros_like(parameter, device='cpu') if left is None else left.detach().cpu()
        b = torch.zeros_like(parameter, device='cpu') if right is None else right.detach().cpu()
        a, b = a.double(), b.double()
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise RuntimeError('nonfinite WM diagnostic gradient: '+name)
        row = np.array([float((a*a).sum()), float((b*b).sum()), float((a*b).sum())])
        for group in gradient_groups(name): totals[group] += row
    result = {}
    for group, (aa, bb, ab) in totals.items():
        na, nb = float(np.sqrt(aa)), float(np.sqrt(bb))
        result[group] = dict(occupancy_norm=na, motion_norm=nb,
            cosine=float(np.clip(ab/(na*nb), -1, 1)) if na and nb else None,
            occupancy_to_motion_norm_ratio=na/nb if nb else None,
            dot=float(ab), pre_clip=True)
    return result


def gradient_batch(model, provider, rows, *, object_weight=1., lovasz_weight=1.):
    """One training-mode forward/window, all six full-grid losses, NO step.

    Matches the accepted sequential network/two-horizon execution. Render VJPs
    are returned to ALL live roots, including CCR's linked future queries;
    comparing only output XY gradients would miss the shared feature path.
    """
    model.train()
    named = [(n, p) for n, p in model.transport.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    accumulated = [[None]*len(named), [None]*len(named)]
    terms = defaultdict(float)
    for row in rows:
        fw = forward_window(model, provider, row, 'gt')
        geometry, existence = source_auxiliary(fw['output'], row[0], fw['prep'].state)
        direct = object_weight*(geometry+existence)/len(rows)
        if not bool(torch.isfinite(direct)): raise RuntimeError('nonfinite direct motion audit loss')
        motion = (torch.autograd.grad(direct, parameters, retain_graph=True, allow_unused=True)
                  if direct.requires_grad else [None]*len(named))
        scene, bridge = GroupedScene(fw['prep'], provider.pcfg.grid, provider.device), RenderGradientBridge(fw)
        labels = np.asarray(row[1]['future_gt_occ'])
        for first in (0, 2, 4):
            hs = [first, first+1]
            probability = scene.render_group(hs, **bridge.leaves, evidence=fw['evidence'], plan=fw['plan'])
            losses, ce, lovasz = grouped_loss(probability, labels[hs], lovasz_weight=lovasz_weight)
            if not bool(torch.isfinite(losses).all()): raise RuntimeError('nonfinite audit loss')
            terms['CE'] += float(ce.detach().sum())/(6*len(rows))
            terms['Lovasz'] += float(lovasz.detach().sum())/(6*len(rows))
            if losses.requires_grad: (losses/(6*len(rows))).sum().backward()
            del probability, losses, ce, lovasz
        roots = [(bridge.original[k], leaf.grad) for k, leaf in bridge.leaves.items()
                 if leaf.grad is not None and bridge.original[k].requires_grad]
        occ = (torch.autograd.grad([v for v, _ in roots], parameters,
               grad_outputs=[g for _, g in roots], allow_unused=True)
               if roots else [None]*len(named))
        for bucket, gradients in zip(accumulated, (occ, motion)):
            for i, value in enumerate(gradients):
                if value is not None:
                    value = value.detach().float().cpu()
                    bucket[i] = value.clone() if bucket[i] is None else bucket[i]+value
        terms['source_geometry'] += float(geometry.detach())/len(rows)
        terms['existence'] += float(existence.detach())/len(rows)
        del fw, geometry, existence, direct, scene, bridge, roots, occ, motion
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('read-only gradient audit unexpectedly accumulated parameter .grad')
    return dict(windows=len(rows), losses=dict(terms), groups=compare_gradients(named, *accumulated),
        scope='TRAIN; actual weighted full6 loss/batch normalization; training mode; pre-clip; no optimizer')


def edit_quality(before, after, target):
    before, after, target = (np.asarray(v) for v in (before, after, target))
    changed = before != after
    correct, was_correct = after == target, before == target
    return dict(changed=int(changed.sum()), corrected=int((changed & correct & ~was_correct).sum()),
        damaged=int((changed & was_correct & ~correct).sum()),
        wrong_to_wrong=int((changed & ~was_correct & ~correct).sum()),
        added=int((changed & (before == 17) & (after != 17)).sum()),
        removed_to_free=int((changed & (before != 17) & (after == 17)).sum()),
        relabeled_occupied=int((changed & (before != 17) & (after != 17)).sum()))


def action_funnel(fw, outputs, target, h, *, remove_threshold=.95):
    """GT labels audits only, never changes support/scores/thresholds.

    Histograms are evidence-ROW weighted (duplicates disclosed), whereas actual
    changed/corrected/damaged and legal coverage are UNIQUE VOXEL counts.
    """
    evidence, plan = fw['evidence'], fw['plan']
    scores = fw['actions'].detach().float().cpu().numpy()
    baseline, add, both, gt = [np.asarray(v).reshape(-1) for v in
        (outputs['transport'][h], outputs['add_only'][h], outputs['add_remove'][h], target)]
    result = dict(add_quality=edit_quality(baseline, add, gt),
        remove_incremental_quality=edit_quality(add, both, gt),
        add_errors_not_undoable_by_remove=int(((baseline == 17) & (add != 17) & (add != gt)).sum()),
        remove_changes_to_new_add_voxels=int(((add != baseline) & (both != add)).sum()),
        baseline_semantic_errors=int((baseline != gt).sum()), actions={})
    for index, name, threshold in ((0, 'ADD', .5), (1, 'REMOVE', remove_threshold)):
        rows = np.flatnonzero(plan.legal[:, h, index] & (plan.flat[:, h] >= 0))
        at = plan.flat[rows, h]
        destination = evidence.classes[rows] if index == 0 else plan.fallback[rows, h]
        prior, truth, score = baseline[at], gt[at], scores[rows, h, index]
        labels = dict(correcting=(prior != truth) & (destination == truth),
            damaging=(prior == truth) & (destination != truth),
            wrong_to_wrong=(prior != truth) & (destination != truth))
        groups = {}
        for role, use in (('all', np.ones(len(rows), bool)), ('static', evidence.actor[rows] < 0),
                          ('dynamic', evidence.actor[rows] >= 0)):
            groups[role] = dict(rows=int(use.sum()), unique_voxels=int(len(np.unique(at[use]))),
                accepted_rows=int((use & (score >= threshold)).sum()),
                accepted_unique_voxels=int(len(np.unique(at[use & (score >= threshold)]))),
                correctable_unique_voxels=int(len(np.unique(at[use & labels['correcting']]))),
                histogram={quality: np.histogram(score[use & mask], SCORE_EDGES)[0].tolist()
                           for quality, mask in labels.items()},
                mean_score={quality: float(score[use & mask].mean()) if (use & mask).any() else None
                            for quality, mask in labels.items()})
        result['actions'][name] = groups
    return result


@torch.no_grad()
def evaluation_window(model, provider, row, *, remove_threshold=.95):
    """Reuse ONE actual hard forward's WM/geometry; read soft actions separately.

    Three fixed soft/hard routes, not an oracle or threshold sweep. Continuous
    soft head context is the training context, not the rounded hard readout.
    """
    model.eval()
    fw = forward_window(model, provider, row, 'gt', soft=False)
    hard = hard_predictions(fw, remove_threshold=remove_threshold)
    mapped, context = continuous_context(fw['evidence'], fw['plan'], fw['prep'].state,
        fw['targets'], fw['yaws'], fw['poses'], provider.pcfg.grid)
    upload = lambda a: torch.as_tensor(np.array(a, copy=True), device=provider.device)
    evidence, plan = fw['evidence'], fw['plan']
    actor = upload(evidence.actor).long()
    with torch.autocast(device_type=provider.device.type, dtype=torch.bfloat16,
                        enabled=provider.device.type == 'cuda'):
        linked = model.columns.project_sources(fw['output'])
        encoded = model.columns.encode(upload(evidence.features), upload(evidence.labels), actor,
                                       upload(evidence.classes), linked)
        logits = model.columns.decode(encoded, actor, context, upload(plan.base), upload(plan.fallback),
                                      upload(plan.legal), linked)
    actions = model.columns.probabilities(logits, actor)
    difference = (actions-fw['actions'].to(provider.device)).abs()
    head_gap = dict(max_abs=float(difference.max()) if difference.numel() else 0.,
                   sum_abs=float(difference.sum()), elements=difference.numel())
    without_remove = torch.stack((actions[..., 0], torch.zeros_like(actions[..., 1])), -1)
    scene = GroupedScene(fw['prep'], provider.pcfg.grid, provider.device)
    predictions = {'hard_'+key: [value[h] for h in (1, 3, 5)] for key, value in hard.items()}
    for name, current, edits in (('soft_transport', actions, False),
                                ('soft_add_only', without_remove, True), ('soft_add_remove', actions, True)):
        predictions[name] = []
        # Score only report horizons; gradient audit above still supervises ALL6.
        for h in (1, 3, 5):
            p = scene.render_group([h], poses=fw['poses'], targets=fw['targets'], yaws=fw['yaws'],
                mapped=mapped, actions=current, evidence=evidence, plan=plan, edits=edits)
            predictions[name].append(p[0].argmax(0).reshape(scene.shape).byte().cpu().numpy())
            del p
    # Only NOW access future labels. No label decides the score or candidate.
    funnels = [action_funnel(fw, hard, row[1]['future_gt_occ'][h], h, remove_threshold=remove_threshold)
               for h in (1, 3, 5)]
    return predictions, funnels, head_gap


def merge_counts(left, right):
    """Sum nested integer counters/histograms; means are kept per-window."""
    for key, value in right.items():
        if key == 'mean_score': continue
        if isinstance(value, dict): merge_counts(left.setdefault(key, {}), value)
        elif isinstance(value, list): left[key] = (np.asarray(left.get(key, np.zeros(len(value))), np.int64)+value).tolist()
        elif isinstance(value, (int, np.integer)): left[key] = left.get(key, 0)+int(value)
    return left
