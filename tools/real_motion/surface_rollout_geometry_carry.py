"""Inference candidates, NOT a lossless backend: carry causal render geometry.

No source adapter, annotation, future mask, metric or GT argument is accepted.
Original model/renderer/cache files stay unchanged. Candidates change only the
second block; its motion encoder and all learned future poses stay authoritative.
"""
from dataclasses import replace
import numpy as np
import torch

from real_motion.geometry import relative_transform
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.strong_w2det import inverse_warp, majority_fill
from real_motion.source_evidence_audit import transform_points
from tools.real_motion.joint_surface_long_rollout_common import require_history_only

PROTOCOL = 'causal_static_background_and_predicted_se2_history_carry_v1'
ROUTES = ('baseline', 'static_carry', 'se2_carry', 'combined')


def static_mask(labels):
    a = np.asarray(labels)
    return (a != 17) & ~np.isin(a, DYNAMIC_CLASS_IDS)


def _warp_sequence(labels, source_pose, targets, grid, device):
    matrices = [relative_transform(source_pose, p) for p in targets]
    if torch.device(device).type == 'cuda':
        from real_motion.runtime_fastpath import inverse_warp_sequence_cuda_exact
        return inverse_warp_sequence_cuda_exact(labels, matrices, grid=grid, free_label=17, device=device)
    return [inverse_warp(labels, m, grid, 17) for m in matrices]


def direct_static_backgrounds(first, predictions, target_poses, provider):
    """Original t0 static -> future ONCE, plus first-block CCR static additions.

    Exactly the existing static semantic partition, nearest warp and Strong
    fill rule. Original t0 free/dynamic holes are NOT treated as static points.
    New CCR static labels are retained in their predicted 3s frame and only
    write into background-free voxels. No future visibility is synthesized.
    """
    require_history_only(first.raw)
    if len(predictions) != 6 or len(target_poses) != 6:
        raise ValueError('six first predictions and six target poses required')
    sem = np.asarray(first.raw['history_occ'][-1], np.uint8)
    original = np.full_like(sem, 17); original[static_mask(sem)] = sem[static_mask(sem)]
    rows = _warp_sequence(original, first.raw['history_poses'][-1], target_poses,
                          provider.pcfg.grid, provider.device)
    additions = np.full_like(sem, 17)
    final = np.asarray(predictions[-1])
    # First CCR is add-only. Carry exactly its static additions, not every
    # distorted/re-rasterized static voxel in the complete synthetic history.
    fresh = static_mask(final) & (np.asarray(first.baseline[-1]) == 17)
    additions[fresh] = final[fresh]
    novel_rows = _warp_sequence(additions, first.raw['future_poses'][-1], target_poses,
                                provider.pcfg.grid, provider.device) if fresh.any() else None
    result = []; audit = dict(initial_static_voxels=int(static_mask(original).sum()),
        carried_ccr_static_voxels=int(fresh.sum()), projected_ccr_static_voxels=0)
    for h, (warped, known) in enumerate(rows):
        if torch.device(provider.device).type == 'cuda':
            from real_motion.v18_execution_trial import majority_fill_native_exact
            out = majority_fill_native_exact(warped, ~known, kernel=provider.strong.fill_kernel,
                min_fraction=provider.strong.fill_min_fraction, device=provider.device)
        else:
            out = majority_fill(warped, ~known, kernel=provider.strong.fill_kernel,
                                min_fraction=provider.strong.fill_min_fraction)
        if novel_rows is not None:
            novel = novel_rows[h][0]; write = (out == 17) & static_mask(novel)
            out[write] = novel[write]; audit['projected_ccr_static_voxels'] += int(write.sum())
        if np.isin(out, DYNAMIC_CLASS_IDS).any():
            raise RuntimeError('dynamic class entered static carry')
        result.append(out)
    return result, audit


def apply_static_backgrounds(second, backgrounds):
    """Replace second background BEFORE CCR; moving foreground stays byte exact."""
    if any(len(a) != 6 for a in (backgrounds, second.baseline, second.owners, second.fallbacks)):
        raise ValueError('six complete static/background/ownership horizons required')
    baselines, fallbacks = [], []
    audit = dict(background_changed=0, background_added=0, background_removed=0,
                 background_relabelled=0, dynamic_foreground_changed=0)
    for old, own, fall, bg in zip(second.baseline, second.owners, second.fallbacks, backgrounds):
        old, own, fall, bg = map(np.asarray, (old, own, fall, bg))
        if old.shape != bg.shape or np.isin(bg, DYNAMIC_CLASS_IDS).any():
            raise ValueError('invalid static background shape/classes')
        protected = (own >= 0) | np.isin(old, DYNAMIC_CLASS_IDS)
        new = bg.copy(); new[protected] = old[protected]
        restored = fall.copy()
        # Preserve another dynamic layer under an overlapping dynamic source.
        replace_fallback = (own >= 0) & ~np.isin(fall, DYNAMIC_CLASS_IDS)
        restored[replace_fallback] = bg[replace_fallback]
        audit['background_changed'] += int((new != old).sum())
        audit['background_added'] += int(((old == 17) & (new != 17)).sum())
        audit['background_removed'] += int(((old != 17) & (new == 17)).sum())
        audit['background_relabelled'] += int(((old != 17) & (new != 17) & (old != new)).sum())
        if not np.array_equal(new[protected], old[protected]):
            raise RuntimeError('static carry changed dynamic foreground')
        baselines.append(new); fallbacks.append(restored)
    return replace(second, baseline=baselines, fallbacks=fallbacks), audit


def predicted_rigid_registrations(first, predictions, second, grid):
    """Known predicted SE(2) histories replace ICP only on unique owner matches.

    The existing center/velocity handoff is untouched. No yaw extrapolation or
    duplicate rotation of future shapes: these transforms align PAST predicted
    components to their predicted 3s pose. Splits/merges keep the original ICP.
    """
    if len(predictions) != 6 or len(first.owners) != 6 or len(first.yaws) != 6:
        raise ValueError('six predicted owner/pose horizons required')
    classes = np.asarray([c['class_id'] for c in first.state['current']], np.int64)
    centers = np.asarray(first.targets[-4:], np.float64).reshape(4, len(classes), 3).copy()
    angles = np.asarray(first.yaws[-4:], np.float64).reshape(4, len(classes))
    if not np.isfinite(centers).all() or not np.isfinite(angles).all():
        raise ValueError('nonfinite predicted rigid trajectory')
    centers[:, :, 2] = [c['centroid_world'][2] for c in first.state['current']]
    ownership = []
    for own, dense in zip(first.owners[-4:], predictions[-4:]):
        ids = np.asarray(own).copy(); valid = ids >= 0
        if np.any(ids[valid] >= len(classes)): raise ValueError('invalid owner identity')
        good = valid.copy(); good[valid] = np.asarray(dense)[valid] == classes[ids[valid]]
        ids[~good] = -1; ownership.append(ids)
    frames = second.state['components_by_frame'][-4:]
    if len(frames) != 4: raise ValueError('four predicted component histories required')
    registrations = [list(row) for row in second.registrations]
    pairs = second.state['motion_handoff_audit']['source_identity_pairs']
    audit = dict(unique_current_pairs=len(pairs), exact_se2_registrations=0,
                 ambiguous_or_missing_history_registrations=0, centroid_gate_fallbacks=0)
    origin = np.array([grid.x_min, grid.y_min, grid.z_min]); step = np.asarray(grid.voxel_size)
    history_candidates = []
    for f, frame in enumerate(frames):
        mapping = {}
        for k, comp in enumerate(frame):
            cells = np.asarray(comp['voxel_indices'], np.int64)
            ids = np.unique(ownership[f][tuple(cells.T)])
            ids = [int(j) for j in ids if j >= 0 and classes[j] == comp['class_id']]
            for j in ids: mapping.setdefault(j, []).append((k, ids))
        history_candidates.append(mapping)
    for current_i, original_j in pairs:
        if (not 0 <= current_i < len(registrations) or not 0 <= original_j < len(classes)
                or second.state['current'][current_i]['class_id'] != classes[original_j]):
            raise ValueError('rigid handoff identity mismatch')
        for f in range(3):
            matches = history_candidates[f].get(original_j, [])
            if len(matches) != 1 or len(matches[0][1]) != 1:
                audit['ambiguous_or_missing_history_registrations'] += 1; continue
            cells = np.asarray(frames[f][matches[0][0]]['voxel_indices'], np.int64)
            delta = angles[-1, original_j] - angles[f, original_j]
            c, s = np.cos(delta), np.sin(delta)
            matrix = np.eye(4); matrix[:2, :2] = ((c, -s), (s, c))
            matrix[:2, 3] = centers[-1, original_j, :2] - matrix[:2, :2] @ centers[f, original_j, :2]
            world = transform_points(origin+(cells+.5)*step, second.raw['history_poses'][f])
            aligned = transform_points(world, matrix)
            # Same fixed 4m safety gate as the existing center handoff; this
            # guards a tiny provenance island embedded in unrelated geometry.
            offset = aligned.mean(0)[:2] - np.asarray(second.state['current'][current_i]['centroid_world'])[:2]
            if np.linalg.norm(offset) > 4.:
                audit['centroid_gate_fallbacks'] += 1; continue
            registrations[current_i][f] = (matrix, cells.copy())
            audit['exact_se2_registrations'] += 1
    return replace(second, registrations=registrations), audit


def candidates(first, predictions, second, target_poses, provider):
    """Four isolated views; first block/state/model tensors never mutated."""
    backgrounds, static_audit = direct_static_backgrounds(first, predictions, target_poses, provider)
    stat, static_changes = apply_static_backgrounds(second, backgrounds)
    rigid, rigid_audit = predicted_rigid_registrations(first, predictions, second, provider.pcfg.grid)
    both = replace(stat, registrations=rigid.registrations)
    return dict(baseline=second, static_carry=stat, se2_carry=rigid, combined=both), {
        'static':{**static_audit, **static_changes}, 'se2':rigid_audit}
