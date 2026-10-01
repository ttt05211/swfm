"""Effect-safety tests, not claims of real nuScenes improvements."""
from __future__ import annotations
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.rigid_transport import RasterizedRigidComponent
from real_motion.causal_column_completion import (ColumnConfig, ColumnPlan, GENERATE, REFINE, KEEP, ADD, REMOVE,
    FREE, UNKNOWN, CONTEXT_DIM, PROTOCOL, FEATURE_PROTOCOL, action_targets, sample_queries, compose_dense,
    compose_sparse, sparse_layout, sparse_counts, actions_from_probabilities, acceptance_gate)
from real_motion.causal_column_model import CausalColumnModel, column_loss
from real_motion.source_evidence_audit import register_history_shape, transform_points
from tools.real_motion import causal_column_common as common
from tools.real_motion import train_p0_f9_causal_columns as trainer
from tools.real_motion.static_evidence_selector_common import finite_json, bank_fingerprint


def plan_fixture():
    # SAME future columns for generation and two ordered sources. Visible source
    # 1 is class4; deleting it restores source0 class5, not empty space.
    base = np.array([[4, FREE], [4, FREE], [4, FREE]], np.uint8)
    fallback = base.copy(); fallback[2, 0] = 5
    legal = np.zeros((3, 2, 3), bool); legal[..., KEEP] = True
    legal[:, 1, ADD] = True; legal[2, 0, REMOVE] = True
    return ColumnPlan(np.array([[0, 0]]*3), np.array([GENERATE, REFINE, REFINE]),
        np.array([-3, 0, 1]), np.array([11, 5, 4]), np.array([[0, 1]]*3),
        base, fallback, legal, np.zeros((3, CONTEXT_DIM), np.float32))


def scene_fixture():
    grid = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(16, 12, 2))
    hist = np.full((6, *grid.shape_hwd), FREE, np.uint8)
    hist[:, 7, :, 0] = 11; hist[:, 7, 6:, 0] = 13
    hist[:, 2, 3, 0] = 11
    hist[4, 2, 6, 0] = 4; hist[5, 4, 6, 0] = 4
    base = np.full(grid.shape_hwd, FREE, np.uint8)
    base[7, :, 0] = hist[5, 7, :, 0]
    base[2, 3, 1] = 11
    base[6, 6, 0] = 4
    truth = base.copy(); truth[2, 3, 1] = FREE; truth[2, 3, 0] = 11
    truth[6, 6, 0] = 13; truth[6, 7, 0] = 4
    truth[8, :, 0] = hist[5, 7, :, 0]
    memory = hist[5].copy(); memory[memory == 4] = FREE
    footprint = np.zeros(grid.shape_hwd[:2], bool); footprint[:8] = True
    own = np.full_like(base, -1, dtype=np.int32); own[6, 6, 0] = 0
    fall = base.copy(); fall[6, 6, 0] = 13
    comp = dict(class_id=4, centroid_world=np.array([4.5, 6.5, .5]), voxel_indices=np.array([[4, 6, 0]]))
    reg = np.eye(4); reg[0, 3] = 2
    registrations = [[None]*4+[(reg, np.array([[2, 6, 0]])), (np.eye(4), comp['voxel_indices'])]]
    state = dict(current=[comp], current_pose=np.eye(4), world_to_future=[np.eye(4)]*6)
    raw = dict(history_occ=hist, history_observed=np.ones_like(hist, bool), history_poses=[np.eye(4)]*6,
        future_poses=[np.eye(4)]*6, future_gt_occ=[truth.copy() for _ in range(6)])
    prep = common.PreparedColumns(SimpleNamespace(scene_name='dev', t0_token='t0', future_tokens=tuple('abcdef')), raw,
        state, [base.copy() for _ in range(6)], [own.copy() for _ in range(6)], [fall.copy() for _ in range(6)],
        [[RasterizedRigidComponent(4, np.array([[6, 6, 0]]), 1)]]*6,
        [[np.array([6.5, 6.5, .5])]]*6, [[0.]]*6, registrations,
        np.stack([footprint]*6), np.stack([memory]*6), {'registration_accepted': 1})
    return prep, grid, ColumnConfig(width=16, semantic_dim=4, z_bins=2)


def tensors(features):
    return {k: torch.as_tensor(np.asarray(v)) for k, v in features.items()}


@pytest.mark.parametrize('padding', (1, 2, 4))
def test_local_support_roi_matches_full_grid_dilation_and_order(padding):
    from scipy.ndimage import binary_dilation
    rng = np.random.default_rng(17)
    for shape in ((1, 1), (4, 7), (200, 200)):
        for _ in range(20):
            xy = np.column_stack([rng.integers(0, n, size=12) for n in shape])
            mask = np.zeros(shape, bool); mask[tuple(xy.T)] = True
            expected = np.argwhere(binary_dilation(mask, iterations=padding))
            assert np.array_equal(common.padded_support_xy(xy, shape, padding), expected)


def test_fixed_frontier_reuse_preserves_live_plans_features_and_gt_actions():
    prep, grid, cfg = scene_fixture()
    geometry = common.fixed_candidate_geometry(prep.memory, prep.footprints, grid, cfg)
    for alteration in (False, True):
        if alteration: prep.baseline[0][8, 6, 1] = 13
        original = common.candidate_plan(prep, 0, grid, cfg)
        prep.fixed_candidate_geometry = geometry
        cached = common.candidate_plan(prep, 0, grid, cfg)
        assert all(np.array_equal(v, getattr(cached, k)) for k, v in vars(original).items())
        assert all(np.array_equal(v, common.sample_column_features(prep, 0, cached, grid, cfg)[k])
            for k, v in common.sample_column_features(prep, 0, original, grid, cfg).items())
        historical = (prep.memory[0] == 11)|(prep.memory[0] == 13)
        from scipy.ndimage import binary_dilation
        xy = np.argwhere(historical.any(2))
        old = binary_dilation(historical[tuple(xy.T)], structure=np.ones((1, 3)), iterations=1)
        assert np.array_equal(old, geometry[0]['static_allowed'][tuple(xy.T)])
        prep.fixed_candidate_geometry = None


def test_remove_labels_actual_restored_class_not_any_gt_mismatch():
    plan = plan_fixture()
    for gt, expected in ((5, REMOVE), (4, KEEP), (11, KEEP), (FREE, KEEP)):
        targets = action_targets(plan, np.array([gt, 5]))
        assert targets[2, 0] == expected
        assert targets[1, 1] == ADD and targets[2, 1] == KEEP
        assert targets[0, 1] == KEEP  # road generation cannot relabel a car GT


def test_last_source_wins_remove_restores_lower_source_and_generation_free_only():
    plan = plan_fixture(); dense = np.array([[[4, FREE]]], np.uint8)
    acts = np.full((3, 2), KEEP); acts[:, 1] = ADD; acts[2, 0] = REMOVE
    joint = compose_dense(dense, plan, acts)
    assert joint.tolist() == [[[5, 4]]]
    assert compose_dense(dense, plan, acts, enable_refine=False).tolist() == [[[4, 11]]]
    assert compose_dense(dense, plan, acts, enable_generation=False).tolist() == [[[5, 4]]]
    assert dense.tolist() == [[[4, FREE]]]
    bad = acts.copy(); bad[1, 0] = REMOVE
    with pytest.raises(ValueError, match='illegal'): compose_dense(dense, plan, bad)
    with pytest.raises(ValueError, match='different plan'):
        compose_sparse(plan.subset(slice(None)), acts, layout=sparse_layout(plan))
    with pytest.raises(ValueError, match='baseline mismatch'):
        compose_dense(np.array([[[13, FREE]]]), plan, acts)


@pytest.mark.parametrize('seed', range(10))
def test_layered_sparse_matches_independent_full_source_recomposition(seed):
    rng = np.random.default_rng(seed); shape = (5, 4, 2)
    bg = np.where(rng.random(shape) < .2, 11, FREE).astype(np.uint8)
    components = []
    for c in (4, 5, 4):
        indices = np.argwhere(rng.random(shape) < .35)
        components.append(RasterizedRigidComponent(c, indices, len(indices)))
    base, owner, fall = common.component_layers(bg, components)
    rows, targets = [], []
    for i, comp in enumerate(components):
        xy = np.argwhere(np.ones(shape[:2], bool)); flat = ((xy[:, :1]*shape[1]+xy[:, 1:])*shape[2]+np.arange(shape[2]))
        b = base.reshape(-1)[flat]; f = b.copy(); visible = owner.reshape(-1)[flat] == i
        f[visible] = fall.reshape(-1)[flat][visible]
        legal = np.zeros((*b.shape, 3), bool); legal[..., KEEP] = True
        legal[..., ADD] = b == FREE
        legal[..., REMOVE] = visible & (b == comp.class_id) & (f != b)
        rows.append(ColumnPlan(xy, np.full(len(xy), REFINE), np.full(len(xy), i), np.full(len(xy), comp.class_id),
            flat, b, f, legal, np.zeros((len(xy), CONTEXT_DIM))))
        act = np.zeros_like(b, np.int64)
        act[legal[..., REMOVE] & (rng.random(b.shape) < .4)] = REMOVE
        act[legal[..., ADD] & (rng.random(b.shape) < .2)] = ADD
        targets.append(act)
    plan = ColumnPlan(**{k: np.concatenate([getattr(r, k) for r in rows]) for k in vars(rows[0])})
    action = np.concatenate(targets)
    edited = []
    for comp, row, act in zip(components, rows, targets):
        source = np.zeros(shape, bool); source[tuple(comp.voxel_indices.T)] = True
        source.reshape(-1)[row.flat[act == REMOVE]] = False
        source.reshape(-1)[row.flat[act == ADD]] = True
        edited.append(RasterizedRigidComponent(comp.class_id, np.argwhere(source), int(source.sum())))
    expected = bg.copy()
    for comp in edited: expected[tuple(comp.voxel_indices.T)] = comp.class_id
    actual = compose_dense(base, plan, action)
    assert np.array_equal(actual, expected)
    gt = rng.integers(0, 18, size=shape, dtype=np.uint8); moving = rng.random(shape) < .5
    ids, before, after = compose_sparse(plan, action)
    sparse = sparse_counts(common.Metrics.counts(base, gt, moving, FREE), before, after,
                           gt.reshape(-1)[ids], moving.reshape(-1)[ids], common.DYN)
    assert all(np.array_equal(a, b) for a, b in zip(sparse, common.Metrics.counts(actual, gt, moving, FREE)))


def test_candidates_real_grid_entry_ownership_no_future_gt_and_no_ghost_history():
    prep, grid, cfg = scene_fixture(); plan = common.candidate_plan(prep, 5, grid, cfg)
    g = plan.kind == GENERATE
    assert len(plan) and g.any() and (plan.xy[g, 0] >= 8).all()
    # Unobserved cells INSIDE visited grid are not GRID_ENTRY.
    prep.raw['history_observed'][:] = False
    assert np.array_equal(common.candidate_plan(prep, 5, grid, cfg).xy, plan.xy)
    assert not plan.legal[g, :, REMOVE].any()
    # Generation must see an actual frontier-anchor full-Z patch, even when its
    # output query is further outside history than the patch half-width.
    far = np.flatnonzero(g & (plan.xy[:, 0] >= 10))
    assert len(far)
    anchored = common.sample_column_features(prep, 5, plan.subset(far[:1]), grid, cfg)
    assert np.isin(anchored['history'][0, :, 3, 3, 0], (11, 13)).all()
    assert (anchored['flags'][0, :, 3, 3, 0] & 2).all()
    dyn = plan.actor == 0
    assert dyn.any() and plan.legal[dyn, :, REMOVE].sum() == 1
    ix = np.flatnonzero(dyn & (plan.xy[:, 0] == 6) & (plan.xy[:, 1] == 6))[0]
    assert plan.fallback[ix, 0] == 13
    features = common.sample_column_features(prep, 5, plan.subset([ix]), grid, cfg)
    # Query centre6 -> t0 source4 -> history source2, not an ego-only ghost.
    assert features['history'][0, 4, 3, 3, 0] == features['history'][0, 5, 3, 3, 0] == 4
    assert (features['flags'][0, 4:6, 3, 3, 0] & 2).all()
    assert (features['history'][0, :4] == UNKNOWN).all()
    other = copy.deepcopy(prep); other.raw['future_gt_occ'] = [np.zeros(grid.shape_hwd, np.uint8)]*6
    other.raw['future_annotations'] = {'malicious': 'never used'}
    other_plan = common.candidate_plan(other, 5, grid, cfg)
    assert all(np.array_equal(v, getattr(other_plan, k)) for k, v in vars(plan).items())
    other_features = common.sample_column_features(other, 5, other_plan.subset([ix]), grid, cfg)
    assert all(np.array_equal(v, other_features[k]) for k, v in features.items())
    changed_gen = common.sample_column_features(other, 5, other_plan.subset(far[:1]), grid, cfg)
    assert all(np.array_equal(v, changed_gen[k]) for k, v in anchored.items())
    prep.registrations[0][:5] = [None]*5
    assert not (common.candidate_plan(prep, 5, grid, cfg).actor >= 0).any()


def test_real_causal_registration_matrix_accounts_for_sorted_points_and_preserves_world_z():
    # Actual ICP, not a mocked accepted matrix. Reverse order to catch translation
    # being recovered from mismatched sorted original/result points.
    from real_motion.rigid_transport import rigid_source_points_world
    grid = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(.4, .4, .4), shape_hwd=(30, 30, 4))
    indices = np.array([[x, y, 1] for x in range(4, 8) for y in range(5, 8)])
    past = dict(class_id=4, voxel_indices=indices[::-1].copy(), centroid_world=np.array([2.4, 2.6, .6]))
    current = {**past, 'voxel_indices': indices+np.array([3, 0, 0]), 'centroid_world': np.array([3.6, 2.6, .6])}
    reference = rigid_source_points_world(current['voxel_indices'], np.eye(4), grid=grid)
    state = dict(current=[current], velocities=np.array([[2.4, 0, 0]]), source_world_points=[reference])
    with patch.object(common, 'extract_instances_cropped_exact', return_value=[past]):
        regs, pts, _, audit = common.causal_source_history(np.zeros((6, *grid.shape_hwd)), [np.eye(4)]*6, state, grid, None, 2)
    assert audit['registration_accepted'] > 0
    r = regs[0][4][0]; transformed = transform_points(pts[4][0], r)
    result = register_history_shape(pts[4][0], reference)
    assert np.allclose(np.sort(transformed, axis=0), np.sort(result.points, axis=0))
    assert np.array_equal(transformed[:, 2], pts[4][0][:, 2])
    motion = common.pose_motion(np.array([0, 0, 4]), np.array([1, 2, 40]), np.pi/2)
    assert transform_points(np.array([[1., 0, 3]]), motion).tolist() == [[1., 3., 3.]]


def test_train_sampling_importance_recovers_all_three_populations():
    prep, grid, cfg = scene_fixture(); plan = common.candidate_plan(prep, 1, grid, cfg)
    target = action_targets(plan, prep.raw['future_gt_occ'][1])
    ids, weights = sample_queries(plan, target, 4, np.random.default_rng(9))
    groups = (plan.kind == GENERATE, (plan.kind == REFINE)&(plan.actor < 0), plan.actor >= 0)
    for group in groups:
        assert group[ids].any()
        assert weights[group[ids]].sum() == group.sum()
        for positive in (True, False):
            bucket = group & ((target != KEEP).any(1) == positive)
            assert weights[bucket[ids]].sum() == bucket.sum()


def test_actual_frozen_renderer_layered_preparation_and_empty_source_window():
    from real_motion.strong_w2det import StrongW2DetConfig
    prep, grid, cfg = scene_fixture()
    source = prep.state['current'][0]; indices = source['voxel_indices']
    rec = dict(anchors_xy_t0_m=torch.tensor([[[4.5, 6.5]]*6]))
    anchor = prep.raw['history_occ'][5].copy()
    prior = RasterizedRigidComponent(4, indices, len(indices))
    state = {**prep.state, 'rec': rec, 'future_poses': prep.raw['future_poses'],
        'anchors': [anchor.copy() for _ in range(6)], 'baseline_by_hi': [[prior]]*6,
        'baseline_clear_flat_by_hi': [np.ravel_multi_index(indices.T, grid.shape_hwd)]*6,
        'source_world_points': [np.array([[4.5, 6.5, .5]])], 'source_rel_xy': [np.zeros((1, 2))],
        'source_z_t0': np.array([.5]), 'velocities': {0: np.array([4., 0, 0])}, 'gpu': None}
    outputs = dict(residual_xy_m=torch.tensor([[[2., 0.]]*6]), yaw_delta_rad=torch.zeros(1, 6))
    provider = common.FrozenColumns.__new__(common.FrozenColumns)
    provider.pcfg = SimpleNamespace(grid=grid, free_label=FREE, frame_dt_s=.5)
    provider.device = torch.device('cpu'); provider.workers = 2; provider.strong = StrongW2DetConfig(); provider.model = None
    # Only the expensive pretrained forward/nuScenes loader is replaced. Actual
    # renderer, CLEAR/WRITE compositor, static memory, source association and
    # footprint construction run, including comparison to original _forecast_once.
    with patch.object(common.FrozenXYV18, 'prepare', return_value=(prep.window, prep.raw, state, outputs)), \
        patch.object(common.runtime, '_stage_gpu_inputs'), patch.object(common.runtime, '_release_gpu_inputs'), \
        patch.object(common.runtime, '_exactness_check'):
        result = provider.prepare_columns(None, rec, include_gt=True)
    expected = common.runtime._forecast_once(None, state, provider.pcfg, provider.strong, provider.device, precomputed_out=outputs)
    assert all(np.array_equal(a, b) for a, b in zip(expected, result.baseline))
    assert result.baseline[0][4, 6, 0] == FREE and result.baseline[0][6, 6, 0] == 4
    assert result.owners[0][6, 6, 0] == 0 and result.fallbacks[0][6, 6, 0] == FREE
    assert provider.columns_checked
    # Windows with zero t0 sources still generate/static-refine; no fake source
    # or torch-cat(empty) crash, and all six baseline predictions are preserved.
    state = {**state, 'current': [], 'velocities': {}, 'source_world_points': [], 'source_rel_xy': [],
        'source_z_t0': np.empty(0), 'baseline_by_hi': [[]]*6, 'baseline_clear_flat_by_hi': [np.empty(0, np.int64)]*6,
        'rec': dict(anchors_xy_t0_m=torch.empty(0, 6, 2))}
    outputs = dict(residual_xy_m=torch.empty(0, 6, 2), yaw_delta_rad=torch.empty(0, 6))
    provider.columns_checked = False
    with patch.object(common.FrozenXYV18, 'prepare', return_value=(prep.window, prep.raw, state, outputs)), \
        patch.object(common.runtime, '_stage_gpu_inputs'), patch.object(common.runtime, '_release_gpu_inputs'), \
        patch.object(common.runtime, '_exactness_check'):
        result = provider.prepare_columns(None, state['rec'], include_gt=False)
    assert result.registrations == [] and len(result.baseline) == 6
    assert not (common.candidate_plan(result, 1, grid, cfg).actor >= 0).any()


def test_prior_correction_legal_softmax_and_disabled_gate_even_when_probability_is_one():
    plan = plan_fixture(); model = CausalColumnModel(ColumnConfig(width=16, semantic_dim=4, z_bins=2))
    model.generation_pos_weight.fill_(10); model.refine_class_weights.copy_(torch.tensor([1., 10, 20]))
    probability = model.calibrated_probabilities(torch.full((3, 2), np.log(10)),
        torch.log(torch.tensor([1., 10, 20])).repeat(3, 2, 1), torch.as_tensor(plan.kind), torch.as_tensor(plan.legal)).numpy()
    assert np.allclose(probability[:, 1], [.5, .5, 0])
    assert np.allclose(probability[2, 0], [.5, 0, .5])
    exact = np.zeros((3, 2, 3), np.float32); exact[..., ADD] = 1; exact[2, 0] = [0, 0, 1]
    assert not actions_from_probabilities(plan, exact, (None, None, None)).any()
    assert actions_from_probabilities(plan, exact, (.99, .99, .99))[2, 0] == REMOVE
    with pytest.raises(ValueError, match='probabilities'): actions_from_probabilities(plan, exact*.5, (.9,)*3)
    model.generation_pos_weight.fill_(float('nan'))
    with pytest.raises(RuntimeError, match='nonfinite'): model.calibrated_probabilities(torch.zeros(3, 2), torch.zeros(3, 2, 3),
        torch.as_tensor(plan.kind), torch.as_tensor(plan.legal))


def test_shared_network_learns_both_heads_full_z_and_illegal_gradients_are_zero():
    torch.manual_seed(21); prep, grid, cfg = scene_fixture()
    plan = common.candidate_plan(prep, 1, grid, cfg)
    target = action_targets(plan, prep.raw['future_gt_occ'][1])
    # Small balanced synthetic optimization with actual causal feature sampler.
    ids = np.r_[np.flatnonzero(plan.kind == GENERATE)[:4],
                np.flatnonzero((plan.actor == -2)), np.flatnonzero(plan.actor >= 0)]
    small = plan.subset(ids); features = tensors(common.sample_column_features(prep, 1, small, grid, cfg))
    legal, y = torch.as_tensor(small.legal), torch.as_tensor(target[ids]); kind = features['kind']
    model = CausalColumnModel(cfg); model.generation_pos_weight.fill_(5)
    model.refine_class_weights.copy_(torch.tensor([1., 5, 5]))
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    losses = []
    for step in range(35):
        optimizer.zero_grad(); g, r = model(**features); g.retain_grad(); r.retain_grad()
        loss, stats = column_loss(model, g, r, kind, legal, y, torch.ones(len(ids)))
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        assert (g.grad[~legal[..., ADD]] == 0).all()
        assert (r.grad[~legal] == 0).all()
        if step == 2:
            assert model.query.weight.grad.abs().sum() > 0
            assert model.column.weight.grad.abs().sum() > 0
            assert model.decoder[0].attention.in_proj_weight.grad.abs().sum() > 0
            assert model.generation.weight.grad.abs().sum() > 0
            assert model.refinement.weight.grad.abs().sum() > 0
    assert losses[-1] < losses[0]*.5
    g, r = model(**features); p = model.calibrated_probabilities(g, r, kind, legal).detach().numpy()
    learned = actions_from_probabilities(small, p, (.5, .5, .5))
    assert (learned[y.numpy() == ADD] == ADD).any()
    assert (learned[y.numpy() == REMOVE] == REMOVE).any()
    # Missing source frames/out-of-grid padding are not free evidence or NaNs.
    unknown = {**features, 'history': torch.full_like(features['history'], UNKNOWN), 'flags': torch.zeros_like(features['flags'])}
    assert all(torch.isfinite(v).all() for v in model(**unknown))
    swapped = {**features, 'history': features['history'].flip(-1), 'flags': features['flags'].flip(-1)}
    assert not torch.allclose(model(**features)[0], model(**swapped)[0])


def test_noop_not_success_nor_single_horizon_or_moving_degradation():
    zero = dict(IoU=0., mIoU=0., MovingMicro=0., MovingMacro=0., per_horizon={str(h): dict(IoU=0., mIoU=0., MovingMicro=0., MovingMacro=0.) for h in (1, 2, 3)})
    reports = {k: dict(delta_vs_v18_pp=copy.deepcopy(zero), quality={'added_semantic_tp': 0, 'corrected': 0}) for k in common.VARIANTS}
    assert not acceptance_gate(reports)['pass']
    for r in reports.values():
        r['delta_vs_v18_pp']['mIoU'] = .01; r['quality'] = dict(added_semantic_tp=1, corrected=1)
    assert acceptance_gate(reports)['pass']
    reports['joint']['delta_vs_v18_pp']['per_horizon']['3']['MovingMicro'] = -.00001
    assert not acceptance_gate(reports)['pass']
    reports['joint']['delta_vs_v18_pp']['mIoU'] = None
    assert not acceptance_gate(reports)['pass']


def fake_provider():
    prep, grid, cfg = scene_fixture(); calls = []
    def prepare(source, record, include_gt):
        calls.append((record['scene_name'], record['t0_token'], include_gt))
        result = copy.deepcopy(prep)
        result.window = SimpleNamespace(scene_name=record['scene_name'], t0_token=record['t0_token'], future_tokens=tuple('abcdef'))
        if not include_gt: result.raw.pop('future_gt_occ')
        return result
    return SimpleNamespace(prepare_columns=prepare, pcfg=SimpleNamespace(grid=grid), device=torch.device('cpu'), workers=1,
        sha='a'*64), prep, cfg, calls


def moving_fixture(prep): return [(np.ones_like(prep.baseline[0], bool), None)]*6


def test_train_bank_full_population_weights_not_positive_sample_counts_and_budget_guard():
    provider, prep, cfg, calls = fake_provider(); records = [dict(scene_name='train', t0_token='a')]
    bank, meta = trainer.prepare_bank(provider, None, records, cfg, 7, 8)
    expected = np.zeros(2); ref = np.zeros(3)
    for h in range(6):
        plan = common.candidate_plan(prep, h, provider.pcfg.grid, cfg); y = action_targets(plan, prep.raw['future_gt_occ'][h])
        valid = (plan.kind == GENERATE)[:, None] & plan.legal[..., ADD]
        expected += np.bincount((y[valid] == ADD).astype(int), minlength=2)
        valid = (plan.kind == REFINE)[:, None] & plan.legal[..., 1:].any(-1)
        ref += np.bincount(y[valid], minlength=3)
    assert np.array_equal(meta['full_unsampled_generation_counts'], expected)
    assert np.array_equal(meta['full_unsampled_refine_action_counts'], ref)
    assert expected.sum() > (bank['kind'] == GENERATE).sum()*cfg.z_bins
    assert all(bank[k].dtype == np.uint8 for k in ('history', 'flags', 'target'))
    with pytest.raises(RuntimeError, match='RAM budget'): trainer.prepare_bank(provider, None, records, cfg, 7, 0)


def test_four_way_evaluation_shared_prepare_subset_equals_separate_and_causal_deployment():
    provider, prep, cfg, calls = fake_provider(); model = CausalColumnModel(cfg)
    rows = [dict(scene_name=f'dev{i%2}', t0_token=str(i)) for i in range(4)]
    subset = [(r['scene_name'], r['t0_token']) for r in rows[:2]]
    with patch.object(common, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        combined = common.evaluate_columns(provider, SimpleNamespace(nusc=None), rows, model, (None,)*3, dev64_keys=subset)
        assert len(calls) == 4
        separate = common.evaluate_columns(provider, SimpleNamespace(nusc=None), rows[:2], model, (None,)*3)
    assert finite_json(combined['dev64']) == finite_json(separate['all'])
    assert len(combined['all']['variants']) == 6 and not combined['all']['gate']['pass']
    _, pred = common.forecast_columns(provider, None, rows[0], model, (None,)*3)
    assert calls[-1][-1] is False and all(np.array_equal(a, b) for a, b in zip(pred, prep.baseline))
    with patch.object(common, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        with pytest.raises(RuntimeError, match='incomplete'): common.evaluate_columns(provider, SimpleNamespace(nusc=None), rows,
            model, (None,)*3, dev64_keys=[('absent', 'missing')])


def test_train_only_calibration_selects_actual_safe_composed_metrics_not_noop():
    provider, prep, cfg, calls = fake_provider(); model = CausalColumnModel(cfg)
    def synthetic_scores(model, prepared, h, plan, *args):
        # An explicit synthetic oracle stub tests calibration plumbing, never
        # used in any production feature/forward path or real gain claim.
        y = action_targets(plan, prepared.raw['future_gt_occ'][h])
        p = np.zeros((*y.shape, 3), np.float32)
        np.put_along_axis(p, y[..., None], 1, axis=-1)
        return p
    with patch.object(common, 'gt_moving_support_sequence', return_value=moving_fixture(prep)), \
        patch.object(common, 'predict_probabilities', side_effect=synthetic_scores) as predictor:
        gates, report = common.calibrate_columns(provider, SimpleNamespace(nusc=None),
            [dict(scene_name='held_train', t0_token='c')], model)
        assert len(calls) == 1 and predictor.call_count == 3  # not 84 renders/forwards
        assert report['population'] == 'held_out_TRAIN_scenes_not_dev'
        assert report['both_branches_have_safe_correct_edits'] and all(t is not None for t in gates)
        assert len(report['candidates']) == 64
        evaluation = common.evaluate_columns(provider, SimpleNamespace(nusc=None),
            [dict(scene_name='dev', t0_token='d')], model, gates)
    assert evaluation['all']['gate']['pass']
    assert evaluation['all']['variants']['refine']['quality']['source_layer_REMOVE_decisions'] > 0


def test_checkpoint_fail_closed_roles_prior_weights_identity_nonfinite_and_reload(tmp_path):
    model = CausalColumnModel(ColumnConfig(width=16, semantic_dim=4, z_bins=2))
    ck = dict(protocol=PROTOCOL, feature_protocol=FEATURE_PROTOCOL, training_contract=trainer.CONTRACT,
        base_checkpoint_sha256='a', runtime_config_fingerprint='b', model_config=asdict(model.config),
        state_dict=model.state_dict(), checkpoint_role='calibrated_candidate', mode='screen', screen_pass=True,
        successful_updates=1, thresholds=(.99, .99, .99), TRAIN_weights={'generation_pos_weight': 1., 'refine_class_weights': [1., 1., 1.]})
    path = tmp_path/'candidate.pt'; torch.save(ck, path)
    _, loaded = trainer.load_columns(path, 'cpu', base_sha='a', config_sha='b')
    assert not any(p.requires_grad for p in loaded.parameters())
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    for change in ({'screen_pass': False}, {'mode': 'smoke'}, {'successful_updates': 0}, {'checkpoint_role': 'resume_last'},
                   {'thresholds': (None,)*3}, {'thresholds': (.1, .1, .1)}, {'base_checkpoint_sha256': 'x'},
                   {'TRAIN_weights': {'generation_pos_weight': 2., 'refine_class_weights': [1., 1., 1.]}}):
        torch.save({**ck, **change}, path)
        with pytest.raises(RuntimeError): trainer.load_columns(path, 'cpu', base_sha='a', config_sha='b')
    bad = copy.deepcopy(ck); bad['state_dict']['generation.weight'][0, 0] = float('nan'); torch.save(bad, path)
    with pytest.raises(RuntimeError, match='nonfinite'): trainer.load_columns(path, 'cpu', base_sha='a', config_sha='b')


def test_optimizer_sampling_resume_matches_next_actual_update(tmp_path):
    provider, _, cfg, _ = fake_provider()
    bank, w = trainer.prepare_bank(provider, None, [dict(scene_name='train', t0_token='a')], cfg, 7, 8)
    torch.manual_seed(19); model = CausalColumnModel(cfg)
    model.generation_pos_weight.fill_(w['generation_pos_weight'])
    model.refine_class_weights.copy_(torch.tensor(w['refine_class_weights']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = np.random.default_rng(4)
    trainer.train_step(model, optimizer, bank, rng, torch.device('cpu'), 8)
    ck = dict(protocol=PROTOCOL, feature_protocol=FEATURE_PROTOCOL, training_contract=trainer.CONTRACT,
        base_checkpoint_sha256='a', runtime_config_fingerprint='b', model_config=asdict(cfg),
        state_dict=copy.deepcopy(model.state_dict()), checkpoint_role='resume_last', TRAIN_weights=w,
        optimizer=copy.deepcopy(optimizer.state_dict()), sampling_rng_state=copy.deepcopy(rng.bit_generator.state))
    path = tmp_path/'last.pt'; torch.save(ck, path)
    ck, resumed = trainer.load_columns(path, 'cpu', base_sha='a', config_sha='b', allow_diagnostic=True)
    resumed.requires_grad_(True)
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=3e-4, weight_decay=.01); opt2.load_state_dict(ck['optimizer'])
    rng2 = np.random.default_rng(); rng2.bit_generator.state = ck['sampling_rng_state']
    a = trainer.train_step(model, optimizer, bank, rng, torch.device('cpu'), 8)
    b = trainer.train_step(resumed, opt2, bank, rng2, torch.device('cpu'), 8)
    assert a == b
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in model.state_dict().items())


def test_full_cli_synthetic_smoke_resume_freezes_baseline_calibration_and_summary(tmp_path):
    provider, prep, _, calls = fake_provider()
    train = [dict(scene_name=f'train{s}', t0_token=f'{s}:{i}') for s in range(10) for i in range(4)]
    dev = [dict(scene_name='dev', t0_token=f'd{i}') for i in range(64)]
    files = {k: tmp_path/k for k in ('train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint', 'train-info', 'dev-info')}
    for f in files.values(): f.write_bytes(b'original')
    manifest = dict(parent_keys=[('dev', f'd{i}') for i in range(512)], selected_key_fingerprint=trainer.DEV64_FP,
        manifest_fingerprint='b'*64)
    keys = [(r['scene_name'], r['t0_token']) for r in dev]
    config = Path(__file__).resolve().parents[1]/'configs/real_motion_occfm.yaml'
    raw_before = copy.deepcopy(prep.raw); originals = {k: f.read_bytes() for k, f in files.items()}
    def run(out, resume=None):
        argv = ['train', '--config', str(config), '--dataroot', str(tmp_path), '--out-dir', str(out), '--device', 'cpu',
            '--mode', 'smoke', '--batch-size', '4', '--cpu-workers', '1']
        for k, f in files.items(): argv += ['--'+k, str(f)]
        if resume: argv += ['--resume', str(resume)]
        with patch('sys.argv', argv), patch.object(trainer, 'make_prepare_config', return_value=provider.pcfg), \
            patch.object(trainer, 'load_manifest', return_value=(manifest, keys, None)), \
            patch.object(trainer, 'load_cache', side_effect=[({}, train), ({}, dev)]), \
            patch.object(trainer, 'FrozenColumns', return_value=provider), \
            patch.object(trainer, 'NuScenesWindowSource', return_value=SimpleNamespace(nusc=None)), \
            patch.object(common, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
            trainer.main()
    out = tmp_path/'run'; run(out)
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert summary['successful_updates'] == 2 and summary['train_windows'] == 4
    assert summary['calibration_windows'] == 2 and summary['evaluation']['all']['windows'] == 2
    assert len(summary['evaluation']['all']['variants']) == 6
    assert summary['route'] == 'smoke_only_not_effectiveness_evidence' and not summary['screen_pass']
    assert len(list(out.glob('*.pt'))) == 2
    assert {k: f.read_bytes() for k, f in files.items()} == originals
    assert all(np.array_equal(prep.raw[k], raw_before[k]) for k in ('history_occ', 'history_observed', 'future_gt_occ'))
    ck = torch.load(out/'candidate.pt', weights_only=False)
    assert ck['thresholds'] == tuple(summary['thresholds'])
    assert not set(s for s, _ in ck['train_keys']) & set(s for s, _ in ck['calibration_keys'])
    assert not set(s for s, _ in ck['train_keys']) & {'dev'}
    with pytest.raises(RuntimeError, match='cannot be deployed'):
        trainer.load_columns(out/'candidate.pt', 'cpu', base_sha=provider.sha, config_sha=ck['runtime_config_fingerprint'])
    # Final last.pt resumes without extra updates and reproduces fixed thresholds
    # and all ablations. A candidate.pt is never a training-resume checkpoint.
    resumed = tmp_path/'resume'; run(resumed, out/'last.pt')
    resumed_summary = json.loads((resumed/'summary.json').read_text(encoding='utf-8'))
    assert resumed_summary['evaluation'] == summary['evaluation']
    assert resumed_summary['thresholds'] == summary['thresholds']
    with pytest.raises(RuntimeError, match='resume population'): run(tmp_path/'bad-resume', out/'candidate.pt')
    # Read-only expanded entrypoint: original artifact and frozen gate unchanged;
    # explicit diagnostic permission cannot promote failed/smoke candidates.
    from tools.real_motion import eval_p0_f9_causal_columns as expanded
    digest = expanded.sha256(out/'candidate.pt')
    argv = ['eval', '--config', str(config), '--checkpoint', str(out/'candidate.pt'), '--dataroot', str(tmp_path),
        '--dev-cache', str(files['dev-cache']), '--population-manifest', str(files['population-manifest']),
        '--base-checkpoint', str(files['base-checkpoint']), '--dev-info', str(files['dev-info']),
        '--out-dir', str(tmp_path/'expanded'), '--device', 'cpu', '--population', 'dev64', '--allow-diagnostic']
    calls.clear()
    with patch('sys.argv', argv), patch.object(expanded, 'CLEAN_SHA256', provider.sha), \
        patch.object(expanded, 'make_prepare_config', return_value=provider.pcfg), \
        patch.object(expanded, 'load_manifest', return_value=(manifest, keys, None)), \
        patch.object(expanded, 'load_cache', return_value=({}, dev)), patch.object(expanded, 'FrozenColumns', return_value=provider), \
        patch.object(expanded, 'NuScenesWindowSource', return_value=SimpleNamespace(nusc=None)), \
        patch.object(common, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        expanded.main()
    result = json.loads((tmp_path/'expanded/evaluation.json').read_text(encoding='utf-8'))
    assert result['reports']['all']['windows'] == len(calls) == 64
    assert not result['original_screen_pass_unchanged']
    assert result['thresholds_from_original_TRAIN'] == summary['thresholds']
    assert expanded.sha256(out/'candidate.pt') == digest
    assert not list((tmp_path/'expanded').glob('*.pt'))
