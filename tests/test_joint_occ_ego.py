"""Small actual forward/backward tests, not fabricated nuScenes quality claims."""
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from real_motion.canonical_causal_repair import (build_compact_canonical_support, build_canonical_evidence,
                                                map_canonical_evidence, compose_canonical)
from real_motion.surface_canonical_repair import SurfaceAtlas, augment_evidence, augment_projection
from real_motion.ego_navigation import poses_from_se2, ego_history_features
from tools.joint_occ_experiments.model import (JointOccEgo, HistoryEgoHead, torch_poses, gradient_scale,
                                              ego_geometry_loss, source_auxiliary)
from tools.joint_occ_experiments.soft_geometry import (SoftScene, occupancy_loss, lovasz_softmax,
    projected_points, source_targets, effective_yaw, continuous_context, splat, sparse_splat, sparse_layer)
from tools.joint_occ_experiments.data import history_only_payload, HistoryInputs, prepare_projection
from tools.joint_occ_experiments.engine import forward_window, train_arm, hard_predictions, evaluate
from tools.joint_occ_experiments.run import checkpoint, restore_checkpoint, paired_update, parser, rng_state, restore_rng
from test_canonical_causal_repair import scene
from test_joint_causal_columns import fixture as old_fixture
from test_height_field_screen import assert_nested_equal
from test_surface_canonical_repair import assert_geometry_equal


def fixture():
    torch.set_num_threads(1)
    _, _, old, _, record = old_fixture()
    config = replace(old.transport.v17_config, history_frames=4)
    model = JointOccEgo(config, width=16)
    grid, prep = scene()
    current = prep.state['current'][0]
    current['voxel_indices'] = np.array([[2, 2, 1]], np.int64)
    prep.state.update(source_world_points=[np.array([[2.5, 2.5, 1.5]])],
        source_rel_xy=[np.array([[0., 0.]])], source_z_t0=np.array([1.5]),
        column_backgrounds=[x.copy() for x in prep.fallbacks])
    prep.state['column_backgrounds'][0][2, 2, 1] = 11
    for b in prep.state['column_backgrounds']: b[2, 2, 1] = 11
    # Correct the unrelated original test record to this current source/grid.
    record['source_centroid_xy_t0_m'] = torch.tensor([[2.5, 2.5]])
    record['anchors_xy_t0_m'] = torch.tensor([[[2.8, 2.5]]*6])
    record['kta_displacement_xy_m'] = torch.tensor([[[.3, 0.]]*6])
    record['target_source_displacement_xy_m'] = torch.tensor([[[.5, .3]]*6])
    record['target_source_residual_xy_m'] = torch.tensor([[[.2, .3]]*6])
    record.update(scene_name='TRAIN', t0_token='t0', history_tokens=['h0', 'h1', 'h2', 't0'],
                  future_tokens=[f'f{h}' for h in range(6)])
    compact = build_compact_canonical_support(prep, grid)
    raw = prep.raw
    raw['future_gt_occ'] = prep.baseline.copy()
    raw['future_gt_occ'][:, 3, 2, 1] = 4
    raw['_column_causal_preparation'] = dict(prepared_state=prep.state, registrations=prep.registrations,
        _ccr_compact_support=compact, _ccr_compact_conflicts='MUST_NOT_REUSE', audit={}, memory='POISON')
    provider = SimpleNamespace(device=torch.device('cpu'), pcfg=SimpleNamespace(grid=grid, frame_dt_s=.5),
                               workers=1, strong=SimpleNamespace(free_label=17))
    history = HistoryInputs(provider, ram_mib=1).get(record, raw)
    e = ego_history_features(raw['history_poses'])
    inputs = e, np.full(6, 2, np.int64), np.tile(np.array([.2, 0., .02], np.float32), (6, 1))
    row = record, raw, inputs, history
    optimizer = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=5e-4, initial_lr=5e-4),
        dict(params=model.columns.parameters(), lr=3e-4, initial_lr=3e-4),
        dict(params=model.ego.parameters(), lr=3e-4, initial_lr=3e-4)])
    return model, optimizer, provider, row


def test_ego_parameter_budget_same_history_control_and_continuous_pose_convention():
    head = HistoryEgoHead()
    assert sum(p.numel() for p in head.parameters()) == 8710
    current = np.eye(4)
    a = .1
    current[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    current[:3, 3] = [1324, -540, 2]
    se2 = torch.tensor([[h*.3, h*.1, h*.07] for h in range(6)], requires_grad=True)
    expected = poses_from_se2(current, se2.detach().numpy())
    actual = torch_poses(current, se2)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-10)
    actual[:, :3, :3].sum().backward()
    assert se2.grad[:, 2].abs().sum() > 0
    assert np.all(actual.detach().numpy()[:, 2, 3] == current[2, 3])


def test_gradient_ramp_changes_only_backward_not_predicted_forward():
    value = torch.tensor([1., 2.], requires_grad=True)
    prediction = gradient_scale(value, .1)
    assert torch.equal(prediction, value)
    prediction.sum().backward()
    assert torch.allclose(value.grad, torch.full_like(value, .1))


def test_legacy_surface_head_not_changed_and_new_both_actions_active():
    from real_motion.surface_canonical_repair import SurfaceCanonicalRepairHead
    from tools.joint_occ_experiments.model import RepairBothActions
    x, actors = torch.zeros(2, 6, 2), torch.tensor([-2, 0])
    assert (SurfaceCanonicalRepairHead(8, 16).probabilities(x, actors)[..., 1] == 0).all()
    assert (RepairBothActions(8, 16).probabilities(x, actors) == .5).all()


def test_history_cache_extract_discards_future_projection_and_learned_banks():
    model, _, p, row = fixture()
    raw = row[1]
    before = copy.deepcopy(raw['_column_causal_preparation'])
    pure = history_only_payload(raw['_column_causal_preparation'])
    assert set(pure['state']) == {'current_pose', 'current', 'source_world_points', 'source_rel_xy', 'source_z_t0'}
    assert 'column_backgrounds' not in pure['state'] and '_ccr_compact_conflicts' not in pure
    descriptors = HistoryInputs(p, ram_mib=1)
    first = descriptors.get(row[0], raw)
    raw['future_gt_occ'] = 'POISON'
    raw['future_poses'] = 'POISON'
    second = descriptors.get(row[0], raw)
    assert first is second and descriptors.hits == 1
    assert_geometry_equal(before['prepared_state'], raw['_column_causal_preparation']['prepared_state'])


def test_parallel_history_descriptor_single_flight_is_byte_equal_to_serial():
    _, _, provider, row = fixture()
    descriptors = HistoryInputs(provider, ram_mib=1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda _: descriptors.get(row[0], row[1]), range(8)))
    reference = HistoryInputs(provider, ram_mib=0).get(row[0], row[1])
    assert descriptors.misses == 1 and descriptors.hits == 7
    assert all(value is actual[0] for value in actual)
    for key in ('features', 'labels', 'actor', 'classes', 'world', 'presence'):
        np.testing.assert_array_equal(getattr(actual[0][1], key), getattr(reference[1], key))
    assert descriptors.stats()['pending'] == 0


def test_live_torch_context_matches_original_geometry_and_has_pose_gradients():
    model, _, provider, row = fixture()
    model.eval()
    fw = forward_window(model, provider, row, 'gt')
    old = augment_projection(fw['evidence'], fw['plan'], fw['prep'].state['current_pose'],
                             fw['prep'].state['world_to_future'], provider.pcfg.grid)
    mapped, context = continuous_context(fw['evidence'], fw['plan'], fw['prep'].state,
        fw['targets'], fw['yaws'], fw['poses'], provider.pcfg.grid)
    np.testing.assert_allclose(context.detach().numpy(), old.context, atol=2e-6)
    poses = fw['poses'].clone().requires_grad_()
    _, live = continuous_context(fw['evidence'], fw['plan'], fw['prep'].state,
        fw['targets'], fw['yaws'], poses, provider.pcfg.grid)
    live.sum().backward()
    assert poses.grad.abs().sum() > 0


def test_inverse_warp_identity_axis_order_probability_normalization_and_pose_gradient():
    model, _, provider, row = fixture()
    fw = forward_window(model, provider, row, 'gt')
    scene = SoftScene(fw['prep'], provider.pcfg.grid, provider.device)
    pose = torch.eye(4, dtype=torch.float64, requires_grad=True)
    p = scene.background(pose, 0)
    assert torch.allclose(p.sum(0), torch.ones(p.shape[1]), atol=1e-6)
    at = np.ravel_multi_index((5, 5, 0), provider.pcfg.grid.shape_hwd)
    assert p[11, at] == 1
    # Edge geometry is continuous; a perturbation moves the predicted surface.
    weighted = p[11] @ torch.arange(p.shape[1], dtype=p.dtype)
    weighted.backward()
    assert pose.grad.abs().sum() > 0


def test_soft_source_render_and_action_probabilities_are_normalized_and_differentiable():
    model, _, provider, row = fixture()
    fw = forward_window(model, provider, row, 'gt')
    scene = SoftScene(fw['prep'], provider.pcfg.grid, provider.device)
    probability = scene.render(1, fw['poses'][1], fw['targets'], fw['yaws'], fw['mapped'],
                               fw['evidence'], fw['plan'], fw['actions'])
    assert probability.min() >= 0 and probability.max() <= 1
    assert torch.allclose(probability.sum(0), torch.ones(probability.shape[1]), atol=1e-6)
    loss, _, _ = occupancy_loss(probability, torch.tensor(row[1]['future_gt_occ'][1]))
    grads = torch.autograd.grad(loss, [fw['actions'], fw['output']['residual_xy_m'], fw['output']['yaw_delta_rad']], allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_soft_remove_restores_lower_layer_not_unconditional_free():
    model, _, provider, row = fixture()
    row[0]['anchors_xy_t0_m'][:] = torch.tensor([2.5, 2.5])
    fw = forward_window(model, provider, row, 'gt')
    scene = SoftScene(fw['prep'], provider.pcfg.grid, provider.device)
    # The raw current-frame source cell becomes FREE in the static template.
    # The actual renderer fallback is road11: REMOVE must learn that semantic,
    # not accidentally receive FREE supervision from the soft template.
    actions = torch.zeros_like(fw['actions'])
    actions[np.asarray(fw['evidence'].actor) == 0, :, 1] = 1
    p = scene.render(0, fw['poses'][0], fw['targets'], fw['yaws'], fw['mapped'], fw['evidence'], fw['plan'], actions)
    at = np.ravel_multi_index((2, 2, 1), provider.pcfg.grid.shape_hwd)
    assert p[11, at] == 1 and p[17, at] == 0
    hard = compose_canonical(fw['prep'].baseline, fw['evidence'], fw['plan'],
                            actions[..., 0].numpy(), actions[..., 1].numpy())
    assert hard[0][2, 2, 1] == 11


def test_splat_out_of_bounds_does_not_clamp_occupied_points_to_grid_edge():
    _, _, provider, _ = fixture()
    points = torch.tensor([[-20., 3., 1.], [500., 2., 3.]], requires_grad=True)
    p = splat(points, torch.ones(2), provider.pcfg.grid, reduction='sum')
    assert p.sum() == 0
    p.sum().backward()
    assert torch.isfinite(points.grad).all()


def test_full_grid_losses_ignore_unknown_keep_empty_and_semantic_zero_classes():
    p = torch.eye(18)
    labels = torch.arange(18)
    ce, _, lov = occupancy_loss(p, labels)
    assert ce < .01 and lov < .01
    wrong = p.roll(1, 0).requires_grad_()
    loss, _, _ = occupancy_loss(wrong, labels)
    assert loss > 5
    loss.backward()
    assert wrong.grad.abs().sum() > 0
    unknown = torch.full((18,), 255)
    z, _, _ = occupancy_loss(wrong, unknown)
    assert z == 0
    assert lovasz_softmax(p, torch.full((18,), 17)) >= 0


def test_actual_six_horizon_step_updates_wm_repair_and_ego_no_action_bce():
    model, opt, provider, row = fixture()
    before = copy.deepcopy(model.state_dict())
    stats = train_arm(model, opt, provider, [row], 'gt', update=1, total_updates=3, probe=True)
    assert stats['CE'] > 0 and stats['Lovasz'] > 0
    assert stats['gradient_links']['occupancy_to_xy'] > 0
    assert stats['gradient_links']['occupancy_to_actions'] > 0
    for key in ('transport.residual_head.weight', 'columns.static_readout.3.weight',
                'columns.readout.3.weight', 'ego.readout.weight'):
        assert not torch.equal(before[key], model.state_dict()[key]), key
    assert all(p.grad is None for p in model.parameters())


def test_empty_source_and_empty_evidence_window_still_trains_ego_without_constant_loss_backward():
    model, opt, provider, original = fixture()
    record = {k: v[:0] if isinstance(v, torch.Tensor) and v.ndim and len(v) == 1 else v
              for k, v in original[0].items()}
    raw = copy.deepcopy(original[1])
    raw['history_occ'][:] = 17; raw['future_gt_occ'][:] = 17
    causal = raw['_column_causal_preparation']
    causal['prepared_state'].update(current=[], source_world_points=[], source_rel_xy=[], source_z_t0=np.empty(0),
        column_backgrounds=[np.full(provider.pcfg.grid.shape_hwd, 17, np.uint8) for _ in range(6)])
    causal['registrations'] = []
    prep = SimpleNamespace(raw=raw, state=causal['prepared_state'], registrations=[])
    causal['_ccr_compact_support'] = build_compact_canonical_support(prep, provider.pcfg.grid)
    history = HistoryInputs(provider).get(record, raw)
    assert len(history[1]) == 0
    before = model.ego.readout.weight.detach().clone()
    result = train_arm(model, opt, provider, [(record, raw, original[2], history)], 'gt', update=1, total_updates=3)
    assert result['sources'] == result['canonical_points'] == 0
    assert np.isfinite(result['CE']) and not torch.equal(before, model.ego.readout.weight)


def test_same_init_rng_and_exact_resume_after_completed_pair(monkeypatch):
    model, opt, provider, row = fixture()
    other = copy.deepcopy(model)
    opts = {'gt': opt, 'pred': torch.optim.AdamW(other.parameters(), lr=.001)}
    opts['pred'].param_groups[0]['initial_lr'] = .001
    models = dict(gt=model, pred=other)
    contract = dict(protocol='test', schedule=3)
    # Test recovery on a small real Adam update; paired renderer tested above.
    def simple_arm(model, optimizer, provider, rows, mode, **kwargs):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(p.square().sum() for p in model.parameters())*torch.rand(())
        loss.backward(); optimizer.step()
        return dict(loss=float(loss.detach()))
    monkeypatch.setattr('tools.joint_occ_experiments.run.train_arm', simple_arm)
    first = paired_update(models, opts, provider, [row])
    assert first['gt']['loss'] == first['pred']['loss']
    saved = copy.deepcopy(checkpoint(models, opts, contract, 0, 1, 1, {}))
    paired_update(models, opts, provider, [row])
    expected = copy.deepcopy({k: m.state_dict() for k, m in models.items()})
    restore_checkpoint(saved, models, opts, contract)
    paired_update(models, opts, provider, [row])
    assert_nested_equal(expected, {k: m.state_dict() for k, m in models.items()})
    with pytest.raises(RuntimeError, match='differs'):
        restore_checkpoint(saved, models, opts, {**contract, 'schedule': 20})


def test_cli_default_does_not_expand_train_or_pick_dev_threshold():
    names = ('train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint', 'dataroot',
             'train-info', 'dev-info', 'ccr-history-cache', 'ccr-val-history-cache', 'out-dir')
    args = parser().parse_args([item for n in names for item in ('--'+n, 'unused')])
    assert args.epochs == 3 and args.train_fraction == .2 and args.eval_population == 'dev512'
    assert args.remove_threshold == .95 and args.window_batch_size == 4


def test_sparse_layers_and_batched_lovasz_preserve_dense_forward_and_gradients():
    _, _, provider, _ = fixture()
    points = torch.tensor([[2.2, 2.4, 1.5], [3.1, 2.1, 1.4]], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([.3, .7], requires_grad=True)
    flat, small = sparse_splat(points, values, provider.pcfg.grid)
    dense = splat(points, values, provider.pcfg.grid)
    torch.testing.assert_close(small, dense[flat])
    start = torch.rand(18, len(dense)); start /= start.sum(0, keepdim=True)
    expected = start*(1-dense[None]); expected[4] += dense
    actual = sparse_layer(start.clone(), flat, small, 4)
    torch.testing.assert_close(expected, actual)
    a = torch.autograd.grad(expected.square().sum(), (points, values), retain_graph=True)
    b = torch.autograd.grad(actual.square().sum(), (points, values))
    for left, right in zip(a, b): torch.testing.assert_close(left, right)
    labels = torch.arange(len(dense)) % 18
    p = start.clone().requires_grad_()
    vectorized = lovasz_softmax(p, labels)
    terms = []
    for cls in range(17):
        fg = (labels == cls).float()
        error, order = (fg-p[cls]).abs().sort(descending=True, stable=True)
        f = fg[order]; g = 1-(f.sum()-f.cumsum(0))/(f.sum()+(1-f).cumsum(0))
        g = torch.cat((g[:1], g[1:]-g[:-1]))
        terms.append((error*g).sum())
    reference = torch.stack(terms).mean()
    torch.testing.assert_close(reference, vectorized)
    a = torch.autograd.grad(reference, p, retain_graph=True)[0]
    b = torch.autograd.grad(vectorized, p)[0]
    torch.testing.assert_close(a, b)


def test_pred_never_uses_gt_future_pose_background_or_occupancy_and_ego_has_occ_gradient():
    from real_motion.strong_w2det import StrongW2DetConfig
    model, opt, provider, row = fixture()
    provider.strong = StrongW2DetConfig(free_label=17)
    model.eval()
    first = forward_window(model, provider, row, 'pred')
    # Poison every old GT-conditioned cache field, not merely future labels.
    raw = copy.deepcopy(row[1])
    raw['future_poses'] = 'POISON'
    raw['future_gt_occ'] = 'POISON'
    raw['_column_causal_preparation']['prepared_state']['column_backgrounds'] = 'POISON'
    raw['_column_causal_preparation']['prepared_state']['world_to_future'] = 'POISON'
    second = forward_window(model, provider, (row[0], raw, row[2], row[3]), 'pred')
    assert torch.equal(first['poses'], second['poses'])
    assert torch.equal(first['actions'], second['actions'])
    assert all(np.array_equal(a, b) for a, b in zip(first['prep'].baseline, second['prep'].baseline))
    scene = SoftScene(first['prep'], provider.pcfg.grid, provider.device)
    prob = scene.render(1, first['poses'][1], first['targets'], first['yaws'], first['mapped'],
                        first['evidence'], first['plan'], first['actions'])
    loss, _, _ = occupancy_loss(prob, torch.as_tensor(row[1]['future_gt_occ'][1]))
    gradient = torch.autograd.grad(loss, first['ego'], retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    stats = train_arm(model, opt, provider, [row], 'pred', update=1, total_updates=3, probe=True)
    assert stats['gradient_links']['occupancy_to_ego'] > 0


def test_soft_add_stencil_cannot_overwrite_neighboring_original_occupancy():
    model, _, provider, row = fixture()
    fw = forward_window(model, provider, row, 'gt')
    scene = SoftScene(fw['prep'], provider.pcfg.grid, provider.device)
    candidate = np.flatnonzero((fw['evidence'].actor < 0) & fw['plan'].legal[:, 0, 0])
    assert len(candidate)
    ids = candidate[0]
    mapped = fw['mapped'].clone()
    # A legal originally-free candidate overlaps an occupied neighbor in its
    # trilinear stencil; the neighbor still cannot be changed by ADD.
    mapped[ids, 0] = torch.tensor([5.1, 5.5, .5], dtype=torch.float64)
    actions = torch.zeros_like(fw['actions']); actions[ids, 0, 0] = 1
    no_edit = scene.render(0, fw['poses'][0], fw['targets'], fw['yaws'], mapped, fw['evidence'], fw['plan'], actions*0)
    edited = scene.render(0, fw['poses'][0], fw['targets'], fw['yaws'], mapped, fw['evidence'], fw['plan'], actions)
    occupied = scene.baselines[0] != 17
    torch.testing.assert_close(no_edit[:, occupied], edited[:, occupied])


def test_new_execution_readout_matches_reference_and_keeps_remove_active():
    from tools.joint_occ_experiments.execution import BothActionsExecution
    model, _, provider, row = fixture()
    model.eval().requires_grad_(False)
    reference = forward_window(model, provider, row, 'gt', soft=False)
    session = BothActionsExecution(model.columns, provider.device, graphs=False)
    provider.paired_eval_sessions = {id(model.columns): session}
    actual = forward_window(model, provider, row, 'gt', soft=False)
    torch.testing.assert_close(reference['actions'], actual['actions'], rtol=0, atol=0)
    assert (actual['actions'][..., 1] > 0).all()
    assert session.stats()['counts']['points'] == len(reference['evidence'])
    session.close()


def test_actual_hard_evaluation_integer_resume_and_soft_hard_gap(monkeypatch):
    from real_motion.strong_w2det import StrongW2DetConfig
    model, _, provider, row = fixture()
    provider.strong = StrongW2DetConfig(free_label=17)
    models = {'gt': model, 'pred': copy.deepcopy(model)}
    source = SimpleNamespace(nusc=None)
    # Only annotations/support mocked; actual two-model hard prediction, three
    # compositors and full-grid metric accumulation run here.
    monkeypatch.setattr('tools.joint_occ_experiments.engine.gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr('tools.joint_occ_experiments.engine.moving_support_masks',
                        lambda rows, shape: np.ones((6, *shape), bool))
    saved = []
    expected = evaluate(models, provider, source, [[row], [row]], save=lambda state: saved.append(copy.deepcopy(state)))
    first = evaluate(models, provider, source, [[row]], save=lambda state: saved.append(copy.deepcopy(state)))
    state = json.loads(json.dumps(saved[-1]))
    resumed = evaluate(models, provider, source, [[row], [row]], saved=state)
    assert first['windows'] == 1 and expected['windows'] == resumed['windows'] == 2
    # Absent classes are NaN, not zero; JSON canonicalization compares the
    # identical absent pattern without treating NaN != NaN as a resume error.
    assert json.dumps(expected['variants'], sort_keys=True) == json.dumps(resumed['variants'], sort_keys=True)
    assert expected['trajectory'] == resumed['trajectory']
    assert json.dumps(expected['surrogate_gap'], sort_keys=True) == json.dumps(resumed['surrogate_gap'], sort_keys=True)


def test_cli_smoke_then_same_directory_resume_preserves_paired_optimizer_and_rng(tmp_path, monkeypatch):
    from tools.joint_occ_experiments import run
    model, _, provider, row = fixture()
    config = tmp_path/'config.yaml'; config.write_text('test')
    base = tmp_path/'base.pt'; torch.save(dict(model_config=asdict(model.transport.v17_config)), base)
    paths = {name: tmp_path/name for name in ('train-cache', 'dev-cache', 'train-info', 'dev-info', 'population-manifest')}
    for path in paths.values(): path.write_text('test')
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    keys = [('DEV', r['t0_token']) for r in dev[:512]]
    monkeypatch.setattr(run, 'require_cuda', lambda name: torch.device('cpu'))
    monkeypatch.setattr(run, 'load_cache', lambda path: ({}, train if path == str(paths['train-cache']) else dev))
    monkeypatch.setattr(run, 'load_manifest', lambda path: ({'parent_keys': keys}, keys[:64], None))
    monkeypatch.setattr(run, 'load_runtime_config', lambda *args: None)
    monkeypatch.setattr(run, 'make_prepare_config', lambda cfg: provider.pcfg)
    monkeypatch.setattr(run, 'PilotProvider', lambda *args: provider)
    source_calls = []
    # Match the real adapter signature: its second positional argument is the
    # dataset VERSION, not the split info pickle. Check both fresh and resume.
    def window_source(dataroot, version='v1.0-trainval', info_pkl=None, verbose=False):
        assert dataroot == str(tmp_path)
        assert version == 'v1.0-trainval'
        assert info_pkl in (str(paths['train-info']), str(paths['dev-info']))
        source_calls.append(info_pkl)
        return SimpleNamespace(nusc=None)
    monkeypatch.setattr(run, 'NuScenesWindowSource', window_source)
    class Disposable:
        kernels = pool = None
        def __init__(self, *args): pass
        def close(self): pass
    monkeypatch.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
    monkeypatch.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
    def caches(provider, args, root):
        provider.ccr_history_cache = provider.ccr_val_history_cache = SimpleNamespace(stats=lambda: {}, close=lambda: None)
        return dict(train={'namespace': 'train'}, val={'namespace': 'val'})
    monkeypatch.setattr(run, 'attach_caches', caches)
    monkeypatch.setattr(run, 'prefetched_batches', lambda provider, source, groups, descriptors, **k:
                        ([(record, row[1], row[2], row[3]) for record in group] for group in groups))
    # Dense real backward is covered separately. Here exercise the real CLI,
    # schedule/cursor/checkpoint/optimizer/RNG and final eval session lifecycle.
    def small_arm(model, optimizer, provider, rows, mode, update, total_updates, **kwargs):
        optimizer.zero_grad(set_to_none=True)
        value = sum(p.square().sum() for p in model.parameters())*torch.rand(())/100
        value.backward(); optimizer.step()
        return dict(windows=len(rows), seconds=.01, stages_seconds={}, CE=float(value.detach()), gradient_links={})
    monkeypatch.setattr(run, 'train_arm', small_arm)
    def fixed_eval(models, provider, source, batches, **kwargs):
        assert all(not m.training and not any(p.requires_grad for p in m.parameters()) for m in models.values())
        return dict(windows=sum(len(rows) for rows in batches), variants={}, trajectory={})
    monkeypatch.setattr(run, 'evaluate', fixed_eval)
    out = tmp_path/'stopped'
    args = ['--config', str(config), '--base-checkpoint', str(base), '--dataroot', str(tmp_path),
            '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
            '--train-fraction', '.00004', '--out-dir', str(out)]
    args += [item for name, path in paths.items() for item in ('--'+name, str(path))]
    assert run.main(threading.Event(), args+['--max-updates', '1']) == 0
    first = torch.load(out/'last.pt', weights_only=False)
    assert first['updates'] == 1 and first['epoch'] == 1 and first['batch'] == 0
    assert run.main(threading.Event(), args+['--resume', str(out/'last.pt')]) == 0
    resumed = torch.load(out/'last.pt', weights_only=False)
    assert resumed['updates'] == 3 and resumed['epoch'] == 3
    assert json.loads((out/'results.json').read_text())['evaluation']['windows'] == 512
    other = tmp_path/'uninterrupted'
    full_args = args.copy(); full_args[full_args.index(str(out))] = str(other)
    assert run.main(threading.Event(), full_args) == 0
    uninterrupted = torch.load(other/'last.pt', weights_only=False)
    assert_nested_equal(uninterrupted['models'], resumed['models'])
    assert_nested_equal(uninterrupted['optimizers'], resumed['optimizers'])
    assert source_calls == [str(paths['train-info']), str(paths['dev-info'])]*3
