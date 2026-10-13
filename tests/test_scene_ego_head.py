"""Independent component/Pred-forward tests, not nuScenes or FPS claims."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.ego_navigation import ego_history_features, poses_from_se2
from tools.ego_experiments.ego_kinematic import history_state, integrate
from tools.joint_occ_experiments.model import torch_poses, ego_geometry_loss
from tools.scene_ego_head.model import (SceneEgoHead, SceneEgoConfig, HistoricalScene,
                                       prepare_scene, accumulate_increments)
from tools.scene_ego_head.integration import SceneEgoRefiner, forward_pred_window, rebuild_pred_backgrounds


def example(device='cpu'):
    torch.set_num_threads(1)
    grid = OccupancyGrid(x_min=-8, y_min=-8, z_min=-1,
                         voxel_size=(1., 1., 1.), shape_hwd=(16, 16, 4))
    poses = np.repeat(np.eye(4)[None], 4, 0)
    poses[:, 0, 3] = [-3, -2, -1, 0]
    e = torch.as_tensor(ego_history_features(poses), device=device)[None]
    field = torch.randn(4, 16, 8, 8, device=device, requires_grad=True)
    known = torch.ones(4, 16, 16, 4, device=device, dtype=torch.bool)
    scene = prepare_scene(field, observed=known, history_poses=poses, ego_history=e[0], grid=grid, pool_side=2)
    head = SceneEgoHead(SceneEgoConfig(16, 32, 4, 2)).to(device)
    commands = torch.full((1, 6), 2, device=device)
    return head, e, commands, field, scene, poses, grid, known


def open_readout(head):
    with torch.no_grad(): head.readout[-1].weight.normal_(std=.005)


def test_default_parameter_budget_and_zero_residual_prior():
    assert 100000 <= SceneEgoHead().specification()['parameters'] <= 200000
    head, e, cmd, _, scene, *_ = example()
    for mode in range(3):
        out = head(e, cmd*0+mode, scene)
        torch.testing.assert_close(out['se2'], integrate(history_state(e)), rtol=1e-6, atol=1e-6)
        assert out['branch_increments'].shape == (1, 6, 3, 3)
        assert not out['residual_increments'].count_nonzero()


def test_mode_selection_precedes_accumulation_and_selected_gradient_only():
    branches = torch.zeros(1, 6, 3, 3, requires_grad=True)
    with torch.no_grad():
        branches[:, :, :, 0] = torch.tensor([1., 10., 100.])
        branches[:, :, :, 2] = .1
    cmd = torch.tensor([[0, 1, 2, 0, 1, 2]])
    steps, absolute = accumulate_increments(branches, cmd)
    torch.testing.assert_close(absolute[0, :, 0], torch.tensor([1., 11., 111., 112., 122., 222.]))
    torch.testing.assert_close(absolute[0, :, 2], torch.arange(1, 7)*.1)
    absolute.sum().backward()
    for h, mode in enumerate(cmd[0]):
        assert branches.grad[0, h, mode].abs().sum() > 0
        assert branches.grad[0, h, torch.arange(3) != mode].count_nonzero() == 0
    onehot = torch.nn.functional.one_hot(cmd, 3)
    torch.testing.assert_close(accumulate_increments(branches, onehot)[1], absolute)
    with pytest.raises(ValueError): accumulate_increments(branches, torch.full((1, 6, 3), 1/3))


def test_pooling_axes_centres_pose_and_global_frame_invariance():
    head, e, _, field, _, poses, grid, known = example()
    # Each 2x2 pooled token contains four coarse cells on each axis.
    field = torch.arange(8.)[:, None].expand(8, 8)[None, None].expand(4, 16, -1, -1)
    first = prepare_scene(field, observed=known, history_poses=poses, ego_history=e[0], grid=grid, pool_side=2)
    torch.testing.assert_close(first.features[0, :4, 0], torch.tensor([1.5, 1.5, 5.5, 5.5]))
    torch.testing.assert_close(first.geometry[0, 12:16, :2]*40,
                               torch.tensor([[-4., -4.], [-4., 4.], [4., -4.], [4., 4.]]))
    transform = np.eye(4); transform[:3, 3] = [1500, -800, 4]
    a = .3; transform[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    world = transform[None] @ poses
    other = prepare_scene(field, observed=known, history_poses=world, ego_history=e[0], grid=grid, pool_side=2)
    torch.testing.assert_close(first.geometry, other.geometry, atol=1e-6, rtol=1e-6)
    wrong = known.float()
    with pytest.raises(ValueError): prepare_scene(field, observed=wrong, history_poses=poses, ego_history=e[0], grid=grid)


def test_partial_observation_weighted_pool_and_nondivisible_bins():
    _, e, _, _, _, poses, _, _ = example()
    grid = OccupancyGrid(x_min=0, y_min=0, z_min=0,
                         voxel_size=(1., 1., 1.), shape_hwd=(20, 20, 4))
    field = torch.ones(4, 16, 10, 10)
    field[:, :, 0, 0] = 999.
    known = torch.ones(4, 20, 20, 4, dtype=torch.bool); known[:, :2, :2] = False
    scene = prepare_scene(field, observed=known, history_poses=poses, ego_history=e[0], grid=grid, pool_side=8)
    torch.testing.assert_close(scene.features, torch.ones_like(scene.features))
    # Bin0 coarse centres are 1 and 3 -> 2m; not a uniform 1.25m proxy.
    torch.testing.assert_close(scene.geometry[0, -64, :2]*40, torch.tensor([2., 2.]))


def test_scene_and_position_affect_prediction_no_global_mean_shortcut():
    head, e, cmd, field, scene, *_ = example(); open_readout(head)
    a = head(e, cmd, scene)['se2']
    changed = HistoricalScene(scene.features.flip(1), scene.geometry, scene.valid)
    b = head(e, cmd, changed)['se2']
    assert not torch.allclose(a, b, atol=1e-7, rtol=0)
    a.square().mean().backward()
    assert field.grad.norm() > 0
    assert head.scene.weight.grad.norm() > 0


def test_all_unknown_scene_is_finite_and_no_invalid_feature_dependency():
    head, e, cmd, _, scene, *_ = example(); open_readout(head)
    invalid = torch.zeros_like(scene.valid)
    first = head(e, cmd, HistoricalScene(scene.features, scene.geometry, invalid))['se2']
    second = head(e, cmd, HistoricalScene(scene.features*100, scene.geometry+20, invalid))['se2']
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_zero_init_first_step_blocks_scene_then_opens_and_exact_restore():
    head, e, cmd, field, scene, *_ = example()
    target = integrate(history_state(e))+torch.tensor([.4, -.2, .03])
    adam = torch.optim.AdamW(head.parameters(), lr=1e-3)
    def step():
        adam.zero_grad(set_to_none=True)
        # Rebuild token graph, as a live encoder would, for each update.
        local = replace(scene, features=scene.features.detach().clone().requires_grad_())
        loss = ego_geometry_loss(head(e, cmd, local)['se2'], target)
        loss.backward(); gradient = local.features.grad.clone(); adam.step()
        return gradient
    assert step().count_nonzero() == 0
    model_state, adam_state = copy.deepcopy(head.state_dict()), copy.deepcopy(adam.state_dict())
    rng = torch.get_rng_state()
    assert step().norm() > 0
    expected, expected_adam = copy.deepcopy(head.state_dict()), copy.deepcopy(adam.state_dict())
    head.load_state_dict(model_state); adam.load_state_dict(adam_state); torch.set_rng_state(rng)
    step()
    for key, value in expected.items(): torch.testing.assert_close(head.state_dict()[key], value, atol=0, rtol=0)
    for key, values in expected_adam['state'].items():
        for name, value in values.items(): torch.testing.assert_close(adam.state_dict()['state'][key][name], value, atol=0, rtol=0)


def refiner_fixture(device='cpu'):
    from test_shared_semantic_refiner import fixture
    old, _, provider, row = fixture(device)
    model = SceneEgoRefiner(old.transport.config, feature_width=16, z_bins=4,
        memory_width=4, width=8, chunk_size=32, ego_width=32, pool_side=2).to(device)
    return model, provider, row


def causal_background(historical, state, poses, provider):
    assert set(historical) == {'history_occ', 'history_observed', 'history_poses'}
    assert 'column_backgrounds' not in state and 'world_to_future' not in state
    shape = np.asarray(historical['history_occ']).shape[1:]
    return [np.full(shape, 17, np.uint8) for _ in range(6)]


def test_pred_forward_future_poison_and_input_immutability():
    model, provider, row = refiner_fixture()
    calls = []
    handle = model.transport.history_encoder.register_forward_hook(lambda *args: calls.append(1))
    model.eval(); expected = forward_pred_window(model, provider, row, background_builder=causal_background)
    handle.remove()
    assert len(calls) == 1  # no second scene encoder or duplicate history forward
    raw = copy.deepcopy(row[1]); old_poses = np.asarray(raw['history_poses']).copy()
    for key in ('future_poses', 'future_gt_occ', 'future_annotations', 'future_masks', '_column_causal_preparation'):
        raw[key] = 'POISON'
    causal = copy.deepcopy(row[3]); causal[0]['state']['column_backgrounds'] = 'POISON'
    ego = (*row[2][:2], 'POISON EGO LABEL')
    actual = forward_pred_window(model, provider, (row[0], raw, ego, causal), background_builder=causal_background)
    for key in ('poses', 'prior', 'memory', 'logits'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    np.testing.assert_array_equal(raw['history_poses'], old_poses)
    assert causal[0]['state']['column_backgrounds'] == 'POISON'


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_complete_refiner_occ_gradient_to_scene_head_and_poses(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from tools.shared_history_experiments.refine.engine import semantic_loss
    model, provider, row = refiner_fixture(device); open_readout(model.ego)
    fw = forward_pred_window(model, provider, row, background_builder=causal_background)
    for tensor in (fw['poses'], fw['ego']['se2'], fw['field']): tensor.retain_grad()
    losses, _, _ = semantic_loss(fw['logits'], row[1]['future_gt_occ'])
    losses.mean().backward()
    assert fw['poses'].grad.norm() > 0 and fw['ego']['se2'].grad.norm() > 0
    assert fw['field'].grad.norm() > 0 and model.ego.readout[-1].weight.grad.norm() > 0
    assert model.ego.attention.in_proj_weight.grad.norm() > 0
    assert torch.isfinite(fw['logits']).all()


def test_tilt_pose_matches_existing_convention_and_prediction_drives_background():
    head, e, cmd, _, scene, *_ = example(); open_readout(head)
    current = np.eye(4); a = .04
    current[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    current[:3, 3] = [2200, -300, 2.]
    se2 = head(e, cmd, scene)['se2'][0]
    poses = torch_poses(current, se2)
    np.testing.assert_allclose(poses.detach().numpy(), poses_from_se2(current, se2.detach().numpy()), atol=1e-10)
    assert torch.all(poses[:, 2, 3] == current[2, 3])


def test_default_background_builder_receives_pred_not_gt(monkeypatch):
    from tools.scene_ego_head import integration as impl
    model, provider, row = refiner_fixture(); provider.strong.max_match_speed_mps = 100.
    captured = []
    monkeypatch.setattr(impl.runtime, 'extract_instances_cropped_exact', lambda *a, **k: [])
    monkeypatch.setattr(impl.runtime, 'match_instances', lambda *a, **k: [])
    def strong(occ, current, poses, *args, **kwargs):
        captured.append(poses.copy())
        return [np.full_like(occ, 17) for _ in range(6)], [[] for _ in range(6)]
    monkeypatch.setattr(impl.runtime, '_strong_all_horizons', strong)
    monkeypatch.setattr(impl.runtime, 'baseline_clear_flat_indices', lambda *a, **k: [])
    monkeypatch.setattr(impl, 'compose_component_replacements_fast_exact', lambda a, *args, **k: a)
    with torch.no_grad(): model.ego.readout[-1].bias.reshape(6, 3, 3)[..., 0] = .1
    fw = forward_pred_window(model, provider, row)
    np.testing.assert_array_equal(captured[0], fw['poses'].detach().numpy())
    assert not np.array_equal(captured[0], row[1]['future_poses'])
    assert 'column_backgrounds' not in row[3][0]['state']


def test_real_cpu_strong_background_and_pred_forward_without_gt_future_cache():
    from real_motion.strong_w2det import StrongW2DetConfig
    model, provider, row = refiner_fixture()
    provider.strong = StrongW2DetConfig()
    with torch.no_grad(): model.ego.readout[-1].bias.reshape(6, 3, 3)[..., 0] = .05
    raw = copy.deepcopy(row[1])
    for key in ('future_poses', 'future_gt_occ', '_column_causal_preparation'):
        raw[key] = 'POISON'
    fw = forward_pred_window(model, provider, (row[0], raw, (*row[2][:2], 'POISON'), row[3]))
    assert torch.isfinite(fw['logits']).all()
    assert len(fw['state']['column_backgrounds']) == 6
    assert all(b.dtype == np.uint8 for b in fw['state']['column_backgrounds'])
    assert not torch.equal(fw['poses'], torch.eye(4).expand(6, -1, -1))
