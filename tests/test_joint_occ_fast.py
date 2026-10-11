"""Forward, chain-rule and optimizer parity, not a claimed CUDA speed result."""
import copy

import pytest
import torch

from tools.joint_occ_experiments.fast_autograd import compose_sparse
from tools.joint_occ_experiments.soft_geometry import (
    SoftScene, sparse_layer, lovasz_softmax, lovasz_softmax_reference,
)
from tools.joint_occ_experiments.engine import forward_window, train_arm
from test_joint_occ_ego import fixture


def reference_composition(background, operations, alphas):
    result = background.clone()
    for (kind, ids, cls, destination), alpha in zip(operations, alphas):
        if kind == 'overlay':
            result = sparse_layer(result, ids, alpha, cls)
        else:
            local = result[:, ids].clone()
            amount = local[cls].clone()*alpha
            local[cls] = local[cls]-amount
            local = local.scatter_add(0, destination[None], amount[None])
            result[:, ids] = local
    return result


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_mixed_overlapping_sparse_layers_forward_and_every_gradient(dtype):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(871)
    background = torch.rand(18, 91, dtype=dtype, generator=generator)
    background = (background/background.sum(0)).requires_grad_()
    operations, alphas = [], []
    for i in range(25):
        ids = torch.randperm(91, generator=generator)[:19]
        kind = 'overlay' if i % 3 else 'remove'
        destination = torch.randint(18, (len(ids),), generator=generator) if kind == 'remove' else None
        operations.append((kind, ids, i % 18, destination))
        alphas.append(torch.rand(len(ids), dtype=dtype, generator=generator).requires_grad_())
    expected = reference_composition(background, operations, alphas)
    actual = compose_sparse(background, operations, alphas)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn(expected.shape, dtype=dtype, generator=generator)
    reference_gradients = torch.autograd.grad((expected*weights).sum(), (background, *alphas))
    actual_gradients = torch.autograd.grad((actual*weights).sum(), (background, *alphas))
    for a, b in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)


def test_sparse_composition_gradcheck_zero_edit_and_same_class_fallback():
    background = torch.rand(3, 5, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 3])
    operations = [('overlay', ids, 0, None), ('remove', ids, 0, torch.tensor([0, 2]))]
    alpha = torch.tensor([.21, .37], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([.61, .42], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda b, a, r: compose_sparse(b, operations, [a, r]),
                                    (background, alpha, beta))
    assert compose_sparse(background, [], []) is background
    unchanged = compose_sparse(background, [('remove', ids, 0, torch.tensor([0, 0]))], [beta])
    torch.testing.assert_close(unchanged, background)
    assert torch.autograd.grad(unchanged.sum(), beta)[0].abs().sum() == 0


@pytest.mark.parametrize('labels', [
    torch.arange(180) % 18,
    torch.full((180,), 255),
    torch.full((180,), 17),
    torch.cat((torch.arange(120) % 18, torch.full((60,), 255))),
])
@pytest.mark.parametrize('tied', [False, True])
def test_lovasz_exact_objective_and_gradients_with_ties_ignored_and_free(labels, tied):
    generator = torch.Generator().manual_seed(922)
    probability = (torch.full((18, 180), .5) if tied else
                   torch.rand(18, 180, generator=generator)).requires_grad_()
    expected = lovasz_softmax_reference(probability, labels)
    actual = lovasz_softmax(probability, labels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a = torch.autograd.grad(actual, probability)[0]
    b = torch.autograd.grad(expected, probability)[0]
    torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-8)


def test_lovasz_exact_match_zero_subgradient():
    probability = torch.eye(18).requires_grad_()
    actual = lovasz_softmax(probability, torch.arange(18))
    assert actual == 0
    assert torch.autograd.grad(actual, probability)[0].abs().sum() == 0


def test_real_scene_all_six_fused_outputs_and_boundary_gradients_equal_reference():
    model, _, provider, row = fixture()
    model.eval()
    fw = forward_window(model, provider, row, 'gt')
    # Fractional coordinates/pose exercise geometry derivatives, not only labels.
    inputs = {k: fw[k].detach().clone().requires_grad_() for k in ('poses', 'targets', 'yaws', 'mapped', 'actions')}
    offset = torch.zeros_like(inputs['poses']); offset[:, 0, 3] = .037
    inputs['poses'] = inputs['poses']+offset
    scene = SoftScene(fw['prep'], provider.pcfg.grid, provider.device)
    for h in range(6):
        args = (h, inputs['poses'][h], inputs['targets'], inputs['yaws'], inputs['mapped'],
                fw['evidence'], fw['plan'], inputs['actions'])
        actual = scene.render(*args)
        expected = scene.render_reference(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.linspace(-1., 2., actual.numel()).reshape_as(actual)
        a = torch.autograd.grad((actual*weights).sum(), tuple(inputs.values()), retain_graph=True, allow_unused=True)
        b = torch.autograd.grad((expected*weights).sum(), tuple(inputs.values()), retain_graph=True, allow_unused=True)
        for left, right in zip(a, b):
            assert (left is None) == (right is None)
            if left is not None: torch.testing.assert_close(left, right, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize('mode', ['gt', 'pred'])
def test_actual_adam_step_losses_all_parameters_and_moments_match_reference(mode):
    from real_motion.strong_w2det import StrongW2DetConfig
    model, optimizer, provider, row = fixture()
    provider.strong = StrongW2DetConfig(free_label=17)
    initial, adam, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), torch.get_rng_state()
    outcomes, moments, stats, traversals = [], [], [], []
    for optimized in (False, True):
        model.load_state_dict(initial); optimizer.load_state_dict(copy.deepcopy(adam)); torch.set_rng_state(rng)
        calls = []
        hook = model.transport.residual_head.weight.register_hook(lambda gradient: calls.append(gradient.norm().item()))
        stats.append(train_arm(model, optimizer, provider, [row, row], mode,
                               update=1, total_updates=3, probe=True, optimized=optimized))
        hook.remove(); traversals.append(len(calls))
        outcomes.append(copy.deepcopy(model.state_dict())); moments.append(copy.deepcopy(optimizer.state_dict()))
    for key in ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry', 'grad_norm'):
        assert stats[0][key] == pytest.approx(stats[1][key], rel=3e-5, abs=2e-6), key
    assert traversals[1] == 2 and traversals[0] > traversals[1]  # ONCE per window, not per horizon.
    for key in ('occupancy_to_xy', 'occupancy_to_yaw', 'occupancy_to_actions'):
        assert stats[1]['gradient_links'][key] > 0
        assert stats[0]['gradient_links'][key] == pytest.approx(stats[1]['gradient_links'][key], rel=1e-4, abs=1e-6)
    if mode == 'pred': assert stats[1]['gradient_links']['occupancy_to_ego'] > 0
    for key in outcomes[0]:
        torch.testing.assert_close(outcomes[0][key], outcomes[1][key], rtol=3e-5, atol=2e-6, msg=key)
    for key, state in moments[0]['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, moments[1]['state'][key][name], rtol=3e-5, atol=2e-6)
