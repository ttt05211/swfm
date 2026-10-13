import copy
import pytest
import torch
from tools.shared_history_experiments.refine.model import SemanticRefiner
from tools.shared_history_experiments.refiner_batch_readout.optimized import (
    BACKENDS, batch_forward, upsample_xy, regular_forward, replace_forward, replay_update)


def test_xy_upsample_matches_grid_centres_preserves_z():
    x = torch.arange(2*3*4*5*4.).reshape(2, 3, 4, 5, 4).requires_grad_()
    shape = (8, 10, 4)
    xyz = torch.stack(torch.meshgrid(*(torch.arange(s)+.5 for s in shape), indexing='ij'), -1).reshape(-1, 3)
    grid = (xyz/xyz.new_tensor(shape)*2-1)[:, [2, 1, 0]][None, :, None, None].expand(2, -1, -1, -1, -1)
    ref = torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode='border')[:, :, :, 0, 0]
    out = upsample_xy(x, shape)
    torch.testing.assert_close(out, ref, rtol=2e-6, atol=3e-5)
    grad = torch.randn_like(out)
    a = torch.autograd.grad(out, x, grad, retain_graph=True)[0]
    b = torch.autograd.grad(ref, x, grad)[0]
    torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(upsample_xy(x, shape, volume=True), ref, rtol=2e-6, atol=3e-5)


@pytest.mark.parametrize('backend', BACKENDS[1:])
def test_real_readout_nonzero_head_shape_time_and_gradients(backend):
    torch.manual_seed(12); torch.set_num_threads(1)
    model = SemanticRefiner(source_dim=16, feature_width=16, z_bins=4,
                            memory_width=4, width=8, chunk_size=31)
    with torch.no_grad(): model.readout[-1].weight.normal_(std=.03)
    memories = [torch.randn(6, 4, 4, 4, 4, requires_grad=True) for _ in range(4)]
    priors = [torch.rand(6, 18, 256, requires_grad=True) for _ in range(4)]
    reference = batch_forward(model, memories, priors, (8, 8, 4), 'reference')
    actual = batch_forward(model, memories, priors, (8, 8, 4), backend)
    for a, b in zip(actual, reference): torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    tensors = [*memories, *priors, model.readout[-1].weight]
    ga = torch.autograd.grad(sum(z.square().mean()/4 for z in actual), tensors, retain_graph=True)
    gb = torch.autograd.grad(sum(z.square().mean()/4 for z in reference), tensors)
    for a, b in zip(ga, gb): torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)


def test_instance_hooks_restored_on_failure_and_invalid_lattice():
    model = SemanticRefiner()
    original = model.forward.__func__
    with pytest.raises(RuntimeError):
        with replace_forward(model, lambda *a: None): raise RuntimeError('forced')
    assert model.forward.__func__ is original and 'forward' not in model.__dict__
    with pytest.raises(ValueError): upsample_xy(torch.ones(1, 1, 2, 2, 1), (5, 4, 1))


def test_complete_b4_replay_preserves_input_and_occ_only_links():
    from test_full_gt_training import fixture
    torch.manual_seed(13)
    model, adam, provider, row = fixture()
    initial = copy.deepcopy(model.state_dict()); opt = copy.deepcopy(adam.state_dict())
    for backend in BACKENDS:
        model.load_state_dict(initial); adam.load_state_dict(opt)
        result = replay_update(model, adam, provider, [row]*4, backend)
        assert result['windows'] == 4
        assert 'forward' not in model.columns.__dict__
        assert not torch.equal(initial['columns.readout.4.weight'], model.columns.readout[-1].weight)
        assert not list(model.ego.parameters())
