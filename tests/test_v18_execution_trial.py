from contextlib import nullcontext
import numpy as np
import pytest
import torch
from real_motion.v18_execution_trial import majority_fill_sparse_cuda_exact, majority_fill_native_exact, reuse_v18_projections, FrozenV18Graph, MOTION_KEYS
from real_motion.runtime_fastpath import majority_fill_sparse_5x5x1
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.motion_transport import FEATURE_DIM


@pytest.mark.parametrize('history', [4, 6])
def test_reused_v18_projections_preserve_forward_and_gradients(history):
    torch.manual_seed(7)
    model = LocalSpatialTemporalWorldModelV18SE2(LocalSTWMV17Config(
        history_frames=history, d_model=16, semantic_dim=8, heads=2, blocks=1, decoder_blocks=1))
    # Nonzero learned heads exercise the complete path.
    torch.nn.init.normal_(model.residual_head.weight, std=.02)
    torch.nn.init.normal_(model.yaw_head.weight, std=.02)
    values = [torch.randn(2, FEATURE_DIM), torch.randint(18, (2, 6, 20, 20)),
              torch.randn(2, 6, 2), torch.randn(2, 6, 5), torch.randint(2, (2, 6, 20, 20))]
    outputs, gradients = [], []
    for optimized in (False, True):
        model.zero_grad(set_to_none=True)
        with reuse_v18_projections(model) if optimized else nullcontext():
            result = model(*values, return_latents=True)
            sum(v.float().square().sum() for v in result.values()).backward()
        outputs.append({k: v.detach().clone() for k, v in result.items()})
        gradients.append({k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None})
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=2e-5, atol=2e-5)
    assert not hasattr(model, 'reuse_source_projection')
    assert not hasattr(model.decoder[0], 'reuse_context_norm')


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_sparse_majority_exact_on_cuda_or_cpu(seed):
    rng = np.random.default_rng(seed)
    sem = rng.integers(0, 18, (27, 29, 3), dtype=np.uint8)
    unknown = rng.random(sem.shape) < .35
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    expected = majority_fill_sparse_5x5x1(sem, unknown)
    np.testing.assert_array_equal(majority_fill_sparse_cuda_exact(sem, unknown, device=device, chunk=97), expected)


def test_sparse_majority_all_known_or_unknown_and_negative_label():
    sem = np.full((9, 8, 2), 17, np.uint8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for value in (False, True):
        np.testing.assert_array_equal(majority_fill_sparse_cuda_exact(sem, np.full(sem.shape, value), device=device), sem)


@pytest.mark.parametrize('seed', [11, 12])
def test_native_majority_matches_frozen_scipy_including_edges(seed, monkeypatch, tmp_path):
    from real_motion.native_column_cpu import prepare_native
    from real_motion.strong_w2det import majority_fill
    import shutil
    if not any(shutil.which(name) for name in ('cl', 'c++', 'g++', 'clang++')):
        pytest.skip('explicit C++ compiler unavailable; no package installation')
    prepare_native(tmp_path/'build')
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'native')
    rng = np.random.default_rng(seed)
    for unknown_probability in (0., .01, .3, .7, .99, 1.):
        sem = rng.integers(0, 18, (23, 25, 4), dtype=np.uint8)
        unknown = rng.random(sem.shape) < unknown_probability
        expected = majority_fill(sem, unknown)
        np.testing.assert_array_equal(majority_fill_native_exact(sem, unknown), expected)
    # 3/10 threshold and 4/4 tied winners; do not replace scipy rounding with
    # a different >= comparison or unconditional ascending-class argmax.
    for labels in ([1]*3+[2]*3+[3]*2+[4]*2, [1]*4+[2]*4):
        sem = np.full((5, 5, 1), 17, np.uint8)
        unknown = np.ones(sem.shape, bool)
        positions = [(x,y,0) for x in range(5) for y in range(5) if (x,y)!=(2,2)]
        for coordinate, label in zip(positions, labels):
            sem[coordinate] = label; unknown[coordinate] = False
        np.testing.assert_array_equal(majority_fill_native_exact(sem, unknown), majority_fill(sem, unknown))
    with pytest.raises(ValueError, match='rejected'):
        majority_fill_native_exact(np.full((2, 3, 1), 18, np.uint8), np.zeros((2,3,1),bool))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_four_frame_capture_safe_preprocessing_equals_original_and_gradient(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('actual CUDA unavailable')
    from real_motion.local_history_contract import four_frame_motion_inputs, _EXCLUDED, _OFFSET, _SEGMENT, _VALID
    torch.manual_seed(123)
    x = torch.randn(3, FEATURE_DIM, device=device, requires_grad=True)
    tube = torch.randint(18, (3, 6, 4, 4), device=device)
    mask = torch.randint(2, tube.shape, device=device)
    reference = x.clone(); reference[:, _EXCLUDED] = 0.
    offsets = reference[:, _OFFSET].reshape(3,4,2)
    segments = reference[:, _SEGMENT].reshape(3,3,2)
    valid = reference[:, _VALID].reshape(3,4,1)
    velocities = torch.stack((segments[:,0], .5*(segments[:,0]+segments[:,1]),
                             .5*(segments[:,1]+segments[:,2]), segments[:,2]),1)
    old_frame = torch.cat((offsets*valid, velocities*valid, valid),-1)
    actual = four_frame_motion_inputs(x, tube, None, mask)
    for a,b in zip(actual, (reference,tube[:,-4:],old_frame,mask[:,-4:])):
        assert torch.equal(a,b)
    old_gradient = torch.autograd.grad(reference.square().sum()+old_frame.square().sum(), x)[0]
    new_gradient = torch.autograd.grad(actual[0].square().sum()+actual[2].square().sum(), x)[0]
    torch.testing.assert_close(new_gradient, old_gradient, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA graph required')
def test_frozen_graph_exact_no_storage_alias_and_rejects_weight_change():
    torch.manual_seed(9)
    model = LocalSpatialTemporalWorldModelV18SE2(LocalSTWMV17Config(
        history_frames=4, d_model=16, semantic_dim=8, heads=2, blocks=1, decoder_blocks=1)).cuda().eval().requires_grad_(False)
    torch.nn.init.normal_(model.residual_head.weight, std=.02)
    values = [torch.randn(2, FEATURE_DIM), torch.randint(18, (2, 6, 20, 20)),
              torch.randn(2, 6, 2), torch.randn(2, 6, 5), torch.randint(2, (2, 6, 20, 20))]
    record = dict(zip(MOTION_KEYS, values))
    graph = FrozenV18Graph(model, 'cuda', entries=1)
    with pytest.raises(RuntimeError, match='cannot train'):
        graph(record)
    try:
        with torch.no_grad():
            first = graph(record)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                reference = model(*[x.cuda().float() if i in (0, 2, 3) else x.cuda() for i, x in enumerate(values)], return_latents=True)
            assert all(torch.equal(reference[k], first[k]) for k in first)
            saved = {k: v.clone() for k, v in first.items()}
            other = {**record, 'features': record['features']+1}
            second = graph(other)
            assert all(torch.equal(saved[k], first[k]) for k in first)
            assert first['future_transport_queries'].data_ptr() != second['future_transport_queries'].data_ptr()
            bad = {**record, 'local_semantic_tube': record['local_semantic_tube'].clone()}
            bad['local_semantic_tube'][0, -1, 0, 0] = 18
            with pytest.raises(ValueError, match='invalid historical'):
                graph(bad)
            model.residual_head.bias.add_(.01)
            with pytest.raises(RuntimeError, match='changed weights'):
                graph(record)
    finally:
        graph.close()
