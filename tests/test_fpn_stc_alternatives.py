import copy
import numpy as np
import pytest
import torch
from tools.fpn_stc_alternatives import core


def test_embedding_conditional_expectation_fallback_and_no_weight_changes():
    encoder = torch.nn.Module(); encoder.semantic = torch.nn.Embedding(19, 4)
    original = encoder.semantic.weight.detach().clone(); counts = np.eye(18, dtype=np.int64)
    torch.testing.assert_close(core.embedding_table(counts, original, 1.), original, rtol=0, atol=0)
    counts[0] = 0; counts[1] = 0; counts[1, 3] = 8
    table = core.embedding_table(counts, original, .5)
    torch.testing.assert_close(table[0], original[0], rtol=0, atol=0)
    torch.testing.assert_close(table[1], (original[1]+original[3])/2)
    with pytest.raises(RuntimeError):
        with core.embedding_override(encoder, table):
            torch.testing.assert_close(encoder.semantic(torch.tensor([1, 18])), table[[1, 18]])
            raise RuntimeError('stop')
    assert not encoder.semantic._forward_hooks
    torch.testing.assert_close(encoder.semantic.weight, original, rtol=0, atol=0)


def test_tiled_blend_preserves_pointwise_logits_and_covers_edges():
    shape = (16, 16, 4); prior = torch.rand(6, 18, int(np.prod(shape))); memory = torch.rand(6, 4, 8, 8, 4)
    class Columns:
        def __call__(self, m, p, hs, shape): return p*2-1
    actual = core.tiled_logits(Columns(), memory, prior, shape, side=12)
    torch.testing.assert_close(actual, prior*2-1, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(core.tiled_logits(Columns(), memory, prior, shape, side=16), prior*2-1, rtol=0, atol=0)


def test_head_sampler_and_fixed_162_fit_preserve_source_samples():
    rng = np.random.default_rng(21); z = torch.randn(6, 18, 128)
    activations = [torch.randn(1, 8, 128) for _ in range(6)]
    target = rng.integers(0, 18, (3, 8, 8, 2), dtype=np.uint8)
    samples = core.sample_head(z, activations, target, seed=21, cap=2)
    assert samples['features'].shape[1] == 8 and samples['logits'].shape[1] == 18
    assert samples['weights'].sum() == pytest.approx(target.size)
    before = copy.deepcopy(samples); fit = core.fit_head(samples, steps=20)
    assert fit['parameters'] == 162 and fit['steps'] == 20 and len(fit['weight']) == 18
    assert fit['trace'][-1]['loss'] < fit['trace'][0]['loss']
    conditional = core.fit_head(samples, steps=20, conditional=True)
    assert conditional['parameters'] == 153 and conditional['conditional']
    assert conditional['bias'][-1] == 0. and conditional['weight'][-1] == [0.]*8
    assert conditional['trace'][-1]['loss'] < conditional['trace'][0]['loss']
    for k in before: np.testing.assert_array_equal(samples[k], before[k])


def test_real_head_capture_zero_patch_and_tiled_identity_source_read_only():
    from test_fpn_unet import fixture
    from tools.fpn_unet_all.run import AllPredictor
    body, old, row = fixture(); pcfg = old.provider.pcfg; old.close()
    body.eval().requires_grad_(False)
    with torch.no_grad(): body.columns.head.weight.normal_(std=.02)
    p = AllPredictor(body, pcfg, 'cpu', workers=1); p.kind = 'stc'
    before = {k: v.clone() for k, v in body.state_dict().items()}
    try:
        raw = {k: row[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
        prepared = p.prepare(row[0], raw, times=np.arange(4)*.5)
        with core.capture_readout(body.columns) as captured: fw = p.forward(prepared, poses=row[1]['future_poses'])
        assert torch.cat(captured).shape == (6, 8, int(np.prod(fw['shape'])))
        patch = dict(weight=np.zeros((18, 8)), bias=np.zeros(18))
        with core.readout_patch(body.columns, patch): changed = body.columns(fw['memory'], fw['prior'], range(6), fw['shape'])
        torch.testing.assert_close(changed, fw['logits'], rtol=0, atol=0)
        weight = torch.randn(18, 8)*.1; bias = torch.randn(18)*.1
        with core.readout_patch(body.columns, dict(weight=weight, bias=bias)):
            changed = body.columns(fw['memory'], fw['prior'], range(6), fw['shape'])
        expected = fw['logits']+torch.einsum('oc,hcv->hov', weight, torch.cat(captured))+bias[None, :, None]
        torch.testing.assert_close(changed, expected, rtol=2e-5, atol=2e-6)
        single = core.tiled_logits(body.columns, fw['memory'], fw['prior'], fw['shape'], side=fw['shape'][0])
        torch.testing.assert_close(single, fw['logits'], rtol=0, atol=0)
        tiled = core.tiled_logits(body.columns, fw['memory'], fw['prior'], fw['shape'], side=fw['shape'][0]//2)
        assert torch.isfinite(tiled).all() and tiled.shape == single.shape
        assert not body.columns.head._forward_hooks and not body.columns.head._forward_pre_hooks
        for k, v in before.items(): torch.testing.assert_close(body.state_dict()[k], v, rtol=0, atol=0)
    finally: p.close()
