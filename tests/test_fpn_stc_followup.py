import copy
import numpy as np
import pytest
import torch
from tools.fpn_stc_followup import core
from tools.fpn_stc_local import calibration as cal


def test_semantic_bias_preserves_binary_support_and_input_logits():
    torch.manual_seed(21); z = torch.randn(6, 18, 32); before = z.clone()
    values = core.readouts(z, (4, 4, 2), dict(identity=np.zeros(17), changed=np.arange(17)/8))
    np.testing.assert_array_equal(values['identity'], values['free_minus1'])
    np.testing.assert_array_equal(values['changed'] != 17, values['free_minus1'] != 17)
    torch.testing.assert_close(z, before, rtol=0, atol=0)
    gt = np.random.default_rng(21).integers(0, 18, (3, 4, 4, 2), dtype=np.uint8)
    counts = cal.counts(values, gt, names=list(values))
    assert cal.report(counts[1])['average']['IoU'] == cal.report(counts[3])['average']['IoU']


def test_stratified_fit_samples_recover_support_mass_and_are_deterministic():
    torch.manual_seed(21); z = torch.randn(6, 18, 128)
    gt = np.random.default_rng(21).integers(0, 18, (3, 8, 8, 2), dtype=np.uint8)
    samples = core.sample_fit(z, gt, seed=21, cap=2)
    other = core.sample_fit(z, gt, seed=21, cap=2)
    for k in samples: np.testing.assert_array_equal(samples[k], other[k])
    gate = z[[1, 3, 5], :17].max(1).values >= z[[1, 3, 5], 17]-1
    support = gate.numpy() | (gt.reshape(3, -1) != 17)
    assert samples['logits'].shape[1] == 17
    assert samples['weights'].sum() == pytest.approx(support.sum())
    for cls in range(18):
        assert samples['weights'][samples['labels'] == cls].sum() == pytest.approx(((gt.reshape(3, -1) == cls) & support).sum())


def test_bias_fit_is_small_final_fixed_objective_and_samples_unchanged():
    rng = np.random.default_rng(21); labels = np.tile([0, 1, 17], 100)
    z = np.full((300, 17), -10., np.float32); z[:, :2] = rng.normal(size=(300, 2)); z[:, 0] += 2
    sample = dict(logits=z, labels=labels, occupied=np.ones(300, bool), weights=np.ones(300, np.float32))
    before = copy.deepcopy(sample)
    for objective in ('ce', 'soft_iou'):
        bias, trace = core.fit_bias(sample, objective, steps=20)
        assert len(bias) == 17 and abs(sum(bias)) < 1e-5
        assert trace[-1]['step'] == 20 and trace[-1]['loss'] < trace[0]['loss']
    for k in before: np.testing.assert_array_equal(sample[k], before[k])


def test_real_memory_temperature_identity_read_only_and_hook_cleanup():
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
        history = copy.deepcopy(prepared[3][0]['memory_reads'])
        fw = p.forward(prepared, poses=row[1]['future_poses'])
        identity = core.memory_logits(body, p.provider, prepared, fw, 1.)
        torch.testing.assert_close(identity, fw['logits'], rtol=0, atol=0)
        for t in (2., float('inf')):
            changed = core.memory_logits(body, p.provider, prepared, fw, t)
            assert torch.isfinite(changed).all() and changed.shape == identity.shape
        probability = core.temporal_prior_logits(body, p.provider, prepared, fw)
        assert torch.isfinite(probability).all() and probability.shape == identity.shape
        with pytest.raises(RuntimeError):
            with core.importance_temperature(body.columns, 2.): raise RuntimeError('stop')
        assert not body.columns.importance._forward_hooks
        for k, v in before.items(): torch.testing.assert_close(body.state_dict()[k], v, rtol=0, atol=0)
        for k, v in history.items(): np.testing.assert_array_equal(prepared[3][0]['memory_reads'][k], v)
    finally: p.close()
