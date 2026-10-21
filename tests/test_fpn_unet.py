"""Small contract/gradient/recovery checks; no extra scientific training arms."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.fpn_unet import model, adapter, evaluation, run
from tools.shared_history_experiments.full_gt.engine import semantic_loss


def fixture(device='cpu'):
    from test_source_scene_stage2 import fixture as old
    original, runner, row = old(device)
    body = model.make_model(replace(original.transport.config, decoder_blocks=2),
        feature_width=16, z_bins=4, memory_width=4).to(device)
    return body, runner, row


def test_FPN_learned_levels_shapes_and_validity():
    torch.manual_seed(21)
    encoder = model.FPNEncoder(z_bins=4, width=16)
    sem = torch.randint(0, 18, (4, 32, 40, 4)); known = torch.ones_like(sem, dtype=torch.bool)
    levels = encoder(sem, known)
    assert [tuple(v.shape) for v in levels] == [(4, 16, 16, 20), (4, 16, 8, 10), (4, 16, 4, 5)]
    # Fine P2 must depend on the actual coarse learned stage via top-down fusion.
    levels[0].square().mean().backward()
    assert encoder.down[1][0].weight.grad.abs().sum() > 0
    assert all(not isinstance(v, (torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d)) for v in encoder.modules())
    encoder.eval(); hidden = torch.zeros_like(known)
    torch.testing.assert_close(encoder(sem, hidden)[0], encoder(sem*0, hidden)[0], rtol=0, atol=0)
    with pytest.raises(ValueError): encoder(sem[:, :31], known[:, :31])


def test_UNet_channels_prior_identity_and_full3D_gradients():
    refiner = model.UNetRefiner(feature_width=16, memory_width=16, z_bins=4).train()
    assert refiner.middle[0].in_channels == 24 and refiner.middle[0].out_channels == 16
    prior = torch.rand(6, 18, 16*16*4, requires_grad=True)
    memory = torch.randn(6, 16, 8, 8, 4, requires_grad=True)
    seen = []
    hook = refiner.middle.register_forward_pre_hook(lambda _, a: seen.append(tuple(a[0].shape)))
    output = refiner(memory, prior, range(6), (16, 16, 4)); hook.remove()
    assert seen == [(2, 24, 8, 8, 4)]*3
    torch.testing.assert_close(output, refiner.prior_logits(prior), rtol=0, atol=0)
    # After opening the zero head, both prior and aligned memory receive gradients.
    with torch.no_grad(): refiner.head.weight.normal_(std=.02)
    output = refiner(memory, prior, range(6), (16, 16, 4))
    output.square().mean().backward()
    assert memory.grad.abs().sum() > 0 and prior.grad.abs().sum() > 0
    assert refiner.low[0].weight.grad.abs().sum() > 0


def test_same_UNet_train_eval_and_horizon_grouping():
    refiner = model.UNetRefiner(memory_width=4, z_bins=4)
    with torch.no_grad(): refiner.head.weight.normal_(std=.02)
    memory = torch.randn(6, 4, 4, 4, 4); prior = torch.rand(6, 18, 8*8*4)
    actual = refiner.train()(memory, prior, range(6), (8, 8, 4))
    expected = refiner.eval()(memory, prior, range(6), (8, 8, 4))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    pieces = torch.cat([refiner(memory[h:h+1], prior[h:h+1], [h], (8, 8, 4)) for h in range(6)])
    torch.testing.assert_close(expected, pieces, rtol=1e-5, atol=2e-6)
    with pytest.raises(ValueError): refiner(memory, prior, range(6), (8, 8, 5))


def test_actual_transport_memory_refine_occupancy_backward_and_checkpoint():
    body, runner, row = fixture()
    try:
        # No future labels may be consumed by the actual prediction path.
        poison = row[0], dict(row[1], future_gt_occ=object()), row[2], row[3]
        fw = adapter.forward_window(body, runner.provider, poison, fresh=True)
        assert fw['logits'].shape[0] == 6 and fw['memory'].shape[:2] == (6, 4)
        loss = semantic_loss(fw['logits'], row[1]['future_gt_occ'])[0].mean()
        loss.backward()
        for p in (body.columns.head.weight, body.transport.residual_head.weight):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        # Initial zero residual readout deliberately blocks memory-branch gradients.
        # Open it and motion heads, then test the complete learned feature path.
        body.zero_grad(set_to_none=True)
        with torch.no_grad():
            body.columns.head.weight.normal_(std=.02)
            body.transport.residual_head.weight.normal_(std=.02)
            body.transport.yaw_head.weight.normal_(std=.005)
        fw = adapter.forward_window(body, runner.provider, row, fresh=True)
        semantic_loss(fw['logits'], row[1]['future_gt_occ'])[0].mean().backward()
        for p in (body.transport.history_encoder.down[1][0].weight, body.columns.lift.weight,
                  body.columns.low[0].weight, body.transport.ego_future_proj[-1].weight):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        saved = dict(protocol=run.PROTOCOL, contract=dict(model=body.configs()), models={'gt': body.state_dict()})
        restored = model.checkpoint_model(saved)
        for k, v in body.state_dict().items(): torch.testing.assert_close(restored.state_dict()[k], v, rtol=0, atol=0)
        with pytest.raises(RuntimeError): model.checkpoint_model(dict(saved, protocol='old_ego_query'))
        assert not any('source_query' in n or 'phase' in n for n, _ in body.named_parameters())
    finally: runner.close()


def test_final_evaluation_four_routes_same_integer_resume(tmp_path, monkeypatch):
    body, runner, row = fixture()
    try:
        for token, sem in zip(row[0]['history_tokens'], row[1]['history_occ']):
            p = tmp_path/row[0]['scene_name']/str(token); p.mkdir(parents=True)
            np.savez(p/'labels.npz', semantics=sem)
        calls = []; forward = evaluation.forward_window
        def forecast(*a, **k): calls.append('forecast'); return forward(*a, **k)
        def moving(*a, **k): assert calls == ['forecast', 'forecast']; return []
        monkeypatch.setattr(evaluation, 'forward_window', forecast)
        monkeypatch.setattr(evaluation.gt, 'gt_moving_support_sequence', moving)
        monkeypatch.setattr(evaluation.gt, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
        states = []
        first = evaluation.evaluate(body, runner.provider, SimpleNamespace(nusc=None), [[row]], stc_root=tmp_path,
            save=lambda x: states.append(copy.deepcopy(x)))
        second = evaluation.evaluate(body, runner.provider, SimpleNamespace(nusc=None), [[row]], stc_root=tmp_path, saved=states[-1])
        assert first['keys'] == second['keys'] and len(first['variants']) == 4 and first['windows'] == 1
        for name in evaluation.NAMES:
            for metric in ('mIoU', 'IoU', 'MovingMicro'):
                assert first['variants'][name][metric] == second['variants'][name][metric]
        # Per-class reports can contain NaN for absent classes; compare the
        # underlying resumed integer counts rather than NaN == NaN.
        restored, _ = evaluation.restore(states[-1])
        for name, fields in states[-1]['metrics'].items():
            for field, counts in fields.items(): np.testing.assert_array_equal(getattr(restored[name], field), counts)
        assert calls == ['forecast', 'forecast']
    finally: runner.close()


def test_resume_cursor_and_single_epoch_summary():
    contract = dict(world_size=1)
    saved = dict(protocol=run.PROTOCOL, contract=contract, epoch=0, batch=1, updates=1, rank_rng=[{}])
    assert run.validate_resume(saved, contract, [[1, 2]]) == (0, 1, 1)
    saved.update(epoch=1, batch=0, updates=2)
    assert run.validate_resume(saved, contract, [[1, 2]]) == (1, 0, 2)
    with pytest.raises(RuntimeError): run.validate_resume(dict(saved, protocol='previous'), contract, [[1, 2]])
    with pytest.raises(RuntimeError): run.validate_resume(dict(saved, updates=3), contract, [[1, 2]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_real_cuda_BF16_optimizer_step():
    if not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA required')
    body, runner, row = fixture('cuda')
    try:
        optimizer = torch.optim.AdamW(body.parameters(), lr=3e-4)
        before = body.columns.head.weight.detach().clone()
        fw = adapter.forward_window(body, runner.provider, row, fresh=True)
        loss = semantic_loss(fw['logits'], row[1]['future_gt_occ'])[0].mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(body.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        assert torch.isfinite(loss) and not torch.equal(before, body.columns.head.weight)
    finally: runner.close()


def test_actual_cuda_runner_resume_and_final_eval_once(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA')
    from dataclasses import asdict
    body, runner, row = fixture()
    try:
        names = ('config', 'train_cache', 'dev_cache', 'base_checkpoint', 'train_info', 'dev_info', 'eval_reference', 'population_manifest')
        paths = {k: tmp_path/k for k in names}
        for p in paths.values(): p.write_text('fixture')
        torch.save(dict(model_config=asdict(body.transport.config)), paths['base_checkpoint'])
        stc = tmp_path/'stc'
        for token, sem in zip(row[0]['history_tokens'], row[1]['history_occ']):
            folder = stc/row[0]['scene_name']/str(token); folder.mkdir(parents=True)
            np.savez(folder/'labels.npz', semantics=sem)
        eval_rows = [dict(row[0], t0_token='dev'+str(i)) for i in range(64)]
        keys = [list(k) for k in run.base.record_keys(eval_rows)]
        source = SimpleNamespace(nusc=None)
        class Provider:
            def __init__(self, *args):
                self.pcfg, self.strong = runner.provider.pcfg, runner.provider.strong
                self.device, self.workers, self.joint = args[3], 1, args[5]
            def attach_tracks(self, *a, **k): return {}
            def close_tracks(self): pass
        def caches(provider, *a):
            provider.ccr_history_cache = SimpleNamespace(close=lambda: None)
            provider.ccr_val_history_cache = SimpleNamespace(close=lambda: None)
            return {}
        monkeypatch.setattr(run, 'DirectProvider', Provider)
        monkeypatch.setattr(run, 'make_model', lambda cfg: model.make_model(cfg, feature_width=16, z_bins=4, memory_width=4))
        monkeypatch.setattr(run.base, 'attach_caches', caches)
        monkeypatch.setattr(run.base, 'load_cache', lambda *a: ({}, []))
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: {})
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda *a: runner.provider.pcfg)
        monkeypatch.setattr(run.base, 'NuScenesWindowSource', lambda *a, **k: source)
        monkeypatch.setattr(run.base, 'CachedColumnSource', lambda *a, **k: source)
        monkeypatch.setattr(run, 'plan', lambda *a: ([['TRAIN', 'one']], eval_rows, [[[row[0]], [row[0]]]]))
        monkeypatch.setattr(run, 'batches', lambda provider, source, groups, history, **kw: ( [row] for group in groups ))
        calls = []
        def evaluate(*a, **k):
            calls.append(1)
            return dict(windows=64, keys=keys, variants={r: dict(mIoU=1., IoU=2., MovingMicro=3.) for r in evaluation.NAMES})
        monkeypatch.setattr(run, 'evaluate', evaluate)
        out = tmp_path/'new'
        argv = ['--out-dir', str(out), '--stc-root', str(stc), '--dataroot', str(tmp_path),
            '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
            '--device', 'cuda', '--cpu-workers', '1', '--prefetch-workers', '1']
        for k, p in paths.items(): argv += ['--'+k.replace('_', '-'), str(p)]
        run.main(argv+['--max-updates', '1'])
        partial = torch.load(out/'last.pt', weights_only=False)
        assert partial['updates'] == 1 and partial['epoch'] == 0 and not calls
        run.main(argv+['--resume', str(out/'last.pt')])
        final = torch.load(out/'last.pt', weights_only=False)
        assert final['updates'] == 2 and final['epoch'] == 1 and len(calls) == 1
        model.checkpoint_model(final)
        run.main(argv+['--resume', str(out/'last.pt')])
        again = torch.load(out/'last.pt', weights_only=False)
        assert len(calls) == 1
        for k, v in final['models']['gt'].items(): torch.testing.assert_close(again['models']['gt'][k], v, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match='NEW output'): run.main(argv)
    finally: runner.close()
