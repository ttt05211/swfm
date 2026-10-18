"""Joint encoder/AdaGN experiment: common initialization, masks, full6 gradients, recovery."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_scene_integrated.model import make_model, checkpoint_model, common_key
from tools.source_scene_kta_query.adapter import forward_window
from tools.source_scene_kta_query import evaluation
from tools.source_scene_integrated import report

KW = dict(feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32)


def fixture(device='cpu'):
    from test_source_scene_stage2 import fixture as old
    previous, runner, row = old(device)
    return make_model(previous.transport.config, **KW).to(device), runner, row


def test_common_fresh_weights_post_rng_and_architecture():
    from tools.source_scene_kta_query.model import make_model as old
    model, runner, row = fixture()
    try:
        cfg = model.transport.config
        torch.manual_seed(21); control = old(cfg, **KW); expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(21); model = make_model(cfg, **KW)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        previous = control.state_dict()
        for k, v in model.state_dict().items():
            mapped = common_key(k)
            if mapped in previous and v.shape == previous[mapped].shape:
                torch.testing.assert_close(v, previous[mapped], rtol=0, atol=0)
        assert not hasattr(model.transport, 'readers') and not hasattr(model.transport, 'temporal')
        assert len(model.transport.encoder_blocks) == 2
        assert len(model.transport.decoder) == len(control.transport.decoder)
        assert model.configs()['kta_conditioning'] is True
        assert model.configs()['encoder']['global_attention'] is False
        for block in model.columns.blocks:
            assert not block.norm.affine
            assert (block.affine.weight == 0).all() and (block.affine.bias == 0).all()
        with torch.no_grad():
            fw = forward_window(model, runner.provider, row, fresh=True)
            torch.testing.assert_close(fw['logits'], model.columns.prior_logits(fw['prior']), rtol=0, atol=0)
    finally: runner.close()


def test_AdaGN_only_modulates_residual_and_zero_projection():
    from tools.source_scene_integrated.refiner import AdaGNBlock
    block = AdaGNBlock(8, 2)
    value = torch.randn(6, 8, 4, 4, 4); embedding = torch.randn(6, 8)
    expected = value+block.convs(block.norm(value))
    torch.testing.assert_close(block(value, embedding), expected, rtol=0, atol=0)
    block.affine.weight.data.normal_()
    block.affine.bias.data.normal_()
    block.convs[-1].weight.data.zero_(); block.convs[-1].bias.data.zero_()
    torch.testing.assert_close(block(value, embedding), value, rtol=0, atol=0)


def test_AdaGN_condition_gradient_opens_after_first_projection_update():
    from tools.source_scene_integrated.refiner import AdaGNBlock
    block = AdaGNBlock(8, 1)
    value = torch.randn(6, 8, 4, 4, 4)
    embedding = torch.randn(6, 8, requires_grad=True)
    optimizer = torch.optim.SGD(block.parameters(), lr=.1)
    block(value, embedding).square().mean().backward()
    assert block.affine.weight.grad.abs().sum() > 0
    assert (embedding.grad == 0).all()  # Expected with zero condition projection.
    optimizer.step(); optimizer.zero_grad(); embedding.grad = None
    block(value, embedding).square().mean().backward()
    assert torch.isfinite(embedding.grad).all() and embedding.grad.abs().sum() > 0


def test_full_cuda_integrated_runner_stop_resume_and_control_reused(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA')
    import json
    from dataclasses import asdict
    from tools.source_scene_integrated import run
    from tools.source_scene_kta_query.run import PROTOCOL as PREVIOUS
    from tools.source_scene_kta_query.model import make_model as old_model
    model, runner, row = fixture()
    try:
        paths = {k: tmp_path/k for k in ('config', 'train_cache', 'dev_cache', 'base_checkpoint', 'train_info', 'dev_info', 'eval_reference', 'population_manifest')}
        for path in paths.values(): path.write_text('fixture')
        torch.save(dict(model_config=asdict(model.transport.config)), paths['base_checkpoint'])
        stc = tmp_path/'stc'
        for token, sem in zip(row[0]['history_tokens'], row[1]['history_occ']):
            directory = stc/row[0]['scene_name']/str(token); directory.mkdir(parents=True)
            np.savez(directory/'labels.npz', semantics=sem)
        inventory = sorted((str((stc/row[0]['scene_name']/str(t)/'labels.npz').resolve()),
                            (stc/row[0]['scene_name']/str(t)/'labels.npz').stat().st_size,
                            (stc/row[0]['scene_name']/str(t)/'labels.npz').stat().st_mtime_ns) for t in row[0]['history_tokens'])
        eval_rows = [dict(row[0], t0_token='dev'+str(i)) for i in range(64)]
        kwargs = dict(feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32)
        control_model = old_model(model.transport.config, **kwargs)
        control = dict(status='complete', updates=2, contract=dict(protocol=PREVIOUS, arm='kta_query_only', registration='none',
            source_policy='shared_pinned', model=control_model.configs(), world_size=1,
            parameters=sum(p.numel() for p in control_model.parameters()),
            train_keys=[['T', 't']], eval_keys=run.base.record_keys(eval_rows), epochs=1, total_updates=2, epochs_updates=[2],
            seed=20261011, objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.),
            learning_rates=[.0005, .0003], schedule='whole_1_epoch_cosine_floor0.01', known='TRAIN_OCC_STC_dense_valid',
            precision='original_BF16_geometry_FP64_H6',
            data={k: run.base.sha256(paths[k]) for k in ('config', 'train_cache', 'dev_cache', 'base_checkpoint', 'train_info', 'dev_info')},
            overrides=[], STC_inventory=inventory, implementation={}),
            evaluations={'1': dict(windows=64, keys=run.base.record_keys(eval_rows), variants={
                r: dict(mIoU=10., IoU=20., MovingMicro=5.) for r in ('occ_gt_transport', 'occ_gt_semantic_refine', 'stc_gt_transport', 'stc_gt_semantic_refine')})})
        reference = tmp_path/'reader'; reference.mkdir(); (reference/'results.json').write_text(json.dumps(control))
        control = json.loads((reference/'results.json').read_text())
        reference_sha = run.base.sha256(reference/'results.json')
        class Provider:
            def __init__(self, *args):
                self.pcfg, self.strong = runner.provider.pcfg, runner.provider.strong
                self.device = args[3]; self.workers = 1; self.joint = args[5]
            def attach_tracks(self, root, **kwargs):
                assert root.parent == tmp_path/'new'; return dict(registration_computed=False)
            def close_tracks(self): pass
            def load_raw_columns(self, *args, **kwargs): return row[1]
        tokens = {str(t): int(i*500000) for i, t in enumerate(row[0]['history_tokens'])}
        source = SimpleNamespace(nusc=SimpleNamespace(get=lambda _, t: {'timestamp': tokens[str(t)]}))
        def caches(provider, *args):
            provider.ccr_history_cache = SimpleNamespace(close=lambda: None)
            provider.ccr_val_history_cache = SimpleNamespace(close=lambda: None)
            return {}
        monkeypatch.setattr(run, 'DirectProvider', Provider)
        monkeypatch.setattr(run, 'make_model', lambda cfg: make_model(cfg, **kwargs))
        monkeypatch.setattr(run.base, 'attach_caches', caches)
        monkeypatch.setattr(run.base, 'load_cache', lambda *a: ({}, []))
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: {})
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda *a: runner.provider.pcfg)
        monkeypatch.setattr(run.base, 'NuScenesWindowSource', lambda *a, **k: source)
        monkeypatch.setattr(run.base, 'CachedColumnSource', lambda *a, **k: source)
        # Plan has one epoch with two global groups.
        monkeypatch.setattr(run, 'plan', lambda *a: ([['T', 't']], eval_rows, [[[row[0]], [row[0]]]]))
        monkeypatch.setattr(run, 'evaluate', lambda *a, **k: copy.deepcopy(control['evaluations']['1']))
        out = tmp_path/'new'
        argv = ['--out-dir', str(out), '--reference-dir', str(reference), '--stc-root', str(stc),
                '--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
                '--device', 'cuda', '--cpu-workers', '1', '--prefetch-workers', '1']
        for k, path in paths.items(): argv += ['--'+k.replace('_', '-'), str(path)]
        run.main(argv+['--max-updates', '1'])
        partial = json.loads((out/'results.json').read_text()); assert partial['updates'] == 1 and partial['epoch'] == 0
        run.main(argv+['--resume', str(out/'last.pt')])
        final = json.loads((out/'results.json').read_text()); assert final['status'] == 'complete' and final['updates'] == 2
        assert (out/'comparison.txt').is_file() and reference_sha == run.base.sha256(reference/'results.json')
        checkpoint_model(torch.load(out/'last.pt', weights_only=False))
        with pytest.raises(RuntimeError, match='NEW output'): run.main(argv)
    finally: runner.close()


def test_reject_wrong_control_and_ambiguous_directory(tmp_path):
    from tools.source_scene_kta_query.run import PROTOCOL
    model, runner, _ = fixture()
    try:
        base = dict(status='complete', updates=1, contract=dict(protocol=PROTOCOL, epochs=1,
            model={**model.configs(), 'variant': 'source_scene_stage2_kta_query', 'kta_conditioning': True},
            registration='none', source_policy='shared_pinned', known='TRAIN_OCC_STC_dense_valid',
            total_updates=1, eval_keys=[['V', str(i)] for i in range(64)]),
            evaluations={'1': dict(windows=64, keys=[['V', str(i)] for i in range(64)])})
        for name in ('one', 'two'):
            folder = tmp_path/('scene_stage2_kta_query_1ep_'+name); folder.mkdir()
            (folder/'results.json').write_text(json.dumps(base))
        with pytest.raises(RuntimeError, match='candidates'): report.select_reference(tmp_path)
        assert report.select_reference(tmp_path, tmp_path/'scene_stage2_kta_query_1ep_one') == (tmp_path/'scene_stage2_kta_query_1ep_one').resolve()
        bad = copy.deepcopy(base); bad['contract']['model']['kta_conditioning'] = False
        (tmp_path/'scene_stage2_kta_query_1ep_one/results.json').write_text(json.dumps(bad))
        with pytest.raises(RuntimeError): report.reference(tmp_path/'scene_stage2_kta_query_1ep_one')
    finally: runner.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_query_depends_on_KTA_history_unchanged_and_full6_gradient_no_ICP(device, monkeypatch):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    from real_motion import shared_source_track
    from tools.shared_history_experiments.full_gt.engine import semantic_loss
    model, runner, row = fixture(device)
    try:
        monkeypatch.setattr(shared_source_track, 'register_history_shape', lambda *a, **k: pytest.fail('ICP called'))
        with torch.no_grad(): model.columns.readout[-1].weight.normal_(std=.1)
        record = dict(row[0], kta_displacement_xy_m=row[0]['kta_displacement_xy_m']+.2,
                      anchors_xy_t0_m=row[0]['anchors_xy_t0_m']+.2)
        row = record, *row[1:]
        fw = forward_window(model, runner.provider, row)
        loss = semantic_loss(fw['logits'], row[1]['future_gt_occ'])[0].mean(); loss.backward()
        for p in (model.transport.kta_future_proj[0].weight, model.transport.residual_head.weight,
                  model.transport.history_yaw.weight, model.transport.encoder_blocks[0].scene.offsets.weight, model.transport.scene_global.weight,
                  model.transport.encoder_blocks[0].temporal_attention.in_proj_weight,
                  model.columns.blocks[0].affine.weight):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        reads = {k: torch.as_tensor(v, device=device) for k, v in row[3][0]['shared_reads'].items()}
        times = torch.as_tensor(row[2][0][:, -1], device=device)
        kta = torch.as_tensor(record['kta_displacement_xy_m'], device=device)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            a = model.transport(fw['field'], reads, times, object(), kta)
            b = model.transport(fw['field'], reads, times, object(), kta+1.)
        assert not torch.equal(a['future_transport_queries'], b['future_transport_queries'])
        for key in ('history_source_context', 'history_yaw_to_t0_rad'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert (a['history_yaw_to_t0_rad'][:, -1] == 0).all()
        with pytest.raises(ValueError, match='KTA'): model.transport(fw['field'], reads, times, None, None)
        # Existing future labels/commands cannot enter the query or encoder.
        with torch.no_grad():
            poisoned = row[0], dict(row[1], future_gt_occ=object(), future_poses=object()), row[2], row[3]
            fresh = forward_window(model, runner.provider, poisoned, poses=row[1]['future_poses'], fresh=True)
            assert fresh['logits'].shape[0] == 6
    finally: runner.close()


def test_empty_sources_checkpoint_isolation_and_resume():
    from tools.source_scene_integrated.run import PROTOCOL, validate_resume
    model, runner, row = fixture()
    try:
        saved = dict(protocol=PROTOCOL, contract=dict(model=model.configs()), models={'gt': model.state_dict()})
        loaded = checkpoint_model(saved)
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, loaded.state_dict()[k], rtol=0, atol=0)
        with pytest.raises(RuntimeError): checkpoint_model(dict(saved, protocol='stage2_no_query'))
        reads = {k: torch.as_tensor(v) for k, v in row[3][0]['shared_reads'].items()}
        for k in ('reference_valid', 'reference_grid', 'reference_xy_t0'): reads[k] = reads[k][:0]
        out = model.transport(torch.zeros(4, 16, 4, 4), reads, torch.arange(4), None, torch.empty(0, 6, 2))
        assert out['history_yaw_to_t0_rad'].shape == (0, 4)
        c = dict(world_size=2); state = dict(protocol=PROTOCOL, contract=c, epoch=0, batch=1, updates=1, rank_rng=[{}, {}])
        assert validate_resume(state, c, [[1, 2]]) == (0, 1, 1)
        for bad in (dict(state, updates=2), dict(state, protocol='old')):
            with pytest.raises(RuntimeError): validate_resume(bad, c, [[1, 2]])
    finally: runner.close()


def test_real_eval_learned_KTA_integer_resume_and_no_ICP(tmp_path, monkeypatch):
    from real_motion import shared_source_track
    model, runner, row = fixture()
    try:
        for token, sem in zip(row[0]['history_tokens'], row[1]['history_occ']):
            folder = tmp_path/row[0]['scene_name']/str(token); folder.mkdir(parents=True)
            np.savez(folder/'labels.npz', semantics=sem)
        monkeypatch.setattr(shared_source_track, 'register_history_shape', lambda *a, **k: pytest.fail('ICP called'))
        monkeypatch.setattr(evaluation.gt, 'gt_moving_support_sequence', lambda *a, **kw: [])
        monkeypatch.setattr(evaluation.gt, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
        monkeypatch.setattr(evaluation, 'box_motion_rows', lambda *a: [])
        snapshots = []
        kwargs = dict(stc_root=tmp_path, save=lambda x: snapshots.append(copy.deepcopy(x)))
        first = evaluation.evaluate(model, runner.provider, SimpleNamespace(nusc=None), [[row]], **kwargs)
        state = snapshots[-1]
        second = evaluation.evaluate(model, runner.provider, SimpleNamespace(nusc=None), [[row]], saved=state, **kwargs)
        assert snapshots[-1] == state
        assert first['windows'] == 1 and len(first['variants']) == len(first['kta_variants']) == 4
        for r in first['variants']:
            for metric in ('mIoU', 'IoU', 'MovingMicro'):
                assert first['variants'][r][metric] == second['variants'][r][metric]
                assert first['variants'][r][metric] == first['kta_variants'][r][metric]  # Fresh zero future heads.
        with pytest.raises(RuntimeError): evaluation.restore(dict(state, protocol='old'))
    finally: runner.close()

def test_mask_after_each_residual_full_four_history_decoder_context(monkeypatch):
    model, runner, row = fixture()
    try:
        with torch.no_grad():
            fw = forward_window(model, runner.provider, row, fresh=True)
        reads = {k: torch.as_tensor(v) for k, v in row[3][0]['shared_reads'].items()}
        reads['reference_valid'] = reads['reference_valid'].clone()
        reads['reference_valid'][:, 0] = False
        valid = reads['reference_valid']
        times = torch.as_tensor(row[2][0][:, -1])
        kta = torch.as_tensor(row[0]['kta_displacement_xy_m'])
        seen = []
        hooks = [block.register_forward_hook(lambda _, args, out: seen.append(out.detach())) for block in model.transport.encoder_blocks]
        def capture(_, args): assert args[1].shape[1] == 4
        hooks.append(model.transport.decoder[0].register_forward_pre_hook(capture))
        with torch.no_grad(): output = model.transport(fw['field'], reads, times, None, kta)
        for hook in hooks: hook.remove()
        assert len(seen) == 2
        for tokens in seen:
            assert (tokens[~valid] == 0).all() and torch.isfinite(tokens).all()
        assert (output['history_yaw_to_t0_rad'][:, 0] == 0).all()
        assert (output['history_yaw_to_t0_rad'][:, -1] == 0).all()
    finally: runner.close()


def test_scene_sampling_retains_original_multiscale_local_math():
    from tools.source_scene_stage1.model import SceneReader, ReaderConfig
    from tools.source_scene_integrated.attention import MultiScaleSceneAttention
    model, runner, row = fixture()
    try:
        cfg = model.transport.config
        old = SceneReader(cfg.d_model, cfg.heads, 4).eval()
        new = MultiScaleSceneAttention(cfg.d_model, cfg.heads, 4).eval()
        state = old.state_dict()
        new.load_state_dict({k: state[k.replace('output.', 'local_output.')] for k in new.state_dict()})
        reads = {k: torch.as_tensor(v) for k, v in row[3][0]['shared_reads'].items()}
        valid = reads['reference_valid'].bool()
        q = torch.randn(4, len(valid), cfg.d_model)
        pyramid = [torch.randn(4, cfg.d_model, side, side) for side in (4, 2, 1)]
        seen = []
        hook = old.local_output.register_forward_hook(lambda _, args, out: seen.append(out.detach()))
        with torch.no_grad():
            content = pyramid[-1].flatten(2).transpose(1, 2)
            old(q, pyramid, content, torch.zeros_like(content), reads, valid)
            actual = new(old.norm(q), pyramid, reads, valid)
        hook.remove()
        torch.testing.assert_close(actual, seen[0], rtol=0, atol=0)
    finally: runner.close()


def test_encoder_block_does_not_mix_sources():
    model, runner, row = fixture()
    try:
        reads = {k: torch.as_tensor(v) for k, v in row[3][0]['shared_reads'].items()}
        for key in ('reference_valid', 'reference_grid', 'reference_xy_t0'):
            reads[key] = reads[key][:1].expand(2, *reads[key].shape[1:]).clone()
        valid = reads['reference_valid'].bool()
        tokens = torch.randn(2, 4, model.transport.config.d_model)
        other = tokens.clone(); other[1] = torch.randn_like(other[1])*10
        pyramid = [torch.randn(4, model.transport.config.d_model, side, side) for side in (4, 2, 1)]
        block = model.transport.encoder_blocks[0].eval()
        with torch.no_grad():
            a = block(tokens, pyramid, reads, valid)
            b = block(other, pyramid, reads, valid)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    finally: runner.close()
