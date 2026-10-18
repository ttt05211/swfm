"""No ICP path, pinned identity/KTA parity, rotation geometry and occupancy gradients."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_scene_stage2 import track
from tools.source_scene_stage2.model import make_model, checkpoint_model
from tools.source_scene_stage2.adapter import DirectPredictor, forward_window
from tools.source_scene_stage2.geometry import direct_move
from tools.source_scene_stage2.run import PROTOCOL, validate_resume


def fixture(device='cpu'):
    from test_source_scene_stage1 import fixture as old
    base, provider, original = old()
    model = make_model(base.transport.config, feature_width=16, z_bins=4, memory_width=4,
                       width=8, chunk_size=32).to(device)
    runner = DirectPredictor(model, provider.pcfg, device, workers=1)
    raw = {k: original[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
    raw['history_occ'] = np.full_like(raw['history_occ'], 17)
    for f in range(4): raw['history_occ'][f, 1+f//2:4+f//2, 2:5, 1] = 4
    raw['history_observed'] = np.ones_like(raw['history_occ'], dtype=bool)
    row = runner.prepare(original[0], raw, times=np.arange(4)*.5)
    assert len(row[0]['source_class_id']) > 0
    poses = original[1]['future_poses']
    # Train consumes the original immutable GT-conditioned Strong background,
    # whereas live evaluation computes it afresh. Build that reference here.
    from tools.shared_source_track.adapter import backgrounds
    bg = backgrounds(row[1], row[3][0]['state'], poses, runner.provider, row[1]['_shared_source_track'])
    raw = dict(row[1], future_poses=poses, future_gt_occ=original[1]['future_gt_occ'],
        _column_causal_preparation=dict(row[1]['_column_causal_preparation'],
            prepared_state=dict(row[3][0]['state'], column_backgrounds=bg)))
    return model, runner, (row[0], raw, row[2], row[3])


def test_identity_and_kta_exact_no_registration_even_if_old_icp_rejects(monkeypatch):
    from test_source_scene_stage1 import fixture as old
    from real_motion import shared_source_track as previous
    from real_motion.source_evidence_audit import Registration
    base, provider, row = old()
    from real_motion.strong_w2det import StrongW2DetConfig
    provider.strong = StrongW2DetConfig(free_label=17)
    raw = {k: row[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
    raw['history_occ'] = np.array(raw['history_occ'], copy=True)
    raw['history_occ'][:, 2:5, 2:5, 1] = 4
    frames, actual = track.from_raw(raw, provider, np.arange(4)*.5)
    monkeypatch.setattr(previous, 'register_history_shape', lambda points, ref, **kwargs:
        Registration(points, False, 0., 10., 0.))
    control = previous.build_track(frames, raw['history_poses'], np.arange(4)*.5, provider.pcfg.grid, provider.strong)
    for k in ('links', 'centers', 'track_valid', 'times'): np.testing.assert_array_equal(getattr(actual, k), getattr(control, k))
    a = previous.kta_and_anchors(frames[-1], actual, raw['history_poses'][-1])
    b = previous.kta_and_anchors(frames[-1], control, raw['history_poses'][-1])
    for x, y in zip(a, b): np.testing.assert_array_equal(x, y)
    assert actual.track_valid[:, :3].any() and not control.registration_valid[:, :3].any()
    monkeypatch.setattr(previous, 'register_history_shape', lambda *a, **k: pytest.fail('old ICP called'))
    track.from_raw(raw, provider, np.arange(4)*.5)
    assert not hasattr(actual, 'registrations')


def test_direct_rotation_not_center_only_and_yaw_gradients():
    points = torch.tensor([[1., 0., 3.]], dtype=torch.float64)
    source = torch.zeros_like(points); dest = torch.tensor([[5., 6., 7.]], dtype=torch.float64)
    hy = torch.tensor([np.pi/2], dtype=torch.float64, requires_grad=True)
    fy = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
    result = direct_move(points, source, dest, hy, fy)
    torch.testing.assert_close(result, torch.tensor([[5., 7., 3.]], dtype=torch.float64))
    result[:, 0].sum().backward()
    assert hy.grad.abs().sum() > 0 and fy.grad.abs().sum() > 0


def test_shared_fresh_modules_and_query_independent_of_KTA_classes():
    from tools.source_scene_stage1.model import make_model as old
    model, runner, row = fixture()
    try:
        cfg = model.transport.config; kwargs = dict(feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32)
        torch.manual_seed(21); control = old('reader', cfg, **kwargs)
        torch.manual_seed(21); candidate = make_model(cfg, **kwargs)
        for key, value in candidate.state_dict().items():
            if key in control.state_dict(): torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
        assert not hasattr(candidate.transport, 'kta_future_proj')
        fw = forward_window(model.eval(), runner.provider, row)
        reads = {k: torch.as_tensor(v) for k, v in row[3][0]['shared_reads'].items()}
        times = torch.as_tensor(row[2][0][:, -1])
        a = model.transport(fw['field'], reads, times, object(), object())
        b = model.transport(fw['field'], reads, times, torch.ones(500), torch.randn(100, 3))
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert a['history_yaw_to_t0_rad'].count_nonzero() == 0
        assert a['residual_xy_m'].count_nonzero() == 0
        assert torch.equal(fw['logits'].argmax(1), fw['prior'].argmax(1))
        poison = dict(row[1], future_gt_occ=object(), future_poses=object())
        h = runner.history.build(poison)
        np.testing.assert_array_equal(h[0]['shared_reads']['reference_valid'], row[1]['_shared_source_track'].track_valid)
        assert 'registrations' not in h[0]
        # Unlike ICP points, these coordinates stay in their original frame's
        # world position until the differentiable memory transport.
        np.testing.assert_array_equal(h[0]['memory_reads']['centers_world'], row[1]['_shared_source_track'].centers)
    finally: runner.close()


def test_evaluation_same_occ_stc_integer_resume_and_no_ICP(tmp_path, monkeypatch):
    from tools.source_scene_stage2 import evaluation
    from real_motion import shared_source_track
    model, runner, row = fixture()
    try:
        for token, sem in zip(row[0]['history_tokens'], row[1]['history_occ']):
            folder = tmp_path/row[0]['scene_name']/str(token); folder.mkdir(parents=True)
            np.savez(folder/'labels.npz', semantics=sem)
        monkeypatch.setattr(shared_source_track, 'register_history_shape', lambda *a, **k: pytest.fail('ICP called'))
        monkeypatch.setattr(evaluation.gt, 'gt_moving_support_sequence', lambda *a, **k: [])
        monkeypatch.setattr(evaluation.gt, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
        saved = []
        args = dict(stc_root=tmp_path, save=lambda s: saved.append(copy.deepcopy(s)))
        report = evaluation.evaluate(model, runner.provider, SimpleNamespace(nusc=None), [[row]], **args)
        first = copy.deepcopy(saved[-1])
        again = evaluation.evaluate(model, runner.provider, SimpleNamespace(nusc=None), [[row]], saved=first, **args)
        assert saved[-1] == first
        for route in report['variants']:
            for key in ('mIoU', 'IoU', 'MovingMicro'):
                assert report['variants'][route][key] == again['variants'][route][key]
        assert report['windows'] == 1 and set(report['variants']) == {'occ_gt_transport', 'occ_gt_semantic_refine', 'stc_gt_transport', 'stc_gt_semantic_refine'}
        with pytest.raises(RuntimeError): evaluation.evaluate(model, runner.provider, None, [], saved={'protocol': 'stage1'}, **args)
    finally: runner.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_full6_occupancy_gradients_and_fresh_cached_entrance(device, monkeypatch):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    from tools.shared_history_experiments.full_gt.engine import semantic_loss
    model, runner, row = fixture(device)
    try:
        model.eval()
        # Nonzero residual makes the memory path visible immediately, without
        # adding an alignment target/loss or training on any VAL sample.
        with torch.no_grad(): model.columns.readout[-1].weight.normal_(std=.1)
        fw = forward_window(model, runner.provider, row)
        loss = semantic_loss(fw['logits'], row[1]['future_gt_occ'])[0].mean(); loss.backward()
        for p in (model.transport.history_yaw.weight, model.transport.residual_head.weight,
                  model.transport.yaw_head.weight, model.transport.readers[0].offsets.weight,
                  model.transport.history_encoder.stem[0].weight if hasattr(model.transport.history_encoder, 'stem') else next(model.transport.history_encoder.parameters())):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        assert (fw['output']['history_yaw_to_t0_rad'][:, -1] == 0).all()
        with torch.no_grad():
            cached = forward_window(model, runner.provider, row)
            fresh = forward_window(model, runner.provider, row, fresh=True)
            torch.testing.assert_close(cached['prior'], fresh['prior'], rtol=0, atol=0)
        # Validity masks, not arbitrary numeric features: missing history does
        # not acquire a nonzero yaw from the temporal model.
        reads = {k: torch.as_tensor(v, device=device) for k, v in row[3][0]['shared_reads'].items()}
        reads['reference_valid'][:, 0] = False
        field = fw['field'].detach(); times = torch.as_tensor(row[2][0][:, -1], device=device)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            result = model.transport(field, reads, times, None, None)
        assert (result['history_yaw_to_t0_rad'][:, 0] == 0).all()
    finally: runner.close()


def test_checkpoint_resume_guards_and_empty_sources():
    model, runner, row = fixture()
    try:
        saved = dict(protocol=PROTOCOL, contract=dict(model=model.configs()), models={'gt': model.state_dict()})
        loaded = checkpoint_model(saved)
        for k, v in model.state_dict().items(): torch.testing.assert_close(loaded.state_dict()[k], v, rtol=0, atol=0)
        with pytest.raises(RuntimeError): checkpoint_model(dict(saved, protocol='stage1'))
        c = dict(world_size=2); state = dict(protocol=PROTOCOL, contract=c, epoch=1, batch=0, updates=3, rank_rng=[{}, {}])
        assert validate_resume(state, c, [[1, 2, 3]]) == (1, 0, 3)
        with pytest.raises(RuntimeError): validate_resume(dict(state, updates=4), c, [[1, 2, 3]])
        with pytest.raises(RuntimeError): validate_resume(dict(state, protocol='stage1'), c, [[1, 2, 3]])
        reads = {k: torch.as_tensor(v) for k, v in row[3][0]['shared_reads'].items()}
        for k in ('reference_valid', 'reference_grid', 'reference_xy_t0'): reads[k] = reads[k][:0]
        out = model.transport(torch.zeros(4, 16, 4, 4), reads, torch.arange(4), None, None)
        assert out['history_yaw_to_t0_rad'].shape == (0, 4)
    finally: runner.close()


def test_actual_two_step_adam_rng_recovery_exact():
    from tools.real_motion.joint_training_distributed import DistributedTraining
    from tools.dense_valid_ddp.engine import train_step
    model, runner, row = fixture(); ctx = DistributedTraining(False); rng = np.random.default_rng(21)
    opt = torch.optim.AdamW([dict(params=model.parameters(), lr=.001, initial_lr=.001)])
    try:
        train_step(model, opt, runner.provider, [row], ctx, global_windows=1, update=1, total_updates=3, forward=forward_window)
        saved = copy.deepcopy(dict(model=model.state_dict(), adam=opt.state_dict(), rank_rng=[ctx.rng_state(rng)]))
        train_step(model, opt, runner.provider, [row], ctx, global_windows=1, update=2, total_updates=3, forward=forward_window)
        expected = copy.deepcopy(model.state_dict()); adam = copy.deepcopy(opt.state_dict())
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['adam']); ctx.restore_rng(saved['rank_rng'], rng)
        train_step(model, opt, runner.provider, [row], ctx, global_windows=1, update=2, total_updates=3, forward=forward_window)
        for key, value in expected.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for key, values in adam['state'].items():
            for name, value in values.items(): torch.testing.assert_close(opt.state_dict()['state'][key][name], value, rtol=0, atol=0)
    finally: runner.close()


def test_cached_provider_discards_old_registration_and_never_falls_back(monkeypatch):
    from tools.source_scene_stage2.adapter import DirectProvider
    from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
    from real_motion import shared_source_track
    model, runner, row = fixture()
    try:
        timestamps = {str(t): int(i*500000) for i, t in enumerate(row[0]['history_tokens'])}
        source = SimpleNamespace(nusc=SimpleNamespace(get=lambda _, t: {'timestamp': timestamps[str(t)]}))
        p = DirectProvider.__new__(DirectProvider)
        p.pcfg, p.strong = runner.provider.pcfg, runner.provider.strong
        p.ccr_history_cache_mode = p.ccr_val_history_cache_mode = 'require'
        p.ccr_history_cache_source, p.ccr_val_history_cache_source = source, object()
        seen = []
        def cached(key, raw, build):
            np.testing.assert_array_equal(raw['future_poses'], np.array([0], np.uint8))
            seen.append(key); return build(), False
        p.track_cache = SimpleNamespace(get_or_build=cached)
        poison = dict(row[1], future_poses=object(), future_gt_occ=object(),
            _column_causal_preparation=dict(row[1]['_column_causal_preparation'], registrations=object(), _ccr_compact_support=object()))
        monkeypatch.setattr(PilotProvider, 'load_raw_columns', lambda *a, **k: poison)
        monkeypatch.setattr(shared_source_track, 'register_history_shape', lambda *a, **k: pytest.fail('ICP called'))
        loaded = p.load_raw_columns(source, row[0], include_gt=True)
        assert len(seen) == 1 and 'registrations' not in loaded['_column_causal_preparation']
        assert not hasattr(loaded['_shared_source_track'], 'registrations')
        assert loaded['future_poses'] is poison['future_poses']  # not read by identities
        with pytest.raises(RuntimeError, match='fallback'):
            p.load_raw_columns(object(), row[0], include_gt=True)
    finally: runner.close()


def test_full_cuda_runner_stop_resume_and_control_reused(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA')
    import json
    from dataclasses import asdict
    from tools.source_scene_stage2 import run
    from tools.source_scene_stage1.run import PROTOCOL as PREVIOUS
    from tools.source_scene_stage1.model import make_model as old_model
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
        control_model = old_model('reader', model.transport.config, **kwargs)
        control = dict(status='complete', updates=2, contract=dict(protocol=PREVIOUS, arm='reader',
            source_policy='shared_pinned', model=control_model.configs(), world_size=1,
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
