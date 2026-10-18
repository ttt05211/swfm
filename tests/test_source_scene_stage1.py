"""Isolated reader, original memory parity, full6 gradients and resume guards."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_scene_stage1.model import make_model, SceneBody, SceneReader, ReaderConfig
from tools.source_scene_stage1.history import StageHistory
from tools.source_scene_stage1.run import validate_resume, PROTOCOL
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs


def fixture(device='cpu'):
    from test_full_gt_training import fixture as old
    torch.set_num_threads(1)
    base, _, provider, row = old()
    model = make_model('reader', base.transport.config, feature_width=16, z_bins=4,
                       memory_width=4, width=8, chunk_size=32).to(device)
    provider.device = torch.device(device); provider.joint = model
    raw = dict(row[1], history_observed=np.ones_like(row[1]['history_occ'], dtype=bool))
    history = StageHistory(provider, arm='reader', ram_mib=0).build(raw)
    return model, provider, (row[0], raw, row[2], history)


def test_shared_fresh_initialization_and_no_old_motion_features():
    model, _, _ = fixture()
    kwargs = dict(feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32)
    torch.manual_seed(21); baseline = make_model('baseline', model.transport.config, **kwargs)
    torch.manual_seed(21); reader = make_model('reader', model.transport.config, **kwargs)
    for key, value in reader.state_dict().items():
        if key in baseline.state_dict() and baseline.state_dict()[key].shape == value.shape:
            torch.testing.assert_close(value, baseline.state_dict()[key], rtol=0, atol=0)
    assert not any(name in dict(reader.transport.named_modules()) for name in ('motion_condition', 'class_embedding', 'source_adapter', 'existence_head'))
    assert 'kta_future_proj' in dict(reader.transport.named_modules())


def test_memory_bytes_validity_and_no_legacy_patch_construction(monkeypatch):
    model, provider, row = fixture()
    expected = ResidualHistoryInputs(provider, ram_mib=0).build(row[1])
    monkeypatch.setattr('tools.shared_history_experiments.refine.history.history_reads', lambda *a, **k: pytest.fail('old patch built'))
    actual = StageHistory(provider, arm='reader', ram_mib=0).build(dict(row[1], future_poses=object(), future_gt_occ=object()))
    for key, value in expected[0]['memory_reads'].items(): np.testing.assert_array_equal(actual[0]['memory_reads'][key], value)
    np.testing.assert_array_equal(actual[0]['shared_reads']['reference_valid'], expected[0]['shared_reads']['source_valid'])
    np.testing.assert_allclose(actual[0]['shared_reads']['reference_xy_t0'], expected[0]['shared_reads']['source_center'], atol=1e-6)
    assert set(actual[0]['shared_reads']).isdisjoint({'source_mask', 'source_grid', 'velocity'})


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_full_six_initial_kta_poison_class_and_gradients(device):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA required')
    model, provider, row = fixture(device)
    fw = gt.forward_window(model, provider, row)
    torch.testing.assert_close(fw['logits'], model.columns.prior_logits(fw['prior']), rtol=0, atol=0)
    assert fw['output']['residual_xy_m'].count_nonzero() == 0
    reads = {k: torch.as_tensor(v, device=device) for k, v in row[3][0]['shared_reads'].items()}
    times = torch.as_tensor(row[2][0][:, -1], device=device)
    kta = torch.as_tensor(row[0]['kta_displacement_xy_m'], device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
        output = model.transport(fw['field'], reads, times, object(), kta)
    torch.testing.assert_close(output['future_transport_queries'], fw['output']['future_transport_queries'], rtol=0, atol=0)
    adam = torch.optim.AdamW([dict(params=model.parameters(), lr=1e-3, initial_lr=1e-3)])
    for update in (1, 2):
        result = gt.train_arm(model, adam, provider, [row], 'gt', update=update, total_updates=4, probe=True)
    assert result['gradient_links']['occupancy_to_xy'] > 0
    assert result['gradient_links']['occupancy_to_memory'] > 0
    assert result['gradient_links']['occupancy_to_shared_history'] > 0
    assert model.transport.readers[0].offsets.weight.grad is None  # optimizer cleanup
    fw = gt.forward_window(model, provider, row)
    loss = gt.semantic_loss(fw['logits'], row[1]['future_gt_occ'])[0].mean(); loss.backward()
    assert model.transport.readers[0].offsets.weight.grad.abs().sum() > 0
    assert model.transport.position.projection.weight.grad.abs().sum() > 0


def test_spacetime_identifiability_invalid_mask_and_empty_sources():
    model, _, row = fixture(); transport = model.transport.eval()
    reads = {k: torch.as_tensor(v) for k, v in row[3][0]['shared_reads'].items()}
    n = len(reads['reference_valid']); dim = transport.config.d_model
    assert n > 0
    field = torch.ones(4, 16, 4, 4)
    kta = torch.zeros(n, 6, 2); times = torch.linspace(-1, 0, 4)
    reads['reference_valid'][:] = True
    same = transport(field, reads, times, object(), kta)['future_transport_queries']
    other = copy.deepcopy(reads); other['reference_xy_t0'][:, :3, 0] += torch.tensor([3., 2., 1.])
    moved = transport(field, other, times, object(), kta)['future_transport_queries']
    assert not torch.allclose(same, moved)
    changed_time = transport(field, reads, times*.5, object(), kta)['future_transport_queries']
    assert not torch.allclose(same, changed_time)
    reads['reference_valid'][:, 0] = False
    a = transport(field, reads, times, object(), kta)['future_transport_queries']
    other = copy.deepcopy(reads)
    other['reference_xy_t0'][:, 0] = 1000.; other['reference_grid'][:, 0] = -1000.
    b = transport(field, other, times, object(), kta)['future_transport_queries']
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for k in ('reference_valid', 'reference_xy_t0', 'reference_grid'): reads[k] = reads[k][:0]
    empty = transport(field, reads, times, object(), kta[:0])
    assert empty['future_transport_queries'].shape == (0, 6, dim)


def test_single_source_local_head_selection_shapes():
    reader = SceneReader(8, 2, 3)
    q = torch.randn(4, 1, 8)
    reads = dict(reference_grid=torch.zeros(1, 4, 2), t0_xy_to_native_grid=torch.eye(2)[None].repeat(4, 1, 1)*.1,
                 scene_bounds=torch.tensor([-4., -4., 4., 4.]))
    output = reader(q, [torch.randn(4, 8, s, s) for s in (8, 2, 1)], torch.randn(4, 1, 8),
                    torch.randn(4, 1, 8), reads, torch.ones(1, 4, dtype=torch.bool))
    assert output.shape == q.shape and torch.isfinite(output).all()


def test_resume_rejects_changed_arm_budget_and_incomplete_cursor():
    groups = [[1, 2], [3, 4, 5]]; contract = dict(arm='reader', world_size=1)
    saved = dict(protocol=PROTOCOL, contract=contract, epoch=1, batch=1, updates=3, rank_rng=[{}])
    assert validate_resume(saved, contract, groups) == (1, 1, 3)
    with pytest.raises(RuntimeError): validate_resume(saved, dict(contract, arm='baseline'), groups)
    with pytest.raises(RuntimeError): validate_resume(dict(saved, updates=2), contract, groups)


@pytest.mark.parametrize('arm', ['baseline', 'reader'])
def test_checkpoint_roundtrip_explicit_model_and_old_protocol_rejected(arm):
    from tools.source_scene_stage1.model import checkpoint_model
    model, _, _ = fixture()
    original = make_model(arm, model.transport.config, feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32)
    saved = dict(protocol=PROTOCOL, contract=dict(arm=arm, source_policy='shared_pinned', model=original.configs()),
                 models={'gt': original.state_dict()})
    actual = checkpoint_model(saved)
    for key, value in original.state_dict().items(): torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(RuntimeError): checkpoint_model(dict(saved, protocol='old_gt_body'))


def test_actual_reader_adam_rng_recovery_exact(tmp_path):
    from tools.real_motion.joint_training_recovery import save_resume_checkpoint
    from tools.real_motion.joint_training_distributed import DistributedTraining
    from tools.dense_valid_ddp.engine import train_step
    model, provider, row = fixture()
    ctx = DistributedTraining(False); rng = np.random.default_rng(21)
    opt = torch.optim.AdamW([dict(params=model.parameters(), lr=.001, initial_lr=.001)])
    train_step(model, opt, provider, [row], ctx, global_windows=1, update=1, total_updates=4)
    saved = dict(model=copy.deepcopy(model.state_dict()), adam=copy.deepcopy(opt.state_dict()), rank_rng=[ctx.rng_state(rng)])
    save_resume_checkpoint(tmp_path/'last.pt', saved)
    train_step(model, opt, provider, [row], ctx, global_windows=1, update=2, total_updates=4)
    expected, expected_adam = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    restored = torch.load(tmp_path/'last.pt', weights_only=False)
    model.load_state_dict(restored['model']); opt.load_state_dict(restored['adam']); ctx.restore_rng(restored['rank_rng'], rng)
    train_step(model, opt, provider, [row], ctx, global_windows=1, update=2, total_updates=4)
    for k, value in expected.items(): torch.testing.assert_close(model.state_dict()[k], value, atol=0, rtol=0)
    for k, fields in expected_adam['state'].items():
        for name, value in fields.items(): torch.testing.assert_close(opt.state_dict()['state'][k][name], value, atol=0, rtol=0)


def test_fixed_training_plan_and_comparison_reject_mismatched_population(tmp_path):
    import json
    from tools.source_scene_stage1.run import plan
    from tools.source_scene_stage1.report import compare
    from tools.perception_robustness.evaluation import ROUTES
    train = [dict(scene_name=f'T{i//50}', t0_token=f't{i}', features=np.empty((i % 5, 0))) for i in range(20430)]
    dev = [dict(scene_name=f'V{i%150}', t0_token=f'v{i}', features=[]) for i in range(4369)]
    keys = [[r['scene_name'], r['t0_token']] for r in dev[:64]]
    reference = tmp_path/'reference.json'; reference.write_text(json.dumps(dict(contract=dict(eval_keys=keys))))
    selected, evaluation, groups = plan(train, dev, reference, 20261011)
    assert len(selected) == len(set(map(tuple, selected))) == 4086
    for epoch in groups:
        flat = [(r['scene_name'], r['t0_token']) for batch in epoch for r in batch]
        assert len(flat) == 4086 and set(flat) == set(map(tuple, selected))
    fields = ('train_keys', 'eval_keys', 'epochs', 'total_updates', 'epochs_updates', 'seed', 'learning_rates',
              'schedule', 'objective', 'known', 'source_policy', 'source_implementation', 'data', 'overrides',
              'precision', 'world_size', 'STC_inventory', 'implementation')
    common = dict.fromkeys(fields, 'same'); common.update(protocol=PROTOCOL, eval_keys=keys, epochs=1)
    values = {name: dict(mIoU=20., IoU=30., MovingMicro=10.) for name in ROUTES}
    for arm in ('baseline', 'reader'):
        folder = tmp_path/arm; folder.mkdir()
        result = dict(status='complete', contract=dict(common, arm=arm),
                      evaluations={'1': dict(windows=64, keys=keys, variants=values)})
        (folder/'results.json').write_text(json.dumps(result))
    assert len(groups) == 1
    report = compare(tmp_path)
    assert 'epoch1 stc_gt_semantic_refine' in report and 'epoch2' not in report
    result['contract']['known'] = 'lidar_known'
    (tmp_path/'reader/results.json').write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match='known'): compare(tmp_path)


def test_one_epoch_complete_cursor_and_old_schedule_rejected():
    groups = [[1, 2, 3]]; contract = dict(epochs=1, world_size=1)
    saved = dict(protocol=PROTOCOL, contract=contract, epoch=1, batch=0, updates=3, rank_rng=[{}])
    assert validate_resume(saved, contract, groups) == (1, 0, 3)
    with pytest.raises(RuntimeError): validate_resume(dict(saved, epoch=2), contract, groups)
    with pytest.raises(RuntimeError): validate_resume(dict(saved, batch=1), contract, groups)
    with pytest.raises(RuntimeError):
        validate_resume(dict(saved, protocol='source_scene_reader_stage1_matched_TRAIN20x2_H6_v1'), contract, groups)


@pytest.mark.parametrize('arm', ['baseline', 'reader'])
def test_actual_shared_track_occ_stc_evaluation_resume(arm, tmp_path, monkeypatch):
    from test_shared_source_track import fixture as tracks
    from tools.source_scene_stage1 import evaluation
    runner, rec, raw = tracks()
    try:
        old = runner.body
        model = make_model(arm, old.transport.config, feature_width=old.transport.history_encoder.width,
            z_bins=old.columns.z_bins, memory_width=old.columns.memory_width,
            width=old.columns.width, chunk_size=old.columns.chunk_size)
        runner.body = runner.provider.joint = model
        runner.history = StageHistory(runner.provider, arm=arm, ram_mib=0)
        poses = np.tile(np.eye(4), (6, 1, 1))
        row = runner.prepare(rec, raw, times=np.arange(4)*.5)
        fw = runner.forward(row, poses=poses)
        historical = dict(row[1], future_poses=poses, future_gt_occ=np.tile(raw['history_occ'][-1], (6, 1, 1, 1)))
        historical['_column_causal_preparation']['prepared_state']['column_backgrounds'] = fw['state']['column_backgrounds']
        row = row[0], historical, row[2], row[3]
        for token in rec['history_tokens']:
            folder = tmp_path/rec['scene_name']/token; folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder/'labels.npz', semantics=raw['history_occ'][0])
        monkeypatch.setattr(gt, 'gt_moving_support_sequence', lambda *a, **k: [])
        monkeypatch.setattr(gt, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
        saved = []
        kwargs = dict(arm=arm, stc_root=tmp_path, save=lambda s: saved.append(copy.deepcopy(s)))
        a = evaluation.evaluate(model, runner.provider, SimpleNamespace(nusc=None), [[row]], **kwargs)
        first = copy.deepcopy(saved[-1]); old_state = copy.deepcopy(model.state_dict())
        b = evaluation.evaluate(model, runner.provider, SimpleNamespace(nusc=None), [[row]], saved=first, **kwargs)
        assert first == saved[-1] and b['windows'] == a['windows'] == 1
        for route in a['variants']:
            for metric in ('mIoU', 'IoU', 'MovingMicro'):
                assert a['variants'][route][metric] == b['variants'][route][metric]
        for key, value in old_state.items(): torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    finally: runner.close()
