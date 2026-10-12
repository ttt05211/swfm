"""Focused geometry/gradient/causality/restore tests, not real-data score claims."""
import copy
import json
import threading
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.shared_history_experiments.history import SharedHistoryInputs, normalized_xy, compact_geometry
from tools.shared_history_experiments.model import SharedJoint, SharedEncoder, sample_points, PROTOCOL
from tools.shared_history_experiments import engine, run
from tools.joint_occ_experiments import run as legacy
from tools.joint_occ_experiments.soft_inference.engine import soft_predictions
from tools.joint_occ_experiments.soft_inference.fast import predict
from real_motion.canonical_causal_repair import materialize_compact_canonical
from test_joint_occ_ego import fixture as old_fixture


def fixture():
    old, _, provider, row = old_fixture()
    model = SharedJoint(old.transport.v17_config, width=16, z_bins=4)
    provider.joint = model
    history = SharedHistoryInputs(provider, ram_mib=1).get(row[0], row[1])
    row = (*row[:3], history)
    optimizer = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=5e-4, initial_lr=5e-4),
        dict(params=model.columns.parameters(), lr=3e-4, initial_lr=3e-4),
        dict(params=model.ego.parameters(), lr=3e-4, initial_lr=3e-4)])
    return model, optimizer, provider, row


def test_geometry_order_metric_points_and_surface_preserved_without_old_descriptors():
    _, _, p, row = fixture()
    causal, evidence = row[3]
    prep = SimpleNamespace(raw=row[1], state=causal['state'], registrations=causal['registrations'])
    old = materialize_compact_canonical(causal['support'], prep, p.pcfg.grid, np.arange(len(evidence)))
    for key in ('world', 'actor', 'classes', 'presence'):
        np.testing.assert_array_equal(getattr(old, key), getattr(evidence, key))
    assert not evidence.features[:, :27].any()
    assert evidence.features[:, 27:].shape[1] == 12
    assert not evidence.audit['legacy_27_descriptors_used']


def test_height_layers_unknown_and_source_mask_are_not_collapsed():
    torch.manual_seed(10)
    encoder = SharedEncoder(z_bins=4, width=16)
    a = torch.full((4, 8, 8, 4), 17)
    b = a.clone(); a[:, 3, 3, 0] = 4; b[:, 3, 3, 3] = 4
    known = torch.ones_like(a, dtype=torch.bool)
    assert encoder(a, known).shape == (4, 16, 4, 4)
    assert not torch.equal(encoder(a, known), encoder(b, known))
    known[:] = False
    torch.testing.assert_close(encoder(a, known), encoder(b, known), rtol=0, atol=0)
    _, _, _, row = fixture()
    reads = row[3][0]['shared_reads']
    assert reads['source_mask'].sum() > 0


def test_feature_sampling_axis_centres_and_encoder_gradient():
    _, _, p, _ = fixture()
    field = torch.arange(16.).reshape(1, 1, 4, 4).expand(4, -1, -1, -1).clone().requires_grad_()
    grid = torch.as_tensor(normalized_xy(np.array([[1., 3., 0.]]), p.pcfg.grid))[:, None].expand(-1, 4, -1)
    out = sample_points(field, grid)
    torch.testing.assert_close(out[0, :, 0], torch.ones(4))  # X0,Y1, not X1,Y0
    out.sum().backward(); assert field.grad.abs().sum() == 4


def test_sources_and_canonical_reads_follow_actual_past_positions():
    _, _, p, row = fixture()
    raw = copy.deepcopy(row[1])
    reg = raw['_column_causal_preparation']['registrations'][0]
    reg[0] = (np.eye(4), np.array([[1, 2, 1]], np.int64))
    reg[0][0][0, 3] = 1  # historical object moved into its t0 canonical position
    history = SharedHistoryInputs(p, ram_mib=0).get(row[0], raw)
    reads = history[0]['shared_reads']; evidence = history[1]
    assert reads['source_center'][0, 0, 0] == 1.5
    at = np.flatnonzero((evidence.actor == 0) & np.all(evidence.world == [2.5, 2.5, 1.5], 1))[0]
    expected = normalized_xy(np.array([[1.5, 2.5, 1.5]]), p.pcfg.grid)[0]
    np.testing.assert_allclose(reads['point_grid'][at, 0], expected)


def test_one_shared_encoder_forward_and_no_future_label_input():
    model, _, p, row = fixture(); model.eval()
    calls = []
    hook = model.transport.history_encoder.register_forward_hook(lambda *args: calls.append(1))
    expected = engine.forward_window(model, p, row)
    raw = copy.deepcopy(row[1]); raw['future_gt_occ'] = 'POISON FUTURE OCCUPANCY'
    raw['future_annotations'] = 'POISON'
    poisoned = engine.forward_window(model, p, (row[0], raw, row[2], row[3]))
    hook.remove(); assert len(calls) == 2  # exactly ONCE per entire window
    for key in ('actions', 'mapped', 'targets'):
        torch.testing.assert_close(expected[key], poisoned[key], rtol=0, atol=0)
    assert torch.count_nonzero(expected['actions'][..., 1]) == 0
    assert not hasattr(model.transport, 'spatial_stem')
    assert not hasattr(model.columns, 'encoder')  # no old 27+semantic-tube encoder


def test_full_ce_lovasz_backward_updates_shared_motion_ccr_and_direct_auxiliary():
    model, adam, p, row = fixture()
    before = copy.deepcopy(model.state_dict())
    stats = engine.train_arm(model, adam, p, [row], 'gt', update=1, total_updates=3, probe=True)
    assert all(np.isfinite(stats[k]) for k in ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry'))
    assert stats['gradient_links']['occupancy_to_shared_history'] > 0
    assert stats['gradient_links']['occupancy_to_actions'] > 0
    for prefix in ('transport.history_encoder.', 'transport.residual_head.', 'columns.', 'ego.'):
        assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items() if k.startswith(prefix))


@torch.no_grad()
def test_full_six_fast_soft_outputs_match_training_renderer():
    model, _, p, row = fixture(); model.eval()
    fw = engine.forward_window(model, p, row)
    reference = soft_predictions(p, fw, ('transport', 'add_only'), range(6))
    for name, frames in reference.items():
        actual = predict(p, fw, name, range(6), group_size=3)
        for a, b in zip(actual, frames): np.testing.assert_array_equal(a, b)


def test_empty_sources_and_all_missing_history_have_finite_readouts():
    model, _, p, row = fixture()
    reads = {k: torch.as_tensor(v) for k, v in row[3][0]['shared_reads'].items()}
    field = model.transport.history_encoder(torch.as_tensor(row[1]['history_occ']), torch.as_tensor(row[1]['history_observed']))
    reads['point_valid'][:] = False
    encoded = model.columns.encode_history(field, reads, torch.tensor([-1., -.66, -.33, 0.]), row[3][1])
    assert torch.isfinite(encoded).all()
    reads = {k: v[:0] if k.startswith('source_') else v for k, v in reads.items()}
    output = model.transport(field, reads, torch.tensor([-1., -.66, -.33, 0.]), torch.empty(0, dtype=torch.long), torch.empty(0, 6, 2))
    assert output['future_transport_queries'].shape == (0, 6, model.transport.config.d_model)


def test_strict_new_protocol_restores_adam_rng_and_rejects_old_checkpoint(monkeypatch):
    model, adam, p, row = fixture()
    monkeypatch.setattr(legacy, 'PROTOCOL', PROTOCOL+'_shared')
    contract = {'prototype': 'shared', 'epochs': 2, 'train_fraction': .2}
    engine.train_arm(model, adam, p, [row], 'gt', update=1, total_updates=2)
    saved = copy.deepcopy(legacy.checkpoint({'gt': model}, {'gt': adam}, contract, 0, 1, 1, {}))
    engine.train_arm(model, adam, p, [row], 'gt', update=2, total_updates=2)
    expected = copy.deepcopy(model.state_dict())
    legacy.restore_checkpoint(saved, {'gt': model}, {'gt': adam}, contract)
    engine.train_arm(model, adam, p, [row], 'gt', update=2, total_updates=2)
    for k, v in expected.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    saved['protocol'] = 'old_protocol'
    with pytest.raises(RuntimeError): legacy.restore_checkpoint(saved, {'gt': model}, {'gt': adam}, contract)


def test_actual_soft_evaluation_integer_counts_restore_and_source_read_only(monkeypatch):
    model, _, p, row = fixture()
    monkeypatch.setattr(engine, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    states = []; before = copy.deepcopy(model.state_dict())
    source = SimpleNamespace(nusc=None)
    first = engine.evaluate({'gt': model}, p, source, [[row]], save=lambda v: states.append(copy.deepcopy(v)))
    saved = json.loads(json.dumps(states[-1]))
    second = engine.evaluate({'gt': model}, p, source, [[row], [row]], saved=saved)
    whole = engine.evaluate({'gt': model}, p, source, [[row], [row]])
    assert first['windows'] == 1 and second['windows'] == whole['windows'] == 2
    assert json.dumps(second['variants'], sort_keys=True) == json.dumps(whole['variants'], sort_keys=True)
    assert saved['windows'] == 1 and len(saved['errors']) == 1
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    stop = threading.Event(); stop.set()
    with pytest.raises(InterruptedError): engine.evaluate({'gt': model}, p, source, [[row]], stop_event=stop)


@pytest.mark.parametrize('variant', ['shared', 'legacy_control'])
def test_real_cli_new_population_epoch_snapshots_resume_and_eval_only(tmp_path, monkeypatch, variant):
    model, _, provider, row = fixture()
    config = tmp_path/'config.yaml'; config.write_text('test')
    base = tmp_path/'base.pt'; torch.save({'model_config': asdict(model.transport.config)}, base)
    paths = {n: tmp_path/n for n in ('train-cache', 'dev-cache', 'train-info', 'dev-info', 'population-manifest')}
    for path in paths.values(): path.write_text('test')
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    keys = [('DEV', r['t0_token']) for r in dev[:512]]
    monkeypatch.setattr(legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(legacy, 'load_cache', lambda path: ({}, train if path == str(paths['train-cache']) else dev))
    monkeypatch.setattr(legacy, 'load_manifest', lambda _: ({'parent_keys': keys}, keys[:64], None))
    monkeypatch.setattr(legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(legacy, 'make_prepare_config', lambda _: provider.pcfg)
    monkeypatch.setattr(legacy, 'PilotProvider', lambda *args: provider)
    monkeypatch.setattr(legacy, 'NuScenesWindowSource', lambda root, *, info_pkl: SimpleNamespace(nusc=None))
    class Disposable:
        kernels = pool = None
        def __init__(self, *a): pass
        def close(self): pass
    monkeypatch.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
    monkeypatch.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
    def caches(p, *a):
        p.ccr_history_cache = p.ccr_val_history_cache = SimpleNamespace(stats=lambda: {}, close=lambda: None)
        return {'train': {'namespace': 'train'}, 'val': {'namespace': 'val'}}
    monkeypatch.setattr(legacy, 'attach_caches', caches)
    # Keep the REAL 4086-window selection; shorten only mocked epoch grouping.
    def groups(records, *a):
        assert len(records) == 4086
        return [[records[0]], [records[1]]]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    monkeypatch.setattr(legacy, 'prefetched_batches', lambda p, source, groups, descriptors, **k:
        ([(record, row[1], row[2], row[3]) for record in group] for group in groups))
    def step(m, adam, p, rows, mode, **kw):
        adam.zero_grad(set_to_none=True)
        loss = sum(v.square().sum() for v in m.parameters())*.0001*torch.rand(())
        loss.backward(); adam.step()
        return dict(windows=len(rows), seconds=.01, stages_seconds={}, CE=float(loss.detach()), gradient_links={})
    monkeypatch.setattr(engine, 'train_arm', step)
    def evaluate(models, *a, **k):
        assert all(not m.training for m in models.values())
        return dict(windows=512, variants={}, trajectory={})
    monkeypatch.setattr(engine, 'evaluate', evaluate)
    out = tmp_path/'pilot'
    args = ['--config', str(config), '--base-checkpoint', str(base), '--dataroot', str(tmp_path),
        '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
        '--out-dir', str(out), '--variant', variant]
    args += [v for n, p in paths.items() for v in ('--'+n, str(p))]
    original_constructor = legacy.JointOccEgo
    assert run.main(threading.Event(), args+['--max-updates', '2']) == 0
    saved = torch.load(out/'last.pt', weights_only=False)
    assert len(saved['contract']['train_keys']) == 4086 and saved['contract']['epochs'] == 2
    assert saved['updates'] == 2 and saved['epoch'] == 1 and (out/'epoch_0001.pt').is_file()
    assert saved['protocol'] == PROTOCOL+'_'+variant and legacy.JointOccEgo is original_constructor
    epoch_bytes = (out/'epoch_0001.pt').read_bytes()
    assert run.main(threading.Event(), args+['--resume', str(out/'last.pt')]) == 0
    final = torch.load(out/'last.pt', weights_only=False)
    assert final['updates'] == 4 and final['epoch'] == 2 and (out/'epoch_0002.pt').is_file()
    assert (out/'epoch_0001.pt').read_bytes() == epoch_bytes
    last_bytes = (out/'last.pt').read_bytes(); summary_bytes = (out/'summary.txt').read_bytes()
    evaluation_args = args.copy(); evaluation_args[evaluation_args.index(str(out))] = str(out/'eval_epoch1')
    assert run.main(threading.Event(), evaluation_args+['--resume', str(out/'epoch_0001.pt'), '--evaluate-only']) == 0
    assert (out/'last.pt').read_bytes() == last_bytes and (out/'summary.txt').read_bytes() == summary_bytes
    wrong = args.copy(); wrong[wrong.index(variant)] = 'legacy_control' if variant == 'shared' else 'shared'
    with pytest.raises(RuntimeError, match='differs'): run.main(threading.Event(), wrong+['--resume', str(out/'last.pt')])


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason='CUDA/BF16 not available')
def test_actual_cuda_bf16_shared_encoder_source_and_point_readout_update_and_fast_parity():
    model, _, p, row = fixture()
    p.device = torch.device('cuda'); model = model.to(p.device)
    adam = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=5e-4, initial_lr=5e-4),
        dict(params=model.columns.parameters(), lr=3e-4, initial_lr=3e-4),
        dict(params=model.ego.parameters(), lr=3e-4, initial_lr=3e-4)])
    stats = engine.train_arm(model, adam, p, [row], 'gt', update=1, total_updates=2, probe=True)
    assert stats['gradient_links']['occupancy_to_shared_history'] > 0
    model.eval()
    with torch.no_grad():
        fw = engine.forward_window(model, p, row)
        reference = soft_predictions(p, fw, ('transport', 'add_only'), range(6))
        for name in reference:
            actual = predict(p, fw, name, range(6), group_size=3)
            for a, b in zip(actual, reference[name]): np.testing.assert_array_equal(a, b)
