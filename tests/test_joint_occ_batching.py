"""Real small-grid gradients/Adam, not claims about nuScenes or GPU speed."""
import copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch
from test_joint_occ_ego import fixture
from tools.joint_occ_experiments.engine import forward_window, train_arm as reference_step
from tools.joint_occ_experiments.soft_geometry import SoftScene, occupancy_loss, sparse_splat
from tools.joint_occ_experiments.batching.geometry import GroupedScene, grouped_loss, grouped_splat
from tools.joint_occ_experiments.batching.engine import forward_batch, train_arm


@pytest.mark.parametrize('size', [1, 2, 3])
def test_grouped_render_preserves_six_outputs_and_boundary_gradients(size):
    model, _, provider, row = fixture()
    fw = forward_window(model, provider, row, 'gt')
    from tools.joint_occ_experiments.model import torch_poses
    fw['poses'] = torch_poses(np.eye(4), torch.tensor([[h*.07, -h*.04, h*.02] for h in range(6)]))
    fields = ('poses', 'targets', 'yaws', 'mapped', 'actions')
    def run(grouped):
        values = {key: fw[key].detach().clone().requires_grad_(True) for key in fields}
        scene = GroupedScene(fw['prep'], provider.pcfg.grid, provider.device)
        if grouped:
            predictions = torch.cat([scene.render_group(list(range(h, min(h+size, 6))),
                **values, evidence=fw['evidence'], plan=fw['plan']) for h in range(0, 6, size)])
        else:
            predictions = torch.stack([scene.render(h, values['poses'][h], values['targets'], values['yaws'],
                values['mapped'], fw['evidence'], fw['plan'], values['actions']) for h in range(6)])
        # Nonconstant upstream also checks horizon/class addressing in backward.
        upstream = torch.linspace(-.2, .7, predictions.numel()).reshape(predictions.shape)
        gradients = torch.autograd.grad((predictions*upstream).sum(), list(values.values()))
        return predictions.detach(), gradients
    old, gradients = run(False); new, batched = run(True)
    torch.testing.assert_close(new, old, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(batched, gradients):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('ignore', [False, True])
def test_loss_each_horizon_class_mean_and_gradient(ignore):
    torch.manual_seed(7)
    probability = torch.rand(3, 18, 48).softmax(1).requires_grad_()
    labels = np.full((3, 48), 17, np.int64)
    labels[0, :6] = 4; labels[0, 6:15] = 11
    labels[1, :4] = 13
    if ignore: labels[0, 31:] = 255; labels[2] = 255
    weight = torch.tensor([.8, -.4, 1.2])
    actual = grouped_loss(probability, labels, lovasz_weight=.7)
    expected_rows = [occupancy_loss(p, torch.as_tensor(t), lovasz_weight=.7,
        class_ids=[int(c) for c in np.unique(t) if c < 17]) for p, t in zip(probability, labels)]
    expected = tuple(torch.stack([term[i] for term in expected_rows]) for i in range(3))
    for a, b in zip(actual, expected): torch.testing.assert_close(a, b, atol=3e-7, rtol=2e-6)
    a = torch.autograd.grad((actual[0]*weight).sum(), probability, retain_graph=True)[0]
    b = torch.autograd.grad((expected[0]*weight).sum(), probability)[0]
    torch.testing.assert_close(a, b, atol=2e-7, rtol=3e-5)


@pytest.mark.parametrize('reduction', ['sum', 'amax'])
def test_splat_no_cross_horizon_leak_with_ties_and_outside(reduction):
    _, _, p, _ = fixture()
    points = torch.tensor([[2.3, 2.2, 1.2], [2.3, 2.2, 1.2], [-30, 0, 0]], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([.3, .3, .9], requires_grad=True)
    ids, field = grouped_splat(points, values, torch.tensor([0, 0, 1]), p.pcfg.grid, reduction)
    from real_motion.canonical_causal_repair import grid_arrays
    volume = int(np.prod(grid_arrays(p.pcfg.grid)[2]))
    dense = field.new_zeros(2*volume).scatter(0, ids, field).reshape(2, volume)
    for slot, indices in enumerate(([0, 1], [2])):
        expected_ids, expected = sparse_splat(points[indices], values[indices], p.pcfg.grid, reduction=reduction)
        reference = expected.new_zeros(volume).scatter(0, expected_ids, expected)
        torch.testing.assert_close(dense[slot], reference)
    dense.sum().backward()
    assert torch.isfinite(points.grad).all() and values.grad[2] == 0


@pytest.mark.parametrize('count', [1, 4])
def test_packed_motion_and_ccr_keep_window_actor_identity(count):
    model, _, provider, row = fixture()
    rows = [copy.deepcopy(row) for _ in range(count)]
    for i, item in enumerate(rows): item[0]['features'] = item[0]['features']+i*.01
    with ThreadPoolExecutor(max_workers=2) as pool:
        packed = forward_batch(model, provider, rows, projection_pool=pool)
    for item, actual in zip(rows, packed):
        expected = forward_window(model, provider, item, 'gt')
        for key in ('actions', 'targets', 'yaws', 'mapped', 'ego'):
            torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-5)
        for key in expected['output']:
            torch.testing.assert_close(actual['output'][key], expected['output'][key], atol=2e-6, rtol=2e-5)
        np.testing.assert_array_equal(actual['evidence'].actor, expected['evidence'].actor)
        np.testing.assert_array_equal(actual['plan'].legal, expected['plan'].legal)


@pytest.mark.parametrize('size', [1, 2])
def test_two_actual_packed_adam_updates_match_original_batch4(size):
    torch.manual_seed(12)
    model, optimizer, provider, row = fixture()
    batched, adam = copy.deepcopy((model, optimizer))
    rows = [copy.deepcopy(row) for _ in range(4)]
    for i, item in enumerate(rows): item[0]['features'] = item[0]['features']+i*.02
    for update in (1, 2):
        expected = reference_step(model, optimizer, provider, rows, 'gt', update=update, total_updates=20, probe=True)
        actual = train_arm(batched, adam, provider, rows, 'gt', update=update, total_updates=20, probe=True, horizon_group=size)
        for key in ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry', 'grad_norm'):
            assert actual[key] == pytest.approx(expected[key], rel=2e-4, abs=2e-5)
        for key, value in actual['gradient_links'].items():
            assert value == pytest.approx(expected['gradient_links'][key], rel=2e-4, abs=2e-5)
        from tools.joint_occ_experiments.performance.probe import assert_close_state
        assert_close_state(model.state_dict(), batched.state_dict())
        assert_close_state(optimizer.state_dict(), adam.state_dict())
    assert actual['windows'] == 4 and actual['sources'] == 4


def test_no_future_labels_read_by_packed_forward():
    model, _, provider, row = fixture()
    row[1]['future_gt_occ'] = 'POISON'
    forwards = forward_batch(model, provider, [row, row])
    assert len(forwards) == 2


def empty_row(row, *, empty_evidence=False):
    row = copy.deepcopy(row)
    record, raw, inputs, history = row
    for key, value in record.items():
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == 1: record[key] = value[:0]
    causal, evidence = history
    state = causal['state']
    for key in ('current', 'source_world_points', 'source_rel_xy'): state[key] = []
    state['source_z_t0'] = np.empty(0, np.float64)
    causal['registrations'] = []
    chosen = np.zeros(len(evidence), bool) if empty_evidence else evidence.actor < 0
    for key in ('features', 'labels', 'actor', 'classes', 'world', 'presence'):
        setattr(evidence, key, getattr(evidence, key)[chosen])
    return record, raw, inputs, (causal, evidence)


@pytest.mark.parametrize('mixed', [False, True])
def test_empty_sources_and_empty_canonical_rows_have_valid_backward(mixed):
    model, optimizer, provider, row = fixture()
    batched, adam = copy.deepcopy((model, optimizer))
    rows = [empty_row(row, empty_evidence=True), empty_row(row)]
    if mixed: rows += [row]
    expected = reference_step(model, optimizer, provider, rows, 'gt', update=1, total_updates=20)
    actual = train_arm(batched, adam, provider, rows, 'gt', update=1, total_updates=20)
    from tools.joint_occ_experiments.performance.probe import assert_close_state
    assert_close_state(model.state_dict(), batched.state_dict())
    assert_close_state(optimizer.state_dict(), adam.state_dict())
    assert actual['CE'] == pytest.approx(expected['CE'], abs=2e-6)


def test_readonly_probe_actual_updates_restore_nonempty_adam_and_rng():
    from test_joint_occ_speed_probe import saved_fixture
    from tools.joint_occ_experiments.batching import probe
    from tools.joint_occ_experiments.run import rng_state
    model, optimizer, provider, row, saved = saved_fixture()
    saved['contract'].update(window_batch=4, source_budget=128)
    immutable = copy.deepcopy(saved)
    report = probe.measure(model, optimizer, provider, saved, [[row, row], [row]], progress=lambda *a, **k: None)
    assert all(row['numerical_parity'] for row in report['warmup'].values())
    assert all(row['numerical_parity'] for row in report['final_state_parity'].values())
    assert all(len(rows) == 2 for rows in report['timed_passes'].values())
    assert not any(row['pass_gate'] for row in report['candidates'].values()) # CPU cannot pass GPU memory gate.
    assert probe.base.assert_close_state(model.state_dict(), saved['models']['gt']) == 0
    assert probe.base.assert_close_state(optimizer.state_dict(), saved['optimizers']['gt']) == 0
    for key in ('models', 'optimizers', 'contract', 'epoch', 'batch', 'updates', 'reports'):
        assert probe.base.assert_close_state(saved[key], immutable[key]) == 0
    assert torch.equal(saved['rng']['torch'], immutable['rng']['torch'])
    np.testing.assert_array_equal(saved['rng']['numpy'][1], immutable['rng']['numpy'][1])
    assert saved['rng']['python'] == immutable['rng']['python']
    assert torch.equal(rng_state()['torch'], saved['rng']['torch'])


def test_resume_allows_only_explicit_execution_change_preserves_adam_cursor_rng():
    from test_joint_occ_speed_probe import saved_fixture
    from tools.joint_occ_experiments.batching.run import import_contract, validate_receipt
    from tools.joint_occ_experiments.run import restore_checkpoint, rng_state
    model, optimizer, _, _, saved = saved_fixture()
    saved['contract'].update(implementation='old', window_batch=4, source_budget=128)
    current = {**saved['contract'], 'implementation': 'new'}
    imported, changed = import_contract(saved, current, 'old')
    assert changed and saved['contract']['implementation'] == 'old'
    cursor = restore_checkpoint(imported, {'gt': model}, {'gt': optimizer}, current)
    assert cursor[:3] == (saved['epoch'], saved['batch'], saved['updates'])
    assert torch.equal(rng_state()['torch'], saved['rng']['torch'])
    assert import_contract(imported, current, 'old') == (imported, False)
    with pytest.raises(RuntimeError): import_contract(saved, {**current, 'window_batch': 8}, 'old')
    with pytest.raises(RuntimeError): restore_checkpoint(imported, {'gt': model}, {'gt': optimizer}, saved['contract'])
    candidate = dict(execution_fingerprint='new', pass_gate=True, numerical_parity=True,
        baseline_repeat_parity=True, faster_each_pass=True, memory_headroom_ge_10percent=True, speedup=1.2)
    proof = dict(status='complete', source_snapshot_sha256='source', candidates=dict(packed_h2=candidate))
    validate_receipt(proof, 'packed_h2', 'source', 'new')
    for key in ('pass_gate', 'numerical_parity', 'baseline_repeat_parity', 'faster_each_pass', 'memory_headroom_ge_10percent'):
        bad = copy.deepcopy(proof); bad['candidates']['packed_h2'][key] = False
        with pytest.raises(RuntimeError): validate_receipt(bad, 'packed_h2', 'source', 'new')
    with pytest.raises(RuntimeError): validate_receipt(proof, 'packed_h2', 'other_source', 'new')


def test_launcher_original_import_then_new_backend_resume_source_readonly(monkeypatch, tmp_path):
    from test_joint_occ_speed_probe import saved_fixture
    from tools.joint_occ_experiments.batching import run as wrapper
    from tools.joint_occ_experiments.batching.probe import execution_fingerprint
    from tools.joint_occ_experiments.performance.probe import legacy_fingerprint
    import json
    from pathlib import Path
    model, optimizer, provider, row, saved = saved_fixture()
    root = Path(wrapper.__file__).resolve().parents[3]
    saved['contract'].update(implementation=legacy_fingerprint(root), window_batch=4, source_budget=128)
    original = tmp_path/'original'; original.mkdir()
    torch.save(saved, original/'last.pt')
    before = (original/'last.pt').read_bytes()
    fingerprint = execution_fingerprint(root, 2)
    proof = tmp_path/'proof.json'
    proof.write_text(json.dumps(dict(status='complete', source_snapshot_sha256=wrapper.legacy.sha256(original/'last.pt'),
        candidates=dict(packed_h2=dict(execution_fingerprint=fingerprint, pass_gate=True, numerical_parity=True,
            baseline_repeat_parity=True, faster_each_pass=True, memory_headroom_ge_10percent=True, speedup=1.2)))), encoding='utf-8')
    original_step, original_restore, original_hash, original_parser = (wrapper.legacy.train_arm,
        wrapper.legacy.restore_checkpoint, wrapper.legacy.stable_json_fingerprint, wrapper.legacy.parser)
    def fake_main(event, argv):
        args = wrapper.legacy.parser().parse_args(argv)
        state = torch.load(args.resume, weights_only=False)
        current = {**state['contract'], 'implementation': wrapper.legacy.stable_json_fingerprint(
            {'tools/joint_occ_experiments/run.py': 'unchanged_original_code'})}
        assert current['implementation'] == fingerprint
        cursor = wrapper.legacy.restore_checkpoint(state, {'gt': model}, {'gt': optimizer}, current)
        assert cursor[2] == state['updates']
        wrapper.legacy.train_arm(model, optimizer, provider, [row, row], 'gt', update=state['updates']+1, total_updates=7)
        destination = Path(args.out_dir); destination.mkdir(exist_ok=True)
        torch.save(wrapper.legacy.checkpoint({'gt': model}, {'gt': optimizer}, current, 0,
            state['batch']+1, state['updates']+1, {}), destination/'last.pt')
        return 0
    monkeypatch.setattr(wrapper.legacy, 'main', fake_main)
    output = tmp_path/'batched'
    args = []
    for name in ('config', 'train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint',
                 'dataroot', 'train-info', 'dev-info', 'ccr-history-cache', 'ccr-val-history-cache'):
        args += ['--'+name, str(tmp_path/name)]
    args += ['--mode', 'gt', '--out-dir', str(output), '--resume', str(original/'last.pt'), '--batching-proof', str(proof)]
    assert wrapper.main(argv=args) == 0
    assert (original/'last.pt').read_bytes() == before
    assert json.loads((output/'batching_import.json').read_text())['source_updates'] == saved['updates']
    args[args.index('--resume')+1] = str(output/'last.pt')
    assert wrapper.main(argv=args) == 0
    restored = torch.load(output/'last.pt', weights_only=False)
    assert restored['updates'] == saved['updates']+2
    assert wrapper.legacy.train_arm is original_step and wrapper.legacy.restore_checkpoint is original_restore
    assert wrapper.legacy.stable_json_fingerprint is original_hash and wrapper.legacy.parser is original_parser
