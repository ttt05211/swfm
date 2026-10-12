"""Real small-grid forward/backward and recovery, NOT nuScenes quality scores."""
import copy
import threading
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_shared_history_pilot import fixture as old_fixture
from tools.shared_history_experiments.refine.model import ResidualJoint
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.shared_history_experiments.refine import engine
from tools.shared_history_experiments.refine.geometry import aligned_memory
from tools.joint_occ_experiments.batching.geometry import GroupedScene
from tools.joint_occ_experiments.data import prepare_projection
from tools.joint_occ_experiments import run as legacy


def fixture(device='cpu'):
    old, _, provider, row = old_fixture()
    provider.device = torch.device(device)
    model = ResidualJoint(old.transport.config, feature_width=16, z_bins=4,
                          memory_width=4, width=8, chunk_size=32).to(device)
    provider.joint = model
    history = ResidualHistoryInputs(provider, ram_mib=1).get(row[0], row[1])
    row = (*row[:3], history)
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=1e-3, initial_lr=1e-3)])
    return model, optimizer, provider, row


def test_zero_init_full6_transport_parity_and_one_encoder(monkeypatch):
    model, _, p, row = fixture(); calls = []
    hook = model.transport.history_encoder.register_forward_hook(lambda *a: calls.append(1))
    fw = engine.forward_window(model, p, row); hook.remove()
    assert len(calls) == 1 and fw['memory'].shape == (6, 4, 4, 4, 4)
    torch.testing.assert_close(fw['logits'], model.columns.prior_logits(fw['prior']), rtol=0, atol=0)
    assert torch.equal(fw['logits'].argmax(1), fw['prior'].argmax(1))
    # Compare existing edits=False continuous Transport, not its hard raster.
    _, _, _, old_row = old_fixture()
    prep, evidence, plan = prepare_projection(p, row[0], row[1], fw['output'],
        row[1]['future_poses'], old_row[3], use_gt_cache=True)
    scene = GroupedScene(prep, p.pcfg.grid, p.device)
    expected = scene.render_group([0, 1, 2], fw['poses'], fw['targets'], fw['yaws'],
        torch.zeros(len(evidence), 6, 3), evidence, plan, torch.zeros(len(evidence), 6, 2), edits=False)
    torch.testing.assert_close(fw['prior'][:3], expected, atol=2e-6, rtol=2e-6)


def test_history_no_surface_or_canonical_abi_and_future_gt_poison(monkeypatch):
    model, _, p, row = fixture()
    raw = copy.deepcopy(row[1]); raw['future_gt_occ'] = 'POISON'; raw['future_masks'] = 'POISON'
    raw['_column_causal_preparation']['_ccr_compact_support'] = 'POISON'
    raw['_column_causal_preparation']['_ccr_compact_conflicts'] = 'POISON'
    monkeypatch.setattr('real_motion.surface_canonical_repair.SurfaceAtlas.from_compact', lambda *a: pytest.fail('Surface used'))
    history = ResidualHistoryInputs(p, ram_mib=0).build(raw)
    assert history[1] is None and 'support' not in history[0]
    expected = engine.forward_window(model, p, row)
    actual = engine.forward_window(model, p, (*row[:1], raw, row[2], history))
    torch.testing.assert_close(actual['logits'], expected['logits'], rtol=0, atol=0)


def test_full_grid_relabel_and_chunk_numerical_parity():
    model, _, p, row = fixture()
    with torch.no_grad():
        model.columns.readout[-1].weight.normal_(std=.01)
        model.columns.readout[-1].bias[16] = 20.
    first = engine.forward_window(model, p, row)
    model.columns.chunk_size = 4096
    second = engine.forward_window(model, p, row)
    torch.testing.assert_close(first['logits'], second['logits'], rtol=2e-5, atol=2e-6)
    # A class outside old road/sidewalk/source candidates is editable everywhere.
    assert torch.all(second['logits'].argmax(1) == 16)


def test_memory_permutation_invariant_and_motion_pose_gradients():
    model, _, p, row = fixture(); fw = engine.forward_window(model, p, row)
    field = fw['field']; lifted = model.columns.lift_history(field)
    poses = fw['poses'].clone().requires_grad_()
    targets = fw['targets'].detach().clone().requires_grad_()
    yaws = fw['yaws'].detach().clone().requires_grad_()
    memory = aligned_memory(model.columns, lifted, fw['output'], row[3], row[1], targets, yaws, poses, p.pcfg.grid)
    other_history = copy.deepcopy(row[3])
    reads = other_history[0]['memory_reads']; order = np.arange(len(reads['world']))[::-1]
    for key in ('world', 'actor', 'frame', 'cell', 'coverage'): reads[key] = reads[key][order].copy()
    other = aligned_memory(model.columns, lifted, fw['output'], other_history, row[1], targets, yaws, poses, p.pcfg.grid)
    torch.testing.assert_close(memory, other, rtol=2e-5, atol=2e-6)
    weight = torch.arange(memory.numel(), device=memory.device).reshape(memory.shape).float()/memory.numel()
    (memory*weight).sum().backward()
    assert targets.grad.norm() > 0 and poses.grad.norm() > 0


def test_unknown_not_observed_free_and_empty_sources():
    model, _, p, row = fixture()
    raw = copy.deepcopy(row[1]); raw['history_observed'][:] = False
    history = ResidualHistoryInputs(p, ram_mib=0).build(raw)
    assert not history[0]['memory_reads']['static_valid'].any()
    assert not len(history[0]['memory_reads']['world'])
    fw = engine.forward_window(model, p, (row[0], raw, row[2], history))
    torch.testing.assert_close(fw['memory'], model.columns.empty[None, :, None, None, None].expand_as(fw['memory']))
    record = copy.deepcopy(row[0]); raw = copy.deepcopy(row[1])
    for k in ('features', 'source_class_id', 'source_centroid_xy_t0_m', 'anchors_xy_t0_m', 'kta_displacement_xy_m'):
        record[k] = record[k][:0]
    state = raw['_column_causal_preparation']['prepared_state']
    for k in ('current', 'source_world_points', 'source_rel_xy'): state[k] = []
    state['source_z_t0'] = np.empty(0)
    raw['_column_causal_preparation']['registrations'] = []
    history = ResidualHistoryInputs(p, ram_mib=0).build(raw)
    actual = engine.forward_window(model, p, (record, raw, row[2], history))
    assert torch.isfinite(actual['logits']).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_full_occ_gradients_and_exact_cpu_adam_rng_recovery(device):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('CUDA/BF16 unavailable')
    model, adam, p, row = fixture(device)
    before = copy.deepcopy(model.state_dict())
    first = engine.train_arm(model, adam, p, [row], 'gt', update=1, total_updates=4, probe=True)
    assert first['training_backend'] == 'shared_full6_semantic_residual'
    assert not torch.equal(before['columns.readout.4.weight'], model.state_dict()['columns.readout.4.weight'])
    # Zero last layer initially blocks upstream refiner gradients; after one update they open.
    old_protocol = legacy.PROTOCOL
    try:
        legacy.PROTOCOL = 'TEST_RESIDUAL'
        contract = dict(epochs=2)
        saved = copy.deepcopy(legacy.checkpoint({'gt': model}, {'gt': adam}, contract, 0, 1, 1, {}))
        second = engine.train_arm(model, adam, p, [row], 'gt', update=2, total_updates=4, probe=True)
        assert second['gradient_links']['occupancy_to_memory'] > 0
        assert second['gradient_links']['occupancy_to_shared_history'] > 0
        assert second['gradient_links']['occupancy_to_xy'] > 0
        assert not torch.equal(before['columns.lift.weight'], model.state_dict()['columns.lift.weight'])
        if device == 'cpu':
            expected, expected_adam = copy.deepcopy(model.state_dict()), copy.deepcopy(adam.state_dict())
            legacy.restore_checkpoint(saved, {'gt': model}, {'gt': adam}, contract)
            engine.train_arm(model, adam, p, [row], 'gt', update=2, total_updates=4)
            for k, v in expected.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
            for k, values in expected_adam['state'].items():
                for name, v in values.items(): torch.testing.assert_close(adam.state_dict()['state'][k][name], v, rtol=0, atol=0)
    finally: legacy.PROTOCOL = old_protocol


def test_integer_eval_resume_readonly_weights_and_future_labels_after_prediction(monkeypatch):
    model, _, p, row = fixture()
    monkeypatch.setattr(engine, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    before = copy.deepcopy(model.state_dict()); first_states = []
    engine.evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[row]], save=lambda s: first_states.append(copy.deepcopy(s)))
    saved = copy.deepcopy(first_states[-1]); untouched = copy.deepcopy(saved)
    resumed, full = [], []
    engine.evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[row], [row]], saved=saved,
                    save=lambda s: resumed.append(copy.deepcopy(s)))
    engine.evaluate({'gt': model}, p, SimpleNamespace(nusc=None), [[row], [row]], save=lambda s: full.append(copy.deepcopy(s)))
    assert resumed[-1]['metrics'] == full[-1]['metrics'] and saved == untouched
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def test_semantic_loss_full_grid_ignore_free_and_gradient():
    logits = torch.randn(6, 18, 8, requires_grad=True)
    labels = np.full((6, 8), 17); labels[:, 0] = 4; labels[:, 1] = 255; labels[:, 2] = -1
    losses, ce, lovasz = engine.semantic_loss(logits, labels)
    assert torch.isfinite(losses).all() and torch.all(lovasz > 0)
    losses.mean().backward()
    assert torch.count_nonzero(logits.grad[:, :, 1:3]) == 0
    assert logits.grad[:, :, 0].norm() > 0


def test_real_cli_new_protocol_epoch_snapshots_resume_and_eval_readonly(tmp_path, monkeypatch):
    from tools.shared_history_experiments.refine import run
    from tools.shared_history_experiments.refine.model import PROTOCOL
    model, _, p, row = fixture()
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
    monkeypatch.setattr(legacy, 'make_prepare_config', lambda _: p.pcfg)
    def provider(*args): p.joint = args[-2]; return p
    monkeypatch.setattr(legacy, 'PilotProvider', provider)
    monkeypatch.setattr(legacy, 'NuScenesWindowSource', lambda root, *, info_pkl: SimpleNamespace(nusc=None))
    class Disposable:
        kernels = pool = None
        def __init__(self, *a): pass
        def close(self): pass
    monkeypatch.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
    monkeypatch.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
    def caches(provider, *a):
        provider.ccr_history_cache = provider.ccr_val_history_cache = SimpleNamespace(stats=lambda: {}, close=lambda: None)
        return {'train': {'namespace': 'train'}, 'val': {'namespace': 'val'}}
    monkeypatch.setattr(legacy, 'attach_caches', caches)
    def groups(records, *a):
        assert len(records) == 4086
        return [[records[0]], [records[1]]]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    monkeypatch.setattr(legacy, 'prefetched_batches', lambda provider, source, groups, descriptors, **k:
        ([(record, row[1], row[2], row[3]) for record in group] for group in groups))
    original_model = run.ResidualJoint
    monkeypatch.setattr(run, 'ResidualJoint', lambda cfg: original_model(cfg, feature_width=16, z_bins=4,
        memory_width=4, width=8, chunk_size=32))
    # Actual renderer/network/CE/Lovasz/backward runs; dataset only is simulated.
    monkeypatch.setattr(engine, 'evaluate', lambda *a, **k: dict(windows=512, variants={}, trajectory={}))
    out = tmp_path/'pilot'
    args = ['--config', str(config), '--base-checkpoint', str(base), '--dataroot', str(tmp_path),
        '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path), '--out-dir', str(out)]
    args += [value for name, path in paths.items() for value in ('--'+name, str(path))]
    from tools.shared_history_experiments.refine.experiment import probe
    probe_dir = tmp_path/'discarded_probe'
    probe_args = args.copy(); probe_args[probe_args.index(str(out))] = str(probe_dir)
    original_train, original_save = engine.train_arm, legacy.save_resume_checkpoint
    # Real CPU training steps run, but CPU cannot satisfy the production CUDA gate.
    assert probe.main(threading.Event(), probe_args+['--device', 'cpu'], warmup=2, measured=2) == 1
    assert not list(probe_dir.glob('*.pt'))
    import json
    probe_report = json.loads((probe_dir/'probe.json').read_text())
    assert probe_report['scientific_updates_published'] == 0 and probe_report['measured_updates'] == 2
    assert engine.train_arm is original_train and legacy.save_resume_checkpoint is original_save
    assert run.main(threading.Event(), args+['--max-updates', '2']) == 0
    saved = torch.load(out/'last.pt', weights_only=False)
    assert saved['protocol'] == PROTOCOL and saved['epoch'] == 1 and saved['updates'] == 2
    assert saved['contract']['model']['repair']['action_heads'] is False
    assert (out/'epoch_0001.pt').is_file()
    assert run.main(threading.Event(), args+['--resume', str(out/'last.pt')]) == 0
    assert (out/'epoch_0002.pt').is_file()
    last, summary = (out/'last.pt').read_bytes(), (out/'summary.txt').read_bytes()
    evaluation = args.copy(); evaluation[evaluation.index(str(out))] = str(out/'eval')
    assert run.main(threading.Event(), evaluation+['--resume', str(out/'epoch_0001.pt'), '--evaluate-only']) == 0
    assert (out/'last.pt').read_bytes() == last and (out/'summary.txt').read_bytes() == summary
    saved['protocol'] = 'OLD_CCR'
    wrong = out/'old.pt'; torch.save(saved, wrong)
    with pytest.raises(RuntimeError, match='differs'): run.main(threading.Event(), args+['--resume', str(wrong)])
