"""Real small-grid execution parity/gradients/benchmark, NOT L40S throughput."""
import copy
import json
from pathlib import Path
import threading
from dataclasses import asdict
from types import SimpleNamespace
import pytest
import torch
from test_shared_semantic_refiner import fixture
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.refine import engine
from tools.shared_history_experiments.refine.geometry import aligned_memory
from tools.shared_history_experiments.refine.model import SemanticRefiner
from tools.shared_history_experiments.refiner_execution.optimized import BACKENDS, Coordinates, execution, projected_memory, _SharedQueryCast
from tools.shared_history_experiments.refiner_execution.probe import measure, tensor_check
from tools.shared_history_experiments.refiner_execution import probe


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('backend', BACKENDS[1:])
def test_forward_full6_finite_and_motion_gradients(device, backend):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1); torch.manual_seed(21)
    model, _, provider, row = fixture(device)
    model.columns.chunk_size = 17
    with torch.no_grad(): model.columns.readout[-1].weight.normal_(std=.02)
    expected = engine.forward_window(model, provider, row)
    loss, _, _ = engine.semantic_loss(expected['logits'], row[1]['future_gt_occ'])
    target = [expected['output']['residual_xy_m'], expected['memory'], expected['field']]
    grads = torch.autograd.grad(loss.mean(), target)
    with execution(backend):
        actual = engine.forward_window(model, provider, row)
        loss, _, _ = engine.semantic_loss(actual['logits'], row[1]['future_gt_occ'])
        actual_grads = torch.autograd.grad(loss.mean(),
            [actual['output']['residual_xy_m'], actual['memory'], actual['field']])
    torch.testing.assert_close(actual['logits'], expected['logits'], rtol=2e-4, atol=2e-5)
    assert torch.equal(actual['logits'].argmax(1), expected['logits'].argmax(1))
    for a, b in zip(actual_grads, grads):
        assert torch.isfinite(a).all() and a.norm() > 0
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)


@pytest.mark.parametrize('batch_horizons', [False, True])
def test_differentiable_poses_targets_and_yaws(batch_horizons):
    model, _, provider, row = fixture()
    fw = engine.forward_window(model, provider, row)
    poses = fw['poses'].detach().requires_grad_()
    targets = fw['targets'].detach().requires_grad_()
    yaws = fw['yaws'].detach().requires_grad_()
    lifted = model.columns.lift_history(fw['field'])
    args = (model.columns, lifted, fw['output'], row[3], row[1], targets, yaws, poses, provider.pcfg.grid)
    expected = aligned_memory(*args)
    actual = projected_memory(*args, batch_horizons=batch_horizons)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    weight = torch.arange(actual.numel()).reshape(actual.shape).float()/actual.numel()
    first = torch.autograd.grad((expected*weight).sum(), [poses, targets, yaws], retain_graph=True)
    second = torch.autograd.grad((actual*weight).sum(), [poses, targets, yaws])
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)
    assert second[0].norm() > 0 and second[1].norm() > 0


def test_cache_bounded_no_learned_features_and_hooks_restored():
    cache = Coordinates()
    a = cache.get((8, 8, 4), torch.device('cpu'), 17)
    assert cache.get((8, 8, 4), torch.device('cpu'), 17) is a
    assert all(not t.requires_grad for chunk in a for t in chunk)
    cache.get((10, 8, 4), torch.device('cpu'), 17)
    cache.get((12, 8, 4), torch.device('cpu'), 17)
    assert len(cache.rows) == 2
    old_forward, old_memory = SemanticRefiner.forward, engine.aligned_memory
    with pytest.raises(RuntimeError):
        with execution('memory_h6'):
            assert engine.aligned_memory is not old_memory
            raise RuntimeError('failed experimental step')
    assert SemanticRefiner.forward is old_forward and engine.aligned_memory is old_memory


def test_probe_real_adam_and_rng_restored_no_state_snapshots_on_gpu():
    torch.set_num_threads(1); torch.manual_seed(21)
    model, adam, provider, row = fixture()
    model.columns.chunk_size = 17
    # Use already learned readout, as real frozen epoch2 does.
    with torch.no_grad(): model.columns.readout[-1].weight.normal_(std=.02)
    engine.train_arm(model, adam, provider, [row], 'gt', update=1, total_updates=4)
    before = copy.deepcopy(model.state_dict()); original_adam = copy.deepcopy(adam.state_dict())
    rng = legacy.rng_state()
    saved = dict(models={'gt': before}, optimizers={'gt': original_adam}, rng=rng, updates=1,
        contract=dict(weights=[1., 1., 1.], total_updates=4))
    report = measure(model, adam, provider, saved, [[row], [row]], passes=2,
        progress=lambda *a, **k: None)
    assert report['scientific_updates_published'] == 0 and not report['automatic_promotion']
    assert set(report['seconds_per_window']) == set(BACKENDS)
    assert all(len(r) == 2 for r in report['timed_passes'].values())
    assert all(report['warmup'][n]['loss']['pass_gate'] for n in BACKENDS)
    for key, value in before.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for key, values in original_adam['state'].items():
        for name, value in values.items(): torch.testing.assert_close(adam.state_dict()['state'][key][name], value, rtol=0, atol=0)
    assert torch.equal(legacy.rng_state()['torch'], rng['torch'])
    assert all(not c['pass_gate'] for c in report['candidates'].values())  # CPU cannot validate CUDA headroom


def test_probe_interrupted_restores_weights_and_hook():
    model, adam, provider, row = fixture()
    before = copy.deepcopy(model.state_dict()); old = SemanticRefiner.forward
    saved = dict(models={'gt': before}, optimizers={'gt': copy.deepcopy(adam.state_dict())},
        rng=legacy.rng_state(), updates=0, contract=dict(weights=[1.,1.,1.], total_updates=4))
    event = threading.Event(); event.set()
    with pytest.raises(InterruptedError): measure(model, adam, provider, saved, [[row]], stop_event=event)
    assert SemanticRefiner.forward is old
    for key, value in before.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_numerical_failure_is_fail_not_exception_or_tolerance_waiver():
    report = tensor_check({'g': torch.ones(4)}, {'g': torch.zeros(4)})
    assert not report['pass_gate'] and report['mismatched_elements'] == 4
    assert report['budgets'] == dict(rtol=3e-5, atol=2e-7)


def test_shared_cast_preserves_per_chunk_BF16_rounding_not_naive_gradient_sum():
    value = torch.arange(8).bfloat16().requires_grad_()
    weights = [torch.full((8,), v) for v in (1.2345, -1.2306, .00003)]
    reference = torch.autograd.grad(sum((value.float()*w).sum() for w in weights), value)[0]
    cached = value.detach().float()
    candidate = torch.autograd.grad(sum((_SharedQueryCast.apply(value, cached)*w).sum() for w in weights), value)[0]
    converted = value.float()
    naive = torch.autograd.grad(sum((converted*w).sum() for w in weights), value)[0]
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    assert not torch.equal(naive, reference)


def test_real_cli_reads_frozen_source_no_original_changes_no_DEV_rescoring(tmp_path, monkeypatch):
    from test_shared_semantic_refiner import test_real_cli_new_protocol_epoch_snapshots_resume_and_eval_readonly
    # Reuse the real small-network CLI harness: only the nuScenes dataset is mocked.
    test_real_cli_new_protocol_epoch_snapshots_resume_and_eval_readonly(tmp_path, monkeypatch)
    source = tmp_path/'pilot/epoch_0002.pt'
    receipts = {p: p.read_bytes() for p in (source, tmp_path/'pilot/last.pt', tmp_path/'pilot/summary.txt')}
    out = tmp_path/'speed'
    args = [value for name, path in (
        ('config', tmp_path/'config.yaml'), ('base-checkpoint', tmp_path/'base.pt'),
        ('train-cache', tmp_path/'train-cache'), ('dev-cache', tmp_path/'dev-cache'),
        ('population-manifest', tmp_path/'population-manifest'), ('dataroot', tmp_path),
        ('train-info', tmp_path/'train-info'), ('dev-info', tmp_path/'dev-info'),
        ('ccr-history-cache', tmp_path), ('ccr-val-history-cache', tmp_path),
        ('out-dir', out), ('resume', source)) for value in ('--'+name, str(path))]
    assert probe.main(argv=args+['--probe-steps', '2', '--profile']) == 0
    report = json.loads((out/'speed.json').read_text())
    assert report['scientific_updates_published'] == 0
    assert report['source_checkpoint'] == str(source)
    assert report['source_update'] == 4
    assert all(k[0] != 'DEV' for k in json.loads((out/'population.json').read_text()))
    assert report['profiles']['reference']['top_cpu']
    assert (out/'reference_trace.json').is_file()
    assert not (out/'last.pt').exists()
    for path, value in receipts.items(): assert path.read_bytes() == value
