import copy
import json
from pathlib import Path
import pytest
import torch

from tools.joint_occ_experiments.performance import probe
from tools.joint_occ_experiments.run import checkpoint, rng_state
from real_motion.strong_w2det import StrongW2DetConfig
from test_joint_occ_ego import fixture


def saved_fixture():
    model, optimizer, provider, row = fixture()
    provider.strong = StrongW2DetConfig(free_label=17)
    # Nonempty Adam moments: this must preserve a real resumed state, not only init.
    probe.train_arm(model, optimizer, provider, [row], 'gt', update=1, total_updates=7)
    contract = dict(weights=[1., 1., 1.], total_updates=7)
    saved = copy.deepcopy(checkpoint({'gt': model}, {'gt': optimizer}, contract, 0, 1, 1, {}))
    return model, optimizer, provider, row, saved


def test_actual_read_only_probe_resets_same_adam_rng_and_allows_no_published_updates():
    model, optimizer, provider, row, saved = saved_fixture()
    original = copy.deepcopy(saved)
    result = probe.measure(model, optimizer, provider, saved, [[row], [row, row]], progress=lambda *a, **k: None)
    assert set(result['timed_passes']) == set(probe.BACKENDS)
    assert all(len(passes) == 2 for passes in result['timed_passes'].values())
    assert all(p['windows'] == 3 for passes in result['timed_passes'].values() for p in passes)
    assert all(p['numerical_parity'] for p in result['final_state_parity'].values())
    assert result['source_weights_optimizer_rng_unchanged']
    assert result['route'] == 'probe_only_no_automatic_training_backend_change'
    assert probe.assert_close_state(model.state_dict(), saved['models']['gt']) == 0
    assert probe.assert_close_state(optimizer.state_dict(), saved['optimizers']['gt']) == 0
    for key in ('models', 'optimizers', 'contract', 'epoch', 'batch', 'updates'):
        assert probe.assert_close_state(saved[key], original[key]) == 0
    assert torch.equal(rng_state()['torch'], original['rng']['torch'])


def test_probe_rejects_empty_and_one_sided_measurements():
    model, optimizer, provider, row, saved = saved_fixture()
    for batches, passes in (([], 2), ([[row]], 1)):
        with pytest.raises(ValueError, match='two alternating'):
            probe.measure(model, optimizer, provider, saved, batches, passes=passes)


def test_report_retains_original_budget_and_exact_failure_path():
    original = dict(model={'bias': torch.zeros(128)})
    changed = copy.deepcopy(original)
    changed['model']['bias'][7] = 2.2003543563187122e-5
    report = probe.state_parity(original, changed)
    assert report['numerical_parity'] is False and report['mismatched_elements'] == 1
    assert report['budgets'] == dict(rtol=2e-4, atol=2e-5)
    assert report['worst_tensors'][0]['path'] == 'state.model.bias'
    assert report['max_state_abs_difference'] > 2e-5
    with pytest.raises(RuntimeError, match='structure/dtype'):
        probe.state_parity(original, dict(model={'bias': torch.zeros(127)}))


def test_numeric_failure_finishes_all_timings_but_never_passes_gate(monkeypatch):
    from tools.joint_occ_experiments import engine
    model, optimizer, provider, row, saved = saved_fixture()
    real_step, original_loss = probe.train_arm, engine.occupancy_loss
    def drift(*args, **kwargs):
        stats = real_step(*args, **kwargs)
        if engine.occupancy_loss is not original_loss and not kwargs['probe'] and kwargs['update'] == saved['updates']+2:
            with torch.no_grad(): next(model.transport.parameters()).flatten()[7] += .01
        return stats
    monkeypatch.setattr(probe, 'train_arm', drift)
    result = probe.measure(model, optimizer, provider, saved, [[row], [row]], progress=lambda *a, **k: None)
    assert all(len(rows) == 2 for rows in result['timed_passes'].values())
    for name in probe.BACKENDS[1:]:
        assert result['warmup'][name]['loss_gradient_audit']['pass_gate']
        assert not result['candidates'][name]['numerical_parity']
        assert not result['candidates'][name]['pass_gate']
        assert result['timed_passes'][name][0]['final_state_parity']['mismatched_elements'] > 0
    assert probe.assert_close_state(model.state_dict(), saved['models']['gt']) == 0
    assert probe.assert_close_state(optimizer.state_dict(), saved['optimizers']['gt']) == 0


def test_cli_source_snapshot_and_cursor_unmodified(monkeypatch, tmp_path):
    model, optimizer, provider, row, saved = saved_fixture()
    root = Path(probe.__file__).resolve().parents[3]
    paths = {name: tmp_path/(name+'.data') for name in ('config', 'train_cache', 'dev_cache',
        'population_manifest', 'base_checkpoint', 'train_info', 'dev_info')}
    for path in paths.values(): path.write_bytes(b'original immutable input')
    saved['contract'].update(modes=['gt'], training_backend='fused', model=model.configs(),
        implementation=probe.legacy_fingerprint(root), seed=27, epochs=1, window_batch=1, source_budget=128,
        lr=[5e-4, 3e-4, 3e-4], train_keys=[['TRAIN', 't0'], ['TRAIN', 't1']], total_updates=2,
        data={name: probe.legacy.sha256(path) for name, path in paths.items() if name not in ('config', 'population_manifest')},
        cache_namespace=dict(train='TRAIN_NAMESPACE', val='VAL_NAMESPACE'),
        execution=dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda))
    source = tmp_path/'original_gt'
    source.mkdir(); torch.save(saved, source/'last.pt')
    source_bytes = (source/'last.pt').read_bytes()
    record2 = dict(row[0], t0_token='t1')
    monkeypatch.setattr(probe.legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(probe, 'JointOccEgo', lambda config, width: model)
    monkeypatch.setattr(probe.legacy, 'PilotProvider', lambda *args: provider)
    monkeypatch.setattr(probe.legacy, 'load_runtime_config', lambda *args: None)
    monkeypatch.setattr(probe.legacy, 'make_prepare_config', lambda _: provider.pcfg)
    monkeypatch.setattr(probe.legacy, 'CachedColumnSource', lambda *args, **kwargs: object())
    def source_reader(dataroot, *, info_pkl):
        assert dataroot == str(tmp_path) and info_pkl == str(paths['train_info'])
    monkeypatch.setattr(probe.legacy, 'NuScenesWindowSource', source_reader)
    monkeypatch.setattr(probe.legacy, 'load_cache', lambda _: ({}, [row[0], record2]))
    monkeypatch.setattr(probe.legacy, 'record_keys', lambda _: None)
    monkeypatch.setattr(probe.legacy, 'align_records', lambda records, keys: records)
    monkeypatch.setattr(probe.legacy, 'epoch_groups', lambda records, *args: [[r] for r in records])
    monkeypatch.setattr(probe, 'attach_caches', lambda *args: dict(train={'namespace': 'TRAIN_NAMESPACE'}, val={'namespace': 'VAL_NAMESPACE'}))
    monkeypatch.setattr(probe, 'prefetched_batches', lambda p, s, jobs, d, **kw: ([row] for _ in jobs))
    class Execution:
        def __init__(self, *args): pass
        def close(self): pass
        def map(self, evidence, prep, grid):
            from real_motion.canonical_causal_repair import map_canonical_evidence
            return map_canonical_evidence(evidence, prep, grid)
    import real_motion.canonical_repair_execution as canonical
    import real_motion.surface_projection_execution as surface
    import real_motion.strong_majority_execution as majority
    monkeypatch.setattr(canonical, 'CanonicalCpuExecution', Execution)
    monkeypatch.setattr(surface, 'SurfaceMapExecution', Execution)
    monkeypatch.setattr(majority, 'ParallelNativeMajority', Execution)
    out = tmp_path/'speed_only'
    args = [item for name, path in paths.items() for item in ('--'+name.replace('_', '-'), str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path/'cache_train'),
             '--ccr-val-history-cache', str(tmp_path/'cache_val'), '--resume', str(source/'last.pt'),
             '--out-dir', str(out), '--probe-steps', '1', '--mode', 'gt']
    assert probe.main(args) == 0
    assert (source/'last.pt').read_bytes() == source_bytes
    assert (out/'source_snapshot.pt').read_bytes() == source_bytes
    assert not (out/'last.pt').exists()
    result = json.loads((out/'speed.json').read_text())
    assert result['scientific_updates_published'] == 0 and result['source_completed_updates'] == 1
    assert result['status'] == 'complete' and (out/'summary.txt').is_file()
