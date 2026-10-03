"""Large-batch attention and no-waste recovery of interrupted timing probes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import numpy as np

from real_motion.causal_column_completion import ColumnConfig
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.local_st_world_model import batch_bounded_self_attention, SpatialTemporalBlock
from real_motion.local_training_profile import trial_summary
from real_motion.strong_w2det import StrongW2DetConfig
from tools.real_motion import benchmark_p0_f9_joint_local_warm as bench
from tools.real_motion.local_warm_cache_common import geometry_namespace, warm_causal_cache
from test_causal_geometry_cache import raw_fixture


@pytest.mark.parametrize('history', [4, 6])
@pytest.mark.parametrize('limit', [1, 5, 17, 18])
def test_bounded_attention_preserves_output_and_input_parameter_gradients(history, limit):
    torch.manual_seed(31)
    a = torch.nn.MultiheadAttention(8, 2, batch_first=True).double()
    b = copy.deepcopy(a)
    q = torch.randn(17, history, 8, dtype=torch.float64, requires_grad=True)
    r = q.detach().clone().requires_grad_()
    expected = a(q, q, q, need_weights=False)[0]
    with patch.object(b, 'forward', wraps=b.forward) as calls:
        actual = batch_bounded_self_attention(b, r, limit)
        assert calls.call_count == (17+limit-1)//limit
        assert all(call.args[0].shape[1:] == (history, 8) for call in calls.call_args_list)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    if limit >= 17: assert torch.equal(actual, expected)
    weights = torch.randn_like(expected)
    (expected*weights).sum().backward(); (actual*weights).sum().backward()
    torch.testing.assert_close(q.grad, r.grad, rtol=1e-12, atol=1e-12)
    assert a.state_dict().keys() == b.state_dict().keys()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-11, atol=1e-11)


def test_bounded_attention_rejects_nonindependent_layout_or_stochastic_dropout():
    q = torch.randn(7, 4, 8)
    for kwargs in (dict(batch_first=False), dict(batch_first=True, dropout=.1)):
        with pytest.raises(ValueError, match='batch_first and zero dropout'):
            batch_bounded_self_attention(torch.nn.MultiheadAttention(8, 2, **kwargs), q, 3)
    with pytest.raises(ValueError, match='positive'):
        batch_bounded_self_attention(torch.nn.MultiheadAttention(8, 2, batch_first=True), q, 0)


def test_small_cpu_st_block_keeps_exact_original_path():
    block = SpatialTemporalBlock(8, 2)
    with patch.object(block.temporal_attn, 'forward', wraps=block.temporal_attn.forward) as calls, \
            patch('real_motion.local_st_world_model.batch_bounded_self_attention', side_effect=AssertionError('small path changed')):
        out = block(torch.randn(2, 4, 8, 3, 3, requires_grad=True))
        out.sum().backward()
        assert calls.call_count == 1 and calls.call_args.args[0].shape == (18, 4, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA launch regression requires GPU')
@pytest.mark.parametrize('history', [4, 6])
def test_cuda_st_block_over_sdpa_grid_limit(history):
    block = SpatialTemporalBlock(8, 2).cuda().train()
    # 660 * 10 * 10 = 66000 independent temporal attention batches.
    x = torch.randn(660, history, 8, 10, 10, device='cuda', requires_grad=True)
    with patch('real_motion.local_st_world_model.batch_bounded_self_attention', wraps=batch_bounded_self_attention) as bounded:
        with torch.autocast('cuda', dtype=torch.bfloat16): out = block(x)
        out.float().square().mean().backward()
        torch.cuda.synchronize()
        assert bounded.call_count == 1
        assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


def contract(tmp_path):
    c = dict(protocol=bench.PROTOCOL, out=str(tmp_path), history_frames=4,
        typical_keys=[['s', str(i)] for i in range(16)], stress_keys=[['s', 'dense']], repeats=2)
    weights = dict(generation_pos_weight=1., refine_class_weights=[1., 1., 1.])
    bench.write_json(tmp_path/'diagnostic_weights.json', weights)
    return c, weights


def save_trial(out, c, weights, name, windows, workers=4, persistent=True, status='ok'):
    t = dict(name=name, window_batch=windows, source_budget=windows*32, workers=workers, persistent=persistent)
    if status == 'ok':
        rows = [dict(wall_seconds=.2*windows, windows=windows, sources=windows*10,
            causal_geometry_cache_hits=windows, host_stage_seconds={'prepare': .1}, peak_memory_mib=2., peak_reserved_mib=3.)
            for _ in range(32//windows)]
        row = dict(**t, status=status, rows=rows, measurement=trial_summary(rows), history_frames=4,
            diagnostic_weights=weights, capacity_peak_reserved_mib=3., available_memory_mib=100.)
    else: row = dict(**t, status=status)
    bench.write_json(out/(name+'.json'), row)
    return t


def saved_small_trials(tmp_path):
    c, weights = contract(tmp_path)
    for name, windows, workers, persistent in [('legacy_b4', 4, 4, False), ('pool4_b4', 4, 4, True),
            ('pool6_b4', 4, 6, True), ('pool4_b8', 8, 4, True), ('pool4_b16', 16, 4, True)]:
        save_trial(tmp_path, c, weights, name, windows, workers, persistent)
    return c, weights


def test_finish_existing_only_profiles_never_rewarms_or_retrains_completed_trials(tmp_path):
    c, _ = saved_small_trials(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob('pool*.json')}
    launches = []
    def launch(phase, t):
        launches.append((phase, t['name']))
        assert phase == 'profile'
        (tmp_path/'cpu_profile.txt').write_text('tiny untimed profile')
        return 0
    assert bench.run_contract(c, continuation=True, finish_existing=True, launch=launch) == 0
    assert launches == [('profile', 'pool4_b4')]
    assert all((tmp_path/n).read_bytes() == b for n, b in before.items())
    s = json.loads((tmp_path/'summary.json').read_text())
    assert s['status'] == 'completed_existing_trials_only' and s['cpu_profile_status'] == 'complete'
    assert len(s['reused_trials']) == 5 and s['scientific_epochs'] == 0


def test_continue_only_missing_trial_and_oom_stops_expansion(tmp_path):
    c, weights = saved_small_trials(tmp_path); calls = []
    def launch(phase, t):
        calls.append((phase, t['name']))
        if phase == 'trial':
            assert t['window_batch'] == 32
            save_trial(tmp_path, c, weights, t['name'], 32, status='oom'); return 42
        return 0
    bench.run_contract(c, continuation=True, launch=launch)
    assert calls == [('trial', 'pool4_b32'), ('profile', 'pool4_b4')]
    assert json.loads((tmp_path/'summary.json').read_text())['trials'][-1]['status'] == 'oom'


def test_non_oom_failure_still_reports_all_completed_trials_no_auto_retry(tmp_path):
    c, _ = saved_small_trials(tmp_path); calls = []
    def launch(phase, t):
        calls.append((phase, t['name']))
        raise RuntimeError('trial pool4_b32 invalid configuration argument')
    with pytest.raises(RuntimeError, match='invalid configuration'):
        bench.run_contract(c, continuation=True, launch=launch)
    s = json.loads((tmp_path/'summary.json').read_text())
    assert s['status'] == 'failed_partial' and len(s['trials']) == 5
    assert s['cpu_profile_status'] == 'not_run' and calls == [('trial', 'pool4_b32')]
    assert 'invalid configuration' in (tmp_path/'summary.txt').read_text()


@pytest.mark.parametrize('mutation', ['history', 'weights', 'summary', 'keys', 'warm', 'fingerprint'])
def test_reusing_changed_trial_fails_closed(tmp_path, mutation):
    c, weights = contract(tmp_path)
    t = save_trial(tmp_path, c, weights, 'pool4_b4', 4)
    p = tmp_path/'pool4_b4.json'; row = json.loads(p.read_text())
    if mutation == 'history': row['history_frames'] = 6
    if mutation == 'weights': row['diagnostic_weights']['generation_pos_weight'] = 2.
    if mutation == 'summary': row['measurement']['windows_per_second'] *= 2
    if mutation == 'keys': c['typical_keys'] = c['typical_keys'][:-1]
    if mutation == 'warm': row['rows'][0]['causal_geometry_cache_hits'] = 0
    if mutation == 'fingerprint': row['benchmark_contract_fingerprint'] = 'changed'
    bench.write_json(p, row)
    with pytest.raises(RuntimeError): bench.load_completed_trial(tmp_path, t, c)


def real_continuation_fixture(tmp_path, *, legacy=False):
    c, _ = contract(tmp_path)
    rows = [dict(scene_name='s', t0_token=str(i)) for i in range(16)]
    bench.atomic_checkpoint(tmp_path/'records.pt', dict(typical=rows, stress=[dict(scene_name='s', t0_token='dense')]))
    for n in ('base', 'info'): (tmp_path/n).write_text(n)
    c.update(records_sha=bench.sha256(tmp_path/'records.pt'), base_checkpoint=str(tmp_path/'base'),
        base_sha=bench.sha256(tmp_path/'base'), train_info=str(tmp_path/'info'),
        info_fingerprints={'train': bench.sha256(tmp_path/'info')}, cache_fingerprints={'train': 'cache'},
        config='cfg', dataroot=str(tmp_path), geometry_cache=str(tmp_path/'cache'))
    cfg = {'a': 1}; pcfg = SimpleNamespace(free_label=17, grid=SimpleNamespace(shape_hwd=(5, 5, 3)))
    provider = SimpleNamespace(strong=StrongW2DetConfig(free_label=17), joint=SimpleNamespace(
        transport=SimpleNamespace(config=SimpleNamespace(history_frames=4)), columns=SimpleNamespace(config=ColumnConfig(z_bins=3))))
    namespace = geometry_namespace(cfg, provider, c['info_fingerprints'], c['cache_fingerprints'], c['dataroot'])
    cache = CausalGeometryCache(c['geometry_cache'], namespace, ram_bytes=0, reserve_bytes=0)
    # Produce the report through the REAL cache/prefill writer, not an invented
    # schema containing fields that stats() never supplied on the server.
    raw = raw_fixture(); raw['future_gt_occ'] = None
    def load(source, record, *, include_gt):
        assert not include_gt
        value, _ = cache.get_or_build((record['scene_name'], record['t0_token']), raw,
            lambda: {'memory': np.zeros(3)}, defer_write=True)
        return {**raw, '_column_causal_preparation': value}
    prefill = SimpleNamespace(causal_geometry_cache=cache, joint=torch.nn.Linear(1, 1),
        device=torch.device('cpu'), load_raw_columns=load,
        prepare_columns=lambda *a, **kw: SimpleNamespace(state={'fixed': np.ones(2)}))
    warm = warm_causal_cache(prefill, None, rows+[dict(scene_name='s', t0_token='dense')]); cache.close()
    assert warm['complete'] and warm['windows'] == 17
    assert warm['cache']['directory'] == str(cache.root)
    assert warm['cache']['namespace'] == cache.namespace
    if legacy:
        # Exactly the original producer schema: every statistic, no path or
        # namespace. Preserve that report unchanged across recovery.
        warm['cache'].pop('directory'); warm['cache'].pop('namespace')
    bench.write_json(tmp_path/'warm_cache.json', warm)
    return c, cfg, pcfg, cache


@pytest.mark.parametrize('legacy', [False, True])
def test_actual_prefill_report_continuation_validates_identity_and_namespace(tmp_path, legacy):
    c, cfg, pcfg, _ = real_continuation_fixture(tmp_path, legacy=legacy)
    before = (tmp_path/'warm_cache.json').read_bytes()
    with patch.object(bench, 'load_runtime_config', return_value=cfg), patch.object(bench, 'make_prepare_config', return_value=pcfg):
        assert bench.validate_continuation(c, tmp_path) == c
        assert (tmp_path/'warm_cache.json').read_bytes() == before
        c['history_frames'] = 6
        with pytest.raises(RuntimeError, match='namespace'): bench.validate_continuation(c, tmp_path)
        c['history_frames'] = 4; c['typical_keys'].reverse()
        with pytest.raises(RuntimeError, match='identity/order changed'): bench.validate_continuation(c, tmp_path)


@pytest.mark.parametrize('legacy', [False, True])
def test_empty_or_missing_actual_namespace_is_not_rebuilt(tmp_path, legacy):
    c, cfg, pcfg, cache = real_continuation_fixture(tmp_path, legacy=legacy)
    # Rename within this test's isolated directory; never delete user artifacts.
    saved = cache.root.with_name('saved-cache')
    cache.root.rename(saved)
    with patch.object(bench, 'load_runtime_config', return_value=cfg), patch.object(bench, 'make_prepare_config', return_value=pcfg):
        with pytest.raises(RuntimeError, match='namespace/cache missing'): bench.validate_continuation(c, tmp_path)
        assert not cache.root.exists()
        cache.root.mkdir()
        with pytest.raises(RuntimeError, match='namespace/cache missing'): bench.validate_continuation(c, tmp_path)


def test_legacy_real_artifacts_continue_to_missing_batch_without_rewriting_completed_data(tmp_path):
    c, cfg, pcfg, _ = real_continuation_fixture(tmp_path, legacy=True)
    weights = json.loads((tmp_path/'diagnostic_weights.json').read_text())
    for name, windows, workers, persistent in [('legacy_b4', 4, 4, False), ('pool4_b4', 4, 4, True),
            ('pool6_b4', 4, 6, True), ('pool4_b8', 8, 4, True), ('pool4_b16', 16, 4, True)]:
        save_trial(tmp_path, c, weights, name, windows, workers, persistent)
    bench.write_json(tmp_path/'contract.json', c)
    immutable = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    launches = []
    def launch(phase, t):
        launches.append((phase, t['name']))
        if phase == 'trial':
            save_trial(tmp_path, c, weights, t['name'], t['window_batch'], status='oom'); return 42
        (tmp_path/'cpu_profile.txt').write_text('test-only orchestration profile stub')
        return 0
    # Exercise the real CLI parsing + old-format contract validation + parent
    # recovery together. Only the CUDA child launcher is replaced on CPU.
    real_run = bench.run_contract
    def run_with_cpu_launcher(*args, **kwargs): return real_run(*args, **kwargs, launch=launch)
    with patch('sys.argv', ['benchmark', '--continue-run', str(tmp_path)]), \
            patch.object(bench, 'load_runtime_config', return_value=cfg), \
            patch.object(bench, 'make_prepare_config', return_value=pcfg), \
            patch.object(bench, 'run_contract', side_effect=run_with_cpu_launcher):
        assert bench.main() == 0
    assert launches == [('trial', 'pool4_b32'), ('profile', 'pool4_b4')]
    assert all(p.read_bytes() == old for p, old in immutable.items())
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['reused_trials']) == 5 and summary['status'] == 'complete'


def test_cpu_comparison_reuses_real_population_prior_cache_but_never_old_report(tmp_path):
    original = tmp_path/'original'; original.mkdir()
    c, cfg, pcfg, _ = real_continuation_fixture(original, legacy=True)
    bench.write_json(original/'contract.json', c)
    weights = json.loads((original/'diagnostic_weights.json').read_text())
    before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
    out = tmp_path/'new-cpu-comparison'; calls = []
    def launch(phase, t):
        calls.append((phase, t['name']))
        assert phase != 'warm'
        if phase == 'trial':
            saved_c = json.loads((out/'contract.json').read_text())
            save_trial(out, saved_c, weights, t['name'], t['window_batch'])
            row = json.loads((out/(t['name']+'.json')).read_text()); row.update(t)
            bench.write_json(out/(t['name']+'.json'), row)
        return 0
    with patch.object(bench, 'load_runtime_config', return_value=cfg), patch.object(bench, 'make_prepare_config', return_value=pcfg):
        bench.compare_cpu_paths(original, out, max_window_batch=16, launch=launch)
        with pytest.raises(RuntimeError, match='NEW CPU comparison'): bench.compare_cpu_paths(original, out, launch=launch)
    assert calls == [('trial', 'reference_b4'), ('trial', 'previous_b4'), ('trial', 'optimized_b4'), ('trial', 'optimized_b8'),
        ('trial', 'optimized_b16'), ('profile', 'optimized_b4')]
    assert all(p.read_bytes() == b for p, b in before.items())
    summary = json.loads((out/'summary.json').read_text())
    assert summary['recommendation']['recommended_trial'] == 'optimized_b4'
    assert summary['cpu_comparison']['same_batch4_speedup'] == 1.0
    assert summary['cpu_comparison']['same_batch4_speedup_vs_previous'] == 1.0
    assert (out/'diagnostic_weights.json').read_bytes() == (original/'diagnostic_weights.json').read_bytes()
    assert (out/'records.pt').read_bytes() == (original/'records.pt').read_bytes()


def test_native_comparison_only_two_batch4_children_no_prefill_or_capacity_sweep(tmp_path):
    original = tmp_path/'original'; original.mkdir()
    c, cfg, pcfg, _ = real_continuation_fixture(original, legacy=True)
    bench.write_json(original/'contract.json', c)
    weights = json.loads((original/'diagnostic_weights.json').read_text())
    before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
    out = tmp_path/'native-comparison'; calls = []
    def launch(phase, t):
        calls.append((phase, t['name'], t['backend']))
        assert phase != 'warm' and t['window_batch'] == 4 and t['source_budget'] == 128
        if phase == 'trial':
            saved_c = json.loads((out/'contract.json').read_text())
            save_trial(out, saved_c, weights, t['name'], t['window_batch'])
            row = json.loads((out/(t['name']+'.json')).read_text()); row.update(t)
            row['execution'] = {'native_cpu': {'calls': {'rows': 100}} if t['backend'] == 'native' else None}
            bench.write_json(out/(t['name']+'.json'), row)
        return 0
    with patch.object(bench, 'load_runtime_config', return_value=cfg), patch.object(bench, 'make_prepare_config', return_value=pcfg):
        bench.compare_cpu_paths(original, out, max_window_batch=32, launch=launch, native_compare=True)
    assert calls == [('trial', 'optimized_b4', 'numpy'), ('trial', 'native_b4', 'native'), ('profile', 'optimized_b4', 'numpy')]
    assert all(p.read_bytes() == b for p, b in before.items())
    summary = json.loads((out/'summary.json').read_text())
    assert summary['native_comparison']['speedup'] == 1.0
    assert summary['native_comparison']['native_artifact']['calls']['rows'] == 100


def test_native_bundle_only_two_matched_children_and_does_not_rewrite_previous_artifacts(tmp_path):
    original = tmp_path/'original'; original.mkdir()
    c, cfg, pcfg, _ = real_continuation_fixture(original, legacy=True)
    # Comparing from an earlier comparison must not inherit conflicting flags.
    c['native_comparison'] = True
    bench.write_json(original/'contract.json', c)
    weights = json.loads((original/'diagnostic_weights.json').read_text())
    before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
    out = tmp_path/'native-bundle'; calls = []
    def launch(phase, t):
        calls.append((phase, t['name'], t['backend'], t['bundle']))
        assert phase != 'warm' and t['window_batch'] == 4 and t['source_budget'] == 128
        assert t['workers'] == 4 and t['persistent'] and t['optimize_kernels'] and t['optimize_cpu']
        if phase == 'trial':
            saved_c = json.loads((out/'contract.json').read_text())
            save_trial(out, saved_c, weights, t['name'], 4)
            row = json.loads((out/(t['name']+'.json')).read_text()); row.update(t)
            row['execution'] = {'native_cpu': {'calls': {'support_many': 100} if t['bundle'] else {'support': 100}}}
            for r in row['rows']:
                r.update(full_candidate_columns=600, compact_candidate_columns=600 if t['bundle'] else 0,
                    materialized_candidate_columns=100 if t['bundle'] else 600, compact_descriptor_bytes=18000 if t['bundle'] else 0)
            bench.write_json(out/(t['name']+'.json'), row)
        return 0
    with patch.object(bench, 'load_runtime_config', return_value=cfg), patch.object(bench, 'make_prepare_config', return_value=pcfg):
        bench.compare_cpu_paths(original, out, max_window_batch=128, launch=launch, native_bundle_compare=True)
    assert calls == [('trial','previous_native_b4','native',False), ('trial','optimized_native_b4','native',True),
        ('profile','previous_native_b4','native',False)]
    assert all(p.read_bytes() == b for p,b in before.items())
    summary = json.loads((out/'summary.json').read_text())
    comparison = summary['native_bundle_comparison']
    assert comparison['speedup'] == 1.0 and comparison['population_audit']['full_candidate_columns'] == 4800
    assert comparison['population_audit']['materialized_candidate_columns'] == 800
    assert comparison['native_artifact']['calls']['support_many'] == 100
    assert 'native_comparison' not in summary and 'cpu_comparison' not in summary
    assert (out/'records.pt').read_bytes() == (original/'records.pt').read_bytes()
    assert (out/'diagnostic_weights.json').read_bytes() == (original/'diagnostic_weights.json').read_bytes()


@pytest.mark.parametrize('speedup,verified,oom,expected', [(1.2,6,0,True), (1.01,6,0,False),
    (1.2,0,0,False), (1.2,6,1,False)])
def test_gpu_compare_reuses_artifacts_only_two_children_and_gate_is_honest(tmp_path, speedup, verified, oom, expected):
    original = tmp_path/'original'; original.mkdir()
    c, cfg, pcfg, _ = real_continuation_fixture(original, legacy=True)
    c['native_bundle_comparison'] = True
    bench.write_json(original/'contract.json', c)
    weights = json.loads((original/'diagnostic_weights.json').read_text())
    before = {p:p.read_bytes() for p in original.rglob('*') if p.is_file()}
    out = tmp_path/'gpu'; calls = []
    def launch(phase, t):
        calls.append((phase,t['column_feature_backend']))
        assert phase == 'trial' and t['workers'] == 6 and t['window_batch'] == 4 and t['source_budget'] == 128
        saved = json.loads((out/'contract.json').read_text())
        save_trial(out, saved, weights, t['name'], 4, workers=6)
        row = json.loads((out/(t['name']+'.json')).read_text()); row.update(t)
        row['total_memory_mib'] = 100
        if t['column_feature_backend'] == 'gpu':
            for r in row['rows']:
                r['wall_seconds'] /= speedup
                r.update(gpu_feature_horizons=24,gpu_feature_fallback_horizons=0,
                    gpu_feature_verified_horizons=verified,gpu_feature_oom_windows=oom)
            row['measurement'] = trial_summary(row['rows'])
        bench.write_json(out/(t['name']+'.json'), row)
        return 0
    with patch.object(bench,'load_runtime_config',return_value=cfg), patch.object(bench,'make_prepare_config',return_value=pcfg):
        bench.compare_cpu_paths(original, out, launch=launch, gpu_feature_compare=True)
    assert calls == [('trial','cpu'),('trial','gpu')]
    assert all(p.read_bytes() == data for p,data in before.items())
    summary = json.loads((out/'summary.json').read_text())
    assert summary['gpu_feature_comparison']['pass_gate'] == expected
    assert summary['cpu_profile_status'] == 'not_run'
    assert 'native_bundle_comparison' not in summary and 'cpu_comparison' not in summary
