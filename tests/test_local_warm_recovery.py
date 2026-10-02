"""Large-batch attention and no-waste recovery of interrupted timing probes."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from real_motion.causal_column_completion import ColumnConfig
from real_motion.causal_geometry_cache import PROTOCOL as GEOMETRY_PROTOCOL
from real_motion.local_st_world_model import batch_bounded_self_attention, SpatialTemporalBlock
from real_motion.local_training_profile import trial_summary
from real_motion.strong_w2det import StrongW2DetConfig
from tools.real_motion import benchmark_p0_f9_joint_local_warm as bench
from tools.real_motion.local_warm_cache_common import geometry_namespace


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


def test_legacy_v1_continuation_validates_record_identity_and_geometry_namespace(tmp_path):
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
    directory = Path(c['geometry_cache'])/hashlib.sha256((GEOMETRY_PROTOCOL+namespace).encode()).hexdigest()
    bench.write_json(tmp_path/'warm_cache.json', dict(complete=True, windows=17, cache={'directory': str(directory)}))
    with patch.object(bench, 'load_runtime_config', return_value=cfg), patch.object(bench, 'make_prepare_config', return_value=pcfg):
        assert bench.validate_continuation(c, tmp_path) == c
        c['history_frames'] = 6
        with pytest.raises(RuntimeError, match='namespace changed'): bench.validate_continuation(c, tmp_path)
        c['history_frames'] = 4; c['typical_keys'].reverse()
        with pytest.raises(RuntimeError, match='identity/order changed'): bench.validate_continuation(c, tmp_path)
