"""CPU throughput changes must preserve labels, draws, live poses and gradients."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from real_motion.causal_column_sampling import ColumnFeatureSampler, ColumnHistoryIndex
from real_motion.joint_causal_columns import JointCausalColumns
from real_motion.strong_w2det import StrongW2DetConfig
from tools.real_motion import causal_column_common as columns
from tools.real_motion import joint_column_common as common
from tools.real_motion import joint_column_full_common as full
from test_joint_causal_columns import fixture, optimizers, provider_for
from test_causal_column_sampling import fixture as sampling_fixture


@pytest.mark.parametrize('history', [4, 6])
@pytest.mark.parametrize('seed', range(4))
def test_sparse_features_bit_exact_reference_all_coordinates_and_ownership(history, seed):
    prep, grid, cfg, plan = sampling_fixture(seed)
    for k in ('history_occ', 'history_observed', 'history_poses'): prep.raw[k] = prep.raw[k][-history:]
    prep.registrations = [r[-history:] for r in prep.registrations]
    small = plan.subset(np.r_[np.arange(12), np.arange(405, 416), np.arange(800, 819)])
    fast = ColumnFeatureSampler(prep, 3, small, grid, cfg, columns.pose_motion, max_cache_mib=0)
    actual = fast.sample(small, columns.sample_column_features)
    expected = columns.sample_column_features(prep, 3, small, grid, cfg)
    assert all(np.array_equal(v, actual[k]) for k, v in expected.items())
    prep.cpu_pipeline_optimized = False
    reference = ColumnFeatureSampler(prep, 3, small, grid, cfg, columns.pose_motion, max_cache_mib=0).sample(small, columns.sample_column_features)
    assert all(np.array_equal(v, reference[k]) for k, v in actual.items())


@pytest.mark.parametrize('limit', [0, .001, 8])
def test_membership_index_bounds_empty_duplicates_and_sorted_fallback(limit):
    prep, grid, _, _ = sampling_fixture(2)
    prep.registrations.append([(np.eye(4), np.empty((0, 3), np.int64))]*6)
    index = ColumnHistoryIndex(prep, grid, max_membership_mib=limit)
    for actor in (0, 1):
        for f, reg in enumerate(prep.registrations[actor]):
            if reg is None: continue
            owned = np.unique(np.ravel_multi_index(reg[1].T, grid.shape_hwd))
            test = np.r_[-100, np.arange(np.prod(grid.shape_hwd)), owned[::-1], 1000000]
            assert np.array_equal(index.contains(actor, f, test), np.isin(test, owned))
    assert index.table_bytes <= int(limit*2**20)


def test_selected_horizons_share_one_index_without_gt_or_pose_caching():
    prep, grid, joint, _, _ = fixture()
    selected = common.select_online_columns(prep, joint.columns.config, grid, np.random.default_rng(44))
    expected = [common.sample_online_column(prep, row, grid, joint.columns.config) for row in selected]
    with patch('real_motion.causal_column_sampling.ColumnHistoryIndex', wraps=ColumnHistoryIndex) as factory:
        actual = common.sample_online_columns(prep, selected, grid, joint.columns.config)
        assert factory.call_count == 1
    assert all(np.array_equal(a[k], b[k]) for a, b in zip(expected, actual) for k in a)
    assert not hasattr(prep.column_history_index, 'prepared') and not hasattr(prep.column_history_index, 'targets')
    prep.targets = [[p+np.array([1., .5, 0.]) for p in frame] for frame in prep.targets]
    prep.yaws = [[yaw+.13 for yaw in frame] for frame in prep.yaws]
    changed = common.sample_online_columns(prep, selected, grid, joint.columns.config)
    for row, arrays in zip(selected, changed):
        expected = columns.sample_column_features(prep, row[0], row[1], grid, joint.columns.config)
        assert all(np.array_equal(expected[k], arrays[k]) for k in expected)


@pytest.mark.parametrize('history', [4, 6])
def test_fast_candidates_every_field_labels_and_rng_equal_reference(history):
    prep, grid, joint, _, _ = fixture()
    for k in ('history_occ', 'history_observed', 'history_poses'): prep.raw[k] = prep.raw[k][-history:]
    prep.registrations = [r[-history:] for r in prep.registrations]
    slow = copy.deepcopy(prep); slow.cpu_pipeline_optimized = False
    fast = copy.deepcopy(prep); fast.cpu_pipeline_optimized = True
    a = common.build_online_column_candidates(slow, joint.columns.config, grid)
    b = common.build_online_column_candidates(fast, joint.columns.config, grid)
    for (ha, pa, ya), (hb, pb, yb) in zip(a, b):
        assert ha == hb and np.array_equal(ya, yb)
        assert all(np.array_equal(v, getattr(pb, k)) for k, v in vars(pa).items())
    x, y = np.random.default_rng(83), np.random.default_rng(83)
    sa = common.select_online_columns(slow, joint.columns.config, grid, x, candidates=a)
    sb = common.select_online_columns(fast, joint.columns.config, grid, y, candidates=b)
    assert x.bit_generator.state == y.bit_generator.state
    for aa, bb in zip(sa, sb):
        assert aa[0] == bb[0] and np.array_equal(aa[2], bb[2]) and np.array_equal(aa[3], bb[3])
        assert all(np.array_equal(v, getattr(bb[1], k)) for k, v in vars(aa[1]).items())


@pytest.mark.parametrize('history', [4, 6])
@pytest.mark.parametrize('windows', [1, 4])
def test_optimized_full_updates_loss_parameters_optimizer_rng_and_source_gradient_exact(history, windows):
    prep, grid, original, control, rec = fixture()
    joint = JointCausalColumns(replace(original.transport.v17_config, history_frames=history), original.columns.config)
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    reference = copy.deepcopy(joint)
    for k in ('history_occ', 'history_observed', 'history_poses'): prep.raw[k] = prep.raw[k][-history:]
    prep.registrations = [r[-history:] for r in prep.registrations]
    p, q = provider_for(prep, grid, joint), provider_for(prep, grid, reference)
    opt, _ = optimizers(joint, control); ropt, _ = optimizers(reference, control)
    rng, rrng = np.random.default_rng(46), np.random.default_rng(46)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for update in (1, 2, 3):
            a = full.train_full_batch(joint, opt, p, None, [(rec, None)]*windows, rng, update, 100,
                optimize_cpu=True, sampling_pool=pool, probe=True)
            b = full.train_full_batch(reference, ropt, q, None, [(rec, None)]*windows, rrng, update, 100,
                optimize_cpu=False, sampling_pool=pool, probe=True)
            assert a['loss'] == b['loss'] and a['source_query_gradient_norm'] == b['source_query_gradient_norm'] > 0
            assert a['sampled_columns'] == b['sampled_columns'] and rng.bit_generator.state == rrng.bit_generator.state
            assert all(torch.equal(v, reference.state_dict()[k]) for k, v in joint.state_dict().items())
            for k, state in opt.state_dict()['state'].items():
                assert all(torch.equal(v, ropt.state_dict()['state'][k][name]) for name, v in state.items())


def real_warm_fixture(history):
    prep, grid, old, control, rec = fixture()
    joint = JointCausalColumns(replace(old.transport.v17_config, history_frames=history), old.columns.config)
    prep.window.history_tokens = tuple('abcdef')
    raw = copy.deepcopy(prep.raw); raw['history_occ'][raw['history_occ'] == 4] = 17
    for f in range(6): raw['history_occ'][f, 1+f:4+f, 5:8, 1] = 4
    for k in ('history_occ', 'history_observed', 'history_poses'): raw[k] = raw[k][-history:]
    rec.update(scene_name='scene', t0_token='t0', history_tokens=tuple('abcdef'), future_tokens=tuple('ghijkl'))
    pcfg = SimpleNamespace(grid=grid, free_label=17, frame_dt_s=.5); strong = StrongW2DetConfig()
    evidence = full.prepare_causal_evidence(raw, pcfg, strong, 1)
    rec['source_centroid_xy_t0_m'] = torch.tensor([c['centroid_world'][:2] for c in evidence['current']]).float()
    rec['anchors_xy_t0_m'] = rec['source_centroid_xy_t0_m'][:, None].repeat(1, 6, 1)+rec['kta_displacement_xy_m']
    fixed = full.build_fixed_geometry({**raw, 'future_gt_occ': 'forbidden'}, rec, pcfg, strong, 1, joint.columns.config)
    provider = full.FullJointColumnProvider.__new__(full.FullJointColumnProvider)
    provider.pcfg, provider.strong, provider.device, provider.workers = pcfg, strong, torch.device('cpu'), 4
    provider.joint, provider.model, provider.columns_checked = joint, joint.transport, False
    return prep, grid, joint, control, rec, provider, {**raw, '_column_causal_preparation': fixed, '_causal_geometry_cache_hit': True}


@pytest.mark.parametrize('history', [4, 6])
def test_real_parallel_warm_renderer_identical_live_pose_and_gradient_not_detached(history):
    prep, grid, joint, control, rec, provider, raw = real_warm_fixture(history)
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    opt, _ = optimizers(joint, control); caller = get_ident(); thread_ids = []
    original = provider.prepare_columns_cpu
    def cpu(*args):
        assert get_ident() != caller; thread_ids.append(get_ident()); return original(*args)
    provider.prepare_columns_cpu = cpu
    with patch.object(columns, 'window_from_record', return_value=prep.window), \
            patch.object(columns.runtime, 'window_from_record', return_value=prep.window):
        out = joint.motion(rec, provider.device)
        assert not provider.can_prepare_cpu(raw)
        baseline = provider.prepare_columns(None, rec, include_gt=True, raw_window=raw, outputs=out)
        assert provider.can_prepare_cpu(raw)
        out['_column_render_numpy'] = {k: columns.numpy(out[k]) for k in ('residual_xy_m', 'yaw_delta_rad')}
        parallel = original(rec, raw, out)
        for name in ('baseline', 'owners', 'fallbacks', 'targets', 'yaws'):
            assert all(np.array_equal(a, b) for a, b in zip(getattr(baseline, name), getattr(parallel, name)))
        assert parallel.outputs['future_transport_queries'].requires_grad
        stats = full.train_full_batch(joint, opt, provider, None, [(rec, raw)]*4, np.random.default_rng(45), 1, 20, probe=True)
        assert stats['parallel_warm_preparations'] == 4 and len(thread_ids) == 4
        assert stats['source_query_gradient_norm'] > 0 and stats['optimizer_updated']
        bad = copy.deepcopy(raw); bad['_column_causal_preparation']['current'][0]['class_id'] = 5
        with pytest.raises(RuntimeError, match='source identity mismatch'): original(rec, bad, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual CUDA for CPU worker/readback/gradient integration')
@pytest.mark.parametrize('history', [4, 6])
def test_cuda_live_readback_parallel_warm_updates_equal_reference(history):
    prep, grid, joint, control, rec, provider, raw = real_warm_fixture(history)
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    joint.cuda(); provider.device = torch.device('cuda')
    ref = copy.deepcopy(joint); rp = copy.copy(provider); rp.joint = ref; rp.model = ref.transport
    opt, _ = optimizers(joint, control); ropt, _ = optimizers(ref, control)
    x, y = np.random.default_rng(61), np.random.default_rng(61)
    with patch.object(columns, 'window_from_record', return_value=prep.window), \
            patch.object(columns.runtime, 'window_from_record', return_value=prep.window):
        # Real first-use frozen renderer validation stays on the CUDA caller.
        for p, m in ((provider, joint), (rp, ref)):
            out = m.motion(rec, p.device)
            p.prepare_columns(None, rec, include_gt=True, raw_window=raw, outputs=out)
        for step in (1, 2):
            a = full.train_full_batch(joint, opt, provider, None, [(rec, raw)]*4, x, step, 20, probe=True)
            b = full.train_full_batch(ref, ropt, rp, None, [(rec, raw)]*4, y, step, 20, probe=True, optimize_cpu=False)
            assert a['parallel_warm_preparations'] == 4 and a['source_query_gradient_norm'] > 0
            assert a['sampled_columns'] == b['sampled_columns'] and x.bit_generator.state == y.bit_generator.state
            assert a['loss'] == pytest.approx(b['loss'], rel=1e-5, abs=1e-5)
            for k, value in joint.state_dict().items():
                torch.testing.assert_close(value, ref.state_dict()[k], rtol=1e-4, atol=2e-6)
