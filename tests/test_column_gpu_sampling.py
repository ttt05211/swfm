"""Byte exactness, guarded boundaries, live gradients and CPU-only workers."""
import copy
from dataclasses import replace
from threading import get_ident

import numpy as np
import pytest
import torch

from real_motion.column_gpu_sampling import GpuColumnSampler, pack_column_window
from real_motion.column_cpu_pipeline import HorizonCpuPool
from real_motion.joint_causal_columns import JointCausalColumns
from tools.real_motion import causal_column_common as common
from tools.real_motion import joint_column_full_common as full
from tools.real_motion.joint_column_common import sample_online_column
from test_causal_column_sampling import fixture
from test_column_horizon_pipeline import assert_state_equal
from test_joint_causal_columns import fixture as joint_fixture, provider_for, optimizers
from test_native_column_cpu import compiled


def selected_rows(plan):
    ids = np.r_[np.arange(42), np.arange(400, 442), np.arange(800, 830)]
    small = plan.subset(ids)
    return [(h, small, np.zeros(small.base.shape, np.int64), np.ones(len(small), np.float32)) for h in range(6)]


@pytest.fixture
def exact_training_determinism(device, monkeypatch):
    # CUDA duplicate-source gathers can otherwise use nondeterministic atomic
    # gradient additions, even for CPU-backend vs CPU-backend. Test exactness
    # under deterministic kernels; do NOT change the user's training settings.
    if device == 'cpu':
        yield; return
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    try: yield
    finally: torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize('frames', [4, 6])
@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('chunk', [1, 32])
def test_cpu_emulation_same_bytes_rotations_missing_registration_padding_repeated_anchors(frames, seed, chunk):
    prep, grid, cfg, plan = fixture(seed)
    prep.raw = {**prep.raw, **{k: prep.raw[k][-frames:] for k in ('history_occ','history_observed','history_poses')}}
    prep.registrations = [r[-frames:] for r in prep.registrations]
    # Overlapping ownership must remain multi-owner, not last-writer wins.
    prep.registrations.append(copy.deepcopy(prep.registrations[0]))
    prep.state['current'].append(copy.deepcopy(prep.state['current'][0]))
    prep.targets = [r*2 for r in prep.targets]; prep.yaws = [r*2 for r in prep.yaws]
    rows = selected_rows(plan)
    rows[0][1].actor[-15:] = 1
    prep.raw['future_gt_occ'] = object()  # any read by sampler would fail
    sampler = GpuColumnSampler('cpu', allow_cpu=True, chunk_queries=chunk)
    before = copy.deepcopy(prep.raw['history_occ'])
    actual, stats = sampler.sample(prep, rows, grid, cfg, common.pose_motion)
    assert stats['gpu_feature_verified_horizons'] > 0
    for row, values in zip(rows, actual):
        expected = sample_online_column(prep, row, grid, cfg)
        for key in expected:
            value = values[key].numpy() if isinstance(values[key], torch.Tensor) else values[key]
            assert np.array_equal(value, expected[key]), key
    assert np.array_equal(prep.raw['history_occ'], before)


def test_boundary_whole_horizon_replay_and_empty_and_budget_fallback():
    prep, grid, cfg, plan = fixture(1)
    prep.raw['history_poses'] = [np.eye(4)]*6
    prep.raw['future_poses'] = [np.eye(4)]*6
    # Centres map EXACTLY onto integer grid planes, requiring original BLAS.
    prep.raw['future_poses'][0] = np.eye(4)
    prep.raw['future_poses'][0][0, 3] = .5*grid.voxel_size[0]
    row = selected_rows(plan)[0]
    sampler = GpuColumnSampler('cpu', allow_cpu=True)
    actual, stats = sampler.sample(prep, [row], grid, cfg, common.pose_motion)
    assert stats['gpu_feature_boundary_horizons'] == 1
    assert torch.equal(actual[0]['history'], torch.as_tensor(sample_online_column(prep, row, grid, cfg)['history']))
    empty, stats = sampler.sample(prep, [], grid, cfg, common.pose_motion)
    assert empty == [] and stats['gpu_feature_horizons'] == 0
    assert sampler.windows == 1
    sampler = GpuColumnSampler('cpu', allow_cpu=True, max_working_mib=1)
    _, stats = sampler.sample(prep, selected_rows(plan), grid, cfg, common.pose_motion)
    assert stats['gpu_feature_budget_windows'] == 1 and stats['gpu_feature_fallback_horizons'] == 6


def test_oom_only_falls_back_and_exactness_error_is_fatal(monkeypatch):
    prep, grid, cfg, plan = fixture()
    rows = selected_rows(plan)
    sampler = GpuColumnSampler('cpu', allow_cpu=True)
    def oom(*args): raise torch.cuda.OutOfMemoryError('test temporary gather allocation')
    monkeypatch.setattr(sampler, '_gather', oom)
    _, stats = sampler.sample(prep, rows, grid, cfg, common.pose_motion)
    assert stats['gpu_feature_oom_windows'] == 1 and stats['gpu_feature_fallback_horizons'] == 6
    def wrong(*args):
        n = sum(len(row[1]) for row in rows)
        a = torch.zeros((n, 6, cfg.patch, cfg.patch, cfg.z_bins), dtype=torch.uint8)
        return a, a.clone(), np.zeros(n, bool)
    monkeypatch.setattr(sampler, '_gather', wrong)
    with pytest.raises(RuntimeError, match='exactness failed'): sampler.sample(prep, rows, grid, cfg, common.pose_motion)
    def bug(*args): raise ValueError('not an OOM')
    monkeypatch.setattr(sampler, '_gather', bug)
    with pytest.raises(ValueError, match='not an OOM'): sampler.sample(prep, rows, grid, cfg, common.pose_motion)


@pytest.mark.parametrize('horizons', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires actual CUDA'))])
def test_real_adamw_rng_gradients_and_main_thread_gpu_gather_unchanged(monkeypatch, horizons, device, exact_training_determinism):
    prep, grid, original, control, rec = joint_fixture()
    motion_cfg = replace(original.transport.v17_config, history_frames=4)
    joint = JointCausalColumns(motion_cfg, original.columns.config)
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    reference = copy.deepcopy(joint)
    joint.to(device); reference.to(device); control.to(device)
    for k in ('history_occ','history_observed','history_poses'): prep.raw[k] = prep.raw[k][-4:]
    prep.registrations = [r[-4:] for r in prep.registrations]
    opt, _ = optimizers(joint, control); refopt, _ = optimizers(reference, control)
    p, q = provider_for(prep, grid, joint), provider_for(prep, grid, reference)
    p.device = q.device = torch.device(device)
    rows = [(rec, None)]*4
    rng, ref_rng = np.random.default_rng(71), np.random.default_rng(71)
    monkeypatch.setattr(full, 'bundle_enabled', lambda: True)
    monkeypatch.setattr(full, 'horizon_pipeline_enabled', lambda: horizons)
    sampler = GpuColumnSampler(device, allow_cpu=device == 'cpu')
    caller = get_ident(); sample = sampler.sample
    def on_main(*args, **kwargs):
        assert get_ident() == caller
        return sample(*args, **kwargs)
    sampler.sample = on_main
    with HorizonCpuPool(2) as pool:
        for update in (22523, 22524):
            ref = full.train_full_batch(reference, refopt, q, None, rows, ref_rng, update, 77172, probe=True,
                                        sampling_pool=pool)
            actual = full.train_full_batch(joint, opt, p, None, rows, rng, update, 77172, probe=True,
                                           sampling_pool=pool, column_feature_sampler=sampler)
            for key in ('loss','motion_loss','column_loss','grad_norm','column_grad_norm',
                        'source_query_gradient_norm','sampled_columns'):
                assert actual[key] == ref[key], key
            assert actual['source_query_gradient_norm'] > 0
            assert_state_equal(joint.state_dict(), reference.state_dict())
            assert_state_equal(opt.state_dict(), refopt.state_dict())
            assert rng.bit_generator.state == ref_rng.bit_generator.state


def test_requires_explicit_cuda_and_verified_configuration():
    with pytest.raises(ValueError, match='requires CUDA'): GpuColumnSampler('cpu')
    with pytest.raises(ValueError, match='mandatory verification'):
        GpuColumnSampler('cpu', allow_cpu=True, verify_first=0)


def test_stale_window_packing_rejected_not_reused_for_new_live_poses():
    prep, grid, cfg, plan = fixture()
    rows = selected_rows(plan)
    packed = pack_column_window(prep, rows, grid, cfg, common.pose_motion)
    sampler = GpuColumnSampler('cpu', allow_cpu=True)
    other = copy.deepcopy(prep)
    with pytest.raises(ValueError, match='stale GPU'):
        sampler.sample(other, rows, grid, cfg, common.pose_motion, packed=packed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual CUDA')
@pytest.mark.parametrize('frames', [4, 6])
def test_actual_cuda_gather_bytes(frames):
    prep, grid, cfg, plan = fixture(5)
    for k in ('history_occ','history_observed','history_poses'): prep.raw[k] = prep.raw[k][-frames:]
    prep.registrations = [r[-frames:] for r in prep.registrations]
    sampler = GpuColumnSampler('cuda')
    rows = selected_rows(plan)
    values, stats = sampler.sample(prep, rows, grid, cfg, common.pose_motion)
    for row, actual in zip(rows, values):
        expected = sample_online_column(prep, row, grid, cfg)
        assert np.array_equal(actual['history'].cpu().numpy(), expected['history'])
        assert np.array_equal(actual['flags'].cpu().numpy(), expected['flags'])
    assert stats['gpu_feature_verified_horizons'] > 0


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires actual CUDA'))])
def test_native_compact_actual_cuda_or_emulated_features_match_real_native_cpu(compiled, monkeypatch, device):
    from test_column_cpu_kernels import candidate_fixture
    from tools.real_motion import joint_column_common as joint_common
    prep, grid, cfg = candidate_fixture(8)
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND','native')
    monkeypatch.setenv('SWFM_COLUMN_CPU_BUNDLE','1')
    prep.cpu_kernels_optimized = prep.cpu_bundle_optimized = True
    plans = joint_common.build_online_column_candidates(prep, cfg, grid, defer_context=True)
    rows = joint_common.select_online_columns(prep, cfg, grid, np.random.default_rng(83), candidates=plans)
    assert rows and all(labels is None for _,_,labels in plans)
    sampler = GpuColumnSampler(device, allow_cpu=device == 'cpu')
    arrays, stats = sampler.sample(prep, rows, grid, cfg, common.pose_motion)
    for row, actual in zip(rows, arrays):
        expected = sample_online_column(prep, row, grid, cfg)
        assert np.array_equal(actual['history'].cpu().numpy(), expected['history'])
        assert np.array_equal(actual['flags'].cpu().numpy(), expected['flags'])
    assert compiled.info()['calls']['compact_materialize'] > 0
