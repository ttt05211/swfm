"""Scientific safety for warm timing and strict four-history Local inputs."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from test_causal_geometry_cache import raw_fixture
from test_joint_causal_columns import fixture, optimizers, provider_for
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.local_history_contract import four_frame_motion_inputs
from real_motion.local_training_profile import StageTimer, CpuProfiles, trial_summary, recommend_trials
from real_motion.joint_causal_columns import JointCausalColumns, FULL4_PROTOCOL, FULL4_CONTRACT, FULL_PROTOCOL, FULL_CONTRACT, LINK_PROTOCOL
from real_motion.motion_transport import FEATURE_NAMES
from real_motion.prepared import load_nuscenes_window_raw, PrepareConfig
from real_motion.causal_column_sampling import ColumnFeatureSampler
from tools.real_motion import causal_column_common as columns
from tools.real_motion.joint_column_full_common import train_full_batch
from tools.real_motion.local_warm_cache_common import warm_causal_cache, geometry_namespace
from tools.real_motion.benchmark_p0_f9_joint_local_warm import measured_batches
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint


def test_cache_flush_completes_async_and_remains_reusable(tmp_path):
    raw = raw_fixture(); c = CausalGeometryCache(tmp_path, 'test', ram_bytes=0, reserve_bytes=0)
    c.store(('s', 'a'), raw, {'v': np.arange(5)}, asynchronous=True); c.flush()
    assert c.is_persisted(('s', 'a'), raw)
    c.store(('s', 'b'), raw, {'v': np.arange(3)}, asynchronous=True); c.flush(); c.close()
    fresh = CausalGeometryCache(tmp_path, 'test', ram_bytes=0, reserve_bytes=0)
    for key in ('a', 'b'): assert fresh.get_or_build(('s', key), raw, lambda: pytest.fail('already persisted'))[1]
    fresh.close()


def test_four_namespace_separate_and_legacy_namespace_exactly_preserved(tmp_path):
    from dataclasses import asdict
    from real_motion.strong_w2det import StrongW2DetConfig
    from real_motion.v21_source_induction import stable_json_fingerprint
    _, _, six, _, _ = fixture()
    p = SimpleNamespace(joint=six, strong=StrongW2DetConfig())
    info, caches, cfg = {'train': 'i', 'dev': 'j'}, {'train': 'k', 'dev': 'l'}, {'a': 1}
    before = stable_json_fingerprint(dict(runtime_config=cfg, strong=asdict(p.strong),
        columns=asdict(six.columns.config), info=info, caches=caches, dataroot=str(tmp_path.resolve())))
    assert geometry_namespace(cfg, p, info, caches, tmp_path) == before
    p.joint = JointCausalColumns(replace(six.transport.v17_config, history_frames=4), six.columns.config)
    assert geometry_namespace(cfg, p, info, caches, tmp_path) != before


def test_four_parent_models_and_empty_source_supported():
    from real_motion.local_st_world_model import LocalSpatialTemporalWorldModel, LocalSTWMConfig
    from real_motion.local_st_world_model_v17 import LocalSpatialTemporalWorldModelV17
    _, _, old, _, rec = fixture()
    four = JointCausalColumns(replace(old.transport.v17_config, history_frames=4), old.columns.config)
    for k in ('features', 'local_semantic_tube', 'frame_motion_features', 'target_source_mask_tube', 'kta_displacement_xy_m'): rec[k] = rec[k][:0]
    assert four.motion(rec, torch.device('cpu'))['residual_xy_m'].shape == (0, 6, 2)
    model = LocalSpatialTemporalWorldModel(LocalSTWMConfig(history_frames=4, d_model=8, semantic_dim=4, heads=2, blocks=1, decoder_blocks=1))
    assert model(rec['features'], rec['local_semantic_tube'], rec['kta_displacement_xy_m'])['residual_xy_m'].shape == (0, 6, 2)
    model = LocalSpatialTemporalWorldModelV17(replace(old.transport.v17_config, history_frames=4))
    assert model(rec['features'], rec['local_semantic_tube'], rec['kta_displacement_xy_m'], rec['frame_motion_features'], rec['target_source_mask_tube'])['residual_xy_m'].shape == (0, 6, 2)


def test_warm_preserves_rng_parameters_mode_and_never_loads_gt(tmp_path):
    raw = raw_fixture(); raw['future_gt_occ'] = None
    joint = torch.nn.Linear(2, 2); joint.train(); before = copy.deepcopy(joint.state_dict()); rng = torch.get_rng_state().clone()
    c = CausalGeometryCache(tmp_path, 'warm', ram_bytes=0, reserve_bytes=0)
    def load(source, record, *, include_gt):
        assert not include_gt
        result = copy.deepcopy(raw)
        value, hit = c.get_or_build((record['scene_name'], record['t0_token']), raw, lambda: {'memory': np.ones(2)}, defer_write=True)
        result['_column_causal_preparation'] = value
        return result
    def prep(source, record, *, include_gt, raw_window):
        assert not include_gt and not joint.training and not torch.is_grad_enabled()
        torch.rand(2)  # fork_rng must restore even if the model/diagnostic consumes RNG
        return SimpleNamespace(state={'rec': record, 'window': object(), 'gpu': torch.ones(2), 'fixed': np.zeros(2)})
    p = SimpleNamespace(causal_geometry_cache=c, joint=joint, device=torch.device('cpu'), load_raw_columns=load, prepare_columns=prep)
    rows = [dict(scene_name='train', t0_token=str(i)) for i in range(9)]
    report = warm_causal_cache(p, None, rows)
    assert report['complete'] and report['windows'] == 9 and report['optimizer_steps'] == 0
    assert joint.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, joint.state_dict()[k]) for k, v in before.items())
    for r in rows:
        value, hit = c.get_or_build((r['scene_name'], r['t0_token']), raw, lambda: pytest.fail('warm'))
        assert hit and set(value['prepared_state']) == {'fixed'}
    c.close()


def test_warm_quota_failure_is_not_reported_complete(tmp_path):
    raw = raw_fixture(); raw['future_gt_occ'] = None; raw['_column_causal_preparation'] = {}
    c = CausalGeometryCache(tmp_path, 'budget', max_bytes=0, ram_bytes=0, reserve_bytes=0)
    p = SimpleNamespace(causal_geometry_cache=c, joint=torch.nn.Linear(1, 1), device=torch.device('cpu'),
        load_raw_columns=lambda *a, **k: raw, prepare_columns=lambda *a, **k: SimpleNamespace(state={'fixed': np.eye(2)}))
    with pytest.raises(RuntimeError, match='warm cache incomplete'):
        warm_causal_cache(p, None, [dict(scene_name='s', t0_token='t')])
    c.close()


def test_persistent_pool_profiles_preserve_actual_updates_and_rng():
    prep, grid, a, control, rec = fixture(); b = copy.deepcopy(a)
    opt, _ = optimizers(a, control); opt2, _ = optimizers(b, control)
    p, q = provider_for(prep, grid, a), provider_for(prep, grid, b)
    r1, r2 = np.random.default_rng(33), np.random.default_rng(33); profiles = CpuProfiles()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for update in (1, 2):
            x = train_full_batch(a, opt, p, None, [(rec, None)]*2, r1, update, 10)
            y = train_full_batch(b, opt2, q, None, [(rec, None)]*2, r2, update, 10,
                sampling_pool=pool, sampling_workers=6, profile=True, cpu_profiles=profiles)
            assert x['loss'] == y['loss'] and 'host_stage_seconds' in y
            assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
            assert r1.bit_generator.state == r2.bit_generator.state
    assert 'candidate_workers' in profiles.text() and 'patch_workers' in profiles.text()


def test_stage_summary_safe_recommendation_does_not_reward_oom():
    stats = trial_summary([dict(wall_seconds=2., windows=8, sources=50, host_stage_seconds={'prepare': .5},
        online_candidate_wait_seconds=.3, peak_memory_mib=4, peak_reserved_mib=6, causal_geometry_cache_hits=8)])
    assert stats['windows_per_second'] == 4 and stats['cache_hits'] == 8
    assert stats['train15_hours_if_representative'] == 20430*15*.25/3600
    rows = [dict(status='ok', measurement={'windows_per_second': s}, window_batch=b, source_budget=b*32,
        capacity_peak_reserved_mib=m, available_memory_mib=100) for b, s, m in ((4, 3.9, 50), (8, 4, 80), (16, 6, 99))]
    rows.append(dict(status='oom', window_batch=32))
    decision = recommend_trials(rows)
    assert decision['recommended']['window_batch'] == 4 and decision['largest_safe']['window_batch'] == 8
    assert recommend_trials([dict(status='oom')])['recommended'] is None


def test_four_encoder_old_observations_and_cross_boundary_velocity_cannot_change_output():
    prep, grid, old, control, rec = fixture()
    joint = JointCausalColumns(replace(old.transport.v17_config, history_frames=4), old.columns.config).eval()
    # Nonzero heads: zero initialization must not hide a leaked input.
    torch.nn.init.normal_(joint.transport.residual_head.weight); torch.nn.init.normal_(joint.transport.yaw_head.weight)
    torch.nn.init.normal_(joint.transport.existence_head.weight)
    assert joint.transport.time_embedding.shape[1] == joint.columns.history_frames == 4
    output = joint.motion(rec, torch.device('cpu')); altered = copy.deepcopy(rec)
    altered['local_semantic_tube'][:, :2] = 0; altered['target_source_mask_tube'][:, :2] ^= 1
    altered['frame_motion_features'][:] = 10000  # reconstructed rather than trusting cached averages
    for i, name in enumerate(FEATURE_NAMES):
        if name.startswith(('hist_offset_0_', 'hist_offset_1_', 'hist_vel_0_', 'hist_vel_1_')) or name in ('hist_valid_0', 'hist_valid_1'):
            altered['features'][:, i] += 12345
    assert all(torch.equal(v, joint.motion(altered, torch.device('cpu'))[k]) for k, v in output.items())
    x, tube, fm, mask = four_frame_motion_inputs(rec['features'], rec['local_semantic_tube'], rec['frame_motion_features'], rec['target_source_mask_tube'])
    assert tube.shape[1] == fm.shape[1] == mask.shape[1] == 4
    assert torch.equal(fm[:, 0, 2], x[:, FEATURE_NAMES.index('hist_vel_2_x')]*fm[:, 0, 4])
    for key in ('local_semantic_tube', 'target_source_mask_tube'):
        rec[key] = rec[key][:, -4:]
    assert all(torch.equal(v, joint.motion(rec, torch.device('cpu'))[k]) for k, v in output.items())


def test_four_raw_loader_never_opens_excluded_occupancy_or_pose():
    calls = []
    w = SimpleNamespace(scene_name='s', history_tokens=tuple('abcdef'), future_tokens=tuple('ghijkl'))
    class Source:
        def load_occ3d(self, scene, token, require_lidar_mask=True):
            assert token not in ('a', 'b'); calls.append(token); return np.full((2, 2, 1), 17), np.ones((2, 2, 1), bool)
        def load_semantics(self, scene, token): return np.full((2, 2, 1), 17)
        def pose(self, token):
            assert token not in ('a', 'b'); return np.eye(4)
        def official_trajectory(self, *a, **k): return np.zeros((12, 2), np.float32)
    raw = load_nuscenes_window_raw(Source(), w, PrepareConfig(), include_gt=False, active_history_frames=4)
    assert calls == list('cdef') and len(raw['history_poses']) == 4 and len(raw['future_poses']) == 6


def test_four_sampler_bit_exact_reference_no_t0_only_dynamic_refine():
    prep, grid, joint, control, rec = fixture()
    for k in ('history_occ', 'history_observed', 'history_poses'): prep.raw[k] = prep.raw[k][-4:]
    prep.registrations = [row[-4:] for row in prep.registrations]
    plan = columns.candidate_plan(prep, 0, grid, joint.columns.config)
    sampler = ColumnFeatureSampler(prep, 0, plan, grid, joint.columns.config, columns.pose_motion, workers=3)
    a = columns.sample_column_features(prep, 0, plan, grid, joint.columns.config); b = sampler.sample(plan, columns.sample_column_features)
    assert a['history'].shape[1] == 4 and all(np.array_equal(v, b[k]) for k, v in a.items())
    prep.registrations[0][:-1] = [None]*3
    assert not np.any(columns.candidate_plan(prep, 0, grid, joint.columns.config).actor >= 0)


def test_four_actual_full_update_keeps_transport_gradient_link():
    prep, grid, old, control, rec = fixture()
    joint = JointCausalColumns(replace(old.transport.v17_config, history_frames=4), old.columns.config)
    for k in ('history_occ', 'history_observed', 'history_poses'): prep.raw[k] = prep.raw[k][-4:]
    prep.registrations = [row[-4:] for row in prep.registrations]
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    opt, _ = optimizers(joint, control); p = provider_for(prep, grid, joint)
    stats = train_full_batch(joint, opt, p, None, [(rec, None)], np.random.default_rng(8), 1, 10, probe=True, profile=True)
    assert stats['optimizer_updated'] and stats['source_query_gradient_norm'] > 0 and np.isfinite(stats['loss'])


def test_four_real_provider_warm_and_live_renderer_exactness(tmp_path):
    from tools.real_motion import joint_column_full_common as full
    from real_motion.strong_w2det import StrongW2DetConfig
    prep, grid, old, control, rec = fixture()
    joint = JointCausalColumns(replace(old.transport.v17_config, history_frames=4), old.columns.config)
    prep.window.history_tokens = tuple(f'h{i}' for i in range(6))
    raw = copy.deepcopy(prep.raw)
    raw['history_occ'][raw['history_occ'] == 4] = 17
    for f in range(6): raw['history_occ'][f, 1+f:4+f, 5:8, 1] = 4
    for k in ('history_occ', 'history_observed', 'history_poses'): raw[k] = raw[k][-4:]
    raw['future_gt_occ'] = None
    rec.update(scene_name='scene', t0_token='t0', sample_id='actual_four')
    pcfg = SimpleNamespace(grid=grid, free_label=17, frame_dt_s=.5)
    strong = StrongW2DetConfig()
    evidence = full.prepare_causal_evidence(raw, pcfg, strong, 1)
    center = torch.tensor(np.asarray([c['centroid_world'][:2] for c in evidence['current']]), dtype=torch.float32)
    rec['source_centroid_xy_t0_m'] = center
    rec['anchors_xy_t0_m'] = center[:, None, :].repeat(1, 6, 1)+rec['kta_displacement_xy_m']
    p = full.FullJointColumnProvider.__new__(full.FullJointColumnProvider)
    p.pcfg, p.strong, p.device, p.workers = pcfg, strong, torch.device('cpu'), 1
    p.joint, p.model, p.columns_checked = joint, joint.transport, False
    p.causal_geometry_cache = CausalGeometryCache(tmp_path, 'actual_four', ram_bytes=0, reserve_bytes=0)
    before = copy.deepcopy(joint.state_dict()); rng = torch.get_rng_state().clone()
    with patch.object(columns, 'window_from_record', return_value=prep.window), \
         patch.object(columns.runtime, 'window_from_record', return_value=prep.window), \
         patch.object(columns, 'load_nuscenes_window_raw', side_effect=lambda *a, **k: copy.deepcopy(raw)):
        warm_causal_cache(p, None, [rec])
        assert p.columns_checked and torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(v, joint.state_dict()[k]) for k, v in before.items())
        cached = p.load_raw_columns(None, rec, include_gt=False)
        assert cached['_causal_geometry_cache_hit'] and len(cached['history_occ']) == 4
        out = joint.motion(rec, p.device)
        prepared = p.prepare_columns(None, rec, include_gt=False, raw_window=cached, outputs=out)
        assert len(prepared.registrations[0]) == 4 and prepared.outputs['future_transport_queries'].requires_grad
        assert prepared.memory.shape[0] == 6
    p.causal_geometry_cache.close()


def test_four_checkpoint_roundtrip_and_protocol_mismatch_rejected(tmp_path):
    from dataclasses import asdict
    prep, grid, old, control, rec = fixture()
    joint = JointCausalColumns(replace(old.transport.v17_config, history_frames=4), old.columns.config)
    ck = dict(protocol=FULL4_PROTOCOL, training_contract=FULL4_CONTRACT, source_link=LINK_PROTOCOL,
        reference_checkpoint_sha256='a', runtime_config_fingerprint='b', checkpoint_role='resume_last',
        model_configs=joint.configs(), state_dict=joint.state_dict(),
        TRAIN_weights={'generation_pos_weight': 1., 'refine_class_weights': [1., 1., 1.]})
    f = tmp_path/'four.pt'; torch.save(ck, f)
    _, reloaded = load_joint(f, torch.device('cpu'), reference_sha='a', config_sha='b', allow_diagnostic=True)
    assert reloaded.transport.config.history_frames == 4
    assert all(torch.equal(v, reloaded.state_dict()[k]) for k, v in joint.state_dict().items())
    ck.update(protocol=FULL_PROTOCOL, training_contract=FULL_CONTRACT); torch.save(ck, f)
    with pytest.raises(RuntimeError, match='four-frame checkpoint/protocol'):
        load_joint(f, torch.device('cpu'), reference_sha='a', config_sha='b', allow_diagnostic=True)


def test_measured_batches_use_real_cpu_update_and_enforce_warm_hits():
    prep, grid, joint, control, rec = fixture(); p = provider_for(prep, grid, joint)
    p.load_raw_columns = lambda *a, **k: {'_causal_geometry_cache_hit': True}
    opt, _ = optimizers(joint, control)
    rows = measured_batches(joint, opt, p, None, [rec]*4, windows=2, sources=10, workers=4)
    assert len(rows) == 2 and trial_summary(rows)['cache_hits'] == 4
    p.load_raw_columns = lambda *a, **k: {'_causal_geometry_cache_hit': False}
    with pytest.raises(RuntimeError, match='persisted warm disk'):
        measured_batches(joint, opt, p, None, [rec], windows=2, sources=10, workers=4)
