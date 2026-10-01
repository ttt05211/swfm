"""Optimized cache must reproduce old features, not merely similar metrics."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from real_motion.geometry import OccupancyGrid
from real_motion.causal_column_completion import ColumnConfig, ColumnPlan, CONTEXT_DIM
from real_motion.causal_column_model import CausalColumnModel
from real_motion.causal_column_sampling import ColumnFeatureSampler
from tools.real_motion import causal_column_common as common


def fixture(seed=0, full=False):
    rng = np.random.default_rng(seed)
    shape = (200, 200, 16) if full else (35, 31, 4)
    grid = OccupancyGrid(x_min=-4, y_min=-6, z_min=-1, voxel_size=(.4, .4, .4), shape_hwd=shape)
    cfg = ColumnConfig(width=8, heads=2, semantic_dim=4, z_bins=shape[2])
    def pose(angle, translation):
        out = common.pose_motion(np.zeros(3), translation, angle); out[2, 3] = translation[2]
        roll = .03*np.sin(angle+seed*.17); pitch = .05*np.cos(angle+seed*.31)
        rx = np.array([[1,0,0], [0,np.cos(roll),-np.sin(roll)], [0,np.sin(roll),np.cos(roll)]])
        ry = np.array([[np.cos(pitch),0,np.sin(pitch)], [0,1,0], [-np.sin(pitch),0,np.cos(pitch)]])
        out[:3,:3] = out[:3,:3]@rx@ry
        return out
    poses = [pose(.013*(f-5), np.array([.08*f, -.11*f, .004*f])) for f in range(6)]
    future = [pose(.11+.002*h, np.array([.71, -.53, .008*h])) for h in range(6)]
    indices = np.array([[x, y, 1] for x in range(5, 9) for y in range(7, 10)])
    reg = common.pose_motion(np.zeros(3), np.array([.51, -.63, 0.]), .07)
    registrations = [[None, (reg, indices), None, (reg, indices), (np.eye(4), indices), (np.eye(4), indices)]]
    prep = SimpleNamespace(raw={'history_occ': rng.integers(0, 18, (6, *shape), dtype=np.uint8),
        'history_observed': rng.random((6, *shape)) > .3, 'history_poses': poses, 'future_poses': future},
        registrations=registrations, state={'current': [dict(centroid_world=np.array([.3, .2, .4]))]},
        targets=[[np.array([.7, -.5, .4])]]*6, yaws=[[.24]]*6)
    side = 100 if full else 20
    xy = np.argwhere(np.ones((side, side), bool))
    # All three groups have overlapping read patches; generation/static share
    # geometry but DIFFERENT class-specific membership. Padding crosses borders.
    evidence = np.concatenate((xy, xy+np.array([3, 4]), xy[:100]+np.array([1, 2])))
    actors = np.r_[np.full(len(xy), -3), np.full(len(xy), -2), np.zeros(100)].astype(np.int32)
    n, z = len(actors), shape[2]; cls = np.where(actors >= 0, 4, rng.choice((11, 13), n)).astype(np.uint8)
    base = np.full((n, z), 17, np.uint8); legal = np.zeros((n, z, 3), bool); legal[..., :2] = True
    plan = ColumnPlan(evidence.copy(), (actors != -3).astype(np.uint8), actors, cls,
        np.tile(np.arange(z), (n, 1)), base, base.copy(), legal, np.zeros((n, CONTEXT_DIM), np.float32), evidence)
    return prep, grid, cfg, plan


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('workers', [1, 4])
def test_dense_and_sparse_feature_equality_all_frames_ownership_padding_rotation_z(seed, workers):
    prep, grid, cfg, plan = fixture(seed)
    sampler = ColumnFeatureSampler(prep, 3, plan, grid, cfg, common.pose_motion, workers=workers)
    assert set(sampler.maps) == {-1, 0}
    for start in range(0, len(plan), 127):
        small = plan.subset(slice(start, start+127))
        expected = common.sample_column_features(prep, 3, small, grid, cfg)
        actual = sampler.sample(small, common.sample_column_features)
        assert all(np.array_equal(expected[k], actual[k]) for k in common.FEATURE_KEYS)
    assert sampler.cache_bytes <= 64*2**20
    assert (sampler.maps[0][1][[0, 2]] == 18).all()


def test_cache_budget_sparse_fallback_and_no_future_gt_or_stale_window_cache():
    prep, grid, cfg, plan = fixture()
    original = copy.deepcopy(prep.raw)
    sampler = ColumnFeatureSampler(prep, 3, plan, grid, cfg, common.pose_motion, max_cache_mib=0)
    assert not sampler.maps and sampler.cache_bytes == 0
    actual = sampler.sample(plan, common.sample_column_features)
    expected = common.sample_column_features(prep, 3, plan, grid, cfg)
    assert all(np.array_equal(actual[k], expected[k]) for k in common.FEATURE_KEYS)
    # The original arrays remain unchanged. A fresh window gets a fresh mapper.
    assert np.array_equal(prep.raw['history_occ'], original['history_occ'])
    changed = copy.deepcopy(prep); changed.raw['future_gt_occ'] = 'must never be read'
    changed.raw['history_occ'][:] = 17
    fresh = ColumnFeatureSampler(changed, 3, plan, grid, cfg, common.pose_motion, workers=3)
    expected = common.sample_column_features(changed, 3, plan, grid, cfg)
    actual = fresh.sample(plan, common.sample_column_features)
    assert all(np.array_equal(actual[k], expected[k]) for k in common.FEATURE_KEYS)
    empty = plan.subset([])
    sampler = ColumnFeatureSampler(prep, 3, empty, grid, cfg, common.pose_motion)
    assert sampler.sample(empty, common.sample_column_features)['history'].shape == (0, 6, 7, 7, cfg.z_bins)


def test_existing_model_weights_batching_and_probabilities_unchanged():
    prep, grid, cfg, plan = fixture(3)
    # Keep enough dense queries to exercise map construction while using CPU NN.
    plan = plan.subset(np.r_[np.arange(50), np.arange(400, 450), np.arange(800, 850)])
    torch.manual_seed(18); model = CausalColumnModel(cfg).eval(); model.column_sampling_workers = 3
    with torch.no_grad():
        model.generation.weight.normal_(); model.refinement.weight.normal_()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    actual = common.predict_probabilities(model, prep, 3, plan, grid, torch.device('cpu'), batch_size=32)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(plan), 32):
            small = plan.subset(slice(start, start+32))
            arrays = common.sample_column_features(prep, 3, small, grid, cfg)
            values = {k: torch.as_tensor(v) for k, v in arrays.items()}
            g, r = model(**values)
            chunks.append(model.calibrated_probabilities(g, r, values['kind'], torch.as_tensor(small.legal)).numpy())
    assert np.array_equal(actual, np.concatenate(chunks))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert model.last_prediction_profile['queries'] == len(plan)


def test_readonly_same_window_profile_cli_keeps_original_artifact_and_checks_real_features(tmp_path, capsys):
    from unittest.mock import patch
    from test_causal_columns import fake_provider, moving_fixture
    from tools.real_motion import profile_p0_f9_causal_columns as profiler
    provider, prep, cfg, _ = fake_provider(); model = CausalColumnModel(cfg).eval()
    model_dir = tmp_path/'original'; model_dir.mkdir()
    checkpoint = model_dir/'candidate.pt'; checkpoint.write_bytes(b'original checkpoint')
    info = tmp_path/'nuscenes_infos_val_temporal_v3_scene.pkl'; info.write_bytes(b'info identity')
    record = {'scene_name': 'dev', 't0_token': 'original220'}
    ck = {'checkpoint_role': 'calibrated_candidate', 'dev_keys': [('dev', 'original220')],
          'info_fingerprints': {'dev': profiler.sha256(info)}, 'thresholds': (None, None, None)}
    argv = ['profile', '--model-dir', str(model_dir), '--window', '1', '--dataroot', str(tmp_path)]
    with patch('sys.argv', argv), patch.object(torch.cuda, 'is_available', return_value=True), \
        patch.object(torch.cuda, 'is_bf16_supported', return_value=True), \
        patch.object(profiler, 'load_columns', return_value=(ck, model)), \
        patch.object(profiler.common, 'FrozenColumns', return_value=provider), \
        patch.object(profiler, 'NuScenesWindowSource', return_value=SimpleNamespace(nusc=None)), \
        patch.object(profiler, 'load_cache', return_value=({}, [record])), \
        patch.object(common, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        profiler.main()
    output = capsys.readouterr().out
    assert 'REFERENCE FEATURE EXACTNESS:' in output and 'SECOND_WINDOW_TIMING:' in output
    assert checkpoint.read_bytes() == b'original checkpoint'
    assert list(model_dir.iterdir()) == [checkpoint]


@pytest.mark.parametrize('change', ['unchanged', 'reserialize', 'screen_pass', 'weight', 'dtype',
                                  'threshold', 'contract', 'population', 'remove_pass', 'regress_pass'])
def test_profile_snapshot_audit_allows_only_serialization_or_final_screen_pass(tmp_path, capsys, change):
    import hashlib
    from tools.real_motion import profile_p0_f9_causal_columns as profiler
    checkpoint = tmp_path/'candidate.pt'
    original = {'state_dict': {'weight': torch.tensor([1., 2.])}, 'screen_pass': change == 'regress_pass',
                'thresholds': (.5, .5, None), 'training_contract': {'frozen': True},
                'dev_keys': [('dev', 'original220')]}
    torch.save(original, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    current = copy.deepcopy(original)
    if change == 'screen_pass': current['screen_pass'] = True
    elif change == 'weight': current['state_dict']['weight'][0] += 1
    elif change == 'dtype': current['state_dict']['weight'] = current['state_dict']['weight'].double()
    elif change == 'threshold': current['thresholds'] = (.75, .5, None)
    elif change == 'contract': current['training_contract']['frozen'] = False
    elif change == 'population': current['dev_keys'] = [('dev', 'other')]
    elif change == 'remove_pass': del current['screen_pass']
    elif change == 'regress_pass': current['screen_pass'] = False
    if change != 'unchanged':
        # Different serialization prefix changes bytes even when all fields match.
        other = tmp_path/'external-finalization.pt'; torch.save(current, other)
        other.replace(checkpoint)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest
    before = checkpoint.read_bytes()
    if change in ('unchanged', 'reserialize', 'screen_pass'):
        profiler.verify_checkpoint_snapshot(checkpoint, digest, original)
        output = capsys.readouterr().out
        assert 'CHECKPOINT AUDIT:' in output
        if change != 'unchanged': assert 'weights_thresholds_and_contracts_identical' in output
    else:
        with pytest.raises(RuntimeError, match='weights/thresholds/contracts changed'):
            profiler.verify_checkpoint_snapshot(checkpoint, digest, original)
    assert checkpoint.read_bytes() == before
