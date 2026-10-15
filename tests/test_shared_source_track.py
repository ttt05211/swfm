"""Fixed identities/KTA, registration validity, cached/live GT/Pred integration."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from real_motion import shared_source_track as track
from real_motion.strong_w2det import match_instances, StrongW2DetConfig
from real_motion.geometry import OccupancyGrid
from tools.shared_source_track import adapter
from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance


def component(x, cls=4):
    ijk = np.array([[i, j, k] for i in (x, x+1) for j in (1, 2) for k in (0, 1)], np.int64)
    return dict(class_id=cls, voxel_indices=ijk, centroid_world=(ijk+.5).mean(0), voxel_count=len(ijk))


def fixture(device='cpu'):
    from test_full_gt_training import fixture as old
    body, _, provider, _ = old(); body.to(device).eval().requires_grad_(False)
    provider.pcfg.free_label = 17
    runner = adapter.SharedTrackPredictor(body, provider.pcfg, device, workers=1)
    shape = provider.pcfg.grid.shape_hwd
    raw = dict(history_occ=np.full((4, *shape), 17, np.uint8), history_observed=np.ones((4, *shape), bool),
               history_poses=np.tile(np.eye(4), (4, 1, 1)))
    for f in range(4): raw['history_occ'][f][tuple(component(f+1)['voxel_indices'].T)] = 4
    raw['history_occ'][:, 1:3, 5:7, 0] = 11  # static evidence makes ego-conditioned backgrounds distinguishable
    rec = dict(scene_name='VAL', t0_token='t0', history_tokens=['h0', 'h1', 'h2', 't0'], future_tokens=[f'f{i}' for i in range(6)])
    return runner, rec, raw


def test_pair_exact_baseline_ties_height_classes_missing_and_no_older_fallback():
    rng = np.random.default_rng(21)
    before = [dict(class_id=int(c), centroid_world=p) for c, p in zip(rng.integers(4, 7, 32), rng.normal(size=(32, 3))*10)]
    now = [dict(class_id=int(c), centroid_world=p) for c, p in zip(rng.integers(4, 7, 40), rng.normal(size=(40, 3))*10)]
    for previous, current in ((before, now), ([], now), ([component(2)]*2, [component(3)])):
        links, velocity = track.match_pair(previous, current)
        expected = match_instances(previous, current, .5, max_speed_mps=25.)
        assert set(velocity) == set(expected)
        for k in velocity: np.testing.assert_array_equal(velocity[k], expected[k])
    frames = [[component(1)], [component(2)], [], [component(4)]]
    value = track.build_track(frames, np.tile(np.eye(4), (4, 1, 1)), np.arange(4)*.5,
        OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(24, 8, 4)), StrongW2DetConfig())
    assert value.links[0, 2] == -1 and value.track_valid[0, 1] and not value.pair_velocity
    assert not track.kta_and_anchors(frames[-1], value, np.eye(4))[1].any()


def test_registration_failure_does_not_change_identity_or_enable_source_reads(monkeypatch):
    p, rec, raw = fixture()
    try:
        monkeypatch.setattr(track, 'register_history_shape', lambda *a, **k: SimpleNamespace(accepted=False))
        row = p.prepare(rec, raw, times=np.arange(4)*.5)
        source = row[1]['_shared_source_track']
        assert source.track_valid.all() and source.registration_valid.tolist() == [[False, False, False, True]]
        assert row[3][0]['shared_reads']['source_valid'].tolist() == [[False, False, False, True]]
        assert set(row[3][0]['memory_reads']['frame']) == {3}
    finally: p.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_single_association_live_kta_bytes_and_independent_pose_backgrounds(device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from tools.final_semantic_evaluation.predictor import Predictor
    from tools.scene_ego_head.diagnostics.engine import external_forward
    torch.set_num_threads(1); p, rec, raw = fixture(device)
    baseline = Predictor(p.body, p.provider.pcfg, device, workers=1)
    calls = dict(pair=0, extract=0)
    pair, extract = track.match_pair, adapter.runtime.extract_instances_cropped_exact
    def counted_pair(*a, **k): calls['pair'] += 1; return pair(*a, **k)
    def counted_extract(*a, **k): calls['extract'] += 1; return extract(*a, **k)
    try:
        with torch.no_grad(), deterministic_entrance():
            old = baseline.prepare(rec, raw, times=np.arange(4)*.5)
            monkeypatch.setattr(track, 'match_pair', counted_pair)
            monkeypatch.setattr(adapter.runtime, 'extract_instances_cropped_exact', counted_extract)
            row = p.prepare(rec, raw, times=np.arange(4)*.5)
            assert calls == dict(pair=1, extract=4)
            for key in ('kta_displacement_xy_m', 'anchors_xy_t0_m', 'source_centroid_xy_t0_m'):
                np.testing.assert_array_equal(row[0][key], old[0][key])
            gt_poses = np.tile(np.eye(4), (6, 1, 1))
            reference = external_forward(p.body, p.provider, row, gt_poses)
            monkeypatch.setattr(adapter.runtime, 'match_instances', lambda *a, **k: pytest.fail('second association'))
            actual = p.forward(row, poses=gt_poses)
            assert calls == dict(pair=1, extract=5)  # +1 ONLY from reference external_forward
            for a, b in zip(actual['state']['column_backgrounds'], reference['state']['column_backgrounds']):
                np.testing.assert_array_equal(a, b)
            torch.testing.assert_close(actual['logits'], reference['logits'], atol=1e-6, rtol=1e-5)
            pred_poses = gt_poses.copy(); pred_poses[:, 0, 3] = np.arange(1, 7)*.7
            other = p.forward(row, poses=pred_poses)
            assert any(not np.array_equal(a, b) for a, b in zip(actual['state']['column_backgrounds'], other['state']['column_backgrounds']))
            assert calls == dict(pair=1, extract=5)
            poisoned = dict(row[1], future_poses=object(), future_gt_occ=object())
            result = p.forward((row[0], poisoned, row[2], row[3]), poses=gt_poses)
            torch.testing.assert_close(result['logits'], actual['logits'], atol=0, rtol=0)
    finally: p.close(); baseline.close()


def test_cached_training_new_namespace_old_record_unchanged_and_no_old_match(tmp_path, monkeypatch):
    from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
    from tools.dense_valid_ddp.engine import DenseTrainHistory, dense_rows
    p, rec, raw = fixture()
    try:
        row = p.prepare(rec, raw, times=np.arange(4)*.5)
        expected = p.forward(row, poses=np.tile(np.eye(4), (6, 1, 1)))
        raw_old = copy.deepcopy(row[1]); raw_old.pop('_shared_source_track')
        fixed = raw_old['_column_causal_preparation']; fixed.pop('source_track')
        fixed['registrations'] = 'POISON old independent association'
        fixed['prepared_state']['column_backgrounds'] = expected['state']['column_backgrounds']
        raw_old['future_poses'] = np.tile(np.eye(4), (6, 1, 1))
        raw_old['future_gt_occ'] = np.tile(raw['history_occ'][-1], (6, 1, 1, 1))
        record = row[0]; saved_record = copy.deepcopy(record)
        provider = adapter.TrackProvider.__new__(adapter.TrackProvider)
        provider.__dict__.update(vars(p.provider)); provider.ccr_history_cache_mode = provider.ccr_val_history_cache_mode = 'require'
        provider.attach_tracks(tmp_path/'new_tracks', rank=0, implementation={'test': 'v1'})
        monkeypatch.setattr(adapter.PilotProvider, 'load_raw_columns', lambda *a, **k: raw_old)
        monkeypatch.setattr(adapter.runtime, 'match_instances', lambda *a, **k: pytest.fail('old association executed'))
        source = SimpleNamespace(nusc=SimpleNamespace(get=lambda _, token: {'timestamp': ['h0', 'h1', 'h2', 't0'].index(token)*500000}))
        a = provider.load_raw_columns(source, record, include_gt=True)
        assert not a['_shared_track_cache_hit'] and a['_shared_source_track'].track_valid.all()
        monkeypatch.setattr(track, 'from_raw', lambda *a, **k: pytest.fail('cache hit rebuilt association'))
        b = provider.load_raw_columns(source, record, include_gt=True)
        assert b['_shared_track_cache_hit'] and b['_shared_source_track'] is a['_shared_source_track']
        assert raw_old['_column_causal_preparation']['registrations'] == 'POISON old independent association'
        for key, value in saved_record.items():
            if isinstance(value, np.ndarray): np.testing.assert_array_equal(record[key], value)
        assert not any('future' in field for field in vars(b['_shared_source_track']))
        hist = DenseTrainHistory(provider, ram_mib=0).get(record, b)
        published = dense_rows([(record, b, row[2], hist)])[0]
        assert hist[0]['registrations'] is b['_shared_source_track'].registrations
        # GT background stays byte-identical; new history association is consumed.
        from tools.shared_history_experiments.full_gt.engine import forward_window
        actual = forward_window(p.body, provider, published)
        for x, y in zip(actual['state']['column_backgrounds'], expected['state']['column_backgrounds']): np.testing.assert_array_equal(x, y)
        provider.close_tracks()
    finally: p.close()


def test_real_occ_stc_shared_evaluator_integer_resume(tmp_path, monkeypatch):
    from tools.shared_source_track import evaluation
    from tools.dense_valid_ddp.engine import merge_evaluation
    p, rec, raw = fixture()
    try:
        row = p.prepare(rec, raw, times=np.arange(4)*.5)
        poses = np.tile(np.eye(4), (6, 1, 1))
        fw = p.forward(row, poses=poses)
        row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds'] = fw['state']['column_backgrounds']
        row[1].update(future_poses=poses, future_gt_occ=np.tile(raw['history_occ'][-1], (6, 1, 1, 1)))
        for i, token in enumerate(rec['history_tokens']):
            path = tmp_path/rec['scene_name']/token; path.mkdir(parents=True)
            np.savez_compressed(path/'labels.npz', semantics=raw['history_occ'][i])
        monkeypatch.setattr(evaluation.gt, 'gt_moving_support_sequence', lambda *a, **kw: [])
        monkeypatch.setattr(evaluation.gt, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
        monkeypatch.setattr(adapter.runtime, 'match_instances', lambda *a, **kw: pytest.fail('old independent association'))
        states = []
        fn = evaluation.evaluator(tmp_path)
        fn({'gt': p.body}, p.provider, SimpleNamespace(nusc=None), [[row]], save=lambda s: states.append(s))
        assert states[-1]['windows'] == 1 and len(states[-1]['metrics']) == 4
        saved = copy.deepcopy(states[-1]); states.clear()
        fn({'gt': p.body}, p.provider, SimpleNamespace(nusc=None), [[row]], saved=saved, save=lambda s: states.append(s))
        assert states[-1] == saved
        merged = merge_evaluation(states[-1:], [[rec['scene_name'], rec['t0_token']]], protocol=evaluation.PROTOCOL)
        assert merged['windows'] == 1
    finally: p.close()
