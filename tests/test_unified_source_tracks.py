"""Isolated geometry, real small-network CPU/CUDA, causality and resume checks."""
import copy
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from real_motion.geometry import OccupancyGrid
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.source_evidence_audit import associate_backwards
from tools.unified_source_tracks import geometry as g, engine, run
from tools.final_semantic_evaluation.predictor import Predictor
from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


def component(x, cls=4):
    ijk = np.array([[i, j, k] for i in (x, x+1) for j in (1, 2) for k in (0, 1)], np.int64)
    return dict(class_id=cls, voxel_indices=ijk, centroid_world=(ijk+.5).mean(0), voxel_count=len(ijk))


def manual_track(centers, valid=None, times=None):
    c = np.asarray(centers, np.float64).reshape(1, 4, 3)
    valid = np.asarray([[True]*4] if valid is None else [valid], bool)
    links = np.where(valid, 0, -1)
    return g.SourceTracks(links, c, valid, np.asarray([-.5*3, -1., -.5, 0.] if times is None else times), [], {})


def test_actual_time_linear_slope_two_single_and_anchor():
    times = np.array([100., 100.45, 101.05, 101.6])
    centers = np.c_[3+2*(times-times[-1]), 5-.8*(times-times[-1]), np.arange(4)]
    track = manual_track(centers, times=times)
    np.testing.assert_allclose(g.track_velocity(track, 'fit'), [[2., -.8, 0.]], atol=1e-14)
    track.valid[:] = [False, True, False, True]
    np.testing.assert_allclose(g.track_velocity(track, 'fit'), g.track_velocity(track, 'two'), atol=1e-14)
    track.valid[:] = [False, False, False, True]
    assert not g.track_velocity(track, 'fit').any()
    with pytest.raises(ValueError, match='increasing'): g.checked_times([0, 1, 1, 2])


def test_multiframe_jitter_reduction_and_acceleration_negative_control():
    times = np.arange(4)*.5
    c = np.c_[2*times+np.array([.02, -.02, -.2, .2]), np.zeros((4, 2))]
    t = manual_track(c)
    assert abs(g.track_velocity(t, 'fit')[0, 0]-2) < abs(g.track_velocity(t, 'two')[0, 0]-2)
    c[:, 0] = times**2  # current derivative = 3; recent secant is better here
    t = manual_track(c)
    assert abs(g.track_velocity(t, 'fit')[0, 0]-3) > abs(g.track_velocity(t, 'two')[0, 0]-3)


def test_identity_order_matches_existing_association_and_raw_not_aligned_centers():
    frames = [[component(f+1), component(15-f, 5)] for f in range(4)]
    seed = {0: np.array([2., 0, 0]), 1: np.array([-2., 0, 0])}
    before = copy.deepcopy(frames)
    linked, _ = associate_backwards(frames, frames[-1], seed)
    track = g.tracks_from_frames(frames, np.arange(4)*.5, seed,
        OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(24, 8, 4)),
        np.tile(np.eye(4), (4, 1, 1)), StrongW2DetConfig())
    np.testing.assert_array_equal(track.links, linked)
    np.testing.assert_allclose(g.track_velocity(track, 'fit')[:, 0], [2., -2.])
    assert track.registrations[0][0] is not None
    # Original centroids move; registered shape centers become t0 centers.
    assert track.centers[0, 0, 0] != track.centers[0, -1, 0]
    for a, b in zip(frames, before):
        for x, y in zip(a, b): np.testing.assert_array_equal(x['voxel_indices'], y['voxel_indices'])


def test_ambiguity_gate_gap_and_empty_no_new_association_rules():
    current = [component(5)]; previous = [component(4), component(4)]
    frames = [[component(3)], [component(4)], previous, current]
    track = g.tracks_from_frames(frames, np.arange(4)*.5, {0: np.array([2., 0, 0])},
        OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(24, 8, 4)),
        np.tile(np.eye(4), (4, 1, 1)), StrongW2DetConfig())
    assert track.links[0, 2] == -1 and track.audit['ambiguous'] == 1
    assert track.valid[0, 1]  # existing gap bridge remains, not a new rule
    empty = g.tracks_from_frames([[]]*4, np.arange(4)*.5, {}, OccupancyGrid(),
        np.tile(np.eye(4), (4, 1, 1)), StrongW2DetConfig())
    assert g.track_velocity(empty, 'fit').shape == (0, 3)


def test_kta_error_is_displacement_not_absolute_box_center_and_common_valid_mask():
    track = manual_track(np.c_[np.arange(4), np.zeros((4, 2))])
    d = np.full((1, 6, 2), 2.); valid = np.ones((1, 6), bool); valid[0, 3] = False
    e = g.motion_errors({a: d.copy() for a in g.ARMS}, track, d, valid)
    assert e['baseline']['all'] == [[0.], [], [0.]]
    assert e['unified_fit']['three_plus'] == e['baseline']['all']


def real_fixture(device='cpu'):
    from test_full_gt_training import fixture
    model, _, p, row = fixture(); model.eval().requires_grad_(False).to(device)
    p.pcfg.free_label = 17
    runner = Predictor(model, p.pcfg, device, workers=1)
    shape = p.pcfg.grid.shape_hwd
    raw = dict(history_occ=np.full((4, *shape), 17, np.uint8),
        history_observed=np.ones((4, *shape), bool), history_poses=np.tile(np.eye(4), (4, 1, 1)))
    for f in range(4): raw['history_occ'][f][tuple(component(f+1)['voxel_indices'].T)] = 4
    record = dict(scene_name='VAL', t0_token='t0', history_tokens=['h0', 'h1', 'h2', 't0'], future_tokens=[f'f{i}' for i in range(6)])
    return runner, record, raw


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_frozen_model_common_track_reads_memory_baseline_noop_and_source_unchanged(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    p, rec, raw = real_fixture(device); before = tensor_state_fingerprint(p.body.state_dict())
    original = copy.deepcopy(raw)
    try:
        with torch.no_grad(), deterministic_entrance():
            row = p.prepare(rec, raw, times=np.arange(4)*.5)
            track = g.build_tracks(row, p.provider, np.arange(4)*.5)
            for arm, mode in (('unified_two', 'two'), ('unified_fit', 'fit')):
                new = g.make_row(row, track, g.track_velocity(track, mode), p.history)
                assert new[3][0]['registrations'] is track.registrations
                np.testing.assert_array_equal(new[0]['anchors_xy_t0_m'], new[0]['source_centroid_xy_t0_m'][:, None]+new[0]['kta_displacement_xy_m'])
                np.testing.assert_allclose(new[3][0]['shared_reads']['source_center'][0, :, 0], [2., 3., 4., 5.])
                assert set(new[3][0]['memory_reads']['frame']) == {0, 1, 2, 3}
                first = p.forward(row, poses=np.tile(np.eye(4), (6, 1, 1)))
                other = engine.unified_forward(p, new, first, np.tile(np.eye(4), (6, 1, 1)))
                for key in ('prior', 'memory', 'logits'): torch.testing.assert_close(other[key], first[key], rtol=1e-5, atol=1e-6)
                for key in p.dense(first): np.testing.assert_array_equal(p.dense(first)[key], p.dense(other)[key])
                assert all(x is y for x, y in zip(other['state']['column_backgrounds'], first['state']['column_backgrounds']))
        assert before == tensor_state_fingerprint(p.body.state_dict())
        assert all(v.grad is None for v in p.body.parameters())
        for k in raw: np.testing.assert_array_equal(raw[k], original[k])
    finally: p.close()


def test_engine_all_predictions_before_labels_resume_counts_and_single_window_no_mutation(monkeypatch):
    runner, rec, raw = real_fixture(); calls = []; snapshots = []; event = threading.Event()
    class Source:
        def poses(self, job): return np.tile(np.eye(4), (6, 1, 1))
        def history(self, job, kind): return rec, raw, np.arange(4)*.5
        def targets(self, job, indices):
            assert calls.count('dense') >= 6
            return [raw['history_occ'][-1].copy()]*3
        def moving(self, *a): return [np.ones(raw['history_occ'].shape[1:], bool)]*3
    old_dense = runner.dense
    def dense(v): calls.append('dense'); return old_dense(v)
    monkeypatch.setattr(runner, 'dense', dense)
    def labels(source, job, row):
        assert calls.count('dense') >= 6
        return np.tile(np.array([2., 0.])*(np.arange(1, 7)*.5)[:, None], (len(row[0]['source_class_id']), 1, 1)), np.ones((len(row[0]['source_class_id']), 6), bool)
    keys = [['VAL', 'a'], ['VAL', 'b']]; jobs = [object(), object()]
    def save(v): snapshots.append(copy.deepcopy(v)); event.set()
    try:
        with pytest.raises(InterruptedError): engine.evaluate(runner, Source(), jobs, keys, {}, kinds=('occ', 'stc'),
            save=save, stop_event=event, checkpoint_every=1, labeler=labels)
        assert snapshots[-1]['keys'] == keys[:1]
        before = copy.deepcopy(snapshots[-1])
        resumed = engine.evaluate(runner, Source(), jobs, keys, {}, kinds=('occ', 'stc'), saved=before, labeler=labels)
        assert resumed['windows'] == 2 and snapshots[-1] == before
        assert resumed['kta_errors']['occ']['unified_fit']['all']['mean_m'] == [0., 0., 0.]
        bad = copy.deepcopy(before); bad['counts']['occ/baseline/transport']['confusion'][0][0][0] += 1
        with pytest.raises(RuntimeError, match='totals'): engine.evaluate(runner, Source(), jobs, keys, {}, kinds=('occ', 'stc'), saved=bad, labeler=labels)
    finally: runner.close()


def test_label_motion_future_annotations_evaluation_only_and_preserves_source_offset(monkeypatch):
    comp = component(5); comp['centroid_world'] = np.array([6., 2., 1.])
    ann = dict(instance_token='instance', class_id=4, center_world=np.array([5., 2., 1.]))
    monkeypatch.setattr(engine, 'dynamic_annotations', lambda *a: [ann])
    monkeypatch.setattr(engine, 'annotation_map', lambda _, token: {'instance': {**ann,
        'center_world': ann['center_world']+np.array([float(token), 0., 0.])}})
    source = SimpleNamespace(source=SimpleNamespace(nusc=object()))
    row = ({}, {'history_poses': [np.eye(4)]*4}, (), ({'state': {'current': [comp]}}, None))
    displacement, valid = engine.label_motion(source, SimpleNamespace(t0_token='t0', future_tokens=[str(i) for i in range(1, 7)]), row)
    np.testing.assert_array_equal(displacement[0, :, 0], np.arange(1, 7))
    assert valid.all()  # source component offset 1m did not contaminate displacement


def test_actual_cli_snapshot_contract_resume_and_old_source_bytes_read_only(tmp_path, monkeypatch):
    from test_full_gt_training import fixture
    model, _, p, _ = fixture(); model.eval().requires_grad_(False)
    (tmp_path/'original_training').mkdir()
    source = tmp_path/'original_training/source.pt'; torch.save({'source': 'unchanged'}, source)
    info = tmp_path/'info.pkl'; info.write_bytes(b'info')
    cache = tmp_path/'cache.pt'; cache.write_bytes(b'cache')
    original = source.read_bytes(); data = {k: run.file_sha256(v) for k, v in (('dev_info', info), ('dev_cache', cache))}
    ck = dict(contract=dict(data=data, eval_keys=[['VAL', str(i)] for i in range(64)], train_keys=[['TRAIN', 't']]), epoch=2, updates=10292)
    monkeypatch.setattr(run, 'load_body', lambda *a: (ck, model))
    monkeypatch.setattr(run, 'make_prepare_config', lambda *a: SimpleNamespace(grid=p.pcfg.grid, free_label=17, frame_dt_s=.5))
    monkeypatch.setattr(run, 'load_runtime_config', lambda *a: {})
    monkeypatch.setattr(run, 'validate_grid', lambda *a: None)
    monkeypatch.setattr(run, 'load_cache', lambda *a: ({}, [{'scene_name':'VAL', 't0_token':str(i)} for i in range(64)]))
    jobs = [SimpleNamespace(scene_name='VAL', t0_token=str(i)) for i in range(64)]
    monkeypatch.setattr(run, 'nusc_jobs', lambda *a: (jobs, {}))
    monkeypatch.setattr(run, 'NuscInputs', lambda *a, **k: SimpleNamespace(preflight=lambda *a: ['fixed-inventory']))
    monkeypatch.setattr(run, 'Predictor', lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(run, 'audit_gt_entrance', lambda *a: dict(pass_gate=True))
    def evaluate(predictor, inputs, jobs, keys, contract, **kw):
        kw['save'](dict(contract=contract, keys=keys))
        return dict(status='complete', windows=64, reports={}, kta_errors={}, track_audits={}, seconds=1.)
    monkeypatch.setattr(run, 'evaluate', evaluate)
    out = tmp_path/'out'
    args = ['--gt-checkpoint', str(source), '--out-dir', str(out), '--dataroot', str(tmp_path),
        '--dev-info', str(info), '--dev-cache', str(cache), '--inputs', 'occ', '--device', 'cpu']
    assert run.main(argv=args) == 0
    contract = json.loads((out/'contract.json').read_text())
    assert contract['arms'] == list(g.ARMS) and contract['population']['actual_windows'] == 64
    assert run.main(argv=[*args, '--resume']) == 0
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError): run.main(argv=args)
    with pytest.raises(RuntimeError, match='contract'): run.main(argv=[*args, '--resume', '--cpu-workers', '2'])
