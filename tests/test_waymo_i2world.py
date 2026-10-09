"""Exact public sampling/metric semantics plus history-only model integration."""
from threading import Event
import json
import pickle

import numpy as np
import pytest
import torch

from real_motion.waymo_i2world import (LABEL_MAP, SHAPE, WaymoI2WorldSource, WaymoMetrics,
                                      fingerprint, remap_labels)
from tools.real_motion.waymo_zero_shot_common import evaluate_windows, restore


def source_fixture(tmp_path, lengths=(21, 21), shape=(8, 8, 4)):
    infos = []; poses = {}; timestamp = 0
    for scene, length in enumerate(lengths):
        poses[scene] = {}
        for frame in range(length):
            sample = 1000000 + 1000*scene + frame
            infos.append(dict(timestamp=timestamp, image=dict(image_idx=sample),
                              annos='MUST NOT be passed to predictor'))
            pose = np.eye(4); pose[0, 3] = frame*.01
            poses[scene][frame] = [dict(ego2global=pose)]
            timestamp += 100000
    source = WaymoI2WorldSource(infos[::-1], poses, tmp_path, shape=shape, cache_mib=1)
    for row in source.frames:
        row.path.parent.mkdir(parents=True, exist_ok=True)
        lab = np.full(shape, 23, np.uint8); lab[:, :, 0] = 13
        lab[2:4, 2:4, 1:3] = 1
        np.savez_compressed(row.path, voxel_label=lab, origin_voxel_state=np.zeros(shape))
    return source


def test_literal_map_all_classes_and_explicit_free15():
    original = np.array(list(LABEL_MAP), np.uint8).reshape(2, 2, 4)
    before = original.copy()
    mapped = remap_labels(original, shape=original.shape)
    assert mapped.ravel().tolist() == list(LABEL_MAP.values())
    assert np.array_equal(original, before)
    original[1, 1, 3] = 15
    with pytest.raises(ValueError, match='unmapped'):
        remap_labels(original, shape=original.shape)
    assert remap_labels(original, raw_free_label=15, shape=original.shape)[1, 1, 3] == 17
    for bad in (17, 22, 255):
        original[1, 1, 3] = bad
        with pytest.raises(ValueError, match='unmapped'):
            remap_labels(original, shape=original.shape)


def test_global_stride_and_exact_official_boundary_padding(tmp_path):
    source = source_fixture(tmp_path)
    # A per-scene [::5] implementation would incorrectly start scene1 at frame0.
    assert [(r.scene, r.frame) for r in source.frames] == [(0, i) for i in (0, 5, 10, 15, 20)] + [(1, i) for i in (4, 9, 14, 19)]
    assert source.windows[0].history == (0, 0, 0, 0)
    assert source.windows[1].history == (0, 0, 0, 1)
    assert source.windows[3].future == (4, 4, 4, 4, 4, 4)
    assert source.windows[4].future == (4, 4, 4, 4, 4, 4)
    assert source.windows[5].history == (5, 5, 5, 5)
    assert len(source.windows) == len(source.frames) == 9  # no boundary window dropped
    assert source.metadata['actual_report_dt_s_including_padded_targets']['3']['zero_time_targets'] == 2


def test_prediction_only_loads_history_and_uses_no_visibility_annotations(tmp_path, monkeypatch):
    source = source_fixture(tmp_path); window = source.windows[1]
    read = []; original = source.occupancy
    monkeypatch.setattr(source, 'occupancy', lambda i: (read.append(i), original(i))[1])
    record, raw = source.prediction_inputs(window)
    assert read == list(window.history)
    assert raw['future_gt_occ'] is None
    assert raw['history_observed'].all()  # upstream dense input, not origin_voxel_state
    assert set(raw) == {'history_occ', 'history_observed', 'history_poses', 'future_poses', 'future_gt_occ'}
    assert len(record['history_tokens']) == 4 and len(record['future_tokens']) == 6
    assert not any(k in record for k in ('annos', 'target', 'valid_frame'))


def test_preflight_and_cache_are_bounded_and_source_changes_fail(tmp_path):
    source = source_fixture(tmp_path); audit = source.preflight(source.windows)
    assert audit['files'] == 9
    a = source.occupancy(0)
    assert a is source.occupancy(0) and not a.flags.writeable
    assert source.cache_hits == 1 and source.cache_bytes <= source.cache_limit
    path = source.frames[0].path
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='changed'):
        source.occupancy(0)
    source.frames[1].path.unlink()
    with pytest.raises(FileNotFoundError, match='missing'):
        source.preflight(source.windows)


def test_reject_wrong_rate_duplicate_bad_pose_or_nuscenes_infos(tmp_path):
    poses = {0: {i: [dict(ego2global=np.eye(4))] for i in range(10)}}
    infos = [dict(timestamp=i*100000, image=dict(image_idx=1000000+i)) for i in range(10)]
    with pytest.raises(ValueError, match='LIST'):
        WaymoI2WorldSource({'infos': infos}, poses, tmp_path)
    for row in infos:
        row['timestamp'] *= 2
    with pytest.raises(ValueError, match='2Hz'):
        WaymoI2WorldSource(infos, poses, tmp_path)
    for row in infos:
        row['timestamp'] //= 2
    infos[5]['image']['image_idx'] = 1000000
    with pytest.raises(ValueError, match='duplicate'):
        WaymoI2WorldSource(infos, poses, tmp_path)
    infos[5]['image']['image_idx'] = 1000005; poses[0][5][0]['ego2global'][0, 0] = 2
    with pytest.raises(ValueError, match='non-rigid'):
        WaymoI2WorldSource(infos, poses, tmp_path)


def test_official_zero_iou_exclusion_and_standard_metric_do_not_mutate():
    gt = np.array([11, 11, 13, 17], np.uint8).reshape(2, 2, 1)
    pred = np.array([11, 11, 17, 17], np.uint8).reshape(2, 2, 1)
    original = pred.copy(); target = gt.copy()
    meter = WaymoMetrics(); meter.add([pred]*6, [gt]*3)
    result = meter.report()
    assert result['average']['IoU'] == pytest.approx(200/3)
    assert result['average']['i2world_mIoU'] == 100
    assert result['average']['standard_mIoU'] == 50
    assert result['horizons']['1']['i2world_zero_IoU_classes_excluded'] == ['sidewalk']
    assert np.array_equal(pred, original) and np.array_equal(gt, target)
    # Upstream rounds each horizon before its displayed average.
    assert result['average']['i2world_rounded_IoU'] == pytest.approx(66.67)


def test_horizon_mean_not_global_pool_and_missing_classes():
    gt = np.array([11, 17], np.uint8).reshape(2, 1, 1)
    miss = np.full_like(gt, 17)
    meter = WaymoMetrics(); meter.add([gt, gt, gt, miss, gt, miss], [gt]*3)
    row = meter.report()
    assert row['average']['IoU'] == pytest.approx(100/3)
    assert row['average']['i2world_mIoU'] is None  # undefined official all-zero horizons
    assert row['average']['standard_mIoU'] == pytest.approx(100/3)
    empty = WaymoMetrics().report()
    assert all(v is None for v in empty['average'].values())
    with pytest.raises(ValueError, match='integers'):
        WaymoMetrics(np.zeros((3, 18, 18), float))


def test_metric_counts_match_literal_upstream_reference_random_grids():
    rng = np.random.default_rng(719)
    targets = [rng.integers(0, 18, (17, 13, 4), dtype=np.uint8) for _ in range(3)]
    predictions = [rng.integers(0, 18, (17, 13, 4), dtype=np.uint8) for _ in range(6)]
    meter = WaymoMetrics(); meter.add(predictions, targets); report = meter.report()
    for j, h in enumerate((1, 3, 5)):
        gt, pred = targets[j], predictions[h]
        hist = np.bincount(18*gt.ravel().astype(int)+pred.ravel(), minlength=324).reshape(18, 18)
        assert np.array_equal(hist, meter.counts[j])
        score = np.diag(hist)/(hist.sum(0)+hist.sum(1)-np.diag(hist))
        score[score == 0] = np.nan
        expected = round(np.nanmean(score[:17])*100, 2)
        binary_gt = (gt != 17).astype(int); binary_pred = (pred != 17).astype(int)
        binary = np.bincount(2*binary_gt.ravel()+binary_pred.ravel(), minlength=4).reshape(2, 2)
        iou = round(100*binary[1, 1]/(binary[1].sum()+binary[:, 1].sum()-binary[1, 1]), 2)
        assert report['horizons'][str(j+1)]['i2world_rounded_mIoU'] == expected
        assert report['horizons'][str(j+1)]['i2world_rounded_IoU'] == iou


def test_official_metadata_file_reader_strips_annotations_and_records_hashes(tmp_path):
    source = source_fixture(tmp_path, lengths=(6,))
    infos = [dict(timestamp=i*100000, image=dict(image_idx=1000000+i), annos=dict(boxes='not inputs')) for i in range(6)]
    poses = {0: {i: [dict(ego2global=np.eye(4))] for i in range(6)}}
    for name, value in (('waymo_infos_val.pkl', infos), ('cam_infos_vali.pkl', poses)):
        with (tmp_path/name).open('wb') as handle:
            pickle.dump(value, handle)
    restored = WaymoI2WorldSource.from_files(tmp_path, shape=source.shape)
    assert len(restored.metadata['source_files']) == 2
    assert all(len(row['sha256']) == 64 for row in restored.metadata['source_files'])
    assert not hasattr(restored.frames[0], 'annos')


def test_audit_cli_does_not_load_model_or_future_gt(tmp_path, monkeypatch):
    from tools.real_motion import eval_p0_f9_joint_surface_waymo as cli
    source = source_fixture(tmp_path/'data')
    monkeypatch.setattr(cli.WaymoI2WorldSource, 'from_files', lambda *a, **k: source)
    monkeypatch.setattr(cli, 'load_evaluation_model', lambda *a, **k: (_ for _ in ()).throw(AssertionError('model loaded')))
    monkeypatch.setattr(source, 'metric_targets', lambda w: (_ for _ in ()).throw(AssertionError('future GT loaded')))
    out = tmp_path/'audit'
    args = ['--waymo-root', str(source.root), '--out-dir', str(out), '--audit-only', '--expected-scenes', '2']
    assert cli.main(argv=args) == 0
    audit = json.loads((out/'audit.json').read_text())
    assert audit['future_GT_loaded'] is False and audit['model_loaded'] is False
    with pytest.raises(SystemExit):
        cli.main(argv=args)  # never replace a completed audit/result


def fake_predict(record, raw, *, verify):
    assert raw['future_gt_occ'] is None and raw['history_occ'].shape[0] == 4
    predictions = [raw['history_occ'][-1].copy() for _ in range(6)]
    return predictions, predictions, dict(added=0, removed=0), dict(readout=0.)


def test_window_boundary_resume_and_exact_integer_counts(tmp_path):
    source = source_fixture(tmp_path); selected = source.windows
    contract = dict(windows=len(selected), weight='frozen', protocol='test')
    uninterrupted = {}; split = {}; event = Event()
    result = evaluate_windows(source, selected, fake_predict, contract,
                              save=lambda s: uninterrupted.update(s))
    def progress(row):
        if row['window'] == 3:
            event.set()
    first = evaluate_windows(source, selected, fake_predict, contract, stop_event=event,
                             progress=progress, save=lambda s: split.update(s))
    assert first['status'] == 'stopped' and first['completed_windows'] == 3
    seen = []
    def resumed(*args, verify):
        seen.append(verify); return fake_predict(*args, verify=verify)
    second = evaluate_windows(source, selected, resumed, contract, saved=split)
    assert second['status'] == 'complete' and seen[0] is True
    assert result['reports'] == second['reports']
    assert restore(uninterrupted, contract, voxel_count=np.prod(source.shape))['completed_windows'] == 9
    with pytest.raises(RuntimeError, match='fingerprint/contract'):
        restore(split, {**contract, 'weight': 'changed'}, voxel_count=np.prod(source.shape))
    split['counts']['joint'][0][0][0] += 1
    split['fingerprint'] = fingerprint({k: v for k, v in split.items() if k != 'fingerprint'})
    with pytest.raises(RuntimeError, match='integer metric'):
        restore(split, contract, voxel_count=np.prod(source.shape))


def test_incomplete_forecast_never_loads_future_gt(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    monkeypatch.setattr(source, 'metric_targets', lambda w: (_ for _ in ()).throw(AssertionError('future GT read')))
    def incomplete(record, raw, **kwargs):
        return [raw['history_occ'][-1]]*5, [], {}, {}
    saved = {}
    with pytest.raises(RuntimeError, match='SIX'):
        evaluate_windows(source, source.windows, incomplete, dict(windows=9), save=lambda s: saved.update(s))
    assert saved['completed_windows'] == 0


@pytest.mark.parametrize('mode', ['numpy', 'native_parallel'])
def test_real_waymo_four_history_surface_predictions_match_numpy_reference(tmp_path, mode):
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from tools.real_motion.eval_p0_f9_joint_surface_waymo import WaymoSurfaceProvider
    from tools.real_motion.joint_surface_long_rollout_common import SurfaceBlockExecution, verify_first_block
    shape = (32, 32, 4)
    source = source_fixture(tmp_path, lengths=(36,), shape=shape)
    cfg = PrepareConfig(grid=OccupancyGrid(-6.4, -6.4, -1., (.4,)*3, shape))
    model = JointSurfaceCCR(LocalSTWMV17Config(history_frames=4, d_model=16, semantic_dim=4,
                             blocks=1, decoder_blocks=1), width=16, z_bins=4).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    provider = WaymoSurfaceProvider(model, cfg, 'cpu', 1)
    execution = SurfaceBlockExecution(provider, mode=mode, workers=2, query_workers=2, graphs=False)
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for window in (source.windows[0], source.windows[3], source.windows[-1]):
            record, raw = source.prediction_inputs(window)
            prep = provider.prepare_columns(None, record, include_gt=False, raw_window=raw)
            dense, edits, _, probability = execution.predict(prep)
            verify_first_block(provider, prep.state['rec'], prep, dense, probability, execution)
            assert len(dense) == 6 and all(x.shape == shape for x in dense)
            assert edits['removed'] == 0 and raw['future_gt_occ'] is None
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    finally:
        execution.close(); torch.set_num_threads(threads)
