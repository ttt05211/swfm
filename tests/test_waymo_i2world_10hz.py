"""Native-frame official indices, one-pass metrics, isolation and recovery."""
from copy import deepcopy
import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real_motion.waymo_i2world import WaymoI2WorldSource, WaymoMetrics, fingerprint
from real_motion.waymo_i2world_10hz import (PROTOCOL, REPORT_KEYS, REPORT_SECONDS,
                                          WaymoI2World10HzSource, format_10hz_reports)
from tools.real_motion.waymo_zero_shot_common import evaluate_windows, restore


def metadata(lengths=(9, 8), dt_us=100000, gap=None):
    infos, poses, tick = [], {}, 0
    for scene, length in enumerate(lengths):
        poses[scene] = {}
        for frame in range(length):
            tick += (gap or {}).get((scene, frame), 0)
            infos.append(dict(timestamp=tick, image=dict(image_idx=1000000+1000*scene+frame),
                              annos='NOT predictor input'))
            pose = np.eye(4); pose[0, 3] = frame*.01
            poses[scene][frame] = [dict(ego2global=pose)]
            tick += dt_us
    return infos[::-1], poses


def fixture(tmp_path, *, shape=(8, 8, 4), lengths=(9, 8), gap=None):
    infos, poses = metadata(lengths, gap=gap)
    source = WaymoI2World10HzSource(infos, poses, tmp_path, shape=shape, cache_mib=1)
    for row in source.frames:
        row.path.parent.mkdir(parents=True, exist_ok=True)
        lab = np.full(shape, 23, np.uint8); lab[:, :, 0] = 13
        x = 2+row.frame%2; lab[x:x+2, 2:4, 1:3] = 1
        np.savez_compressed(row.path, voxel_label=lab)
    return source


def test_all_native_anchors_and_boundary_no_scene_crossing(tmp_path):
    source = fixture(tmp_path)
    assert len(source.frames) == len(source.windows) == 17
    assert [(r.scene, r.frame) for r in source.frames] == [(0, i) for i in range(9)]+[(1, i) for i in range(8)]
    assert source.windows[0].history == (0, 0, 0, 0)
    assert source.windows[8].future == (8,)*6
    assert source.windows[9].history == (9,)*4
    assert source.windows[6].future == (7, 8, 8, 8, 8, 8)
    assert source.metadata['load_interval'] == 1 and source.metadata['sample_hz'] == 10
    assert source.metadata['upstream_eval_times'] == [1, 3, 5]
    assert source.metadata['report_future_native_steps'] == [2, 4, 6]
    assert source.metadata['report_nominal_seconds'] == [.2, .4, .6]
    spans = source.metadata['actual_report_dt_s_including_padded_targets']
    assert set(spans) == set(REPORT_KEYS)
    assert [spans[key]['max'] for key in REPORT_KEYS] == list(REPORT_SECONDS)


def test_two_hz_source_and_metadata_inputs_are_unchanged(tmp_path):
    infos, poses = metadata(); before = deepcopy(infos)
    original = WaymoI2WorldSource(infos, poses, tmp_path)
    original_manifest, original_metadata = original.manifest_fingerprint, deepcopy(original.metadata)
    native = WaymoI2World10HzSource(infos, poses, tmp_path)
    again = WaymoI2WorldSource(infos, poses, tmp_path)
    assert original_metadata == original.metadata == again.metadata
    assert original_manifest == again.manifest_fingerprint != native.manifest_fingerprint
    assert infos == before
    assert [(r.scene, r.frame) for r in original.frames] == [(0, 0), (0, 5), (1, 1), (1, 6)]


def test_native_gaps_audited_without_retime_or_drop(tmp_path):
    source = fixture(tmp_path, lengths=(21,), gap={(0, 3):300000})
    assert len(source.frames) == 21 and source.frames[3].timestamp_us == 600000
    audit = source.metadata['timestamp_gap_audit']
    assert audit['pair_count'] == 20 and audit['outlier_count'] == 1
    assert audit['frame_step_histogram'] == {'1':20}
    assert audit['resampled'] is False and audit['dropped_windows'] == 0
    assert audit['examples'][0]['dt_s'] == .4
    spans = source.metadata['actual_report_dt_s_including_padded_targets']
    assert spans['eval_time_5']['max'] == .9


@pytest.mark.parametrize('dt_us', [100, 20000, 200000, 500000, 100000000])
def test_wrong_native_rate_or_units_fail_closed(tmp_path, dt_us):
    infos, poses = metadata(dt_us=dt_us)
    with pytest.raises(ValueError, match='not nominal 10Hz'):
        WaymoI2World10HzSource(infos, poses, tmp_path)


@pytest.mark.parametrize('corrupt', ['duplicate', 'backward', 'pose', 'zero_time'])
def test_native_metadata_corruption_not_hidden(tmp_path, corrupt):
    infos, poses = metadata(lengths=(10,)); infos.reverse()
    if corrupt == 'duplicate': infos[3]['image']['image_idx'] = infos[2]['image']['image_idx']
    if corrupt == 'backward': infos[2]['image']['image_idx'], infos[3]['image']['image_idx'] = (
        infos[3]['image']['image_idx'], infos[2]['image']['image_idx'])
    if corrupt == 'pose': poses[0][2][0]['ego2global'][0,0] = 2
    if corrupt == 'zero_time': infos[3]['timestamp'] = infos[2]['timestamp']
    with pytest.raises(ValueError):
        WaymoI2World10HzSource(infos, poses, tmp_path)


def predict(record, raw, *, verify):
    assert raw['future_gt_occ'] is None and len(raw['history_occ']) == 4
    frames = [np.roll(raw['history_occ'][-1], h, axis=0) for h in range(6)]
    return frames, frames, dict(added=0, removed=0), dict(readout=0.)


def test_one_pass_equals_three_independent_eval_time_counts_and_correct_targets(tmp_path):
    source = fixture(tmp_path); contract = dict(protocol=PROTOCOL, windows=len(source.windows))
    calls, target_indices = [], []
    original = source.metric_targets
    def targets(window):
        target_indices.append([window.future[h] for h in (1, 3, 5)])
        return original(window)
    source.metric_targets = targets
    def one(*args, **kwargs):
        calls.append(args[0]['t0_token']); return predict(*args, **kwargs)
    report = evaluate_windows(source, source.windows, one, contract)
    assert len(calls) == len(source.windows)  # not three model passes
    assert target_indices[0] == [2, 4, 6]
    separate = []
    for h in (1, 3, 5):
        matrix = np.zeros((18,18), np.int64)
        for window in source.windows:
            record, raw = source.prediction_inputs(window)
            p = predict(record, raw, verify=False)[0][h]
            gt = source.occupancy(window.future[h])
            matrix += np.bincount(18*gt.ravel().astype(np.int64)+p.ravel(), minlength=324).reshape(18,18)
        separate.append(matrix)
    meter = WaymoMetrics(np.stack(separate), windows=len(calls))
    assert report['reports']['joint'] == meter.report()
    before = deepcopy(report['reports']); formatted = format_10hz_reports(report['reports'])
    assert report['reports'] == before  # presentation cannot mutate counts or original 2Hz report
    assert set(formatted['joint']['horizons']) == set(REPORT_KEYS)
    for key, h, seconds in zip(REPORT_KEYS, (1,3,5), REPORT_SECONDS):
        row = formatted['joint']['horizons'][key]
        assert row['eval_time'] == h and row['future_native_step'] == h+1 and row['nominal_seconds'] == seconds
    assert formatted['joint']['average'] == before['joint']['average']


def test_native_whole_window_resume_and_cross_protocol_rejection(tmp_path):
    source = fixture(tmp_path); contract = dict(protocol=PROTOCOL, windows=len(source.windows))
    complete = evaluate_windows(source, source.windows, predict, contract)
    event = Event(); saved = {}; verify = []
    def stop(row):
        if row['window'] == 3: event.set()
    part = evaluate_windows(source, source.windows, predict, contract, stop_event=event,
                            progress=stop, save=lambda s:saved.update(s))
    assert part['status'] == 'stopped' and part['completed_windows'] == 3
    def resumed(*args, **kw):
        verify.append(kw['verify']); return predict(*args, **kw)
    result = evaluate_windows(source, source.windows, resumed, contract, saved=saved)
    assert result['reports'] == complete['reports'] and verify[0] is True
    with pytest.raises(RuntimeError, match='contract'):
        restore(saved, {**contract, 'protocol':'p0_f9_surface_ccr_i2world_waymo_2hz_v1'}, voxel_count=np.prod(source.shape))


def test_incomplete_native_forecast_never_loads_future_gt(tmp_path, monkeypatch):
    source = fixture(tmp_path)
    monkeypatch.setattr(source, 'metric_targets', lambda w:pytest.fail('future GT read'))
    def incomplete(record, raw, **kw):
        return [raw['history_occ'][-1]]*5, [], {}, {}
    with pytest.raises(RuntimeError, match='SIX'):
        evaluate_windows(source, source.windows, incomplete, dict(windows=len(source.windows)))


def test_native_audit_cli_is_history_only_and_no_model(tmp_path, monkeypatch):
    from tools.real_motion import eval_p0_f9_joint_surface_waymo_10hz as cli
    source = fixture(tmp_path/'data'); out = tmp_path/'audit'
    monkeypatch.setattr(cli.WaymoI2World10HzSource, 'from_files', lambda *a, **kw:source)
    monkeypatch.setattr(cli, 'load_evaluation_model', lambda *a, **kw:pytest.fail('model loaded'))
    monkeypatch.setattr(source, 'metric_targets', lambda w:pytest.fail('future GT read'))
    args = ['--waymo-root', str(source.root), '--out-dir', str(out), '--audit-only', '--expected-scenes','2']
    assert cli.main(argv=args) == 0
    audit = json.loads((out/'audit.json').read_text())
    assert audit['windows'] == 17 and audit['data']['load_interval'] == 1
    assert audit['future_GT_loaded'] is False and audit['model_loaded'] is False
    with pytest.raises(SystemExit): cli.main(argv=args)


def test_native_full_cli_snapshot_resume_relabels_results_without_source_writes(tmp_path, monkeypatch):
    from tools.real_motion import eval_p0_f9_joint_surface_waymo_10hz as cli
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    source = fixture(tmp_path/'data'); out = tmp_path/'eval'; ckpt = tmp_path/'weights'/'mean.pt'
    ckpt.parent.mkdir()
    ckpt.write_bytes(b'FROZEN SOURCE WEIGHTS')
    digest = cli.file_sha256(ckpt); calls = []
    monkeypatch.setattr(cli.WaymoI2World10HzSource, 'from_files', lambda *a, **kw:source)
    monkeypatch.setattr(cli.torch.cuda, 'is_available', lambda:True)
    monkeypatch.setattr(cli, 'SHAPE', source.shape)
    monkeypatch.setattr(cli, 'load_runtime_config', lambda p:None)
    monkeypatch.setattr(cli, 'make_prepare_config', lambda c:PrepareConfig(
        grid=OccupancyGrid(-40,-40,-1,(.4,)*3,source.shape)))
    monkeypatch.setattr(cli, 'load_evaluation_model', lambda *a, **kw:(
        dict(source_epochs=list(cli.AVERAGE_EPOCHS), averaging=True),
        SimpleNamespace(transport=SimpleNamespace(config=SimpleNamespace(history_frames=4)))))
    class Provider:
        def __init__(self, *a, **kw): pass
        def prepare_columns(self, source_unused, record, *, include_gt, raw_window):
            assert not include_gt and raw_window['future_gt_occ'] is None
            rows = predict(record, raw_window, verify=False)
            return SimpleNamespace(baseline=rows[0], state=dict(rec=record), raw=raw_window)
    event = Event()
    class Execution:
        def __init__(self, *a, **kw): pass
        def predict(self, prep):
            calls.append(prep.state['rec']['t0_token'])
            if len(calls) == 3: event.set()
            return prep.baseline, dict(added=0,removed=0), {}, None
        def close(self): pass
    monkeypatch.setattr(cli, 'WaymoSurfaceProvider', Provider)
    monkeypatch.setattr(cli, 'SurfaceBlockExecution', Execution)
    checked = []
    monkeypatch.setattr(cli, 'verify_first_block', lambda *a:checked.append(True))
    args = ['--waymo-root',str(source.root),'--out-dir',str(out),'--checkpoint',str(ckpt),
            '--config',str(ckpt),'--expected-scenes','2','--no-graphs']
    assert cli.main(event, argv=args) == 0
    partial = json.loads((out/'waymo_validation.json').read_text())
    assert partial['status'] == 'stopped' and partial['completed_windows'] == 3
    assert cli.main(argv=[*args,'--resume']) == 0
    final = json.loads((out/'waymo_validation.json').read_text())
    assert final['status'] == 'complete' and final['completed_windows'] == 17
    assert len(calls) == 17 and len(checked) == 2
    assert set(final['reports']['joint']['horizons']) == set(REPORT_KEYS)
    assert cli.file_sha256(ckpt) == digest
    assert cli.main(argv=[*args,'--resume']) == 0 and len(calls) == 17


@pytest.mark.parametrize('mode', ['numpy', 'native_parallel'])
def test_native_real_cpu_surface_all_six_match_reference(tmp_path, mode):
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from tools.real_motion.eval_p0_f9_joint_surface_waymo import WaymoSurfaceProvider
    from tools.real_motion.joint_surface_long_rollout_common import SurfaceBlockExecution, verify_first_block
    shape = (32,32,4); source = fixture(tmp_path, shape=shape, lengths=(16,))
    pcfg = PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1,(.4,)*3,shape))
    model = JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,
                           blocks=1,decoder_blocks=1),width=16,z_bins=4).eval().requires_grad_(False)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    provider = WaymoSurfaceProvider(model,pcfg,'cpu',1)
    execution = SurfaceBlockExecution(provider,mode=mode,workers=2,query_workers=2,graphs=False)
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for window in (source.windows[0],source.windows[4],source.windows[-1]):
            record, raw = source.prediction_inputs(window)
            prep = provider.prepare_columns(None,record,include_gt=False,raw_window=raw)
            dense, edits, _, probabilities = execution.predict(prep)
            verify_first_block(provider,prep.state['rec'],prep,dense,probabilities,execution)
            assert len(dense) == 6 and edits['removed'] == 0 and raw['future_gt_occ'] is None
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    finally:
        execution.close(); torch.set_num_threads(threads)
