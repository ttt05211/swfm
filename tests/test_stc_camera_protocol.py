from dataclasses import replace
import json
from threading import Event
import zipfile
import io

import numpy as np
import pytest
import torch

from real_motion.stc_camera_protocol import (Frame, PLAN_SHA, PLAN_TAG, SETTINGS,
    STCFourSettingSource, validate_pose, semantics)
from real_motion.waymo_i2world import fingerprint
from tools.real_motion.stc_camera_evaluation import evaluate, restore
from tools.real_motion.prepare_stc_camera_data import extract


def fixture(tmp_path, *, shape=(8, 8, 4), length=14):
    root, stc, plans = [tmp_path / x for x in ('nuscenes', 'stc', 'plans')]
    (plans / 'records').mkdir(parents=True)
    catalog = {}; scene = 'scene-0001'; tokens = [f'{i:032x}' for i in range(1, length + 1)]
    for i, t in enumerate(tokens):
        pose = np.eye(4); pose[0, 3] = i * .04
        catalog[t] = Frame(t, scene, tokens[i - 1] if i else '', tokens[i + 1] if i + 1 < length else '',
                           i * 500000, validate_pose(pose))
        sem = np.full(shape, 17, np.uint8); sem[:, :, 0] = 11
        sem[2:4, 2:4, 1:3] = 4
        for directory in (root / 'gts', stc):
            p = directory / scene / t; p.mkdir(parents=True)
            np.savez_compressed(p / 'labels.npz', semantics=sem,
                                mask_lidar=np.zeros(shape, bool), mask_camera=np.zeros(shape, bool))
    keys = []
    for i in range(3, length - 6):
        key = scene + '__' + tokens[i]; keys.append(key)
        future = np.stack([catalog[t].pose for t in tokens[i + 1:i + 7]])
        future[:, 1, 3] = np.arange(1, 7) * .1
        np.savez_compressed(plans / 'records' / (key + '.npz'), format='stochocc_transport_cache_v2',
            protocol_tag=PLAN_TAG, scene=scene, history_tokens=tokens[i - 3:i + 1], future_tokens=tokens[i + 1:i + 7],
            future_e2g=future,
            # Deliberate garbage: a successful loader MUST NOT read these.
            history_semantic=np.array(['BEVStereo must not be read'], object),
            future_semantic=np.array(['future GT must not be read'], object),
            base_semantic=np.array(['old forecast must not be read'], object))
    manifest = dict(format='stochocc_transport_cache_v2', come_protocol_setting='camera_pred',
        come_protocol_tag=PLAN_TAG, future_ego='predicted', come_trajectory_sha256=PLAN_SHA,
        predicted_yaw_semantics='each_future_json_yaw_offset_relative_to_current_pose_non_cumulative',
        hist_last=4, keys=keys, num_windows=len(keys), num_scenes=1)
    (plans / 'manifest.json').write_text(json.dumps(manifest))
    source = STCFourSettingSource(root, stc, plans, catalog, shape=shape, cache_mib=1)
    return source


def test_stc_full_history_and_pred_world_poses_ignore_all_old_semantics(tmp_path):
    s = fixture(tmp_path); selected, audit = s.select('all'); s.preflight(selected)
    w = selected[0]
    rec, raw = s.prediction_inputs(w, 'stc_pred')
    assert len(rec['history_tokens']) == 4 and len(rec['future_tokens']) == 6
    assert raw['future_gt_occ'] is None and raw['history_observed'].all()
    assert raw['history_occ'][0, 2, 2, 1] == 4  # not masked to free
    assert len(raw['future_poses']) == 6
    assert raw['future_poses'][-1][1, 3] == pytest.approx(.6)
    assert s.catalog[w.future[-1]].pose[1, 3] == 0
    _, gt = s.prediction_inputs(w, 'occ_gt')
    assert not gt['history_observed'].any()  # baseline's original lidar history
    assert gt['future_poses'][-1][1, 3] == 0
    assert audit['windows'] == len(selected)
    assert set(raw) == {'history_occ', 'history_observed', 'history_poses', 'future_poses', 'future_gt_occ'}


def test_pred_inputs_invariant_to_future_gt_pose_and_masks(tmp_path):
    s = fixture(tmp_path); s.preflight(s.windows); w = s.windows[0]
    _, old = s.prediction_inputs(w, 'stc_pred')
    for t in w.future:
        new = np.eye(4); new[0, 3] = 1000
        s.catalog[t] = replace(s.catalog[t], pose=validate_pose(new))
    _, new = s.prediction_inputs(w, 'stc_pred')
    assert all(np.array_equal(a, b) for a, b in zip(old['future_poses'], new['future_poses']))
    assert np.array_equal(old['history_occ'], new['history_occ'])


@pytest.mark.parametrize('bad', ['pose', 'order', 'tag', 'scene', 'future_z'])
def test_reject_wrong_plan_pose_identity_or_tag(tmp_path, bad):
    s = fixture(tmp_path); w = s.windows[0]; path = s.plan_cache / 'records' / (w.key + '.npz')
    with np.load(path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    if bad == 'pose':
        data['future_e2g'][0, 0, 0] = 2
    elif bad == 'order':
        data['history_tokens'] = data['history_tokens'][::-1]
    elif bad == 'future_z':
        data['future_e2g'][0, 2, 3] = 1
    elif bad == 'tag':
        data['protocol_tag'] = 'old_cumulative_yaw'
    else:
        data['scene'] = 'scene-0002'
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        s.preflight([w])


def test_missing_stc_never_substitutes_gt_and_changed_inputs_rejected(tmp_path):
    s = fixture(tmp_path); w = s.windows[0]
    path = s._path('stc', w.scene, w.history[0]); path.unlink()
    with pytest.raises(FileNotFoundError):
        s.preflight([w])
    np.savez_compressed(path, semantics=np.full(s.shape, 17, np.uint8))
    s.preflight([w]); s.prediction_inputs(w, 'stc_pred')
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='changed'):
        s.prediction_inputs(w, 'stc_pred')


def test_population_intersection_is_explicit_no_prefix_fallback(tmp_path):
    s = fixture(tmp_path); parent = [(w.scene, w.t0) for w in s.windows]
    parent.insert(0, ('scene-0001', 'unknown'))
    selected, audit = s.select('dev512', parent)
    assert selected == s.windows and audit['missing_parent_keys'] == [['scene-0001', 'unknown']]
    with pytest.raises(ValueError, match='fewer than 64'):
        s.select('dev64', parent)
    with pytest.raises(ValueError, match='duplicate'):
        s.select('dev512', parent + parent[-1:])


def test_semantic_cast_does_not_wrap_and_pose_validation():
    for dtype, label in [(np.uint16, 273), (np.int16, -1), (np.float32, 4)]:
        with pytest.raises(ValueError):
            semantics(np.full((2, 2, 1), label, dtype), (2, 2, 1))
    for pose in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.diag([1, 1, -1, 1])):
        with pytest.raises(ValueError):
            validate_pose(pose)


def fake_predict(record, raw, verify):
    assert raw['future_gt_occ'] is None
    return [raw['history_occ'][-1].copy() for _ in range(6)], {'test': 0.}


def test_four_setting_atomic_resume_and_all_forecasts_before_targets(tmp_path, monkeypatch):
    s = fixture(tmp_path); s.preflight(s.windows); contract = dict(windows=len(s.windows), weight='fixed')
    count = [0]; actual = s.metric_targets
    def predict(*args, verify):
        count[0] += 1; return fake_predict(*args, verify)
    def targets(w):
        assert count[0] % 4 == 0
        return actual(w)
    monkeypatch.setattr(s, 'metric_targets', targets)
    complete = evaluate(s, s.windows, predict, contract)
    event = Event(); saved = {}
    first = evaluate(s, s.windows, predict, contract, stop_event=event,
        progress=lambda row: event.set() if row['window'] == 2 else None, save=lambda v: saved.update(v))
    assert first['status'] == 'stopped' and first['completed_windows'] == 2
    seen = []
    def resumed(*args, verify):
        seen.append(verify); return predict(*args, verify=verify)
    final = evaluate(s, s.windows, resumed, contract, saved=saved)
    assert final['reports'] == complete['reports'] and seen[:4] == [True] * 4
    assert final['status'] == 'complete'
    with pytest.raises(RuntimeError, match='contract'):
        restore(saved, contract | {'weight': 'other'}, np.prod(s.shape))
    saved['counts']['stc_pred'][0][0][0] += 1
    saved['fingerprint'] = fingerprint({k: v for k, v in saved.items() if k != 'fingerprint'})
    with pytest.raises(ValueError, match='counts'):
        restore(saved, contract, np.prod(s.shape))


def test_mid_setting_failure_commits_no_partial_window_or_gt_read(tmp_path, monkeypatch):
    s = fixture(tmp_path); s.preflight(s.windows); saved = {}; calls = [0]
    monkeypatch.setattr(s, 'metric_targets', lambda w: pytest.fail('future target accessed'))
    def broken(*args, verify):
        calls[0] += 1
        if calls[0] == 3:
            return [], {}
        return fake_predict(*args, verify)
    with pytest.raises(RuntimeError, match='SIX'):
        evaluate(s, s.windows, broken, dict(windows=len(s.windows)), save=lambda v: saved.update(v))
    assert saved['completed_windows'] == 0
    assert all(np.asarray(v).sum() == 0 for v in saved['counts'].values())


def npz_bytes(**arrays):
    buffer = io.BytesIO(); np.savez(buffer, **arrays); return buffer.getvalue()


def test_compact_extraction_lossless_and_resume_no_overwrite(tmp_path):
    path = tmp_path / 'official.zip'; sem = np.arange(18, dtype=np.int32).reshape(3, 3, 2)
    name = 'stc-results/scene-0001/' + 'a' * 32 + '/labels.npz'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name, npz_bytes(semantics=sem, mask_camera=np.zeros_like(sem)))
    out = tmp_path / 'compact'
    a = extract(path, out, shape=sem.shape, expected_size=0)
    dest = out / 'scene-0001' / ('a' * 32) / 'labels.npz'
    with np.load(dest, allow_pickle=False) as z:
        assert z.files == ['semantics'] and z['semantics'].dtype == np.uint8
        assert np.array_equal(z['semantics'], sem)
    assert a['frames'] == 1 and a['status'] == 'complete'
    with pytest.raises(FileExistsError):
        extract(path, out, shape=sem.shape, expected_size=0)
    assert extract(path, out, resume=True, shape=sem.shape, expected_size=0)['files'] == a['files']
    np.savez_compressed(dest, semantics=np.zeros_like(sem))
    with pytest.raises(RuntimeError, match='changed'):
        extract(path, out, resume=True, shape=sem.shape, expected_size=0)


@pytest.mark.parametrize('name', ['../outside', '/outside', 'stc-results/scene-0001/x/labels.npz'])
def test_unsafe_or_wrong_zip_names_rejected(tmp_path, name):
    path = tmp_path / 'bad.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name, npz_bytes(semantics=np.zeros((3, 3, 2), np.uint8)))
    with pytest.raises(ValueError):
        extract(path, tmp_path / 'out', expected_size=0, shape=(3, 3, 2))


@pytest.mark.parametrize('backend', ['numpy', 'native_parallel'])
def test_real_four_setting_model_no_weight_mutations_and_exactness(tmp_path, backend):
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from tools.real_motion.eval_p0_f9_joint_surface_waymo import WaymoSurfaceProvider
    from tools.real_motion.joint_surface_long_rollout_common import SurfaceBlockExecution, verify_first_block
    s = fixture(tmp_path, shape=(32, 32, 4)); s.preflight(s.windows[:1])
    model = JointSurfaceCCR(LocalSTWMV17Config(history_frames=4, d_model=16, semantic_dim=4,
        blocks=1, decoder_blocks=1), width=16, z_bins=4).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = PrepareConfig(grid=OccupancyGrid(-6.4, -6.4, -1., (.4,) * 3, s.shape))
    provider = WaymoSurfaceProvider(model, cfg, 'cpu', 1)
    execution = SurfaceBlockExecution(provider, mode=backend, workers=1, query_workers=1, graphs=False)
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        @torch.no_grad()
        def predict(rec, raw, verify):
            prep = provider.prepare_columns(None, rec, include_gt=False, raw_window=raw)
            dense, _, details, probability = execution.predict(prep)
            if verify:
                verify_first_block(provider, prep.state['rec'], prep, dense, probability, execution)
            return dense, details
        result = evaluate(s, s.windows[:1], predict, dict(windows=1))
        assert result['status'] == 'complete' and set(result['reports']) == set(SETTINGS)
        assert result['verified_settings'] == list(SETTINGS)
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    finally:
        execution.close(); torch.set_num_threads(threads)


def test_catalog_read_matches_standard_lidar_top_ego_pose_without_annotations(tmp_path):
    from real_motion.stc_camera_protocol import load_catalog
    base = tmp_path / 'v1.0-trainval'; base.mkdir()
    rows = dict(scene=[dict(token='scene', name='scene-0001')],
        sample=[dict(token='t', scene_token='scene', prev='', next='', timestamp=0)],
        sensor=[dict(token='lidar', channel='LIDAR_TOP'), dict(token='cam', channel='CAM_FRONT')],
        calibrated_sensor=[dict(token='lidarcal', sensor_token='lidar'), dict(token='camcal', sensor_token='cam')],
        sample_data=[dict(sample_token='t', is_key_frame=True, calibrated_sensor_token='lidarcal', ego_pose_token='ego'),
                     dict(sample_token='t', is_key_frame=True, calibrated_sensor_token='camcal', ego_pose_token='cameraego')],
        ego_pose=[dict(token='ego', translation=[1, 2, 3], rotation=[1, 0, 0, 0]),
                  dict(token='cameraego', translation=[999, 999, 999], rotation=[1, 0, 0, 0])])
    for name, data in rows.items():
        (base / (name + '.json')).write_text(json.dumps(data))
    catalog, provenance = load_catalog(tmp_path, {'scene-0001'})
    assert np.array_equal(catalog['t'].pose[:3, 3], [1, 2, 3])
    assert len(provenance) == 6 and not (base / 'sample_annotation.json').exists()


def test_audit_cli_reads_no_model_or_future_targets(tmp_path, monkeypatch):
    from tools.real_motion import eval_p0_f9_joint_surface_stc as cli
    s = fixture(tmp_path)
    monkeypatch.setattr(cli.STCFourSettingSource, 'from_files', lambda *a, **kw: s)
    monkeypatch.setattr(cli, 'load_evaluation_model', lambda *a, **kw: pytest.fail('model loaded'))
    monkeypatch.setattr(s, 'metric_targets', lambda *a: pytest.fail('future target read'))
    out = tmp_path / 'audit'
    args = ['--dataroot', str(s.root), '--stc-root', str(s.stc_root), '--plan-cache', str(s.plan_cache),
            '--population', 'all', '--out-dir', str(out), '--audit-only']
    assert cli.main(argv=args) == 0
    report = json.loads((out / 'audit.json').read_text())
    assert report['future_targets_loaded'] is False and report['model_loaded'] is False
    with pytest.raises(SystemExit):
        cli.main(argv=args)
