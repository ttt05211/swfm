from dataclasses import replace
import json

import numpy as np
import pytest

from real_motion.planner_origin_audit import PlannerRows, audit, rebuild_poses, sha256, summary
from real_motion.stc_camera_protocol import Frame, PLAN_SHA, PLAN_TAG, STCFourSettingSource


def fixture(tmp_path, *, stationary=False):
    scene = 'scene-0001'
    tokens = [f'{i:032x}' for i in range(14)]
    catalog = {}; rows = {}
    for i, token in enumerate(tokens):
        pose = np.eye(4)
        pose[:3, 3] = [0. if stationary else i * .3, 1., 2.]
        # Preserve a nontrivial t0 yaw and tilt; JSON yaw is relative, not global.
        a, b = .2, .04
        pose[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]]) @ np.array(
            [[1., 0., 0.], [0., np.cos(b), -np.sin(b)], [0., np.sin(b), np.cos(b)]])
        catalog[token] = Frame(token, scene, tokens[i-1] if i else '',
            tokens[i+1] if i+1 < len(tokens) else '', i*500000, pose)
        row = [[*pose[:2, 3], 0.]]
        row += [[pose[0, 3]+j*.4, 1.+j*.1, j*.02] for j in range(1, 7)]
        rows[f'{scene}-{240+i}'] = row  # global-style suffix != scene ordinal
    original = tmp_path / 'planner.json'
    original.write_text(json.dumps(dict(trajs=dict(reversed(list(rows.items()))))), encoding='utf-8')
    plans = tmp_path / 'cache'; (plans / 'records').mkdir(parents=True)
    keys = []
    for i in range(3, 8):
        key = f'{scene}__{tokens[i]}'; keys.append(key)
        p = rebuild_poses(catalog[tokens[i]].pose, rows[f'{scene}-{240+i}'])
        np.savez_compressed(plans / 'records' / (key + '.npz'),
            format='stochocc_transport_cache_v2', protocol_tag=PLAN_TAG, scene=scene,
            history_tokens=tokens[i-3:i+1], future_tokens=tokens[i+1:i+7], future_e2g=p,
            # Reading either forbidden member with allow_pickle=False MUST fail.
            history_semantic=np.array(['do not read'], dtype=object),
            future_semantic=np.array(['no future GT'], dtype=object))
    manifest = dict(format='stochocc_transport_cache_v2', come_protocol_setting='camera_pred',
        come_protocol_tag=PLAN_TAG, future_ego='predicted', come_trajectory_sha256=PLAN_SHA,
        predicted_yaw_semantics='each_future_json_yaw_offset_relative_to_current_pose_non_cumulative',
        hist_last=4, keys=keys, num_windows=len(keys), num_scenes=1)
    (plans / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    # These paths do not exist: audit must never need GT occupancy, STC or sensors.
    source = STCFourSettingSource(tmp_path / 'no_nuscenes', tmp_path / 'no_stc', plans, catalog, cache_mib=0)
    return source, original, rows


def test_nonzero_suffix_sorted_ordinal_and_all_read_only(tmp_path):
    source, path, _ = fixture(tmp_path)
    files = [p for p in tmp_path.rglob('*') if p.is_file()]
    before = {p: sha256(p) for p in files}
    result = audit(source, path, expected_sha=sha256(path))
    assert result['statuses'] == {'compatible_unique_xy': 5}
    assert result['decision'] == 'correspondence_consistent_no_row_shift_evidence'
    assert result['full_scene_rows']['origin_xy_compatible'] == 14
    assert result['cache_reconstruction_pass'] == 5
    row = result['window_audit'][0]
    assert row['t0_ordinal'] == 3 and row['json_key'] == 'scene-0001-243'
    assert row['cumulative_yaw_pose_max_abs_error'] > .01
    assert before == {p: sha256(p) for p in files}
    assert not source.plans and not source.cache
    assert 'no model/GPU' in summary(result)


def test_stationary_origin_ambiguity_not_false_unique_proof(tmp_path):
    source, path, _ = fixture(tmp_path, stationary=True)
    result = audit(source, path, expected_sha=sha256(path))
    assert result['statuses'] == {'compatible_ambiguous_xy': 5}
    assert len(result['window_audit'][0]['compatible_frame_ordinals']) == 14
    assert result['full_scene_rows']['unique_xy'] == 0


def test_shifted_json_origin_detected_independently_of_matching_cache(tmp_path):
    source, path, rows = fixture(tmp_path)
    entries = list(rows)
    # Both original JSON selection and cached future poses consistently use the
    # wrong row. Reconstructing cache alone would falsely validate its t0 origin.
    for i in range(len(entries)-1):
        rows[entries[i]] = rows[entries[i+1]]
    path.write_text(json.dumps(dict(trajs=rows)), encoding='utf-8')
    for window in source.windows:
        i = int(window.t0, 16)
        record = source.plan_cache / 'records' / (window.key+'.npz')
        with np.load(record, allow_pickle=False) as z:
            fields = {k: z[k] for k in ('format', 'protocol_tag', 'scene', 'history_tokens', 'future_tokens')}
        fields['future_e2g'] = rebuild_poses(source.catalog[window.t0].pose, rows[entries[i]])
        np.savez_compressed(record, **fields)
    result = audit(source, path, expected_sha=sha256(path))
    assert result['statuses'] == {'mismatch': 5}
    assert result['cache_reconstruction_pass'] == 5
    assert result['mismatches'][0]['nearest_frame_time_offset_s'] == .5
    assert result['mismatches'][0]['current_xy_error_m'] == pytest.approx(.3)
    assert result['decision'] == 'inspect_mismatches_no_automatic_correction'


@pytest.mark.parametrize('change', ['pose', 'cumulative_yaw', 'order', 'tag', 'scene'])
def test_bad_cache_pose_or_identity(tmp_path, change):
    source, path, rows = fixture(tmp_path)
    window = source.windows[0]
    record = source.plan_cache / 'records' / (window.key+'.npz')
    with np.load(record, allow_pickle=False) as z:
        fields = {k: z[k] for k in ('format', 'protocol_tag', 'scene', 'history_tokens', 'future_tokens', 'future_e2g')}
    if change == 'pose':
        fields['future_e2g'][0, 0, 3] += .5
    elif change == 'cumulative_yaw':
        fields['future_e2g'] = rebuild_poses(source.catalog[window.t0].pose, rows['scene-0001-243'], cumulative=True)
    elif change == 'order':
        fields['history_tokens'] = fields['history_tokens'][::-1]
    else:
        fields['protocol_tag' if change == 'tag' else 'scene'] = 'bad'
    np.savez_compressed(record, **fields)
    if change in ('pose', 'cumulative_yaw'):
        result = audit(source, path, expected_sha=sha256(path))
        assert result['statuses']['mismatch'] == 1
        assert result['cache_reconstruction_pass'] == 4
    else:
        with pytest.raises(ValueError, match='identity'):
            audit(source, path, expected_sha=sha256(path))


def test_wrong_official_hash_fail_closed(tmp_path):
    source, path, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        audit(source, path)


@pytest.mark.parametrize('change', ['count', 'duplicate_suffix', 'nan', 'duplicate_key', 'bad_chain'])
def test_invalid_rows_or_chronology(tmp_path, change):
    source, path, rows = fixture(tmp_path)
    if change == 'count':
        rows.pop('scene-0001-240')
    elif change == 'duplicate_suffix':
        rows['scene-0001-0240'] = rows['scene-0001-240']
        rows.pop('scene-0001-253')
    elif change == 'nan':
        rows['scene-0001-240'][0][0] = float('nan')
    elif change == 'bad_chain':
        frame = next(iter(source.catalog.values()))
        source.catalog[frame.token] = replace(frame, next='wrong')
    if change == 'duplicate_key':
        path.write_text('{"trajs":{},"trajs":{}}', encoding='utf-8')
    else:
        path.write_text(json.dumps(dict(trajs=rows)), encoding='utf-8')
    with pytest.raises(ValueError):
        PlannerRows(source.catalog, path, expected_sha=sha256(path))


def test_manifest_changed_and_json_changed_during_audit(tmp_path):
    source, path, _ = fixture(tmp_path)
    def progress(i, n):
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(RuntimeError, match='JSON changed'):
        audit(source, path, expected_sha=sha256(path), progress=progress)
    manifest = source.plan_cache / 'manifest.json'
    manifest.write_bytes(manifest.read_bytes() + b' ')
    with pytest.raises(RuntimeError, match='manifest changed'):
        audit(source, path, expected_sha=sha256(path))


def test_cache_changed_after_read_rejected(tmp_path):
    source, path, _ = fixture(tmp_path)
    record = source.plan_cache / 'records' / (source.windows[0].key + '.npz')
    def progress(i, n):
        record.write_bytes(record.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='cache record changed during audit'):
        audit(source, path, expected_sha=sha256(path), progress=progress)


@pytest.mark.parametrize('bad_out', ['existing', 'inside_cache', 'contains_cache', 'inside_dataroot'])
def test_cli_refuses_unsafe_or_existing_output_before_read(tmp_path, monkeypatch, bad_out):
    from tools.real_motion import audit_p0_f9_original_planner_t0 as cli
    source, path, _ = fixture(tmp_path)
    existing = tmp_path / 'existing'; existing.mkdir()
    out = dict(existing=existing, inside_cache=source.plan_cache / 'report',
               contains_cache=tmp_path, inside_dataroot=source.root / 'report')[bad_out]
    monkeypatch.setattr('sys.argv', ['audit', '--dataroot', str(source.root), '--plan-cache', str(source.plan_cache),
        '--planner-json', str(path), '--out-dir', str(out)])
    with pytest.raises((ValueError, FileExistsError)):
        cli.main()


def test_cli_metadata_only_end_to_end(tmp_path, monkeypatch):
    from tools.real_motion import audit_p0_f9_original_planner_t0 as cli
    source, path, _ = fixture(tmp_path)
    out = tmp_path / 'report'
    monkeypatch.setattr('sys.argv', ['audit', '--dataroot', str(source.root), '--plan-cache', str(source.plan_cache),
        '--planner-json', str(path), '--out-dir', str(out)])
    monkeypatch.setattr(cli, 'load_catalog', lambda root, scenes: (source.catalog, []))
    # Synthetic JSON only for this test. Production CLI has no hash override.
    monkeypatch.setattr(cli, 'audit', lambda s, p, **kw: audit(s, p, expected_sha=sha256(p), **kw))
    cli.main()
    report = json.loads((out / 'evaluation.json').read_text(encoding='utf-8'))
    assert report['statuses'] == {'compatible_unique_xy': 5}
    assert (out / 'summary.txt').is_file() and not source.root.exists()


def test_nonfinite_rows_and_tilt_z_preservation():
    current = np.eye(4); current[2, 3] = 3.
    rows = np.zeros((7, 3)); rows[1:, :2] = [10., 20.]
    result = rebuild_poses(current, rows)
    assert np.all(result[:, 2, 3] == 3.)
    assert np.array_equal(result[:, :2, 3], np.tile([10., 20.], (6, 1)))
    rows[0, 0] = np.inf
    with pytest.raises(ValueError, match='finite'):
        rebuild_poses(current, rows)
