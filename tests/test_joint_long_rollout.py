"""Causal/open-loop semantics, four-history ABI, metric and recovery safety."""
import copy
import json
from types import SimpleNamespace
from threading import Event
import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.prepared import PrepareConfig
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.nuscenes_adapter import WindowTokens
from real_motion.motion_transport import FEATURE_NAMES
from real_motion.local_history_contract import four_frame_motion_inputs
from real_motion.joint_causal_columns import FULL4_EXT_PROTOCOL
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import joint_long_rollout_common as common
from tools.real_motion import eval_p0_f9_joint_zero_shot_long_rollout as cli
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256


def window(scene, token):
    return WindowTokens(scene, (token+'h0', token+'h1', token+'h2', token), token,
        tuple(token+f'f{i}' for i in range(12)))


def record(w):
    return dict(scene_name=w.scene_name, t0_token=w.t0_token,
        history_tokens=('excluded0', 'excluded1', *w.history_tokens), future_tokens=w.future_tokens[:6])


def test_population_is_explicit_balanced_complete_and_not_old_dev64():
    windows = [window(s, f'{s}{i}') for s in ('a', 'b') for i in range(36)]
    rows = [record(w) for w in windows]
    parent = [(w.scene_name, w.t0_token) for w in windows]
    chosen, audit = common.select_long_population(rows, windows[:-5], parent, 'dev64')
    assert len(chosen) == 64 and audit['eligible_windows'] == 67
    assert audit['excluded_short_future_keys'] == [list(k) for k in parent[-5:]]
    assert [w.scene_name for w, _ in chosen[:4]] == ['a', 'b', 'a', 'b']
    all_chosen, _ = common.select_long_population(rows, windows[:-5], parent, 'dev512')
    assert len(all_chosen) == 67  # NEVER silently invent 512 windows.
    with pytest.raises(RuntimeError, match='fewer than 64'):
        common.select_long_population(rows, windows[:63], parent, 'dev64')


def test_population_rejects_duplicates_missing_and_wrong_future_order():
    w = window('dev', 't'); row = record(w); key = [('dev', 't')]
    with pytest.raises(RuntimeError, match='duplicate cache'):
        common.select_long_population([row, row], [w], key, 'dev512')
    with pytest.raises(RuntimeError, match='duplicate long'):
        common.select_long_population([row], [w, w], key, 'dev512')
    with pytest.raises(RuntimeError, match='missing'):
        common.select_long_population([row], [w], [('dev', 'absent')], 'dev512')
    row['future_tokens'] = row['future_tokens'][::-1]
    with pytest.raises(RuntimeError, match='order mismatch'):
        common.select_long_population([row], [w], key, 'dev512')


def test_timestamps_reject_scene_crossing_gaps_and_rate_change():
    w = window('s', 't'); tokens = w.history_tokens+w.future_tokens
    samples = {t: dict(timestamp=i*500000, scene_token='s', next=tokens[i+1] if i < 15 else '')
        for i, t in enumerate(tokens)}
    nusc = SimpleNamespace(get=lambda table, t: samples[t])
    common.validate_timestamps(nusc, w)
    samples[tokens[7]]['timestamp'] += 300000
    with pytest.raises(RuntimeError, match='2Hz'): common.validate_timestamps(nusc, w)
    samples[tokens[7]]['timestamp'] -= 300000
    samples[tokens[5]]['next'] = tokens[7]
    with pytest.raises(RuntimeError, match='noncontiguous'): common.validate_timestamps(nusc, w)
    samples[tokens[5]]['scene_token'] = 'other'
    with pytest.raises(RuntimeError, match='scene boundary'): common.validate_timestamps(nusc, w)


def test_nominal_timestamps_allow_cumulative_jitter_and_audit_actual_horizons():
    w = window('s', 't'); tokens = w.history_tokens+w.future_tokens
    samples = {t: dict(timestamp=1532402927647951+i*510000, scene_token='s',
        next=tokens[i+1] if i < 15 else '') for i, t in enumerate(tokens)}
    nusc = SimpleNamespace(get=lambda table, t: samples[t])
    row = common.validate_timestamps(nusc, w)
    assert row['max_nominal_deviation_s'] == pytest.approx(.12)
    audit = common.summarize_timestamps([row])
    assert audit['windows_exceeding_old_60ms_cumulative_check'] == 1
    assert audit['actual_report_times_s']['6.0']['median'] == pytest.approx(6.12)
    assert audit['horizons_are_nominal_keyframe_steps']
    # A local 100ms keyframe jitter also remains legal; not a missing frame.
    samples[tokens[7]]['timestamp'] += 100000
    common.validate_timestamps(nusc, w)


def test_timestamp_guards_still_reject_wrong_rate_nonmonotonic_and_nonfinite():
    w = window('s', 't'); tokens = w.history_tokens+w.future_tokens
    samples = {t: dict(timestamp=i*600000, scene_token='s', next=tokens[i+1] if i < 15 else '')
        for i, t in enumerate(tokens)}
    nusc = SimpleNamespace(get=lambda table, t: samples[t])
    with pytest.raises(RuntimeError, match='2Hz'): common.validate_timestamps(nusc, w)
    for i, token in enumerate(tokens): samples[token]['timestamp'] = i*500000
    samples[tokens[7]]['timestamp'] = samples[tokens[6]]['timestamp']
    with pytest.raises(RuntimeError, match='strictly increasing'): common.validate_timestamps(nusc, w)
    samples[tokens[7]]['timestamp'] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'): common.validate_timestamps(nusc, w)


def test_inherited_visibility_uses_initial_evidence_not_prediction_or_future_mask():
    grid = OccupancyGrid(0, 0, 0, (1., 1., 1.), (5, 4, 2))
    obs = np.zeros((4, *grid.shape_hwd), bool); obs[0, 2, 1, 0] = True
    raw = dict(history_observed=obs, history_poses=[np.eye(4)]*4,
        future_observed=np.ones_like(obs), history_occ=np.zeros_like(obs, np.uint8))
    translated = np.eye(4); translated[0, 3] = 1
    inherited = common.inherited_observation_masks(raw, [translated]*4, grid, workers=2)
    assert inherited.sum() == 4 and inherited[:, 1, 1, 0].all()
    raw['history_occ'][:] = 17; raw['future_observed'][:] = False
    assert np.array_equal(inherited, common.inherited_observation_masks(raw, [translated]*4, grid))
    raw['history_observed'][:] = False
    assert not common.inherited_observation_masks(raw, [translated]*4, grid).any()


def state_fixture(empty=False):
    grid = OccupancyGrid(-6.4, -6.4, -1, (.4, .4, .4), (32, 32, 4))
    history = np.full((4, *grid.shape_hwd), 17, np.uint8)
    history[:, :, :, 0] = 11
    if not empty:
        for f in range(4): history[f, 10+f:13+f, 10:13, 1:3] = 4
    pcfg = PrepareConfig(grid=grid)
    state = common.build_four_history_state(history, [np.eye(4)]*4, [np.eye(4)]*6,
        pcfg, StrongW2DetConfig(), torch.device('cpu'))
    return history, pcfg, state


@pytest.mark.parametrize('empty', (False, True))
def test_actual_builder_four_history_empty_abi_and_causal_motion(empty):
    history, pcfg, state = state_fixture(empty)
    rec = state['rec']; n = len(state['current'])
    assert n == (0 if empty else 1)
    assert rec['local_semantic_tube'].shape == (n, 6, 20, 20)
    assert (rec['local_semantic_tube'][:, :2] == 17).all()
    assert not rec['target_source_mask_tube'][:, :2].any()
    for name in ('hist_valid_0', 'hist_valid_1', 'hist_vel_1_x'):
        assert torch.count_nonzero(rec['features'][:, FEATURE_NAMES.index(name)]) == 0
    if not empty:
        assert np.isclose(state['velocities'][0][0], .8)
        assert torch.allclose(rec['kta_displacement_xy_m'][0, :, 0], torch.arange(1, 7)*.4, atol=1e-6)
    view = four_frame_motion_inputs(rec['features'], rec['local_semantic_tube'],
        rec['frame_motion_features'], rec['target_source_mask_tube'])
    assert view[1].shape[1] == 4 and view[2].shape[1] == 4
    common.assert_four_inputs_equal(rec, rec)
    assert not any(k in rec for k in ('existence', 'target_yaw_rad', 'supervised_source', 'future_gt_occ'))


def test_first_block_reconstruction_matches_last_four_of_legacy_six_history_cache():
    last4, pcfg, rebuilt = state_fixture()
    history6 = np.full((6, *pcfg.grid.shape_hwd), 17, np.uint8)
    history6[:, :, :, 0] = 11
    for f in range(6): history6[f, 8+f:11+f, 10:13, 1:3] = 4
    assert np.array_equal(history6[-4:], last4)
    cached = common.legacy._build_block_state(history6, [np.eye(4)]*6, [np.eye(4)]*6,
        pcfg, StrongW2DetConfig(), torch.device('cpu'))
    common.assert_four_inputs_equal(cached['rec'], rebuilt['rec'])


def test_synthetic_preparation_only_last_four_joint_predictions_and_no_source_api(monkeypatch):
    history, pcfg, state = state_fixture()
    first_predictions = [history[0].copy(), history[1].copy(), *list(history)]
    first_predictions[0][:] = 2; first_predictions[1][:] = 3  # excluded predicted slots
    first_raw = dict(history_observed=np.ones_like(history, bool), history_poses=[np.eye(4)]*4,
        future_gt_occ=np.zeros_like(history))  # must NOT carry this field over
    seen = {}
    class Provider:
        device = torch.device('cpu'); workers = 1; strong = StrongW2DetConfig()
        def prepare_columns(self, source, rec, *, include_gt, raw_window):
            assert source is None and include_gt is False
            seen.update(record=rec, raw=raw_window)
            return SimpleNamespace(raw=raw_window)
    provider = Provider(); provider.pcfg = pcfg
    provider.joint = SimpleNamespace(columns=SimpleNamespace(config=SimpleNamespace()))
    monkeypatch.setattr(common, 'prepare_causal_evidence', lambda *a, **k: {})
    prep = common.synthetic_preparation(first_predictions, first_raw, [np.eye(4)]*12, window('s', 't'), provider)
    assert np.array_equal(prep.raw['history_occ'], history)
    assert prep.raw['future_gt_occ'] is None and len(prep.raw['history_poses']) == 4
    assert seen['record']['t0_token'] == 'tf5'
    assert seen['record']['future_tokens'] == tuple('t'+f'f{i}' for i in range(6, 12))
    first_predictions[-1][:] = 17
    assert not np.array_equal(prep.raw['history_occ'][-1], first_predictions[-1])


def test_dense_gate_checks_all_six_including_unreported_half_seconds():
    a = [np.full((2, 2, 2), 17, np.uint8) for _ in range(6)]; b = copy.deepcopy(a)
    common.assert_dense_equal(a, b)
    b[0][0, 0, 0] = 4
    with pytest.raises(RuntimeError, match='horizon=0'): common.assert_dense_equal(a, b)


def test_metrics_raw_aggregation_empty_support_and_separate_averages():
    raw = common.legacy._new_raw()
    gt = np.array([[[4, 17]]], np.uint8); pred = np.array([[[4, 4]]], np.uint8)
    for hi in range(6):
        common.legacy._update_raw(raw, hi, pred if hi >= 3 else gt, gt, np.zeros_like(gt, bool), 17)
    with np.errstate(all='raise'): result = common.finalize_metrics(raw)
    assert result['average_1s_2s_3s']['IoU'] == 100
    assert result['average_4s_5s_6s']['IoU'] == 50
    assert np.isnan(result['average_4s_5s_6s']['MovingMicro'])
    # Empty supports are serialized NA, not zero or invalid JSON NaN.
    assert cli.finite_json(result)['per_horizon']['6.0']['MovingMicro'] is None


def test_histogram_metrics_match_frozen_dense_counts_all_classes_and_horizons():
    rng = np.random.default_rng(19)
    reference, actual = common.legacy._new_raw(), common.legacy._new_raw()
    for h in range(6):
        for _ in range(2):
            pred = rng.integers(0, 18, (7, 9, 4), dtype=np.uint8)
            gt = rng.integers(0, 18, pred.shape, dtype=np.uint8)
            moving = rng.random(pred.shape) < .3
            common.legacy._update_raw(reference, h, pred, gt, moving, 17)
            common.update_metrics(actual, h, pred, gt, moving, 17)
    for key in reference: assert np.array_equal(reference[key], actual[key]), key


@pytest.mark.parametrize('empty', (False, True))
def test_real_joint_network_renderer_and_both_blocks_no_gt_or_weight_changes(empty):
    from real_motion.joint_causal_columns import JointCausalColumns
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.causal_column_completion import ColumnConfig
    from tools.real_motion.joint_column_common import JointColumnProvider
    from tools.real_motion.joint_column_full_common import prepare_causal_evidence
    from concurrent.futures import ThreadPoolExecutor
    history, pcfg, state = state_fixture(empty)
    torch.manual_seed(19)
    joint = JointCausalColumns(LocalSTWMV17Config(history_frames=4, d_model=16, semantic_dim=4,
        blocks=1, decoder_blocks=1), ColumnConfig(width=16, semantic_dim=4, layers=1, z_bins=4))
    joint.eval().requires_grad_(False)
    original = {k: v.clone() for k, v in joint.state_dict().items()}
    # Skip only checkpoint I/O; preparation, Strong gates, registration, rigid
    # renderer, candidates, linked queries, and the complete network are REAL.
    provider = JointColumnProvider.__new__(JointColumnProvider)
    provider.joint = joint; provider.model = joint.transport; provider.reference = None
    provider.control = None; provider.reference_enabled = False; provider.latents_checked = True
    provider.pcfg = pcfg; provider.strong = StrongW2DetConfig(); provider.workers = 2
    provider.device = torch.device('cpu')
    w = window('s', 't'); rec = state['rec']
    rec.update(record(w))
    raw = dict(history_occ=history, history_observed=np.ones_like(history, bool),
        history_poses=[np.eye(4)]*4, future_poses=[np.eye(4)]*6, future_gt_occ=None)
    evidence = prepare_causal_evidence(raw, pcfg, provider.strong, 2, state=state, column_config=joint.columns.config)
    evidence['prepared_state'] = state; raw['_column_causal_preparation'] = evidence
    previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        first = provider.prepare_columns(None, rec, include_gt=False, raw_window=raw)
        with ThreadPoolExecutor(max_workers=2) as pool:
            reference, _ = common.predict_joint_block(first, joint.columns, pcfg.grid, provider.device,
                batch_size=256, candidate_pool=pool)
            joint.columns.column_inference_optimized = True
            joint.columns.column_inference_verify_remaining = 3
            pred1, _ = common.predict_joint_block(first, joint.columns, pcfg.grid, provider.device,
                batch_size=256, candidate_pool=pool)
            common.assert_dense_equal(reference, pred1)
            # The normal deployment path independently prepares and renders.
            provider.load_raw_columns = lambda source, record, *, include_gt: raw
            joint.columns.column_inference_optimized = False
            _, deployed = common.columns.forecast_columns(provider, None, rec, joint.columns,
                common.THRESHOLDS, 256)
            common.assert_dense_equal(deployed, pred1)
            joint.columns.column_inference_optimized = True
            future_poses = []
            for i in range(12):
                pose = np.eye(4); angle = .003*(i+1)
                pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                pose[0, 3] = .1*(i+1)
                future_poses.append(pose)
            second = common.synthetic_preparation(pred1, raw, future_poses, w, provider)
            pred2, _ = common.predict_joint_block(second, joint.columns, pcfg.grid, provider.device,
                batch_size=256, candidate_pool=pool)
            from real_motion.causal_rollout_handoff import handoff_from_prepared
            carry = handoff_from_prepared(first, pred1[-1])
            linked = common.synthetic_preparation(pred1, raw, future_poses, w, provider,
                motion_handoff=carry, component_frames=second.state['components_by_frame'])
            linked_pred, _ = common.predict_joint_block(linked, joint.columns, pcfg.grid, provider.device,
                batch_size=256, candidate_pool=pool)
            assert len(linked_pred) == 6 and linked.state['motion_handoff_audit']['memory_only_sources_added'] == 0
            assert linked.state['current'] is second.state['current']
            assert np.array_equal(linked.raw['history_occ'], second.raw['history_occ'])
            assert linked.raw['future_gt_occ'] is None
            if empty: assert linked.state['motion_handoff_audit']['matched_sources'] == 0
        assert provider.columns_checked and joint.columns.column_inference_verify_remaining == 0
        assert np.array_equal(second.raw['history_occ'], np.stack(pred1[-4:]))
        assert second.raw['future_gt_occ'] is None and len(second.raw['history_occ']) == 4
        assert np.array_equal(second.state['current_pose'], future_poses[5])
        assert len(pred1+pred2) == 12 and all(x.shape == pcfg.grid.shape_hwd for x in pred1+pred2)
        assert all(np.isin(x, np.arange(18)).all() for x in pred1+pred2)
        for key, value in joint.state_dict().items(): assert torch.equal(value, original[key]), key
    finally: torch.set_num_threads(previous_threads)


def test_resume_contract_counts_and_exactness_fail_closed():
    contract = {'population': [['s', 't']], 'thresholds': [.5, .5, None]}
    saved = dict(contract_fingerprint=stable_json_fingerprint(contract), completed_windows=1,
        first_block_exactness_passed=True, raw_counts={k: v.tolist() for k, v in common.legacy._new_raw().items()})
    assert common.validate_resume_state(saved, contract, 2)[0] == 1
    with pytest.raises(RuntimeError, match='contract changed'):
        common.validate_resume_state(saved, {**contract, 'thresholds': [.5, .5, .95]}, 2)
    bad = copy.deepcopy(saved); bad['raw_counts']['occ_inter'][0] = 1
    with pytest.raises(RuntimeError, match='intersection'): common.validate_resume_state(bad, contract, 2)
    bad = copy.deepcopy(saved); bad['first_block_exactness_passed'] = False
    with pytest.raises(RuntimeError, match='exactness'): common.validate_resume_state(bad, contract, 2)


@pytest.mark.parametrize('comparison', (False, True))
@pytest.mark.parametrize('aligned', (False, True))
def test_cli_interrupt_resume_counts_original_t0_and_readonly_checkpoint(tmp_path, monkeypatch, comparison, aligned):
    # Real CLI orchestration/serialization; model/nuScenes math tested separately.
    files = {}
    for k in ('config', 'checkpoint', 'dev-cache', 'population-manifest', 'base-checkpoint', 'dev-info'):
        files[k] = tmp_path/k; files[k].write_bytes(k.encode())
    original = files['checkpoint'].read_bytes(); out = tmp_path/'eval'
    grid = OccupancyGrid(0, 0, 0, (1., 1., 1.), (2, 2, 2)); pcfg = PrepareConfig(grid=grid)
    windows = [window('dev', 'a'), window('dev', 'b')]; records = [record(w) for w in windows]
    parent = [('dev', 'a'), ('dev', 'b')]
    cfg = {}; digest = lambda path: CLEAN_SHA256 if Path(path) == files['base-checkpoint'] else real_sha(path)
    from pathlib import Path
    real_sha = cli.sha256
    ck = dict(protocol=FULL4_EXT_PROTOCOL, cursor_epoch=19, cursor_batch=0, successful_updates=97748,
        attempted_updates=97748, cache_fingerprints={'dev': real_sha(files['dev-cache'])},
        info_fingerprints={'dev': real_sha(files['dev-info'])}, dev_manifest_fingerprint='manifest',
        dev_keys=parent, train_keys=[('train', 't')])
    class Joint:
        transport = SimpleNamespace(config=SimpleNamespace(history_frames=4))
        columns = SimpleNamespace(history_frames=4)
        def eval(self): return self
        def requires_grad_(self, flag): assert flag is False; return self
    initial = np.full((4, *grid.shape_hwd), 17, np.uint8)
    raw = dict(history_occ=initial, history_observed=np.zeros_like(initial, bool),
        history_poses=[np.eye(4)]*4, future_poses=[np.eye(4)]*6, future_gt_occ=None)
    predictions = [np.full(grid.shape_hwd, 17, np.uint8) for _ in range(6)]
    class Source:
        nusc = SimpleNamespace()
        def iter_windows(self, **kwargs): assert kwargs == dict(history=4, future=12); return iter(windows)
        def pose(self, token): return np.eye(4)
        def load_semantics(self, scene, token):
            if not resumed[0]: event.set()  # interrupt only AFTER predictions, while scoring first window
            return predictions[0]
    class Provider:
        def __init__(self, *args): self.strong = StrongW2DetConfig(); self.workers = 1; self.pcfg = pcfg; self.device = torch.device('cpu')
        def prepare_columns(self, source, record, *, include_gt, raw_window):
            assert not include_gt and raw_window.get('future_gt_occ') is None
            return SimpleNamespace(raw=raw_window, baseline=predictions, state={'current': [], 'components_by_frame': [], 'rec': record})
    event = Event(); resumed = [False]; supports = []
    monkeypatch.setattr(cli, 'sha256', digest)
    monkeypatch.setattr(cli, 'load_runtime_config', lambda *a: cfg)
    monkeypatch.setattr(cli, 'make_prepare_config', lambda *a: pcfg)
    monkeypatch.setattr(cli, 'load_joint', lambda *a, **k: (ck, Joint()))
    monkeypatch.setattr(cli, 'load_manifest', lambda *a: (dict(selected_key_fingerprint=cli.DEV64_FP,
        manifest_fingerprint='manifest', parent_keys=parent), parent, None))
    monkeypatch.setattr(cli, 'load_cache', lambda *a: ({}, records))
    monkeypatch.setattr(cli, 'NuScenesWindowSource', lambda *a, **k: Source())
    monkeypatch.setattr(cli, 'CachedColumnSource', lambda source, *a: source)
    monkeypatch.setattr(cli, 'EvaluationJointColumnProvider', Provider)
    if aligned:
        official = tmp_path/'official.pkl'; official.write_bytes(b'official')
        monkeypatch.setattr(cli.genie, 'verify_info', lambda *a: cli.genie.INFO_SHA256)
        monkeypatch.setattr(cli.genie, 'validate_grid', lambda *a: None)
        monkeypatch.setattr(cli.genie, 'AlignedColumnProvider', Provider)
        monkeypatch.setattr(cli.genie, 'select_population', lambda *a: (list(zip(windows, records)), dict(
            population='all', selected_keys=parent, eligible_windows=2, requested_parent_windows=48,
            scenes=1, missing_six_history_cache_keys=[list(parent[0])], alignment=cli.genie.POPULATION_PROTOCOL)))
    monkeypatch.setattr(cli, 'prefetch_raw_columns', lambda p, s, rows, **k: ((r, copy.deepcopy(raw)) for r in rows))
    monkeypatch.setattr(common, 'validate_timestamps', lambda *a: dict(
        intervals_s=[.5]*15, relative_times_s=((np.arange(16)-3)*.5).tolist(), max_nominal_deviation_s=0.))
    monkeypatch.setattr(common, 'predict_joint_block', lambda *a: (copy.deepcopy(predictions), {'added': 0}))
    monkeypatch.setattr(common, 'build_four_history_state', lambda *a: {'rec': {}})
    monkeypatch.setattr(common, 'assert_four_inputs_equal', lambda *a: None)
    monkeypatch.setattr(common, 'synthetic_preparation', lambda *a, **k: SimpleNamespace(state={
        'current': [], 'components_by_frame': [], 'motion_handoff_audit': {'matched_sources': 0}}))
    monkeypatch.setattr(cli, 'handoff_from_prepared', lambda *a, **k: object())
    monkeypatch.setattr(cli, 'forecast_clean_reference', lambda *a: copy.deepcopy(predictions+predictions))
    monkeypatch.setattr(cli.runtime, '_stage_gpu_inputs', lambda *a: None)
    monkeypatch.setattr(cli.runtime, '_release_gpu_inputs', lambda *a: None)
    monkeypatch.setattr(cli.runtime, '_forecast_once', lambda *a: predictions)
    monkeypatch.setattr(cli.columns, 'forecast_columns', lambda *a: (None, predictions))
    def support(nusc, token, ftokens, horizons, **kw):
        supports.append((token, tuple(horizons)))
        return [(np.zeros(grid.shape_hwd, bool), [], {})]*6
    monkeypatch.setattr(cli, 'gt_moving_support_sequence', support)
    argv = ['eval']+[x for k, path in files.items() for x in ('--'+k, str(path))]
    argv += ['--dataroot', str(tmp_path), '--out-dir', str(out), '--population', 'dev512', '--device', 'cpu']
    if aligned:
        argv += ['--population', 'all', '--population-alignment', 'geniedrive_code10s', '--geniedrive-info', str(official)]
    if comparison:
        argv += ['--handoff-modes', 'redetect,reconciled,transport_history']
        if not aligned: argv += ['--compare-e14']
    monkeypatch.setattr('sys.argv', argv)
    assert cli.main(event) == 130
    saved = json.loads((out/'evaluation_state.json').read_text())
    assert saved['completed_windows'] == 1 and not (out/'evaluation.json').exists()
    if comparison:
        assert set(saved['comparison_raw_counts']) == {'reconciled', 'transport_history'} | (set() if aligned else {'clean_E14_native6'})
    resumed[0] = True; event.clear(); monkeypatch.setattr('sys.argv', argv+['--resume'])
    assert cli.main(event) == 0
    result = json.loads((out/'evaluation.json').read_text())
    assert result['windows'] == 2 and result['first_block_exactness_passed']
    assert supports == [('a', common.REPORT_HORIZONS), ('b', common.REPORT_HORIZONS)]
    assert files['checkpoint'].read_bytes() == original
    assert 'average_4s_5s_6s' in (out/'summary.txt').read_text()
    if comparison:
        assert set(result['handoff_comparison']) == set(cli.HANDOFF_MODES) | (set() if aligned else {'clean_E14_native6'})
        assert result['no_automatic_route_selection']
        if not aligned: assert result['handoff_comparison']['clean_E14_native6']['history_frames'] == 6
    if aligned:
        assert result['public_code_alignment_only'] and not result['paper_table_population_verified']
        assert set(result['geniedrive_code_compatibility']) == (set(cli.HANDOFF_MODES) if comparison else {'redetect'})
        assert 'GENIEDRIVE PUBLIC CODE COMPATIBILITY' in (out/'summary.txt').read_text()
