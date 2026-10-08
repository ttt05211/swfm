import copy
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from real_motion.canonical_causal_repair import build_canonical_evidence, compose_canonical, map_canonical_evidence
from real_motion.canonical_causal_repair import CanonicalRepairHead
from real_motion.surface_canonical_repair import SurfaceCanonicalRepairHead
from tools.real_motion import surface_ccr_validation_common as common
from tools.real_motion import surface_ccr_screen_common as surface
from tools.real_motion import validate_p0_f9_surface_ccr_expanded as cli
from tools.real_motion.height_field_screen_recovery import payload
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from test_canonical_causal_repair import scene
from test_height_field_screen import contract


def one_window(accumulator, key):
    gt = [np.array([[[11, 17, 4, 13]]], np.uint8) for _ in range(6)]
    predictions = {name: [g.copy() for g in gt] for name in common.VARIANTS}
    predictions['baseline'][1][0, 0, 0] = 17
    evidence = SimpleNamespace(actor=np.array([-2, 0]))
    target = np.ones((2, 6, 2), bool); valid = np.ones_like(target)
    score = np.zeros((2, 6, 2), np.float32); score[..., 0] = .6
    accumulator.update(dict(scene_name=key[0], t0_token=key[1]), predictions, gt,
        [np.ones_like(g, bool) for g in gt], evidence, target, valid,
        {'surface_CCR': score, 'frozen_B': score}, dev512_keys={('devscene', '1')}, dev512_scenes={'devscene'})


def compare_state(a, b):
    if isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: compare_state(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): compare_state(x, y)
    else: assert a == b


def test_integer_progress_resume_subsets_and_scene_disjoint_selection(tmp_path):
    full, first, resumed = common.Accumulator(), common.Accumulator(), common.Accumulator()
    keys = [('devscene', '1'), ('devscene', '2'), ('otherscene', '3')]
    for key in keys: one_window(full, key)
    one_window(first, keys[0])
    path = tmp_path/'evaluation_progress.pt'
    c = {'weights': 'pinned', 'population': keys}
    common.save_progress(path, c, first, {'fps_done': True}, {'seconds': 1.})
    speed, perf = common.load_progress(path, c, resumed)
    assert speed['fps_done'] and perf['seconds'] == 1
    for key in keys[1:]: one_window(resumed, key)
    compare_state(full.state_dict(), resumed.state_dict())
    report = resumed.report()['subsets']
    assert report['full4369']['windows'] == 3 and report['dev512']['windows'] == 1
    assert report['outside_dev512_windows']['windows'] == 2
    assert report['outside_dev512_scenes']['windows'] == report['outside_dev512_scenes']['scenes'] == 1
    assert report['full4369']['metrics']['surface_CCR']['mIoU'] == 100
    assert report['full4369']['action_ADD']['surface_CCR/static/all6']['precision'] == 1
    with pytest.raises(RuntimeError, match='identical'):
        common.load_progress(path, {**c, 'weights': 'wrong'}, common.Accumulator())
    state = full.state_dict(); state['cursor'] = 2
    with pytest.raises(RuntimeError, match='cursor'): resumed.load_state_dict(state)
    bad = full.state_dict(); bad['scenes']['devscene']['metrics']['baseline']['oi'] = np.zeros(3, np.float32)
    with pytest.raises(RuntimeError, match='integer'): resumed.load_state_dict(bad)


def test_known_execution_fix_resume_still_rejects_every_scientific_change(tmp_path):
    state = common.Accumulator(); one_window(state, ('devscene', '1'))
    path = tmp_path/'progress.pt'
    old = dict(implementation=cli.LEGACY_EXECUTION_IMPLEMENTATION, weights='same',
               population=[('devscene','1')], thresholds=[.5, None], val_cache_namespace='same')
    new = {**old, 'implementation': 'new execution logging'}
    common.save_progress(path, old, state, {'fps_done': True}, {'seconds': 2.})
    kwargs = dict(compatible_implementations=(cli.LEGACY_EXECUTION_IMPLEMENTATION,))
    with pytest.raises(RuntimeError, match='identical'):
        common.load_progress(path, new, common.Accumulator())
    resumed = common.Accumulator()
    speed, _ = common.load_progress(path, new, resumed, **kwargs)
    compare_state(state.state_dict(), resumed.state_dict())
    assert speed['fps_done']
    for field, value in (('weights','changed'), ('population',[('other','1')]),
                         ('thresholds',[.6,None]), ('val_cache_namespace','changed')):
        with pytest.raises(RuntimeError, match='identical'):
            common.load_progress(path, {**new,field:value}, common.Accumulator(), **kwargs)


def test_legacy_execution_resume_requires_unchanged_head_file(monkeypatch):
    monkeypatch.setattr(cli, 'sha256', lambda path: cli.UNCHANGED_SURFACE_HEAD_SHA256)
    assert cli.compatible_execution_implementations('unused') == (cli.LEGACY_EXECUTION_IMPLEMENTATION,)
    monkeypatch.setattr(cli, 'sha256', lambda path: 'modified head')
    assert cli.compatible_execution_implementations('unused') == ()


def surface_checkpoint():
    baseline = CanonicalRepairHead(8)
    head = SurfaceCanonicalRepairHead(8); head.initialize_from(baseline)
    c = {**contract(), 'protocol': surface.PROTOCOL, 'teacher_sha256': 'teacher', 'config_fingerprint': 'config',
         'dev_manifest_fingerprint': 'manifest', 'model': surface.model_contract(head), 'static_only_training': True,
         'thresholds': dict(CCR_ADD=.5, CCR_REMOVE=None, old_Local=(.5, .5, None)),
         'train_fraction': 1., 'epochs': 3, 'epoch_batches': [1, 1, 1],
         'epoch_batch_sizes': [[1], [1], [1]], 'schedule_steps': 3}
    saved = payload(head, torch.optim.AdamW(head.parameters()), np.random.default_rng(1), c,
                    epoch=3, batch=0, updates=3, executed=3, reports={}, protocol=surface.PROTOCOL)
    return saved, baseline


def test_fixed_third_epoch_loader_rejects_wrong_teacher_cursor_and_dynamic_mutation():
    saved, baseline = surface_checkpoint()
    args = dict(teacher_sha='teacher', config_fp='config', manifest_fp='manifest', device=torch.device('cpu'))
    head = cli.load_surface(saved, baseline, **args)
    assert not head.training and not any(p.requires_grad for p in head.parameters())
    with pytest.raises(RuntimeError, match='contract'):
        cli.load_surface(saved, baseline, **{**args, 'teacher_sha': 'wrong'})
    bad = copy.deepcopy(saved); bad.update(epoch=2, updates=2, executed=2)
    with pytest.raises(RuntimeError, match='third'): cli.load_surface(bad, baseline, **args)
    bad = copy.deepcopy(saved); bad['head']['encoder.0.weight'].add_(1)
    with pytest.raises(RuntimeError, match='shared CCR'): cli.load_surface(bad, baseline, **args)
    bad = copy.deepcopy(saved); bad['head']['phase.weight'][0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'): cli.load_surface(bad, baseline, **args)


def test_paired_fps_has_all_six_live_phase_and_full_20_population(monkeypatch):
    grid, prep = scene(); plain = build_canonical_evidence(prep, grid)
    provider = SimpleNamespace(device=torch.device('cpu'), pcfg=SimpleNamespace(grid=grid), workers=2)
    teacher = torch.nn.Module(); teacher.transport = torch.nn.Identity()
    baseline = CanonicalRepairHead(8).eval().requires_grad_(False)
    head = SurfaceCanonicalRepairHead(8); head.initialize_from(baseline); head.eval().requires_grad_(False)
    output = {'history_source_context': torch.randn(1, 8), 'future_transport_queries': torch.randn(1, 6, 8)}
    records = [dict(scene_name='scene-'+str(i%18), t0_token=str(i), features=torch.zeros(1+i%3, 8)) for i in range(64)]
    def history(*args, **kwargs):
        return SimpleNamespace(canonical_evidence=plain, current_pose=np.eye(4), future_poses=[np.eye(4)]*6)
    calls = []
    def forecast(h, provider, motion, model, probability_fn, **kwargs):
        plan = map_canonical_evidence(h.canonical_evidence, prep, grid)
        assert plan.context.shape[-1] == 8  # phase computed inside timed callback
        p = probability_fn(model, h.canonical_evidence, plan, output, provider.device)
        calls.append(type(model))
        dense = compose_canonical(prep.baseline, h.canonical_evidence, plan, p[..., 0], p[..., 1])
        assert len(dense) == 6
        return dict(probability=p, dense=dense, stages_seconds={'shared_encode_six_readouts': .001})
    monkeypatch.setattr(common, 'prepare_history', history); monkeypatch.setattr(common, 'forecast_six', forecast)
    report = common.paired_speed(provider, None, records, teacher, head, baseline)
    assert report['probability_and_six_dense_parity_windows'] == 20
    assert report['dynamic_byte_parity_windows'] == 20 and len(report['trials']) == 180
    assert report['selected_execution'] == 'surface_eager'  # no CUDA graph on CPU
    assert not report['actual_cuda'] and report['memory_mib'] is None
    assert calls.count(SurfaceCanonicalRepairHead) == 160
    assert calls.count(CanonicalRepairHead) == 80


def test_summary_has_iou_all_horizons_and_does_not_claim_independence():
    state = common.Accumulator(); one_window(state, ('devscene', '1'))
    result = dict(status='complete', **state.report())
    text = cli.summary(result)
    assert 'IoU=' in text and 'MovingMacro=' in text
    assert 'road11=' in text and '3.0s' in text and 'NOT independent tests' in text


def test_actual_expanded_cli_count_resume_and_reuse_completed_fps(monkeypatch, tmp_path):
    # External nuScenes IO/old Local/FPS are mocked; new head forward, phase,
    # compositor, integer counts, snapshots and recovery run for real.
    from test_ccr_screen import fixture
    teacher, provider, rows, baseline = fixture()
    teacher.transport = torch.nn.Identity(); teacher.transport.config = SimpleNamespace(history_frames=4)
    teacher.columns = SimpleNamespace(source_dim=8, config=None)
    prep = provider.prepare_columns()
    prep.window = SimpleNamespace(t0_token='0', future_tokens=tuple(str(i) for i in range(6)))
    provider.joint = teacher
    provider.prepare_columns = lambda *args, **kwargs: copy.deepcopy(prep)
    files = {}
    for name in ('config', 'checkpoint', 'ccr-checkpoint', 'frozen-b-checkpoint', 'base-checkpoint',
                 'dev-cache', 'population-manifest', 'dev-info'):
        files[name] = tmp_path/name; files[name].write_bytes(name.encode())
    for name in ('dataroot', 'ccr-val-history-cache'):
        files[name] = tmp_path/name; files[name].mkdir()
    records = [dict(scene_name='devscene' if i < 512 else 'other', t0_token=str(i), features=torch.zeros(1, 8))
               for i in range(4369)]
    keys = tuple((r['scene_name'], r['t0_token']) for r in records)
    ck = dict(cursor_epoch=19, model_configs={}, cache_fingerprints={'dev': cli.sha256(files['dev-cache'])},
        info_fingerprints={'dev': cli.sha256(files['dev-info'])}, dev_manifest_fingerprint='manifest', dev_keys=keys[:512])
    saved, _ = surface_checkpoint()
    head = SurfaceCanonicalRepairHead(8); head.initialize_from(baseline)
    with torch.no_grad(): head.static_readout[-1].bias.add_(4)
    saved['head'] = head.state_dict()
    saved['contract'].update(teacher_sha256=cli.sha256(files['checkpoint']),
        config_fingerprint=cli.stable_json_fingerprint({}), warm_start_head_sha256='placeholder')
    # Real B serialized input; loader mocked because its scheduling contract is
    # covered separately, while original shared-weight equality is NOT mocked.
    torch.save({'head': baseline.state_dict()}, files['frozen-b-checkpoint'])
    saved['contract']['warm_start_head_sha256'] = cli.sha256(files['frozen-b-checkpoint'])
    torch.save(saved, files['ccr-checkpoint'])
    monkeypatch.setattr(cli, 'CLEAN_SHA256', cli.sha256(files['base-checkpoint']))
    monkeypatch.setattr(cli, 'load_runtime_config', lambda *args: {})
    monkeypatch.setattr(cli, 'make_prepare_config', lambda cfg: provider.pcfg)
    monkeypatch.setattr(cli, 'require_cuda', lambda device: torch.device('cpu'))
    monkeypatch.setattr(cli, 'load_joint', lambda *args, **kwargs: (ck, teacher))
    monkeypatch.setattr(cli, 'load_point_head', lambda *args, **kwargs: baseline.eval().requires_grad_(False))
    monkeypatch.setattr(cli, 'load_manifest', lambda *args: ({'parent_keys': keys[:512], 'manifest_fingerprint': 'manifest'}, keys[:64], None))
    monkeypatch.setattr(cli, 'load_cache', lambda *args: ({}, records))
    monkeypatch.setattr(cli, 'PilotProvider', lambda *args: provider)
    monkeypatch.setattr(cli, 'NuScenesWindowSource', lambda *args, **kwargs: SimpleNamespace(nusc=None))
    monkeypatch.setattr(cli, 'namespace', lambda *args: 'fixed')
    monkeypatch.setattr(cli, 'validate_manifest', lambda *args: {'complete': True})
    monkeypatch.setattr(cli, 'CausalGeometryCache', lambda *args, **kwargs: SimpleNamespace(namespace='fixed', stats=lambda: {}, close=lambda: None))
    monkeypatch.setattr(cli, 'prefetch_raw_columns', lambda p, s, records: ((r, copy.deepcopy(prep.raw)) for r in records))
    monkeypatch.setattr(cli.ccr, 'old_execution', lambda *args: nullcontext())
    monkeypatch.setattr(cli, '_attach_old_local_fixed_geometry', lambda *args: None)
    monkeypatch.setattr(cli.columns, 'candidate_plan', lambda *args: None)
    monkeypatch.setattr(cli.columns, 'predict_probabilities', lambda *args: None)
    monkeypatch.setattr(cli, 'actions_from_probabilities', lambda *args: None)
    monkeypatch.setattr(cli, 'compose_dense', lambda before, *args: before.copy())
    monkeypatch.setattr(cli, 'gt_moving_support_sequence', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, 'moving_support_masks', lambda *args: [np.ones(provider.pcfg.grid.shape_hwd, bool)]*6)
    speed_calls = []
    def speed(*args, **kwargs):
        speed_calls.append(1)
        return dict(selected_execution='surface_graph', six_frame_amortized_FPS={'surface_graph': 40.},
            six_frame_mean_seconds={'surface_graph': .15}, p90_six_ms={'surface_graph': 160.},
            stages_mean_ms={'surface_graph': {'test': 1.}}, boundary='test-only mock',
            surface_descriptor_prepare_seconds_per_window=0.)
    monkeypatch.setattr(cli, 'paired_speed', speed)
    execution_calls = []
    execution_type = cli.SurfaceExecution
    def execution(*args, **kwargs):
        execution_calls.append(kwargs)
        return execution_type(*args, **kwargs)
    monkeypatch.setattr(cli, 'SurfaceExecution', execution)
    argv = [v for name, path in files.items() for v in ('--'+name, str(path))]
    argv += ['--ccr-cpu-execution', 'numpy', '--ccr-cpu-workers', '1']
    full, stop, resumed = [tmp_path/n for n in ('uninterrupted', 'stop', 'resumed')]
    assert cli.main(argv + ['--out-dir', str(full), '--max-windows', '4']) == 0
    assert cli.main(argv + ['--out-dir', str(stop), '--max-windows', '2']) == 0
    # Reproduce a stopped ORIGINAL invocation, with its exact implementation
    # fingerprint. Resume preserves integer counts and already completed FPS.
    old_progress = torch.load(stop/'evaluation_progress.pt', weights_only=False)
    old_progress['contract']['implementation'] = cli.LEGACY_EXECUTION_IMPLEMENTATION
    torch.save(old_progress, stop/'evaluation_progress.pt')
    assert cli.main(argv + ['--out-dir', str(resumed), '--max-windows', '4', '--resume-eval', str(stop/'evaluation_progress.pt')]) == 0
    assert len(speed_calls) == 2  # third invocation reuses completed FPS
    a = torch.load(full/'evaluation_progress.pt', weights_only=False)
    b = torch.load(resumed/'evaluation_progress.pt', weights_only=False)
    compare_state(a['accumulator'], b['accumulator'])
    assert b['accumulator']['cursor'] == 4
    assert (resumed/'ccr_checkpoint_snapshot.pt').is_file()
    assert all(v['graphs'] is False for v in execution_calls)
    import json
    result = json.loads((resumed/'expanded_validation.json').read_text())
    assert result['quality_eval_execution'] == 'eager'
    assert result['speed']['selected_execution'] == 'surface_graph'
    assert 'saved prefix' in result['timing_note']
    assert result['val_cache'] == {}
    row = json.loads((resumed/'progress.jsonl').read_text().splitlines()[-1])
    assert row['surface_execution'] == 'eager'
    assert row['surface_execution_counts']['eager_chunks'] > 0
    assert row['surface_execution_counts'].get('captures_verified',0) == 0
    assert row['input_wait_seconds'] >= 0 and row['canonical_points'] > 0
