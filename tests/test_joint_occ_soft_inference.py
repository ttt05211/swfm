"""Small real readout + timing boundary + full cursor tests; NOT nuScenes scores/FPS."""
import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.joint_occ_experiments.soft_inference import engine, run
from tools.joint_occ_experiments.diagnostics import audit
from tools.joint_occ_experiments.diagnostics.run import model_digest
from tools.joint_occ_experiments.execution import BothActionsExecution
from tools.joint_occ_experiments.model import PROTOCOL
from tools.joint_occ_experiments.performance.probe import legacy_fingerprint
from test_joint_occ_ego import fixture


def test_real_readouts_equal_previous_audit_and_ignore_future_labels():
    model, _, provider, row = fixture()
    model.eval().requires_grad_(False); digest = model_digest(model)
    expected, _, _ = audit.evaluation_window(model, provider, row)
    row[1]['future_gt_occ'] = 'POISON_FUTURE_LABELS'
    actual = engine.quality_predictions(model, provider, row)
    full = engine.quality_predictions(model, provider, row, horizons=range(6))
    for name in engine.ROUTES:
        assert len(full[name]) == 6
        for i, h in enumerate((1, 3, 5)):
            np.testing.assert_array_equal(actual[name][i], expected[name][i])
            np.testing.assert_array_equal(actual[name][i], full[name][h])
    assert model_digest(model) == digest and all(p.grad is None for p in model.parameters())


def test_fresh_timing_rebuilds_strong_and_composes_only_one_hard_route(monkeypatch):
    from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
    model, _, provider, row = fixture()
    model.eval().requires_grad_(False)
    session = BothActionsExecution(model.columns, provider.device)
    provider.paired_eval_sessions = {id(model.columns): session}
    provider.strong.max_match_speed_mps = 30
    # A stationary causal prior, unlike the original artificial fixture's
    # constant displacement at every future horizon.
    row[0]['kta_displacement_xy_m'] = torch.zeros(1, 6, 2)
    row[0]['anchors_xy_t0_m'] = row[0]['source_centroid_xy_t0_m'][:, None].expand(-1, 6, -1)
    backgrounds = copy.deepcopy(row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds'])
    expected = engine.quality_predictions(model, provider, row, horizons=range(6))
    calls = []
    historical = []
    def extract(*a, **kw): historical.append('extract'); return []
    def match(*a, **kw): historical.append('match'); return {}
    monkeypatch.setattr(runtime, 'extract_instances_cropped_exact', extract)
    monkeypatch.setattr(runtime, 'match_instances', match)
    def fresh(*a, **kw):
        calls.append(kw); return backgrounds, [[] for _ in range(6)]
    monkeypatch.setattr(runtime, '_strong_all_horizons', fresh)
    monkeypatch.setattr(engine, 'compose_component_replacements_fast_exact', lambda anchor, *a, **kw: anchor.copy())
    monkeypatch.setattr(runtime, 'baseline_clear_flat_indices', lambda *a, **kw: np.empty(0, np.int64))
    row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds'] = 'POISON_CACHED_STRONG'
    row[1]['future_gt_occ'] = 'POISON_LABELS'
    original = engine.compose_canonical; compositions = []
    def compose(*a, **kw): compositions.append(kw); return original(*a, **kw)
    monkeypatch.setattr(engine, 'compose_canonical', compose)
    try:
        prepared, _ = engine.prepare_speed_row(provider, row)
        assert historical == ['extract', 'match'] and not calls
        for route in engine.SPEED_ROUTES:
            actual = engine.infer(model, provider, prepared, route)
            assert engine.mismatch_voxels(expected[route], actual) == 0
        assert historical == ['extract', 'match']  # no matching inside infer
        assert len(calls) == 3 and len(compositions) == 1
        assert '_speed_history' not in row[1]  # original raw/cache untouched
        assert compositions[0]['thresholds'] == (.5, .95)
    finally: session.close()


def test_timing_alternates_complete_routes_and_retains_failures(monkeypatch):
    _, _, provider, row = fixture()
    volume = np.zeros((2, 2, 2), np.uint8)
    expected = {r: [volume]*6 for r in engine.ROUTES}
    monkeypatch.setattr(engine, 'quality_predictions', lambda *a, **kw: expected)
    preparations = []
    def prepare(*a): preparations.append('prepare'); return row, .03
    monkeypatch.setattr(engine, 'prepare_speed_row', prepare)
    order = []
    def infer(*args, **kwargs):
        route = args[3]; order.append(route); return [volume.copy() for _ in range(6)]
    monkeypatch.setattr(engine, 'infer', infer)
    timing = engine.measure_window(None, provider, row, 0)
    assert preparations == ['prepare'] and timing['excluded_history_prepare_seconds'] == .03
    assert order == list(engine.SPEED_ROUTES)*2+list(reversed(engine.SPEED_ROUTES))
    assert timing['six_frame_byte_parity']
    report = engine.speed_summary([timing], actual_cuda=True)
    assert report['valid_same_quality_route_timing']
    assert report['timing_boundary_version'] == 2
    assert all(r['measurements'] == 2 for r in report['routes'].values())
    assert not engine.speed_summary([timing], actual_cuda=False)['valid_same_quality_route_timing']
    def wrong(*args, **kwargs): return [np.ones_like(volume)]*6
    monkeypatch.setattr(engine, 'infer', wrong)
    failed = engine.measure_window(None, provider, row, 1)
    assert not engine.speed_summary([failed], actual_cuda=True)['valid_same_quality_route_timing']
    with pytest.raises(RuntimeError, match='ALL SIX'): engine.mismatch_voxels([volume]*3, [volume]*3)


def test_cli_quality_then_speed_stop_resume_512_and_source_rng_unchanged(monkeypatch, tmp_path):
    model, _, provider, row = fixture(); initializer = copy.deepcopy(model)
    paths = {name: tmp_path/(name+'.data') for name in ('config', 'train_cache', 'dev_cache',
        'population_manifest', 'base_checkpoint', 'train_info', 'dev_info')}
    for path in paths.values(): path.write_bytes(b'immutable')
    dev = [dict(row[0], scene_name='DEV', t0_token='dev'+str(i)) for i in range(512)]
    root = Path(run.__file__).resolve().parents[3]
    c = dict(modes=['gt'], training_backend='fused', model=model.configs(), epochs=3, seed=21,
        total_updates=3088, eval_population='dev512', eval_keys=[['DEV', r['t0_token']] for r in dev],
        train_keys=[['TRAIN', 'train0']], implementation=legacy_fingerprint(root), thresholds=[.5, .95],
        data={name: run.legacy.sha256(path) for name, path in paths.items() if name not in ('config', 'population_manifest')},
        cache_namespace=dict(train='TRAIN_NS', val='VAL_NS'),
        execution=dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda))
    source = tmp_path/'gt3'; source.mkdir()
    torch.save(dict(protocol=PROTOCOL, contract=c, epoch=3, batch=0, updates=3088,
        models={'gt': model.state_dict()}, optimizers={'gt': 'UNTOUCHED'}, rng='UNTOUCHED'), source/'last.pt')
    original = (source/'last.pt').read_bytes(); original_rng = torch.get_rng_state()
    monkeypatch.setattr(run.legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(run, 'JointOccEgo', lambda config, width: copy.deepcopy(initializer))
    monkeypatch.setattr(run.legacy, 'PilotProvider', lambda *a: provider)
    monkeypatch.setattr(run.legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(run.legacy, 'make_prepare_config', lambda _: provider.pcfg)
    monkeypatch.setattr(run.legacy, 'CachedColumnSource', lambda *a, **kw: SimpleNamespace(nusc=None))
    def reader(dataroot, *, info_pkl): assert info_pkl in (str(paths['train_info']), str(paths['dev_info']))
    monkeypatch.setattr(run.legacy, 'NuScenesWindowSource', reader)
    monkeypatch.setattr(run.legacy, 'load_cache', lambda _: ({}, dev))
    monkeypatch.setattr(run, 'attach_caches', lambda *a: dict(train={'namespace': 'TRAIN_NS'}, val={'namespace': 'VAL_NS'}))
    monkeypatch.setattr(run, 'prefetched_batches', lambda p, s, jobs, d, **kw:
        ([(r, *row[1:]) for r in group] for group in jobs))
    class Execution:
        def __init__(self, *a): pass
        def close(self): pass
    import real_motion.canonical_repair_execution as canonical
    import real_motion.surface_projection_execution as surface
    import real_motion.strong_majority_execution as majority
    monkeypatch.setattr(canonical, 'CanonicalCpuExecution', Execution)
    monkeypatch.setattr(surface, 'SurfaceMapExecution', Execution)
    monkeypatch.setattr(majority, 'ParallelNativeMajority', Execution)
    monkeypatch.setattr(run, 'gt_moving_support_sequence', lambda *a, **kw: None)
    monkeypatch.setattr(run, 'moving_support_masks', lambda *_: np.ones_like(row[1]['future_gt_occ'], bool))
    model.eval().requires_grad_(False)
    predictions = engine.quality_predictions(model, provider, row)
    event = threading.Event()
    def quality(*a, **kw): event.set(); return predictions
    monkeypatch.setattr(run, 'quality_predictions', quality)
    out = tmp_path/'evaluation'
    args = [item for name, path in paths.items() for item in ('--'+name.replace('_', '-'), str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path/'train'),
        '--ccr-val-history-cache', str(tmp_path/'val'), '--resume', str(source/'last.pt'),
        '--out-dir', str(out), '--mode', 'gt', '--eval-population', 'dev512', '--speed-windows', '2']
    assert run.main(args, event) == 0
    first = json.loads((out/'state.json').read_text())
    assert first['windows'] == 1 and not first['speed_windows']
    event.clear(); monkeypatch.setattr(run, 'quality_predictions', lambda *a, **kw: predictions)
    def speed(m, p, r, index, **kw):
        event.set()
        routes = engine.OPTIMIZED_SPEED_ROUTES if kw.get('compare_optimized') else engine.SPEED_ROUTES
        return dict(key=[r[0]['scene_name'], r[0]['t0_token']], six_frame_byte_parity=True,
            mismatched_voxels={r: 0 for r in routes},
            samples=[dict(route=r, repeat=j, seconds=.1, peak_reserved_mib=None, peak_allocated_mib=None)
                     for j in range(2) for r in routes])
    monkeypatch.setattr(run, 'measure_window', speed)
    assert run.main(args+['--resume-evaluation'], event) == 0
    second = json.loads((out/'state.json').read_text())
    assert second['windows'] == 512 and len(second['speed_windows']) == 1
    event.clear()
    assert run.main(args+['--resume-evaluation']) == 0
    final = json.loads((out/'evaluation.json').read_text())
    assert final['status'] == 'complete' and final['scientific_updates'] == 0 and final['speed']['windows'] == 2
    assert not final['speed']['valid_same_quality_route_timing']  # local CPU test is not GPU/FPS evidence
    counts = json.loads((out/'state.json').read_text())['metric_counts']
    for name in engine.ROUTES:
        for key, values in first['metric_counts'][name].items():
            np.testing.assert_array_equal(counts[name][key], np.asarray(values)*512)
    assert (source/'last.pt').read_bytes() == original == (out/'source_snapshot.pt').read_bytes()
    assert torch.equal(original_rng, torch.get_rng_state()) and not (out/'last.pt').exists()
    with pytest.raises(RuntimeError, match='contract changed'):
        run.main(args+['--resume-evaluation', '--speed-repeats', '3'])
    contract = json.loads((out/'contract.json').read_text())
    second['speed_windows'][0]['key'] = ['WRONG', 'WINDOW']
    with pytest.raises(RuntimeError, match='population'): run.validate_state(second, contract)
    # A separate speed-only output uses old integer counts, not old timing.
    from tools.joint_occ_experiments.soft_inference.receipts import completed_quality, discover_completed
    original_files = {name: (out/name).read_bytes() for name in
                      ('contract.json', 'state.json', 'evaluation.json', 'source_snapshot.pt')}
    def forbidden(*a, **kw): raise AssertionError('512-window scoring must NOT run again')
    monkeypatch.setattr(run, 'quality_predictions', forbidden)
    monkeypatch.setattr(run, 'gt_moving_support_sequence', forbidden)
    retime = tmp_path/'speed_only'
    retime_args = args.copy(); retime_args[retime_args.index('--out-dir')+1] = str(retime)
    assert run.main(retime_args+['--reuse-quality-from', str(out), '--compare-optimized-speed']) == 0
    reused = json.loads((retime/'evaluation.json').read_text())
    assert reused['status'] == 'complete' and reused['metrics'] == final['metrics']
    assert reused['contract']['quality_reuse']['old_timing_discarded']
    assert reused['contract']['compare_optimized_speed'] and len(reused['speed']['routes']) == 5
    with pytest.raises(RuntimeError, match='contract changed'):
        run.main(retime_args+['--resume-evaluation', '--compare-optimized-speed', '--fast-horizons', '2'])
    assert json.loads((retime/'state.json').read_text())['metric_counts'] == counts
    assert all((out/name).read_bytes() == value for name, value in original_files.items())
    # v1 receipts remain usable despite their old timing implementation.
    old_contract = json.loads((out/'contract.json').read_text())
    old_contract['protocol'] = 'joint_occ_frozen_soft_hard_dev512_paired_inference_v1'
    old_contract['implementation'] = 'OLD_BOUNDARY_SOURCE'
    old_result = json.loads((out/'evaluation.json').read_text())
    old_result['contract'] = old_contract
    (out/'contract.json').write_text(json.dumps(old_contract))
    (out/'evaluation.json').write_text(json.dumps(old_result))
    _, imported, receipt = completed_quality(out)
    assert imported['windows'] == 512 and not imported['speed_windows']
    assert receipt['old_timing_discarded']
    bad_contract = copy.deepcopy(old_contract); bad_contract['thresholds'] = [.6, .95]
    with pytest.raises(RuntimeError, match='differs: thresholds'):
        completed_quality(out, bad_contract)
    old_result['status'] = 'stopped'
    (out/'evaluation.json').write_text(json.dumps(old_result))
    with pytest.raises(RuntimeError, match='completed compatible'): completed_quality(out)
    with pytest.raises(RuntimeError, match='no completed'): discover_completed(tmp_path)
    assert (source/'last.pt').read_bytes() == original and torch.equal(original_rng, torch.get_rng_state())
