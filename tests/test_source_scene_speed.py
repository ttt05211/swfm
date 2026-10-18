"""Read-only training resource arithmetic + actual matched H6 entrance."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_scene_speed.report import training, pair
from tools.source_scene_speed.engine import parameters, prepare, fresh, aggregate


def test_training_weighted_by_windows_startup_and_last_replay(tmp_path):
    rows = [dict(event='scene_stage1_train', update=i, windows=4, seconds_per_window=.5) for i in range(1, 19)]
    rows += [dict(event='scene_stage1_train', update=18, windows=1, seconds_per_window=2.)]
    (tmp_path/'progress.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    value = training(tmp_path)
    assert value['post_startup']['windows'] == 5
    assert value['post_startup']['seconds_per_window'] == .8
    assert value['replayed_log_rows'] == 1
    assert value['post_startup']['updates'] == [17, 18]
    rows[-1]['seconds_per_window'] = float('nan')
    (tmp_path/'progress.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError): training(tmp_path)


def test_aggregate_keeps_byte_fail_and_ratio_of_total_time():
    def row(s, changes):
        return dict(seconds=s, changed_voxels=changes, reference_repeat_changed_voxels=1,
                    peak_allocated_mib=5, peak_reserved_mib=6, profile={})
    report = aggregate([row(1., 0), row(3., 2)], 2)
    assert report['window_fps'] == .5 and report['output_frame_fps'] == 3
    assert report['six_frame_ms'] == 2000 and report['byte_failed_calls'] == 1
    assert not report['strict_byte_parity']
    with pytest.raises(ValueError): aggregate([row(1., 0)], 2)


def test_full_runner_single_cuda_small_grid_and_source_readonly(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA required')
    from tools.source_scene_speed import run
    from tools.source_scene_stage1.model import make_model
    from tools.source_scene_stage1.run import PROTOCOL
    from test_source_scene_stage1 import fixture
    base, provider, original = fixture()
    root = tmp_path/'pilot'; paths = {k: tmp_path/k for k in ('config', 'dev_cache', 'dev_info')}
    for path in paths.values(): path.write_text('fixture')
    key = [original[0]['scene_name'], original[0]['t0_token']]
    common = dict(protocol=PROTOCOL, source_policy='shared_pinned', known='TRAIN_OCC_STC_dense_valid',
        train_keys=[['TRAIN', 't']], eval_keys=[key], world_size=2, epochs=1, total_updates=18,
        schedule='fixture_cosine', objective={'CE': 1, 'Lovasz': 1}, seed=21, learning_rates=[.001, .001],
        precision='BF16', execution={}, data={k: run.base.sha256(v) for k, v in paths.items()},
        overrides=[], source_implementation={}, implementation={}, STC_inventory=[])
    sources = {}
    for arm in ('baseline', 'reader'):
        folder = root/arm; folder.mkdir(parents=True)
        model = make_model(arm, base.transport.config, feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32)
        c = dict(common, arm=arm, parameters=parameters(model)['total'], model=model.configs())
        (folder/'results.json').write_text(json.dumps(dict(status='complete', contract=c)))
        (folder/'progress.jsonl').write_text('\n'.join(json.dumps(dict(event='scene_stage1_train',
            update=i, windows=4, seconds_per_window=.2)) for i in range(1, 19)))
        torch.save(dict(protocol=PROTOCOL, contract=c, updates=18, epoch=1, batch=0, models={'gt': model.state_dict()}), folder/'last.pt')
        sources[str(folder/'last.pt')] = run.base.sha256(folder/'last.pt')
    raw = {k: original[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
    raw['history_occ'] = np.array(raw['history_occ'], copy=True)
    raw['history_occ'][:, 2:5, 2:5, 1] = 4
    class Inputs:
        def __init__(self, *args, **kwargs): pass
        def history(self, *args, **kwargs): return original[0], raw, np.arange(4)*.5
        def poses(self, *args): return original[1]['future_poses']
    monkeypatch.setattr(run, 'NuscInputs', Inputs)
    monkeypatch.setattr(run.base, 'make_prepare_config', lambda *args: provider.pcfg)
    monkeypatch.setattr(run.base, 'load_runtime_config', lambda *args: {})
    monkeypatch.setattr(run.base, 'load_cache', lambda *args: ({}, [original[0]]))
    out = tmp_path/'speed'
    argv = ['--root', str(root), '--out', str(out), '--dataroot', str(tmp_path), '--stc-root', str(tmp_path),
            '--windows', '1', '--repeats', '1', '--cpu-workers', '1']
    for k, path in paths.items(): argv += ['--'+k.replace('_', '-'), str(path)]
    run.main(argv)
    result = json.loads((out/'results.json').read_text())
    assert result['status'] == 'complete' and result['scientific_updates'] == 0
    assert len((out/'samples.jsonl').read_text().splitlines()) == 4
    for arm in ('baseline', 'reader'):
        for modality in ('gt', 'stc'):
            assert result['arms'][arm]['fps'][modality]['calls'] == 1
            assert result['arms'][arm]['fps'][modality]['output_frame_fps'] > 0
    for path, sha in sources.items(): assert run.base.sha256(path) == sha
    with pytest.raises(RuntimeError, match='NEW speed output'): run.main(argv)


@pytest.mark.parametrize('arm', ['baseline', 'reader'])
def test_fresh_actual_h6_matches_live_reference_and_never_caches_strong(arm, monkeypatch):
    from test_source_scene_stage1 import fixture
    from tools.source_scene_stage1.model import make_model
    from tools.source_scene_stage1.history import StageHistory
    from tools.shared_source_track.adapter import SharedTrackPredictor
    import tools.source_scene_speed.engine as engine
    base, provider, original = fixture()
    model = make_model(arm, base.transport.config, feature_width=16, z_bins=4,
                       memory_width=4, width=8, chunk_size=32).eval().requires_grad_(False)
    with torch.no_grad():
        model.columns.readout[-1].weight.normal_(std=.2)
        model.transport.residual_head.weight.normal_(std=.01)
    p = parameters(model)
    assert p['total'] == sum(p[k] for k in ('encoder', 'motion_ex_encoder', 'refiner'))
    raw = {k: original[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
    raw['history_occ'] = np.array(raw['history_occ'], copy=True)
    raw['history_occ'][:, 2:5, 2:5, 1] = 4  # real component above original min6
    runner = SharedTrackPredictor(model, provider.pcfg, 'cpu', workers=1)
    runner.history = StageHistory(runner.provider, arm=arm, ram_mib=0)
    poses = original[1]['future_poses']; state = copy.deepcopy(model.state_dict())
    calls = []; builder = engine.backgrounds
    def counted(*args, **kwargs): calls.append(1); return builder(*args, **kwargs)
    monkeypatch.setattr(engine, 'backgrounds', counted)
    try:
        row = runner.prepare(original[0], raw, times=np.arange(4)*.5)
        assert len(row[0]['source_class_id']) > 0
        ready = prepare(row, poses, 'cpu')
        fw = runner.forward(row, poses=poses)
        expected = fw['logits'].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
        profile = {}
        a = fresh(model, runner.provider, ready, profile=profile)
        b = fresh(model, runner.provider, ready)
        np.testing.assert_array_equal(a, expected); np.testing.assert_array_equal(b, expected)
        assert len(calls) == 2 and set(profile) == {'live_KTA', 'encoder_motion_lift', 'fresh_Strong_CLEAR', 'transport', 'memory', 'refiner', 'argmax_CPU'}
        with pytest.raises(ValueError): prepare((row[0], dict(row[1], future_gt_occ=object()), *row[2:]), poses, 'cpu')
        for key, value in state.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    finally: runner.close()
