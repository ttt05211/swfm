"""Read-only matching/log arithmetic + actual no-ICP full6 CPU/CUDA speed entrance."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tools.source_scene_integrated_speed import engine, report, run


def source_pair(root, config, paths, key, inventory=None):
    from tools.source_scene_kta_query import model as query, run as query_run
    from tools.source_scene_integrated import model as integrated, run as integrated_run
    kw = dict(feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32)
    common = dict(source_policy='shared_pinned', registration='none', known='TRAIN_OCC_STC_dense_valid',
        train_keys=[['TRAIN', 't']], eval_keys=[key]+[['V', str(i)] for i in range(63)],
        world_size=2, epochs=1, total_updates=18, epochs_updates=[18], seed=21,
        schedule='fixture_cosine', objective=dict(CE=1., Lovasz=1., ego=0., motion=0., existence=0.),
        learning_rates=[.001, .001], precision='BF16', execution={}, overrides=[], STC_inventory=[list(v) for v in inventory or []],
        data={k: run.base.sha256(v) for k, v in paths.items()}, implementation={})
    folders = []
    for name, package, protocol, arm, event in (
            ('query', query, query_run.PROTOCOL, 'kta_query_only', 'scene_stage2_kta_query_train'),
            ('integrated', integrated, integrated_run.PROTOCOL, 'integrated_encoder_AdaGN', 'scene_integrated_train')):
        folder = root/name; folder.mkdir(parents=True); folders.append(folder)
        model = package.make_model(config, **kw)
        with torch.no_grad():
            model.columns.readout[-1].weight.normal_(std=.1)
            model.transport.residual_head.weight.normal_(std=.01)
            model.transport.history_yaw.weight.normal_(std=.01)
        c = dict(common, protocol=protocol, arm=arm, model=model.configs(), parameters=engine.parameters(model)['total'])
        if name == 'integrated': c['reference_results_sha256'] = run.base.sha256(folders[0]/'results.json')
        result = dict(status='complete', contract=c, updates=18, epoch=1,
            evaluations={'1': dict(windows=64, keys=common['eval_keys'])})
        (folder/'results.json').write_text(json.dumps(result))
        (folder/'progress.jsonl').write_text('\n'.join(json.dumps(dict(event=event,
            update=i, windows=4 if i < 18 else 1, seconds_per_window=.2)) for i in range(1, 19)))
        torch.save(dict(protocol=protocol, contract=c, updates=18, epoch=1, batch=0,
                        models={'gt': model.state_dict()}), folder/'last.pt')
    return folders


def test_matching_complete_source_logs_and_control_binding(tmp_path):
    from test_source_scene_kta_query import fixture
    model, runner, row = fixture()
    try:
        p = tmp_path/'config'; p.write_text('fixture')
        a, b = source_pair(tmp_path, model.transport.config, {'config': p}, ['V', 'v'])
        before = {str(f): run.base.sha256(f) for d in (a, b) for f in d.iterdir()}
        records, loaded = run.load_pair(a, b)
        assert set(loaded) == {'KTA_query', 'integrated'}
        assert records['integrated']['training']['post_startup']['windows'] == 5
        for f, sha in before.items(): assert run.base.sha256(f) == sha
        v = json.loads((b/'results.json').read_text()); v['contract']['known'] = 'lidar_known'
        (b/'results.json').write_text(json.dumps(v))
        with pytest.raises(ValueError): report.pair(a, b)
        v['contract']['known'] = 'TRAIN_OCC_STC_dense_valid'; v['contract']['reference_results_sha256'] = 'wrong'
        (b/'results.json').write_text(json.dumps(v))
        with pytest.raises(ValueError, match='original KTA-query'): report.pair(a, b)
    finally: runner.close()


def test_log_event_startup_duplicate_and_malformed_timing(tmp_path):
    rows = [dict(event='wanted', update=i, windows=4, seconds_per_window=.5) for i in range(1, 19)]
    rows += [dict(event='wanted', update=18, windows=1, seconds_per_window=2.),
             dict(event='other', update=19, windows=4, seconds_per_window=.01)]
    path = tmp_path/'progress.jsonl'; path.write_text('\n'.join(map(json.dumps, rows)))
    v = report.training(tmp_path, 'wanted')
    assert v['post_startup']['seconds_per_window'] == .8 and v['replayed_log_rows'] == 1
    rows[-2]['seconds_per_window'] = float('nan'); path.write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError): report.training(tmp_path, 'wanted')


def test_selector_requires_unique_completed_integrated(tmp_path):
    from tools.source_scene_integrated.run import PROTOCOL
    value = dict(status='complete', epoch=1, updates=18, contract=dict(protocol=PROTOCOL, total_updates=18))
    for name in ('a', 'b'):
        d = tmp_path/('scene_integrated_1ep_'+name); d.mkdir(); (d/'results.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='explicitly'): report.select_integrated(tmp_path)
    assert report.select_integrated(tmp_path, d) == d.resolve()


def test_changed_STC_files_or_root_rejected(tmp_path):
    job = dict(scene_name='scene', history_tokens=['a']*4)
    path = tmp_path/'scene/a/labels.npz'; path.parent.mkdir(parents=True); path.write_bytes(b'fixture')
    st = path.stat(); inventory = [(str(path.resolve()), st.st_size, st.st_mtime_ns)]
    run.validate_stc(tmp_path, [job], inventory)
    path.write_bytes(b'changed fixture')
    with pytest.raises(RuntimeError, match='differs'): run.validate_stc(tmp_path, [job], inventory)


@pytest.mark.parametrize('arm', ['KTA_query', 'integrated'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fresh_full6_matches_actual_query_entrance_without_ICP_or_cache(arm, device, monkeypatch):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    from test_source_scene_kta_query import fixture
    from tools.source_scene_integrated.model import make_model
    from tools.source_scene_kta_query.adapter import forward_window
    from real_motion import shared_source_track
    model, runner, original = fixture(device)
    try:
        if arm == 'integrated': model = make_model(model.transport.config, feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32).to(device)
        model.eval().requires_grad_(False)
        with torch.no_grad():
            model.columns.readout[-1].weight.normal_(std=.15)
            model.transport.residual_head.weight.normal_(std=.01)
            model.transport.history_yaw.weight.normal_(std=.01)
        raw = {k: original[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
        monkeypatch.setattr(shared_source_track, 'register_history_shape', lambda *a, **k: pytest.fail('ICP called'))
        row = runner.prepare(original[0], raw, times=np.arange(4)*.5)
        poses = original[1]['future_poses']; ready = engine.prepare(row, poses, device)
        state = copy.deepcopy(model.state_dict()); builder = engine.backgrounds; calls = []
        monkeypatch.setattr(engine, 'backgrounds', lambda *a, **k: (calls.append(1), builder(*a, **k))[1])
        with torch.no_grad():
            fw = forward_window(model, runner.provider, row, poses=poses, fresh=True)
            expected = fw['logits'].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
        profile = {}; first = engine.fresh(model, runner.provider, ready, profile=profile)
        second = engine.fresh(model, runner.provider, ready)
        assert len(calls) == 2 and first.shape == second.shape == expected.shape
        if device == 'cpu':
            np.testing.assert_array_equal(first, expected); np.testing.assert_array_equal(second, expected)
        assert set(profile) == {'live_KTA', 'encoder_motion_lift', 'fresh_Strong_CLEAR', 'transport', 'memory', 'refiner', 'argmax_CPU'}
        for k, v in state.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
        with pytest.raises(ValueError): engine.prepare((row[0], dict(row[1], future_gt_occ=object()), *row[2:]), poses, device)
    finally: runner.close()


def test_actual_single_gpu_runner_two_passes_both_modalities_readonly(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA')
    from test_source_scene_kta_query import fixture
    model, runner, original = fixture()
    try:
        paths = {k: tmp_path/k for k in ('config', 'dev_cache', 'dev_info')}
        for p in paths.values(): p.write_text('fixture')
        key = [original[0]['scene_name'], original[0]['t0_token']]
        inventory = []
        for token in original[0]['history_tokens']:
            p = tmp_path/'stc'/original[0]['scene_name']/str(token)/'labels.npz'
            p.parent.mkdir(parents=True); p.write_bytes(b'fixture')
            st = p.stat(); inventory.append((str(p.resolve()), st.st_size, st.st_mtime_ns))
        a, b = source_pair(tmp_path/'pilot', model.transport.config, paths, key, inventory)
        before = {str(f): run.base.sha256(f) for d in (a, b) for f in d.iterdir()}
        raw = {k: original[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
        class Inputs:
            def __init__(self, *a, **k): pass
            def history(self, *a, **k): return original[0], raw, np.arange(4)*.5
            def poses(self, *a): return original[1]['future_poses']
        monkeypatch.setattr(run, 'NuscInputs', Inputs)
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda *a: runner.provider.pcfg)
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: {})
        monkeypatch.setattr(run.base, 'load_cache', lambda *a: ({}, [original[0]]))
        out = tmp_path/'speed'
        argv = ['--control-dir', str(a), '--integrated-dir', str(b), '--out', str(out),
                '--dataroot', str(tmp_path), '--stc-root', str(tmp_path/'stc'), '--windows', '1', '--repeats', '2', '--cpu-workers', '1']
        for k, p in paths.items(): argv += ['--'+k.replace('_', '-'), str(p)]
        run.main(argv)
        result = json.loads((out/'results.json').read_text()); assert result['status'] == 'complete'
        assert result['scientific_updates'] == 0
        samples = [json.loads(v) for v in (out/'samples.jsonl').read_text().splitlines()]
        assert len(samples) == 8
        assert [v['arm'] for v in samples] == ['KTA_query']*2+['integrated']*4+['KTA_query']*2
        for arm in ('KTA_query', 'integrated'):
            for kind in ('gt', 'stc'):
                assert result['arms'][arm]['fps'][kind]['calls'] == 2
                assert result['arms'][arm]['fps'][kind]['output_frame_fps'] > 0
        for f, sha in before.items(): assert run.base.sha256(f) == sha
        with pytest.raises(RuntimeError, match='NEW speed output'): run.main(argv)
    finally: runner.close()
