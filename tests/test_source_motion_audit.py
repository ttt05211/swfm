"""Frozen counterfactual geometry, matched metrics and conditional box truth."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_motion_audit import engine, statistics as stats, run


def test_intervention_only_changes_future_motion():
    output = dict(residual_xy_m=torch.ones(2, 6, 2), yaw_delta_rad=torch.ones(2, 6),
                  Q=torch.randn(2, 6, 8), history_yaw_to_t0_rad=torch.randn(2, 4))
    original = copy.deepcopy(output)
    for mode in engine.MODES:
        changed = engine.intervention(output, mode)
        assert changed['Q'] is output['Q']
        assert changed['history_yaw_to_t0_rad'] is output['history_yaw_to_t0_rad']
        assert bool(changed['residual_xy_m'].count_nonzero()) == (mode in ('learned', 'XY_only'))
        assert bool(changed['yaw_delta_rad'].count_nonzero()) == (mode in ('learned', 'yaw_only'))
    for k in output: torch.testing.assert_close(output[k], original[k], rtol=0, atol=0)
    with pytest.raises(ValueError): engine.intervention(output, 'all_geometry_zero')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actual_original_forward_frozen_full6_and_no_future_labels(device, monkeypatch):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 GPU')
    from test_source_scene_stage2 import fixture
    model, runner, row = fixture(device)
    try:
        model.eval().requires_grad_(False)
        with torch.no_grad():
            model.columns.readout[-1].weight.normal_(std=.1)
            model.transport.residual_head.bias.fill_(.15)
            model.transport.yaw_head.bias.fill_(.08)
            model.transport.history_yaw.weight.normal_(std=.05)
        weights = copy.deepcopy(model.state_dict())
        poses = row[1]['future_poses']
        # No future pose/occupancy payload remains in the model's raw inputs.
        row = (row[0], {k: v for k, v in row[1].items() if not k.startswith('future_')}, *row[2:])
        calls = []; direct = engine.forward_window
        def observed(*args, **kwargs):
            fw = direct(*args, **kwargs)
            calls.append({k: fw[k].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                          for k in ('prior', 'logits')})
            return fw
        monkeypatch.setattr(engine, 'forward_window', observed)
        routes, d = engine.predictions(model, runner.provider, row, poses)
        assert len(calls) == 2  # Original + independent original repeat, not one network per intervention.
        assert set(routes) == {m+'/'+r for m in engine.MODES for r in ('transport', 'semantic_refine')}
        for key, value in routes.items():
            assert value.shape == (6, *runner.provider.pcfg.grid.shape_hwd) and value.dtype == np.uint8
            assert value.min() >= 0 and value.max() < 18
        np.testing.assert_array_equal(routes['learned/transport'], calls[0]['prior'])
        np.testing.assert_array_equal(routes['learned/semantic_refine'], calls[0]['logits'])
        for name, key in (('transport', 'prior'), ('semantic_refine', 'logits')):
            assert d['reference_repeat_changed_voxels'][name] == np.count_nonzero(calls[0][key] != calls[1][key])
        assert np.linalg.norm(d['learned_world_displacement']-d['KTA_world_displacement']) > 0
        assert np.all(d['history_yaw_rad'][:, -1] == 0)
        for k, v in weights.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
        assert all(p.grad is None for p in model.parameters())
    finally: runner.close()


def test_integer_full_grid_metrics_match_original_and_regions_partition():
    rng = np.random.default_rng(21)
    p = rng.integers(0, 18, (6, 4, 7, 2), dtype=np.uint8)
    g = rng.integers(0, 18, (3, 4, 7, 2), dtype=np.uint8)
    masks = rng.random(g.shape) > .4
    counts = stats.window_counts({'route': p}, g, masks)['route']
    old = stats.Metrics()
    for i, h in enumerate(stats.HORIZONS): old.update(i, p[h], g[i], masks[i])
    report = stats.finish(counts)
    for name, value in old.compute().items(): assert report[name] == value
    assert counts['full'].sum() == g.size and counts['moving'].sum() == masks.sum()
    assert np.all(counts['full']-counts['moving'] >= 0)
    # Binary FP/FN in the two complementary regions recover full-grid counts.
    for i in range(3):
        a = report['regions']['moving_support'][str(i+1)]
        b = report['regions']['outside_moving_support'][str(i+1)]
        f = counts['full'][i]
        assert a['false_occupied']+b['false_occupied'] == f[17, :17].sum()
        assert a['missed_occupied']+b['missed_occupied'] == f[:17, 17].sum()
    assert stats.confusion(np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)).sum() == 0
    with pytest.raises(ValueError): stats.confusion(np.array([255]), np.array([17]))


def test_edit_counts_distinguish_free_damage_and_wrong_to_wrong():
    g = np.array([[4, 17, 4, 4]])
    before = np.array([[17, 17, 4, 2]])
    after = np.array([[4, 4, 17, 3]])
    preds = {m+'/'+r: np.repeat(x[None], 6, axis=0) for m, x in
             [('learned', after), ('KTA_future', before)] for r in ('transport', 'semantic_refine')}
    masks = np.repeat(np.array([[[True, False, False, False]]]), 3, axis=0)
    report = stats.edits(preds, np.repeat(g[None], 3, axis=0), masks)
    for values in report.values():
        assert values['full/corrected'] == 3 and values['full/damaged'] == 6
        assert values['full/damaged_correct_free'] == 3 and values['full/wrong_to_other_wrong'] == 3
        assert values['moving/corrected'] == 3 and values['outside_moving/damaged'] == 6


def test_post_prediction_box_displacement_avoids_component_box_center_offset(monkeypatch):
    from test_source_scene_stage2 import fixture
    model, runner, row = fixture()
    try:
        count = len(row[0]['source_class_id'])
        # Deliberately offset GT box centers 100m: displacement comparison must
        # not count that center offset as trajectory error.
        def boxes(_, token):
            h = 0 if token == row[0]['t0_token'] else list(row[0]['future_tokens']).index(token)+1
            return [dict(instance_token='i', center=[100+h, 200, 0], rotation=np.eye(3))]
        monkeypatch.setattr(stats, 'boxes_for_sample', boxes)
        assignments = [dict(token='i', status='assigned', purity=.8, fragments_for_instance=count)]*count
        monkeypatch.setattr(stats, 'attribute', lambda *a: assignments)
        times = {row[0]['t0_token']: 0, **{t: (h+1)*500000 for h, t in enumerate(row[0]['future_tokens'])}}
        nusc = SimpleNamespace(get=lambda table, token: {'timestamp': times[token]})
        learned = np.zeros((count, 6, 3)); learned[:, :, 0] = np.arange(1, 7)
        d = dict(residual_xy_m=np.ones((count, 6, 2)), effective_future_yaw_rad=np.zeros((count, 6)),
                 learned_world_displacement=learned, KTA_world_displacement=np.zeros_like(learned))
        rows = stats.box_motion_rows(row, d, nusc, runner.provider.pcfg.grid)
        assert len(rows) == count*6 and all(r['group'] == 'moving' for r in rows)
        assert all(r['learned_error_m'] == 0 for r in rows)
        assert rows[5]['KTA_error_m'] == 6 and rows[0]['GT_speed_mps'] == 2
        s = stats.source_summary(rows)
        assert s['all']['XY_helped'] == count*6 and s['all']['XY_harmed'] == 0
        assert s['groups']['unresolved']['samples'] == 0
    finally: runner.close()


def test_restore_strict_integer_counts_prefix_and_empty_summary():
    c = dict(keys=[['scene', 't0']]); counts = stats.empty_counts()
    saved = dict(protocol=run.PROTOCOL, contract=c, counts={'occ_gt': {'learned/transport': counts}},
                 keys=c['keys'], rows=[{}])
    state = run.restore(json.loads(json.dumps(saved, default=lambda x: x.tolist())), c)
    assert state['counts']['occ_gt']['learned/transport']['full'].dtype.kind in 'iu'
    for name, changed in [('protocol', 'old'), ('keys', [['other', 't0']]), ('rows', [])]:
        broken = copy.deepcopy(saved); broken[name] = changed
        with pytest.raises(RuntimeError): run.restore(broken, c)
    broken = copy.deepcopy(saved); broken['counts']['occ_gt']['learned/transport']['full'] = np.zeros((3, 18, 18), float)
    with pytest.raises(RuntimeError): run.restore(broken, c)
    assert 'windows=0/64' in run.summary(dict(status='stopped', windows=0, metrics={'occ_gt': {}}))


def test_cuda_entry_stop_resume_no_optimizer_and_labels_after_both_forecasts(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 GPU')
    import threading
    from test_source_scene_stage2 import fixture
    model, runner, row = fixture()
    try:
        paths = {k: tmp_path/k for k in ('config', 'dev_cache', 'dev_info')}
        for p in paths.values(): p.write_text('fixture')
        md = tmp_path/'v1.0-trainval'; md.mkdir()
        for name in ('sample', 'sample_annotation', 'instance', 'category', 'sample_data', 'ego_pose'):
            (md/(name+'.json')).write_text('[]')
        archive = tmp_path/'input.npz'; archive.write_bytes(b'fixture')
        records = [dict(row[0], t0_token='dev'+str(i)) for i in range(64)]
        source_contract = dict(total_updates=3, known='TRAIN_OCC_STC_dense_valid', overrides=[],
            model=model.configs(), implementation={}, eval_keys=[list(k) for k in run.base.record_keys(records)],
            data={k: run.base.sha256(p) for k, p in paths.items()})
        ckpt = tmp_path/'epoch2.pt'
        torch.save(dict(protocol=run.SOURCE_PROTOCOL, epoch=2, batch=0, updates=3,
                        contract=source_contract, models={'gt': model.state_dict()}), ckpt)
        before = run.base.sha256(ckpt)
        events = []; stop = threading.Event(); forecast_calls = []
        raw = {k: row[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
        class Inputs:
            source = SimpleNamespace(nusc=None)
            def __init__(self, *a, **k): pass
            def path(self, *a): return archive
            def stat(self, p): return p.stat().st_size, p.stat().st_mtime_ns
            def history(self, w, kind):
                events.append('history'); return row[0], dict(raw, history_observed=np.zeros_like(raw['history_observed'])), np.arange(4)*.5
            def poses(self, w): return row[1]['future_poses']
            def targets(self, w, indices):
                assert events[-2:] == ['history', 'forecast'] and events.count('forecast') == 2
                events.clear(); events.append('labels')
                return np.repeat(raw['history_occ'][-1][None], 3, axis=0)
            def moving(self, *a):
                assert events == ['labels']; events.clear()
                return np.ones((3, *raw['history_occ'].shape[1:]), bool)
        class Predictor:
            def __init__(self, body, *a, **k): self.provider = runner.provider
            def prepare(self, record, historical, **kw):
                assert historical['history_observed'].all(); return row
            def close(self): pass
        def predict(body, provider, actual, poses):
            assert events[-1] == 'history'; events.append('forecast'); forecast_calls.append(1)
            if len(forecast_calls) == 2: stop.set()
            arr = np.repeat(raw['history_occ'][-1][None], 6, axis=0)
            return ({m+'/'+r: arr for m in engine.MODES for r in ('transport', 'semantic_refine')},
                dict(history_yaw_rad=np.zeros((1, 4)), history_valid=np.ones((1, 4), bool),
                     reference_repeat_changed_voxels=dict(transport=0, semantic_refine=0)))
        monkeypatch.setattr(run, 'NuscInputs', Inputs)
        monkeypatch.setattr(run, 'DirectPredictor', Predictor)
        monkeypatch.setattr(run, 'predictions', predict)
        monkeypatch.setattr(run, 'box_motion_rows', lambda *a: [])
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: {})
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda *a: runner.provider.pcfg)
        monkeypatch.setattr(run.base, 'load_cache', lambda *a: ({}, records))
        monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **k: pytest.fail('optimizer forbidden'))
        argv = ['--checkpoint', str(ckpt), '--out', str(tmp_path/'audit'), '--dataroot', str(tmp_path), '--stc-root', str(tmp_path)]
        for k, p in paths.items(): argv += ['--'+k.replace('_', '-'), str(p)]
        run.main(argv, event=stop)
        state = json.loads((tmp_path/'audit/state.json').read_text())
        assert len(state['keys']) == 1 and len(forecast_calls) == 2
        assert json.loads((tmp_path/'audit/results.json').read_text())['status'] == 'stopped_at_complete_window'
        run.main(argv+['--resume'])
        final = json.loads((tmp_path/'audit/results.json').read_text())
        assert final['windows'] == 64 and final['status'] == 'complete' and len(forecast_calls) == 128
        for routes in final['metrics'].values(): assert len(routes) == 8
        assert run.base.sha256(ckpt) == before
        run.main(argv+['--resume'])
        assert len(forecast_calls) == 128  # Completed state does not forecast again.
    finally: runner.close()
