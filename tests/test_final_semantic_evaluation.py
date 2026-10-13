"""Frozen new-body evaluation adapters; synthetic tests are NOT dataset scores."""
import copy
import threading
from types import SimpleNamespace
from dataclasses import asdict
import numpy as np
import pytest
import torch
from test_full_gt_training import fixture
from tools.final_semantic_evaluation import metrics, engine, predictor, sources
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.geniedrive_eval_alignment import compatibility_metrics
from tools.scene_ego_head.model import SceneEgoConfig, SceneEgoHead
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


def test_metric_matches_current_formal_metrics_and_zero_classes():
    rng = np.random.default_rng(21); counts = metrics.empty(3); old = Metrics()
    for _ in range(2):
        for h in range(3):
            p = rng.integers(0, 18, (8, 8, 4), dtype=np.uint8)
            g = rng.integers(0, 18, p.shape, dtype=np.uint8); mask = rng.random(p.shape) > .2
            metrics.add(counts, h, p, g, mask); old.update(h, p, g, mask)
    report, reference = metrics.report(counts, (1., 2., 3.)), old.compute()
    for key, value in report['average'].items(): assert value == pytest.approx(reference[key], abs=1e-12)
    for h, row in report['per_horizon'].items():
        for key in ('IoU', 'mIoU', 'MovingMicro', 'MovingMacro'):
            assert row[key] == pytest.approx(reference['per_horizon'][h][key], abs=1e-12)
    with pytest.raises(ValueError): metrics.add(counts, 0, np.array([255]), np.array([17]))


def test_new_predictor_real_small_network_forward_without_future_inputs():
    model, _, p, old = fixture(); model.eval().requires_grad_(False)
    p.pcfg.free_label = 17
    runner = predictor.Predictor(model, p.pcfg, 'cpu', workers=1)
    try:
        raw = {k: old[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
        row = runner.prepare(old[0], raw)
        assert set(row[1]) == {*raw, '_column_causal_preparation'}
        assert not any(k.startswith('future') for k in row[1])
        assert row[3][1] is None and 'memory_reads' in row[3][0]
        out = runner.forward(row, poses=old[1]['future_poses'])
        dense = runner.dense(out)
        assert dense['semantic_refine'].shape == (6, *p.pcfg.grid.shape_hwd)
        np.testing.assert_array_equal(dense['transport'], dense['semantic_refine'])  # zero residual initialization
        with pytest.raises(RuntimeError, match='historical'):
            runner.prepare(old[0], {**raw, 'future_poses': object()})
        with pytest.raises(RuntimeError, match='occupancy'):
            runner.prepare(old[0], {**raw, 'future_gt_occ': object()})
    finally: runner.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_new_adapter_same_pose_gate_and_internal_head(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model, _, p, old = fixture(); model.to(device).eval().requires_grad_(False)
    with torch.no_grad(): model.columns.readout[-1].weight.normal_(std=.01)
    head = SceneEgoHead(SceneEgoConfig(feature_width=16, width=16, heads=4, pool_side=2)).to(device).eval()
    runner = predictor.Predictor(model, p.pcfg, device, workers=1, head=head)
    try:
        raw = {k: copy.deepcopy(old[1][k]) for k in ('history_occ', 'history_observed', 'history_poses')}
        raw['history_occ'][:, 2:4, 2:4, 1:3] = 4  # actual >=6 voxel dynamic component
        row = runner.prepare(old[0], raw)
        assert len(row[0]['source_class_id']) > 0
        rec = {**row[0], 'features': torch.zeros(len(row[0]['source_class_id']), 1), 'sample_id': 'synthetic'}
        source = SimpleNamespace(history=lambda _: (rec, raw, None), poses=lambda _: old[1]['future_poses'])
        report = predictor.audit_gt_entrance(runner, source, rec)
        assert report['pass_gate'], report
        fw = runner.forward(row, commands=np.full(6, 2, np.int64))
        assert fw['ego']['se2'].shape == (1, 6, 3)
        assert runner.dense(fw)['semantic_refine'].shape == (6, *p.pcfg.grid.shape_hwd)
    finally: runner.close()


def test_real_head10_only_bound_to_body(tmp_path):
    model, _, _, _ = fixture(); config = SceneEgoConfig(feature_width=16, width=16, heads=4, pool_side=2)
    head = SceneEgoHead(config)
    contract = dict(schedule=dict(epochs=10, batch_size=64), source_train_keys=[['a', 'b']],
        stage=dict(source_sha256='bodySHA'), body_state_sha256=tensor_state_fingerprint(model.state_dict()), head=asdict(config))
    from tools.scene_ego_head.frozen_gt.training import PROTOCOL
    saved = dict(protocol=PROTOCOL, contract=contract, epochs=10, final_epoch_only=True, head=head.state_dict(), updates=10,
                 state_sha256=tensor_state_fingerprint(head.state_dict()))
    path = tmp_path/'head.pt'; torch.save(saved, path)
    loaded = predictor.load_head10(path, model, 'bodySHA', 'cpu')
    assert not any(p.requires_grad for p in loaded.parameters())
    with pytest.raises(RuntimeError, match='head10'): predictor.load_head10(path, model, 'otherSHA', 'cpu')
    saved['contract']['schedule']['epochs'] = 100; torch.save(saved, path)
    with pytest.raises(RuntimeError, match='head10'): predictor.load_head10(path, model, 'bodySHA', 'cpu')


class FakePredictor:
    def __init__(self, source):
        self.source = source; self.rows = []; self.calls = 0
        self.provider = SimpleNamespace(pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=(2, 2, 2))), workers=1)
    def prepare(self, record, raw, **kw):
        assert not any(k.startswith('future') for k in raw)
        self.rows.append(copy.deepcopy(raw)); return record, raw
    def forward(self, row, *, poses=None, commands=None):
        self.calls += 1; self.source.forecast_count += 1
        values = np.stack([np.full((2, 2, 2), (i+1 if self.calls % 2 else i+7), np.uint8) for i in range(6)])
        return dict(semantic_refine=values, transport=values,
                    ego=dict(se2=torch.zeros(1, 6, 3)))
    def dense(self, out): return {k: out[k] for k in ('semantic_refine', 'transport')}


class FakeSource:
    def __init__(self, forecasts_before_labels=1):
        self.forecast_count = 0; self.required = forecasts_before_labels
        self.source = SimpleNamespace(nusc=SimpleNamespace(get=lambda _, t: dict(timestamp=int(t[1:])*500000)))
    def history(self, job, kind='occ'):
        return dict(scene_name=job.scene_name, t0_token=job.t0_token, history_tokens=job.history_tokens,
            future_tokens=job.future_tokens[:6]), dict(history_occ=np.full((4, 2, 2, 2), 17, np.uint8),
            history_observed=np.ones((4, 2, 2, 2), bool), history_poses=[np.eye(4)]*4), np.arange(4)*.5
    def poses(self, _): return [np.eye(4)]*12
    def commands(self, _): return np.full(6, 2, np.int64)
    def targets(self, _, indices):
        assert self.forecast_count >= self.required
        return [np.full((2, 2, 2), 4, np.uint8)]*len(indices)
    def moving(self, *a):
        assert self.forecast_count >= self.required
        return [np.ones((2, 2, 2), bool)]*len(a[1])


def jobs(count=1):
    return [SimpleNamespace(scene_name='VAL', t0_token='a'+str(i), history_tokens=('h0', 'h1', 'h2', 'h3'),
                            future_tokens=tuple('f'+str(j) for j in range(12))) for i in range(count)]


@pytest.mark.parametrize('task,required', [('gt', 1), ('stc', 2), ('long6', 2)])
def test_engine_future_labels_after_all_routes_and_strict_rollout(task, required, monkeypatch):
    src = FakeSource(required); pred = FakePredictor(src); pop = jobs(); keys = [['VAL', 'a0']]
    inherited = np.zeros((4, 2, 2, 2), bool)
    monkeypatch.setattr(engine, 'inherited_observation_masks', lambda *a: inherited.copy())
    contract = dict(population=dict(alignment='pinned'))
    result = engine.evaluate(pred, src, pop, task, keys, contract)
    assert result['windows'] == 1 and result['scientific_updates'] == 0
    if task == 'long6':
        np.testing.assert_array_equal(pred.rows[1]['history_occ'][:, 0, 0, 0], [3, 4, 5, 6])
        np.testing.assert_array_equal(pred.rows[1]['history_observed'], inherited)
        assert list(result['reports']['semantic_refine_rollout']['per_horizon']) == [str(float(i)) for i in range(1, 7)]
        assert result['rollout']['static_carry'] is False
        assert 'public_code_compatibility' in result
    if task == 'stc': assert set(result['reports']) == {'stc_gt/transport', 'stc_gt/semantic_refine', 'stc_pred/transport', 'stc_pred/semantic_refine'}


def test_resume_complete_window_and_invalid_counts():
    pop = jobs(2); keys = [['VAL', 'a0'], ['VAL', 'a1']]; event = threading.Event(); snapshots = []
    def save(s): snapshots.append(copy.deepcopy(s)); event.set()
    src = FakeSource(); pred = FakePredictor(src)
    with pytest.raises(InterruptedError):
        engine.evaluate(pred, src, pop, 'gt', keys, {}, save=save, stop_event=event, checkpoint_every=1)
    assert snapshots[-1]['keys'] == keys[:1]
    states = []; src = FakeSource()
    resumed = engine.evaluate(FakePredictor(src), src, pop, 'gt', keys, {}, saved=snapshots[-1], save=states.append)
    assert resumed['windows'] == 2
    # Fake predictor resets call index; gt labels here use per-window same unchanging prediction.
    assert all(np.asarray(v['confusion']).sum() == 48 for v in states[-1]['counts'].values())
    bad = copy.deepcopy(snapshots[-1]); bad['counts']['gt/transport']['confusion'][0][0][0] += 1
    with pytest.raises(RuntimeError, match='totals'):
        engine.evaluate(pred, src, pop, 'gt', keys, {}, saved=bad)
    bad = copy.deepcopy(snapshots[-1]); bad['keys'][0][1] = 'wrong'
    with pytest.raises(RuntimeError, match='identity'):
        engine.evaluate(pred, src, pop, 'gt', keys, {}, saved=bad)


def test_stc_history_reads_no_sensor_mask_and_future_target_reads_no_mask(tmp_path):
    shape = (200, 200, 16); frame = tmp_path/'s'/'t'/'labels.npz'; frame.parent.mkdir(parents=True)
    np.savez_compressed(frame, semantics=np.full(shape, 4, np.uint8))  # no GT mask fields
    source = sources.NuscInputs.__new__(sources.NuscInputs)
    source.stc = tmp_path; source.source = SimpleNamespace(dataroot=str(tmp_path.parent))
    from collections import OrderedDict
    source.cache = OrderedDict(); source.bytes = 0; source.limit = 0; source.inventory = {}
    sem, seen = source.frame('stc', 's', 't')
    assert np.all(sem == 4) and seen.all()
    source.path = lambda *a: frame
    assert np.all(source.targets(SimpleNamespace(scene_name='s', future_tokens=['t']), [0])[0] == 4)


@pytest.mark.parametrize('task', ['waymo2', 'waymo10'])
def test_waymo_literal_reports_and_no_future_in_history(task):
    source = FakeSource()
    def inputs(job):
        rec, raw, _ = source.history(job)
        return rec, {**raw, 'future_poses': source.poses(job)[:6]}
    source.prediction_inputs = inputs; source.metric_targets = lambda _: source.targets(None, [1, 3, 5])
    result = engine.evaluate(FakePredictor(source), source, jobs(), task, [['VAL', 'a0']], {})
    rows = result['reports']['gt/semantic_refine']['horizons']
    if task == 'waymo10':
        assert [r['nominal_seconds'] for r in rows.values()] == [.2, .4, .6]
        assert list(rows) != ['1', '2', '3']
    else: assert list(rows) == ['1', '2', '3']


def test_long_compatibility_same_integer_counts():
    counts = metrics.empty(6)
    for h in range(6): metrics.add(counts, h, np.array([0, 1, 17]), np.array([0, 2, 17]))
    v = counts['confusion']; tp = np.diagonal(v, axis1=1, axis2=2)
    legacy = compatibility_metrics(dict(sem_inter=tp[:, :17], sem_union=(v.sum(1)+v.sum(2)-tp)[:, :17],
        occ_inter=v[:, :17, :17].sum((1, 2)), occ_union=v.sum((1, 2))-v[:, 17, 17]))
    assert metrics.report(counts, range(1, 7), moving=False)['average']['mIoU'] == pytest.approx(100/3)
    assert legacy['per_horizon']['1.0']['mIoU'] == 100.
