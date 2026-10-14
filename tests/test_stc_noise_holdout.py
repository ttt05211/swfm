"""Synthetic-domain read-only population, shared realization and integer resume."""
import copy
from dataclasses import asdict
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_noise_holdout import run
from tools.stc_noise_adaptation import augmentation as aug
from tools.stc_noise_adaptation import data
from test_stc_noise_adaptation import empirical_value, profile_files
from test_perception_robustness import real_fixture


def test_selection_excludes_whole_consumed_scenes_not_just_keys(monkeypatch):
    records = [dict(scene_name='SEEN', t0_token=str(i), features=[0]) for i in range(600)]
    records += [dict(scene_name='UNSEEN'+str(i%3), t0_token='hold'+str(i), features=[0]) for i in range(90)]
    contract = dict(train_keys=run.legacy.record_keys(records[:600]), seed=21,
                    window_batch=4, source_budget=128, total_updates=500)
    monkeypatch.setattr(run.legacy, 'epoch_groups', lambda records, *a: [[r] for r in records])
    selected, audit = run.select_holdout(records, contract)
    assert len(selected) == 64 and len({r['t0_token'] for r in selected}) == 64
    assert {r['scene_name'] for r in selected}.isdisjoint({'SEEN'})
    assert audit['consumed_windows'] == 500 and audit['eligible_windows'] == 90
    assert audit['body_pretraining_saw_these_TRAIN_scenes']
    with pytest.raises(RuntimeError, match='no window-only fallback'):
        run.select_holdout(records[:650], contract)


def test_confusion_scores_standard_full_grid_and_invalid_label_ignore():
    target = np.array([11, 4, 17, 17, 255], np.uint8).reshape(1, 5)
    pred = np.array([11, 17, 4, 17, 4], np.uint8).reshape(1, 5)
    c = run.confusion(np.repeat(pred[None], 6, axis=0), np.repeat(target[None], 6, axis=0))
    m = run.scores(c)
    assert m['mIoU'] == 50. and m['IoU'] == pytest.approx(100/3)
    assert c.sum() == 12
    with pytest.raises(ValueError): run.scores(c.astype(float))
    with pytest.raises(ValueError): run.scores(-c)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_three_models_same_simulator_input_no_steps_resume(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    m, _, p, row = real_fixture(device)
    models = {arm: copy.deepcopy(m).eval().requires_grad_(False) for arm in run.ARMS}
    originals = {arm: copy.deepcopy(model.state_dict()) for arm, model in models.items()}
    noise = aug.NoiseModel(empirical_value())
    # Pin a selected and an unselected RNG coin, without modifying the 50% recipe.
    records = []
    for selected in (True, False):
        rec = dict(row[0])
        rec['t0_token'] = next(str(i) for i in range(100) if (aug.rng_for(dict(rec, t0_token=str(i)), 21).random() < .5) == selected)
        rec['history_tokens'] = [*rec['history_tokens'][:-1], rec['t0_token']]
        records.append(rec)
    history = copy.deepcopy(row[1]['history_occ'])
    calls = []
    forward = run.gt.forward_window
    def checked(model, provider, value):
        assert not torch.is_grad_enabled()
        # Forecast gets no future occupancy access; targets are retained only for later score.
        raw = dict(value[1], future_gt_occ=object())
        calls.append((id(model), value[0]['t0_token'], raw['history_occ'].tobytes()))
        return forward(model, provider, (value[0], raw, value[2], value[3]))
    monkeypatch.setattr(run.gt, 'forward_window', checked)
    rows = [(rec, row[1], row[2], row[3]) for rec in records]
    states = []
    event = threading.Event()
    def save(state):
        states.append(copy.deepcopy(state))
        if len(state['rows']) == 1: event.set()
    part = run.evaluate(models, p, iter(rows), noise, 21, save=save, event=event)
    assert len(part['rows']) == 1 and part['rows'][0]['audit']['selected']
    before = copy.deepcopy(part)
    resumed = run.evaluate(models, p, iter(rows), noise, 21, saved=part)
    assert part == before and len(resumed['rows']) == 2
    assert not resumed['rows'][1]['audit']['selected']
    grouped = {}
    for model, key, sem in calls: grouped.setdefault(key, []).append(sem)
    # Three model pairs receive identical clean and corrupted realizations.
    for values in grouped.values():
        assert len(values) == 6 and values[0] == values[2] == values[4] and values[1] == values[3] == values[5]
    result = run.report(resumed)
    assert result['all']['windows'] == 2 and result['corrupted_only']['windows'] == 1
    assert set(result['all']['metrics']) == set(run.ARMS)
    np.testing.assert_array_equal(history, row[1]['history_occ'])
    for arm, model in models.items():
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, originals[arm][k], atol=0, rtol=0)
        assert all(v.grad is None for v in model.parameters())
    bad = copy.deepcopy(resumed); bad['rows'][0]['counts']['clean']['clean_dense']['transport'][0][0][0] = -1
    with pytest.raises(ValueError): run.evaluate(models, p, iter(rows), noise, 21, saved=bad)
    bad = copy.deepcopy(resumed); bad['rows'][0]['key'][1] = 'bad'
    with pytest.raises(RuntimeError, match='prefix'): run.evaluate(models, p, iter(rows), noise, 21, saved=bad)


def test_actual_cli_reads_only_original_three_checkpoints(tmp_path, monkeypatch):
    m, _, provider, row = real_fixture()
    parent, full = tmp_path/'paired', tmp_path/'stc_full_unfreeze500_fixture'
    parent.mkdir(); full.mkdir()
    files = {name: tmp_path/name for name in ('config', 'train_cache', 'dev_cache', 'population_manifest',
                                             'base_checkpoint', 'train_info', 'dev_info')}
    for f in files.values(): f.write_text('fixture')
    torch.save(dict(model_config=asdict(m.transport.config)), files['base_checkpoint'])
    records = [dict(row[0], scene_name='SEEN', t0_token=str(i)) for i in range(600)]
    records += [dict(row[0], scene_name='HELD', t0_token='held'+str(i),
                    history_tokens=[*row[0]['history_tokens'][:-1], 'held'+str(i)]) for i in range(70)]
    datahash = {k: run.legacy.sha256(f) for k, f in files.items()}
    fields = dict({k: str(f) for k, f in files.items()}, device='cuda', cpu_workers=1, frame_cache_mib=0,
        descriptor_ram_mib=0, geometry_ram_mib=0, dataroot=str(tmp_path), override=[], seed=21,
        ccr_history_cache=str(tmp_path), ccr_val_history_cache=str(tmp_path))
    contract = dict(protocol='frozen', train_keys=[list(k) for k in run.legacy.record_keys(records[:600])], seed=21,
        window_batch=4, source_budget=128, total_updates=500, data=datahash, source_GT_sha256='GT')
    source = dict(contract={'model': m.configs()}, models={'gt': m.state_dict()})
    (parent/'empirical/frozen').mkdir(parents=True)
    torch.save(source, parent/'empirical/frozen/GTepoch2.pt')
    contract['source_GT_sha256'] = run.legacy.sha256(parent/'empirical/frozen/GTepoch2.pt')
    profile_files(parent/'profile', contract['source_GT_sha256'])
    contract['noise_identity'] = run.load_model(parent/'profile/noise_model.json')[1]
    workflow = dict(args=fields, data=datahash)
    (parent/'workflow.json').write_text(json.dumps(workflow))
    refs = {a: dict(status='complete', updates=500, contract=dict(contract, augmentation=a),
        evaluation={'variants': {'gt': {'stc_gt_semantic_refine': dict(mIoU=18., IoU=28.)}}}) for a in ('clean', 'empirical')}
    for a in refs:
        (parent/a/'train').mkdir(parents=True, exist_ok=True)
        (parent/a/'results.json').write_text(json.dumps(refs[a]))
        torch.save(dict(updates=500, contract=refs[a]['contract'], models={'gt': m.state_dict()}), parent/a/'train/last.pt')
    fr = dict(status='complete', updates=500, contract=dict(contract, protocol=run.FULL_PROTOCOL),
        evaluation=refs['clean']['evaluation'])
    (full/'train').mkdir(); (full/'results.json').write_text(json.dumps(fr))
    torch.save(dict(updates=500, contract=fr['contract'], models={'gt': m.state_dict()}), full/'train/last.pt')
    (full/'reference.json').write_text(json.dumps({'reference': {str(parent/'workflow.json'): run.legacy.sha256(parent/'workflow.json')}}))
    assert run.choose_full(parent) == full
    before = {str(f): f.read_bytes() for d in (parent, full) for f in d.rglob('*') if f.is_file()}
    monkeypatch.setattr(run, 'references', lambda _: (parent, workflow, refs))
    monkeypatch.setattr(run.legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(run.legacy, 'load_cache', lambda _: ({}, records))
    monkeypatch.setattr(run.legacy, 'epoch_groups', lambda records, *a: [[r] for r in records])
    monkeypatch.setattr(run.legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(run.legacy, 'make_prepare_config', lambda _: provider.pcfg)
    def make_provider(*args): provider.joint = args[-2]; return provider
    monkeypatch.setattr(run.legacy, 'PilotProvider', make_provider)
    monkeypatch.setattr(run.legacy, 'NuScenesWindowSource', lambda *a, **k: object())
    monkeypatch.setattr(run.legacy, 'CachedColumnSource', lambda src, *a, **kw: src)
    class Disposable:
        kernels = pool = None
        def __init__(self, *a): pass
        def close(self): pass
    monkeypatch.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
    monkeypatch.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
    def caches(p, *a):
        p.ccr_history_cache = p.ccr_val_history_cache = Disposable()
    monkeypatch.setattr(run.legacy, 'attach_caches', caches)
    provider.load_raw_columns = lambda *a, **k: row[1]
    monkeypatch.setattr(run.gt, 'history_inputs', lambda *a: row[2])
    def evaluate(models, provider, rows, noise, seed, **kw):
        selected = list(rows)
        assert len(selected) == 64 and all(r[0]['scene_name'] == 'HELD' for r in selected)
        assert all(r[1]['history_observed'].all() for r in selected)
        for model in models.values():
            for k, v in model.state_dict().items(): torch.testing.assert_close(v, m.state_dict()[k], rtol=0, atol=0)
        c = np.zeros((3, 18, 18), np.int64); c[:, 11, 11] = 3
        result = dict(protocol=run.PROTOCOL, rows=[dict(key=[r[0]['scene_name'], r[0]['t0_token']],
            audit={'selected': True}, counts={arm: {inp: {rt: c.tolist() for rt in run.ROUTES}
                                                   for inp in run.INPUTS} for arm in run.ARMS}) for r in selected])
        kw['save'](result); return result
    monkeypatch.setattr(run, 'evaluate', evaluate)
    out = tmp_path/'evaluation'
    args = ['--reference-dir', str(parent), '--out-dir', str(out)]
    assert run.main(args) == 0
    result = json.loads((out/'results.json').read_text())
    assert result['status'] == 'complete' and result['model_weights_unchanged']
    assert result['subsets']['corrupted_only']['windows'] == 64
    assert run.main(args+['--resume']) == 0
    assert before == {str(f): f.read_bytes() for d in (parent, full) for f in d.rglob('*') if f.is_file()}
    with pytest.raises(RuntimeError, match='NEW output'): run.main(args)
