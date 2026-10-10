from copy import deepcopy
import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_surface_ego_head import config, feature_row
from tools.ego_diagnostics import surface_train_replay as replay
from real_motion.ego_trajectory_head import HistoryEgoTrajectoryHead
from tools.real_motion.ego_trajectory_common import stack_features, fingerprint, digest_file
from tools.real_motion.surface_ego_ablation_common import bank_reports, TRAIN_NAMES


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def data(n=7):
    return [dict(key=['scene-'+str(i % 3), str(i)], features=feature_row(),
        commands=np.full(6, i % 3, np.int64),
        target=np.c_[np.arange(1, 7)*.2, np.zeros(6), np.arange(1, 7)*.01].astype(np.float32))
        for i in range(n)]


def models():
    # Nonzero learned readout + visible tokens: a prior-only parity is insufficient.
    head = HistoryEgoTrajectoryHead(config()).eval().requires_grad_(False)
    with torch.no_grad():
        head.readout.weight.normal_(std=.07); head.readout.bias.normal_(std=.02)
    result = {k: deepcopy(head) for k in ('old320', *TRAIN_NAMES)}
    with torch.no_grad():
        result['B_geometry'].readout.bias.add_(.2)
    return result


def recorded(rows, heads):
    return bank_reports(heads, stack_features([r['features'] for r in rows], 'cpu'),
        torch.from_numpy(np.stack([r['target'] for r in rows])),
        torch.from_numpy(np.stack([r['commands'] for r in rows])))


def materializer(rows):
    def build(i, _):
        r = rows[i]
        return dict(train_features=deepcopy(r['features']), eval_features=deepcopy(r['features']),
            target_train=r['target'].copy(), target_eval=r['target'].copy(),
            commands=r['commands'].copy(), inputs={'history_occ': replay.array_difference([17], [17])})
    return build


def test_selection_command_and_scene_balance_no_error_or_model_inputs():
    rows = data(27)
    # A deliberately heavily imbalanced population still exposes all commands.
    for r in rows[9:]: r['commands'][:] = 2
    selected = replay.select_indices(rows, 12)
    assert selected == replay.select_indices(rows, 12)
    assert len(set(selected)) == 12
    assert [int(rows[i]['commands'][-1]) for i in selected[:9]] == [0, 1, 2]*3
    assert replay.select_indices(rows, len(rows)) and len(set(replay.select_indices(rows, 27))) == 27
    with pytest.raises(ValueError): replay.select_indices(rows, 28)
    rows[0]['commands'][1] = 3
    with pytest.raises(ValueError, match='commands'): replay.select_indices(rows, 12)


def test_feature_bytes_numeric_budget_and_invalid_shapes_values():
    a = feature_row(); b = deepcopy(a); b['objects'][0, 0] += 2e-7
    r = replay.feature_difference(a, b)
    assert not r['objects']['exact'] and r['objects']['numerical']
    b['object_valid'][0] = True
    assert not replay.feature_difference(a, b)['object_valid']['numerical']
    b['objects'][0, 0] += .01
    assert not replay.feature_difference(a, b)['objects']['numerical']
    assert not replay.array_difference(np.zeros(2), np.zeros(3))['numerical']
    assert not replay.array_difference(np.zeros(2), np.full(2, np.nan))['numerical']
    assert not replay.array_difference(np.zeros(2, np.float32), np.zeros(2, np.float64))['numerical']
    with pytest.raises(ValueError): replay.feature_difference(a, {**a, 'future': torch.zeros(1)})
    p = np.zeros((6, 3)); q = p.copy(); q[:, 2] = 2*np.pi
    assert replay.trajectory_difference(p, q)['pass_gate']
    q[:, 0] = .01
    assert not replay.trajectory_difference(p, q)['pass_gate']


def test_real_nonzero_heads_batch64_single_and_modes_no_weights_or_features_changed():
    rows = data(67); heads = models()
    for r in rows:
        r['features']['object_valid'][:2] = True; r['features']['surface_valid'][:1] = True
    before = {k: replay.state_digest(m) for k, m in heads.items()}
    features = [r['features'] for r in rows]; commands = np.stack([r['commands'] for r in rows])
    p = replay.predictions(heads, features, commands)
    for variant in (replay.predictions(heads, features, commands, batch_size=1),
                    replay.predictions(heads, features, commands, training_mode=True)):
        for name in p: assert replay.trajectory_difference(p[name], variant[name])['pass_gate']
    assert not np.array_equal(p['old320'], p['prior'])
    assert before == {k: replay.state_digest(m) for k, m in heads.items()}
    assert all(not m.training and all(p.grad is None for p in m.parameters()) for m in heads.values())


def test_train_replay_complete_prefix_resume_contract_and_source_unchanged():
    rows = data(); heads = models(); reports = recorded(rows, heads); contract = {'science': 'read only'}
    before = deepcopy(rows); source_digest = {k: replay.state_digest(m) for k, m in heads.items()}
    whole = replay.audit(rows, list(range(7)), heads, materializer(rows), reports, contract)
    assert whole['pass_gate'] and all(whole['gates'].values())
    saved = []; event = Event()
    partial = replay.audit(rows, list(range(7)), heads, materializer(rows), reports, contract,
        stop_event=event, progress=lambda _: event.set(), save=saved.append)
    assert partial['state']['status'] == 'stopped' and partial['state']['completed_windows'] == 1
    assert saved[0]['completed_windows'] == 0 and saved[0]['window_audits'] == []
    checked = dict(saved[-1]); sha = checked.pop('fingerprint')
    assert sha == fingerprint(checked)
    calls = []
    def again(i, c):
        calls.append(i); return materializer(rows)(i, c)
    resumed = replay.audit(rows, list(range(7)), heads, again, reports, contract, saved=saved[-1])
    assert calls == list(range(1, 7)) and resumed['pass_gate']
    assert resumed['subset'] == whole['subset'] and resumed['full_TRAIN'] == whole['full_TRAIN']
    assert replay.summary_text(resumed).count('exact=7/7') == len(replay.FEATURE_FIELDS)*3
    with pytest.raises(RuntimeError, match='fingerprint'):
        replay.audit(rows, list(range(7)), heads, again, reports, {'science': 'changed'}, saved=saved[-1])
    corrupt = deepcopy(saved[-1]); corrupt['window_audits'][0]['index'] = 3
    value = {k: v for k, v in corrupt.items() if k != 'fingerprint'}; corrupt['fingerprint'] = fingerprint(value)
    with pytest.raises(RuntimeError, match='prefix'):
        replay.audit(rows, list(range(7)), heads, again, reports, contract, saved=corrupt)
    for a, b in zip(before, rows):
        assert a['key'] == b['key']; np.testing.assert_array_equal(a['target'], b['target'])
        for k in a['features']: torch.testing.assert_close(a['features'][k], b['features'][k], rtol=0, atol=0)
    assert source_digest == {k: replay.state_digest(m) for k, m in heads.items()}


@pytest.mark.parametrize('bad', ['feature', 'mask', 'label', 'command', 'reader', 'nonfinite', 'report'])
def test_discrepancies_are_reported_without_training_or_automatic_repair(bad):
    rows = data(2); heads = models(); reports = recorded(rows, heads)
    if bad == 'report': reports['B_geometry']['FDE_3s_m'] += .1
    def live(i, c):
        r = materializer(rows)(i, c)
        if bad == 'feature': r['train_features']['ego_history'][-1, 5] += .1
        if bad == 'mask': r['eval_features']['surface_valid'][0] = True
        if bad == 'label': r['target_train'][0, 0] += .1
        if bad == 'command': r['commands'][0] = (r['commands'][0]+1) % 3
        if bad == 'reader': r['inputs']['history_occ'] = replay.array_difference([17], [11])
        if bad == 'nonfinite': r['eval_features']['objects'][0, 0] = float('nan')
        return r
    result = replay.audit(rows, [0, 1], heads, live, reports, {'test': bad})
    assert result['state']['status'] == 'complete' and not result['pass_gate']
    assert result['route'] == 'interface_discrepancy_do_not_expand_training'
    if bad == 'nonfinite': assert result['state']['window_audits'][0]['predictions']['live_eval'] is None
    if bad == 'report': assert not result['gates']['full_TRAIN_report_reproduced']
    else: assert 'first_mismatch_windows=[{"index": 0' in replay.summary_text(result)


def test_middle_window_exception_never_publishes_partial_window():
    rows = data(2); heads = models(); saved = []
    def fail(*_): raise RuntimeError('middle of fresh extraction')
    with pytest.raises(RuntimeError, match='middle'):
        replay.audit(rows, [0, 1], heads, fail, recorded(rows, heads), {}, save=saved.append)
    assert len(saved) == 1 and saved[0]['completed_windows'] == 0 and saved[0]['window_audits'] == []


@pytest.mark.parametrize('bad', ['batch', 'mode'])
def test_batch_or_mode_dependency_is_detected(bad):
    class Unstable(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config = config(); self.witness = torch.nn.Parameter(torch.zeros(()))
        def forward(self, bank, commands):
            out = replay.historical_prior(bank).clone()
            if len(out) > 1 if bad == 'batch' else self.training: out[..., 0] += .1
            return {'se2': out}
    rows = data(2); heads = {name: Unstable().eval() for name in ('old320', *TRAIN_NAMES)}
    result = replay.audit(rows, [0, 1], heads, materializer(rows), recorded(rows, heads), {})
    assert not result['gates']['head_batch64_vs_single' if bad == 'batch' else 'head_train_mode_vs_eval']


def fake_source(s):
    def get(kind, token):
        if kind == 'scene': return {'name': s.windows[0].scene}
        r = s.catalog[token]
        return dict(prev=r.prev, next=r.next, timestamp=r.timestamp, scene_token='scene-id')
    path = lambda scene, token: s.root/'gts'/scene/token/'labels.npz'
    def load(scene, token):
        with np.load(path(scene, token), allow_pickle=False) as z:
            return np.array(z['semantics']), np.array(z['mask_lidar'], bool)
    return SimpleNamespace(dataroot=str(s.root), nusc=SimpleNamespace(get=get), _label_path=path,
        pose=lambda t: s.catalog[t].pose, load_occ3d=load)


def test_two_history_readers_supervision_after_extraction_no_future_occupancy(tmp_path, monkeypatch):
    from test_stc_camera_protocol import fixture
    s = fixture(tmp_path); source = fake_source(s); w = s.windows[0]
    rows = [{'key': [w.scene, w.t0]}]; records = replay.derive_records(source.nusc, rows)
    calls = []; npz_paths = []; feature = feature_row(); actual_load = np.load
    def load(path, *args, **kw):
        assert path in [source._label_path(w.scene, t) for t in w.history]
        npz_paths.append(path); return actual_load(path, *args, **kw)
    def extract(provider, rec, raw, cfg, timestamps_s):
        assert set(raw) == {'history_occ', 'history_observed', 'history_poses'}
        assert rec == records[0] and len(timestamps_s) == 4
        calls.append('features'); return deepcopy(feature)
    old_pose = source.pose
    def pose(t):
        if t in w.future: assert calls == ['features']*2
        return old_pose(t)
    source.pose = pose
    def commands(*_):
        assert calls == ['features']*2; return np.full(6, 2, np.int64)
    monkeypatch.setattr(np, 'load', load); monkeypatch.setattr(replay, 'extract_history_features', extract)
    monkeypatch.setattr(replay, 'navigation_commands', commands)
    r = replay.TrainMaterializer(source, s.catalog, records, None, s.shape)(0, config())
    assert len(npz_paths) == 8 and calls == ['features']*2
    assert all(v['exact'] for v in r['inputs'].values())
    np.testing.assert_array_equal(r['target_train'], r['target_eval'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actual_frozen_feature_readers_equivalent_without_geometry_ram_cache(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('actual CUDA unavailable')
    from test_stc_camera_protocol import fixture
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from real_motion.waymo_native_execution import prepare_waymo_native
    prepare_waymo_native(tmp_path/'native')
    s = fixture(tmp_path, shape=(32, 32, 4)); source = fake_source(s); w = s.windows[0]
    for t in w.history:
        p = source._label_path(w.scene, t)
        with np.load(p) as z: sem = np.array(z['semantics'])
        np.savez_compressed(p, semantics=sem, mask_lidar=np.ones(s.shape, bool))
    history_sha = {t: digest_file(source._label_path(w.scene, t)) for t in w.history}
    rows = [{'key': [w.scene, w.t0]}]; recs = replay.derive_records(source.nusc, rows)
    joint = JointSurfaceCCR(LocalSTWMV17Config(history_frames=4, d_model=16, semantic_dim=4,
        blocks=1, decoder_blocks=1), width=16, z_bins=4).to(device).eval().requires_grad_(False)
    before = replay.state_digest(joint)
    predictor = replay.Predictor(joint, PrepareConfig(grid=OccupancyGrid(-6.4, -6.4, -1., (.4,)*3, s.shape)),
        device, workers=2, graphs=False, geometry_mib=0)
    monkeypatch.setattr(replay, 'navigation_commands', lambda *_: np.full(6, 2, np.int64))
    try:
        r = replay.TrainMaterializer(source, s.catalog, recs, predictor.provider, s.shape)(0, config())
        assert r['train_features']['object_valid'].any() and r['train_features']['surface_valid'].any()
        assert all(v['exact'] for v in replay.feature_difference(r['train_features'], r['eval_features']).values())
        assert before == replay.state_digest(joint)
        assert history_sha == {t: digest_file(source._label_path(w.scene, t)) for t in w.history}
    finally: predictor.close()


def test_original_bank_fingerprint_excludes_only_paired_files_not_new_replay(tmp_path):
    from test_surface_ego_ablation import source_artifacts
    from tools.real_motion.surface_ego_ablation_common import load_source_bank
    from tools.real_motion.ego_trajectory_common import implementation_fingerprint
    root, source, _ = source_artifacts(tmp_path)
    before = implementation_fingerprint(root)
    new = root/'tools/ego_diagnostics'; new.mkdir(); (new/'surface_train_replay.py').write_text('new audit only')
    assert before == implementation_fingerprint(root)
    rows, _, _ = load_source_bank(source, root)
    assert len(rows) == 3


def test_historical_population_metadata_receipt_and_change_guard(tmp_path):
    from test_stc_camera_protocol import fixture
    s = fixture(tmp_path); source = fake_source(s)
    rows = [{'key': [w.scene, w.t0]} for w in s.windows]
    records = replay.derive_records(source.nusc, rows)
    base = s.root/'v1.0-trainval'; base.mkdir()
    for name in replay.METADATA: (base/(name+'.json')).write_text('[]')
    paths = sorted({source._label_path(r['scene_name'], t) for r in records for t in r['history_tokens']})
    original = dict(metadata_hashes={p.name: digest_file(p) for p in base.glob('*.json')},
        historical_files_fingerprint=fingerprint([[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths]))
    receipt = replay.historical_provenance(source, records, [0], original)
    assert len(receipt) == 6+4
    replay.verify_receipt(receipt)
    wrong = dict(original, historical_files_fingerprint='changed')
    with pytest.raises(RuntimeError, match='population/stat'):
        replay.historical_provenance(source, records, [0], wrong)
    (base/'ego_pose.json').write_text('[1]')
    with pytest.raises(RuntimeError, match='metadata changed'):
        replay.historical_provenance(source, records, [0], original)
    with pytest.raises(RuntimeError, match='source changed'): replay.verify_receipt(receipt)


def test_cli_complete_resume_same_contract_readonly_and_existing_dev_guard(tmp_path, monkeypatch):
    rows = data(3); heads = models(); original = tmp_path/'old'; original.mkdir()
    pairdir = tmp_path/'pair'; pairdir.mkdir(); dataroot = tmp_path/'nuscenes'; dataroot.mkdir()
    cfg = tmp_path/'config.yaml'; cfg.write_text('unchanged config')
    env = {k: v for k, v in sorted(__import__('os').environ.items()) if k.startswith('SWFM_')}
    source_audit = dict(source_dir=str(original), frozen_checkpoint='frozen', original_training=dict(
        feature_execution=dict(device='cpu', torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                               WM_precision='fp32'), runtime_environment=env, runtime_config_sha256=digest_file(cfg)))
    training = dict(source=source_audit, schedule={'max_updates': 2000}, head_config=config().__dict__)
    pair = dict(protocol=replay.PAIR_PROTOCOL, updates=2000, contract=training, config=config().__dict__,
        models={k: heads[k].state_dict() for k in TRAIN_NAMES},
        monitors=[dict(update=2000, reports=recorded(rows, heads))])
    torch.save(pair, pairdir/'pair_last.pt'); (pairdir/'contract.json').write_text(json.dumps(training))
    frozen = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    before = replay.state_digest(frozen); digest = digest_file(pairdir/'pair_last.pt')
    monkeypatch.setattr(replay, 'load_source_bank', lambda *_: (rows, {'state_dict': heads['old320'].state_dict()}, source_audit))
    monkeypatch.setattr(replay, 'load_evaluation_model', lambda *a, **kw: ({}, frozen))
    monkeypatch.setattr(replay, 'NuScenesWindowSource', lambda *_: SimpleNamespace(nusc=None))
    monkeypatch.setattr(replay, 'derive_records', lambda *_: [dict(scene_name=r['key'][0]) for r in rows])
    monkeypatch.setattr(replay, 'historical_provenance', lambda *_: {})
    monkeypatch.setattr(replay, 'load_catalog', lambda *_: ({}, {}))
    monkeypatch.setattr(replay, 'load_runtime_config', lambda *_: {})
    monkeypatch.setattr(replay, 'make_prepare_config', lambda *_: None)
    closed = []
    class Predictor:
        def __init__(self, *a, **kw):
            assert kw['geometry_mib'] == 0 and kw['graphs'] is False
            self.provider = SimpleNamespace(pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=(32, 32, 4))))
        def close(self): closed.append(1)
    monkeypatch.setattr(replay, 'Predictor', Predictor)
    monkeypatch.setattr(replay, 'TrainMaterializer', lambda *_: materializer(rows))
    monkeypatch.setattr(replay, 'verify_sources', lambda *_: None)
    out = tmp_path/'replay'
    args = ['--pair-dir', str(pairdir), '--dataroot', str(dataroot), '--config', str(cfg),
            '--out-dir', str(out), '--windows', '3', '--device', 'cpu']
    assert replay.main(argv=args) == 0
    assert json.loads((out/'audit.json').read_text())['pass_gate'] and len(closed) == 1
    assert replay.main(argv=[*args, '--resume']) == 0
    with pytest.raises(RuntimeError, match='contract'): replay.main(argv=[*args, '--resume', '--windows', '2'])
    with pytest.raises(FileExistsError): replay.main(argv=args)
    assert before == replay.state_digest(frozen) and digest == digest_file(pairdir/'pair_last.pt')
    dev = tmp_path/'different_dev.json'
    dev.write_text(json.dumps(dict(state={'status': 'complete'}, contract={'training': {'wrong': True}})))
    with pytest.raises(RuntimeError, match='different'):
        replay.main(argv=[*args, '--out-dir', str(tmp_path/'new'), '--dev-report', str(dev)])
    assert list(original.iterdir()) == []
