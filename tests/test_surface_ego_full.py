from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_surface_ego_head import config, feature_row
from test_surface_ego_ablation import assert_nested_equal, source_artifacts
from real_motion.ego_trajectory_head import HistoryEgoTrajectoryHead
from tools.ego_experiments import train_surface_ego_full as full
from tools.ego_experiments import eval_surface_ego_full as dev
from tools.real_motion.ego_trajectory_common import atomic_save, digest_file, fingerprint
from tools.real_motion.surface_ego_ablation_common import load_source_bank, tensor_state_fingerprint


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def data(n=11):
    return [dict(key=['train-scene-'+str(i % 4), str(i)], features=feature_row(),
        commands=np.full(6, i % 3, np.int64),
        target=np.c_[np.arange(1, 7)*.2, np.zeros(6), np.arange(1, 7)*.01].astype(np.float32))
        for i in range(n)]


def contract(rows, batch_size=2):
    keys = [r['key'] for r in rows]
    return dict(keys=keys, split=full.scene_split(keys), head_config=asdict(config()),
        schedule=dict(epochs=1, batch_size=batch_size, lr=3e-4, min_lr=3e-6, seed=21, radius_m=10.))


def test_scene_split_is_key_order_independent_disjoint_and_not_target_selected():
    rows = data(80); keys = [r['key'] for r in rows]
    a = full.scene_split(keys); b = full.scene_split(keys[::-1])
    assert a['holdout_scenes'] == b['holdout_scenes']
    assert not set(a['train_scenes']) & set(a['holdout_scenes'])
    assert sorted(a['train_indices']+a['holdout_indices']) == list(range(80))
    assert {keys[i][0] for i in a['holdout_indices']} == set(a['holdout_scenes'])
    for bad, kw in (([['s','a']], {}), (keys+[keys[0]], {}), (keys, {'fraction':.5})):
        with pytest.raises(ValueError): full.scene_split(bad, **kw)


def test_new_experiment_location_does_not_invalidate_original_bank(tmp_path):
    root, source, _ = source_artifacts(tmp_path)
    before = {p:digest_file(p) for p in source.rglob('*') if p.is_file()}
    (root/'tools/ego_experiments').mkdir()
    (root/'tools/ego_experiments/train_surface_ego_full.py').write_text('new experiment only')
    rows, saved, _ = load_source_bank(source, root)
    assert len(rows) == 3 and saved['updates'] == 4
    assert before == {p:digest_file(p) for p in before}


def test_bank_reuses_original_and_recovers_without_mutating_it(tmp_path):
    rows = data(4); before = deepcopy(rows)
    records = [dict(scene_name=r['key'][0], t0_token=r['key'][1]) for r in rows]
    c = contract(rows); out = tmp_path/'bank_run'; out.mkdir()
    def forbidden(*a): pytest.fail('reusable history must not be read or extracted')
    stop = Event(); stop.set()
    partial = full.build_bank(out, records, c, rows, None, forbidden, reader=forbidden, stop_event=stop)
    assert partial['status'] == 'stopped' and partial['windows'] == 0
    inventory = full.build_bank(out, records, c, rows, None, forbidden, reader=forbidden)
    assert inventory['reused'] == 4 and inventory['fresh'] == 0
    loaded = full.load_bank(out, c, inventory)
    for actual, old in zip(loaded, before):
        assert_nested_equal(actual['features'], old['features'])
        np.testing.assert_array_equal(actual['target'], old['target'])
    assert_nested_equal([r['features'] for r in rows], [r['features'] for r in before])
    old_sha = [digest_file(out/'bank'/f'{i:06d}.pt') for i in range(4)]
    again = full.build_bank(out, records, c, rows, None, forbidden, reader=forbidden)
    assert again == inventory
    assert old_sha == [digest_file(out/'bank'/f'{i:06d}.pt') for i in range(4)]
    with pytest.raises(RuntimeError, match='complete'):
        full.load_bank(out, c, inventory|{'shards':inventory['shards'][:-1]})
    with pytest.raises(RuntimeError, match='memory'):
        full.load_bank(out, c, inventory, max_mib=.0001)
    p = out/'bank/000000.pt'; damaged = torch.load(p, weights_only=False)
    damaged['features']['ego_history'][0, 0] += 1; atomic_save(p, damaged)
    with pytest.raises(RuntimeError, match='shard'):
        full.build_bank(out, records, c, rows, None, forbidden, reader=forbidden)
    with pytest.raises(RuntimeError, match='inventory'):
        full.load_bank(out, c, inventory)


def test_history_prefetch_lru_and_supervision_read_boundary(tmp_path, monkeypatch):
    from test_stc_camera_protocol import fixture
    from test_surface_ego_train_replay import fake_source
    s = fixture(tmp_path); source = fake_source(s); w = s.windows[0]
    record = dict(scene_name=w.scene, t0_token=w.t0, history_tokens=w.history, future_tokens=w.future)
    out = tmp_path/'new_bank'; out.mkdir(); calls = []; stop = Event()
    actual_np = np.load; actual_pose = source.pose
    def load(path, *a, **kw):
        assert Path(path) in {source._label_path(w.scene, t) for t in w.history}
        return actual_np(path, *a, **kw)
    monkeypatch.setattr(np, 'load', load)
    def pose(t):
        if t in w.future: assert calls == ['features']
        return actual_pose(t)
    source.pose = pose
    def extract(rec, raw, times):
        assert set(raw) == {'history_occ','history_observed','history_poses'}
        assert rec == record and len(times) == 4
        calls.append('features'); stop.set(); return feature_row()
    def cmd(nusc, future):
        assert future == w.future and calls == ['features']
        return np.full(6, 2)
    monkeypatch.setattr(full, 'navigation_commands', cmd)
    reader = full.HistoryReader(source, ram_mib=.0002)
    a, m = reader.frame(w.scene, w.history[0])
    assert not a.flags.writeable and not m.flags.writeable
    c = dict(keys=[[w.scene,w.t0]])
    inventory = full.build_bank(out, [record], c, [], source, extract, reader=reader,
        workers=2, stop_event=stop)
    assert inventory['status'] == 'complete' and inventory['fresh'] == 1
    assert reader.bytes <= reader.limit
    assert full.load_bank(out, c, inventory)[0]['target'].shape == (6,3)


def test_load_bank_hashes_large_contract_once(tmp_path, monkeypatch):
    rows = data(8); records = [dict(scene_name=r['key'][0], t0_token=r['key'][1]) for r in rows]
    out = tmp_path/'new'; out.mkdir(); c = contract(rows)
    inventory = full.build_bank(out, records, c, rows, None, lambda *a:None)
    calls = []; original = full.fingerprint
    def counted(x):
        if x is c: calls.append(1)
        return original(x)
    monkeypatch.setattr(full, 'fingerprint', counted)
    assert len(full.load_bank(out, c, inventory)) == 8 and len(calls) == 1


def test_fresh_bank_mid_population_recovery_never_reextracts_prefix(tmp_path, monkeypatch):
    records = [dict(scene_name='train',t0_token=str(i),history_tokens=['h']*4,future_tokens=['f']*6) for i in range(3)]
    c = dict(keys=[['train',str(i)] for i in range(3)])
    source = SimpleNamespace(pose=lambda _:np.eye(4), nusc=None)
    monkeypatch.setattr(full, 'navigation_commands', lambda *a:np.full(6,2))
    reader = lambda _:({'history_poses':[np.eye(4)]*4}, [0,.5,1,1.5])
    stop = Event(); seen = []; out = tmp_path/'fresh'; out.mkdir()
    def extract(rec, raw, times):
        seen.append(rec['t0_token']); stop.set(); return feature_row()
    partial = full.build_bank(out, records, c, [], source, extract, reader=reader, stop_event=stop)
    assert partial['status'] == 'stopped' and partial['windows'] == 1 and seen == ['0']
    sha = digest_file(out/'bank/000000.pt')
    def again(rec, raw, times):seen.append(rec['t0_token']);return feature_row()
    inventory = full.build_bank(out, records, c, [], source, again, reader=reader)
    assert inventory['status'] == 'complete' and seen == ['0','1','2']
    assert digest_file(out/'bank/000000.pt') == sha


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_fresh_real_frozen_history_bank_and_no_future_strong(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA unavailable')
    from test_stc_camera_protocol import fixture
    from test_surface_ego_train_replay import fake_source
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from tools.real_motion.stc_shared_execution import Predictor
    from tools.real_motion import joint_long_rollout_common as rollout
    from real_motion.waymo_native_execution import prepare_waymo_native
    prepare_waymo_native(tmp_path/'native')
    s = fixture(tmp_path,shape=(32,32,4));source = fake_source(s);w = s.windows[0]
    rec = dict(scene_name=w.scene,t0_token=w.t0,history_tokens=w.history,future_tokens=w.future)
    joint = JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,
        blocks=1,decoder_blocks=1),width=16,z_bins=4).to(device).eval().requires_grad_(False)
    sha = tensor_state_fingerprint(joint.state_dict())
    predictor = Predictor(joint,PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1.,(.4,)*3,s.shape)),
        device,workers=2,graphs=False,geometry_mib=8)
    raw, times = full.HistoryReader(source)(rec)
    raw['history_observed'][:] = True;raw['history_occ'][:,4:10,4:10,1:3] = 4
    monkeypatch.setattr(full,'navigation_commands',lambda *a:np.full(6,2))
    monkeypatch.setattr(rollout.legacy,'_strong_all_horizons',lambda *a,**kw:pytest.fail('future Strong prohibited'))
    out = tmp_path/'real_bank';out.mkdir();c = dict(keys=[[w.scene,w.t0]])
    def extract(r, history, timestamps):
        return full.extract_history_features(predictor.provider,r,history,config(),timestamps_s=timestamps)
    try:
        expected = extract(rec,raw,times)
        inventory = full.build_bank(out,[rec],c,[],source,extract,reader=lambda _:(raw,times))
        row = full.load_bank(out,c,inventory)[0]
        assert_nested_equal(row['features'], expected)
        assert row['features']['object_valid'].any() and row['features']['surface_valid'].any()
        assert tensor_state_fingerprint(joint.state_dict()) == sha
        assert all(p.grad is None for p in joint.parameters())
    finally:predictor.close()


def test_one_epoch_uses_every_fit_window_once_and_final_weight_even_if_worse(tmp_path, monkeypatch):
    rows = data(); c = contract(rows); initial = HistoryEgoTrajectoryHead(config())
    initial_sha = tensor_state_fingerprint(initial.state_dict()); out = tmp_path/'train'; out.mkdir()
    actual_report = full.report; calls = []
    def worse(head, packed, ids, **kw):
        r = actual_report(head, packed, ids, **kw)
        if ids == c['split']['holdout_indices']:
            r['geometry_loss'] = 1. if not calls else 2.; calls.append(1)
        return r
    monkeypatch.setattr(full, 'report', worse)
    result = full.train(initial, rows, c, out, batch_size=2, checkpoint_every=1)
    assert result['completed_epochs'] == result['evaluated_epoch'] == 1
    assert result['updates'] == int(np.ceil(len(c['split']['train_indices'])/2))
    assert result['holdout']['geometry_loss'] > result['baseline']['geometry_loss']
    final = torch.load(out/'head_epoch1.pt', weights_only=False)
    assert final['selected_epoch'] == 1 and final['training_completed']
    assert final['state_fingerprint'] != initial_sha
    assert final['state_fingerprint'] == tensor_state_fingerprint(initial.state_dict())
    assert not (out/'head_best.pt').exists()


@pytest.mark.parametrize('stop_after', [1,4])
def test_one_epoch_adam_rng_cosine_cursor_exact_resume(tmp_path, monkeypatch, stop_after):
    torch.manual_seed(2); initial = HistoryEgoTrajectoryHead(config()); rows = data(); c = contract(rows)
    whole, interrupted = [tmp_path/k for k in ('whole','interrupted')]
    whole.mkdir(); interrupted.mkdir(); kw = dict(batch_size=2, checkpoint_every=1)
    full.train(deepcopy(initial), rows, c, whole, **kw)
    stop = Event(); original_step = torch.optim.AdamW.step; steps = []
    def step(opt, *a, **kw):
        r = original_step(opt, *a, **kw); steps.append(1)
        if len(steps) == stop_after: stop.set()
        return r
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW, 'step', step)
        partial = full.train(deepcopy(initial), rows, c, interrupted, stop_event=stop, **kw)
    assert partial['updates'] == stop_after
    if stop_after == 1: assert partial['status'] == 'stopped'
    full.train(deepcopy(initial), rows, c, interrupted, resume=True, **kw)
    a = torch.load(whole/'last.pt', weights_only=False); b = torch.load(interrupted/'last.pt', weights_only=False)
    for key in ('state_dict','optimizer','updates','epoch','cursor','order','sampling_rng','epoch_sum','history','baseline'):
        assert_nested_equal(a[key], b[key])
    export_sha = digest_file(interrupted/'head_epoch1.pt'); last_sha = digest_file(interrupted/'last.pt')
    full.train(deepcopy(initial), rows, c, interrupted, resume=True, **kw)
    assert export_sha == digest_file(interrupted/'head_epoch1.pt') and last_sha == digest_file(interrupted/'last.pt')
    with pytest.raises(RuntimeError, match='contract'):
        full.train(deepcopy(initial), rows, c|{'changed':True}, interrupted, resume=True, **kw)
    with pytest.raises(RuntimeError, match='schedule'):
        full.train(deepcopy(initial), rows, c, interrupted, batch_size=4, resume=True)
    b['cursor'] = 1; atomic_save(interrupted/'last.pt', b)
    with pytest.raises(RuntimeError, match='cursor'):
        full.train(deepcopy(initial), rows, c, interrupted, resume=True, **kw)


def test_epoch_mean_weighting_final_short_batch_and_export_tampering(tmp_path, monkeypatch):
    rows = data(11); c = contract(rows)
    monkeypatch.setattr(full, 'geometry_loss', lambda out, target, **kw:out['se2'].sum()*0+len(target))
    out = tmp_path/'run'; out.mkdir(); initial = HistoryEgoTrajectoryHead(config())
    result = full.train(deepcopy(initial), rows, c, out, batch_size=2)
    n = len(c['split']['train_indices']); expected = ((n//2)*4+n%2)/n
    assert result['last_epoch']['train_mean_loss'] == pytest.approx(expected)
    p = out/'head_epoch1.pt'; saved = torch.load(p, weights_only=False); saved['selected_epoch'] = 0
    atomic_save(p, saved)
    with pytest.raises(RuntimeError, match='export'):
        full.train(deepcopy(initial), rows, c, out, batch_size=2, resume=True)
    with pytest.raises(ValueError):full.train(deepcopy(initial), rows, c, out, epochs=2)


def test_full_training_cli_cache_only_resume_then_one_epoch_frozen_sources(tmp_path, monkeypatch):
    rows = data(8); initial = HistoryEgoTrajectoryHead(config())
    origin = tmp_path/'original'; origin.mkdir(); inputs = tmp_path/'inputs'; inputs.mkdir()
    frozen = inputs/'frozen.pt'; frozen.write_bytes(b'immutable WM')
    cache = inputs/'cache.pt'; cache.write_bytes(b'TRAIN cache')
    info = inputs/'info.pkl'; info.write_bytes(b'TRAIN info')
    cfg = inputs/'config.yaml'; cfg.write_bytes(b'runtime geometry')
    dataroot = inputs/'nuscenes'; (dataroot/'v1.0-trainval').mkdir(parents=True)
    (dataroot/'v1.0-trainval/sample.json').write_text('[]')
    p = dataroot/'history.npz'; p.write_bytes(b'historical immutable file')
    metadata = {'sample.json':digest_file(dataroot/'v1.0-trainval/sample.json')}
    old = dict(schedule=dict(lr=3e-4,min_lr=3e-6,seed=21), head_config=asdict(config()),
        feature_execution=dict(device='cpu',torch_version=str(torch.__version__),cuda_version=torch.version.cuda,WM_precision='fp32'),
        runtime_environment={k:v for k,v in sorted(full.os.environ.items()) if k.startswith('SWFM_')},
        train_cache_sha256=digest_file(cache), train_info_sha256=digest_file(info),
        runtime_config_sha256=digest_file(cfg), metadata_hashes=metadata)
    audit = dict(original_training=old, frozen_checkpoint=str(frozen), frozen_sha256=digest_file(frozen))
    records = [dict(scene_name=r['key'][0],t0_token=r['key'][1],history_tokens=['h']*4,future_tokens=['f']*6) for r in rows]
    source = SimpleNamespace(allowed_scenes=set(r['key'][0] for r in rows),nusc=None,_label_path=lambda *a:p)
    monkeypatch.setattr(full, 'load_source_bank', lambda *a:(deepcopy(rows), {}, audit))
    monkeypatch.setattr(full, 'verify_sources', lambda a:None)
    monkeypatch.setattr(full, 'recover_original_initialization', lambda *a:deepcopy(initial))
    monkeypatch.setattr(full, 'load_cache', lambda *a:({}, records))
    monkeypatch.setattr(full, 'NuScenesWindowSource', lambda *a,**kw:source)
    monkeypatch.setattr(full, 'validate_window_identity', lambda *a:None)
    model = torch.nn.Linear(2,2).eval().requires_grad_(False); model_sha = tensor_state_fingerprint(model.state_dict())
    monkeypatch.setattr(full, 'load_evaluation_model', lambda *a,**kw:({},model))
    monkeypatch.setattr(full, 'load_runtime_config', lambda *a:{})
    monkeypatch.setattr(full, 'make_prepare_config', lambda *a:None)
    monkeypatch.setattr(full, 'Predictor', lambda *a,**kw:SimpleNamespace(close=lambda:None))
    out = tmp_path/'expanded'
    args = ['--source-dir',str(origin),'--out-dir',str(out),'--train-cache',str(cache),'--train-info',str(info),
        '--dataroot',str(dataroot),'--config',str(cfg),'--device','cpu','--expected-windows','8','--batch-size','2']
    stop = Event(); stop.set()
    assert full.main(stop, argv=args) == 130
    assert json.loads((out/'bank_inventory.json').read_text())['status'] == 'stopped'
    assert full.main(argv=[*args,'--resume']) == 0
    result = json.loads((out/'training_summary.json').read_text())
    assert result['completed_epochs'] == 1 and result['evaluated_epoch'] == 1
    before = digest_file(out/'head_epoch1.pt')
    assert full.main(argv=[*args,'--resume']) == 0 and before == digest_file(out/'head_epoch1.pt')
    assert tensor_state_fingerprint(model.state_dict()) == model_sha and frozen.read_bytes() == b'immutable WM'
    assert list(origin.iterdir()) == []
    with pytest.raises(RuntimeError, match='schedule'):
        full.main(argv=[*args,'--resume','--batch-size','4'])


def test_final_epoch_export_gates_code_sources_state_and_epoch(tmp_path, monkeypatch):
    rows = data(); out = tmp_path/'run'; out.mkdir(); c = contract(rows)
    root = tmp_path/'code'; root.mkdir()
    for f in full.IMPLEMENTATION:
        p = root/f; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('pinned experiment')
    frozen = tmp_path/'frozen.pt'; frozen.write_bytes(b'WM')
    c.update(feature_geometry_implementation='original-code', implementation={p:digest_file(root/p) for p in full.IMPLEMENTATION},
        frozen_checkpoint=str(frozen),frozen_sha256=digest_file(frozen))
    full.train(HistoryEgoTrajectoryHead(config()), rows, c, out, batch_size=2)
    saved = torch.load(out/'head_epoch1.pt', weights_only=False)
    monkeypatch.setattr(dev, 'implementation_fingerprint', lambda root:'original-code')
    assert dev.validate_selected(saved, root=root) == c
    with pytest.raises(RuntimeError, match='final-epoch1'):
        dev.validate_selected(saved|{'selected_epoch':0}, root=root)
    with pytest.raises(RuntimeError, match='final-epoch1'):
        dev.validate_selected(torch.load(out/'last.pt', weights_only=False), root=root)
    altered = deepcopy(saved); next(iter(altered['state_dict'].values())).add_(.01)
    with pytest.raises(RuntimeError, match='final-epoch1'):
        dev.validate_selected(altered, root=root)
    (root/full.IMPLEMENTATION[0]).write_text('changed')
    with pytest.raises(RuntimeError, match='implementation'):
        dev.validate_selected(saved, root=root)


def test_new_dev_cli_six_routes_readonly_and_atomic_same_directory_resume(tmp_path, monkeypatch):
    from test_stc_camera_protocol import fixture
    from tools.real_motion.eval_p0_f9_surface_ego_head import evaluate
    source = fixture(tmp_path); source.preflight(source.windows)
    training_dir = tmp_path/'trained'; training_dir.mkdir(); rows = data(); c = contract(rows)
    cfg = tmp_path/'config.yaml'; cfg.write_bytes(b'geometry')
    c.update(runtime_config_sha256=digest_file(cfg),
        runtime_environment={k:v for k,v in sorted(full.os.environ.items()) if k.startswith('SWFM_')},
        frozen_checkpoint='read only', execution=dict(device='cpu',torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,WM_precision='fp32'))
    head = HistoryEgoTrajectoryHead(config()); trained = full.train(head, rows, c, training_dir, batch_size=2)
    (training_dir/'summary.txt').write_text(full.training_text(trained,c,{'reused':0,'fresh':len(rows)}))
    sha = digest_file(training_dir/'head_epoch1.pt'); calls = []
    class Predict:
        provider = None
        def full(self, rec, raw, verify):
            assert raw['future_gt_occ'] is None; calls.append('prediction')
            return None,[raw['history_occ'][-1].copy() for _ in range(6)],None,{}
        def close(self):pass
    targets = source.metric_targets
    def labels(w):
        assert calls[-6:] == ['prediction']*6; calls.append('labels'); return targets(w)
    source.metric_targets = labels
    monkeypatch.setattr(dev, 'validate_selected', lambda saved:c)
    monkeypatch.setattr(dev.STCFourSettingSource, 'from_files', lambda *a,**kw:source)
    monkeypatch.setattr(source, 'select', lambda *a:(source.windows, {'population':'dev64','keys':[]}))
    monkeypatch.setattr(dev, 'load_manifest', lambda *a:({'parent_keys':[]},None,None))
    monkeypatch.setattr(dev, 'NuScenesWindowSource', lambda *a:SimpleNamespace(nusc=None))
    monkeypatch.setattr(dev, 'navigation_commands', lambda *a:np.full(6,2))
    wm = torch.nn.Linear(2,2).eval().requires_grad_(False); wm_sha = tensor_state_fingerprint(wm.state_dict())
    monkeypatch.setattr(dev, 'load_evaluation_model', lambda *a,**kw:({},wm))
    monkeypatch.setattr(dev, 'load_runtime_config', lambda *a:{})
    monkeypatch.setattr(dev, 'make_prepare_config', lambda *a:None)
    monkeypatch.setattr(dev, 'Predictor', lambda *a,**kw:Predict())
    feature = feature_row()
    monkeypatch.setattr(dev, 'evaluate', lambda *a,**kw:evaluate(*a,**kw,feature_extractor=lambda *a,**kw:feature))
    out = tmp_path/'dev'; stop = Event()
    args = ['--training-dir',str(training_dir),'--dataroot',str(source.root),'--stc-root',str(tmp_path/'stc'),
        '--plan-cache',str(tmp_path/'planner'),'--population-manifest',str(tmp_path/'manifest.json'),
        '--config',str(cfg),'--out-dir',str(out),'--device','cpu']
    stop.set(); assert dev.main(stop, argv=args) == 130
    assert dev.main(argv=[*args,'--resume']) == 0
    report = json.loads((out/'evaluation.json').read_text())
    assert report['state']['status'] == 'complete' and len(report['reports']) == 6
    assert sha == digest_file(training_dir/'head_epoch1.pt') and tensor_state_fingerprint(wm.state_dict()) == wm_sha
    assert 'EXPANDED FINAL EPOCH1' in (out/'summary.txt').read_text()
