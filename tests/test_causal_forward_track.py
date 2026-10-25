"""Initialization, crossing, gaps, t0 order, causal geometry and frozen replay."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_track import track, run
from tools.causal_forward_track.adapter import TrackPredictor
from tools.source_scene_stage2 import track as old


def component(x, cls=4):
    return dict(class_id=cls, centroid_world=np.array([x, 0., 0.]), voxel_indices=np.array([[1,2,1]]))


def build(frames):
    return track.build(frames, np.repeat(np.eye(4)[None],4,0), np.arange(4)*.5,
                       SimpleNamespace(max_match_speed_mps=25.))


def test_global_greedy_gates_ties_and_input_immutable():
    cost = np.array([[2.,1.],[.5,np.inf]])
    saved = cost.copy()
    assert track.greedy(cost) == [(1,0),(0,1)]
    np.testing.assert_array_equal(cost,saved)
    assert track.greedy(np.full((2,3),np.inf)) == []
    assert track.greedy(np.empty((0,2))) == []
    assert track.greedy(np.ones((2,2))) == [(0,0),(1,1)]


def test_config_checks_allow_only_exact_windows_newline_difference(tmp_path):
    import hashlib
    path = tmp_path/'config.yaml'; path.write_bytes(b'a: 1\r\nb: 2\r\n')
    expected = hashlib.sha256(b'a: 1\nb: 2\n').hexdigest()
    assert run.config_identity(path,expected)['LF_sha256'] == expected
    path.write_bytes(b'a: 2\r\nb: 2\r\n')
    with pytest.raises(ValueError): run.config_identity(path,expected)


def test_bootstrap_unknown_then_motion_compensated_crossing_and_original_t0_order():
    frames = [[component(0),component(10)], [component(2),component(8)],
              [component(4),component(6)], [component(4),component(6)]]
    # At t0 B is first (4), A is second (6). The two objects have crossed.
    actual = build(frames)
    expected = np.array([[1,1,1,0],[0,0,0,1]])
    np.testing.assert_array_equal(actual.links,expected)
    assert actual.audit['matched_without_velocity'] == 2
    assert actual.audit['matched_with_velocity'] == 4
    np.testing.assert_array_equal(actual.pair_velocity[0],[-4,0,0])
    np.testing.assert_array_equal(actual.pair_velocity[1],[4,0,0])
    previous = old.build(frames,np.repeat(np.eye(4)[None],4,0),np.arange(4)*.5,
                         SimpleNamespace(max_match_speed_mps=25.))
    assert previous.links[:,2].tolist() == [0,1]  # nearest without compensation swaps identities


def test_unknown_gap_reconnects_and_missing_observation_is_not_filled():
    actual = build([[component(0)],[],[component(2)],[component(3)]])
    assert actual.links.tolist() == [[0,-1,0,0]]
    assert actual.voxels[0][1] is None and not actual.track_valid[0,1]
    assert actual.audit['gap_reconnections'] == 1
    np.testing.assert_array_equal(actual.pair_velocity[0],[2,0,0])


def test_known_gap_does_not_invent_latest_pair_KTA_or_ghost_current_geometry():
    actual = build([[component(0)],[component(1)],[],[component(3)]])
    assert actual.links.tolist() == [[0,0,-1,0]] and actual.pair_velocity == {}
    assert actual.audit['gap_reconnections'] == 1
    gone = build([[component(0)],[],[],[]])
    assert gone.links.shape == (0,4) and gone.audit['historical_only_tracks'] == 1


def test_unknown_is_not_static_class_gate_speed_gate_and_ambiguity():
    actual = build([[component(0)],[component(3)],[component(6)],[component(9)]])
    assert actual.links.tolist() == [[0,0,0,0]] and actual.pair_velocity[0][0] == 6
    actual = build([[component(0,4)],[component(1,5)],[component(2,5)],[component(3,5)]])
    assert actual.links.tolist() == [[-1,0,0,0]]
    actual = build([[component(0)],[component(100)],[],[component(200)]])
    assert actual.links.tolist() == [[-1,-1,-1,0]] and actual.pair_velocity == {}
    actual = build([[component(0)],[component(-1),component(1)],[],[]])
    assert actual.audit['ambiguous_rows'] > 0


def test_reliable_motion_has_priority_over_unknown_component():
    # Unknown at 3 would have zero distance to current 3, but known 2m/s track
    # predicts 3 exactly and must be considered first.
    known = dict(last=1,has_velocity=True,class_id=4,velocity=np.array([2.,0,0]),
                 centers=np.array([[0.,0,0],[2.,0,0],[0,0,0],[0,0,0]]))
    pairs,_ = track.associate([known],[component(3)],2,np.array([-1.5,-.5,0.,.5]),
        has_velocity=True,speed_gate=25.,ambiguity_margin=.4)
    assert pairs == [(0,0)]
    actual = build([[component(0)],[component(2),component(4)],[component(4)],[component(6)]])
    assert actual.links.tolist() == [[0,0,0,0]]
    assert actual.audit['historical_only_tracks'] == 1


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_frozen_preparation_matches_old_baseline_and_fresh_background_per_policy(monkeypatch,device):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    from test_fpn_unet import fixture
    from tools.fpn_unet.adapter import forward_window
    import tools.fpn_unet.adapter as original_adapter
    body, original, row = fixture(device)
    predictor = TrackPredictor(body.eval().requires_grad_(False),original.provider.pcfg,device,workers=1)
    raw = {k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
    raw_before = copy.deepcopy(raw)
    state = {k:v.clone() for k,v in body.state_dict().items()}
    try:
        legacy = predictor.prepare(row[0],raw,times=np.arange(4)*.5)
        rows,audit = predictor.prepare_pair(row[0],raw,times=np.arange(4)*.5)
        for key in ('kta_displacement_xy_m','anchors_xy_t0_m','source_class_id'):
            np.testing.assert_array_equal(rows['baseline'][0][key],legacy[0][key])
        np.testing.assert_array_equal(rows['baseline'][1]['_shared_source_track'].links,
                                      legacy[1]['_shared_source_track'].links)
        for field in ('shared_reads','memory_reads'):
            for key,value in rows['baseline'][3][0][field].items():
                np.testing.assert_array_equal(value,legacy[3][0][field][key])
        calls = []; real = original_adapter.backgrounds
        monkeypatch.setattr(original_adapter,'backgrounds',lambda *a,**kw:(calls.append(a[-1]),real(*a,**kw))[1])
        first = predictor.infer(rows['baseline'],row[1]['future_poses'])
        second = predictor.infer(rows['forward'],row[1]['future_poses'])
        assert calls[0] is rows['baseline'][1]['_shared_source_track']
        assert calls[1] is rows['forward'][1]['_shared_source_track']
        assert first['logits'].shape[0] == second['logits'].shape[0] == 6
        repeated = forward_window(body,predictor.provider,legacy,poses=row[1]['future_poses'],fresh=True)
        if device == 'cpu': torch.testing.assert_close(first['logits'],repeated['logits'],rtol=0,atol=0)
        else: assert torch.isfinite(first['logits']).all() and torch.isfinite(repeated['logits']).all()
        for key,value in raw.items(): np.testing.assert_array_equal(value,raw_before[key])
        for key,value in body.state_dict().items(): torch.testing.assert_close(value,state[key],rtol=0,atol=0)
        assert not any(p.grad is not None for p in body.parameters())
        with pytest.raises(ValueError): predictor.prepare_pair(row[0],dict(raw,future_gt_occ=object()),times=np.arange(4)*.5)
    finally:
        predictor.close(); original.close()


@pytest.mark.parametrize('kind',['fpn','adaln','full','adaln_full'])
def test_implementation_loader_and_interim_rejection(tmp_path,kind):
    from test_fpn_unet import fixture
    body,predictor,_ = fixture()
    from tools.fpn_unet import run as fpn
    from tools.fpn_unet_full import run as full
    from tools.fpn_adaln_zero import run as ada
    from tools.fpn_adaln_zero_full import run as ada_full
    from tools.fpn_adaln_zero.model import make_model
    if 'adaln' in kind:
        body = make_model(body.transport.config,feature_width=16,z_bins=4,memory_width=4)
    protocol = dict(fpn=fpn.PROTOCOL,adaln=ada.PROTOCOL,full=full.PROTOCOL,adaln_full=ada_full.PROTOCOL)[kind]
    epochs,windows = (2,20430) if 'full' in kind else (1,4086)
    try:
        contract = dict(protocol=protocol,epochs=epochs,total_updates=1031*epochs,known='TRAIN_OCC_STC_dense_valid',
            source_policy='shared_pinned',registration='none',model=body.configs(),
            train_keys=[['TRAIN',str(i)] for i in range(windows)],eval_keys=[['VAL',str(i)] for i in range(64)])
        saved = dict(protocol=protocol,contract=contract,models={'gt':body.state_dict()},epoch=epochs,batch=0,updates=1031*epochs)
        path = tmp_path/'source.pt'; torch.save(saved,path)
        loaded,c,receipt = run.load_source(checkpoint=path)
        assert receipt['model_variant'] == body.configs()['variant'] and not loaded.training
        assert all(not p.requires_grad for p in loaded.parameters())
        saved['contract']['implementation'] = {'tools/real_motion/joint_training_distributed.py':'old_training_sha'}
        torch.save(saved,path)
        _,_,receipt = run.load_source(checkpoint=path)
        assert receipt['unused_training_source_drift']['tools/real_motion/joint_training_distributed.py']['executed'] is False
        saved['contract']['implementation'] = {'tools/source_scene_stage2/track.py':'wrong_inference_sha'}
        torch.save(saved,path)
        with pytest.raises(ValueError,match='original source implementation changed'): run.load_source(checkpoint=path)
        saved['contract']['implementation'] = {}
        saved['updates'] = 500; torch.save(saved,path)
        with pytest.raises(ValueError): run.load_source(checkpoint=path)
    finally: predictor.close()


def test_resume_integer_counts_population_and_unknown_Moving_not_reported():
    contract = dict(keys=[['VAL','0']],source=dict(model_variant='test'),stc_age=False)
    counts = {}
    for n in run.NAMES:
        value = run.metrics.empty(3); value['confusion'][:,4,4] = 8
        counts[n] = {k:v.tolist() for k,v in value.items()}
    result = dict(status='complete',contract=contract,keys=contract['keys'],audits=[{}],
                  moving_available=False,counts=counts)
    run.restore(result,contract,8,False)
    run.summarize(result)
    assert 'MovingMicro' not in result['reports']['occ/baseline/transport']['average']
    assert result['forward_minus_baseline']['stc/semantic_refine']['mIoU'] == 0
    wrong = copy.deepcopy(result); wrong['keys'] = [['VAL','other']]
    with pytest.raises(ValueError): run.restore(wrong,contract,8,False)
    wrong = copy.deepcopy(result); wrong['counts'][run.NAMES[0]]['confusion'][0][4][4] -= 1
    with pytest.raises(ValueError): run.restore(wrong,contract,8,False)


def test_actual_cli_loop_full6_before_labels_stop_resume_no_optimizer(tmp_path,monkeypatch):
    import threading
    from real_motion.geometry import OccupancyGrid
    grid = OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(.4,.4,.4),shape_hwd=(2,2,2))
    pcfg = SimpleNamespace(grid=grid)
    model = torch.nn.Linear(1,1).eval().requires_grad_(False)
    model.columns = SimpleNamespace(horizon_chunk=6)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    source = tmp_path/'source'; source.mkdir(); ckpt = source/'checkpoint.pt'; ckpt.write_bytes(b'not_loaded_fixture')
    bundle = tmp_path/'data.zip'; bundle.write_bytes(b'not_loaded_fixture')
    contract = dict(eval_keys=[['VAL',str(i)] for i in range(64)],train_keys=[['TRAIN','0']],overrides=None)
    receipt = dict(path=str(ckpt),file_sha256=run.base.sha256(ckpt),model_variant='test')
    monkeypatch.setattr(run,'load_source',lambda *a:(model,contract,receipt))
    monkeypatch.setattr(run.base,'make_prepare_config',lambda *_:pcfg)
    calls = dict(infer=0,labels=0); stop = threading.Event()

    class Data:
        def __init__(self,*a,**kw):
            self.pcfg = pcfg
            self.by_key = {('VAL',str(i)):dict(index=i,future_poses=np.repeat(np.eye(4)[None],6,0)) for i in range(64)}
        def history(self,job,*_): return {},{},np.arange(4)*.5
        def targets(self,job):
            assert calls['infer'] == 4*(job['index']+1)  # both modalities AND both policies already inferred
            calls['labels'] += 1
            return np.full((3,2,2,2),4,np.uint8)
        def close(self): pass

    class Predictor:
        def __init__(self,*a,**kw): pass
        def prepare_pair(self,*a,**kw): return {'baseline':0,'forward':1},{}
        def infer(self,*a,**kw):
            calls['infer'] += 1
            if calls['infer'] == 4: stop.set()
            value = torch.zeros(6,18,8); value[:,4] = 1
            return dict(prior=value,logits=value,shape=(2,2,2))
        def close(self): pass
    monkeypatch.setattr(run,'Bundle',Data); monkeypatch.setattr(run,'TrackPredictor',Predictor)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**kw:pytest.fail('optimizer created'))
    args = ['--checkpoint',str(ckpt),'--bundle',str(bundle),'--out-dir',str(tmp_path/'out'),
            '--device','cpu','--limit','16','--horizon-chunk','1']
    assert run.main(args,event=stop) == 130 and calls == dict(infer=4,labels=1)
    stop.clear()
    assert run.main(args+['--resume'],event=stop) == 0 and calls == dict(infer=64,labels=16)
    import json
    result = json.loads((tmp_path/'out'/'results.json').read_text())
    assert result['status'] == 'complete' and len(result['keys']) == 16
    assert all(r['average']['mIoU'] == 100. for r in result['reports'].values())
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
