from copy import deepcopy
from threading import Event
import numpy as np
import pytest
import torch
from test_surface_ego_head import config, feature_row
from test_surface_ego_ablation import assert_nested_equal
from real_motion.ego_navigation import ego_history_features
from real_motion.ego_trajectory_head import HistoryEgoTrajectoryHead
from tools.ego_experiments.ego_kinematic import history_state, integrate, KinematicEgoHead, KinematicPrior
from tools.ego_experiments import train_ego_kinematic as training
from tools.ego_experiments import eval_ego_kinematic as dev
from tools.real_motion.ego_trajectory_common import stack_features, digest_file


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def history(speed=5., acceleration=0., omega=0.):
    t = np.arange(-3,1)*.5; poses = np.repeat(np.eye(4)[None],4,axis=0)
    if omega:
        poses[:,0,3] = speed*(np.cos(omega*t)-1)/omega
        poses[:,1,3] = speed*np.sin(omega*t)/omega
    else: poses[:,1,3] = speed*t+.5*acceleration*t*t
    a = omega*t
    poses[:,0,0] = poses[:,1,1] = np.cos(a);poses[:,1,0] = np.sin(a);poses[:,0,1] = -np.sin(a)
    return torch.tensor(ego_history_features(poses))


@pytest.mark.parametrize('a', [-1.,0.,1.])
def test_current_velocity_not_secant_and_acceleration(a):
    e = history(acceleration=a)[None]; s = history_state(e)
    torch.testing.assert_close(s['velocity'],torch.tensor([[0.,5.]]),atol=2e-5,rtol=1e-5)
    assert float(s['acceleration']) == pytest.approx(a,abs=2e-5)
    p = integrate(s)[0]; t = torch.arange(1,7)*.5
    torch.testing.assert_close(p[:,1],5*t+.5*a*t*t,atol=2e-5,rtol=1e-5)


@pytest.mark.parametrize('omega', [-.2,0.,.2])
def test_turn_xy_yaw_consistency_and_independent_analytic_reference(omega):
    s = history_state(history(omega=omega)[None]); p = integrate(s)[0];t = torch.arange(1,7)*.5
    if omega: target = torch.stack((5*(torch.cos(omega*t)-1)/omega,5*torch.sin(omega*t)/omega,omega*t),1)
    else: target = torch.stack((torch.zeros(6),5*t,torch.zeros(6)),1)
    torch.testing.assert_close(p,target,atol=.002,rtol=1e-5)


def test_stop_reverse_empty_evidence_gradient_and_no_future_fields():
    e = history(speed=1,acceleration=-2)[None];p = integrate(history_state(e))
    assert (p[0,:,1]>=0).all() and torch.allclose(p[0,:,1],torch.full((6,),.25),atol=1e-5)
    row = feature_row();row['ego_history'] = history(speed=-3)
    bank = stack_features([row],'cpu');head = KinematicEgoHead(config())
    out = head(bank,np.full((1,6),2)); assert torch.isfinite(out['se2']).all()
    assert (out['se2'][...,1] < 0).all()
    loss = (out['se2']-torch.ones_like(out['se2'])).square().mean();loss.backward()
    assert head.readout.weight.grad.abs().sum() > 0
    assert not bank['ego_history'].requires_grad
    bank['future_poses'] = torch.ones(1,6,4,4)
    with pytest.raises(ValueError,match='ONLY'):head(bank,np.full((1,6),2))


def test_zero_readout_matches_prior_batch_and_single():
    rows = [feature_row() for _ in range(3)]
    for i,r in enumerate(rows):r['ego_history'] = history(speed=2+i,omega=.02*i)
    bank = stack_features(rows,'cpu');cmd = np.full((3,6),2)
    head = KinematicEgoHead(config()).eval()
    batched = head(bank,cmd)['se2'];prior = KinematicPrior(config())(bank,cmd)['se2']
    torch.testing.assert_close(batched,prior,atol=0,rtol=0)
    for i in range(3):torch.testing.assert_close(head({k:v[i:i+1] for k,v in bank.items()},cmd[i:i+1])['se2'],batched[i:i+1],atol=1e-6,rtol=1e-6)
    with pytest.raises(ValueError):head(bank,np.full((3,6),3))


def test_stationary_can_launch_in_ego_forward_axis_without_future_input():
    row = feature_row();row['ego_history'] = history(speed=0.)
    bank = stack_features([row],'cpu');head = KinematicEgoHead(config(),scene=False)
    p = head(bank,np.full((1,6),2))['se2']; assert p.count_nonzero() == 0
    with torch.no_grad():head.readout.bias[::2] = .2
    p = head(bank,np.full((1,6),2))['se2']
    assert p[0,-1,0] > 0 and p[0,:,1:].count_nonzero() == 0
    p.sum().backward();assert head.readout.bias.grad.abs().sum()>0


def test_vectorized_stop_restart_matches_scalar_recurrence():
    from torch.nn import functional as F
    s = history_state(history(speed=1.)[None])
    controls = torch.tensor([[[-4.,0.],[-4.,0.],[4.,0.]]],requires_grad=True)
    p = integrate(s,controls)
    a = F.interpolate(controls.transpose(1,2),size=48,mode='linear',align_corners=True)[0,0]
    speed = torch.tensor(1.);y = torch.tensor(0.);result = []
    for i in range(48):
        nxt = (speed+a[i]/16).clamp_min(0.);y = y+(speed+nxt)/32;speed = nxt
        if (i+1)%8 == 0:result.append(y)
    expected = torch.stack(result)
    torch.testing.assert_close(p[0,:,1],expected,atol=1e-6,rtol=1e-6)
    assert p[0,-1,1] > p[0,-2,1]  # restart really occurs
    grad, = torch.autograd.grad(p[0,:,1].sum(),controls,retain_graph=True)
    ref_grad, = torch.autograd.grad(expected.sum(),controls)
    torch.testing.assert_close(grad,ref_grad,atol=1e-6,rtol=1e-6)


def test_paired_train_resume_is_exact_and_source_unchanged(tmp_path,monkeypatch):
    from test_surface_ego_full import data
    rows = data(12)
    for i,r in enumerate(rows):r['features']['ego_history'] = history(speed=2+i*.1)
    ids = [i for i,r in enumerate(rows) if r['key'][0] != 'train-scene-0'];held = [i for i in range(12) if i not in ids]
    c = dict(seed=21,source_training=dict(split=dict(train_indices=ids,holdout_indices=held)))
    torch.manual_seed(21);models = {k:KinematicEgoHead(config(),scene=k=='control_scene') for k in training.ARMS}
    legacy = HistoryEgoTrajectoryHead(config()).eval().requires_grad_(False)
    before = deepcopy(rows); a,b = tmp_path/'whole',tmp_path/'resume';a.mkdir();b.mkdir()
    training.train(deepcopy(models),legacy,rows,c,a,checkpoint_every=1)
    stop = Event();real_step = torch.optim.AdamW.step;calls = []
    def step(opt,*args,**kw):
        r = real_step(opt,*args,**kw);calls.append(1)
        if len(calls) == 2:stop.set()
        return r
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW,'step',step)
        assert training.train(deepcopy(models),legacy,rows,c,b,stop_event=stop,checkpoint_every=1)['status']=='stopped'
    training.train(deepcopy(models),legacy,rows,c,b,resume=True,checkpoint_every=1)
    aa = torch.load(a/'last.pt',weights_only=False);bb = torch.load(b/'last.pt',weights_only=False)
    for k in ('models','optimizers','update','epoch','cursor','order','rng','history'):assert_nested_equal(aa[k],bb[k])
    for r,before_row in zip(rows,before):
        assert r['key'] == before_row['key']
        assert_nested_equal(r['features'],before_row['features'])
        for key in ('commands','target'):np.testing.assert_array_equal(r[key],before_row[key])
    sha = digest_file(b/'heads_epoch3.pt');training.train(deepcopy(models),legacy,rows,c,b,resume=True)
    assert digest_file(b/'heads_epoch3.pt') == sha
    with pytest.raises(RuntimeError,match='contract'):training.train(deepcopy(models),legacy,rows,c|{'seed':22},b,resume=True)


def models():
    cfg = config()
    return dict(old_epoch3=HistoryEgoTrajectoryHead(cfg),kinematic_prior=KinematicPrior(cfg),
        control_history=KinematicEgoHead(cfg,scene=False),control_scene=KinematicEgoHead(cfg,scene=True))


def test_eval_all_12_before_gt_scores_exact_resume_and_no_partial_counts(tmp_path):
    from test_stc_camera_protocol import fixture
    source = fixture(tmp_path);source.preflight(source.windows);calls = []
    class Predictor:
        provider = None
        def full(self,rec,raw,verify):
            assert raw['future_gt_occ'] is None
            calls.append(rec['t0_token']);return None,[raw['history_occ'][-1].copy() for _ in range(6)],None,{}
    original = source.metric_targets
    def targets(w):
        assert calls.count(w.t0)>=12
        return original(w)
    source.metric_targets = targets;row = feature_row();extract = lambda *a,**kw:row
    heads = models();contract = dict(modalities=['occ','stc']);stop = Event();saved = {}
    dev.evaluate(source,source.windows,Predictor(),heads,lambda _:np.full(6,2),contract,
        feature_extractor=extract,stop_event=stop,save=lambda s:saved.update(s),progress=lambda _:stop.set())
    resumed = dev.evaluate(source,source.windows,Predictor(),heads,lambda _:np.full(6,2),contract,feature_extractor=extract,saved=saved)
    whole = dev.evaluate(source,source.windows,Predictor(),heads,lambda _:np.full(6,2),contract,feature_extractor=extract)
    assert whole['reports'] == resumed['reports'] and whole['trajectory'] == resumed['trajectory']
    calls.clear();prefix = {}
    class Fail(Predictor):
        def full(self,*a,**kw):
            if len(calls)==5:raise RuntimeError('mid-window')
            return super().full(*a,**kw)
    with pytest.raises(RuntimeError,match='mid-window'):
        dev.evaluate(source,source.windows,Fail(),heads,lambda _:np.full(6,2),contract,feature_extractor=extract,save=lambda s:prefix.update(s))
    assert prefix['completed_windows'] == 0
    assert sum(np.asarray(v).sum() for v in prefix['counts'].values()) == 0


def test_original_feature_fingerprint_does_not_include_new_experiment_files():
    from tools.real_motion.ego_trajectory_common import implementation_fingerprint
    assert 'tools/ego_experiments' not in implementation_fingerprint.__code__.co_consts


def test_cli_fit_reuses_exact_bank_and_source_readonly(tmp_path,monkeypatch):
    from test_surface_ego_partial import fixture
    from tools.ego_experiments import screen_surface_ego_partial as pilot
    from tools.ego_experiments import train_surface_ego_three as three
    parent, _, _, root, _ = fixture(tmp_path,monkeypatch)
    for name in three.IMPLEMENTATION:
        p = root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('immutable three-epoch code')
    source = tmp_path/'partial';pilot.fit(parent,source,max_windows=8,min_windows=4,device='cpu')
    monkeypatch.setattr(three,'ROOT',root)
    initial = HistoryEgoTrajectoryHead(config())
    monkeypatch.setattr(three,'recover_original_initialization',lambda *a:deepcopy(initial))
    old_three = tmp_path/'old_three';three.fit(source,old_three,device='cpu')
    before = {p:digest_file(p) for d in (parent,source,old_three) for p in d.rglob('*') if p.is_file() and p.name != '.evaluation.lock'}
    out = tmp_path/'control'
    assert training.fit(old_three,out,device='cpu') == 0
    selected = torch.load(out/'heads_epoch3.pt',weights_only=False)
    assert selected['epochs'] == 3 and set(selected['models']) == set(training.ARMS)
    old_contract = torch.load(old_three/'last.pt',weights_only=False)['contract']
    assert selected['contract']['source_training']['keys'] == old_contract['keys']
    assert selected['contract']['source_training']['split'] == old_contract['split']
    assert {p:digest_file(p) for p in before} == before
    assert training.fit(old_three,out,device='cpu',resume=True) == 0
    assert {p:digest_file(p) for p in before} == before


def test_real_small_frozen_wm_control_geometry_and_weights_unchanged(tmp_path):
    from test_stc_camera_protocol import fixture
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from real_motion.waymo_native_execution import prepare_waymo_native
    from real_motion.native_column_cpu import prepare_native
    from tools.real_motion.stc_shared_execution import Predictor
    from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
    prepare_waymo_native(tmp_path/'native')
    prepare_native(tmp_path/'column_native')
    source = fixture(tmp_path,shape=(32,32,4));source.preflight(source.windows)
    joint = JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,blocks=1,decoder_blocks=1),width=16,z_bins=4).eval().requires_grad_(False)
    before = tensor_state_fingerprint(joint.state_dict())
    predictor = Predictor(joint,PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1.,(.4,)*3,source.shape)),
        'cpu',workers=2,graphs=False,geometry_mib=8)
    actual = predictor.full; calls = []
    def full(rec,raw,verify):
        prep,dense,prob,stats = actual(rec,raw,verify=False)
        for i,p in enumerate(raw['future_poses']):np.testing.assert_allclose(prep.state['world_to_future'][i],np.linalg.inv(p))
        calls.append(1);return prep,dense,prob,stats
    predictor.full = full
    try:
        result = dev.evaluate(source,source.windows[:1],predictor,models(),lambda _:np.full(6,2),dict(modalities=['occ','stc']))
        assert result['state']['status'] == 'complete' and len(result['reports']) == len(calls) == 12
        assert tensor_state_fingerprint(joint.state_dict()) == before
    finally:predictor.close()
