from copy import deepcopy
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real_motion.ego_trajectory_head import (EgoHeadConfig,HistoryEgoTrajectoryHead,ego_supervision,command_indices)
from real_motion.ego_navigation import (relative_se2,poses_from_se2,ego_history_features,
    replace_future_geometry,navigation_commands)
from tools.real_motion.ego_trajectory_common import (stack_features,save_head,restore_head)
from tools.real_motion.train_p0_f9_surface_ego_head import train_cached


def config():return EgoHeadConfig(object_dim=16,surface_dim=16,width=32,heads=4,layers=1,object_slots=4,surface_side=2)


def feature_row(c=None):
    c=c or config()
    poses=np.repeat(np.eye(4)[None],4,axis=0);poses[:,0,3]=np.arange(4)
    return dict(objects=torch.randn(c.object_slots,c.object_dim),object_geometry=torch.randn(c.object_slots,8),
        object_valid=torch.zeros(c.object_slots,dtype=torch.bool),surfaces=torch.randn(c.surface_side**2,c.surface_dim),
        surface_geometry=torch.randn(c.surface_side**2,3),surface_valid=torch.zeros(c.surface_side**2,dtype=torch.bool),
        ego_history=torch.from_numpy(ego_history_features(poses)))


@pytest.fixture(autouse=True)
def one_thread():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_empty_objects_surfaces_finite_and_absolute_causal_prior():
    head=HistoryEgoTrajectoryHead(config());bank=stack_features([feature_row()],torch.device('cpu'))
    a=head(bank,np.full((1,6),2))
    assert torch.isfinite(a['se2']).all() and a['branches'].shape==(1,6,3,3)
    torch.testing.assert_close(a['se2'][0,:,0],torch.arange(1,7,dtype=torch.float))
    assert a['se2'][0,:,1:].count_nonzero()==0
    bank['future_poses']=torch.zeros(1,6,4,4)
    with pytest.raises(ValueError,match='ONLY'):head(bank,np.full((1,6),2))


@pytest.mark.parametrize('c',[np.full((1,6),3),np.full((1,6),-.1),np.zeros((1,6,3)),np.ones((1,6,3)),np.full((1,6,3),1/3),np.zeros((1,5))])
def test_invalid_commands_fail_closed(c):
    with pytest.raises(ValueError):command_indices(c,batch=1)


def test_navigation_branch_and_periodic_yaw_supervision():
    head=HistoryEgoTrajectoryHead(config());bank=stack_features([feature_row()],torch.device('cpu'))
    out=head(bank,np.eye(3,dtype=np.float32)[[0,1,2,0,1,2]][None]);target=out['se2'].detach().clone()
    target[:,:,2]+=2*np.pi
    loss,_=ego_supervision(out,target);assert abs(loss.item())<1e-6
    target[:,:,0]+=1;loss,_=ego_supervision(out,target);loss.backward()
    assert head.readout.weight.grad.abs().sum()>0
    assert not bank['objects'].requires_grad


def test_se2_roundtrip_t0_tilt_translation_and_non_cumulative_yaw():
    t=np.eye(4);a=.17;b=.7
    rz=np.array([[np.cos(b),-np.sin(b),0],[np.sin(b),np.cos(b),0],[0,0,1]])
    rx=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    t[:3,:3]=rz@rx;t[:3,3]=[3,5,8]
    p=np.c_[np.arange(1,7),np.arange(6)*.3,np.full(6,.4)]
    poses=poses_from_se2(t,p);np.testing.assert_allclose(relative_se2(t,poses),p,atol=1e-6)
    np.testing.assert_array_equal(poses[:,2,3],np.full(6,8))
    np.testing.assert_allclose(poses[:, :3,2],np.broadcast_to(t[:3,2],(6,3)))
    assert relative_se2(t,poses)[-1,2]==pytest.approx(.4) # NOT six times .4.
    raw=dict(history_occ=[],history_observed=[],history_poses=[],future_poses='GT',future_gt_occ=None,
        _column_causal_preparation={'learned_or_future':'must discard'},_waymo_frame_geometry='pure_history')
    clean=replace_future_geometry(raw,poses)
    assert raw['future_poses']=='GT' and '_column_causal_preparation' not in clean
    assert clean['_waymo_frame_geometry']=='pure_history'
    raw['future_gt_occ']=np.zeros(1)
    with pytest.raises(ValueError):replace_future_geometry(raw,poses)


def test_history_ego_uses_actual_known_time_and_no_future_secant():
    poses=np.repeat(np.eye(4)[None],4,axis=0);poses[:,0,3]=[0,.8,2.,3.]
    x=ego_history_features(poses,[0,.4,1.,1.5])
    np.testing.assert_allclose(x[:,5]*10,2.)
    assert x[-1,0]==0 and x[-1,-1]==0
    with pytest.raises(ValueError):ego_history_features(poses,[0,.5,.5,1.])


def test_reject_time_shifted_or_repeated_window_and_corrupted_bank(tmp_path):
    from test_stc_camera_protocol import fixture
    from real_motion.ego_navigation import validate_window_identity
    from tools.real_motion.ego_trajectory_common import row_fingerprint
    source=fixture(tmp_path);w=source.windows[0]
    def get(kind,t):
        if kind=='scene':return dict(name=w.scene)
        r=source.catalog[t];return dict(prev=r.prev,next=r.next,timestamp=r.timestamp,scene_token='s')
    nusc=SimpleNamespace(get=get)
    rec=dict(scene_name=w.scene,t0_token=w.t0,history_tokens=w.history,future_tokens=w.future)
    validate_window_identity(nusc,rec)
    with pytest.raises(ValueError):validate_window_identity(nusc,rec|{'history_tokens':w.history[::-1]})
    with pytest.raises(ValueError):validate_window_identity(nusc,rec|{'future_tokens':w.future[::-1]})
    row=dict(features=feature_row(),target=np.zeros((6,3),np.float32),commands=np.full(6,2),key=[w.scene,w.t0])
    original=row_fingerprint(row);row['features']['ego_history'][0,0]+=1
    assert row_fingerprint(row)!=original


def test_only_selected_mode_receives_output_supervision():
    head=HistoryEgoTrajectoryHead(config());out=head(stack_features([feature_row()],torch.device('cpu')),np.zeros((1,6),np.int64))
    out['branches'].retain_grad();target=out['se2'].detach().clone();target[:,:,0]+=1
    loss,_=ego_supervision(out,target);loss.backward()
    assert out['branches'].grad[:,:,0].abs().sum()>0 and out['branches'].grad[:,:,1:].count_nonzero()==0


def test_official_command_endpoint_padding_and_destination_rows():
    class Nusc:
        def get(self,kind,token):
            i=int(token)
            if kind=='sample':return dict(next=str(i+1) if i<12 else '',data={'LIDAR_TOP':str(i)})
            if kind=='sample_data':return dict(ego_pose_token=str(i),calibrated_sensor_token='0')
            if kind=='ego_pose':return dict(translation=[i*.5,0,0],rotation=[1,0,0,0])
            if kind=='calibrated_sensor':return dict(translation=[0,0,0],rotation=[1,0,0,0])
            raise AssertionError(kind)
    # row 1..6, endpoint >=2 -> right. Tail repeats scene end, not zero/unknown.
    np.testing.assert_array_equal(navigation_commands(Nusc(),[str(i) for i in range(1,7)]),np.zeros(6))
    np.testing.assert_array_equal(navigation_commands(Nusc(),[str(i) for i in range(7,13)]),[0,0,2,2,2,2])


def test_head_adam_rng_exact_resume_and_contract_guard(tmp_path):
    torch.manual_seed(4);head=HistoryEgoTrajectoryHead(config());other=deepcopy(head)
    rows=[dict(features=feature_row(),commands=np.full(6,2),target=np.zeros((6,3),np.float32)) for _ in range(7)]
    contract={'model_sha':'frozen','schedule':'2epochs-batch2','bank':'same'}
    complete=tmp_path/'complete';resume=tmp_path/'resume';complete.mkdir();resume.mkdir()
    args=dict(epochs=2,batch_size=2,lr=3e-4,min_lr=3e-6,seed=13,checkpoint_every=1)
    opt=torch.optim.AdamW(head.parameters(),lr=3e-4);train_cached(head,rows,opt,contract,complete,**args)
    stop=Event();opt2=torch.optim.AdamW(other.parameters(),lr=3e-4);original_step=opt2.step
    def step(*a,**kw):
        x=original_step(*a,**kw);stop.set();return x
    opt2.step=step
    partial=train_cached(other,rows,opt2,contract,resume,stop_event=stop,**args)
    assert partial['status']=='stopped' and partial['updates']==1
    opt2.step=original_step
    train_cached(other,rows,opt2,contract,resume,resume=True,**args)
    for k,v in head.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)
    with pytest.raises(RuntimeError,match='contract'):restore_head(resume/'head_last.pt',other,opt2,contract|{'bank':'different'},np.random.default_rng())
    a=torch.load(complete/'head_last.pt',weights_only=False);b=torch.load(resume/'head_last.pt',weights_only=False)
    assert a['updates']==b['updates']==8 and a['epoch']==b['epoch']==2


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_shared_history_causality_and_frozen_full_geometry(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA unavailable')
    from test_stc_camera_protocol import fixture
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from tools.real_motion.stc_shared_execution import Predictor
    from tools.real_motion.ego_trajectory_common import extract_history_features
    from tools.real_motion import joint_long_rollout_common as rollout
    from real_motion.waymo_native_execution import prepare_waymo_native
    prepare_waymo_native(tmp_path/'native')
    source=fixture(tmp_path,shape=(32,32,4));source.preflight(source.windows)
    joint=JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,blocks=1,decoder_blocks=1),width=16,z_bins=4).to(device).eval().requires_grad_(False)
    before={k:v.clone() for k,v in joint.state_dict().items()}
    pcfg=PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1.,(.4,)*3,source.shape))
    predictor=Predictor(joint,pcfg,device,workers=2,graphs=False,geometry_mib=8)
    try:
        w=source.windows[0];rec,raw=source.prediction_inputs(w,'stc_pred')
        raw['history_occ'][: ,4:10,4:10,1:3]=4
        original=rollout.legacy._strong_all_horizons
        def forbidden(*a,**kw):pytest.fail('history head extractor must NOT run future Strong')
        monkeypatch.setattr(rollout.legacy,'_strong_all_horizons',forbidden)
        a=extract_history_features(predictor.provider,rec,raw,config())
        raw2={**raw,'future_poses':[np.eye(4)*999]*6,'future_gt_occ':np.ones((6,32,32,4),np.uint8)}
        b=extract_history_features(predictor.provider,rec,raw2,config())
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        assert a['surface_valid'].any() and a['object_valid'].any()
        monkeypatch.setattr(rollout.legacy,'_strong_all_horizons',original)
        # Normal route before and after head readout must stay byte-identical.
        _,dense1,prob1,_=predictor.full(rec,raw,verify=False)
        extract_history_features(predictor.provider,rec,raw,config())
        _,dense2,prob2,_=predictor.full(rec,raw,verify=False)
        np.testing.assert_array_equal(prob1,prob2)
        for x,y in zip(dense1,dense2):np.testing.assert_array_equal(x,y)
        shifted=poses_from_se2(raw['history_poses'][-1],np.c_[np.arange(1,7)*.3,np.zeros((6,2))])
        prep,shifted_dense,_,_=predictor.full(rec,replace_future_geometry(raw,shifted),verify=False)
        assert any(not np.array_equal(x,y) for x,y in zip(dense1,shifted_dense))
        for i,p in enumerate(shifted):
            np.testing.assert_array_equal(prep.raw['future_poses'][i],p)
            np.testing.assert_array_equal(prep.state['future_poses'][i],p)
            np.testing.assert_allclose(prep.state['world_to_future'][i],np.linalg.inv(p))
        for key in before:torch.testing.assert_close(before[key],joint.state_dict()[key],rtol=0,atol=0)
    finally:predictor.close()


def test_train_cli_real_frozen_bank_head_only_and_resume(tmp_path,monkeypatch):
    from test_stc_camera_protocol import fixture
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from real_motion.waymo_native_execution import prepare_waymo_native
    from tools.real_motion import train_p0_f9_surface_ego_head as cli
    prepare_waymo_native(tmp_path/'native')
    source=fixture(tmp_path,shape=(32,32,4));source.preflight(source.windows)
    model=JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,blocks=1,decoder_blocks=1),width=16,z_bins=4).eval().requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    records=[dict(scene_name=w.scene,t0_token=w.t0,history_tokens=w.history,future_tokens=w.future) for w in source.windows]
    path=lambda scene,token:source.root/'gts'/scene/token/'labels.npz'
    def load(scene,token):
        with np.load(path(scene,token)) as z:return z['semantics'],z['mask_lidar']
    def metadata(kind,t):
        if kind=='scene':return dict(name=source.windows[0].scene)
        r=source.catalog[t]
        return dict(timestamp=r.timestamp,prev=r.prev,next=r.next,scene_token='scene-id')
    fake_source=SimpleNamespace(allowed_scenes={source.windows[0].scene},_label_path=path,load_occ3d=load,
        pose=lambda t:source.catalog[t].pose,nusc=SimpleNamespace(get=metadata))
    monkeypatch.setattr(cli,'load_cache',lambda *_:({},records))
    monkeypatch.setattr(cli,'NuScenesWindowSource',lambda *a,**kw:fake_source)
    monkeypatch.setattr(cli,'navigation_commands',lambda *_:np.full(6,2))
    monkeypatch.setattr(cli,'load_evaluation_model',lambda *a,**kw:(dict(source_epochs=list(cli.AVERAGE_EPOCHS),averaging=True),model))
    pcfg=PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1.,(.4,)*3,source.shape))
    monkeypatch.setattr(cli,'make_prepare_config',lambda *_:pcfg);monkeypatch.setattr(cli,'load_runtime_config',lambda *_:{})
    inputs=tmp_path/'inputs';inputs.mkdir()
    checkpoint=inputs/'mean.pt';checkpoint.write_bytes(b'frozen checkpoint')
    cache=inputs/'train.pt';cache.write_bytes(b'TRAIN identity')
    info=inputs/'info.pkl';info.write_bytes(b'TRAIN metadata')
    cfg=inputs/'config.yaml';cfg.write_text('config')
    out=tmp_path/'head_screen'
    args=['--dataroot',str(source.root),'--train-cache',str(cache),'--train-info',str(info),
          '--checkpoint',str(checkpoint),'--out-dir',str(out),'--config',str(cfg),'--device','cpu',
          '--train-windows','2','--epochs','2','--batch-size','2','--cpu-workers','2']
    assert cli.main(argv=args)==0
    saved=torch.load(out/'head_last.pt',weights_only=False)
    assert saved['epoch']==2 and saved['updates']==2 and saved['config']['history_frames']==4
    assert 'transport' not in saved['state_dict'] and saved['contract']['command_condition'].startswith('GT-derived')
    for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    assert checkpoint.read_bytes()==b'frozen checkpoint'
    assert cli.main(argv=[*args,'--resume'])==0
    with pytest.raises(RuntimeError,match='contract'):cli.main(argv=[*args,'--resume','--epochs','3'])


def test_full_ego_eval_reads_labels_after_predictions_atomic_resume(tmp_path):
    from test_stc_camera_protocol import fixture
    from tools.real_motion.eval_p0_f9_surface_ego_head import evaluate
    source=fixture(tmp_path);source.preflight(source.windows)
    head=HistoryEgoTrajectoryHead(config());calls=[]
    class Predict:
        provider=None
        def full(self,rec,raw,verify):
            assert raw['future_gt_occ'] is None
            calls.append(('prediction',rec['t0_token']))
            return None,[raw['history_occ'][-1].copy() for _ in range(6)],None,{}
    original=source.metric_targets
    def targets(w):
        assert len([x for x in calls if x==('prediction',w.t0)])>=6
        calls.append(('targets',w.t0));return original(w)
    source.metric_targets=targets
    features=lambda *a,**kw:feature_row()
    contract=dict(modalities=['occ','stc'],model='frozen',windows=len(source.windows))
    stop=Event();saved={}
    partial=evaluate(source,source.windows,Predict(),head,lambda _:np.full(6,2),contract,
        feature_extractor=features,stop_event=stop,save=lambda s:saved.update(s),progress=lambda _:stop.set())
    assert partial['state']['status']=='stopped' and saved['completed_windows']==1
    resumed=evaluate(source,source.windows,Predict(),head,lambda _:np.full(6,2),contract,
        feature_extractor=features,saved=saved)
    full=evaluate(source,source.windows,Predict(),head,lambda _:np.full(6,2),contract,feature_extractor=features)
    assert resumed['reports']==full['reports'] and full['state']['status']=='complete'
    with pytest.raises(RuntimeError,match='contract'):evaluate(source,source.windows,Predict(),head,
        lambda _:np.full(6,2),contract|{'model':'changed'},feature_extractor=features,saved=saved)
