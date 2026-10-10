from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_surface_ego_head import config,feature_row
from tools.real_motion.ego_trajectory_common import stack_features,row_fingerprint,atomic_save,digest_file,fingerprint
from real_motion.ego_trajectory_head import HistoryEgoTrajectoryHead,ego_supervision
from tools.real_motion.surface_ego_ablation_common import (
    PROTOCOL,NEW_PYTHON_FILES,TRAIN_NAMES,geometry_loss,historical_prior,paired_train,
    recover_original_initialization,tensor_state_fingerprint,load_source_bank,trajectory_report,
    unchanged_legacy_implementation,
)


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def rows(n=7):
    return [dict(features=feature_row(),commands=np.full(6,i%3,np.int64),
                 target=np.c_[np.arange(1,7)*.2,np.zeros(6),np.arange(1,7)*.01].astype(np.float32)) for i in range(n)]


def test_geometry_loss_translation_matches_original_xy_and_periodic_yaw():
    x=torch.zeros(2,6,3,requires_grad=True);target=torch.zeros_like(x)
    target[...,0]=.5;target[...,1]=1.5
    geometry=geometry_loss({'se2':x},target)
    original,_=ego_supervision({'se2':x},target)
    torch.testing.assert_close(geometry,original)
    target=torch.zeros_like(x);target[...,2]=2*np.pi
    assert geometry_loss({'se2':x},target).item()<1e-10
    target[...,2]=.05;loss=geometry_loss({'se2':x},target);loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad[...,2].abs().sum()>0
    # At small angles the fixed radius gives meaningful geometric angular cost.
    periodic,_=ego_supervision({'se2':x},target)
    assert loss>periodic*30


@pytest.mark.parametrize('radius',[0,-1,float('nan'),float('inf')])
def test_geometry_radius_fail_closed(radius):
    with pytest.raises(ValueError):geometry_loss({'se2':torch.zeros(1,6,3)},torch.zeros(1,6,3),radius_m=radius)


def test_prior_is_exact_original_zero_readout_and_has_no_nav_or_future_input():
    head=HistoryEgoTrajectoryHead(config());bank=stack_features([feature_row()],torch.device('cpu'))
    a=head(bank,np.full((1,6),0))['se2'];b=head(bank,np.full((1,6),2))['se2']
    torch.testing.assert_close(a,historical_prior(bank),rtol=0,atol=0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_original_initializer_reconstructed_and_rng_restored():
    cfg=config();seed=21
    def loader(*a,**kw):
        return dict(source_epochs=[5,6,8,12,14],averaging=True),torch.nn.Linear(5,7)
    torch.manual_seed(seed);loader('frozen');original=HistoryEgoTrajectoryHead(cfg);witness=torch.get_rng_state()
    saved=dict(config=cfg.__dict__,contract=dict(schedule=dict(seed=seed)),torch_rng=witness)
    torch.manual_seed(9);before=torch.get_rng_state().clone()
    actual=recover_original_initialization(saved,'frozen',loader=loader)
    assert tensor_state_fingerprint(actual.state_dict())==tensor_state_fingerprint(original.state_dict())
    assert torch.equal(before,torch.get_rng_state())
    saved['torch_rng']=before
    with pytest.raises(RuntimeError,match='RNG witness'):recover_original_initialization(saved,'frozen',loader=loader)


def assert_nested_equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:assert_nested_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested_equal(x,y)
    else:assert a==b


def test_paired_lockstep_resume_adam_cursor_monitors_and_complete_readonly(tmp_path):
    torch.manual_seed(2);initial=HistoryEgoTrajectoryHead(config());data=rows();old=deepcopy(initial).eval().requires_grad_(False)
    old_sha=tensor_state_fingerprint(old.state_dict());initial_sha=tensor_state_fingerprint(initial.state_dict())
    complete,resume=[tmp_path/k for k in ('complete','resume')]
    complete.mkdir();resume.mkdir();contract=dict(source='read only',budget=8)
    kw=dict(max_updates=8,batch_size=2,monitor_every=2,checkpoint_every=1,seed=21)
    result=paired_train(initial,data,old,contract,complete,**kw)
    stop=Event();original_step=torch.optim.AdamW.step;calls=[]
    def step(opt,*a,**kw):
        r=original_step(opt,*a,**kw);calls.append(1)
        if len(calls)==1:stop.set() # must still finish B in the same update.
        return r
    from unittest.mock import patch
    with patch.object(torch.optim.AdamW,'step',step):
        partial=paired_train(initial,data,old,contract,resume,stop_event=stop,**kw)
    assert partial['updates']==1 and partial['status']=='stopped' and len(calls)==2
    paired_train(initial,data,old,contract,resume,resume=True,**kw)
    a=torch.load(complete/'pair_last.pt',weights_only=False);b=torch.load(resume/'pair_last.pt',weights_only=False)
    for key in ('models','optimizers','updates','epoch','cursor','order','sampling_rng','epoch_sums','monitors'):
        assert_nested_equal(a[key],b[key])
    assert result['TRAIN_reports']==b['monitors'][-1]['reports']
    assert a['initial_weights']==initial_sha and tensor_state_fingerprint(old.state_dict())==old_sha
    # No mutation of the supplied initial or old learned head.
    assert tensor_state_fingerprint(initial.state_dict())==initial_sha
    digest=digest_file(resume/'pair_last.pt')
    paired_train(initial,data,old,contract,resume,resume=True,**kw)
    assert digest_file(resume/'pair_last.pt')==digest
    with pytest.raises(RuntimeError,match='contract'):
        paired_train(initial,data,old,contract|{'budget':9},resume,resume=True,**kw)
    logs=[json.loads(s) for s in (complete/'progress.jsonl').read_text().splitlines()]
    means=[r for r in logs if r['event']=='paired_epoch_mean']
    assert len(means)==2 and all(r['windows']==len(data) for r in means)


def test_mid_epoch_fixed_update_budget_and_corrupt_cursor_rejection(tmp_path):
    initial=HistoryEgoTrajectoryHead(config());old=deepcopy(initial).eval();data=rows(5)
    out=tmp_path/'pair';out.mkdir();kw=dict(max_updates=2,batch_size=2,monitor_every=10)
    result=paired_train(initial,data,old,{'contract':'same'},out,**kw)
    assert result['cursor']==4 and result['completed_epochs']==0 and result['last_TRAIN_monitor_update']==2
    saved=torch.load(out/'pair_last.pt',weights_only=False);saved['cursor']=1
    atomic_save(out/'pair_last.pt',saved)
    with pytest.raises(RuntimeError,match='cursor mismatch'):
        paired_train(initial,data,old,{'contract':'same'},out,resume=True,**kw)


def test_epoch_mean_is_sample_weighted_not_last_batch(tmp_path,monkeypatch):
    from tools.real_motion import surface_ego_ablation_common as common
    def objective(output,target,**kw):
        loss=output['se2'].sum()*0+len(target)
        return loss,dict(xy=loss.detach(),yaw=loss.detach()*0)
    monkeypatch.setattr(common,'ego_supervision',objective)
    monkeypatch.setattr(common,'geometry_loss',lambda output,target,**kw:objective(output,target)[0])
    initial=HistoryEgoTrajectoryHead(config());old=deepcopy(initial)
    out=tmp_path/'pair';out.mkdir()
    paired_train(initial,rows(5),old,{'science':'same'},out,max_updates=3,batch_size=2)
    logs=[json.loads(s) for s in (out/'progress.jsonl').read_text().splitlines()]
    means=next(r for r in logs if r['event']=='paired_epoch_mean')
    for name in TRAIN_NAMES:assert means['means'][name]['loss']==pytest.approx(1.8)


def test_training_cli_complete_resume_no_retrain_or_source_writes(tmp_path,monkeypatch):
    from tools.real_motion import train_p0_f9_surface_ego_ablation as cli
    initial=HistoryEgoTrajectoryHead(config());data=rows(1)*1024
    source=tmp_path/'source';source.mkdir();out=tmp_path/'pair'
    saved=dict(config=config().__dict__,state_dict=initial.state_dict(),updates=320,
        contract=dict(schedule=dict(batch_size=64,epochs=20,lr=3e-4,min_lr=3e-6,seed=21)))
    audit=dict(frozen_checkpoint='read only',old='source')
    monkeypatch.setattr(cli,'load_source_bank',lambda *a:(data,saved,audit))
    monkeypatch.setattr(cli,'recover_original_initialization',lambda *a:deepcopy(initial))
    monkeypatch.setattr(cli,'verify_sources',lambda a:None)
    args=['--source-dir',str(source),'--out-dir',str(out),'--max-updates','2','--device','cpu']
    assert cli.main(argv=args)==0
    result=json.loads((out/'training_summary.json').read_text());assert result['updates']==2
    digest=digest_file(out/'pair_last.pt')
    assert cli.main(argv=[*args,'--resume'])==0
    assert digest==digest_file(out/'pair_last.pt')
    with pytest.raises(RuntimeError,match='contract'):
        cli.main(argv=[*args,'--resume','--max-updates','3'])
    assert list(source.iterdir())==[]


def source_artifacts(tmp_path,n=3):
    root=tmp_path/'repo';(root/'tools/real_motion').mkdir(parents=True);(root/'real_motion/native').mkdir(parents=True)
    feature=root/'real_motion/features.py';feature.write_text('immutable feature equations\n')
    for name in NEW_PYTHON_FILES:(root/name).write_text('new ablation only\n')
    old_fingerprint=fingerprint({'real_motion/features.py':digest_file(feature)})
    source=tmp_path/'source';(source/'bank').mkdir(parents=True)
    frozen=tmp_path/'frozen.pt';frozen.write_bytes(b'old WM/CCR')
    head=HistoryEgoTrajectoryHead(config());data=rows(n)
    training=dict(keys=[['scene',str(i)] for i in range(n)],head_config=config().__dict__,
        schedule=dict(batch_size=2,epochs=2,seed=21),frozen_checkpoint=str(frozen),frozen_sha256=digest_file(frozen),
        feature_geometry_implementation=old_fingerprint,implementation={'real_motion/features.py':digest_file(feature)})
    (source/'training.json').write_text(json.dumps(training))
    atomic_save(source/'head_last.pt',dict(protocol='surface_frozen_history_ego_se2_cmd_v1',contract=training,
        config=config().__dict__,epoch=2,cursor=0,updates=4,order=[],state_dict=head.state_dict()))
    for i,row in enumerate(data):
        row.update(key=training['keys'][i],contract_fingerprint=fingerprint(training))
        row['content_fingerprint']=row_fingerprint(row);atomic_save(source/'bank'/f'{i:06d}.pt',row)
    return root,source,feature


def test_legacy_bank_reuse_excludes_only_new_files_and_never_writes(tmp_path):
    root,source,feature=source_artifacts(tmp_path)
    before={p:digest_file(p) for p in source.rglob('*') if p.is_file()}
    data,saved,audit=load_source_bank(source,root)
    assert len(data)==3 and saved['updates']==4 and audit['only_added_files_excluded']==list(NEW_PYTHON_FILES)
    assert before=={p:digest_file(p) for p in before}
    feature.write_text('changed old geometry\n')
    with pytest.raises(RuntimeError,match='legacy feature'):load_source_bank(source,root)


def test_bank_corruption_order_and_unrelated_unexcluded_code_fail_closed(tmp_path):
    root,source,feature=source_artifacts(tmp_path)
    path=source/'bank/000001.pt';row=torch.load(path,weights_only=False);row['features']['ego_history'][0,0]+=1
    atomic_save(path,row)
    with pytest.raises(RuntimeError,match='shard'):load_source_bank(source,root)
    (root/'real_motion/other.py').write_text('also included\n')
    training=json.loads((source/'training.json').read_text())
    with pytest.raises(RuntimeError,match='legacy feature'):unchanged_legacy_implementation(root,training)


def test_trajectory_report_all_horizons_commands_and_periodic_angles():
    target=np.zeros((2,6,3));pred=target.copy();pred[...,0]=2;pred[...,2]=2*np.pi
    report=trajectory_report(pred,target,np.array([[0]*6,[2]*6]))
    assert report['ADE_m']==2 and report['FDE_3s_m']==2
    assert max(report['yaw_mean_deg'])<1e-10 and report['command_counts_by_horizon']==[[1,0,1]]*6


def evaluation_models():
    base=HistoryEgoTrajectoryHead(config())
    models={k:deepcopy(base) for k in ('old320',*TRAIN_NAMES)}
    with torch.no_grad():models['B_geometry'].readout.bias[0]=.2
    return models


def test_dev64_shared_feature_once_per_modality_all_predictions_before_gt_atomic_resume(tmp_path):
    from test_stc_camera_protocol import fixture
    from tools.real_motion.eval_p0_f9_surface_ego_ablation import evaluate
    source=fixture(tmp_path);source.preflight(source.windows);models=evaluation_models();calls=[];features=[]
    class Predict:
        provider=None
        def full(self,rec,raw,verify):
            assert raw['future_gt_occ'] is None
            calls.append(rec['t0_token'])
            return None,[raw['history_occ'][-1].copy() for _ in range(6)],None,{}
    def extract(*a,**kw):
        features.append(1);return feature_row()
    original=source.metric_targets
    def targets(w):
        # old320 and A are exactly the same prior: redundant pose routes reused.
        assert calls.count(w.t0)>=8
        return original(w)
    source.metric_targets=targets
    contract=dict(modalities=['occ','stc'],windows=len(source.windows),old='fixed')
    saved={};stop=Event()
    partial=evaluate(source,source.windows,Predict(),models,lambda _:np.full(6,2),contract,
        feature_extractor=extract,stop_event=stop,progress=lambda _:stop.set(),save=lambda s:saved.update(s))
    assert partial['state']['completed_windows']==1 and len(features)==2 and len(partial['reports'])==12
    assert saved['identical_pose_reuses']==4
    resumed=evaluate(source,source.windows,Predict(),models,lambda _:np.full(6,2),contract,
        feature_extractor=extract,saved=saved)
    full=evaluate(source,source.windows,Predict(),models,lambda _:np.full(6,2),contract,feature_extractor=extract)
    assert resumed['reports']==full['reports'] and resumed['trajectory']==full['trajectory']
    with pytest.raises(RuntimeError,match='fingerprint'):
        evaluate(source,source.windows,Predict(),models,lambda _:np.full(6,2),contract|{'old':'changed'},saved=saved)


def test_partial_window_error_does_not_commit_any_counts(tmp_path):
    from test_stc_camera_protocol import fixture
    from tools.real_motion.eval_p0_f9_surface_ego_ablation import evaluate
    source=fixture(tmp_path);source.preflight(source.windows);saved={};calls=[]
    class Fail:
        provider=None
        def full(self,rec,raw,verify):
            calls.append(1)
            if len(calls)==6:raise RuntimeError('middle of window')
            return None,[raw['history_occ'][-1] for _ in range(6)],None,{}
    with pytest.raises(RuntimeError,match='middle'):
        evaluate(source,source.windows,Fail(),evaluation_models(),lambda _:np.full(6,2),
            dict(modalities=['occ','stc']),feature_extractor=lambda *a,**kw:feature_row(),save=lambda s:saved.update(s))
    assert saved['completed_windows']==0 and sum(np.asarray(v).sum() for v in saved['counts'].values())==0


def test_eval_original_prior_old_geometry_scores_match_existing_entry(tmp_path):
    from test_stc_camera_protocol import fixture
    from tools.real_motion.eval_p0_f9_surface_ego_ablation import evaluate
    from tools.real_motion.eval_p0_f9_surface_ego_head import evaluate as original_evaluate
    source=fixture(tmp_path);source.preflight(source.windows);models=evaluation_models()
    feature=feature_row()
    class Predictor:
        provider=None
        def full(self,rec,raw,verify):
            grid=raw['history_occ'][-1].copy()
            if raw['future_poses'][-1][1,3]>.2:grid[0,0,0]=17
            return None,[grid]*6,None,{}
    p=Predictor();extract=lambda *a,**kw:feature
    new=evaluate(source,source.windows,p,models,lambda _:np.full(6,2),dict(modalities=['occ','stc']),feature_extractor=extract)
    old=original_evaluate(source,source.windows,p,models['old320'],lambda _:np.full(6,2),dict(modalities=['occ','stc']),feature_extractor=extract)
    for modality in ('occ','stc'):
        for a,b in (('gt','gt'),('external','external'),('old320','internal')):
            assert new['reports'][modality+'_'+a]==old['reports'][modality+'_'+b]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_frozen_wm_all_ego_routes_geometry_and_readonly(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA unavailable')
    from test_stc_camera_protocol import fixture
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from real_motion.waymo_native_execution import prepare_waymo_native
    from tools.real_motion.stc_shared_execution import Predictor
    from tools.real_motion.eval_p0_f9_surface_ego_ablation import evaluate
    prepare_waymo_native(tmp_path/'native')
    source=fixture(tmp_path,shape=(32,32,4));source.preflight(source.windows)
    joint=JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,blocks=1,decoder_blocks=1),width=16,z_bins=4).to(device).eval().requires_grad_(False)
    before=tensor_state_fingerprint(joint.state_dict())
    predictor=Predictor(joint,PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1.,(.4,)*3,source.shape)),device,workers=2,graphs=False,geometry_mib=8)
    models={k:v.to(device) for k,v in evaluation_models().items()}
    actual=predictor.full;poses=[]
    def full(rec,raw,verify):
        prep,dense,prob,stages=actual(rec,raw,verify=verify)
        for i,p in enumerate(raw['future_poses']):np.testing.assert_allclose(prep.state['world_to_future'][i],np.linalg.inv(p))
        poses.append(np.asarray(raw['future_poses']))
        return prep,dense,prob,stages
    predictor.full=full
    try:
        result=evaluate(source,source.windows[:1],predictor,models,lambda _:np.full(6,2),dict(modalities=['occ','stc']))
        assert result['state']['status']=='complete' and len(result['reports'])==12 and len(poses)==8
        assert tensor_state_fingerprint(joint.state_dict())==before
    finally:predictor.close()
