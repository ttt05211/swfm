from copy import deepcopy
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
from real_motion.stc_trajectory_yaw import tangent_yaw, rz, yaw, wrap, yaw_error_deg
from real_motion.waymo_i2world import fingerprint
from tools.real_motion.eval_p0_f9_stc_trajectory_yaw import ROUTES, evaluate, restore, summary


def pose(x=0., y=0., angle=0., tilt=.03):
    p=np.eye(4); c,s=np.cos(tilt),np.sin(tilt)
    p[:3,:3]=rz(angle) @ np.array([[c,0,s],[0,1,0],[-s,0,c]])
    p[:3,3]=[x,y,7.]; return p


def inputs():
    t=np.arange(-3,1)*.5
    history=[pose(2*v) for v in t]
    planned=[pose(2*v,angle=np.deg2rad(5)) for v in np.arange(1,7)*.5]
    return history,t,planned


def test_correct_yaw_preserve_xyz_body_tilt_so3_and_no_input_mutation():
    h,t,p=inputs(); original=np.asarray(p).copy()
    result,a=tangent_yaw(h,t,p)
    assert a['corrected']==6 and np.allclose([yaw(v) for v in result],0,atol=1e-12)
    assert np.array_equal(result[:,:,3],original[:,:,3])
    assert np.array_equal(result[:,3],original[:,3])
    for old,new in zip(p,result):
        assert np.allclose(rz(-yaw(old))@old[:3,:3],rz(-yaw(new))@new[:3,:3],atol=1e-12)
        assert np.allclose(new[:3,:3].T@new[:3,:3],np.eye(3),atol=1e-12)
        assert np.linalg.det(new[:3,:3])==pytest.approx(1)
    assert np.array_equal(np.asarray(p),original)


@pytest.mark.parametrize('case',['stationary','reverse','large_yaw','zigzag'])
def test_unreliable_paths_fallback_exact_bytes(case):
    h,t,p=inputs()
    if case=='stationary': h=[pose() for _ in h]
    if case=='reverse': h=[pose(-2*v) for v in t]
    if case=='large_yaw': p=[pose(i+1,angle=np.deg2rad(80)) for i in range(6)]
    if case=='zigzag': p=[pose(i+1, y=(-1)**i*5., angle=np.deg2rad(5)) for i in range(6)]
    result,a=tangent_yaw(h,t,p)
    assert result.tobytes()==np.asarray(p).tobytes() and a['corrected']==0


def test_measured_history_clock_quadratic_tangent_no_half_step_bias_and_global_rotation():
    t=np.array([-1.48,-.99,-.51,0.]); angle=np.deg2rad(179)
    rotation=rz(angle)[:2,:2]; origin=np.array([876543.,345678.])
    def make(v,error):
        point=origin+rotation@np.array([3*v,.1*v*v])
        return pose(*point,angle=angle+np.arctan2(.2*v,3)+error)
    h=[make(v,0) for v in t]; planned=[make(v,np.deg2rad(4)) for v in np.arange(1,7)*.5]
    out,a=tangent_yaw(h,t,planned)
    assert a['corrected']==6
    for v,p in zip(np.arange(1,7)*.5,out):
        assert abs(float(wrap(yaw(p)-angle-np.arctan2(.2*v,3))))<1e-8
    assert yaw_error_deg([pose(angle=np.deg2rad(179))]*6,[pose(angle=np.deg2rad(-179))]*6)==pytest.approx([2]*6)


def test_invalid_times_pose_lengths_fail_closed():
    h,t,p=inputs()
    for times in ([0,0,1,2],[0,1,2,3],[0,.5,1,np.nan]):
        with pytest.raises(ValueError): tangent_yaw(h,times,p)
    with pytest.raises(ValueError): tangent_yaw(h[:3],t,p)
    p[0][0,0]=3
    with pytest.raises(ValueError): tangent_yaw(h,t,p)


class Source:
    shape=(4,4,2)
    def __init__(self,unchanged=False):
        h,t,p=inputs()
        if unchanged: p=[pose(i+1) for i in range(6)]
        self.history=tuple('h'+str(i) for i in range(4)); self.future=tuple('f'+str(i) for i in range(6))
        self.catalog={k:SimpleNamespace(pose=v,timestamp=int((time+2)*1e6)) for k,v,time in zip(self.history,h,t)}
        self.catalog.update({k:SimpleNamespace(pose=pose(i+1),timestamp=int((i*.5+2.5)*1e6)) for i,k in enumerate(self.future)})
        self.raw=dict(history_poses=h,future_poses=p,future_gt_occ=None,
            history_occ=np.full((4,*self.shape),17,np.uint8),history_observed=np.ones((4,*self.shape),bool))
        self.windows=[SimpleNamespace(key=str(i),history=self.history,future=self.future,t0=self.history[-1]) for i in range(2)]
        self.calls=0; self.reads=0; self.expected=2 if unchanged else 4
    def prediction_inputs(self,w,setting):
        assert setting in ('occ_pred','stc_pred')
        return dict(scene_name='s',t0_token=w.t0),deepcopy(self.raw)
    def metric_targets(self,w):
        assert self.calls==(self.reads+1)*self.expected
        self.reads+=1; return [np.full(self.shape,17,np.uint8)]*3


class Predictor:
    def __init__(self,source,fail=None): self.source=source; self.fail=fail; self.seen=[]
    def full(self,rec,raw,verify):
        self.source.calls+=1
        if self.source.calls==self.fail: raise RuntimeError('fixture failure')
        self.seen.append(deepcopy(raw))
        return None,[np.full(self.source.shape,17,np.uint8)]*6,None,dict(head=.001)


def test_atomic_resume_gt_boundary_and_all_future_truth_changes_leave_predictions_unchanged():
    source=Source(); contract=dict(windows=2,weights='fixed'); saved={}; stop=Event()
    first=evaluate(source,source.windows,Predictor(source),contract,save=lambda s:saved.update(deepcopy(s)),
                   stop_event=stop,progress=lambda _:stop.set())
    assert first['status']=='stopped' and first['completed_windows']==1
    continued=evaluate(source,source.windows,Predictor(source),contract,saved=saved)
    other=Source(); full=evaluate(other,other.windows,Predictor(other),contract)
    assert continued['reports']==full['reports'] and continued['audits']==full['audits']
    assert 'no training' in summary(full) and full['audits'][0]['forecast_calls']==4
    with pytest.raises(RuntimeError): restore(saved,{**contract,'weights':'different'},source.shape)
    bad=deepcopy(saved); bad.pop('fingerprint'); bad['counts']['b_occ_pred'][0][17][17]+=1
    bad['fingerprint']=fingerprint(bad)
    with pytest.raises(ValueError): restore(bad,contract,source.shape)
    changed=Source()
    for token in changed.future: changed.catalog[token].pose=pose(999,999,1.5)
    predictor=Predictor(changed); evaluate(changed,changed.windows,predictor,contract)
    clean=Source(); original=Predictor(clean); evaluate(clean,clean.windows,original,contract)
    assert all(np.array_equal(a['future_poses'],b['future_poses']) for a,b in zip(predictor.seen,original.seen))


def test_failure_has_no_partial_scoring_and_consistent_yaw_skips_two_forwards():
    source=Source(); saved={}
    with pytest.raises(RuntimeError,match='fixture failure'):
        evaluate(source,source.windows,Predictor(source,fail=3),dict(windows=2),save=lambda s:saved.update(deepcopy(s)))
    assert saved['completed_windows']==0 and source.reads==0
    source=Source(unchanged=True)
    result=evaluate(source,source.windows,Predictor(source),dict(windows=2))
    assert all(a['unchanged_reuses']==2 for a in result['audits']) and source.calls==4
    assert result['reports']['b_occ_pred']==result['reports']['yaw_occ_pred']


def test_unchanged_first_window_does_not_skip_later_real_candidate_exactness():
    source=Source(unchanged=True); source.expected=2
    predictor=Predictor(source); seen=[]; full=predictor.full
    def capture(rec,raw,verify):
        seen.append(verify); return full(rec,raw,verify)
    predictor.full=capture
    def next_window(_):
        source.raw['future_poses']=inputs()[2]
        # First window used two forwards; second uses four.
        source.expected=3
    result=evaluate(source,source.windows,predictor,dict(windows=2),progress=next_window)
    assert result['status']=='complete'
    assert seen==[True,True,False,True,False,True]


def test_future_truth_field_rejected_even_when_candidate_is_reused():
    source=Source(unchanged=True); source.raw['future_gt_occ']=np.zeros(source.shape)
    with pytest.raises(RuntimeError,match='future occupancy'):
        evaluate(source,source.windows,Predictor(source),dict(windows=2))
    assert source.calls==0 and source.reads==0


def test_real_model_reference_bytes_changed_yaw_and_weights_unchanged(tmp_path):
    import torch
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from real_motion.waymo_native_execution import prepare_waymo_native
    from test_stc_camera_protocol import fixture
    from tools.real_motion.stc_shared_execution import Predictor as RealPredictor, bytes_equal
    prepare_waymo_native(tmp_path/'native')
    source=fixture(tmp_path,shape=(32,32,4)); window=source.windows[0]
    source.preflight([window])
    pcfg=PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1.,(.4,)*3,source.shape))
    model=JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,blocks=1,
        decoder_blocks=1),width=16,z_bins=4).eval().requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    predictor=RealPredictor(model,pcfg,'cpu',workers=1,graphs=False)
    try:
        rec,raw=source.prediction_inputs(window,'stc_pred')
        h,t,p=inputs(); raw={**raw,'history_poses':h,'future_poses':p}
        original=predictor.full(rec,dict(raw),verify=True)
        changed,_=tangent_yaw(h,t,p)
        candidate=predictor.full(rec,{**raw,'future_poses':changed},verify=True)
        repeated=predictor.full(rec,dict(raw),verify=True)
        for a,b in zip(original[1],repeated[1]): bytes_equal(a,b,'unchanged original six frames')
        bytes_equal(original[2],repeated[2],'unchanged original probabilities')
        assert len(candidate[1])==6
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    finally: predictor.close(); torch.set_num_threads(threads)


def test_resume_launcher_environment_and_only_dev64_cli():
    from pathlib import Path
    from tools.real_motion.eval_p0_f9_stc_trajectory_yaw import main
    script=(Path(__file__).resolve().parents[1]/'tools/real_motion/run_p0_f9_stc_trajectory_yaw.sh').read_text(encoding='utf-8')
    assert 'STC_YAW_RESUME' in script and 'eval_p0_f9_stc_trajectory_yaw.py' in script
    assert "recorded = json.load(handle)['runtime_environment']" in script
    assert 'os.execve' in script and 'eval ' not in script
    with pytest.raises(SystemExit):
        main(argv=['--dataroot','x','--stc-root','y','--plan-cache','z','--out-dir','w','--population','all'])


def test_cli_output_contract_restore_and_source_checkpoint_readonly(tmp_path,monkeypatch):
    import json
    from test_stc_camera_protocol import fixture
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.waymo_i2world import file_sha256
    from tools.real_motion import eval_p0_f9_stc_trajectory_yaw as cli
    source=fixture(tmp_path); windows=source.windows[:2]
    monkeypatch.setattr(source,'select',lambda *a:(windows,dict(population='dev64')))
    monkeypatch.setattr(cli.base.STCFourSettingSource,'from_files',lambda *a,**kw:source)
    monkeypatch.setattr(cli.base,'load_manifest',lambda *a:(dict(parent_keys=[1]),None,None))
    monkeypatch.setattr(cli.base,'load_runtime_config',lambda *a:None)
    cfg=PrepareConfig(grid=OccupancyGrid(-40,-40,-1.,(.4,)*3,source.shape))
    monkeypatch.setattr(cli.base,'make_prepare_config',lambda *a:cfg)
    monkeypatch.setattr(cli.base,'SHAPE',source.shape)
    monkeypatch.setattr(cli.torch.cuda,'is_available',lambda:True)
    model=SimpleNamespace(transport=SimpleNamespace(config=SimpleNamespace(history_frames=4)))
    monkeypatch.setattr(cli.base,'load_evaluation_model',lambda *a,**kw:
        (dict(source_epochs=list(cli.base.AVERAGE_EPOCHS),averaging=True),model))
    class ModelFixture:
        def __init__(self,*a,**kw): pass
        def close(self): pass
        def full(self,rec,raw,verify=False):
            return None,[raw['history_occ'][-1]]*6,None,{}
    monkeypatch.setattr(cli,'Predictor',ModelFixture)
    checkpoint=tmp_path/'weights/mean.pt'; checkpoint.parent.mkdir(); checkpoint.write_text('fixture')
    config=tmp_path/'config.yaml'; config.write_text('fixture')
    manifest=tmp_path/'manifest.json'; manifest.write_text('fixture')
    digest=file_sha256(checkpoint); out=tmp_path/'new-yaw-screen'
    common=['--dataroot',str(source.root),'--stc-root',str(source.stc_root),'--plan-cache',str(source.plan_cache),
        '--population','dev64','--population-manifest',str(manifest),'--config',str(config),'--checkpoint',str(checkpoint)]
    args=common+['--out-dir',str(out)]
    assert cli.main(argv=args)==0
    first=json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
    assert cli.main(argv=args+['--resume'])==0
    assert json.loads((out/'evaluation.json').read_text(encoding='utf-8'))['reports']==first['reports']
    assert file_sha256(checkpoint)==digest
    with pytest.raises(RuntimeError,match='contract'): cli.main(argv=args+['--resume','--cpu-workers','2'])
    with pytest.raises(SystemExit): cli.main(argv=args)
    with pytest.raises(SystemExit): cli.main(argv=common+['--out-dir',str(source.stc_root/'unsafe')])
