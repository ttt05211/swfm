from copy import deepcopy
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest

from real_motion.geometry import OccupancyGrid
from real_motion.stc_causal_geometry import (stabilize_stc_ground, compensate_planned_ground_pose,
    ground_points, fit_ground, motion_jitter_audit)
from tools.real_motion.eval_p0_f9_stc_causal_geometry import ROUTES, evaluate, restore, pose_errors, summary


def raw_scene(shape=(16,16,8)):
    grid=OccupancyGrid(-8,-8,-2,(1,1,.4),shape)
    sem=np.full((4,*shape),17,np.uint8); sem[:,:, :,3]=11
    poses=[np.eye(4) for _ in range(4)]
    return grid,dict(history_occ=sem,history_observed=np.ones_like(sem,bool),
        history_poses=poses,future_poses=[np.eye(4) for _ in range(6)],future_gt_occ=None)


def test_ground_two_histories_shift_one_cell_preserve_dynamic_source_and_inputs():
    grid,raw=raw_scene(); raw['history_occ'][3,:,:,3]=17; raw['history_occ'][3,:,:,4]=11
    raw['history_occ'][:,3:5,3:5,5]=4
    before=raw['history_occ'].copy(); masks=raw['history_observed'].copy()
    changed,audit=stabilize_stc_ground(raw,grid)
    assert (changed['history_occ'][3,:,:,3]==11).all()
    assert (changed['history_occ'][:,3:5,3:5,5]==4).all()
    assert audit['shifted_columns']==256
    assert np.array_equal(raw['history_occ'],before)
    assert np.array_equal(changed['history_observed'],masks)
    assert changed['future_poses'] is raw['future_poses']


def test_ground_no_free_votes_no_unobserved_invention_no_far_height_snap():
    grid,raw=raw_scene(); raw['history_occ'][:3]=17
    changed,audit=stabilize_stc_ground(raw,grid)
    assert np.array_equal(changed['history_occ'],raw['history_occ']) and audit['changed_voxels']==0
    grid,raw=raw_scene(); raw['history_occ'][3]=17; raw['history_occ'][3,:,:,6]=13
    changed,_=stabilize_stc_ground(raw,grid)
    assert np.array_equal(changed['history_occ'][3],raw['history_occ'][3])


def test_ground_relabel_requires_agreement_and_destination_collision_rejected():
    grid,raw=raw_scene(); raw['history_occ'][3,:,:,3]=13
    changed,audit=stabilize_stc_ground(raw,grid)
    assert (changed['history_occ'][3,:,:,3]==11).all() and audit['relabeled_columns']==256
    grid,raw=raw_scene(); raw['history_occ'][3,:,:,3]=4; raw['history_occ'][3,:,:,4]=11
    changed,audit=stabilize_stc_ground(raw,grid)
    assert np.array_equal(changed['history_occ'][3],raw['history_occ'][3])
    assert audit['protected_destination_columns']==256


def test_ground_identity_is_exact_for_clean_history():
    grid,raw=raw_scene(); changed,audit=stabilize_stc_ground(raw,grid)
    assert np.array_equal(changed['history_occ'],raw['history_occ']) and audit['changed_voxels']==0


def test_pose_flat_ground_identity_and_no_supported_region_fallback():
    grid,raw=raw_scene(); planned=np.repeat(np.eye(4)[None],6,0)
    planned[:,0,3]=np.arange(1,7)
    output,audit=compensate_planned_ground_pose(raw,planned,grid)
    assert np.array_equal(output,planned) and audit['corrected']==0
    raw['history_occ'][:]=17
    output,audit=compensate_planned_ground_pose(raw,planned,grid)
    assert np.array_equal(output,planned) and audit['unsupported']==6


def test_pose_ramp_correction_causal_preserves_planner_xy_yaw_and_so3():
    grid,raw=raw_scene(shape=(32,32,16)); grid=OccupancyGrid(-8,-8,-2,(.5,.5,.2),(32,32,16))
    raw['history_occ'][:]=17
    for x in range(32):
        height=.08*(-8+(x+.5)*.5)-.5
        z=int(np.floor((height+2)/.2)); raw['history_occ'][:,x,:,z]=11
    planned=np.repeat(np.eye(4)[None],6,0); planned[:,0,3]=np.arange(1,7)*.5
    output,audit=compensate_planned_ground_pose(raw,planned,grid); output=np.asarray(output)
    assert audit['corrected']>0 and output[-1,2,3]>0.1
    assert np.array_equal(output[:,:2,3],planned[:,:2,3])
    assert np.allclose(np.arctan2(output[:,1,0],output[:,0,0]),0,atol=1e-12)
    assert np.allclose(output[:,:3,:3].transpose(0,2,1)@output[:,:3,:3],np.eye(3),atol=1e-12)
    assert np.array_equal(planned[:,2,3],np.zeros(6))
    with pytest.raises(ValueError): compensate_planned_ground_pose({**raw,'future_gt_occ':np.ones(1)},planned,grid)


def test_plane_rejects_one_sided_sparse_or_steep_evidence():
    from scipy.spatial import cKDTree
    x,y=np.meshgrid(np.arange(1.,9),np.arange(1.,9)); p=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    assert fit_ground(p,cKDTree(p[:,:2]),[0.,0.]) is None
    p[:,:2]-=4.5; p[:,2]=.3*p[:,0]
    assert fit_ground(p,cKDTree(p[:,:2]),[0.,0.]) is None


def test_pose_heading_error_wraps_and_tilt_is_not_yaw():
    def pose(deg):
        r=np.deg2rad(deg); p=np.eye(4); p[:2,:2]=[[np.cos(r),-np.sin(r)],[np.sin(r),np.cos(r)]]; return p
    a=pose_errors([pose(179)],[pose(-179)])[0]
    assert a['yaw_deg']==pytest.approx(2) and a['tilt_deg']==0


def test_adapters_ignore_future_truth_and_do_not_reuse_raw_future_pose_field():
    grid,raw=raw_scene(); planned=np.repeat(np.eye(4)[None],6,0)
    a,_=compensate_planned_ground_pose(raw,planned,grid); h,_=stabilize_stc_ground(raw,grid)
    other=deepcopy(raw); other['future_poses']=[np.full((4,4),999.)]*6
    b,_=compensate_planned_ground_pose(other,planned,grid); j,_=stabilize_stc_ground(other,grid)
    assert np.array_equal(a,b) and np.array_equal(h['history_occ'],j['history_occ'])


class FakeSource:
    def __init__(self):
        self.grid,self.raw=raw_scene(shape=(8,8,8)); self.shape=self.grid.shape_hwd
        self.windows=[SimpleNamespace(key=str(i),scene='s',t0='t0',future=tuple('abcdef')) for i in range(2)]
        self.catalog={t:SimpleNamespace(pose=np.eye(4)) for t in 'abcdef'}
        self.predictions=0; self.target_reads=0
    def prediction_inputs(self,w,setting): return dict(scene_name='s',t0_token='t0'),deepcopy(self.raw)
    def metric_targets(self,w):
        assert self.predictions%len(ROUTES)==0 and self.predictions>0
        self.target_reads+=1
        return [self.raw['history_occ'][-1]]*3
    def frame(self,*a):
        assert self.target_reads>0
        return self.raw['history_occ'][-1],np.ones(self.shape,bool)


class FakePredictor:
    def __init__(self,source,fail=None): self.source=source; self.fail=fail
    def full(self,rec,raw,verify):
        self.source.predictions+=1
        if self.source.predictions==self.fail: raise RuntimeError('fixture failure')
        prep=SimpleNamespace(raw={},state={'current':[]})
        return prep,[raw['history_occ'][-1].copy()]*6,None,{}


def test_atomic_full_routes_gt_boundary_summary_and_strict_resume():
    source=FakeSource(); contract=dict(windows=2,weights='fixed',protocol='screen')
    stop=Event(); saved={}
    first=evaluate(source,source.windows,FakePredictor(source),source.grid,contract,
        save=lambda s:saved.update(s),stop_event=stop,progress=lambda r:stop.set())
    assert first['completed_windows']==1 and first['status']=='stopped'
    resumed=evaluate(source,source.windows,FakePredictor(source),source.grid,contract,saved=saved)
    other=FakeSource(); full=evaluate(other,other.windows,FakePredictor(other),other.grid,contract)
    assert resumed['reports']==full['reports'] and resumed['audits']==full['audits']
    assert 'No training' in summary(full) and full['reports']['b_occ_gt']['average']['standard_mIoU']==100
    with pytest.raises(RuntimeError): restore(saved,{**contract,'weights':'changed'},source.shape)
    from real_motion.waymo_i2world import fingerprint
    damaged=deepcopy(saved); damaged.pop('fingerprint'); damaged['counts']['b_occ_gt'][0][17][17]+=1
    damaged['fingerprint']=fingerprint(damaged)
    with pytest.raises(ValueError): restore(damaged,contract,source.shape)


def test_partial_route_failure_commits_no_counts_or_audit():
    source=FakeSource(); saved={}; contract=dict(windows=2)
    with pytest.raises(RuntimeError,match='fixture failure'):
        evaluate(source,source.windows,FakePredictor(source,fail=4),source.grid,contract,save=lambda s:saved.update(s))
    assert saved['completed_windows']==0 and saved['audits']==[] and source.target_reads==0
    restore(saved,contract,source.shape)


def test_readonly_motion_audit_reports_centroid_jitter_without_modification():
    frames=[]
    for x in (0.,.8,.6,1.5):
        frames.append(SimpleNamespace(components=[dict(class_id=4,centroid_world=np.array([x,0.,0.]))]))
    velocities={0:np.array([1.8,0.,0.])}; original=velocities[0].copy()
    prep=SimpleNamespace(raw={'_waymo_frame_geometry':frames},state=dict(current=frames[-1].components,velocities=velocities))
    audit=motion_jitter_audit(prep)
    assert audit['tracks_with_3plus_frames']==1 and audit['centroid_fit_rmse_p90_m']>0
    assert not audit['velocities_modified'] and np.array_equal(velocities[0],original)


def test_plane_rejects_diagonal_extrapolation_despite_axis_bracketing():
    from scipy.spatial import cKDTree
    x,y=np.meshgrid(np.linspace(-2,6,20),np.linspace(-2,6,20))
    xy=np.column_stack((x.ravel(),y.ravel())); xy=xy[xy.sum(1)>2]
    assert (xy.min(0)<0).all() and (xy.max(0)>0).all()
    points=np.column_stack((xy,np.zeros(len(xy))))
    assert fit_ground(points,cKDTree(xy),[0,0]) is None


def test_vectorized_ground_matches_slow_column_reference():
    from real_motion.stc_causal_geometry import ground_columns, GROUND
    grid,raw=raw_scene(shape=(12,12,8)); rng=np.random.default_rng(7)
    for f in range(4):
        raw['history_occ'][f]=17
        for x,y in np.ndindex(12,12):
            z=int(rng.integers(2,5)); thickness=int(rng.integers(1,4))
            raw['history_occ'][f,x,y,z:z+thickness]=rng.choice(GROUND)
            if rng.random()<.1: raw['history_occ'][f,x,y,z]=4
    actual,_=stabilize_stc_ground(raw,grid); expected=raw['history_occ'].copy()
    # Identity poses make the reference independent of warp implementation.
    for f in range(4):
        own=ground_columns(raw['history_occ'][f])
        priors=[ground_columns(raw['history_occ'][j]) for j in range(4) if j!=f]
        for x,y in np.ndindex(12,12):
            if not own[3][x,y]: continue
            votes=[sum(p[3][x,y] and p[0][x,y]==c for p in priors) for c in GROUND]
            cls=GROUND[int(np.argmax(votes))]
            heights=[int(p[1][x,y]) for p in priors if p[3][x,y] and p[0][x,y]==cls]
            if len(heights)<2 or max(heights)-min(heights)>1: continue
            z=np.median(heights); old=int(own[1][x,y]); high=int(own[2][x,y])
            dest=min((int(np.floor(z)),int(np.ceil(z))),key=lambda a:(abs(a-old),a))
            shift=dest-old; end=high+shift
            if abs(shift)>1 or dest<0 or end>=8: continue
            column=raw['history_occ'][f,x,y]
            if ((column[dest:end+1]!=17)&~np.isin(column[dest:end+1],GROUND)).any(): continue
            expected[f,x,y,old:high+1]=17; expected[f,x,y,dest:end+1]=cls
    assert np.array_equal(actual['history_occ'],expected)


def test_real_frozen_nine_routes_preserve_original_baselines_weights_and_future_boundary(tmp_path,monkeypatch):
    import torch
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from real_motion.prepared import PrepareConfig
    from real_motion.waymo_native_execution import prepare_waymo_native
    from test_stc_camera_protocol import fixture
    from tools.real_motion.stc_shared_execution import Predictor, bytes_equal
    prepare_waymo_native(tmp_path/'native')
    source=fixture(tmp_path,shape=(32,32,4)); windows=source.windows[:1]; source.preflight(windows)
    cfg=PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1.,(.4,)*3,source.shape))
    model=JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,
        blocks=1,decoder_blocks=1),width=16,z_bins=4).eval().requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    predictor=Predictor(model,cfg,'cpu',workers=1,graphs=False)
    original={k:predictor.full(*source.prediction_inputs(windows[0],k))[1] for k in ('occ_gt','occ_pred','stc_gt','stc_pred')}
    full=predictor.full; finished=[]; targets=source.metric_targets
    def captured(*args,**kwargs):
        output=full(*args,**kwargs); finished.append(output[1]); return output
    def score(window):
        assert len(finished)==len(ROUTES); return targets(window)
    monkeypatch.setattr(predictor,'full',captured); monkeypatch.setattr(source,'metric_targets',score)
    try:
        result=evaluate(source,windows,predictor,cfg.grid,dict(windows=1))
        assert result['status']=='complete' and 'motion_b_stc_gt' in summary(result)
        for i,(route,(setting,_,_)) in enumerate(ROUTES.items()):
            if not route.startswith('b_'): continue
            for h in range(6): bytes_equal(finished[i][h],original[setting][h],route+str(h))
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    finally: predictor.close(); torch.set_num_threads(threads)


def test_cli_new_output_resume_readonly_checkpoint_and_invalid_modes(tmp_path,monkeypatch):
    import json
    from test_stc_camera_protocol import fixture
    from real_motion.prepared import PrepareConfig
    from real_motion.waymo_i2world import file_sha256
    from tools.real_motion import eval_p0_f9_stc_causal_geometry as cli
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
    class PredictorFixture:
        def __init__(self,*a,**kw): pass
        def close(self): pass
        def full(self,rec,raw,verify=False):
            return SimpleNamespace(raw={},state={'current':[]}),[raw['history_occ'][-1]]*6,None,{}
    monkeypatch.setattr(cli,'Predictor',PredictorFixture)
    files={k:tmp_path/k for k in ('weights/mean.pt','config.yaml','population.json')}
    files['weights/mean.pt'].parent.mkdir()
    for path in files.values(): path.write_text('fixture')
    digest=file_sha256(files['weights/mean.pt']); out=tmp_path/'new-screen'
    common=['--dataroot',str(source.root),'--stc-root',str(source.stc_root),'--plan-cache',str(source.plan_cache),
        '--population','dev64','--population-manifest',str(files['population.json']),
        '--config',str(files['config.yaml']),'--checkpoint',str(files['weights/mean.pt'])]
    args=common+['--out-dir',str(out)]
    assert cli.main(argv=args)==0
    first=json.loads((out/'evaluation.json').read_text())
    assert cli.main(argv=args+['--resume'])==0
    assert json.loads((out/'evaluation.json').read_text())['reports']==first['reports']
    assert file_sha256(files['weights/mean.pt'])==digest
    with pytest.raises(RuntimeError,match='contract'): cli.main(argv=args+['--resume','--cpu-workers','2'])
    with pytest.raises(SystemExit): cli.main(argv=args)
    with pytest.raises(SystemExit): cli.main(argv=common+['--out-dir',str(source.stc_root/'unsafe')])
    with pytest.raises(SystemExit): cli.main(argv=args+['--population','all'])


def test_resume_launcher_restores_only_own_swfm_contract_without_shell_eval(tmp_path,monkeypatch):
    import json, os, sys
    from pathlib import Path
    script=(Path(__file__).resolve().parents[1]/'tools/real_motion/run_p0_f9_stc_causal_geometry.sh').read_text(encoding='utf-8')
    launcher=script.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    contract=tmp_path/'contract.json'
    contract.write_text(json.dumps(dict(runtime_environment={'SWFM_TEST':'literal $()'})))
    monkeypatch.setenv('SWFM_STALE','no'); monkeypatch.setattr(sys,'argv',['-',str(contract),'/repo','--resume'])
    captured={}
    monkeypatch.setattr(os,'execve',lambda exe,args,env:captured.update(args=args,env=env))
    exec(compile(launcher,'<resume-launcher>','exec'),{})
    assert captured['env']['SWFM_TEST']=='literal $()' and 'SWFM_STALE' not in captured['env']
    assert captured['args'][-1]=='--resume' and captured['args'][2].endswith('eval_p0_f9_stc_causal_geometry.py')
