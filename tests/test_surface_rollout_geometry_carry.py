"""Causal geometry, foreground safety, actual Surface math and integer recovery."""
import copy
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.strong_w2det import StrongW2DetConfig, inverse_warp
from tools.real_motion import surface_rollout_geometry_carry as carry
from tools.real_motion import eval_p0_f9_surface_geometry_carry as cli
from test_joint_surface_long_rollout import provider_fixture, common
from test_joint_long_rollout import window, record


@dataclass
class View:
    baseline: list
    owners: list
    fallbacks: list
    registrations: list = None
    state: dict = None
    raw: dict = None
    outputs: object = None


def raw_history(labels):
    return dict(history_occ=np.stack([labels]*4),history_observed=np.ones((4,*labels.shape),bool),
                history_poses=[np.eye(4)]*4,future_poses=[np.eye(4)]*6,future_gt_occ=None)


def test_static_direct_projection_avoids_intermediate_quantization_and_retains_ccr():
    grid=OccupancyGrid(0,0,0,(.4,.4,.4),(10,5,2))
    labels=np.full(grid.shape_hwd,17,np.uint8);labels[3,2,0]=11
    raw=raw_history(labels);p3=np.eye(4);p3[0,3]=.12;raw['future_poses'][-1]=p3
    p6=np.eye(4);p6[0,3]=.24
    middle=inverse_warp(labels,np.linalg.inv(p3),grid,17)[0]
    repeated=inverse_warp(middle,np.linalg.inv(p6)@p3,grid,17)[0]
    first=NS(raw=raw,baseline=[middle]*6)
    provider=NS(pcfg=NS(grid=grid),device='cpu',strong=StrongW2DetConfig())
    direct,audit=carry.direct_static_backgrounds(first,[middle]*6,[p6]*6,provider)
    expected=inverse_warp(labels,np.linalg.inv(p6),grid,17)[0]
    assert direct[0][2,2,0]==11 and repeated[2,2,0]==17
    assert np.array_equal(direct[0][:,1:4],expected[:,1:4])
    assert audit['carried_ccr_static_voxels']==0
    novel=middle.copy();novel[5,2,0]=13
    retained,audit=carry.direct_static_backgrounds(first,[middle]*5+[novel],[p3]*6,provider)
    assert retained[0][5,2,0]==13 and audit['carried_ccr_static_voxels']==1
    assert np.array_equal(labels,raw['history_occ'][-1])


def test_static_preserves_dynamic_foreground_and_overlapping_dynamic_fallback():
    base=np.array([17,11,4,4,11],np.uint8).reshape(5,1,1)
    owners=np.array([-1,-1,0,0,-1]).reshape(5,1,1)
    fall=np.array([17,11,11,7,11],np.uint8).reshape(5,1,1)
    bg=np.array([13,17,13,13,12],np.uint8).reshape(5,1,1)
    old=View([base]*6,[owners]*6,[fall]*6,outputs=object())
    new,audit=carry.apply_static_backgrounds(old,[bg]*6)
    assert new.baseline[0].ravel().tolist()==[13,17,4,4,12]
    assert new.fallbacks[0].ravel().tolist()==[17,11,13,7,11]
    assert new.outputs is old.outputs and new.owners is old.owners
    assert audit['background_removed']==6 and audit['dynamic_foreground_changed']==0
    assert np.array_equal(base.ravel(),[17,11,4,4,11])
    with pytest.raises(ValueError):carry.apply_static_backgrounds(old,[bg]*5)
    with pytest.raises(ValueError):carry.apply_static_backgrounds(old,[base]*6)


def rigid_fixture():
    grid=OccupancyGrid(0,0,0,(1,1,1),(8,8,2));cell=np.array([[2,2,0]])
    comp=dict(class_id=4,voxel_indices=cell,centroid_world=np.array([2.5,2.5,.5]))
    dense=np.full(grid.shape_hwd,17,np.uint8);dense[2,2,0]=4
    owner=np.full(grid.shape_hwd,-1,np.int32);owner[2,2,0]=0
    angles=np.array([[0.],[0.],[0.],[0.],[0.],[np.pi/2]])
    centers=np.tile([2.5,2.5,9.],(6,1,1))
    first=NS(state={'current':[comp]},targets=centers,yaws=angles,owners=[owner]*6)
    second=View([dense]*6,[owner]*6,[dense]*6,
        registrations=[[(np.eye(4),cell)]*4],
        state=dict(components_by_frame=[[comp]]*4,current=[comp],
                   motion_handoff_audit={'source_identity_pairs':[[0,0]]}),raw=raw_history(dense),outputs=object())
    return first,[dense]*6,second,grid


def test_predicted_relative_yaw_registration_is_world_se2_not_future_double_rotation():
    first,pred,second,grid=rigid_fixture();before=first.targets.copy()
    new,audit=carry.predicted_rigid_registrations(first,pred,second,grid)
    matrix,cells=new.registrations[0][0]
    np.testing.assert_allclose(matrix@np.array([3.5,2.5,.5,1]),[2.5,3.5,.5,1],atol=1e-12)
    assert audit['exact_se2_registrations']==3
    assert new.registrations[0][3] is second.registrations[0][3]
    assert new.outputs is second.outputs and new.state is second.state
    assert np.array_equal(first.targets,before) and np.array_equal(cells,[[2,2,0]])
    assert np.array_equal(second.registrations[0][0][0],np.eye(4))


@pytest.mark.parametrize('reason',['split','wrong_semantic','large_offset','no_pair'])
def test_unreliable_history_keeps_original_registration(reason):
    first,pred,second,grid=rigid_fixture()
    if reason=='split':second.state['components_by_frame']=[[second.state['current'][0]]*2]*4
    if reason=='wrong_semantic':
        pred=[x.copy() for x in pred]
        for x in pred:x[2,2,0]=7
    if reason=='large_offset':second.state['current'][0]['centroid_world']=np.array([7.5,7.5,.5])
    if reason=='no_pair':second.state['motion_handoff_audit']['source_identity_pairs']=[]
    new,audit=carry.predicted_rigid_registrations(first,pred,second,grid)
    assert audit['exact_se2_registrations']==0
    for f in range(4):assert new.registrations[0][f] is second.registrations[0][f]


@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('mode',['numpy','native_parallel'])
@pytest.mark.parametrize('device',['cpu','cuda'])
@torch.no_grad()
def test_all_routes_real_surface_reference_parity_readonly_and_no_gt(empty,mode,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    threads=torch.get_num_threads();torch.set_num_threads(1)
    provider,w,r,raw=provider_fixture(empty,device)
    before={k:v.clone() for k,v in provider.joint.state_dict().items()}
    exe=common.SurfaceBlockExecution(provider,mode=mode,workers=2,query_workers=2,graphs=True)
    try:
        first=provider.prepare_columns(None,r,include_gt=False,raw_window=raw)
        pred1,_,_,prob=exe.predict(first);common.verify_first_block(provider,r,first,pred1,prob,exe)
        handoff=common.handoff_from_prepared(first,pred1[-1]);poses=[np.eye(4)]*12
        second=common.synthetic_preparation(pred1,raw,poses,w,provider,handoff=handoff)
        base=[v.copy() for v in second.baseline];regs=copy.deepcopy(second.registrations)
        views,audit=carry.candidates(first,pred1,second,poses[6:],provider)
        assert tuple(views)==carry.ROUTES
        for v in views.values():
            dense,edits,_,prob=exe.predict(v);exe.verify(v,dense,prob)
            assert edits['removed']==0 and v.outputs is second.outputs and v.raw is second.raw
        assert all(np.array_equal(a,b) for a,b in zip(second.baseline,base))
        for a,b in zip(second.registrations,regs):
            for x,y in zip(a,b):
                assert (x is None)==(y is None)
                if x is not None:assert np.array_equal(x[0],y[0]) and np.array_equal(x[1],y[1])
        assert all(torch.equal(before[k],v) for k,v in provider.joint.state_dict().items())
        raw['future_gt_occ']=np.zeros((6,*provider.pcfg.grid.shape_hwd),np.uint8)
        with pytest.raises(RuntimeError,match='future'):carry.direct_static_backgrounds(first,pred1,poses[6:],provider)
    finally:exe.close();torch.set_num_threads(threads)


def orchestration(monkeypatch):
    provider,_,_,raw=provider_fixture(empty=True)
    ws=[window(s,t) for s,t in [('train','a'),('train','b'),('dev','c'),('dev','d')]]
    jobs=[('train64' if w.scene_name=='train' else 'dev64',w,record(w)) for w in ws]
    calls=[];event=Event();stop=[False];gt=np.full(provider.pcfg.grid.shape_hwd,11,np.uint8)
    pred=[gt.copy() for _ in range(6)]
    class Execution:
        def predict(self,prep):
            calls.append('predict');return copy.deepcopy(pred),{'added':1,'removed':0,'changed':1},{},np.empty((0,6,2))
        def verify(self,*args):calls.append('verify')
    provider.prepare_columns=lambda *a,**k:NS(state={'rec':a[1]},raw=k['raw_window'])
    monkeypatch.setattr(cli.old,'prefetch_raw_columns',lambda p,s,rows,**k:((r,copy.deepcopy(raw)) for r in rows))
    monkeypatch.setattr(carry,'verify_first_block',lambda *a:None)
    monkeypatch.setattr(cli.old,'handoff_from_prepared',lambda *a,**k:None)
    monkeypatch.setattr(common,'synthetic_preparation',lambda *a,**k:NS())
    monkeypatch.setattr(carry,'candidates',lambda *a,**k:({r:NS() for r in carry.ROUTES},{'static':{'voxels':1}}))
    def moving(*a,**k):
        assert calls.count('predict')==5*(calls.count('moving')+1)
        calls.append('moving');return [(np.zeros(gt.shape,bool),[],{})]*6
    monkeypatch.setattr(cli.old,'gt_moving_support_sequence',moving)
    class Source:
        nusc=object()
        def pose(self,*a):return np.eye(4)
        def load_semantics(self,*a):
            if stop[0]:event.set()
            return gt
    c=dict(routes=[s+':'+r for s in ('train64','dev64') for r in carry.ROUTES],candidate_routes=list(carry.ROUTES))
    return provider,{'train64':Source(),'dev64':Source()},jobs,Execution(),c,event,stop


def test_labels_after_all_routes_scene_totals_and_atomic_resume(monkeypatch):
    p,s,jobs,e,c,event,stop=orchestration(monkeypatch);ledger=[];stop[0]=True
    with pytest.raises(InterruptedError):cli.evaluate(p,s,{},jobs,e,c,save=lambda v:ledger.append(copy.deepcopy(v)),stop_event=event)
    saved=ledger[-1];assert saved['completed_windows']==1
    stop[0]=False;event.clear();done=cli.evaluate(p,s,{},jobs,e,c,saved=saved)
    p,s,jobs,e,c,_,_=orchestration(monkeypatch);reference=cli.evaluate(p,s,{},jobs,e,c)
    for key in ('counts','edits','scene_counts','candidate_audit','comparison_quality'):
        assert done[key]==reference[key]
    result=cli.reports(done,{'train64':{},'dev64':{}},carry.ROUTES)
    assert result['dev64']['combined']['metrics']['average_4s_5s_6s']['mIoU']==100
    corrupted=copy.deepcopy(done);corrupted.pop('fingerprint');corrupted['scene_counts']['dev64:baseline']['dev']['occ_union'][0]+=1
    corrupted['fingerprint']=cli.old.stable_json_fingerprint(corrupted)
    with pytest.raises(RuntimeError,match='scene counts'):cli.restore(corrupted,c,jobs)


def selection_reports():
    metric=dict(average_4s_5s_6s=dict(mIoU=10.,IoU=20.,MovingMicro=5.),
                per_horizon={str(h):dict(mIoU=10.) for h in (4.,5.,6.)})
    return {r:dict(metrics=copy.deepcopy(metric)) for r in carry.ROUTES}


def test_metrics_only_integer_resume_and_no_moving_read(monkeypatch):
    def no_moving(*a,**k):pytest.fail('omitted Moving support was accessed')
    p,s,jobs,e,c,event,stop=orchestration(monkeypatch)
    monkeypatch.setattr(cli.old,'gt_moving_support_sequence',no_moving)
    c['metric_scope']='IoU_mIoU_only';saved=[];stop[0]=True
    with pytest.raises(InterruptedError):
        cli.evaluate(p,s,{},jobs,e,c,save=lambda v:saved.append(copy.deepcopy(v)),stop_event=event)
    assert saved[-1]['completed_windows']==1
    stop[0]=False;event.clear();done=cli.evaluate(p,s,{},jobs,e,c,saved=saved[-1])
    p,s,jobs,e,c,_,_=orchestration(monkeypatch);c['metric_scope']='IoU_mIoU_only'
    monkeypatch.setattr(cli.old,'gt_moving_support_sequence',no_moving)
    reference=cli.evaluate(p,s,{},jobs,e,c)
    for key in ('counts','scene_counts','comparison_quality','candidate_audit','edits'):
        assert done[key]==reference[key]
    corrupt=copy.deepcopy(done);corrupt.pop('fingerprint')
    corrupt['counts']['train64:baseline']['mov_union'][0][0]=1
    corrupt['fingerprint']=cli.old.stable_json_fingerprint(corrupt)
    with pytest.raises(RuntimeError,match='Moving counts'):cli.restore(corrupt,c,jobs)


def test_train_only_fixed_gate_and_recipe_integrity():
    rows=selection_reports();assert cli.choose_train_candidate(rows) is None
    rows['combined']['metrics']['average_4s_5s_6s']['mIoU']+=.1
    assert cli.choose_train_candidate(rows)=='combined'
    value=dict(protocol=cli.PROTOCOL,status='complete',bundle_fingerprint='b',implementation='i',
               reports={'train64':rows,'dev64':selection_reports()},selected_train_route='combined')
    value['result_fingerprint']=cli.old.stable_json_fingerprint(value)
    assert cli.validate_recipe(value,'b','i')=='combined'
    changed=copy.deepcopy(value);changed['reports']['dev64']['baseline']['metrics']['average_4s_5s_6s']['mIoU']=200
    with pytest.raises(RuntimeError,match='content'):cli.validate_recipe(changed,'b','i')
    with pytest.raises(RuntimeError,match='matching'):cli.validate_recipe(value,'different','i')
    rows['combined']['metrics']['per_horizon']['4.0']['mIoU']=9.
    assert cli.choose_train_candidate(rows) is None
    rows['combined']['metrics']['per_horizon']['4.0']['mIoU']=10.
    rows['combined']['metrics']['average_4s_5s_6s']['MovingMicro']=4.
    assert cli.choose_train_candidate(rows) is None


def test_complete_cli_frozen_snapshot_resume_output_and_fail_closed(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from tools.real_motion.joint_surface_checkpoint_selection import evaluation_payload, weight_fingerprint
    p,s,jobs,exe,_,event,stop=orchestration(monkeypatch)
    files={n:tmp_path/n for n in ('config','dev-cache','dev-info','train-cache','train-info','base-checkpoint','population-manifest')}
    for n,f in files.items():f.write_bytes(n.encode())
    actual_sha=cli.old.sha256
    def sha(path):return cli.old.CLEAN_SHA256 if Path(path)==files['base-checkpoint'] else actual_sha(path)
    trained=dict(data={n:sha(files[n.replace('_','-')]) for n in ('dev_cache','dev_info','train_cache','train_info','base_checkpoint')},
        dataroot=str(tmp_path.resolve()),runtime_config_fingerprint=cli.old.stable_json_fingerprint({}),
        implementation=cli.old.training_implementation(Path(cli.__file__).resolve().parents[2]),
        torch_version=str(torch.__version__),dev_manifest_fingerprint='manifest',prior_keys=[['train','a']])
    value=evaluation_payload(p.joint.state_dict(),p.joint.configs(),trained,
        dict(positive_weights=p.joint.columns.positive_weight.tolist()),
        [dict(epoch=i) for i in cli.old.AVERAGE_EPOCHS],average=True)
    mean=tmp_path/'mean.pt';torch.save(value,mean);original=mean.read_bytes()
    run=tmp_path/'train_run';run.mkdir();source_dir=tmp_path/'source';source_dir.mkdir()
    bundle=dict(selection_frozen=True,run_directory=str(run.resolve()),source_comparison_directory=str(source_dir.resolve()),
        audit={'contract':trained},candidates={cli.old.AVERAGE_NAME:dict(path=str(mean),sha256=actual_sha(mean),
            weight_fingerprint=weight_fingerprint(p.joint.state_dict()),source_epochs=list(cli.old.AVERAGE_EPOCHS))})
    bundle['fingerprint']=cli.old.stable_json_fingerprint(bundle)
    monkeypatch.setattr(cli.old,'sha256',sha)
    monkeypatch.setattr(cli.old,'find_frozen_bundle',lambda *a:copy.deepcopy(bundle))
    monkeypatch.setattr(cli.old,'verify_sources',lambda b:None)
    monkeypatch.setattr(cli.old,'load_runtime_config',lambda *a:{})
    monkeypatch.setattr(cli.old,'make_prepare_config',lambda *a:p.pcfg)
    monkeypatch.setattr(cli.old,'load_manifest',lambda *a:(dict(manifest_fingerprint='manifest',
        selected_key_fingerprint=cli.old.DEV64_FP,parent_keys=[['dev',str(i)] for i in range(512)]),[('dev',str(i)) for i in range(64)],None))
    monkeypatch.setattr(cli,'TRAIN_WINDOWS',2);monkeypatch.setattr(cli.old,'VAL_WINDOWS',2)
    monkeypatch.setattr(cli.old,'load_cache',lambda path:({},[r for split,_,r in jobs if (split=='train64')==('train' in str(path))]))
    monkeypatch.setattr(cli.old,'require_cuda',lambda *a:torch.device('cpu'))
    def source(*a,**k):return s['train64' if 'train' in str(k['info_pkl']) else 'dev64']
    monkeypatch.setattr(cli.old,'NuScenesWindowSource',source)
    monkeypatch.setattr(cli.old,'CachedColumnSource',lambda src,*a:src)
    for split,src in s.items():
        src.allowed_scenes={'train' if split=='train64' else 'dev'}
        src.iter_windows=lambda **k:iter(())
    def population(records,src,parent,split):
        chosen=[(w,r) for sp,w,r in jobs if sp==split]
        return chosen,dict(selected_windows=len(chosen),selected_keys=[[w.scene_name,w.t0_token] for w,_ in chosen])
    monkeypatch.setattr(cli,'select_screen',population)
    monkeypatch.setattr(cli.old.rollout,'validate_timestamps',lambda *a:dict(intervals_s=[.5]*15,
        relative_times_s=((np.arange(16)-3)*.5).tolist(),max_nominal_deviation_s=0.))
    monkeypatch.setattr(common,'SurfaceRolloutProvider',lambda *a:p)
    exe.close=lambda:None;monkeypatch.setattr(common,'SurfaceBlockExecution',lambda *a,**k:exe)
    out=tmp_path/'eval'
    argv=[arg for n,f in files.items() for arg in ('--'+n,str(f))]
    argv+=['--run-dir',str(run),'--runs-root',str(tmp_path),'--dataroot',str(tmp_path),'--out-dir',str(out),
           '--ccr-cpu-execution','numpy','--no-graphs','--majority-workers','0']
    stop[0]=True;assert cli.main(event,argv)==130
    assert json.loads((out/'evaluation_state.json').read_text())['completed_windows']==1
    snapshot=out/'checkpoint_snapshot.pt';snapshot.write_bytes(b'bad')
    with pytest.raises(RuntimeError,match='snapshot'):cli.main(event,argv+['--resume'])
    snapshot.write_bytes(original);event.clear();stop[0]=False
    assert cli.main(event,argv+['--resume'])==0
    result=json.loads((out/'evaluation.json').read_text())
    assert result['status']=='complete' and result['completed_windows']==4
    assert result['selected_train_route'] is None and result['source_epochs']==[5,6,8,12,14]
    assert 'dMovingMicro=NA' in (out/'summary.txt').read_text()
    assert mean.read_bytes()==original and snapshot.read_bytes()==original
    assert 'optimizer' not in torch.load(snapshot,weights_only=False)
    with pytest.raises(SystemExit):cli.main(event,argv+['--resume'])
    metadata=tmp_path/'official.pkl';metadata.write_bytes(b'fixture metadata')
    all_args=argv+['--experiment','all','--population','all','--population-alignment','geniedrive_code10s',
                   '--geniedrive-info',str(metadata),'--selection-from',str(out/'evaluation.json'),
                   '--out-dir',str(tmp_path/'all_rejected')]
    with pytest.raises(RuntimeError,match='TRAIN-eligible'):cli.main(event,all_args)
    # Lifecycle-only recipe: abstract TRAIN table passes the separately tested
    # gate. No synthetic accuracy is presented as a real dataset result.
    recipe=copy.deepcopy(result);recipe.pop('result_fingerprint')
    recipe['reports']['train64']=selection_reports()
    recipe['reports']['train64']['combined']['metrics']['average_4s_5s_6s']['mIoU']+=.1
    recipe['selected_train_route']='combined'
    recipe['result_fingerprint']=cli.old.stable_json_fingerprint(recipe)
    selection=tmp_path/'fixture_recipe.json';cli.old.write_json(selection,recipe)
    chosen=[(w,r) for sp,w,r in jobs if sp=='dev64']
    monkeypatch.setattr(cli.old.genie,'select_population',lambda *a:(chosen,dict(selected_windows=2)))
    monkeypatch.setattr(cli.old.genie,'validate_grid',lambda *a:None)
    monkeypatch.setattr(cli.old,'gt_moving_support_sequence',lambda *a,**k:[(np.zeros(p.pcfg.grid.shape_hwd,bool),[],{})]*6)
    all_out=tmp_path/'all'
    assert cli.main(event,all_args+['--selection-from',str(selection),'--out-dir',str(all_out)])==0
    full=json.loads((all_out/'evaluation.json').read_text())
    assert full['candidate_routes']==['baseline','combined'] and set(full['reports'])=={'all'}
    assert set(full['geniedrive_code_compatibility'])=={'baseline','combined'}
    assert full['selected_train_route']=='combined' and mean.read_bytes()==original
    # User accepts the small Moving decline: separate explicit choice, never
    # change the completed screen or relabel its failed original TRAIN gate.
    recipe_bytes=(out/'evaluation.json').read_bytes()
    def no_moving(*a,**k):pytest.fail('metrics-only run read future moving annotations')
    monkeypatch.setattr(cli.old,'gt_moving_support_sequence',no_moving)
    approved_out=tmp_path/'approved_all'
    approved_args=argv+['--experiment','all','--population','all',
        '--population-alignment','geniedrive_code10s','--geniedrive-info',str(metadata),
        '--approved-route','static_carry','--iou-miou-only','--out-dir',str(approved_out)]
    assert cli.main(event,approved_args)==0
    approved=json.loads((approved_out/'evaluation.json').read_text())
    assert approved['candidate_routes']==['baseline','static_carry']
    assert approved['selected_train_route'] is None and approved['user_approved_route']=='static_carry'
    assert 'NOT_TRAIN_gate_pass' in approved['selection_policy']
    for route,row in approved['reports']['all'].items():
        metrics=row['metrics'];reference=full['reports']['all']['baseline']['metrics']
        for group in ('per_horizon','average_1s_2s_3s','average_4s_5s_6s'):
            items=metrics[group].values() if group=='per_horizon' else [metrics[group]]
            for v in items:assert v['MovingMicro'] is None and v['MovingMacro'] is None
        # Same fixture predictions: IoU/mIoU counts are identical even though
        # Moving support is omitted; never present omitted Moving as zero.
        for h,v in metrics['per_horizon'].items():
            for key in ('IoU','mIoU'):assert v[key]==reference['per_horizon'][h][key]
    compact=(approved_out/'summary.txt').read_text()
    assert '1.0s' in compact and '6.0s' in compact and 'avg4--6' in compact
    assert 'dMovingMicro=' not in compact and 'Moving not evaluated' in compact
    assert (out/'evaluation.json').read_bytes()==recipe_bytes and mean.read_bytes()==original
    with pytest.raises(SystemExit):cli.main(event,approved_args+['--selection-from',str(selection)])
    with pytest.raises(SystemExit):cli.main(event,argv+['--approved-route','static_carry'])


@torch.no_grad()
def test_static_only_full_skips_se2_and_matches_screen_predictions(monkeypatch):
    threads=torch.get_num_threads();torch.set_num_threads(1)
    provider,w,r,raw=provider_fixture()
    exe=common.SurfaceBlockExecution(provider,mode='numpy',workers=2,query_workers=2,graphs=False)
    try:
        first=provider.prepare_columns(None,r,include_gt=False,raw_window=raw)
        pred,_,_,_=exe.predict(first);pred_before=[v.copy() for v in pred]
        handoff=common.handoff_from_prepared(first,pred[-1]);poses=[np.eye(4)]*12
        second=common.synthetic_preparation(pred,raw,poses,w,provider,handoff=handoff)
        all_views,_=carry.candidates(first,pred,second,poses[6:],provider)
        expected,_,_,scores=exe.predict(all_views['static_carry'])
        monkeypatch.setattr(carry,'predicted_rigid_registrations',lambda *a:pytest.fail('unused SE2 branch computed'))
        views,audit=carry.candidates(first,pred,second,poses[6:],provider,routes=['baseline','static_carry'])
        actual,_,_,scores2=exe.predict(views['static_carry'])
        assert set(views)=={'baseline','static_carry'} and 'se2' not in audit
        assert np.array_equal(scores,scores2) and all(np.array_equal(a,b) for a,b in zip(expected,actual))
        assert all(np.array_equal(a,b) for a,b in zip(pred,pred_before))
        with pytest.raises(ValueError):carry.candidates(first,pred,second,poses[6:],provider,routes=['typo'])
    finally:exe.close();torch.set_num_threads(threads)


def legacy_upgrade_case():
    from dataclasses import replace
    provider,w,record,raw=provider_fixture()
    provider.pcfg=replace(provider.pcfg,grid=replace(provider.pcfg.grid,x_min=30.,y_min=30.))
    state=common.rollout.build_four_history_state(raw['history_occ'],raw['history_poses'],raw['future_poses'],
        provider.pcfg,provider.strong,provider.device)
    rec={**state['rec'],**{k:record[k] for k in ('scene_name','t0_token','history_tokens','future_tokens')}}
    # Reproduce upgrade_v1_record_targets exactly, including NumPy FP64
    # multiplication of the FP32 normalized features (NOT torch *40).
    source=rec['features'][:,:2].numpy().astype(np.float64)*40.
    upgrade=(rec['anchors_xy_t0_m'].numpy().astype(np.float64)-source[:,None,:]).astype(np.float32)
    assert np.max(np.abs(upgrade-rec['kta_displacement_xy_m'].numpy()))>1e-6
    rec['kta_displacement_xy_m']=torch.from_numpy(upgrade)
    raw['_column_causal_preparation']=common.causal_geometry(raw,state,provider)
    return provider,w,rec,raw


@torch.no_grad()
def test_exact_legacy_upgrade_kta_gate_real_forecast_is_unchanged(capsys):
    threads=torch.get_num_threads();torch.set_num_threads(1)
    p,w,r,raw=legacy_upgrade_case();before=r['kta_displacement_xy_m'].clone()
    exe=common.SurfaceBlockExecution(p,mode='numpy',workers=2,query_workers=2,graphs=False)
    try:
        first=p.prepare_columns(None,r,include_gt=False,raw_window=raw)
        pred,_,_,prob=exe.predict(first)
        carry.verify_first_block(p,r,first,pred,prob,exe)
        assert 'legacy_cache_upgrade_KTA_arithmetic_verified' in capsys.readouterr().out
        assert torch.equal(before,r['kta_displacement_xy_m'])
        again,_,_,prob2=exe.predict(first)
        assert all(np.array_equal(a,b) for a,b in zip(pred,again)) and np.array_equal(prob,prob2)
    finally:exe.close();torch.set_num_threads(threads)


@pytest.mark.parametrize('corruption',['kta','semantic_class','mask'])
def test_cache_rounding_compatibility_does_not_hide_other_input_errors(corruption,monkeypatch):
    p,_,r,raw=legacy_upgrade_case();wrong=copy.deepcopy(r)
    if corruption=='kta':wrong['kta_displacement_xy_m'][0,0,0]+=.001
    elif corruption=='semantic_class':wrong['source_class_id'][0]=7
    else:wrong['target_source_mask_tube'][0,-1,0,0]^=True
    monkeypatch.setattr(common,'verify_first_block',lambda *a:pytest.fail('invalid input reached dense verification'))
    with pytest.raises(RuntimeError):carry.verify_first_block(p,wrong,NS(raw=raw),None,None,None)

