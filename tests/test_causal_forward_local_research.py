"""Scientific invariants for current-source local research; no toy quality claims."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from real_motion.geometry import OccupancyGrid
from real_motion.shared_source_track import kta_and_anchors
from tools.source_scene_stage2.track import SourceIdentities
from tools.causal_forward_local import cache,calibrate,adapt,geometry_screen,feature_screen,residual_calibrate,range_calibrate


def test_research_array_cache_is_lossless_for_strided_FP32(tmp_path):
    a=np.random.default_rng(21).normal(size=(6,18,37)).astype(np.float32).transpose(1,0,2)
    path=tmp_path/'research.npz';cache.save_arrays(path,data=a,key=np.array(['scene','token']))
    with np.load(path,allow_pickle=False) as z:
        assert z['data'].dtype==np.float32 and z['data'].shape==a.shape
        np.testing.assert_array_equal(z['data'],a)
    assert not path.with_suffix('.tmp').exists()


def test_fit_screen_confirmation_and_TRAIN_replay_scene_disjoint():
    ref=[dict(scene_name=f'v{i}',t0_token=str(i),role='reference64') for i in range(64)]
    train=[dict(scene_name=f't{i}',t0_token=str(i),role='clean_train') for i in range(32)]
    b=SimpleNamespace(records=ref+train,by_key={(r['scene_name'],r['t0_token']):r for r in ref+train})
    c=dict(eval_keys=[[r['scene_name'],r['t0_token']] for r in ref]);jobs=cache.jobs(b,c)
    assert len(jobs)==160
    groups={g:{r['scene_name'] for _,r,_,part in jobs if part==g} for g in ('fit32','screen16','confirmation16','source_TRAIN32')}
    assert [len(groups[k]) for k in groups]==[32,16,16,32]
    for name,a in groups.items():
        for other,values in groups.items():
            if name!=other:assert not a&values
    ref[0]['role']='wrong'
    with pytest.raises(ValueError,match='reference64'):cache.jobs(b,c)


def velocity_fixture():
    pos=np.zeros((2,4,3));pos[0,:,0]=[-1.5,-1.,-.5,.5];pos[1,:,0]=2.
    valid=np.array([[1,1,1,1],[0,1,0,1]],bool);links=np.where(valid,0,-1)
    track=SourceIdentities(links,pos,valid,np.array([-1.5,-1.,-.5,0.]),{0:np.array([2.,0.,0.])},[[None]*4 for _ in range(2)],{})
    current=[dict(centroid_world=p,class_id=4) for p in pos[:,-1]];pose=np.eye(4)
    _,kta,anchors=kta_and_anchors(current,track,pose)
    rec=dict(kta_displacement_xy_m=kta,anchors_xy_t0_m=anchors)
    return rec,dict(_shared_source_track=track),None,(dict(state=dict(current=current,current_pose=pose)),None)


def test_LS_only_existing_last_pairs_no_position_or_identity_changes():
    row=velocity_fixture();before=copy.deepcopy(row);identity=geometry_screen.velocity_row(row,0.)
    np.testing.assert_array_equal(identity[0]['kta_displacement_xy_m'],row[0]['kta_displacement_xy_m'])
    changed=geometry_screen.velocity_row(row,.5);a=changed[1]['_shared_source_track'];old=row[1]['_shared_source_track']
    assert set(a.pair_velocity)==set(old.pair_velocity)=={0} and a.pair_velocity[0][0]==pytest.approx(1.65)
    np.testing.assert_array_equal(a.centers,old.centers);np.testing.assert_array_equal(a.track_valid,old.track_valid)
    np.testing.assert_array_equal(changed[0]['kta_displacement_xy_m'][1],np.zeros((6,2)))
    np.testing.assert_array_equal(row[0]['anchors_xy_t0_m'],before[0]['anchors_xy_t0_m'])
    np.testing.assert_array_equal(old.pair_velocity[0],before[1]['_shared_source_track'].pair_velocity[0])
    assert changed[3] is row[3]  # reference reads stay at ACTUAL observed locations.


def test_current_position_filter_agrees_with_direct_memory_without_double_shift():
    from tools.source_scene_ego_conditioned.geometry import direct_move
    row=velocity_fixture();state=row[3][0]['state'];state['source_world_points']=[np.asarray([v['centroid_world']+[.1,0.,0.]]) for v in state['current']]
    observed=np.concatenate(state['source_world_points']);raw=dict(row[1],_column_causal_preparation=dict(prepared_state=state))
    memory=dict(world=observed.copy(),centers_world=raw['_shared_source_track'].centers.copy())
    hist=(dict(state=state,memory_reads=memory,shared_reads={'actual':observed.copy()}),None)
    row=row[0],raw,row[2],hist;before=copy.deepcopy(row)
    changed=geometry_screen.position_row(row,.5,.5);new=changed[3][0]['state']
    assert changed[3][0]['memory_reads'] is memory
    np.testing.assert_array_equal(memory['world'],before[3][0]['memory_reads']['world'])
    np.testing.assert_array_equal(memory['centers_world'],before[3][0]['memory_reads']['centers_world'])
    target=torch.as_tensor(new['current'][0]['centroid_world']);actual=torch.as_tensor(state['current'][0]['centroid_world'])
    point=torch.as_tensor(state['source_world_points'][0]);zero=target.new_tensor(0.)
    direct=direct_move(point,actual,target,zero,zero)
    np.testing.assert_allclose(direct.numpy(),new['source_world_points'][0],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(state['source_world_points'][0],before[3][0]['state']['source_world_points'][0])
    assert float(target[0]-actual[0])==pytest.approx(-.075)


def test_history_pseudo_and_static_probabilities_require_no_future_GT():
    grid=OccupancyGrid(x_min=-4,y_min=-4,z_min=-1,voxel_size=(1.,1.,1.),shape_hwd=(8,8,4))
    sem=np.full((4,8,8,4),17,np.uint8);sem[:3,4,4,1]=11;sem[:,1,1,1]=4;sem[:2,6,6,1]=11
    raw=dict(history_occ=sem,history_observed=np.ones_like(sem,bool),history_poses=np.repeat(np.eye(4)[None],4,0),future_gt_occ=object())
    poses=np.repeat(np.eye(4)[None],6,0);lab=adapt.historical_labels(raw,poses,grid)
    assert (lab[:,4,4,1]==11).all() and (lab[:,1,1,1]==255).all() and (lab[:,6,6,1]==255).all() and (lab[:,0,0,0]==17).all()
    fallback=torch.nn.functional.one_hot(torch.full((8,8,4),17),18).permute(3,0,1,2)[None].float()
    p=geometry_screen.static_probability((None,raw,None,None),grid,'cpu',fallback)
    torch.testing.assert_close(p.sum(1),torch.ones(1,8,8,4),rtol=0,atol=0)
    assert float(p[0,11,4,4,1])==.75 and float(p[0,17,1,1,1])==1.


def test_exact_named_adaptation_scopes_and_no_state_change():
    from tools.fpn_unet.model import UNetRefiner
    m=UNetRefiner();before={k:v.clone() for k,v in m.state_dict().items()};named=dict(m.named_parameters())
    expected={'norm320_history':320,'norm_head482_supervised':482,'decoder_fine7106_history':7106}
    for arm,count in expected.items():
        names=adapt.parameters(m,arm);assert len(names)==len(set(names)) and sum(named[n].numel() for n in names)==count
        assert not any(n.startswith(('lift.','importance.')) or n=='empty' for n in names)
    for n,v in before.items():torch.testing.assert_close(m.state_dict()[n],v,rtol=0,atol=0)


def test_reflection_and_blur_experiment_zero_identity_and_cleanup():
    x=torch.randn(4,16,16,4,4);original=x.clone();assert torch.equal(feature_screen.blur(x,0),x)
    constant=torch.ones_like(x);torch.testing.assert_close(feature_screen.blur(constant,1),constant,rtol=0,atol=0)
    embedding=torch.nn.Embedding(19,4);body=SimpleNamespace(transport=SimpleNamespace(history_encoder=SimpleNamespace(semantic=embedding)))
    sem=torch.randint(0,19,(4,16,16,4));plain=embedding(sem)
    with pytest.raises(RuntimeError):
        with feature_screen.embedding_blur(body,1):
            assert not torch.equal(embedding(sem),plain)
            raise RuntimeError('intentional interruption')
    torch.testing.assert_close(embedding(sem),plain,rtol=0,atol=0)
    torch.testing.assert_close(x,original,rtol=0,atol=0)


def test_global_readout_calibration_ties_and_unit_gain_are_original_argmax():
    rng=np.random.default_rng(21);x=rng.normal(size=(3,18,97)).astype(np.float32);p=rng.random((18,97),dtype=np.float32)
    x[:,3,0]=2.;x[:,17,0]=3.;before=x.copy();method=dict(bias=[0.]*17+[-1.],scale=[1.]*18)
    actual=calibrate.apply_numpy(x,method);adjust=x.copy();adjust[:,17]-=1
    np.testing.assert_array_equal(actual,adjust.argmax(1))
    for h in range(3):np.testing.assert_array_equal(residual_calibrate.predict(x[h],p,1.,0.),x[h].argmax(0))
    np.testing.assert_array_equal(x,before)


def test_range_calibration_sufficient_statistics_match_full18_logits_no_mask():
    torch.manual_seed(21);logits=torch.randn(6,18,37);rho=torch.rand(6,37)
    maximum,winner=logits[:,:17].max(1);free=logits[:,17]
    for a,b in ((0.,0.),(-1.5,0.),(-1.,2.)):
        expected=logits.clone();expected[:,17]+=a+b*rho
        actual=range_calibrate.predicted(maximum.numpy(),free.numpy(),winner.numpy(),rho.numpy(),a,b)
        np.testing.assert_array_equal(actual,expected.argmax(1).numpy())
        assert actual.shape==(6,37) and actual.min()>=0 and actual.max()<=17


def test_radius_uses_conditioned_pose_relative_to_LAST_observation():
    grid=OccupancyGrid(x_min=-4,y_min=-4,z_min=-1,voxel_size=(1.,1.,1.),shape_hwd=(8,8,4))
    current=np.eye(4);current[:3,3]=[100,200,10]
    future=np.repeat(current[None],6,0);future[:,0,3]+=np.arange(1,7)
    field=range_calibrate.radius(grid,current,torch.from_numpy(future))
    point=np.array([-.5,-.5,.5]);index=np.ravel_multi_index((3,3,1),(8,8,4))
    expected=[np.hypot(point[0]+h,point[1])/40 for h in range(1,7)]
    np.testing.assert_allclose(field[:,index],expected,rtol=1e-7,atol=0)
    assert field.shape==(6,8*8*4) and float(field.min())>=0 and float(field.max())<=1
