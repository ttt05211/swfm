import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real_motion.canonical_causal_repair import (
    CanonicalEvidence, CanonicalRepairHead, FEATURE_DIM, STATIC,
    build_canonical_evidence, build_compact_canonical_support, map_canonical_evidence,
    materialize_compact_canonical,
)
from real_motion.geometry import OccupancyGrid
from real_motion.surface_canonical_repair import (
    SurfaceAtlas, SurfaceCanonicalRepairHead, SURFACE_DIM, PHASE_DIM,
    augment_evidence, augment_projection,
)
from tools.real_motion import ccr_screen_common as screen
from tools.real_motion.height_field_screen_recovery import payload, restore
from tools.real_motion import surface_ccr_screen_common as surface_screen
from test_canonical_causal_repair import scene
from test_ccr_screen import fixture
from test_height_field_screen import assert_nested_equal, contract


def assert_geometry_equal(a,b):
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_geometry_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for left,right in zip(a,b):assert_geometry_equal(left,right)
    else:assert a==b


def evidence_at(points, cls=11):
    points=np.asarray(points,np.float64).reshape(-1,3);n=len(points)
    return CanonicalEvidence(np.zeros((n,FEATURE_DIM),np.float32),np.full((n,4),cls,np.uint8),
        np.full(n,STATIC,np.int32),np.full(n,cls,np.uint8),points,np.ones((n,4),bool),{})


def test_surface_height_slope_unknown_and_stacked_layers_are_not_collapsed():
    grid=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1.,1.,1.),shape_hwd=(12,12,8))
    points=np.array([[x,y,1+.2*x] for x in range(3) for y in range(3)],np.float64)
    history=evidence_at(np.concatenate((points,points+[0,0,4])))
    atlas=SurfaceAtlas(history.world,history.classes,history.presence,history.actor,np.eye(4),grid)
    query=evidence_at([[1,1,1.2],[1,1,1.7],[1,1,5.2],[10,10,2]])
    f=atlas.describe(query)
    assert f.shape==(4,SURFACE_DIM) and np.isfinite(f).all()
    assert f[0,0]==f[1,0]==f[2,0]==1 and f[3,0]==0
    assert abs(f[0,1])<1e-5 and abs(f[2,1])<1e-5
    assert f[1,1]==pytest.approx(.5,abs=.002)
    assert f[0,4]==pytest.approx(.2,abs=.002)
    assert f[0,2]<.002 and f[2,2]<.002
    assert f[3,7]==1  # unknown != a confidently fitted zero-height road


@pytest.mark.parametrize('limit',[1,4000000])
def test_compact_atlas_sample_and_full_descriptors_keep_population_and_no_gt(limit):
    grid,prep=scene()
    prep.raw['history_occ'][:,4:7,4:7,0]=11
    plain=build_canonical_evidence(prep,grid,max_lattice_cells=limit)
    compact=build_compact_canonical_support(prep,grid,max_lattice_cells=limit)
    before=copy.deepcopy(compact)
    full=SurfaceAtlas(plain.world,plain.classes,plain.presence,plain.actor,prep.state['current_pose'],grid)
    cached=SurfaceAtlas.from_compact(compact,prep,grid)
    a=augment_evidence(plain,full);b=augment_evidence(plain,cached)
    np.testing.assert_array_equal(a.features,b.features)
    prep.raw['future_gt_occ']='POISON';prep.raw['future_annotations']='POISON'
    ids=np.arange(0,len(plain),2)
    sampled=materialize_compact_canonical(compact,prep,grid,ids)
    np.testing.assert_array_equal(augment_evidence(sampled,cached).features,a.features[ids])
    for key in ('actor','classes','world','presence','labels'):
        np.testing.assert_array_equal(getattr(a,key),getattr(plain,key))
    assert len(a)==len(plain)
    assert_geometry_equal(before.layouts,compact.layouts)
    with pytest.raises(ValueError,match='unaugmented'):
        augment_evidence(a,cached)


def test_live_projection_phase_changes_by_horizon_without_changing_destination():
    grid,prep=scene();plain=build_canonical_evidence(prep,grid)
    atlas=SurfaceAtlas(plain.world,plain.classes,plain.presence,plain.actor,prep.state['current_pose'],grid)
    evidence=augment_evidence(plain,atlas)
    prep.raw['future_poses'][:,0,3]=np.arange(6)*.07
    prep.raw['future_poses'][:,2,3]=np.arange(6)*.03
    prep.state['world_to_future']=np.linalg.inv(prep.raw['future_poses'])
    plan=map_canonical_evidence(plain,prep,grid)
    enriched=augment_projection(evidence,plan,prep.state['current_pose'],prep.state['world_to_future'],grid)
    for key in ('flat','base','fallback','legal'):
        np.testing.assert_array_equal(getattr(plan,key),getattr(enriched,key))
    np.testing.assert_array_equal(enriched.context[...,:8],plan.context)
    assert np.isfinite(enriched.context).all()
    assert not np.array_equal(enriched.context[plain.actor==STATIC,0,8:],enriched.context[plain.actor==STATIC,5,8:])
    assert (enriched.context[plain.actor>=0,:,8:]==0).all()
    bad=replace(plan,flat=plan.flat.copy());bad.flat[plain.actor==STATIC]=-1
    with pytest.raises(RuntimeError,match='destination'):
        augment_projection(evidence,bad,prep.state['current_pose'],prep.state['world_to_future'],grid)


def head_inputs(device):
    torch.manual_seed(17)
    n=24; actors=torch.tensor(([0,-2,-2]*8),device=device)
    classes=torch.where(actors>=0,4,11)
    features=torch.randn(n,FEATURE_DIM+SURFACE_DIM,device=device)
    labels=classes[:,None].expand(-1,4)
    context=torch.randn(n,6,8+PHASE_DIM,device=device)
    base=torch.full((n,6),17,device=device)
    legal=torch.ones(n,6,2,dtype=torch.bool,device=device)
    output={'history_source_context':torch.randn(1,8,device=device,requires_grad=True),
            'future_transport_queries':torch.randn(1,6,8,device=device,requires_grad=True)}
    return features,labels,actors,classes,context,base,legal,output


def forward(head,inputs,plain=False):
    feat,labels,actors,classes,context,base,legal,output=inputs
    live=head.project_sources(output)
    encoded=head.encode(feat[:,:FEATURE_DIM] if plain else feat,labels,actors,classes,live)
    logits=head.decode(encoded,actors,context[...,:8] if plain else context,base,base,legal,live)
    return logits


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_zero_geometry_initialization_and_frozen_dynamic_logits_are_byte_exact(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    inputs=head_inputs(device);base=CanonicalRepairHead(8,16).to(device)
    head=SurfaceCanonicalRepairHead(8,16).to(device);head.initialize_from(base)
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
        old=forward(base,inputs,plain=True);new=forward(head,inputs)
    assert torch.equal(old,new)
    head.freeze_validation()
    with torch.no_grad():
        head.surface.weight.normal_();head.phase.weight.normal_();head.static_readout[-1].bias.add_(1.)
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
        changed=forward(head,inputs)
    dynamic=inputs[2]>=0
    assert torch.equal(old[dynamic],changed[dynamic])
    assert not torch.equal(old[~dynamic],changed[~dynamic])
    assert (head.probabilities(changed,inputs[2])[...,1]==0).all()


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('role',['static','dynamic','mixed'])
def test_role_routing_matches_full_dual_readout_probability_bytes(device,role):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    inputs=list(head_inputs(device));head=SurfaceCanonicalRepairHead(8,16).to(device)
    if role!='mixed':inputs[2].fill_(-2 if role=='static' else 0)
    with torch.no_grad():head.surface.weight.normal_();head.phase.weight.normal_()
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
        original=forward(head,inputs)
        inputs[-1]=head.inference_batch(inputs[-1],inputs[2].cpu().numpy())
        routed=forward(head,inputs)
    assert torch.equal(original,routed)


def test_clean_joint_graph_reaches_original_motion_context_without_extra_module():
    inputs=head_inputs('cpu');head=SurfaceCanonicalRepairHead(8,16)
    head.static_only_training=False;head.requires_grad_(True)
    logits=forward(head,inputs)
    screen._add_only_natural_loss(head,logits,inputs[2],torch.ones_like(logits),torch.ones_like(logits)).backward()
    output=inputs[-1]
    assert output['history_source_context'].grad.abs().sum()>0
    assert output['future_transport_queries'].grad.abs().sum()>0
    assert head.surface.weight.grad.abs().sum()>0 and head.phase.weight.grad.abs().sum()>0


def test_weighted_static_add_loss_keeps_train_weight_and_ignores_dynamic_remove():
    head=SurfaceCanonicalRepairHead(8,16);head.freeze_validation()
    head.positive_weight[0,0]=3.;logits=torch.zeros(2,6,2,requires_grad=True)
    actor=torch.tensor([-2,0]);target=torch.ones_like(logits);weight=torch.ones_like(logits)
    loss=screen._add_only_natural_loss(head,logits,actor,target,weight)
    assert loss.item()==pytest.approx(3*np.log(2))
    loss.backward()
    assert (logits.grad[1]==0).all() and (logits.grad[...,1]==0).all()


def training_fixture(device):
    teacher,provider,rows,baseline=fixture(device)
    head=SurfaceCanonicalRepairHead(8).to(device);head.initialize_from(baseline);head.freeze_validation()
    provider.ccr_batched_head=True;provider.ccr_add_only_natural_bce=True
    grid=provider.pcfg.grid
    def augment(sample,plan,prep):
        full=build_canonical_evidence(prep,grid)
        atlas=SurfaceAtlas(full.world,full.classes,full.presence,full.actor,prep.state['current_pose'],grid)
        sample=augment_evidence(sample,atlas)
        return sample,augment_projection(sample,plan,prep.state['current_pose'],prep.state['world_to_future'],grid)
    provider.ccr_augment_sample=augment
    return teacher,provider,rows,head


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_static_optimizer_step_preserves_base_and_releases_graph(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    teacher,provider,rows,head=training_fixture(device)
    old=copy.deepcopy(head.state_dict());teacher_old=copy.deepcopy(teacher.state_dict())
    opt=torch.optim.AdamW(head.parameters(),lr=.001);rng=np.random.default_rng(23);memory=[]
    for _ in range(4):
        stat=screen.train_step(provider,rows*4,teacher,head,opt,rng)
        assert stat['optimizer_updated'] and np.isfinite(stat['loss'])
        assert all(p.grad is None for p in head.parameters())
        memory.append(stat['allocated_after_mib'])
    for key in CanonicalRepairHead(8).state_dict():
        assert torch.equal(old[key],head.state_dict()[key]),key
    assert not torch.equal(old['static_readout.3.bias'],head.state_dict()['static_readout.3.bias'])
    assert_nested_equal(teacher_old,teacher.state_dict())
    assert max(memory[1:])-min(memory[1:])<1
    provider.ccr_cache.close()


def test_static_next_update_resume_restores_head_optimizer_rng_and_contract():
    teacher,provider,rows,head=training_fixture('cpu')
    opt=torch.optim.AdamW(head.parameters(),lr=.001);rng=np.random.default_rng(25)
    screen.train_step(provider,rows,teacher,head,opt,rng)
    c={**contract(),'model':surface_screen.model_contract(head),'static_only_training':True}
    saved=copy.deepcopy(payload(head,opt,rng,c,epoch=0,batch=1,updates=1,executed=1,
        reports={'train_prior':{'TRAIN_only':True}},protocol=surface_screen.PROTOCOL))
    other=copy.deepcopy(head);other_opt=torch.optim.AdamW(other.parameters(),lr=99);other_rng=np.random.default_rng(9)
    screen.train_step(provider,rows,teacher,head,opt,rng)
    restore(saved,other,other_opt,other_rng,c,protocol=surface_screen.PROTOCOL)
    screen.train_step(provider,rows,teacher,other,other_opt,other_rng)
    assert_nested_equal(head.state_dict(),other.state_dict());assert_nested_equal(opt.state_dict(),other_opt.state_dict())
    assert rng.bit_generator.state==other_rng.bit_generator.state
    with pytest.raises(RuntimeError,match='identical'):
        restore(saved,other,other_opt,other_rng,{**c,'static_only_training':False},protocol=surface_screen.PROTOCOL)
    provider.ccr_cache.close()


def test_completed_epoch2_warm_start_accepts_pinned_weight_role_not_old_optimizer(tmp_path):
    old=CanonicalRepairHead(8);old.positive_weight.copy_(torch.tensor([[3.0331,2.],[2.8068,2.]]))
    opt=torch.optim.AdamW(old.parameters(),lr=.002);rng=np.random.default_rng(32)
    c={**contract(),'protocol':screen.PROTOCOL,'teacher_sha256':'teacher',
       'config_fingerprint':'config','dev_manifest_fingerprint':'fixed',
       'model':screen.model_contract(old),'thresholds':{'CCR_ADD':.5,'CCR_REMOVE':.95},
       'epochs':3,'epoch_batches':[1,1,1],'epoch_batch_sizes':[[1],[1],[1]],'schedule_steps':3}
    saved=payload(old,opt,rng,c,epoch=2,batch=0,updates=2,executed=2,
        reports={'train_prior':{'positive_weights':old.positive_weight.tolist(),'TRAIN_only':True}},
        protocol=screen.PROTOCOL)
    path=tmp_path/'frozen_b.pt';torch.save(saved,path)
    head=SurfaceCanonicalRepairHead(8)
    info=surface_screen.warm_start_head(head,path,teacher_sha256='teacher',config_fingerprint='config',
                                       dev_manifest_fingerprint='fixed')
    assert info['source_epoch']==2 and 'optimizer' not in info
    assert info['train_prior']['TRAIN_only']
    for key in old.state_dict():assert torch.equal(old.state_dict()[key],head.state_dict()[key])
    assert head.static_only_training and not head.encoder[0].weight.requires_grad
    assert head.static_readout[1].weight.requires_grad
    with pytest.raises(RuntimeError,match='manifest'):
        surface_screen.warm_start_head(head,path,teacher_sha256='teacher',config_fingerprint='config',
                                       dev_manifest_fingerprint='other')


def test_surface_contract_rejects_missing_cache_changed_loss_and_fps_budget(tmp_path):
    from tools.real_motion.train_p0_f9_height_shared_field import parser
    p=parser();surface_screen.add_args(p)
    paths=['checkpoint','train-cache','dev-cache','population-manifest','base-checkpoint',
           'dataroot','train-info','dev-info','out-dir']
    a=p.parse_args([item for name in paths for item in ('--'+name,str(tmp_path/name))])
    a.warm_start_head=str(tmp_path/'b.pt');(tmp_path/'b.pt').write_bytes(b'test')
    with pytest.raises(ValueError,match='cache'):
        surface_screen.contract_extra(a,tmp_path)
    a.ccr_history_cache=str(tmp_path/'train')
    a.ccr_add_only_natural_bce=True
    with pytest.raises(ValueError,match='weighted'):
        surface_screen.contract_extra(a,tmp_path)
    a.ccr_add_only_natural_bce=False;a.fps_windows=6
    with pytest.raises(ValueError,match='20-window'):
        surface_screen.contract_extra(a,tmp_path)


def test_surface_cli_stop_resume_actual_optimizer_and_full_monitor_fps_pool(monkeypatch,tmp_path):
    # Mock external IO/metrics, not the sampled head/optimizer/augmentation.
    from test_height_field_screen import mock_cli
    from real_motion.canonical_repair_context import FixedCanonicalCache
    cli,argv=mock_cli(monkeypatch,tmp_path)
    bpath=tmp_path/'b.pt';bpath.write_bytes(b'fixture')
    argv+=['--epochs','3','--train-fraction','.2','--eval-windows','64','--fps-windows','20','--speed-repeats','3',
           '--warm-start-head',str(bpath)]
    monkeypatch.setattr(surface_screen,'contract_extra',lambda *_: {'static_only_training':True})
    def warm(head,*args,**kwargs):
        head.freeze_validation()
        return {'train_prior':{'positive_weights':head.positive_weight.tolist(),'TRAIN_only':True}}
    monkeypatch.setattr(surface_screen,'warm_start_head',warm)
    def setup(provider,args):
        provider.ccr_cache=FixedCanonicalCache(1,neighbors=False);provider.ccr_samples_per_role=8
        provider.ccr_batched_head=True;provider.ccr_add_only_natural_bce=True
        grid=provider.pcfg.grid
        def augment(sample,plan,prep):
            full=build_canonical_evidence(prep,grid)
            atlas=SurfaceAtlas(full.world,full.classes,full.presence,full.actor,prep.state['current_pose'],grid)
            sample=augment_evidence(sample,atlas)
            return sample,augment_projection(sample,plan,prep.state['current_pose'],prep.state['world_to_future'],grid)
        provider.ccr_augment_sample=augment
    monkeypatch.setattr(surface_screen,'setup',setup)
    def evaluate(*args,**kwargs):
        result=cli.evaluate(*args,**kwargs)
        result['variants']['frozen_B']=copy.deepcopy(result['variants']['joint'])
        for row in result['variants'].values():row['metrics']['MovingMacro']=30.
        return result
    monkeypatch.setattr(surface_screen,'evaluate',evaluate)
    def speed(provider,source,records,teacher,head,**kwargs):
        assert len(records)==64  # not first 20, preserving official selection
        result=cli.six_frame_speed()
        result['quality_reference']=kwargs['quality_report']['variants']['frozen_B']['metrics']
        result['dynamic_byte_parity_windows']=20
        result['six_frame_amortized_FPS']['surface_CCR']=60.
        return result
    monkeypatch.setattr(surface_screen,'six_frame_speed',speed)
    full,stop,resume=[tmp_path/n for n in ('full','stop','resume')]
    assert cli.main(argv=argv+['--out-dir',str(full)],backend=surface_screen)==0
    assert cli.main(argv=argv+['--out-dir',str(stop),'--max-updates','1'],backend=surface_screen)==0
    assert cli.main(argv=argv+['--out-dir',str(resume),'--resume',str(stop/'last.pt')],backend=surface_screen)==0
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(resume/'last.pt',weights_only=False)
    assert a['epoch']==b['epoch']==3 and a['updates']==b['updates']==3
    assert_nested_equal(a['head'],b['head']);assert_nested_equal(a['optimizer'],b['optimizer'])
    assert a['numpy_rng']==b['numpy_rng'] and torch.equal(a['torch_rng'],b['torch_rng'])
    assert [r['epoch'] for r in b['reports']['epochs']]==[1,2,3]
    # Stop AFTER final dev evaluation, before FPS, then resume only unfinished
    # reports. A process-local quality variable must not be required.
    def interrupted_speed(*args,**kwargs):raise InterruptedError('injected FPS interruption')
    monkeypatch.setattr(surface_screen,'six_frame_speed',interrupted_speed)
    interrupted=tmp_path/'fps_interrupted';finished=tmp_path/'fps_resumed'
    assert cli.main(argv=argv+['--out-dir',str(interrupted)],backend=surface_screen)==0
    saved=torch.load(interrupted/'last.pt',weights_only=False)
    assert saved['epoch']==3 and 'final_dev512' in saved['reports'] and 'speed' not in saved['reports']
    monkeypatch.setattr(surface_screen,'six_frame_speed',speed)
    assert cli.main(argv=argv+['--out-dir',str(finished),'--resume',str(interrupted/'last.pt')],backend=surface_screen)==0
    completed=torch.load(finished/'last.pt',weights_only=False)
    assert_nested_equal(a['head'],completed['head']);assert_nested_equal(a['optimizer'],completed['optimizer'])
    assert (finished/'frozen_B_snapshot.pt').is_file()


def test_formal_speed_callback_keeps_future_phase_inside_forecast_and_dynamic_parity(monkeypatch):
    from real_motion.canonical_causal_repair import compose_canonical
    grid,prep=scene();plain=build_canonical_evidence(prep,grid)
    provider=SimpleNamespace(device=torch.device('cpu'),pcfg=SimpleNamespace(grid=grid),surface_fps_cpu_workers=2)
    teacher=torch.nn.Module();teacher.transport=torch.nn.Identity()
    base=CanonicalRepairHead(8);head=SurfaceCanonicalRepairHead(8);head.initialize_from(base);head.freeze_validation()
    output={'history_source_context':torch.randn(1,8),'future_transport_queries':torch.randn(1,6,8)}
    records=[dict(scene_name='scene-'+str(i%18),t0_token=str(i),features=torch.zeros(1+i%3,8)) for i in range(64)]
    def history(*args,**kwargs):
        return SimpleNamespace(canonical_evidence=plain,current_pose=np.eye(4),future_poses=[np.eye(4)]*6)
    calls=[]
    def forecast(h,provider,motion,model,probability_fn,**kwargs):
        plan=map_canonical_evidence(h.canonical_evidence,prep,grid)
        # The persisted/prepared history never contains live future phase.
        assert plan.context.shape[-1]==8
        prob=probability_fn(model,h.canonical_evidence,plan,output,provider.device)
        calls.append(type(model))
        dense=compose_canonical(prep.baseline,h.canonical_evidence,plan,prob[...,0],prob[...,1])
        return dict(probability=prob,dense=dense,stages_seconds={'live_phase_and_model':.001})
    monkeypatch.setattr(surface_screen,'prepare_history',history)
    monkeypatch.setattr(surface_screen,'forecast_six',forecast)
    quality={'variants':{'frozen_B':{'metrics':{'mIoU':40.,'MovingMicro':30.}}}}
    report=surface_screen.six_frame_speed(provider,None,records,teacher,head,repeats=3,quality_report=quality)
    assert report['dynamic_byte_parity_windows']==20 and len(report['trials'])==120
    assert calls.count(SurfaceCanonicalRepairHead)==80 and calls.count(CanonicalRepairHead)==80
    assert report['quality_reference']==quality['variants']['frozen_B']['metrics']
    assert report['persistent_val_cache_used'] is False and report['memory_mib'] is None
    assert head.static_readout[1].weight.requires_grad and not head.encoder[0].weight.requires_grad
