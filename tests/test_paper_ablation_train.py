"""Real full-six forwards/gradients, matched budgets and train-only recovery."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
import torch
from tools.paper_ablation_train import model,geometry,control,engine,run,data
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.shared_history_experiments.full_gt.engine import semantic_loss


def tiny(arm,device='cpu'):
    from test_fpn_adaln_zero import fixture
    from tools.causal_forward_recovery.adapter import RecoveryPredictor
    base,old,row=fixture(device)
    body=model.make_model(base.transport.config,arm,feature_width=16,z_bins=4,memory_width=4).to(device)
    predictor=RecoveryPredictor(body,old.provider.pcfg,device,workers=1)
    raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
    prepared=predictor.prepare(row[0],raw,times=np.arange(4)*.5)
    value=(prepared[0],dict(prepared[1],future_poses=row[1]['future_poses'],
        future_gt_occ=row[1]['future_gt_occ']),prepared[2],prepared[3])
    return body,predictor,old,value


@pytest.mark.parametrize('arm',model.ARMS)
def test_removed_paths_common_fresh_state_RNG_and_checkpoint_spec(arm):
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from tools.causal_forward_full.model import make_model as reference
    cfg=LocalSTWMV17Config(history_frames=4,decoder_blocks=2)
    torch.manual_seed(21);ref=reference(cfg);rng=torch.get_rng_state().clone()
    torch.manual_seed(21);body=model.make_model(cfg,arm,expected_reference_spec=ref.configs())
    assert torch.equal(torch.get_rng_state(),rng)
    assert all(k in ref.state_dict() for k in body.state_dict())
    for k,v in body.state_dict().items(): torch.testing.assert_close(v,ref.state_dict()[k],rtol=0,atol=0)
    restored=model.from_spec(json.loads(json.dumps(body.configs())))
    restored.load_state_dict(body.state_dict(),strict=True)
    assert restored.configs()==body.configs()
    names=list(dict(body.named_parameters()))
    if arm=='c_transport': assert not any(k.startswith('columns.') or 'history_yaw' in k for k in names)
    if arm=='d_kta_refiner':
        assert all(k.startswith(('transport.history_encoder.','columns.')) for k in names)
        assert not hasattr(body.transport,'decoder')
    if arm=='m0_no_memory': assert not any(k.startswith(('columns.lift','columns.importance','columns.empty')) for k in names)
    if arm=='m1_ego_memory': assert not any('history_yaw' in k for k in names)
    if arm=='s0_pool': assert not any(k.startswith('transport.history_encoder.down') for k in names)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('arm',model.ARMS)
def test_real_six_future_forward_and_occupancy_backward_without_label_inputs(arm,device):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    body,predictor,old,row=tiny(arm,device)
    try:
        # Open only zero initialization gates for a structural path/gradient
        # test; this is neither a trained model nor a DEV quality experiment.
        with torch.no_grad():
            if model.ARMS[arm]['motion']:
                body.transport.residual_head.weight.normal_(std=.02)
                for block in body.transport.decoder: block.adaLN_modulation[-1].bias.fill_(.02)
            if model.ARMS[arm]['refiner']: body.columns.head.weight.normal_(std=.02)
        poisoned=row[0],dict(row[1],future_gt_occ=object()),row[2],row[3]
        fw=geometry.forward_window(body,predictor.provider,poisoned)
        assert fw['logits'].shape==(6,18,int(np.prod(fw['shape'])))
        loss=semantic_loss(fw['logits'],row[1]['future_gt_occ'])[0].mean()
        assert torch.isfinite(loss);loss.backward()
        if model.ARMS[arm]['motion']: assert body.transport.residual_head.weight.grad.abs().sum()>0
        if model.ARMS[arm]['refiner']: assert body.columns.head.weight.grad.abs().sum()>0
        if arm=='d_kta_refiner':
            assert torch.count_nonzero(fw['output']['residual_xy_m'])==0
            assert torch.count_nonzero(fw['output']['history_yaw_to_t0_rad'])==0
            assert body.transport.history_encoder.stem[0].weight.grad.abs().sum()>0
        if arm=='c_transport': assert fw['memory'] is None
        if arm=='m0_no_memory': assert torch.count_nonzero(fw['memory'])==0
        if arm=='m1_ego_memory': assert torch.count_nonzero(fw['output']['history_yaw_to_t0_rad'])==0
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in body.parameters())
    finally: predictor.close();old.close()


def test_ego_only_memory_world_points_independent_of_source_future_and_gradients():
    body,predictor,old,row=tiny('m1_ego_memory')
    try:
        raw=row[1];sem=torch.as_tensor(raw['history_occ']);known=torch.as_tensor(raw['history_observed'])
        lifted=body.columns.lift_history(body.transport.history_encoder(sem,known)[0])
        poses=torch.as_tensor(raw['future_poses']).double()
        own=geometry.ego_only_memory(body.columns,lifted,row[3],raw,poses,predictor.provider.pcfg.grid)
        # Existing full-memory renderer with targets equal to each historical
        # point's own source center is identical to identity world transport.
        from tools.source_scene_ego_conditioned.geometry import aligned_memory
        desc=row[3][0]['memory_reads']
        # Direct identity check includes static weights and overlaps:
        cloned=copy.deepcopy(row[3]);centers=cloned[0]['memory_reads']['centers_world']
        for i,f in zip(desc['actor'],desc['frame']): centers[i,f]=centers[i,-1]
        targets=torch.as_tensor(centers[:,-1]).double()[:,None].expand(-1,6,-1)
        n=len(targets)
        reference=aligned_memory(body.columns,lifted,dict(history_yaw_to_t0_rad=torch.zeros(n,4)),
            cloned,raw,targets,torch.zeros(n,6),poses,predictor.provider.pcfg.grid)
        torch.testing.assert_close(own,reference,rtol=0,atol=0)
        own.square().mean().backward()
        assert body.columns.lift.weight.grad.abs().sum()>0
        assert body.columns.importance.weight.grad.abs().sum()>0
    finally: predictor.close();old.close()


def test_pooling_grid_and_hidden_semantics():
    from tools.fpn_unet.model import FPNEncoder
    enc=model.PooledPyramid(FPNEncoder(z_bins=4,width=16))
    sem=torch.randint(0,18,(4,32,40,4));known=torch.ones_like(sem,dtype=torch.bool)
    p2,p3,p4=enc(sem,known)
    assert [tuple(x.shape) for x in (p2,p3,p4)]==[(4,16,16,20),(4,16,8,10),(4,16,4,5)]
    torch.testing.assert_close(p3,torch.nn.functional.avg_pool2d(p2,2,2),rtol=0,atol=0)
    torch.testing.assert_close(p4,torch.nn.functional.avg_pool2d(p2,4,4),rtol=0,atol=0)
    known[:]=False
    torch.testing.assert_close(enc(sem,known)[0],enc(sem*0,known)[0],rtol=0,atol=0)


def fake_reference(arm):
    from tools.joint_occ_experiments import run as base
    full=model.ARMS[arm]['fraction']==1.
    rows=[dict(scene_name='TRAIN'+str(i%30),t0_token=str(i),features=[0]*(0 if i%31==0 else 9)) for i in range(20430)]
    count=20430 if full else 4086
    keys=base.select_scene_balanced_round_robin(base.record_keys(rows),count)
    selected=base.align_records(rows,keys)
    sizes=[len(base.epoch_groups(selected,20261011,e,4,128)) for e in range(2 if full else 1)]
    return rows,dict(protocol=control.FULL if full else control.SMALL,epochs=2 if full else 1,
        world_size=2 if full else 1,train_keys=[list(k) for k in keys],eval_keys=[['VAL','t0']],
        total_updates=sum(sizes),epochs_updates=sizes,parameters=2816122,seed=20261011,
        objective=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.),learning_rates=[5e-4,3e-4],
        source_policy='causal_forward_CV_history',model=dict(source_policy='causal_forward_CV_history'),
        known='TRAIN_OCC_STC_dense_valid',registration='none',
        schedule=f'whole_{2 if full else 1}_epoch_cosine_floor0.01')


@pytest.mark.parametrize('arm',model.ARMS)
def test_exact_plan_reference_schedule_and_no_evaluation_reads(tmp_path,arm):
    rows,c=fake_reference(arm);p=tmp_path/'contract.json';p.write_text(json.dumps(c))
    assert control.reference(p,arm)==c
    keys,groups,budget=control.plan(rows,c,arm)
    assert len(keys)==budget['train_windows'] and len(groups)==budget['stop_updates']
    assert sum(map(len,groups))==len(keys) and budget['schedule_reference_updates']==c['total_updates']
    for field,value in (('protocol','old'),('world_size',3),('schedule','whole_1_epoch_wrong')):
        bad=dict(c,**{field:value});p.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError): control.reference(p,arm)
    c['train_keys'][0],c['train_keys'][1]=c['train_keys'][1],c['train_keys'][0]
    with pytest.raises(RuntimeError,match='ordered'): control.plan(rows,c,arm)


def test_train_inputs_hashes_never_read_val_or_stc(tmp_path,monkeypatch):
    c=dict(seed=20261011,world_size=1,learning_rates=[5e-4,3e-4],overrides=[],
        precision='BF16_network_FP64_geometry_full6',execution=dict(torch=str(torch.__version__),
            cuda=torch.version.cuda,cpu_workers=4,prefetch_workers=2),data={},implementation={},model={})
    a=SimpleNamespace(seed=20261011,motion_lr=5e-4,repair_lr=3e-4,override=[],cpu_workers=4,prefetch_workers=2)
    for name in control.TRAIN_FILES:
        p=tmp_path/name;p.write_text('train fixture');setattr(a,name,p);c['data'][name]=control.sha(p)
    for name in ('dev_cache','dev_info','eval_reference'): c['data'][name]='MISSING_VAL_MUST_NOT_OPEN'
    assert set(control.check_inputs(a,SimpleNamespace(world_size=1),c,tmp_path))==set(control.TRAIN_FILES)
    with pytest.raises(RuntimeError,match='world_size'): control.check_inputs(a,SimpleNamespace(world_size=2),c,tmp_path)


def test_RAM_tracks_bounded_no_disk_and_input_sensitive():
    cache=data.MemoryTrackCache(ram_mib=.001);calls=[]
    raw={k:np.zeros((2,2)) for k in ('history_occ','history_observed','history_poses','future_poses')}
    def build(): calls.append(1);return np.zeros(80,np.float64)
    assert cache.get_or_build(('s','t'),raw,build)[1] is False
    assert cache.get_or_build(('s','t'),raw,build)[1] is True
    raw['history_occ'][0,0]=1
    assert cache.get_or_build(('s','t'),raw,build)[1] is False
    assert len(calls)==2 and cache.bytes<=cache.limit and len(cache.rows)==1
    cache.close();assert cache.bytes==0


def test_cli_has_no_eval_or_stc_flags():
    help=run.parser().format_help()
    assert all(name not in help for name in ('--dev-cache','--stc-root','--eval-reference','--evaluate','--epochs'))
    assert '--reference-contract' in help and '--resume' in help


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_train_loop_pause_resume_matches_weights_Adam_and_RNG(tmp_path,device):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    arm='m0_no_memory';body,predictor,old,row=tiny(arm,device)
    original=copy.deepcopy(body.state_dict())
    ctx=DistributedTraining();ctx.device=torch.device(device)
    groups=[[row],[row],[row]];contract=dict(arm=arm,parameters=sum(p.numel() for p in body.parameters()),
        world_size=1,budget=dict(run_epochs=1,train_windows=3,stop_updates=3,schedule_reference_updates=3,
            schedule='whole_1_epoch_cosine_floor0.01'))
    args=SimpleNamespace(motion_lr=5e-4,repair_lr=3e-4)
    make_batches=lambda values:(v for v in values)
    try:
        results=[]
        for name in ('whole','resumed'):
            body.load_state_dict(original);run.seed_all(21);rng=np.random.default_rng(21)
            opt=run.optimizer_for(body,args);out=tmp_path/name;out.mkdir()
            value=engine.execute(body,opt,predictor.provider,groups,contract,ctx,rng,threading.Event(),out,make_batches,
                max_updates=1 if name=='resumed' else 0)
            if name=='resumed':
                assert value['status']=='interrupted' and value['updates']==1 and not (out/'epoch_0001.pt').exists()
                # Deliberately perturb weights and all RNGs before recovery.
                with torch.no_grad(): next(body.parameters()).add_(42)
                run.seed_all(77);rng=np.random.default_rng(77)
                value=engine.execute(body,opt,predictor.provider,groups,contract,ctx,rng,threading.Event(),out,make_batches,
                    resume=out/'last.pt')
            assert value['status']=='complete' and value['updates']==3 and value['epoch']==1
            saved=torch.load(out/'epoch_0001.pt',weights_only=False)
            assert saved['training_only'] and 'evaluations' not in saved and saved['updates']==3
            assert len(list(out.glob('*.pt')))<=3
            assert len((out/'progress.jsonl').read_text().splitlines())==3
            results.append(saved)
            # A completed recovery cannot perform extra optimizer steps.
            engine.execute(body,opt,predictor.provider,groups,contract,ctx,rng,threading.Event(),out,
                lambda _:pytest.fail('completed checkpoint must not load data'),resume=out/'last.pt')
        def equal(a,b):
            if torch.is_tensor(a): torch.testing.assert_close(a,b,rtol=0,atol=0)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a: equal(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                for x,y in zip(a,b): equal(x,y)
            elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
            else: assert a==b
        for key in ('models','optimizers','rank_rng'): equal(results[0][key],results[1][key])
        for field in ('arm','world_size','budget'):
            bad=copy.deepcopy(contract);bad[field]='changed'
            with pytest.raises(RuntimeError,match='SAME'): control.validate_resume(results[0],bad,groups)
    finally: predictor.close();old.close();ctx.close()


def test_transport_only_empty_sources_skip_AdamW_not_fake_gradients(monkeypatch):
    body=torch.nn.Linear(2,2);opt=torch.optim.AdamW(body.parameters(),lr=.1,weight_decay=.1)
    opt.param_groups[0]['initial_lr']=.1
    before=copy.deepcopy(body.state_dict());ctx=DistributedTraining()
    monkeypatch.setattr(engine,'forward_window',lambda *a:dict(logits=torch.zeros(6,18,4)))
    row=({},dict(future_gt_occ=np.full((6,4),17)))
    stats=engine.train_step(body,opt,SimpleNamespace(device=torch.device('cpu')),[row],ctx,
        global_windows=1,update=1,schedule_updates=10)
    assert not stats['optimizer_updated'] and not opt.state
    for k,v in body.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_actual_transport_only_window_with_no_sources():
    body,predictor,old,row=tiny('c_transport')
    try:
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        raw['history_occ']=np.full_like(raw['history_occ'],17)
        raw['history_observed']=np.ones_like(raw['history_observed'],bool)
        p=predictor.prepare(row[0],raw,times=np.arange(4)*.5)
        row=(p[0],dict(p[1],future_poses=row[1]['future_poses'],
            future_gt_occ=np.full_like(row[1]['future_gt_occ'],17)),p[2],p[3])
        assert len(row[0]['source_class_id'])==0
        opt=run.optimizer_for(body,SimpleNamespace(motion_lr=5e-4,repair_lr=3e-4))
        before=copy.deepcopy(body.state_dict());ctx=DistributedTraining()
        stats=engine.train_step(body,opt,predictor.provider,[row],ctx,global_windows=1,update=1,schedule_updates=10)
        assert not stats['optimizer_updated'] and not opt.state
        for k,v in body.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    finally: predictor.close();old.close()


def test_attach_existing_TRAIN_cache_requires_manifest_and_no_VAL(tmp_path,monkeypatch):
    import hashlib
    monkeypatch.setattr(data,'namespace',lambda *args:'test_training_geometry_only')
    digest=hashlib.sha256((data.CACHE_PROTOCOL+'test_training_geometry_only').encode()).hexdigest()
    args=SimpleNamespace(ccr_history_cache=tmp_path,geometry_ram_mib=0,track_ram_mib=0)
    provider=data.TrainProvider.__new__(data.TrainProvider)
    source=object()
    with pytest.raises(RuntimeError,match='NO rebuild'): data.attach_train_cache(provider,args,tmp_path,source)
    folder=tmp_path/digest;folder.mkdir()
    m=dict(complete=True,namespace=digest,windows=20430,artifacts=20430,protocol=data.TRAIN_PROTOCOL,
        future_GT_cached=False,learned_outputs_cached=False,sampled_ids_cached=False,optimizer_or_rng_cached=False)
    (folder/'manifest.json').write_text(json.dumps(m))
    before=sorted(p.name for p in folder.iterdir())
    policy=data.attach_train_cache(provider,args,tmp_path,source)
    try:
        assert not policy['persistent_writes'] and not policy['VAL_opened'] and not policy['STC_opened']
        assert provider.ccr_history_cache_source is source and provider.ccr_val_history_cache is None
        assert provider.ccr_history_cache.limit==0
        assert before==sorted(p.name for p in folder.iterdir())
    finally: provider.close_tracks();provider.ccr_history_cache.close()


def test_training_launcher_single_TRAIN_source_and_no_eval_after_final(tmp_path,monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA')
    from tools.causal_forward_full.model import make_model as reference
    body,predictor,old,row=tiny('m0_no_memory','cuda');calls=[]
    ref=reference(body.transport.config,feature_width=16,z_bins=4,memory_width=4).configs()
    c=dict(model=ref,objective=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.),
        precision='BF16_network_FP64_geometry_full6',learning_rates=[5e-4,3e-4],eval_keys=[['NO_VAL_IO','t0']],
        execution=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cpu_workers=4,prefetch_workers=2))
    contract_path=tmp_path/'reference.json';contract_path.write_text(json.dumps(c))
    monkeypatch.setattr(run,'reference',lambda *a:c)
    monkeypatch.setattr(run,'check_inputs',lambda *a:{})
    monkeypatch.setattr(run.base,'load_cache',lambda *a:({},[]))
    monkeypatch.setattr(run.base,'load_runtime_config',lambda *a:None)
    monkeypatch.setattr(run.base,'make_prepare_config',lambda *a:predictor.provider.pcfg)
    budget=dict(run_epochs=1,train_windows=2,stop_updates=2,schedule_reference_updates=2,
        schedule='whole_1_epoch_cosine_floor0.01')
    monkeypatch.setattr(run,'plan',lambda *a:([['TRAIN','1'],['TRAIN','2']],[[row],[row]],budget))
    def provider_factory(*a):
        p=SimpleNamespace(**vars(predictor.provider));p.joint=a[5]
        p.close_tracks=lambda:None
        p.ccr_history_cache=SimpleNamespace(close=lambda:None)
        return p
    monkeypatch.setattr(run,'TrainProvider',provider_factory)
    monkeypatch.setattr(run,'attach_train_cache',lambda *a:dict(VAL_opened=False,STC_opened=False))
    def source(*a,**kwargs): calls.append(kwargs['info_pkl']);return object()
    monkeypatch.setattr(run.base,'NuScenesWindowSource',source)
    monkeypatch.setattr(run.base,'CachedColumnSource',lambda source,*a,**kw:source)
    monkeypatch.setattr(run,'batches',lambda provider,source,groups,history,**kw:(g for g in groups))
    argv=['--arm','m0_no_memory','--out-dir',str(tmp_path/'new'),'--reference-contract',str(contract_path)]
    for name in ('config','train-cache','base-checkpoint','dataroot','train-info','ccr-history-cache'):
        argv.extend(['--'+name,'TRAIN_ONLY_'+name])
    ctx=DistributedTraining();ctx.device=torch.device('cuda:0')
    try:
        assert run.launch(run.parser().parse_args(argv),threading.Event(),ctx)==0
        assert calls==['TRAIN_ONLY_train-info']
        saved=torch.load(tmp_path/'new/epoch_0001.pt',weights_only=False)
        assert saved['updates']==2 and saved['epoch']==1 and saved['training_only']
        assert saved['contract']['evaluation']=='DISABLED' and 'evaluations' not in saved
        assert not list((tmp_path/'new').glob('*eval*state*'))
        assert len(list((tmp_path/'new').glob('*.pt')))==2
    finally: predictor.close();old.close()
