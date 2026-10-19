"""Pose contract, cross-first arithmetic, no Q memory path, real gradients and recovery."""
import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_scene_ego_conditioned import model as module, adapter, geometry, evaluation, run, report

KW = dict(feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32)


def fixture(device='cpu'):
    from test_source_scene_stage2 import fixture as previous
    original, runner, row = previous(device)
    config = replace(original.transport.config, decoder_blocks=2)
    body = module.make_model(config, **KW).to(device)
    return body, runner, row


def test_fresh_common_initialization_RNG_zero_ego_and_no_source_query():
    body, runner, row = fixture()
    try:
        torch.manual_seed(21); control = module.fresh_control(body.transport.config, **KW); expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(21); body = module.make_model(body.transport.config, **KW)
        torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
        for k,v in body.state_dict().items():
            if k in control.state_dict(): torch.testing.assert_close(v,control.state_dict()[k],rtol=0,atol=0)
        assert not hasattr(body.columns,'source_query') and not any('source_query' in n for n,_ in body.named_parameters())
        kta=torch.as_tensor(row[0]['kta_displacement_xy_m']); ego=torch.randn(6,4)
        initial=body.transport.initial_queries(kta,ego,torch.float32)
        torch.testing.assert_close(initial,control.transport.future_query.expand(len(kta),-1,-1)+control.transport.kta_future_proj(kta/20),rtol=0,atol=0)
        torch.testing.assert_close(initial,body.transport.initial_queries(kta,ego+3.,torch.float32),rtol=0,atol=0)
        assert body.transport.future_query.shape[1]==6
        assert all(isinstance(b,module.CrossFirstFutureBlock) for b in body.transport.decoder)
        saved=dict(protocol=run.PROTOCOL,contract=dict(model=body.configs()),models={'gt':body.state_dict()})
        restored=module.checkpoint_model(saved)
        for k,v in body.state_dict().items(): torch.testing.assert_close(restored.state_dict()[k],v,rtol=0,atol=0)
        with pytest.raises(RuntimeError): module.checkpoint_model(dict(saved,protocol='old_integrated'))
        with pytest.raises(ValueError): module.make_model(replace(body.transport.config,decoder_blocks=4),**KW)
        with pytest.raises(ValueError): body.transport.initial_queries(kta,None,torch.float32)
    finally: runner.close()


def pose(x,y,yaw):
    c,s=np.cos(yaw),np.sin(yaw); p=torch.eye(4,dtype=torch.float64)
    p[:2,:2]=torch.tensor([[c,-s],[s,c]],dtype=p.dtype); p[:2,3]=torch.tensor([x,y],dtype=p.dtype)
    return p


def test_relative_t0_ego_world_gauge_and_differentiable_pose_contract():
    t0=pose(100.,-30.,np.pi/2)
    relative=torch.stack([pose(h*.2,h*.1,.01*h) for h in range(1,7)])
    future=(t0@relative).requires_grad_()
    features=module.relative_ego_features(t0,future)
    expected=torch.tensor([[h*.2,h*.1,np.sin(.01*h),np.cos(.01*h)] for h in range(1,7)],dtype=torch.float32)
    torch.testing.assert_close(features,expected)
    gauge=pose(-40.,50.,-.8)
    torch.testing.assert_close(module.relative_ego_features(gauge@t0,gauge@future),features)
    features.square().sum().backward(); assert future.grad is not None and torch.isfinite(future.grad).all()
    for bad in (torch.zeros(5,4,4),torch.full((6,4,4),float('nan')),torch.zeros(6,4,4)):
        with pytest.raises(ValueError): module.relative_ego_features(t0,bad.double())


def test_cross_first_order_uses_UPDATED_query_and_history_mask_only():
    block=module.CrossFirstFutureBlock(16,4).eval(); q=torch.randn(2,6,16); h=torch.randn(2,4,16)
    valid=torch.tensor([[0,1,1,1],[1,0,1,1]],dtype=torch.bool)
    seen=[]
    hooks=[block.cross_attn.register_forward_pre_hook(lambda m,a,k:seen.append(('cross',a[0].clone(),k)),with_kwargs=True),
           block.self_attn.register_forward_pre_hook(lambda m,a,k:seen.append(('self',a[0].clone(),k)),with_kwargs=True)]
    with torch.no_grad(): actual=block(q,h,valid)
    for hook in hooks: hook.remove()
    with torch.no_grad():
        z=block.cross_ctx_norm(h)
        q1=q+block.cross_attn(block.cross_q_norm(q),z,z,key_padding_mask=~valid,need_weights=False)[0]
        z=block.self_norm(q1); q2=q1+block.self_attn(z,z,z,need_weights=False)[0]
        expected=q2+block.ffn(block.ffn_norm(q2))
    assert [x[0] for x in seen]==['cross','self']
    torch.testing.assert_close(seen[1][1],block.self_norm(q1),rtol=0,atol=0)
    assert 'key_padding_mask' not in seen[1][2] and torch.equal(seen[0][2]['key_padding_mask'],~valid)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    poisoned=h.clone(); poisoned[~valid]=1000*torch.randn_like(poisoned[~valid])
    with torch.no_grad(): torch.testing.assert_close(block(q,poisoned,valid),actual,rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_history_only_memory_no_Q_content_or_weight_horizon_motion_and_gradients(device,monkeypatch):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    from real_motion import shared_source_track
    from tools.shared_history_experiments.full_gt.engine import semantic_loss
    from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance
    body,runner,row=fixture(device)
    try:
        monkeypatch.setattr(shared_source_track,'register_history_shape',lambda *a,**k:pytest.fail('ICP called'))
        # Nonzero KTA exercises the INPUT weight gradient (the original fixture can be stationary).
        record=dict(row[0],kta_displacement_xy_m=row[0]['kta_displacement_xy_m']+.2,
            anchors_xy_t0_m=row[0]['anchors_xy_t0_m']+.2)
        row=record,*row[1:]
        # Open the existing zero-initialized readout so the occupancy objective reaches motion immediately.
        with torch.no_grad(): body.columns.readout[-1].weight.normal_(std=.1)
        # Identity audit ONLY: eliminate existing CUDA scatter reduction drift, retain original precision.
        # Context exits BEFORE normal-mode grid_sample backward; production is never changed.
        with deterministic_entrance():
            fw=adapter.forward_window(body,runner.provider,row,fresh=True)
            assert fw['logits'].shape[0]==fw['memory'].shape[0]==6
            out=dict(fw['output']); out.pop('future_transport_queries')
            args=(body.columns,fw['lifted'],out,row[3],row[1],fw['targets'],fw['yaws'],fw['poses'],runner.provider.pcfg.grid)
            memory=geometry.aligned_memory(*args)
            torch.testing.assert_close(memory,fw['memory'],rtol=0,atol=0)
            out['future_transport_queries']=object()  # Not even a valid tensor; MUST NEVER be consumed.
            torch.testing.assert_close(geometry.aligned_memory(*args),memory,rtol=0,atol=0)
            targets=fw['targets'].detach().clone(); targets[:,5,0]+=.35
            moved=geometry.aligned_memory(*args[:5],targets,*args[6:])
            torch.testing.assert_close(moved[:5],memory[:5],rtol=0,atol=0)
            assert not torch.equal(moved[5],memory[5])
        losses=semantic_loss(fw['logits'],row[1]['future_gt_occ'])[0]
        losses.mean().backward()
        for p in (body.transport.residual_head.weight,body.transport.yaw_head.weight,body.transport.history_yaw.weight,
                  body.transport.ego_future_proj[-1].weight,body.transport.encoder_blocks[0].scene.offsets.weight):
            # Future motion heads are initially zero: encoder geometry gradient still comes via lift/history yaw.
            assert p.grad is not None and torch.isfinite(p.grad).all()
        assert body.transport.residual_head.weight.grad.abs().sum()>0
        assert body.transport.history_yaw.weight.grad.abs().sum()>0
        # Open motion heads, then verify Q/ego receive occupancy gradients WITHOUT a Q-to-memory path.
        body.zero_grad(set_to_none=True)
        with torch.no_grad(): body.transport.residual_head.weight.normal_(std=.02); body.transport.yaw_head.weight.normal_(std=.005)
        second=adapter.forward_window(body,runner.provider,row,fresh=True)
        semantic_loss(second['logits'],row[1]['future_gt_occ'])[0].mean().backward()
        for p in (body.transport.ego_future_proj[-1].weight,body.transport.kta_future_proj[0].weight,
                  body.transport.decoder[0].cross_attn.in_proj_weight,body.transport.encoder_blocks[0].scene.offsets.weight):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    finally: runner.close()


def test_single_pose_entrance_ego_dependency_no_double_motion_and_no_future_label_inputs(monkeypatch):
    body,runner,row=fixture()
    try:
        poses=torch.as_tensor(row[1]['future_poses']).clone(); poses[:,0,3]+=.2
        original_bg=adapter.backgrounds; seen=[]
        def bg(raw,state,p,provider,track):
            seen.append(p.clone()); return original_bg(raw,state,p,provider,track)
        monkeypatch.setattr(adapter,'backgrounds',bg)
        poisoned=row[0],dict(row[1],future_gt_occ=object(),future_poses=object()),row[2],row[3]
        with torch.no_grad():
            a=adapter.forward_window(body,runner.provider,poisoned,poses=poses) # explicit poses force fresh backgrounds
            b=adapter.forward_window(body,runner.provider,row,poses=row[1]['future_poses'],fresh=True)
        torch.testing.assert_close(seen[0],poses,rtol=0,atol=0)
        torch.testing.assert_close(a['poses'],poses,rtol=0,atol=0)
        torch.testing.assert_close(a['ego_features'],module.relative_ego_features(torch.as_tensor(row[1]['history_poses'][-1]),poses))
        # With zero residual heads, changing ego affects projection, NOT world-space source targets.
        torch.testing.assert_close(a['targets'],b['targets'],rtol=0,atol=0)
        torch.testing.assert_close(a['output']['future_transport_queries'],b['output']['future_transport_queries'],rtol=0,atol=0)
        with torch.no_grad(): body.transport.ego_future_proj[-1].weight.normal_(std=.1)
        with torch.no_grad():
            a=adapter.forward_window(body,runner.provider,row,poses=poses)
            b=adapter.forward_window(body,runner.provider,row,poses=row[1]['future_poses'])
        assert not torch.equal(a['output']['future_transport_queries'],b['output']['future_transport_queries'])
        for k in ('history_source_context','history_yaw_to_t0_rad'):
            torch.testing.assert_close(a['output'][k],b['output'][k],rtol=0,atol=0)
        assert not any(p.grad is not None for p in body.parameters())
    finally: runner.close()


def test_empty_source_output_and_zero_branch_gradient_opening():
    body,runner,row=fixture()
    try:
        reads={k:torch.as_tensor(v) for k,v in row[3][0]['shared_reads'].items()}
        for k in ('reference_valid','reference_grid','reference_xy_t0'): reads[k]=reads[k][:0]
        out=body.transport(torch.zeros(4,16,4,4),reads,torch.arange(4),None,torch.empty(0,6,2),torch.zeros(6,4))
        assert out['future_transport_queries'].shape==(0,6,body.transport.config.d_model)
        ego=torch.randn(6,4,requires_grad=True); kta=torch.randn(1,6,2)
        optimizer=torch.optim.SGD(body.transport.ego_future_proj.parameters(),lr=.1)
        body.transport.initial_queries(kta,ego,torch.float32).square().sum().backward()
        assert body.transport.ego_future_proj[-1].weight.grad.abs().sum()>0
        assert (ego.grad==0).all()
        optimizer.step(); optimizer.zero_grad(); ego.grad=None
        body.transport.initial_queries(kta,ego,torch.float32).square().sum().backward()
        assert ego.grad.abs().sum()>0
    finally: runner.close()


def test_real_evaluation_integer_restore_and_future_outputs_before_labels(tmp_path,monkeypatch):
    body,runner,row=fixture()
    try:
        for token,sem in zip(row[0]['history_tokens'],row[1]['history_occ']):
            p=tmp_path/row[0]['scene_name']/str(token);p.mkdir(parents=True);np.savez(p/'labels.npz',semantics=sem)
        calls=[]; actual_forward=evaluation.forward_window
        def forward(*a,**k): calls.append('forecast');return actual_forward(*a,**k)
        def moving(*a,**k): assert calls==['forecast','forecast'];return []
        monkeypatch.setattr(evaluation,'forward_window',forward)
        monkeypatch.setattr(evaluation.gt,'gt_moving_support_sequence',moving)
        monkeypatch.setattr(evaluation.gt,'moving_support_masks',lambda _,shape:np.ones((6,*shape),bool))
        monkeypatch.setattr(evaluation,'box_motion_rows',lambda *a:[])
        states=[]
        first=evaluation.evaluate(body,runner.provider,SimpleNamespace(nusc=None),[[row]],stc_root=tmp_path,save=lambda v:states.append(copy.deepcopy(v)))
        second=evaluation.evaluate(body,runner.provider,SimpleNamespace(nusc=None),[[row]],stc_root=tmp_path,saved=states[-1])
        assert calls==['forecast','forecast']
        for route,m in first['variants'].items():
            for k in ('mIoU','IoU','MovingMicro'): assert m[k]==second['variants'][route][k]
        assert len(first['variants'])==4 and first['windows']==1
        with pytest.raises(RuntimeError): evaluation.restore(dict(states[-1],protocol='old_kta_query'))
    finally:runner.close()


def test_reference_rejects_full_deep_or_incomplete_and_no_automatic_selection(tmp_path):
    from tools.source_scene_integrated.run import PROTOCOL
    body,runner,_=fixture()
    try:
        control=module.fresh_control(body.transport.config,**KW)
        keys=[['DEV',str(i)] for i in range(64)]
        value=dict(status='complete',updates=1031,contract=dict(protocol=PROTOCOL,epochs=1,
            model=control.configs(),registration='none',source_policy='shared_pinned',known='TRAIN_OCC_STC_dense_valid',
            total_updates=1031,eval_keys=keys),evaluations={'1':dict(windows=64,keys=keys)})
        folder=tmp_path/'scene_integrated_1ep_one';folder.mkdir()
        (folder/'results.json').write_text(json.dumps(value))
        assert report.select_reference(tmp_path)==folder
        second=tmp_path/'scene_integrated_1ep_two';second.mkdir()
        (second/'results.json').write_text(json.dumps(value))
        with pytest.raises(RuntimeError,match='candidates'): report.select_reference(tmp_path)
        bad=copy.deepcopy(value);bad['contract']['model']['motion']['decoder_blocks']=4
        (second/'results.json').write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):report.reference(second)
        bad=copy.deepcopy(value);bad['contract']['epochs']=2
        (second/'results.json').write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):report.reference(second)
        c=dict(world_size=2);saved=dict(protocol=run.PROTOCOL,contract=c,epoch=0,batch=1,updates=1,rank_rng=[{},{}])
        assert run.validate_resume(saved,c,[[1,2]])==(0,1,1)
        with pytest.raises(RuntimeError):run.validate_resume(dict(saved,protocol=PROTOCOL),c,[[1,2]])
        with pytest.raises(RuntimeError):run.validate_resume(dict(saved,updates=2),c,[[1,2]])
    finally:runner.close()


def test_actual_cuda_training_checkpoint_resume_frozen_control_and_fingerprint(tmp_path,monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA')
    from tools.source_scene_integrated.run import PROTOCOL as CONTROL_PROTOCOL
    body,runner,row=fixture()
    try:
        paths={k:tmp_path/k for k in ('config','train_cache','dev_cache','base_checkpoint','train_info','dev_info','eval_reference','population_manifest')}
        for p in paths.values():p.write_text('fixture')
        torch.save(dict(model_config=asdict(body.transport.config)),paths['base_checkpoint'])
        stc=tmp_path/'stc'
        for token,sem in zip(row[0]['history_tokens'],row[1]['history_occ']):
            d=stc/row[0]['scene_name']/str(token);d.mkdir(parents=True);np.savez(d/'labels.npz',semantics=sem)
        inventory=sorted((str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in stc.rglob('labels.npz'))
        eval_rows=[dict(row[0],t0_token='dev'+str(i)) for i in range(64)]
        keys=[list(k) for k in run.base.record_keys(eval_rows)]
        controlbody=module.fresh_control(body.transport.config,**KW)
        control=dict(status='complete',updates=2,contract=dict(protocol=CONTROL_PROTOCOL,epochs=1,
            model=controlbody.configs(),world_size=1,parameters=sum(p.numel() for p in controlbody.parameters()),
            registration='none',source_policy='shared_pinned',known='TRAIN_OCC_STC_dense_valid',
            train_keys=[['TRAIN','one']],eval_keys=keys,total_updates=2,epochs_updates=[2],seed=20261011,
            objective=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.),learning_rates=[.0005,.0003],
            schedule='whole_1_epoch_cosine_floor0.01',precision='original_BF16_geometry_FP64_H6',overrides=[],
            data={k:run.base.sha256(paths[k]) for k in ('config','train_cache','dev_cache','base_checkpoint','train_info','dev_info')},
            STC_inventory=inventory,implementation={}),evaluations={'1':dict(windows=64,keys=keys,variants={
                r:dict(mIoU=10.,IoU=20.,MovingMicro=5.) for r in ('occ_gt_transport','occ_gt_semantic_refine','stc_gt_transport','stc_gt_semantic_refine')})})
        reference=tmp_path/'control';reference.mkdir();(reference/'results.json').write_text(json.dumps(control))
        sha=run.base.sha256(reference/'results.json')
        timestamps={str(t):int(i*500000) for i,t in enumerate(row[0]['history_tokens'])}
        source=SimpleNamespace(nusc=SimpleNamespace(get=lambda _,t:{'timestamp':timestamps[str(t)]}))
        class Provider:
            def __init__(self,*args):
                self.pcfg,self.strong=runner.provider.pcfg,runner.provider.strong
                self.device,self.workers,self.joint=args[3],1,args[5]
            def attach_tracks(self,root,**kw):assert root.parent==tmp_path/'new';return {}
            def close_tracks(self):pass
            def load_raw_columns(self,*a,**k):return row[1]
        def caches(provider,*a):
            provider.ccr_history_cache=SimpleNamespace(close=lambda:None)
            provider.ccr_val_history_cache=SimpleNamespace(close=lambda:None)
            return {}
        monkeypatch.setattr(run,'DirectProvider',Provider)
        monkeypatch.setattr(run,'make_model',lambda cfg:module.make_model(cfg,**KW))
        monkeypatch.setattr(run.base,'attach_caches',caches)
        monkeypatch.setattr(run.base,'load_cache',lambda *a:({},[]))
        monkeypatch.setattr(run.base,'load_runtime_config',lambda *a:{})
        monkeypatch.setattr(run.base,'make_prepare_config',lambda *a:runner.provider.pcfg)
        monkeypatch.setattr(run.base,'NuScenesWindowSource',lambda *a,**k:source)
        monkeypatch.setattr(run.base,'CachedColumnSource',lambda *a,**k:source)
        monkeypatch.setattr(run,'plan',lambda *a:([['TRAIN','one']],eval_rows,[[[row[0]],[row[0]]]]))
        calls=[]
        def evaluate(*a,**kw):calls.append(1);return copy.deepcopy(control['evaluations']['1'])
        monkeypatch.setattr(run,'evaluate',evaluate)
        out=tmp_path/'new'
        argv=['--out-dir',str(out),'--reference-dir',str(reference),'--stc-root',str(stc),'--dataroot',str(tmp_path),
            '--ccr-history-cache',str(tmp_path),'--ccr-val-history-cache',str(tmp_path),'--device','cuda',
            '--cpu-workers','1','--prefetch-workers','1']
        for k,p in paths.items():argv+=['--'+k.replace('_','-'),str(p)]
        run.main(argv+['--max-updates','1'])
        partial=torch.load(out/'last.pt',weights_only=False)
        assert partial['updates']==1 and partial['epoch']==0
        run.main(argv+['--resume',str(out/'last.pt')])
        final=torch.load(out/'last.pt',weights_only=False)
        assert final['updates']==2 and final['epoch']==1 and len(calls)==1
        assert 'tools/source_scene_ego_conditioned/adapter.py' in final['contract']['implementation']
        assert final['contract']['model']['memory']['future_query_content'] is False
        assert not any('source_query' in k for k in final['models']['gt'])
        module.checkpoint_model(final)
        assert (out/'comparison.txt').is_file() and run.base.sha256(reference/'results.json')==sha
        original={k:v.clone() for k,v in final['models']['gt'].items()}
        run.main(argv+['--resume',str(out/'last.pt')]);assert len(calls)==1
        again=torch.load(out/'last.pt',weights_only=False)
        for k,v in original.items():torch.testing.assert_close(again['models']['gt'][k],v,rtol=0,atol=0)
        with pytest.raises(RuntimeError,match='NEW output'):run.main(argv)
    finally:runner.close()
