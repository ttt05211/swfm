"""Frozen embedding-OUTPUT intervention, pose invariance, labels ordering and recovery."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_scene_ego_check import engine,run
from tools.source_scene_stage2.adapter import DirectPredictor
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_frozen_output_zero_same_poses_readonly_weights_and_exception_cleanup(device,monkeypatch):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):pytest.skip('BF16 CUDA')
    from test_source_scene_ego_conditioned import fixture
    from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance
    body,old,row=fixture(device);predictor=DirectPredictor(body,old.provider.pcfg,device,1)
    try:
        with torch.no_grad():
            body.transport.ego_future_proj[-1].weight.normal_(std=.1)
            body.transport.ego_future_proj[-1].bias.fill_(.2)
            body.transport.residual_head.weight.normal_(std=.02)
            body.transport.yaw_head.weight.normal_(std=.01)
            body.columns.readout[-1].weight.normal_(std=.1)
        body.eval().requires_grad_(False);before=tensor_state_fingerprint(body.state_dict())
        seen=[];actual=engine.render_future
        def render(*a,**k):
            prepared=a[3];seen.append((prepared['poses'].clone(),id(prepared['state']['column_backgrounds'])))
            return actual(*a,**k)
        monkeypatch.setattr(engine,'render_future',render)
        # Exact structural identity check only; no deployment precision/settings change.
        with deterministic_entrance():predictions,d=engine.paired_forecasts(predictor,row,row[1]['future_poses'],repeat=True)
        assert len(seen)==3 and len({x[1] for x in seen})==1
        for p,_ in seen:torch.testing.assert_close(p,seen[0][0],rtol=0,atol=0)
        assert d['history_byte_equal'] and d['same_future_poses'] and d['shared_fresh_background']
        assert max(d['query_delta_norm'])>0 and max(d['source_XY_delta_m'])>0
        assert all(v.shape[0]==6 for r in predictions.values() for v in r.values())
        assert not body.transport.ego_future_proj._forward_hooks
        value=torch.randn(6,4,device=device)
        with engine.zero_ego_projection(body) as called:
            torch.testing.assert_close(body.transport.ego_future_proj(value),torch.zeros(6,body.transport.config.d_model,device=device),rtol=0,atol=0)
            assert len(called)==1
        with pytest.raises(RuntimeError):
            with engine.zero_ego_projection(body):raise RuntimeError('injected failure')
        assert not body.transport.ego_future_proj._forward_hooks
        assert before==tensor_state_fingerprint(body.state_dict()) and all(p.grad is None for p in body.parameters())
    finally:predictor.close();old.close()


def test_fresh_zero_initialized_branch_is_identical_under_deterministic_audit():
    from test_source_scene_ego_conditioned import fixture
    from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance
    body,old,row=fixture();p=DirectPredictor(body.eval().requires_grad_(False),old.provider.pcfg,'cpu',1)
    try:
        with deterministic_entrance():pred,d=engine.paired_forecasts(p,row,row[1]['future_poses'],repeat=True)
        assert max(d['embedding_norm'])==max(d['query_delta_norm'])==max(d['source_XY_delta_m'])==0
        for r in engine.READOUTS:np.testing.assert_array_equal(pred['ego_on'][r],pred['ego_off'][r])
    finally:p.close();old.close()


def test_all_predictions_before_future_labels_integer_resume_and_reject_bad_state(monkeypatch):
    shape=(2,2,2);p=SimpleNamespace(provider=SimpleNamespace(pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=shape)),workers=1),prepare=lambda *a,**k:None)
    order=[];event=__import__('threading').Event();keys=[['V','one'],['V','two']]
    diagnostic=dict(embedding_norm=[1.],query_delta_norm=[.1],source_XY_delta_m=[.01],source_yaw_delta_deg=[.1],
        history_context_max_abs=0.,history_yaw_max_abs=0.,changed_voxels=dict(transport=1,semantic_refine=1),
        source_count=1,repeated_on_changed_voxels=dict(transport=0,semantic_refine=0),
        repeated_on_logits_max_abs=0.,same_future_poses=True,shared_fresh_background=True,history_byte_equal=True)
    def paired(*a,**k):
        order.append('predict')
        return {m:{r:np.full((6,*shape),17,np.uint8) for r in engine.READOUTS} for m in engine.MODES},copy.deepcopy(diagnostic)
    def targets(*a):assert order[-2:]==['predict','predict'];order.append('labels');return [np.full(shape,17,np.uint8)]*3
    source=SimpleNamespace(poses=lambda *a:np.repeat(np.eye(4)[None],6,0),
        history=lambda *a:({},dict(history_occ=np.full((4,*shape),17,np.uint8),history_observed=np.ones((4,*shape),bool),history_poses=np.repeat(np.eye(4)[None],4,0)),np.arange(4)),
        targets=targets,moving=lambda *a:[np.zeros(shape,bool)]*3)
    monkeypatch.setattr(engine,'paired_forecasts',paired)
    states=[]
    def save(v):states.append(copy.deepcopy(v))
    # Stop request arriving after the first paired window still saves ALL its routes atomically.
    def moving(*a):event.set();return [np.zeros(shape,bool)]*3
    source.moving=moving
    with pytest.raises(InterruptedError):engine.evaluate(p,source,[0,1],keys,{},save=save,stop_event=event)
    assert states[-1]['keys']==keys[:1]
    event.clear();source.moving=lambda *a:[np.zeros(shape,bool)]*3
    result=engine.evaluate(p,source,[0,1],keys,{},saved=states[-1],save=save,stop_event=event)
    assert result['windows']==2 and len(result['reports'])==8 and order.count('predict')==4
    assert result['diagnostics']['occ']['source_XY_delta_m']['mean']==pytest.approx(.01)
    before=len(order);engine.evaluate(p,source,[0,1],keys,{},saved=states[-1]);assert len(order)==before
    bad=copy.deepcopy(states[-1]);bad['counts'][engine.ROUTES[0]]['confusion'][0][17][17]+=1
    with pytest.raises(RuntimeError,match='totals'):engine.restore(bad,{},keys,shape)
    with pytest.raises(RuntimeError):engine.restore(dict(states[-1],protocol='different'),{},keys,shape)


def source_files(folder,body):
    folder.mkdir();c=dict(protocol=run.TRAIN_PROTOCOL,epochs=1,total_updates=1031,epochs_updates=[1031],
        model=body.configs(),known='TRAIN_OCC_STC_dense_valid',source_policy='shared_pinned',registration='none',
        train_keys=[['T',str(i)] for i in range(4086)],eval_keys=[['V',str(i)] for i in range(64)],
        data={},implementation={},STC_inventory=[],overrides=[])
    variant={k:dict(mIoU=10.,IoU=20.,MovingMicro=5.) for k in ('occ_gt_transport','occ_gt_semantic_refine','stc_gt_transport','stc_gt_semantic_refine')}
    result=dict(status='complete',updates=1031,contract=c,evaluations={'1':dict(windows=64,keys=c['eval_keys'],variants=variant)})
    checkpoint=dict(protocol=run.TRAIN_PROTOCOL,contract=c,epoch=1,batch=0,updates=1031,models={'gt':body.state_dict()},optimizers={'gt':{'DO_NOT_USE':True}},rank_rng=['DO_NOT_USE'])
    return c,result,checkpoint


def test_source_selection_loader_and_actual_runner_readonly_resume(tmp_path,monkeypatch):
    from test_source_scene_ego_conditioned import fixture
    body,old,_=fixture()
    try:
        folder=tmp_path/'scene_ego_query_1ep_one';c,result,checkpoint=source_files(folder,body)
        paths={k:tmp_path/k for k in ('config','dev_cache','dev_info')}
        for p in paths.values():p.write_text('fixture')
        c['data']={k:run.base.sha256(p) for k,p in paths.items()}
        torch.save(checkpoint,folder/'epoch_0001.pt');(folder/'results.json').write_text(json.dumps(result))
        assert run.select_source(tmp_path)==folder
        loaded,header,receipt,_=run.load_source(folder)
        assert 'optimizers' not in header and 'rank_rng' not in header
        source_sha=run.base.sha256(folder/'epoch_0001.pt')
        monkeypatch.setattr(run.base,'load_cache',lambda *a:({},[dict(scene_name='V',t0_token=str(i)) for i in range(64)]))
        monkeypatch.setattr(run,'window',lambda r:r)
        monkeypatch.setattr(run,'NuscInputs',lambda *a,**k:SimpleNamespace(preflight=lambda *a:[]))
        monkeypatch.setattr(run.base,'load_runtime_config',lambda *a:{})
        monkeypatch.setattr(run.base,'make_prepare_config',lambda *a:None)
        monkeypatch.setattr(run,'DirectPredictor',lambda *a:SimpleNamespace(close=lambda:None))
        calls=[]
        def evaluate(*a,**k):
            if k['saved'] is not None:return dict(windows=64,reports={},diagnostics={})
            calls.append(1);k['save'](dict(fixture_complete=True));return dict(windows=64,reports={},diagnostics={})
        monkeypatch.setattr(run,'evaluate',evaluate)
        # This workflow test exercises persistence/provenance, not metric formatting (covered separately).
        monkeypatch.setattr(run,'summary',lambda r:r['status']+'\n')
        argv=['--source-dir',str(folder),'--out-dir',str(tmp_path/'audit'),'--device','cpu',
            '--dataroot',str(tmp_path),'--stc-root',str(tmp_path)]
        for k,p in paths.items():argv+=['--'+k.replace('_','-'),str(p)]
        run.main(argv);run.main(argv+['--resume']);assert len(calls)==1
        assert run.base.sha256(folder/'epoch_0001.pt')==source_sha
        assert json.loads((tmp_path/'audit/results.json').read_text())['status']=='complete'
        with pytest.raises(FileExistsError):run.main(argv)
        another=tmp_path/'scene_ego_query_1ep_two';another.mkdir();(another/'results.json').write_text(json.dumps(result))
        with pytest.raises(RuntimeError,match='unique'):run.select_source(tmp_path)
        checkpoint['batch']=1;torch.save(checkpoint,folder/'epoch_0001.pt')
        with pytest.raises(RuntimeError,match='cursor'):run.load_source(folder)
    finally:old.close()


def test_summary_reports_on_minus_off_and_repeat_FAIL_without_selection():
    reports={r:dict(average=dict(mIoU=12. if '/ego_on/' in r else 10.,IoU=20.,MovingMicro=5.)) for r in engine.ROUTES}
    recorded={k:dict(mIoU=11.,IoU=20.,MovingMicro=5.) for k in ('occ_gt_transport','occ_gt_semantic_refine','stc_gt_transport','stc_gt_semantic_refine')}
    result=dict(status='complete',contract=dict(recorded_variants=recorded),
        evaluation=dict(windows=64,reports=reports,diagnostics={k:dict(repeated_on_byte_fail_windows=1) for k in engine.KINDS}))
    text=run.summary(result)
    assert 'ego_on_minus_off={"mIoU": 2.0' in text
    assert 'fresh_on_minus_recorded={"mIoU": 1.0' in text
    assert 'repeated_on_byte_fail_windows": 1' in text and 'never waived' in text
    assert 'NOT a retrained no-ego ablation' in text
