"""Full plan, legacy metadata/raw evaluator, fresh weights and actual CUDA resume."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_full import run,model,evaluation
from tools.causal_forward_train import model as pilot_model
from tools.causal_forward_recovery.adapter import RecoveryPredictor


def test_full_plan_from_selected_C_contract_and_strict_mismatches(tmp_path,monkeypatch):
    from test_causal_forward_train import fake_control
    train=[dict(scene_name='TRAIN',t0_token=str(i),features=[None]) for i in range(20430)]
    dev=[dict(scene_name='VAL',t0_token=str(i),history_tokens=['old1','old2','a','b','c',str(i)]) for i in range(4369)]
    keys=[['VAL',str(i)] for i in range(64)]
    ref=tmp_path/'reference.json';ref.write_text(json.dumps(dict(contract=dict(eval_keys=keys))))
    args=train,dev,ref,20261011;expected=run._plan(*args)
    c=fake_control()['contract'];c.update(protocol=run.PILOT_PROTOCOL,world_size=1,
        source_policy=model.SOURCE_POLICY,parameters=2816122,eval_keys=keys,
        train_keys=[list(k) for k in expected[0][:4086]],
        model=dict(source_policy=model.SOURCE_POLICY,experiment=dict(candidate_implementation={})))
    inventory=set()
    for record in expected[1]:
        for token in record['history_tokens'][-4:]:
            p=tmp_path/'VAL'/token/'labels.npz';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    for p in (tmp_path/'VAL').rglob('labels.npz'):
        s=p.stat();inventory.add((str(p.resolve()),s.st_size,s.st_mtime_ns))
    c['STC_inventory']=[list(v) for v in sorted(inventory)]
    contract=tmp_path/'pilot.json';contract.write_text(json.dumps(c))
    a=SimpleNamespace(pilot_contract=contract,stc_root=tmp_path,seed=20261011,motion_lr=5e-4,repair_lr=3e-4,
        override=[],cpu_workers=4,prefetch_workers=2,epochs=2,train_fraction=1.)
    monkeypatch.setattr(run,'_launch',lambda *a:run.runner.plan(*args))
    factory,plan=run.runner.make_model,run.runner.plan
    actual=run.launch(a,None,SimpleNamespace(world_size=1))
    assert actual==expected and len(actual[0])==20430
    for groups in actual[2]:
        seen=[(r['scene_name'],r['t0_token']) for group in groups for r in group]
        assert len(seen)==len(set(seen))==20430 and set(seen)==set(actual[0])
    for field in ('eval_keys','train_keys','STC_inventory'):
        bad=copy.deepcopy(c)
        if field=='eval_keys':bad[field][:2]=bad[field][1::-1]
        elif field=='train_keys':bad[field][0]=['FOREIGN','missing']
        else:bad[field][0][1]+=1
        contract.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):run.launch(a,None,SimpleNamespace(world_size=1))
        assert run.runner.make_model is factory and run.runner.plan is plan
    contract.write_text(json.dumps(c))
    assert run.launch(a,None,SimpleNamespace(world_size=2))==expected
    monkeypatch.setenv('WORLD_SIZE','3')
    with pytest.raises(RuntimeError,match='ONE or TWO'):run.main([])


def test_same_fresh_C_state_RNG_parameter_count_and_isolated_full_loader():
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    cfg=LocalSTWMV17Config(history_frames=4,decoder_blocks=2)
    torch.manual_seed(21);pilot=pilot_model.make_model(cfg);rng=torch.get_rng_state().clone()
    torch.manual_seed(21);body=model.make_model(cfg,expected_pilot_spec=json.loads(json.dumps(pilot.configs())))
    assert torch.equal(torch.get_rng_state(),rng)
    assert sum(p.numel() for p in body.parameters())==2816122
    for k,v in pilot.state_dict().items():torch.testing.assert_close(body.state_dict()[k],v,rtol=0,atol=0)
    body.experiment=dict(FREE_bias=0.,history_age=False,candidate_implementation={
        'tools/causal_forward_full/model.py':run.sha('tools/causal_forward_full/model.py')})
    saved=dict(protocol=run.PROTOCOL,contract=dict(protocol=run.PROTOCOL,source_policy=model.SOURCE_POLICY,model=body.configs()),
        models={'gt':body.state_dict()})
    restored=model.checkpoint_model(saved)
    assert restored.configs()==body.configs() and not any(p.requires_grad for p in restored.parameters())
    for k,v in body.state_dict().items():torch.testing.assert_close(restored.state_dict()[k],v,rtol=0,atol=0)
    with pytest.raises(RuntimeError):pilot_model.checkpoint_model(saved)
    with pytest.raises(RuntimeError):model.checkpoint_model(dict(saved,protocol=run.PILOT_PROTOCOL))
    bad=copy.deepcopy(saved);bad['contract']['model']['experiment']['candidate_implementation']['tools/causal_forward_full/model.py']='wrong'
    with pytest.raises(RuntimeError,match='implementation'):model.checkpoint_model(bad)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_raw_evaluator_six_token_metadata_forward_C_and_labels_after_H6(tmp_path,monkeypatch,device):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):pytest.skip('BF16 CUDA')
    from test_fpn_adaln_zero import fixture
    old,base,row=fixture(device)
    body=model.make_model(old.transport.config,feature_width=16,z_bins=4,memory_width=4).to(device).eval().requires_grad_(False)
    predictor=RecoveryPredictor(body,base.provider.pcfg,device,workers=1)
    try:
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        record=dict(row[0],history_tokens=('unused_old1','unused_old2')+tuple(row[0]['history_tokens']))
        prepared=predictor.prepare(record,raw,times=np.arange(4)*.5)
        adapter=SimpleNamespace(provider=predictor.provider,history=predictor.history)
        checked=evaluation.NormalizedForwardPrepare.prepare(adapter,record,raw,times=np.arange(4)*.5)
        for key in ('kta_displacement_xy_m','anchors_xy_t0_m','source_class_id'):
            np.testing.assert_array_equal(prepared[0][key],checked[0][key])
        for token,sem in zip(record['history_tokens'][-4:],raw['history_occ']):
            p=tmp_path/record['scene_name']/str(token);p.mkdir(parents=True);np.savez(p/'labels.npz',semantics=sem)
        calls=dict(forward=0,labels=0)
        class GuardedRaw(dict):
            def __getitem__(self,key):
                if key=='future_gt_occ':assert calls['forward']==2;calls['labels']+=1
                return super().__getitem__(key)
        item=(dict(prepared[0],history_tokens=record['history_tokens']),GuardedRaw(prepared[1],
            future_poses=row[1]['future_poses'],future_gt_occ=row[1]['future_gt_occ']),prepared[2],prepared[3])
        original=evaluation.runner.forward_window
        def forward(*a,**kw):
            calls['forward']+=1;value=original(*a,**kw);assert value['logits'].shape[:2]==(6,18);return value
        monkeypatch.setattr(evaluation.runner,'forward_window',forward)
        monkeypatch.setattr(evaluation.runner.gt,'gt_moving_support_sequence',lambda *a,**kw:[
            (np.ones(raw['history_occ'].shape[1:],bool),[],[]) for _ in range(6)])
        report=evaluation.evaluate(body,predictor.provider,SimpleNamespace(nusc=None),[[item]],stc_root=tmp_path)
        assert calls==dict(forward=2,labels=1) and report['windows']==1 and set(report['variants'])==set(evaluation.NAMES)
        assert len(report['variants'])==4 and all('free' not in n for n in report['variants'])
        assert evaluation.runner is not evaluation.original and evaluation.original.PROTOCOL!=evaluation.runner.PROTOCOL
    finally:predictor.close();base.close()


def test_cadence_resume_protocol_and_no_calibration_routes():
    assert not run.evaluation_due(0,0) and not run.evaluation_due(1999,1999)
    assert all(run.evaluation_due(u,1) for u in (2000,4000,6000,8000,10000))
    assert all(run.evaluation_due(u,0) for u in (5145,10292))
    assert run.runner.checkpoint_names(2000,0,2000)==['step_0002000.pt']
    assert run.runner.checkpoint_names(5145,1,0)==['epoch_0001.pt']
    c=dict(world_size=1);saved=dict(protocol=run.PROTOCOL,contract=c,epoch=1,batch=0,updates=2,rank_rng=[{}])
    assert run.runner.validate_resume(saved,c,[[1,2],[3]])==(1,0,2)
    with pytest.raises(RuntimeError):run.runner.validate_resume(dict(saved,protocol=run.PILOT_PROTOCOL),c,[[1,2],[3]])
    assert run.runner.DirectProvider is run.ForwardProvider and run.runner.forward_window is run.forward_window
    assert run.runner.evaluate is evaluation.evaluate and len(run.NAMES)==4


def test_actual_CUDA_full_loop_Adam_RNG_resume_with_forward_model(tmp_path,monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():pytest.skip('BF16 CUDA')
    import test_fpn_unet_full as original_test
    monkeypatch.setattr(original_test,'run',run.runner);monkeypatch.setattr(original_test,'model',model)
    monkeypatch.setattr(run.runner,'parser',run._parser);monkeypatch.setattr(run.runner,'launch',run._launch)
    original_test.test_actual_cuda_two_epoch_pause_resume_Adam_RNG_and_only_two_evals(tmp_path,monkeypatch)
    saved=torch.load(tmp_path/'new/last.pt',weights_only=False)
    assert saved['contract']['source_policy']==model.SOURCE_POLICY
    assert saved['contract']['evaluation']=='every2000_and_epoch_end_same64'
    assert saved['contract']['world_size']==1 and saved['protocol']==run.PROTOCOL
