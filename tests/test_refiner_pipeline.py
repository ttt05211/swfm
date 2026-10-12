"""Actual small-grid staged forward/VJP/drift/control; not nuScenes metrics."""
import copy
import json
from pathlib import Path
import threading
import pytest
import torch
import numpy as np
from test_shared_semantic_refiner import fixture
from tools.joint_occ_experiments import run as legacy
from tools.joint_occ_experiments.fast_autograd import compose_sparse
from tools.shared_history_experiments.refine import engine
from tools.shared_history_experiments.refiner_pipeline.execution import BACKENDS,execution,OrderedOverlay,overlay_values
from tools.shared_history_experiments.refiner_pipeline.staging import PackedInputs,Inverses
from tools.shared_history_experiments.refiner_pipeline.drift import audit,compare,STAGES
from tools.shared_history_experiments.refiner_pipeline import probe
from tools.shared_history_experiments.loss_minimal import run as minimal


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('backend',BACKENDS[1:])
def test_full6_staged_forward_gradients_and_original_hooks(device,backend):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    if device=='cuda' and backend=='staged_compose':pytest.skip('Linux/Triton compiled backend unavailable locally')
    torch.set_num_threads(1);torch.manual_seed(21)
    model,_,p,row=fixture(device)
    with torch.no_grad():model.columns.readout[-1].weight.normal_(std=.02)
    baseline=engine.forward_window(model,p,row)
    loss,_,_=engine.semantic_loss(baseline['logits'],row[1]['future_gt_occ'])
    expected=torch.autograd.grad(loss.mean(),[baseline['output']['residual_xy_m'],baseline['memory'],baseline['field'],baseline['output']['yaw_delta_rad']])
    original=engine.forward_window
    with execution(backend):
        actual=engine.forward_window(model,p,row)
        loss,_,_=engine.semantic_loss(actual['logits'],row[1]['future_gt_occ'])
        gradients=torch.autograd.grad(loss.mean(),[actual['output']['residual_xy_m'],actual['memory'],actual['field'],actual['output']['yaw_delta_rad']])
    assert engine.forward_window is original
    torch.testing.assert_close(actual['logits'],baseline['logits'],rtol=2e-4,atol=2e-5)
    for a,b in zip(gradients,expected):torch.testing.assert_close(a,b,rtol=3e-5,atol=2e-7)
    assert all(g.norm()>0 for g in gradients[:3])


def test_pack_no_precision_change_owns_host_storage():
    arrays={'a':np.arange(12,dtype=np.float64).reshape(4,3),'b':np.arange(3,dtype=np.float64),
            'c':np.array([True,False]),'d':np.arange(4,dtype=np.uint8)}
    p=PackedInputs(arrays,torch.device('cpu'))
    assert len(p.owners)==3
    for k,a in arrays.items():np.testing.assert_array_equal(p[k].numpy(),a)
    p['a'][0,0]=100
    assert arrays['a'][0,0]==0  # caller numpy is unchanged


def test_deferred_inverse_safety_and_pose_gradient():
    matrix=torch.eye(4,dtype=torch.float64).requires_grad_()
    deferred=Inverses(True);value=deferred(matrix)
    expected=torch.linalg.inv(matrix)
    torch.testing.assert_close(value,expected,rtol=0,atol=0)
    value.sum().backward();assert matrix.grad is not None
    deferred.validate()
    singular=Inverses(True);singular(torch.zeros(4,4))
    with pytest.raises(RuntimeError,match='singular'):singular.validate()


def test_ordered_source_overlay_forward_and_alpha_background_gradients():
    torch.manual_seed(21)
    background=torch.rand(18,12,requires_grad=True)
    alpha=[torch.rand(4,requires_grad=True),torch.rand(4,requires_grad=True)]
    ops=[('overlay',torch.tensor([0,1,2,3]),4,None),('overlay',torch.tensor([2,3,4,5]),6,None)]
    masks={c:torch.arange(18)==c for _,_,c,_ in ops}
    reference=compose_sparse(background,ops,alpha)
    candidate=OrderedOverlay.apply(background,tuple(ops),overlay_values,masks,*alpha)
    torch.testing.assert_close(reference,candidate,rtol=0,atol=0)
    weight=torch.rand_like(reference)
    expected=torch.autograd.grad((reference*weight).sum(),[background,*alpha])
    actual=torch.autograd.grad((candidate*weight).sum(),[background,*alpha])
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_frozen_stage_replay_records_first_failure_and_restores():
    model,_,p,row=fixture();model.eval().requires_grad_(False)
    before=copy.deepcopy(model.state_dict());rng=legacy.rng_state()
    saved=dict(models={'gt':before},rng=rng)
    report=audit(model,p,saved,row)
    assert report['scientific_updates']==0 and report['stage_order']==list(STAGES)
    assert all(v['first_numerical_failure'] is None for v in report['reference_replays'])
    assert all(v['first_numerical_failure'] is None for v in report['candidates'].values())
    assert not model.training and not next(model.parameters()).requires_grad
    for name,value in before.items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
    dummy={k:torch.ones(2) for k in STAGES};dummy['logits']=torch.ones(1,2,2)
    changed=copy.deepcopy(dummy);changed['memory'][0]=10
    result=compare(dummy,changed)
    assert result['first_exact_difference']=='memory' and result['first_numerical_failure']=='memory'


def test_no_aux_labels_read_and_existence_head_unchanged():
    model,adam,p,row=fixture();record=copy.deepcopy(row[0])
    before=copy.deepcopy(model.state_dict());aux=engine.source_auxiliary
    for key in ('target_source_displacement_xy_m','se2_target_valid','yaw_label_valid',
                'target_yaw_rad','existence','supervised_source'):record[key]='POISON'
    first=minimal.train(engine.train_arm,model,adam,p,[(record,*row[1:])],'gt',update=1,total_updates=500)
    second=minimal.train(engine.train_arm,model,adam,p,[(record,*row[1:])],'gt',update=2,total_updates=500,probe=True)
    assert engine.source_auxiliary is aux and not first['source_auxiliary_computed']
    assert second['source_geometry']==second['existence']==0
    assert second['gradient_links']['occupancy_to_xy']>0
    assert second['gradient_links']['occupancy_to_shared_history']>0
    for k,v in before.items():
        if 'existence_head' in k:torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


def test_staged_ignores_future_labels_and_masks():
    model,_,p,row=fixture();raw=copy.deepcopy(row[1])
    raw['future_gt_occ']='POISON';raw['future_masks']='POISON';raw['future_annotations']='POISON'
    with execution('staged_sync'):actual=engine.forward_window(model,p,(row[0],raw,row[2],row[3]))
    assert torch.isfinite(actual['logits']).all()


def test_staged_empty_current_sources():
    from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
    model,_,p,row=fixture();record=copy.deepcopy(row[0]);raw=copy.deepcopy(row[1])
    for k in ('features','source_class_id','source_centroid_xy_t0_m','anchors_xy_t0_m','kta_displacement_xy_m'):
        record[k]=record[k][:0]
    state=raw['_column_causal_preparation']['prepared_state']
    for k in ('current','source_world_points','source_rel_xy'):state[k]=[]
    state['source_z_t0']=np.empty(0);raw['_column_causal_preparation']['registrations']=[]
    history=ResidualHistoryInputs(p,ram_mib=0).build(raw);empty=(record,raw,row[2],history)
    expected=engine.forward_window(model,p,empty)
    with execution('staged_sync'):actual=engine.forward_window(model,p,empty)
    torch.testing.assert_close(actual['logits'],expected['logits'],rtol=2e-4,atol=2e-5)


def test_real_pipeline_cli_and_fixed_train_population_readonly(tmp_path,monkeypatch):
    from test_shared_semantic_refiner import test_real_cli_new_protocol_epoch_snapshots_resume_and_eval_readonly
    test_real_cli_new_protocol_epoch_snapshots_resume_and_eval_readonly(tmp_path,monkeypatch)
    source=tmp_path/'pilot/epoch_0002.pt'
    original_bytes={p:p.read_bytes() for p in (source,tmp_path/'pilot/last.pt',tmp_path/'pilot/summary.txt')}
    out=tmp_path/'pipeline'
    args=[value for key,path in (('config',tmp_path/'config.yaml'),('base-checkpoint',tmp_path/'base.pt'),
        ('train-cache',tmp_path/'train-cache'),('dev-cache',tmp_path/'dev-cache'),('population-manifest',tmp_path/'population-manifest'),
        ('dataroot',tmp_path),('train-info',tmp_path/'train-info'),('dev-info',tmp_path/'dev-info'),
        ('ccr-history-cache',tmp_path),('ccr-val-history-cache',tmp_path),('out-dir',out),('resume',source))
        for value in ('--'+key,str(path))]
    assert probe.main(argv=args+['--probe-steps','2','--profile'])==0
    result=json.loads((out/'speed.json').read_text())
    assert set(result['seconds_per_window'])==set(BACKENDS)
    assert result['scientific_updates_published']==0 and not result['fp64_arithmetic_fused']
    assert (out/'stage_replay.json').is_file()
    assert all(k[0]!='DEV' for k in json.loads((out/'population.json').read_text()))
    assert not (out/'last.pt').exists()
    for p,value in original_bytes.items():assert p.read_bytes()==value


def test_minimal_runner_fixed_500_protocol_and_restores(tmp_path,monkeypatch):
    args=[value for key in ('config','base-checkpoint','train-cache','dev-cache','population-manifest',
        'dataroot','train-info','dev-info','ccr-history-cache','ccr-val-history-cache') for value in ('--'+key,str(tmp_path/key))]
    args+=['--out-dir',str(tmp_path/'minimal'),'--epochs','1','--mode','gt']
    original=(minimal.refiner.PROTOCOL,minimal.refiner.ResidualJoint,minimal.refiner.engine.train_arm)
    monkeypatch.setattr(legacy,'epoch_groups',lambda *a:[[i] for i in range(1031)])
    def child(event,argv):
        assert minimal.refiner.PROTOCOL==minimal.PROTOCOL
        assert len(legacy.epoch_groups([],21,0,4,128))==500
        return 0
    monkeypatch.setattr(minimal.refiner,'main',child)
    assert minimal.main(argv=args)==0
    assert (minimal.refiner.PROTOCOL,minimal.refiner.ResidualJoint,minimal.refiner.engine.train_arm)==original
    with pytest.raises(ValueError):minimal.main(argv=args+['--epochs','2'])
