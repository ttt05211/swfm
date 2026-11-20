"""Matched init/objective, actual backward/resume and independent STC evaluation."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from tools.coverage_prior_only import control,engine,run,evaluate,visuals
from tools.coverage_projection import run as joint_run
from tools.coverage_projection.projection import PROJECTION
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.final_semantic_evaluation import metrics


def test_exact_joint_fresh_factory_parameters_and_RNG():
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    cfg=LocalSTWMV17Config(history_frames=4,decoder_blocks=2)
    torch.manual_seed(20261011);old=joint_run.make_model(cfg);rng=torch.get_rng_state().clone()
    torch.manual_seed(20261011);new=run.make_model(cfg)
    assert torch.equal(rng,torch.get_rng_state()) and new.configs()==old.configs()
    assert sum(p.numel() for p in new.parameters())==2816554
    for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)


def joint_fixture():
    keys=[['train',str(i)] for i in range(4086)];groups=[];at=0
    for size,count in ((4,1000),(3,24),(2,7)):
        for _ in range(count):groups.append(keys[at:at+size]);at+=size
    c=dict(protocol=joint_run.PROTOCOL,total_updates=1031,epochs=1,world_size=2,parameters=2816554,
        seed=20261011,learning_rates=[5e-4,3e-4],schedule='whole_1_epoch_cosine_floor0.01',
        known='TRAIN_OCC_STC_dense_valid',source_policy='causal_forward_CV_history',registration='none',
        objective=dict(CE=1.,Lovasz=1.,projection=PROJECTION),train_keys=keys,
        eval_keys=[['val',str(i)] for i in range(64)],group_keys=groups,
        model=dict(variant='forward_FPN_AdaLN_UNet_coverage_projection_v1',source_policy='causal_forward_CV_history'),
        control_sha256='original_sha')
    return dict(protocol=joint_run.PROTOCOL,updates=1031,contract=c,models={'gt':{'w':torch.ones(1)}})


@pytest.mark.parametrize('field,value',[('world_size',1),('total_updates',10292),('parameters',2816122),('seed',42),('objective',control.OBJECTIVE)])
def test_joint_reference_rejects_wrong_budget_architecture_and_source(field,value):
    value_=joint_fixture();value_['contract'][field]=value
    with pytest.raises(ValueError):control.validate_joint(value_)


def test_joint_exact_groups_and_complete_reused_integer_metrics():
    saved=joint_fixture();c=control.validate_joint(saved)
    groups=[[dict(scene_name=s,t0_token=t) for s,t in g] for g in c['group_keys']]
    run.check_groups(groups,c)
    groups[0]=groups[0][::-1]
    with pytest.raises(ValueError,match='ordered global groups'):run.check_groups(groups,c)
    receipt=dict(candidate='joint_sha',control=c['control_sha256'],keys=c['eval_keys'],known='dense_valid',full_grid=True,
        horizons=[1.,2.,3.],indices=[1,3,5],FREE=False,fixed_age=False,calibration=False)
    counts=metrics.empty(3);counts['confusion'][:,17,17]=64*200*200*16-1;counts['confusion'][:,11,11]=1
    values={k:v.tolist() for k,v in counts.items()};report=metrics.report(counts,(1.,2.,3.),moving=False)
    routes=[f'{m}/{k}' for m in ('control','candidate') for k in ('occ','stc')]
    state=dict(receipt=receipt,progress={r:64 for r in routes},counts={r:{n:values for n in ('transport','semantic_refine')} for r in routes})
    result=dict(status='complete',receipt=receipt,reports={r:{n:report for n in ('transport','semantic_refine')} for r in routes})
    assert control.validate_evaluation(result,state,c,'joint_sha') is result
    bad=copy.deepcopy(state);bad['progress']['candidate/stc']=63
    with pytest.raises(ValueError,match='complete original'):control.validate_evaluation(result,bad,c,'joint_sha')
    bad=copy.deepcopy(result);bad['reports']['control/occ']['transport']['average']['mIoU']=10
    with pytest.raises(ValueError,match='scores differ'):control.validate_evaluation(bad,state,c,'joint_sha')
    bad=copy.deepcopy(state);bad['counts']['control/occ']['transport']['confusion'][0][17][17]-=1
    with pytest.raises(ValueError,match='integer counts'):control.validate_evaluation(result,bad,c,'joint_sha')


def test_actual_full6_step_projection_never_called_and_Adam_RNG_resume(tmp_path,monkeypatch):
    from test_coverage_projection import tiny_scene
    from tools.coverage_projection.projection import VolumeProjection
    body,predictor,row,poses,labels=tiny_scene();initial=copy.deepcopy(body.state_dict())
    metadata=[dict(scene_name='train',t0_token=str(i),features=[None]*len(row[0]['source_class_id'])) for i in range(3)]
    groups=[[r] for r in metadata];c=dict(protocol=control.PROTOCOL,objective=control.OBJECTIVE,world_size=1,total_updates=3)
    ctx=DistributedTraining(False)
    @contextmanager
    def batches(values):yield iter([[(row,poses,labels)] for _ in values])
    def forbidden(*a,**k):raise AssertionError('projection must not execute in this arm')
    monkeypatch.setattr(VolumeProjection,'__init__',forbidden);monkeypatch.setattr(VolumeProjection,'loss',forbidden)
    results=[]
    try:
        for label in ('whole','resume'):
            out=tmp_path/label;out.mkdir();body.load_state_dict(initial);torch.manual_seed(31)
            rng=np.random.default_rng(31)
            opt=torch.optim.AdamW([dict(params=body.transport.parameters(),lr=5e-4,initial_lr=5e-4,weight_decay=1e-4),
                dict(params=body.columns.parameters(),lr=3e-4,initial_lr=3e-4,weight_decay=.01)])
            result=engine.execute(body,opt,predictor,groups,c,ctx,rng,threading.Event(),out,batches,max_updates=1 if label=='resume' else 0)
            if label=='resume':
                assert result['status']=='stopped' and result['updates']==1
                with torch.no_grad():next(body.parameters()).add_(10)
                torch.manual_seed(99);rng=np.random.default_rng(99)
                result=engine.execute(body,opt,predictor,groups,c,ctx,rng,threading.Event(),out,batches,resume=True)
            assert result['status']=='complete'
            rows=[json.loads(v) for v in (out/'progress.jsonl').read_text().splitlines()]
            assert all('projection_semantic' not in r and abs(r['loss']-r['CE']-r['Lovasz'])<1e-6 for r in rows)
            results.append(torch.load(out/'last.pt',weights_only=False))
            digest=joint_run.sha(out/'epoch_0001.pt')
            engine.execute(body,opt,predictor,groups,c,ctx,rng,threading.Event(),out,batches,resume=True)
            assert joint_run.sha(out/'epoch_0001.pt')==digest
        for name,value in results[0]['models']['gt'].items():torch.testing.assert_close(value,results[1]['models']['gt'][name],rtol=0,atol=0)
        assert results[0]['optimizer']['param_groups']==results[1]['optimizer']['param_groups']
        for index,state in results[0]['optimizer']['state'].items():
            for key,value in state.items():torch.testing.assert_close(value,results[1]['optimizer']['state'][index][key],rtol=0,atol=0)
        bad=copy.deepcopy(results[1]);bad['protocol']=joint_run.PROTOCOL
        with pytest.raises(ValueError,match='joint weights'):engine.validate_resume(bad,c)
    finally:predictor.close()


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.transport=torch.nn.Linear(2,18);self.columns=torch.nn.Module()
        self.columns.rare=torch.nn.Linear(2,18);self.columns.unused=torch.nn.Linear(2,1)


def _rank(rank,directory):
    torch.set_num_threads(1);os.environ.update(WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank))
    ctx=DistributedTraining(True,backend='gloo',init_method=Path(directory,'rendezvous').resolve().as_uri())
    try:
        with patch.dict(os.environ,WORLD_SIZE='1'):single=DistributedTraining(False)
        torch.manual_seed(17);initial=Tiny().state_dict()
        records=[dict(scene_name='s',t0_token=str(i),id=i,features=[None]*n) for i,n in enumerate((50,5,20,1))]
        groups=[records,records[:3],records[:1],records]
        c=dict(protocol=control.PROTOCOL,objective=control.OBJECTIVE,world_size=2,total_updates=4)
        predictor=SimpleNamespace(provider=SimpleNamespace(device=torch.device('cpu')))
        def rows(group):
            return [((dict(source_class_id=np.zeros(len(r['features']))),dict(x=torch.tensor([[r['id']*.1,.2],[.1,-.3],[.4,r['id']*.2]]),id=r['id']),None,None),
                None,np.array([[0,4,17]]*6)) for r in group]
        def forward(net,p,row,**kw):
            value=net.transport(row[1]['x'])
            if row[1]['id']%2:value=value+net.columns.rare(row[1]['x'])
            return dict(logits=value.T[None].expand(6,-1,-1))
        @contextmanager
        def batches(values):yield iter([rows(g) for g in values])
        def adam(net):return torch.optim.AdamW([dict(params=net.transport.parameters(),lr=.001,initial_lr=.001),
            dict(params=net.columns.parameters(),lr=.001,initial_lr=.001)])
        final=[]
        with patch.object(engine,'forward_window',forward):
            for label in ('whole','resume'):
                net=Tiny();net.load_state_dict(initial);opt=adam(net)
                torch.manual_seed(21+rank);rng=np.random.default_rng(21+rank);out=Path(directory,label)
                if ctx.primary:out.mkdir()
                ctx.barrier()
                result=engine.execute(net,opt,predictor,groups,c,ctx,rng,threading.Event(),out,batches,max_updates=2 if label=='resume' else 0)
                if label=='resume':
                    assert result['updates']==2
                    with torch.no_grad():next(net.parameters()).add_(10)
                    torch.manual_seed(99);rng=np.random.default_rng(99)
                    result=engine.execute(net,opt,predictor,groups,c,ctx,rng,threading.Event(),out,batches,resume=True)
                assert result['status']=='complete';final.append(copy.deepcopy(net.state_dict()))
            for key,value in final[0].items():torch.testing.assert_close(value,final[1][key],rtol=0,atol=0)
            reference=Tiny();reference.load_state_dict(initial);opt=adam(reference)
            for i,group in enumerate(groups):engine.train_step(reference,opt,predictor,rows(group),update=i+1,total=4,context=single)
            for key,value in final[0].items():torch.testing.assert_close(value,reference.state_dict()[key],rtol=1e-5,atol=2e-7)
        state=torch.load(Path(directory,'resume','last.pt'),weights_only=False)
        assert len(state['rank_rng'])==2 and state['epoch']==1 and state['batch']==0
        if ctx.primary:
            progress=[json.loads(v) for v in Path(directory,'resume','progress.jsonl').read_text().splitlines()]
            assert sorted(progress[2]['rank_windows'])==[0,1]
        Path(directory,f'rank{rank}.json').write_text(json.dumps(dict(ok=True)))
    finally:ctx.close()


def test_actual_two_rank_global_mean_empty_rank_and_resume(tmp_path):
    import time
    import torch.multiprocessing as mp
    workers=mp.spawn(_rank,args=(str(tmp_path),),nprocs=2,join=False);deadline=time.monotonic()+90
    try:
        while not workers.join(timeout=1):
            if time.monotonic()>deadline:raise AssertionError('prior-only Gloo timeout')
    finally:
        for child in workers.processes:
            if child.is_alive():child.terminate();child.join(timeout=5)
    assert all(json.loads((tmp_path/f'rank{r}.json').read_text())['ok'] for r in (0,1))


def test_only_new_model_evaluated_actual_STC_rows_resume_and_visuals(tmp_path,monkeypatch):
    from test_coverage_eval_resume import Inputs,Predictor,record
    out=tmp_path/'new';out.mkdir();joint=tmp_path/'joint';(joint/'visuals').mkdir(parents=True)
    shape=(4,4,2);vol=np.prod(shape);gt=np.full(shape,17,np.uint8);gt[0,0,0]=11
    config=tmp_path/'config';config.write_text('config');dev=tmp_path/'dev';dev.write_text('dev');info=tmp_path/'info';info.write_text('info')
    records=[record(i) for i in range(4369)];keys=[[r['scene_name'],r['t0_token']] for r in records[:64]]
    source_counts=metrics.empty(3);source_counts['confusion'][:,17,17]=64*(vol-1);source_counts['confusion'][:,11,11]=64
    report=metrics.report(source_counts,(1.,2.,3.),moving=False)
    old_reports={f'{m}/{k}':{r:copy.deepcopy(report) for r in ('transport','semantic_refine')} for m in ('control','candidate') for k in ('occ','stc')}
    reference=dict(folder=joint,contract=dict(eval_keys=keys,model={}),checkpoint_sha256='joint',contract_sha256='jc',evaluation_sha256='je',
        original_contract=dict(data=dict(dev_info=joint_run.sha(info),dev_cache=joint_run.sha(dev))),
        evaluation=dict(receipt=dict(inventory={'occ':[],'stc':[]}),reports=old_reports))
    c=dict(protocol=control.PROTOCOL,objective=control.OBJECTIVE,total_updates=1031,world_size=2,parameters=2816554,
        implementations={},joint_dir=str(joint),joint_checkpoint_sha256='joint',joint_contract_sha256='jc',joint_evaluation_sha256='je',
        control_sha256='original',eval_keys=keys,model={},train_data=dict(config=joint_run.sha(config)),overrides=[])
    torch.save(dict(protocol=control.PROTOCOL,contract=c,updates=1031,epoch=1,batch=0,
        models={'gt':torch.nn.Linear(1,1).state_dict()}),out/'epoch_0001.pt')
    for kind in ('occ','stc'):
        for i in range(8):
            for role in ('control','candidate'):
                np.savez_compressed(joint/'visuals'/f'{kind}_{i+1:02d}_{role}.npz',gt=np.stack([gt]*3),
                    prediction=np.stack([gt]*3),transport=np.stack([gt]*3),
                    t0=np.full(shape,4 if kind=='occ' else 17,np.uint8),future_poses=np.stack([np.eye(4)]*3))
    source_hash=joint_run.sha(joint/'visuals/occ_01_control.npz');before=copy.deepcopy(old_reports);calls=[];labels=[];stop=threading.Event()
    class Source(Inputs):
        def preflight(self,*a):return []
        def targets(self,w,hs):
            assert len(calls)>len(labels);labels.append(w.t0_token)
            if len(labels)==3:stop.set()
            return [gt for _ in hs]
    def forward(net,p,row,**kw):
        calls.append(row[0]['t0_token']);n=len(row[0]['source_class_id'])
        assert n==(3 if row[1]['history_occ'][0,0,0,0]==4 else 1)
        logits=torch.zeros(6,18,vol);logits[:,17]=2;logits[:,11,0]=3
        return dict(shape=shape,prior=logits,logits=logits,prior_coverage=torch.ones(6,1,vol))
    from tools.joint_occ_experiments import run as base
    monkeypatch.setattr(evaluate,'reference',lambda _:reference);monkeypatch.setattr(base,'load_cache',lambda _:({},records))
    monkeypatch.setattr(evaluate,'load_runtime_config',lambda *a:{})
    monkeypatch.setattr(evaluate,'make_prepare_config',lambda _:SimpleNamespace(grid=SimpleNamespace(x_min=0,y_min=0,z_min=0,voxel_size=[1,1,1],shape_hwd=shape)))
    monkeypatch.setattr(evaluate,'DenseInputs',Source);monkeypatch.setattr(evaluate,'RecoveryPredictor',Predictor)
    monkeypatch.setattr(evaluate,'from_spec',lambda *a,**k:torch.nn.Linear(1,1));monkeypatch.setattr(evaluate,'forward_window',forward)
    args=SimpleNamespace(device='cpu',out_dir=out,dev_cache=dev,dev_info=info,config=config,dataroot=tmp_path,stc_root=tmp_path,cpu_workers=1)
    with pytest.raises(InterruptedError):evaluate.run(args,stop)
    assert json.loads((out/'evaluation_state.json').read_text())['progress']=={'occ':3}
    stop.clear();evaluate.run(args,stop)
    assert len(calls)==128 and len(labels)==128 and old_reports==before
    state=json.loads((out/'evaluation_state.json').read_text());assert state['progress']=={'occ':64,'stc':64}
    for count in state['counts'].values():
        for fields in count.values():assert np.asarray(fields['confusion']).sum()==64*vol*3
    assert joint_run.sha(joint/'visuals/occ_01_control.npz')==source_hash
    result=json.loads((out/'evaluation.json').read_text())
    assert result['reports']['candidate/stc']['semantic_refine']['average']['mIoU']==100
    assert len(list((out/'visuals').glob('*.npz')))==48
    visuals.render(out/'visuals');assert (out/'visuals/index.html').is_file() and (out/'visuals_and_metrics.zip').stat().st_size<2**20


def test_workflow_dual_train_single_eval_and_no_old_weight_resume():
    root=Path(__file__).resolve().parents[1]/'tools/coverage_prior_only'
    script=(root/'workflow.sh').read_text();eval_script=(root/'eval_only.sh').read_text()
    assert '--nproc_per_node=2' in script and 'PILOT_GPUS:-0,1' in script
    assert script.index('export CUDA_VISIBLE_DEVICES="$EVAL_GPU"')<script.index('tools.coverage_prior_only.evaluate')
    assert 'torch.distributed.run' not in eval_script and 'tools.coverage_prior_only.run ' not in eval_script
    assert '--epochs' not in run.parser().format_help() and '--joint-dir' in run.parser().format_help()
