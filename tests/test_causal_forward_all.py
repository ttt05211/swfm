"""Current epoch2 only, complete population, same-GPU workers and integer merge."""
import copy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_all import run,parallel,common
from tools.final_semantic_evaluation import metrics


def make_source(root):
    from test_causal_forward_average import body_and_contract
    body,c=body_and_contract();files={n:root/n for n in ('config','dev_info','dev_cache')}
    for p in files.values():p.write_text('fixture')
    c['data']={n:run.runner.base.sha256(p) for n,p in files.items()}
    folder=root/'adaln_forward_full_2ep_one';folder.mkdir()
    saved=dict(protocol=common.TRAIN_PROTOCOL,contract=c,epoch=2,batch=0,updates=10292,models={'gt':body.state_dict()})
    torch.save(saved,folder/'epoch_0002.pt')
    (folder/'results.json').write_text(json.dumps(dict(status='complete',epoch=2,updates=10292,contract=c)))
    return folder,c,files


def records():
    return [dict(scene_name='VAL'+str(i%150),t0_token=str(i),history_tokens=('old1','old2','a','b','c',str(i)),
        future_tokens=tuple('123456')) for i in range(4369)]


def states_for(keys,sha,voxels,kind,c=None):
    result=[]
    for rank in range(2):
        subset=keys[rank::2];counts=metrics.empty(3)
        counts['confusion'][:,4,4 if rank==0 else 17]=len(subset)*voxels;counts['moving'][:]=counts['confusion']
        header=dict(protocol=run.PROTOCOL,source_sha256=sha,source_contract=c,keys=subset,shard=[rank,2],
            windows=len(subset),parent_windows=4369,known=dict(occ='dense_valid',stc='dense_valid'))
        result.append(dict(contract=dict(base=header,modality=kind),keys=subset,ego_errors=[],
            counts={n:{k:v.tolist() for k,v in counts.items()} for n in ('gt/transport','gt/semantic_refine')}))
    return result


def test_current_epoch2_only_all4369_no_mean_dependency_and_strict_selection(tmp_path):
    jobs,keys=run.select_jobs(records());assert len(jobs)==len(keys)==4369 and len({k[0] for k in keys})==150
    with pytest.raises(RuntimeError):run.select_jobs(records()[:-1])
    source,c,_=make_source(tmp_path);path=source/'epoch_0002.pt';before=run.runner.base.sha256(path)
    loaded,body=common.load_frozen(path,'cpu');assert loaded==c and not body.training and not any(p.requires_grad for p in body.parameters())
    saved=torch.load(path,weights_only=False)
    for bad in (dict(saved,protocol='mean3_or_old'),dict(saved,updates=10000),dict(saved,epoch=1,batch=0)):
        with pytest.raises(RuntimeError):common.validate_source(bad)
    assert common.select_source(None,tmp_path)==path and run.runner.base.sha256(path)==before
    second=tmp_path/'adaln_forward_full_2ep_two';second.mkdir();(second/'results.json').write_bytes((source/'results.json').read_bytes())
    with pytest.raises(RuntimeError,match='ONE'):common.select_source(None,tmp_path)
    assert common.select_source(source,tmp_path)==path


def test_integer_merge_not_average_of_shard_scores_and_reject_duplicate_or_partial():
    _,keys=run.select_jobs(records());states=states_for(keys,'SHA',8,'occ')
    result,_=parallel.merge_states(states,keys,'SHA',8,'occ')
    assert result['occ_gt/semantic_refine']['average']['mIoU']==pytest.approx(100*2185/4369)
    assert result['occ_gt/semantic_refine']['average']['mIoU']!=pytest.approx(50.,abs=1e-6)
    bad=copy.deepcopy(states);bad[1]['keys']=bad[0]['keys']
    with pytest.raises(RuntimeError):parallel.merge_states(bad,keys,'SHA',8,'occ')
    bad=copy.deepcopy(states);bad[0]['counts']['gt/transport']['confusion'][0][4][4]-=1
    with pytest.raises(RuntimeError):parallel.merge_states(bad,keys,'SHA',8,'occ')


def test_coordinator_two_processes_same_gpu_current_module_and_resume(tmp_path,monkeypatch):
    source,c,files=make_source(tmp_path);checkpoint=source/'epoch_0002.pt';_,keys=run.select_jobs(records())
    p=parallel.runner;calls=[]
    monkeypatch.setattr(p.torch.cuda,'device_count',lambda:1);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(p.base,'load_cache',lambda *a:({},records()))
    monkeypatch.setattr(p.base,'load_runtime_config',lambda *a:None)
    monkeypatch.setattr(p.base,'make_prepare_config',lambda *a:SimpleNamespace(grid=SimpleNamespace(shape_hwd=(2,2,2))))
    class Child:
        def __init__(self,cmd,**kw):
            self.pid=1000+len(calls);self.returncode=0;calls.append((cmd,kw['env']['CUDA_VISIBLE_DEVICES']))
            assert 'tools.causal_forward_all.run' in cmd
            rank=int(cmd[cmd.index('--shard-index')+1]);out=Path(cmd[cmd.index('--out-dir')+1]);out.mkdir(exist_ok=True)
            for kind in ('occ','stc'):
                state=states_for(keys,p.base.sha256(checkpoint),8,kind,c)[rank];p.base.write_json(out/f'{kind}_state.json',state)
            p.base.write_json(out/'contract.json',state['contract']['base'])
            p.base.write_json(out/'evaluation.json',dict(status='complete',contract=state['contract']['base']))
        def poll(self):return self.returncode
        def wait(self):return self.returncode
        def terminate(self):self.returncode=130
    monkeypatch.setattr(p.subprocess,'Popen',Child)
    out=tmp_path/'all';args=['--source-dir',str(source),'--out-dir',str(out),'--dataroot',str(tmp_path),'--stc-root',str(tmp_path)]
    for k,path in files.items():args+=['--'+k.replace('_','-'),str(path)]
    before=p.base.sha256(checkpoint);assert parallel.main(argv=args)==0
    assert [v for _,v in calls]==['0','0']
    result=json.loads((out/'evaluation.json').read_text());assert result['status']=='complete' and len(result['reports'])==4
    assert result['contract']['physical_GPUs']==1 and result['contract']['inference_processes']==2
    assert parallel.main(argv=args+['--resume'])==0 and all('--resume' in cmd for cmd,_ in calls[2:])
    assert p.base.sha256(checkpoint)==before and (out/'occ_merged_counts.json').exists()


def test_worker_actual_metric_engine_interrupt_stc_resume_and_source_readonly(tmp_path,monkeypatch):
    source,_,files=make_source(tmp_path);worker=run.runner;event=Event();calls=[]
    jobs,keys=run.select_jobs(records());jobs,keys=jobs[:2],keys[:2]
    monkeypatch.setattr(worker,'select_jobs',lambda *a:(jobs,keys))  # Tiny execution test, not an actual full score.
    monkeypatch.setattr(worker.base,'load_cache',lambda *a:({},[]))
    monkeypatch.setattr(worker.base,'load_runtime_config',lambda *a:None)
    shape=(2,2,2);pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=shape))
    monkeypatch.setattr(worker.base,'make_prepare_config',lambda *a:pcfg)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**kw:pytest.fail('optimizer created'))
    class Inputs:
        def __init__(self,*a,**kw):pass
        def preflight(self,*a):return []
        def history(self,job,kind):
            calls.append((kind,job.t0_token))
            if len(calls)==3:event.set()
            return {},dict(history_observed=np.ones((4,*shape),bool)),np.arange(4)*.5
        def poses(self,*a):return np.repeat(np.eye(4)[None],6,0)
        def targets(self,*a):return np.full((3,*shape),4,np.uint8)
        def moving(self,*a):return np.ones((3,*shape),bool)
    class Predictor:
        def __init__(self,body,pcfg,device,workers):self.provider=SimpleNamespace(pcfg=pcfg,workers=workers)
        def prepare(self,rec,raw,**kw):assert raw['history_observed'].all();return None
        def forward(self,*a,**kw):return None
        def dense(self,*a):return {n:np.full((6,*shape),4,np.uint8) for n in ('transport','semantic_refine')}
        def close(self):pass
    monkeypatch.setattr(worker,'NuscInputs',Inputs);monkeypatch.setattr(worker,'EgoPredictor',Predictor)
    out=tmp_path/'worker';args=['--source-dir',str(source),'--out-dir',str(out),'--device','cpu','--dataroot',str(tmp_path),'--stc-root',str(tmp_path)]
    for k,path in files.items():args+=['--'+k.replace('_','-'),str(path)]
    before=worker.base.sha256(source/'epoch_0002.pt')
    assert run.main(event,args)==130 and len(calls)==3
    event.clear();assert run.main(event,args+['--resume'])==0 and len(calls)==4
    assert run.main(event,args+['--resume'])==0 and len(calls)==4
    result=json.loads((out/'evaluation.json').read_text());assert result['status']=='complete' and len(result['reports'])==4
    assert worker.base.sha256(source/'epoch_0002.pt')==before


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_current_predictor_H6_legacy_metadata_and_no_target_inputs(device,monkeypatch):
    import test_causal_forward_average as inherited
    monkeypatch.setattr(inherited,'run',SimpleNamespace(ForwardMeanPredictor=common.Epoch2Predictor))
    inherited.test_real_current_forward_predictor_six_metadata_H6_no_labels_or_age(device)


def test_progress_valid_prefix_and_inflight_json(tmp_path,monkeypatch):
    import test_source_scene_ego_all as inherited
    monkeypatch.setattr(inherited,'parallel',parallel.runner)
    inherited.test_saved_progress_matches_complete_prefix_and_handles_inflight_write(tmp_path)
