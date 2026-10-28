"""One body, explicit provenance, exact original OCC and fixed STC forward paths."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_stc_only import common,run,parallel
from tools.causal_forward_stc_candidate import common as candidate


def test_provenance_adapter_no_target_reads_no_raw_mutation_and_fail_closed():
    raw=dict(history_occ=object(),history_observed=object(),history_poses=object());record={};times=object();calls=[]
    class Inputs:
        def history(self,job,kind):calls.append(kind);return record,raw,times
        def targets(self,*a):pytest.fail('labels used to select policy')
    for kind in ('occ','stc'):
        rec,value,at=common.ModalInputs(Inputs(),kind).history(None)
        assert rec is record and at is times and value[common.TAG]==kind and common.TAG not in raw
        assert value['history_occ'] is raw['history_occ']
    assert calls==['occ','stc']
    with pytest.raises(ValueError):common.ModalInputs(Inputs(),'unknown')
    predictor=object.__new__(common.InputPolicyPredictor)
    for kind in (None,'gt','guess_from_scores'):
        with pytest.raises(ValueError):predictor.prepare(record,dict(raw,**{common.TAG:kind}))
        with pytest.raises(ValueError):predictor.forward((record,{common.TAG:kind},None,None),poses=None)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_body_original_OCC_fixed_STC_alternation_all_six_and_no_weight_switch(device,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_fpn_adaln_zero import fixture
    from tools.causal_forward_full.model import make_model
    old,base,row=fixture(device)
    body=make_model(old.transport.config,feature_width=16,z_bins=4,memory_width=4).to(device).eval().requires_grad_(False)
    predictor=common.InputPolicyPredictor(body,base.provider.pcfg,device,1)
    calls=[];state_row=candidate.state_row
    monkeypatch.setattr(candidate,'state_row',lambda *a,**kw:(calls.append('stc') or state_row(*a,**kw)))
    try:
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        raw=dict(raw,**{common.TAG:'occ'})
        prepared=predictor.prepare(row[0],raw,times=np.arange(4)*.5)
        assert prepared[1][common.TAG]=='occ' and raw[common.TAG]=='occ'
        clean=prepared[0],{k:v for k,v in prepared[1].items() if k!=common.TAG},prepared[2],prepared[3]
        before={n:v.clone() for n,v in body.state_dict().items()};points=prepared[3][0]['memory_reads']['world'].copy()
        for kind in ('occ','stc','occ','stc'):
            tagged=prepared[0],dict(prepared[1],**{common.TAG:kind},future_gt_occ=object()),prepared[2],prepared[3]
            expected=(common.baseline.Epoch2Predictor.forward if kind=='occ' else candidate.CandidatePredictor.forward)(predictor,clean,poses=row[1]['future_poses'])
            n=len(calls);actual=predictor.forward(tagged,poses=row[1]['future_poses'])
            assert len(calls)-n==(kind=='stc') and actual['logits'].shape[0]==6
            torch.testing.assert_close(actual['logits'],expected['logits'],rtol=0,atol=0)
            assert not actual['logits'].requires_grad
        np.testing.assert_array_equal(prepared[3][0]['memory_reads']['world'],points)
        for n,v in before.items():torch.testing.assert_close(body.state_dict()[n],v,rtol=0,atol=0)
    finally:predictor.close();base.close()


def test_source_same_weights_and_two_policies_declared_without_mutating_contract(tmp_path,monkeypatch):
    from test_causal_forward_all import make_source
    source,c,_=make_source(tmp_path);path=source/'epoch_0002.pt'
    monkeypatch.setattr(candidate.recipe,'SOURCE_SHA256',candidate.base.sha256(path))
    before=copy.deepcopy(c);contract,model=common.load_frozen(path,'cpu');s=contract['evaluation_method']
    assert c==before and 'evaluation_method' not in c
    assert s['same_weights_both_modalities'] and not s['same_inference_policy_both_modalities']
    assert s['input_policies']['occ']['adaptation'] is False
    assert s['input_policies']['stc']['free_intercept']==-2 and 'GT future labels' in s['input_policies']['stc']['calibration']
    assert not any(p.requires_grad for p in model.parameters())
    text=run.summary(dict(status='running',contract=dict(windows=4369),reports={}))
    assert 'DIFFERENT input-specific' in text and 'OCC: original' in text and 'STC SUPERVISED' in text


def test_worker_actual_metric_engine_stop_resume_readonly_source(tmp_path,monkeypatch):
    import test_causal_forward_all as inherited
    original_make=inherited.make_source
    def pinned_source(root):
        source,c,files=original_make(root)
        monkeypatch.setattr(candidate.recipe,'SOURCE_SHA256',candidate.base.sha256(source/'epoch_0002.pt'))
        return source,c,files
    monkeypatch.setattr(inherited,'run',run);monkeypatch.setattr(inherited,'make_source',pinned_source)
    inherited.test_worker_actual_metric_engine_interrupt_stc_resume_and_source_readonly(tmp_path,monkeypatch)
    contract=json.loads((tmp_path/'worker/contract.json').read_text())
    assert contract['source_contract']['evaluation_method']==common.specification()


def test_two_children_same_GPU_integer_merge_policy_mismatch_and_resume(tmp_path,monkeypatch):
    from test_causal_forward_all import make_source,records,states_for
    source,_,files=make_source(tmp_path);checkpoint=source/'epoch_0002.pt';sha=candidate.base.sha256(checkpoint)
    monkeypatch.setattr(candidate.recipe,'SOURCE_SHA256',sha)
    c=common.validate_source(torch.load(checkpoint,weights_only=False));_,keys=run.select_jobs(records());p=parallel.runner;calls=[]
    monkeypatch.setattr(p.torch.cuda,'device_count',lambda:1);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(p.base,'load_cache',lambda *a:({},records()))
    monkeypatch.setattr(p.base,'load_runtime_config',lambda *a:None)
    monkeypatch.setattr(p.base,'make_prepare_config',lambda *a:SimpleNamespace(grid=SimpleNamespace(shape_hwd=(2,2,2))))
    class Child:
        def __init__(self,cmd,**kwargs):
            calls.append((cmd,kwargs['env']['CUDA_VISIBLE_DEVICES']));self.pid=1000+len(calls);self.returncode=0
            assert 'tools.causal_forward_stc_only.run' in cmd
            rank=int(cmd[cmd.index('--shard-index')+1]);folder=Path(cmd[cmd.index('--out-dir')+1]);folder.mkdir(exist_ok=True)
            for kind in ('occ','stc'):
                state=states_for(keys,sha,8,kind,c)[rank];state['contract']['base']['protocol']=run.PROTOCOL
                p.base.write_json(folder/f'{kind}_state.json',state)
            p.base.write_json(folder/'contract.json',state['contract']['base'])
            p.base.write_json(folder/'evaluation.json',dict(status='complete',contract=state['contract']['base']))
        def poll(self):return self.returncode
        def wait(self):return self.returncode
        def terminate(self):self.returncode=130
    monkeypatch.setattr(p.subprocess,'Popen',Child)
    out=tmp_path/'stc_only_all';args=['--source-dir',str(source),'--out-dir',str(out),'--dataroot',str(tmp_path),'--stc-root',str(tmp_path)]
    for n,path in files.items():args+=['--'+n.replace('_','-'),str(path)]
    assert parallel.main(argv=args)==0 and [gpu for _,gpu in calls]==['0','0']
    report=json.loads((out/'evaluation.json').read_text());assert report['status']=='complete' and len(report['reports'])==4
    assert report['contract']['source_contract']['evaluation_method']==common.specification()
    assert parallel.main(argv=args+['--resume'])==0 and all('--resume' in cmd for cmd,_ in calls[2:])
    states=states_for(keys,sha,8,'occ',c)
    for s in states:s['contract']['base']['protocol']=run.PROTOCOL
    _,raw=parallel.merge_states(states,keys,sha,8,'occ')
    assert sum(np.asarray(raw['gt/transport']['confusion'])[0].ravel())==4369*8
    states[0]['contract']['base']['source_contract']=copy.deepcopy(c)
    states[0]['contract']['base']['source_contract']['evaluation_method']['input_policies']['occ']['adaptation']=True
    with pytest.raises(RuntimeError,match='SAME declared'):parallel.merge_states(states,keys,sha,8,'occ')
    assert p.base.sha256(checkpoint)==sha
