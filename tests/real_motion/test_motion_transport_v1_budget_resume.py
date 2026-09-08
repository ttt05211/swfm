from pathlib import Path
from unittest.mock import patch
import torch
from real_motion.motion_transport_v1 import engine
from tests.real_motion.test_motion_transport_v1_acceptance_regressions import _train_cfg,_fake_pipe,_fake_prepare,_fake_forward,_report

def _prov(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}');msp=tmp_path/'msp.pt';msp.write_bytes(b'x');return manifest,msp

def test_epoch_dev_budget_guard_stops_before_eval_and_preserves_dev_pending(tmp_path):
    cfg=_train_cfg(epochs=1,max_hours=100/3600);cfg['training']['wall_clock_final_reserve_seconds']=35.;cfg['training']['wall_clock_next_group_guard_seconds']=1.;ctx=engine.DistContext();manifest,msp=_prov(tmp_path);clock=[0.];events=[]
    def slow_forward(*a,**kw):clock[0]+=50.;return _fake_forward(*a,**kw)
    def slow_eval(*a,**kw):events.append(clock[0]);clock[0]+=30.;return _report(10)
    with patch.object(engine,'prepare_scene',_fake_prepare),patch.object(engine,'forward_losses',slow_forward),patch.object(engine,'motion_pair_count',lambda *a:1),patch.object(engine.time,'monotonic',lambda:clock[0]),patch('real_motion.motion_transport_v1.evaluation.evaluate',slow_eval):
        r=engine.train(_fake_pipe(),None,[(None,'only')],[(None,None)],cfg,ctx,manifest_path=manifest,msp_path=msp,output_dir=tmp_path/'run')
    assert events==[]
    assert r['termination_reason']=='wall_clock_budget' and r['stop_stage']=='epoch_dev' and r['elapsed_s']==50.
    for name in ('latest.pt','last.pt'):
        ck=torch.load(tmp_path/'run'/name,weights_only=False);assert (ck['epoch'],ck['next_group'],ck['phase'],ck['global_step'])==(0,1,'dev_pending',1)

def test_mid_epoch_budget_stop_latest_and_last_resume_identically(tmp_path):
    cfg=_train_cfg(epochs=1);ctx=engine.DistContext();manifest,msp=_prov(tmp_path);base=tmp_path/'base';seen=[]
    def prep(pipe,source,w,c,cfg,**kw):seen.append(c);return _fake_prepare()
    with patch.object(engine,'prepare_scene',prep),patch.object(engine,'forward_losses',_fake_forward),patch.object(engine,'motion_pair_count',lambda *a:1),patch.object(engine,'_budget_should_stop',side_effect=[(False,0.),(True,1.)]),patch('real_motion.motion_transport_v1.evaluation.evaluate',lambda *a,**kw:_report(10)):
        r=engine.train(_fake_pipe(),None,[(None,'a'),(None,'b'),(None,'c')],[(None,None)],cfg,ctx,manifest_path=manifest,msp_path=msp,output_dir=base)
    assert r['stop_stage']=='train_group' and r['global_step']==1
    latest=torch.load(base/'latest.pt',weights_only=False);last=torch.load(base/'last.pt',weights_only=False)
    assert (latest['epoch'],latest['next_group'],latest['phase'],latest['global_step'])==(0,1,'budget_stop',1)
    assert (last['epoch'],last['next_group'],last['phase'],last['global_step'])==(0,1,'budget_stop',1)
    for k,v in latest['model_state_dict'].items():assert torch.equal(v,last['model_state_dict'][k])
    traces=[];states=[]
    for name in ('latest.pt','last.pt'):
        trace=[]
        def resume_prep(pipe,source,w,c,cfg,**kw):trace.append(c);return _fake_prepare()
        out=tmp_path/f'resume_{name.split(".")[0]}';pipe=_fake_pipe()
        with patch.object(engine,'prepare_scene',resume_prep),patch.object(engine,'forward_losses',_fake_forward),patch.object(engine,'motion_pair_count',lambda *a:1),patch('real_motion.motion_transport_v1.evaluation.evaluate',lambda *a,**kw:_report(10)):
            rr=engine.train(pipe,None,[(None,'a'),(None,'b'),(None,'c')],[(None,None)],cfg,ctx,manifest_path=manifest,msp_path=msp,output_dir=out,resume=base/name)
        traces.append(trace);states.append({k:v.detach().clone() for k,v in pipe.source_network.state_dict().items()});assert rr['global_step']==3
    assert traces[0]==traces[1] and len(traces[0])==2
    for k in states[0]:assert torch.equal(states[0][k],states[1][k])
