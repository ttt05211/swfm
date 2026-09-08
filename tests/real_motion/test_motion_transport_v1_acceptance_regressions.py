from __future__ import annotations
import importlib.util, shutil, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

from real_motion.motion_transport_v1 import engine

ROOT=Path(__file__).resolve().parents[2]

def _load_tool(name,rel):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_bootstrap_reports_percentage_points():
    ev=_load_tool('mt_v1_eval_tool',Path('tools/real_motion/eval_motion_transport_v1.py'))
    def row(n):return {m:{h:{'inter':{4:n},'union':{4:100}} for h in ('1.0','2.0','3.0')} for m in ('overall','moving')}
    got=ev.bootstrap_delta({'scene0':{'hard:0':row(50),'hard:16':row(60)}},repeats=20,seed=1)
    for m in ('overall','moving'):
        assert got[m]['mean_pp']==10.0
        assert got[m]['ci95_pp']==[10.0,10.0]
        assert got[m]['p_gt_0']==1.0

def test_profile_group_timing_includes_deliberately_slow_data_preparation():
    prof=_load_tool('mt_v1_profile_tool',Path('tools/real_motion/profile_motion_transport_v1.py'))
    class SlowDS:
        def __len__(self):return 1
        def __getitem__(self,idx):time.sleep(.02);return object(),object()
    class Net(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.));self.activation_checkpointing=False
    class Pipe:
        def __init__(self):self.source_network=Net();self.device=torch.device('cpu');self.grid=object();self.strong_cfg=object()
    def prep(*a,**kw):time.sleep(.03);return SimpleNamespace(decomp=SimpleNamespace(sources=[]),selected=(),targets=None,budget_mode='test')
    def fwd(pipe,rec,cfg,**kw):
        loss=pipe.source_network.w.square();z=loss*0;return loss,z,0,z,torch.zeros((0,6,3)),None,None,{}
    cfg={'paths':{'msp_checkpoint':None},'training':{'accumulation_steps':1,'gradient_clip_norm':1.,'weight_decay':0.,'peak_lr':1e-3},'routing':{'train_later_sparse_budget':16}}
    with patch.object(prof,'MotionTransportV1',lambda *a,**kw:Pipe()),patch.object(prof,'_representative',lambda *a,**kw:[0]),patch.object(prof,'prepare_scene',prep),patch.object(prof,'forward_losses',fwd),patch.object(prof,'motion_pair_count',lambda *a:0):
        _,_,_,stats=prof._pass(cfg,None,SlowDS(),engine.DistContext(),1.,False,0,1,3407)
    assert stats['mean_prepare_s']>=.04,stats
    assert stats['mean_group_s']>=stats['mean_prepare_s']

def _train_cfg(epochs=3,max_hours=4.):
    return {'spec_version':'MT-V1-SPEC-2','training':{'epochs_locked':epochs,'accumulation_steps':1,'initial_seed':3407,'peak_lr':1e-3,'end_lr':1e-5,'weight_decay':0.,'gradient_clip_norm':1.,'lr_warmup_fraction':.05,'max_hours':max_hours,'wall_clock_final_reserve_seconds':0.,'wall_clock_next_group_guard_seconds':0.,'ema':{'enabled':True,'half_life_optimizer_steps':100}},'loss':{'lambda_reference':1.},'evaluation':{'overall_noninferiority_pp':-.1,'stationary_movable_noninferiority_pp':-.2}}

def _fake_pipe():return SimpleNamespace(source_network=torch.nn.Linear(1,1),device=torch.device('cpu'))
def _fake_prepare(*a,**kw):return SimpleNamespace(selected=(0,),targets=None)
def _fake_forward(pipe,rec,cfg,**kw):
    x=pipe.source_network(torch.ones(1,1));v=(x-.1).square().mean();return v,v,1,x.sum()*0,x.expand(1,18).reshape(1,6,3),None,None,{}
def _report(m):return {'delta_vs_kta':{'16':{'overall_pp':0.,'stationary_movable_pp':0.}},'hard':{'16':{'moving':{'mIoU':float(m)}}}}

def test_latest_selection_survives_resume_before_and_after_dev(tmp_path):
    cfg=_train_cfg();ctx=engine.DistContext();ds=[(None,None)];manifest=tmp_path/'manifest.json';manifest.write_text('{}');msp=tmp_path/'msp.pt';msp.write_bytes(b'x');base=tmp_path/'base';pending=tmp_path/'pending.pt';complete=tmp_path/'complete.pt';vals=iter([10.,12.,11.,11.]);real_save=engine.save_checkpoint
    def capture(path,**kw):
        real_save(path,**kw)
        if Path(path).name=='latest.pt' and kw.get('phase')=='dev_pending' and kw.get('epoch')==1:shutil.copy2(path,pending)
        if Path(path).name=='latest.pt' and kw.get('phase')=='epoch_complete' and kw.get('epoch')==2:shutil.copy2(path,complete)
    with patch.object(engine,'prepare_scene',_fake_prepare),patch.object(engine,'forward_losses',_fake_forward),patch.object(engine,'motion_pair_count',lambda *a:1),patch.object(engine,'save_checkpoint',capture),patch('real_motion.motion_transport_v1.evaluation.evaluate',lambda *a,**kw:_report(next(vals))):
        engine.train(_fake_pipe(),None,ds,ds,cfg,ctx,manifest_path=manifest,msp_path=msp,output_dir=base)
    assert torch.load(base/'best.pt',weights_only=False)['selection_state']['best_moving']==12.
    assert torch.load(pending,weights_only=False)['selection_state']['best_moving']==10.
    assert torch.load(complete,weights_only=False)['selection_state']['best_moving']==12.
    vals=iter([12.,11.,11.]);out1=tmp_path/'resume_pending'
    with patch.object(engine,'prepare_scene',_fake_prepare),patch.object(engine,'forward_losses',_fake_forward),patch.object(engine,'motion_pair_count',lambda *a:1),patch('real_motion.motion_transport_v1.evaluation.evaluate',lambda *a,**kw:_report(next(vals))):
        r=engine.train(_fake_pipe(),None,ds,ds,cfg,ctx,manifest_path=manifest,msp_path=msp,output_dir=out1,resume=pending)
    assert r['selection_state']['best_moving']==12.
    vals=iter([11.,11.]);out2=tmp_path/'resume_complete'
    with patch.object(engine,'prepare_scene',_fake_prepare),patch.object(engine,'forward_losses',_fake_forward),patch.object(engine,'motion_pair_count',lambda *a:1),patch('real_motion.motion_transport_v1.evaluation.evaluate',lambda *a,**kw:_report(next(vals))):
        r=engine.train(_fake_pipe(),None,ds,ds,cfg,ctx,manifest_path=manifest,msp_path=msp,output_dir=out2,resume=complete)
    assert r['selection_state']['best_moving']==12.

def test_short_wall_clock_budget_exits_with_recoverable_checkpoint(tmp_path):
    cfg=_train_cfg(epochs=2,max_hours=0.);ctx=engine.DistContext();manifest=tmp_path/'manifest.json';manifest.write_text('{}');msp=tmp_path/'msp.pt';msp.write_bytes(b'x');out=tmp_path/'budget'
    with patch('real_motion.motion_transport_v1.evaluation.evaluate',lambda *a,**kw:{}):
        r=engine.train(_fake_pipe(),None,[(None,None)],[(None,None)],cfg,ctx,manifest_path=manifest,msp_path=msp,output_dir=out)
    assert r['termination_reason']=='wall_clock_budget'
    ck=torch.load(out/'latest.pt',weights_only=False);assert ck['phase']=='budget_stop' and ck['next_group']==0
    # Resume provenance/state is loadable even though no optimizer step occurred.
    m=_fake_pipe().source_network;o=engine.optimizer_for(m,cfg);from real_motion.motion_transport_v1.ema import WarmupEMA
    e=WarmupEMA(m,100);loaded=engine.load_resume(out/'latest.pt',raw=m,ema=e,opt=o,cfg=cfg,ctx=ctx,manifest_path=manifest,msp_path=msp);assert loaded['phase']=='budget_stop'
