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
        assert np.isclose(got[m]['mean_pp'],10.0,atol=1e-12,rtol=0)
        assert np.allclose(got[m]['ci95_pp'],[10.0,10.0],atol=1e-12,rtol=0)
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

def test_null_wall_clock_budget_disables_budget_stop_even_with_large_elapsed_time():
    cfg=_train_cfg(max_hours=None);ctx=engine.DistContext()
    with patch.object(engine.time,'monotonic',return_value=1e9):
        stop,elapsed,required=engine._phase_budget_should_stop(0.,cfg,ctx,current_phase_s=123.,post_phase_reserve_s=456.,next_guard_s=789.)
    assert stop is False and elapsed==1e9 and required==1368.

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
    cfg=_train_cfg(epochs=2,max_hours=1e-9);ctx=engine.DistContext();manifest=tmp_path/'manifest.json';manifest.write_text('{}');msp=tmp_path/'msp.pt';msp.write_bytes(b'x');out=tmp_path/'budget'
    with patch('real_motion.motion_transport_v1.evaluation.evaluate',lambda *a,**kw:{}):
        r=engine.train(_fake_pipe(),None,[(None,None)],[(None,None)],cfg,ctx,manifest_path=manifest,msp_path=msp,output_dir=out)
    assert r['termination_reason']=='wall_clock_budget'
    ck=torch.load(out/'latest.pt',weights_only=False);assert ck['phase']=='budget_stop' and ck['next_group']==0
    m=_fake_pipe().source_network;o=engine.optimizer_for(m,cfg);from real_motion.motion_transport_v1.ema import WarmupEMA
    e=WarmupEMA(m,100);loaded=engine.load_resume(out/'latest.pt',raw=m,ema=e,opt=o,cfg=cfg,ctx=ctx,manifest_path=manifest,msp_path=msp);assert loaded['phase']=='budget_stop'

def test_soft_q16_keeps_empty_source_windows_in_same_sample_set():
    import real_motion.motion_transport_v1.evaluation as ev
    from real_motion.motion_transport_v1.contracts import TrainingTargets,SourceDecomposition,SoftScene,SoftHorizon
    shape=(2,2,1);free=np.full((6,*shape),17,np.uint8);empty_rest=np.zeros((0,3));h=np.arange(1,7)*.5
    class DS:
        def __len__(self):return 2
        def __getitem__(self,i):
            w=SimpleNamespace(scene_name=f'scene{i}',t0_token='t0',future_tokens=tuple(f'f{j}' for j in range(6)))
            c=SimpleNamespace(sample_id=f's{i}',scene_name=f'scene{i}',future_ego_to_world=tuple(np.eye(4) for _ in range(6)))
            return w,c
    class Net:
        def eval(self):return self
    class Pipe:
        def __init__(self):
            self.source_network=Net();self.grid=SimpleNamespace(shape_hwd=shape,x_min=0.,y_min=0.,z_min=0.,voxel_size=(.4,.4,.4))
        def prepare_scene(self,c):
            sources=[] if c.sample_id=='s0' else [SimpleNamespace(source_id=0,class_id=4,crop_eligible=True,fallback_reason='',mapping_coverage=1.,velocity_world=np.zeros(3),dormant_fraction=0.,observed_moving_fraction=1.,points_world=np.zeros((1,3)))]
            return SourceDecomposition(sources,free.copy(),np.zeros((0,3),int),np.zeros(0,np.uint8),empty_rest.copy(),h),[],{}
        def forward_selected(self,c,d,sel,**kw):
            rows=[SoftHorizon(torch.zeros(0,dtype=torch.long),torch.zeros((0,18)),torch.zeros(0,dtype=torch.long)) for _ in range(6)];soft=SoftScene(rows,free.copy(),shape);return None,None,free.copy(),soft,{}
    targets=TrainingTargets(free.copy(),np.ones_like(free,bool),{})
    groups=lambda: {n:np.zeros(shape,bool) for n in ev.GROUP_NAMES}
    def route(sources,budget,*a,**kw):return () if (not sources or budget==0) else (0,)
    with patch.object(ev,'build_training_targets',lambda *a,**kw:targets),patch.object(ev,'route_sources',route),patch.object(ev,'hard_kta_identity',lambda *a,**kw:free.copy()),patch.object(ev,'gt_moving_support_for_horizon',lambda *a,**kw:(np.zeros(shape,bool),[],{})),patch.object(ev,'stationary_movable_support',lambda *a,**kw:np.zeros(shape,bool)),patch.object(ev,'_groups',lambda *a,**kw:groups()):
        rep=ev.evaluate(Pipe(),SimpleNamespace(nusc=object()),DS(),{'targets':{'best_box_source_coverage_min':.8,'second_box_source_coverage_max':.2,'motion_points_per_source_max':64},'input':{'class_count':18},'evaluation':{'main_budget_sources':16}},budgets=(0,16),strategy='msp',include_soft_main=True)
    assert rep['protocol']['hard_windows']==2
    assert rep['protocol']['soft_main_windows']==2
    assert rep['source_calls']['0']==0
    assert rep['source_calls']['16']==1
