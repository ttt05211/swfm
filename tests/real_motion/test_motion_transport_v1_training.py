import copy,torch
from real_motion.motion_transport_v1.engine import DistContext,rank_epoch_indices,optimizer_for,save_checkpoint,load_resume,_budget_should_stop
from real_motion.motion_transport_v1.ema import WarmupEMA
from real_motion.motion_transport_v1.losses import lambda_ratio,calibrated_gradient_ratio,output_gradient_lambda_floor

def cfg():return {'spec_version':'MT-V1-SPEC-2','training':{'weight_decay':.01,'peak_lr':3e-4},'runtime':{'config_path':'ignored'}}
def test_partition_and_ddp_normalization_algebra():
    rows=[]
    for rank in range(2):r,p=rank_epoch_indices(11,DistContext(rank,2,rank,torch.device('cpu')),3,3407);rows+=r;assert p==1
    assert set(rows)==set(range(11)) and len(rows)==12
    w=torch.tensor(.7,requires_grad=True);ls=[(w-1).square()+2,(2*w+.5).square()];rg=[torch.autograd.grad(2*x/2,w,retain_graph=True)[0] for x in ls];gg=torch.autograd.grad(sum(ls)/2,w)[0];assert torch.allclose(sum(rg)/2,gg)
def test_lambda_schedule():assert lambda_ratio(0)==1 and lambda_ratio(.1)==1 and abs(lambda_ratio(.2)-.625)<1e-12 and lambda_ratio(.3)==.25
def test_gradient_calibration_recovery_dominance_and_coordinate_floor():
    # f=.5 means ||g_ce|| <= .5 * ||lambda*g_motion||, hence lambda=10 here
    for cosine in (.3,-1.,-.25):assert calibrated_gradient_ratio(10,2,cosine,max_ce_antagonistic_fraction_of_motion=.5)==10
    assert calibrated_gradient_ratio(0,2,-1)!=calibrated_gradient_ratio(0,2,-1)
    go=torch.tensor([4.,-8.,1.]);gm=torch.tensor([-1.,2.,1.]);assert output_gradient_lambda_floor(go,gm,max_ce_antagonistic_fraction_of_motion=.5)==8.
def test_wall_clock_guard_includes_reserve_and_next_group(monkeypatch):
    c={'training':{'max_hours':1/3600,'wall_clock_final_reserve_seconds':.4,'wall_clock_next_group_guard_seconds':.4}};ctx=DistContext();
    import real_motion.motion_transport_v1.engine as e
    monkeypatch.setattr(e.time,'monotonic',lambda:10.3);stop,elapsed=_budget_should_stop(10.,c,ctx);assert stop and abs(elapsed-.3)<1e-9
    monkeypatch.setattr(e.time,'monotonic',lambda:10.1);stop,_=_budget_should_stop(10.,c,ctx);assert not stop
def test_checkpoint_resume(tmp_path):
    c=cfg();ctx=DistContext();m=torch.nn.Linear(3,2);opt=optimizer_for(m,c);ema=WarmupEMA(m,100);ema.start();loss=m(torch.ones(2,3)).square().mean();loss.backward();opt.step();opt.zero_grad();ema.update();manifest=tmp_path/'m.json';manifest.write_text('{}');msp=tmp_path/'m.pt';msp.write_bytes(b'x');p=tmp_path/'c.pt';state=copy.deepcopy(m.state_dict());save_checkpoint(p,raw=m,ema=ema,opt=opt,cfg=c,ctx=ctx,epoch=2,next_group=7,global_step=19,lambda_ref=12.5,manifest_path=manifest,msp_path=msp,selection_state={'best_epoch':2},phase='train');m2=torch.nn.Linear(3,2);o2=optimizer_for(m2,c);e2=WarmupEMA(m2);ck=load_resume(p,raw=m2,ema=e2,opt=o2,cfg=c,ctx=ctx,manifest_path=manifest,msp_path=msp);assert ck['global_step']==19 and e2.num_updates==1 and ck['phase']=='train'
    for k,v in state.items():assert torch.equal(m2.state_dict()[k],v)
