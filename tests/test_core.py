import torch
from real_motion.support import build_motion_tube,MotionTubeConfig,coverage_and_active_ratio
from real_motion.windows import WindowPlanner,crop_windows,scatter_windows
from real_motion.composition import static_protected_compose

def test_tube_and_coverage():
    x=torch.zeros(2,3,10,10,dtype=torch.bool); x[:,0,5,5]=1; x[:,1,5,6]=1; x[:,2,5,7]=1; y=build_motion_tube(x,MotionTubeConfig((0,1,2),0)); assert y[:,0].sum()==2; assert y[:,1].sum()>y[:,0].sum(); c,a=coverage_and_active_ratio(x,y); assert c==1.0 and 0<a<1

def test_window_roundtrip_nonoverlap():
    s=torch.zeros(1,1,10,10,dtype=torch.bool); s[0,0,1,1]=1; s[0,0,8,8]=1; p=WindowPlanner((4,4),4).plan(s); x=torch.arange(100.).reshape(1,1,10,10); w=crop_windows(x,p); out=scatter_windows(w,p,base=torch.zeros_like(x)); covered=torch.zeros_like(s)
    for k in range(p.valid.shape[1]):
        if p.valid[0,k]: y,x0=p.origins[0,k].tolist(); covered[0,0,y:y+4,x0:x0+4]=1
    assert torch.equal(out[covered],x[covered])

def test_static_protection():
    sta=torch.tensor([[1,1,1,1]]); wm=torch.tensor([[4,4,2,4]]); conf=torch.tensor([[1,0,0,0]],dtype=torch.bool); out=static_protected_compose(sta,wm,conf,dynamic_classes=[4]); assert out.tolist()==[[1,4,1,4]]

def test_planning_support_can_cover_history_without_expanding_loss():
    planning=torch.zeros(1,4,10,10,dtype=torch.bool); planning[0,0,1,1]=1; planning[0,3,8,8]=1; future_loss=torch.zeros(1,2,10,10,dtype=torch.bool); future_loss[0,1,8,8]=1; plan=WindowPlanner((4,4),4).plan(planning); assert plan.valid.sum()>=2; cropped=crop_windows(future_loss,plan); assert int(cropped.sum())==1
