import torch
from real_motion.windows import WindowPlanner,crop_windows,scatter_windows,window_coverage
from real_motion.support import build_motion_tube,MotionTubeConfig


def test_context_does_not_create_history_only_window():
    future=torch.zeros(1,2,10,10,dtype=torch.bool); future[0,1,8,8]=1
    context=torch.zeros(1,4,10,10,dtype=torch.bool); context[0,0,1,1]=1; context[0,3,8,8]=1
    p=WindowPlanner((4,4),4).plan(future,context)
    assert int(p.valid.sum())==1
    assert float(window_coverage(future,p)[0])==1.0


def test_vectorized_crop_scatter_roundtrip_covered():
    s=torch.zeros(1,1,10,10,dtype=torch.bool); s[0,0,2,2]=1; s[0,0,7,7]=1
    p=WindowPlanner((4,4),4).plan(s)
    x=torch.arange(100,dtype=torch.float32).reshape(1,1,10,10)
    w=crop_windows(x,p); out=scatter_windows(w,p,base=torch.full_like(x,-1))
    cover=torch.zeros_like(s)
    for k in range(p.valid.shape[1]):
        if p.valid[0,k]:
            y,x0=p.origins[0,k].tolist(); cover[0,0,y:y+4,x0:x0+4]=1
    assert torch.equal(out[cover],x[cover])
    assert torch.all(out[~cover]==-1)


def test_tube_horizon_dilation():
    k=torch.zeros(3,10,10,dtype=torch.bool); k[:,5,5]=1
    t=build_motion_tube(k,MotionTubeConfig((0,1,2),0))
    assert int(t[0].sum()) < int(t[1].sum()) < int(t[2].sum())

def test_context_tiebreak_can_shift_window_without_sacrificing_future_coverage():
    future=torch.zeros(1,1,8,8,dtype=torch.bool); future[0,0,4,4]=1
    context=future.clone(); context[0,0,2,2]=1
    p=WindowPlanner((4,4),1).plan(future,context)
    y,x0=p.origins[0,0].tolist()
    assert y <= 2 < y+4 and x0 <= 2 < x0+4
    assert y <= 4 < y+4 and x0 <= 4 < x0+4
