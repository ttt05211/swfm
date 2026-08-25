import numpy as np
from real_motion.metrics.moving_miou_v2 import Box3D,GridSpec,is_moving,moving_support,MovingMIoUV2Accumulator,rasterize_oriented_box

def test_interval_contract():
    a=Box3D("x",4,(0,0,0),(2,1,1),0); b=Box3D("x",4,(1,0,0),(2,1,1),0); assert is_moving(a,b,2.0,0.5); assert not is_moving(a,b,2.01,0.5)

def test_dual_support_penalizes_ghost_and_miss():
    g=GridSpec(-5,-5,-1,(0.5,0.5,0.5),(20,20,4)); a=Box3D("x",4,(-1,0,0),(1,1,1),0); b=Box3D("x",4,(1,0,0),(1,1,1),0); s=moving_support(a,b,2.0,g,0.5,0.0); gt=np.full(g.shape_hwd,17,dtype=np.int64); pred=gt.copy(); sa=rasterize_oriented_box(a,g,0); sb=rasterize_oriented_box(b,g,0); gt[sb]=4; pred[sa]=4; acc=MovingMIoUV2Accumulator([4]); acc.update(pred,gt,s); assert acc.compute()["mIoU"]==0.0
