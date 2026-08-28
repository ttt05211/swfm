import numpy as np
from real_motion.metrics.moving_miou_v2 import (
    Box3D, GridSpec, is_moving_world, moving_support,
    MovingMIoUV2Accumulator, MovingMIoUV2MultiHorizon,
)


def test_interval_contract():
    assert is_moving_world((0,0,0),(1,0,0),2.0,0.5)  # equality is moving
    assert not is_moving_world((0,0,0),(1,0,0),2.01,0.5)


def test_dual_support_penalizes_ghost_and_miss():
    g=GridSpec(-5,-5,-1,(0.5,0.5,0.5),(20,20,4))
    a=Box3D("x",4,(-1,0,0),(1,1,1),0)
    b=Box3D("x",4,(1,0,0),(1,1,1),0)
    s=moving_support(a,b,2.0,g,0.5,0.0)
    gt=np.full(g.shape_hwd,17,dtype=np.int64); pred=gt.copy()
    from real_motion.metrics.moving_miou_v2 import rasterize_oriented_box
    sa=rasterize_oriented_box(a,g,0); sb=rasterize_oriented_box(b,g,0)
    gt[sb]=4; pred[sa]=4
    acc=MovingMIoUV2Accumulator(); acc.update(pred,gt,s)
    assert acc.compute()["mIoU"]==0.0


def test_raster_axis0_is_metric_x():
    from real_motion.metrics.moving_miou_v2 import rasterize_oriented_box
    g=GridSpec(0,0,0,(1,1,1),(4,7,2))
    # Center x=1.5, y=5.5 must hit array [1,5,*], not [5,1,*].
    b=Box3D("x",4,(1.5,5.5,0.5),(0.8,0.8,0.8),0)
    s=rasterize_oriented_box(b,g,0)
    assert s[1,5,0]
    assert s.shape==(4,7,2)


def test_multihorizon_contract_averages_horizon_mious_not_voxels():
    metric=MovingMIoUV2MultiHorizon()
    support=np.ones((1,1,2),dtype=bool); gt=np.array([[[4,4]]])
    metric.update(1.0,gt.copy(),gt,support)
    metric.update(2.0,np.array([[[17,17]]]),gt,support)
    metric.update(3.0,np.array([[[4,17]]]),gt,support)
    assert abs(metric.compute()["mIoU"]-50.0)<1e-6
