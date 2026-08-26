import numpy as np, torch
from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import Box3D,GridSpec,is_moving_world,moving_support_from_world_motion,MovingMIoUV2MultiHorizon


def test_composition_write_support_and_static_protection():
    sta=torch.tensor([[1,1,1,1]])
    wm=torch.tensor([[4,4,4,4]])
    conf=torch.tensor([[1,0,0,0]],dtype=torch.bool)
    write=torch.tensor([[1,1,0,1]],dtype=torch.bool)
    out=static_protected_compose(sta,wm,conf,[4],write)
    assert out.tolist()==[[1,4,1,4]]


def test_world_motion_and_dual_box_support():
    g=GridSpec(-5,-5,-1,(0.5,0.5,0.5),(20,20,4))
    a=Box3D('x',4,(-1,0,0),(1,1,1),0); b=Box3D('x',4,(1,0,0),(1,1,1),0)
    assert is_moving_world((0,0,0),(1,0,0),2.0,0.5)
    s=moving_support_from_world_motion((0,0,0),(1,0,0),a,b,2.0,g,0.5,0.0)
    assert s.any()


def test_horizon_first_mean():
    metric=MovingMIoUV2MultiHorizon()
    # Only class 4 has support; other frozen classes absent => ignored.
    support=np.ones((1,1,2),dtype=bool); gt=np.array([[[4,4]]])
    metric.update(1.0,gt.copy(),gt,support)
    metric.update(2.0,np.array([[[17,17]]]),gt,support)
    metric.update(3.0,np.array([[[4,17]]]),gt,support)
    assert abs(metric.compute()['mIoU']-50.0)<1e-6

def test_causal_dynamic_target_is_not_defined_by_future_motion_metric():
    from real_motion.nuscenes_adapter import causal_dynamic_target_semantics
    gt=np.full((2,2,2),17,dtype=np.int64)
    gt[0,0,0]=4      # dynamic semantic object inside causal support
    gt[1,1,0]=1      # static semantic class inside support must stay out of WM target
    sup=np.zeros((2,2),dtype=bool); sup[0,0]=1; sup[1,1]=1
    out=causal_dynamic_target_semantics(gt,sup)
    assert out[0,0,0]==4
    assert out[1,1,0]==17

def test_box_record_roundtrip_for_subset_evaluator():
    from real_motion.nuscenes_adapter import box3d_to_dict,box3d_from_dict
    b=Box3D('tok',4,(1.0,2.0,0.0),(4.0,2.0,1.5),0.3)
    assert box3d_from_dict(box3d_to_dict(b))==b


def test_maneuver_bucket_uses_turn_rate_not_raw_horizon_angle():
    from real_motion.metrics.stratified import maneuver_bucket
    import math
    assert maneuver_bucket(0.2, math.radians(12), 1.0, math.radians(10)) == "turning"
    assert maneuver_bucket(1.2, 0.0, 1.0, math.radians(10)) == "accel/decel"
