import numpy as np
import torch
from types import SimpleNamespace
from real_motion.canonical_causal_repair import RepairPlan
from real_motion.ccr_static_frozen_policies import (
    POLICIES,direct_priority_plan,static_policy_score,best_safe_variant,
)
from tools.real_motion.run_p0_f9_ccr_static_one_shot import (
    _analyze_unique_fp,_make_counts,
)


def _evidence():
    # road direct and sidewalk halo at voxel 1; sidewalk direct and road
    # halo at voxel 2; road and sidewalk direct at voxel 3.
    return SimpleNamespace(
        actor=np.array([-2,-2,-2,-2,-2,-2,0],np.int32),
        classes=np.array([11,13,13,11,11,13,4],np.uint8),
        presence=np.array([[1,0,0,0],[0,0,0,0],
                           [1,0,0,0],[0,0,0,0],
                           [1,0,0,0],[1,0,0,0],
                           [1,0,0,0]],bool),
    )


def _plan():
    flat=np.full((7,6),-1,np.int64)
    flat[:,0]=[1,1,2,2,3,3,4]
    base=np.full((7,6),17,np.uint8)
    legal=np.zeros((7,6,2),bool)
    legal[6,0,0]=True
    return RepairPlan(
        flat=flat,base=base,fallback=base.copy(),
        legal=legal,context=np.zeros((7,6,8),np.float32))


def test_direct_priority_only_reenables_unambiguous_real_static_evidence():
    ev=_evidence();p=_plan();base=[np.full(6,17,np.uint8) for _ in range(6)]
    q,counts=direct_priority_plan(ev,p,base)
    assert q.legal[0,0,0]
    assert not q.legal[1,0,0]
    assert q.legal[2,0,0]
    assert not q.legal[3,0,0]
    assert not q.legal[4,0,0] and not q.legal[5,0,0]
    assert q.legal[6,0,0] and p.legal[6,0,0]
    assert counts["road_direct_over_sidewalk_halo"]==1
    assert counts["sidewalk_direct_over_road_halo"]==1
    assert np.array_equal(p.legal[:6,:,0],np.zeros((6,6),bool))
    base[0][1]=11
    q2,_=direct_priority_plan(ev,p,base)
    assert not q2.legal[0,0,0]


def test_all_static_policies_keep_dynamic_scores():
    ev=_evidence()
    score=np.zeros((7,6,2),np.float32)
    score[...,0]=.6
    head=SimpleNamespace(positive_weight=torch.tensor([[3.0,1.0],[2.8,1.0]]))
    for policy in POLICIES:
        x=static_policy_score(score,ev,head,policy)
        assert np.array_equal(x[6,:,0],score[6,:,0])
        assert not x[...,1].any()
        if policy=="static_corrected05":
            assert not x[:6,:,0].any()
        if policy=="static_direct_only":
            assert not x[[1,3],:,0].any()
            assert (x[[0,2],:,0]==.6).all()
        if policy=="static_halo_corrected05":
            assert not x[[1,3],:,0].any()
            assert (x[[0,2],:,0]==.6).all()


def test_paired_fp_diagnostics_include_halo_only_and_direct():
    ev=_evidence();plan=_plan()
    # enable one B direct road ADD at 1, and one dynamic at 4
    plan.legal[0,0,0]=True
    score=np.zeros((7,6,2),np.float32);score[0,0,0]=.9
    base=np.full(6,17,np.uint8)
    gt=np.array([17,17,17,17,4,17],np.uint8)
    b=np.array([17,11,17,17,4,17],np.uint8)
    old=np.array([17,17,17,17,4,17],np.uint8)
    counts=_make_counts()
    _analyze_unique_fp(counts,base,gt,b,old,b,plan,ev,score,0,11)
    assert counts["B_only_FP"]==1
    assert counts["B_only_FP_GT_free"]==1
    assert counts["B_only_FP_static_with_direct"]==1
    assert counts["B_only_FP_static_halo_only"]==0


def test_nonregression_gate_retains_B_when_a_class_drops():
    def item(road=50,moving=30,m=40):
        return dict(mIoU=m,per_horizon={
            h:dict(mIoU=m,MovingMicro=moving,
                   semantic_per_class={"11":road,"13":40.0})
            for h in ("1.0","2.0","3.0")})
    reference=item()
    higher=item(road=51,m=40.1)
    assert best_safe_variant({"frozen_B":reference,"static_direct_only":higher})=="static_direct_only"
    worse=item(road=49,m=41)
    assert best_safe_variant({"frozen_B":reference,"static_corrected05":worse})=="frozen_B"
