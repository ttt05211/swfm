import numpy as np
from types import SimpleNamespace

from real_motion.causal_column_completion import GENERATE, REFINE
from tools.real_motion.diagnose_p0_f9_ccr_static_surface_gap import (
    _old_class_support, _flatten_support, _static_diagnostic, _finish, _make_counts,
    _static_surface_conflicts, _count_surface_conflicts,
)


def test_old_static_support_separates_gen_refine_from_dynamic():
    plan=SimpleNamespace(
        actor=np.array([-3,-2,0],np.int32),
        kind=np.array([GENERATE,REFINE,REFINE],np.uint8),
        classes=np.array([11,13,11],np.uint8),
        flat=np.array([[1,2],[3,4],[5,6]],np.int64),
        legal=np.ones((3,2,3),bool),
    )
    g=_old_class_support(plan,11,GENERATE,8)
    r=_old_class_support(plan,13,REFINE,8)
    assert np.flatnonzero(g).tolist()==[1,2]
    assert np.flatnonzero(r).tolist()==[3,4]
    assert not g[5] and not g[6]
    assert not r[5] and not r[6]


def test_static_gap_attribution_has_unique_dense_voxel_counts():
    # Baseline is free; GT road at 0, 1, 2. CCR can only propose 0 and 1.
    # Old static succeeds at 2 (GEN-only); CCR misses 1 with legal support.
    base=np.array([17,17,17,17,17,17],np.uint8)
    gt=np.array([11,11,11,17,13,17],np.uint8)
    b=np.array([11,17,17,11,17,17],np.uint8)
    bs=b.copy()
    old=np.array([11,11,11,17,17,17],np.uint8)
    os=old.copy()
    ccr=np.array([1,1,0,1,0,0],bool)
    gen=np.array([0,0,1,0,0,0],bool)
    refine=np.array([1,1,0,0,0,0],bool)
    raw=_make_counts()
    _static_diagnostic(raw,base,gt,b,bs,old,os,ccr,gen,refine,11)
    x=_finish(raw)
    assert x["GT_missing_on_V18_free"]==3
    assert x["CCR_support_GT"]==2
    assert x["Old_only_support_GT"]==1
    assert x["Old_static_win_over_B_joint"]==2
    assert x["Old_static_win_CCR_no_support"]==1
    assert x["Old_static_win_CCR_reachable_not_written"]==1
    assert x["B_static_FP_free"]==1
    assert x["B_static_FP_opposite_surface"]==0
    assert abs(x["GT_missing_CCR_support_recall"]-2/3)<1e-12


def test_flatten_support_ignores_illegal_entries():
    legal=np.array([[True,False],[False,True]])
    ids=np.array([[0,1],[2,3]])
    cls=np.array([True,True])
    mask=_flatten_support(ids,legal,cls,5)
    assert np.flatnonzero(mask).tolist()==[0,3]


def test_static_conflict_masks_partition_direct_halo_evidence():
    # Four exclusive conflict patterns on V18-free target voxels.
    ev=SimpleNamespace(
        actor=np.full(8,-2,np.int32),
        classes=np.array([11,13,13,11,11,13,11,13],np.uint8),
        presence=np.array([[1],[0],[1],[0],[1],[1],[0],[0]],bool))
    flat=np.full((8,6),-1,np.int64)
    flat[:,0]=[1,1,2,2,3,3,4,4]
    plan=SimpleNamespace(
        flat=flat,
        legal=np.zeros((8,6,2),bool),
    )
    base=np.full(6,17,np.uint8)
    masks=_static_surface_conflicts(ev,plan,0,base)
    assert np.flatnonzero(masks["road_direct_sidewalk_pure_halo"]).tolist()==[1]
    assert np.flatnonzero(masks["sidewalk_direct_road_pure_halo"]).tolist()==[2]
    assert np.flatnonzero(masks["both_direct"]).tolist()==[3]
    assert np.flatnonzero(masks["both_pure_halo"]).tolist()==[4]
    assert np.flatnonzero(masks["any"]).tolist()==[1,2,3,4]
    # Keep the frozen actual predictions unchanged; diagnose where Old
    # wins rather than claiming all blocked targets could be recovered.
    gt=np.array([17,11,13,11,13,17],np.uint8)
    b=base.copy()
    old=base.copy();old[1]=11
    counts=_make_counts()
    _count_surface_conflicts(counts,base,gt,b,old,11,masks)
    assert counts["own_real_vs_other_halo_GT"]==1
    assert counts["Old_static_win_own_real_vs_other_halo"]==1
    assert counts["blocked_any_GT"]==2
    assert counts["Old_static_win_blocked_any"]==1


def test_static_conflict_audit_detects_native_planner_mismatch():
    ev=SimpleNamespace(
        actor=np.array([-2,-2],np.int32),
        classes=np.array([11,13],np.uint8),
        presence=np.array([[1],[0]],bool))
    flat=np.full((2,6),-1,np.int64);flat[:,0]=[1,1]
    legal=np.zeros((2,6,2),bool);legal[0,0,0]=True
    plan=SimpleNamespace(flat=flat,legal=legal)
    try:
        _static_surface_conflicts(ev,plan,0,np.full(3,17,np.uint8))
        assert False,"must fail closed when native planner disagrees"
    except RuntimeError as exc:
        assert "planner legality disagree" in str(exc)
