"""Frozen-weight, static-only Point CCR policies for a single paired screen.

Every policy changes only ADD decisions for actor==-2 (road/sidewalk).
Dynamic actor rows, checkpoint, model, V18 baseline and REMOVE-off contract
are immutable. This is a read-only candidate study, not the frozen deployment.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

STATIC_CLASSES=(11,13)
POLICIES=(
    "frozen_B",
    "static_corrected05",
    "static_direct_only",
    "static_halo_corrected05",
    "static_direct_priority",
    "static_halo_corrected05_direct_priority",
)


def direct_priority_plan(evidence,plan,baseline):
    """Resolve ONLY asymmetric direct-history vs opposite pure-halo conflicts.

    A class has direct evidence at a destination when ANY of its projected
    canonical rows has presence in a historical frame. When the other class
    has only halo there, restore ADD legality for the direct class only.

    Both-direct and both-halo conflicts remain disabled. All dynamic rows,
    REMOVE legality, owner layers and occupied V18 targets stay unchanged.
    """
    flat=np.asarray(plan.flat)
    old_legal=np.asarray(plan.legal)
    if flat.ndim!=2 or flat.shape[1]!=6 or old_legal.shape!=(len(flat),6,2):
        raise ValueError("invalid point CCR plan shape")
    presence=np.asarray(evidence.presence,bool)
    if presence.shape!=(len(flat),4):
        raise ValueError("static policy needs exact four-frame presence")
    actors=np.asarray(evidence.actor)
    cls=np.asarray(evidence.classes)
    direct=presence.any(1)
    static=actors==-2
    new_legal=old_legal.copy()
    counts={"road_direct_over_sidewalk_halo":0,"sidewalk_direct_over_road_halo":0}
    for h in range(6):
        base=np.asarray(baseline[h]).ravel()
        volume=len(base)
        destination=flat[:,h]
        valid=(destination>=0)&(destination<volume)
        if np.any((destination>=volume)&(destination>=0)):
            raise RuntimeError("out-of-bounds canonical projection")
        has_direct={}
        has_any={}
        for cid in STATIC_CLASSES:
            any_row=static&(cls==cid)&valid
            direct_row=any_row&direct
            dense=np.zeros(volume,bool)
            dense[destination[direct_row]]=True
            has_direct[cid]=dense
            dense=np.zeros(volume,bool)
            dense[destination[any_row]]=True
            has_any[cid]=dense
        for cid,other,name in (
            (11,13,"road_direct_over_sidewalk_halo"),
            (13,11,"sidewalk_direct_over_road_halo"),
        ):
            # Class cid DIRECT, other has candidates but none are DIRECT.
            # This is GT-independent: no future label enters the rule.
            can=(has_direct[cid]&has_any[other]&~has_direct[other]
                 &(base==17))
            chosen=static&(cls==cid)&direct&valid
            chosen[valid]&=can[destination[valid]]
            if np.any(chosen&old_legal[:,h,0]):
                raise RuntimeError("expected direct-halo collisions to be disabled by frozen planner")
            new_legal[chosen,h,0]=True
            counts[name]+=int(np.count_nonzero(can))
    dyn=actors>=0
    if not np.array_equal(new_legal[dyn],old_legal[dyn]):
        raise RuntimeError("static-only policy changed dynamic plan legality")
    if not np.array_equal(new_legal[...,1],old_legal[...,1]):
        raise RuntimeError("static-only policy changed REMOVE legality")
    if np.any(new_legal[static,:,0] & (np.asarray(plan.base)[static]!=17)):
        raise RuntimeError("static-only policy may not overwrite V18 occupied")
    return replace(plan,legal=new_legal),counts


def static_policy_score(score,evidence,head,policy):
    """Return ADD scores with unchanged dynamic rows; REMOVE always disabled.

    static_corrected05 corresponds to raw weighted sigmoid >= w/(1+w).
    This is a fixed analytic policy, not a DEV-tuned threshold.
    """
    if policy not in POLICIES:
        raise ValueError("unknown static policy")
    score=np.asarray(score,np.float32)
    if score.shape!=(len(evidence),6,2):
        raise ValueError("invalid frozen Point CCR scores")
    result=score.copy()
    result[...,1]=0.0
    static=np.asarray(evidence.actor)==-2
    if np.any((np.asarray(evidence.actor)<0)&~static):
        raise RuntimeError("unexpected static actor code")
    direct=np.asarray(evidence.presence,bool).any(axis=1)
    if "direct_only" in policy:
        result[static&~direct,:,0]=0.0
    weight=float(head.positive_weight[0,0].detach().cpu())
    if not np.isfinite(weight) or weight<1:
        raise RuntimeError("invalid frozen static positive weight")
    if "corrected05" in policy:
        strict=weight/(1.0+weight)
        mask=static if policy=="static_corrected05" else (static&~direct)
        selected=result[mask,:,0]>=strict
        result[mask,:,0]=selected.astype(np.float32)
    if not np.array_equal(result[~static,:,0],score[~static,:,0]):
        raise RuntimeError("static policy modified dynamic logits")
    return result


def best_safe_variant(metrics):
    """Strict descriptive DEV screen; not independently validated selection.

    Candidate must be non-inferior to B at 1/2/3s in overall mIoU,
    MovingMicro, and both road/sidewalk class IoU. Otherwise retain B.
    """
    b=metrics["frozen_B"]
    eligible=[]
    for key,m in metrics.items():
        if key=="frozen_B":continue
        ok=True
        for h in ("1.0","2.0","3.0"):
            p=m["per_horizon"][h];ref=b["per_horizon"][h]
            keys=("mIoU","MovingMicro")
            if any(not np.isfinite(p[k]) or p[k]<ref[k]-1e-9 for k in keys):
                ok=False;break
            for cls in ("11","13"):
                x=p["semantic_per_class"][cls];y=ref["semantic_per_class"][cls]
                if not np.isfinite(x) or x<y-1e-9:ok=False;break
            if not ok:break
        if ok:eligible.append(key)
    return max(eligible,key=lambda k:metrics[k]["mIoU"]) if eligible else "frozen_B"
