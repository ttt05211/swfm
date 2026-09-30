"""Diagnostic-only interventions on the SAME frozen source-centred SE(2).

No training, annotation-based source creation, shape change or new inference
gate. GT validity controls ONLY explicitly labelled offline interventions.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS

PROTOCOL = "p0_f9_v18_motion_gap_audit_v1"
VARIANTS = ("V18_BASE", "KTA_XY_ZERO_YAW", "GT_XY_PRED_YAW", "PRED_XY_GT_YAW",
            "GT_XY_GT_YAW", "GT_XY_GT_YAW_SUPERVISED_ONLY")


def numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
        if str(value.dtype) == "torch.bfloat16": value = value.float()
        return value.cpu().numpy()
    return np.asarray(value)


def motion_states(record, outputs):
    """Return [N,6,2] absolute t0-ego source centres + effective yaw [N,6]."""
    fields = ("anchors_xy_t0_m", "kta_displacement_xy_m", "source_centroid_xy_t0_m",
              "target_source_displacement_xy_m", "target_source_residual_xy_m", "target_yaw_rad",
              "se2_target_valid", "yaw_label_valid", "source_class_id", "supervised_source", "yaw_enabled")
    r = {k: numpy(record[k]) for k in fields}
    n = len(r["source_class_id"])
    for k in ("anchors_xy_t0_m", "kta_displacement_xy_m", "target_source_displacement_xy_m", "target_source_residual_xy_m"):
        if r[k].shape != (n, 6, 2): raise ValueError(f"{k}: expected [N,6,2]")
    for k in ("target_yaw_rad", "se2_target_valid", "yaw_label_valid"):
        if r[k].shape != (n, 6): raise ValueError(f"{k}: expected [N,6]")
    if r["source_centroid_xy_t0_m"].shape != (n, 2) or any(r[k].shape != (n,) for k in ("source_class_id", "supervised_source", "yaw_enabled")):
        raise ValueError("source identity tensor shape mismatch")
    residual, yaw = numpy(outputs["residual_xy_m"]), numpy(outputs["yaw_delta_rad"])
    if residual.shape != (n, 6, 2) or yaw.shape != (n, 6): raise ValueError("model output shape mismatch")
    valid = r["se2_target_valid"].astype(bool)
    enabled = np.isin(r["source_class_id"], YAW_ENABLED_CLASS_IDS)
    if not np.array_equal(enabled, r["yaw_enabled"].astype(bool)):
        raise ValueError("cached yaw class rule differs from frozen deployment")
    if not np.allclose(r["anchors_xy_t0_m"] - r["kta_displacement_xy_m"],
                       r["source_centroid_xy_t0_m"][:, None], rtol=0, atol=2e-4):
        raise ValueError("KTA anchors do not share the cached source-centre pivot")
    if not np.allclose((r["kta_displacement_xy_m"] + r["target_source_residual_xy_m"])[valid],
                       r["target_source_displacement_xy_m"][valid], rtol=0, atol=2e-4):
        raise ValueError("source displacement/residual target contract mismatch")
    pc = r["anchors_xy_t0_m"] + residual
    gc = r["anchors_xy_t0_m"] + r["target_source_residual_xy_m"]
    py = np.where(enabled[:, None], yaw, 0.)
    yaw_valid = valid & r["yaw_label_valid"].astype(bool) & enabled[:, None]
    gy = np.where(yaw_valid, r["target_yaw_rad"], py)
    mask = valid & r["supervised_source"].astype(bool)[:, None]
    states = {"V18_BASE": (pc, py), "KTA_XY_ZERO_YAW": (r["anchors_xy_t0_m"].copy(), np.zeros_like(py)),
        "GT_XY_PRED_YAW": (np.where(valid[..., None], gc, pc), py.copy()),
        "PRED_XY_GT_YAW": (pc.copy(), gy), "GT_XY_GT_YAW": (np.where(valid[..., None], gc, pc), gy.copy()),
        "GT_XY_GT_YAW_SUPERVISED_ONLY": (np.where(mask[..., None], gc, pc), np.where(mask, gy, py))}
    if any(not np.isfinite(a).all() for pair in states.values() for a in pair):
        raise ValueError("nonfinite motion intervention")
    return states


def interaction_decomposition(reports, metric="mIoU"):
    """2x2 counterfactual interaction, NOT three additive independent gains."""
    b = reports["V18_BASE"]["metrics"][metric]
    xy = reports["GT_XY_PRED_YAW"]["metrics"][metric] - b
    yaw = reports["PRED_XY_GT_YAW"]["metrics"][metric] - b
    both = reports["GT_XY_GT_YAW"]["metrics"][metric] - b
    supervised = reports["GT_XY_GT_YAW_SUPERVISED_ONLY"]["metrics"][metric] - b
    return {"xy_only_pp": xy, "yaw_only_pp": yaw, "both_pp": both,
            "interaction_pp": both - xy - yaw,
            "xy_shapley_diagnostic_pp": .5 * (xy + both - yaw),
            "yaw_shapley_diagnostic_pp": .5 * (yaw + both - xy),
            "supervised_only_pp": supervised, "additional_unsupervised_joint_pp": both - supervised}


class MotionErrors:
    """Descriptive source errors, with class/history/supervision strata.

    Derivatives require consecutive valid states; missing GT is never treated
    as a zero displacement. No assumption that GT motion is smooth/constant.
    """
    def __init__(self): self.values = defaultdict(lambda: defaultdict(list))

    def update(self, record, outputs):
        states = motion_states(record, outputs)
        fm = numpy(record["frame_motion_features"])
        n = len(numpy(record["source_class_id"]))
        if fm.shape != (n, 6, 5): raise ValueError("frame motion feature shape mismatch")
        valid = numpy(record["se2_target_valid"]).astype(bool)
        target = numpy(record["target_source_displacement_xy_m"])
        center = numpy(record["source_centroid_xy_t0_m"])
        pred = states["V18_BASE"][0] - center[:, None]
        kta = states["KTA_XY_ZERO_YAW"][0] - center[:, None]
        yaw_valid = valid & numpy(record["yaw_label_valid"]).astype(bool) & numpy(record["yaw_enabled"]).astype(bool)[:, None]
        py, gy = states["V18_BASE"][1], numpy(record["target_yaw_rad"])
        wrap = lambda x: np.arctan2(np.sin(x), np.cos(x))
        groups = {"all": np.ones(n, bool)}
        cid = numpy(record["source_class_id"])
        history = (fm[..., 4] > .5).sum(1)
        sup = numpy(record["supervised_source"]).astype(bool)
        for c in np.unique(cid): groups[f"class/{int(c)}"] = cid == c
        for age in np.unique(history): groups[f"history_valid/{int(age)}"] = history == age
        groups["supervised/yes"], groups["supervised/no"] = sup, ~sup
        turn = np.rad2deg(np.abs(wrap(gy[:, -1])))
        last_yaw_valid = yaw_valid[:, -1]
        groups["turn/straight_lt5deg"] = last_yaw_valid & (turn < 5)
        groups["turn/mild_5to15deg"] = last_yaw_valid & (turn >= 5) & (turn < 15)
        groups["turn/strong_ge15deg"] = last_yaw_valid & (turn >= 15)

        def add(name, values, mask):
            for group, source_mask in groups.items():
                selected = values[mask & source_mask[:, None]]
                self.values[group][name].extend(selected.tolist())
        error = np.linalg.norm(pred - target, axis=-1)
        add("source_center_error_m", error, valid)
        add("kta_source_center_error_m", np.linalg.norm(kta - target, axis=-1), valid)
        add("yaw_error_deg", np.rad2deg(np.abs(wrap(py - gy))), yaw_valid)
        add("zero_yaw_error_deg", np.rad2deg(np.abs(wrap(gy))), yaw_valid)
        for h in (1, 3, 5):
            mask = np.zeros_like(valid); mask[:, h] = valid[:, h]
            add(f"source_center_error/{.5*(h+1):.1f}s_m", error, mask)
        present = np.column_stack((np.ones(n, bool), valid))
        edge = present[:, 1:] & present[:, :-1]
        vp = np.diff(np.concatenate((np.zeros((n, 1, 2)), pred), axis=1), axis=1) / .5
        vg = np.diff(np.concatenate((np.zeros((n, 1, 2)), target), axis=1), axis=1) / .5
        add("velocity_error_mps", np.linalg.norm(vp - vg, axis=-1), edge)
        acceleration_valid = edge[:, 1:] & edge[:, :-1]
        ap, ag = np.diff(vp, axis=1)/.5, np.diff(vg, axis=1)/.5
        add("acceleration_error_mps2", np.linalg.norm(ap-ag, axis=-1), acceleration_valid)
        add("pred_acceleration_mps2", np.linalg.norm(ap, axis=-1), acceleration_valid)
        add("gt_acceleration_mps2", np.linalg.norm(ag, axis=-1), acceleration_valid)

    def compute(self):
        def summarize(values):
            x = np.asarray(values, np.float64)
            if not len(x): return {"count": 0, "mean": None, "median": None, "p90": None}
            return {"count": len(x), "mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.quantile(x, .9))}
        return {group: {key: summarize(v) for key, v in sorted(rows.items())} for group, rows in sorted(self.values.items())}
