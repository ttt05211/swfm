"""Post-hoc motion-difficulty stratification; never used by model inference."""
import math
import numpy as np


def wrap_angle(x):
    return (float(x) + math.pi) % (2*math.pi) - math.pi


def interval_motion_features(record):
    c0=np.asarray(record["center0_world"],dtype=np.float64)
    ch=np.asarray(record["centerh_world"],dtype=np.float64)
    return {
        "displacement_m":float(np.linalg.norm(ch[:2]-c0[:2])),
        "heading_change_rad":abs(wrap_angle(record["yawh_world"]-record["yaw0_world"])),
        "speed_mps":float(record["speed_mps"]),
    }


def maneuver_bucket(delta_speed_mps, turn_rate_radps,
                    speed_change_threshold=1.0,
                    turn_rate_threshold_radps=math.radians(10.0)):
    dv=abs(float(delta_speed_mps)); rate=abs(float(turn_rate_radps))
    speed_hard=dv>=speed_change_threshold
    turn=rate>=turn_rate_threshold_radps
    if turn and speed_hard: return "turning+speed-change"
    if turn: return "turning"
    if speed_hard: return "accel/decel"
    return "uniform/easy"


def quantile_difficulty(values, q=(1/3,2/3)):
    """Return easy/medium/hard labels and the two frozen calibration cutoffs."""
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or len(x)==0: raise ValueError("values must be non-empty 1D")
    cuts=np.quantile(x,q)
    labels=np.where(x<=cuts[0],"easy",np.where(x<=cuts[1],"medium","hard"))
    return labels, tuple(float(v) for v in cuts)


def kta_difficulty(error_m, cuts):
    e=float(error_m); c1,c2=map(float,cuts)
    if not np.isfinite(e): return None
    if e<=c1: return "easy"
    if e<=c2: return "medium"
    return "hard"


def record_subset_labels(record, speed_change_threshold=1.0,
                         turn_rate_threshold_radps=math.radians(10.0),
                         kta_cuts=None):
    """Return analysis labels for one already Moving-mIoU-eligible GT instance."""
    labels=[]
    dv=float(record.get("delta_speed_mps",float("nan")))
    h=float(record.get("horizon_s",float("nan")))
    if not np.isfinite(h) or h<=0:
        h=float(record.get("dt_s",float("nan")))
    yaw0=float(record.get("yaw0_world",float("nan")))
    yawh=float(record.get("yawh_world",float("nan")))
    if np.isfinite(dv) and np.isfinite(h) and h>0 and np.isfinite(yaw0) and np.isfinite(yawh):
        rate=abs(wrap_angle(yawh-yaw0))/h
        labels.append(maneuver_bucket(dv,rate,speed_change_threshold,turn_rate_threshold_radps))
    if kta_cuts is not None:
        kd=kta_difficulty(record.get("kta_center_error_m",float("nan")),kta_cuts)
        if kd is not None:
            labels.append("kta-"+kd)
    return labels
