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
