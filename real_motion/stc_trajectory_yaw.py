"""Fixed causal trajectory-tangent yaw probe; no occupancy or future truth inputs."""
import math
import numpy as np
from real_motion.stc_camera_protocol import validate_pose

PROTOCOL = 'p0_f9_frozen_planner_tangent_yaw_dev64_v1'
RULES = dict(history_frames=4, future_frames=6, future_dt_s=.5,
    minimum_speed_mps=1., maximum_speed_mps=40., maximum_history_fit_rmse_m=.2,
    maximum_sideslip_deg=10., maximum_acceleration_mps2=8.,
    maximum_turn_rate_deg_s=45., maximum_correction_deg=15.,
    derivative='local_three_point_quadratic_at_waypoint',
    calibration='four_measured_history_quadratic_velocity_vs_t0_body_heading',
    rotation='left_world_Rz_preserving_translation_and_yaw_free_body_tilt')


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def yaw(p):
    return math.atan2(float(p[1, 0]), float(p[0, 0]))


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def tangent_yaw(history_poses, history_times_s, planned_poses):
    """Replace reliable planned yaw only. Times are measured HISTORY seconds.

    Future slots remain nominal .5s; no actual future pose/time is accepted.
    All gates are fixed before scoring. A gate failure returns exact originals.
    """
    if len(history_poses) != 4 or len(planned_poses) != 6:
        raise ValueError('yaw probe requires exactly four histories and six planned poses')
    h = np.stack([validate_pose(p) for p in history_poses])
    p = np.stack([validate_pose(v) for v in planned_poses])
    t = np.asarray(history_times_s, dtype=np.float64)
    if t.shape != (4,) or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError('finite strictly increasing measured historical seconds required')
    t = t - t[-1]
    if not ((np.diff(t) >= .25) & (np.diff(t) <= .75)).all():
        raise ValueError('historical clock outside nominal 2Hz guard')
    result = p.copy()
    audit = dict(corrected=0, history_rejection=None, fallback_reasons={},
                 history_speed_mps=None, history_fit_rmse_m=None, sideslip_deg=None,
                 correction_deg=[0.] * 6, original_tangent_gap_deg=[None] * 6)
    def reject(reason):
        audit['history_rejection'] = reason
        audit['fallback_reasons'][reason] = 6
        return result, audit
    def fallback(reason):
        audit['fallback_reasons'][reason] = audit['fallback_reasons'].get(reason, 0) + 1
    # Subtract t0 position before fitting: stable at large global coordinates.
    xy = h[:, :2, 3] - h[-1, :2, 3]
    design = np.column_stack((np.ones(4), t, t*t))
    coef = np.linalg.lstsq(design, xy, rcond=None)[0]
    velocity = coef[1]; speed = float(np.linalg.norm(velocity))
    rmse = float(np.sqrt(np.mean(np.sum((design @ coef - xy)**2, axis=1))))
    slip = float(wrap(yaw(h[-1]) - math.atan2(velocity[1], velocity[0])))
    audit.update(history_speed_mps=speed, history_fit_rmse_m=rmse, sideslip_deg=math.degrees(slip))
    if not RULES['minimum_speed_mps'] <= speed <= RULES['maximum_speed_mps']:
        return reject('history_speed')
    if rmse > RULES['maximum_history_fit_rmse_m']: return reject('history_fit')
    if abs(math.degrees(slip)) > RULES['maximum_sideslip_deg']: return reject('reverse_or_sideslip')
    if np.linalg.norm(2*coef[2]) > RULES['maximum_acceleration_mps2']: return reject('history_acceleration')
    path = np.vstack((h[-1, :2, 3], p[:, :2, 3])) - h[-1, :2, 3]
    velocities = []
    accelerations = []
    for i in range(1, 7):
        indices = np.arange(i-1, i+2) if i < 6 else np.arange(4, 7)
        local_t = (indices-i)*RULES['future_dt_s']
        c = np.linalg.solve(np.column_stack((np.ones(3), local_t, local_t**2)), path[indices])
        velocities.append(c[1]); accelerations.append(2*c[2])
    velocities = np.asarray(velocities)
    speeds = np.linalg.norm(velocities, axis=1)
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    heading_steps = wrap(np.diff(np.r_[yaw(h[-1])-slip, headings]))
    for i in range(6):
        target = float(wrap(headings[i] + slip))
        delta = float(wrap(target-yaw(p[i])))
        audit['original_tangent_gap_deg'][i] = math.degrees(delta)
        if not RULES['minimum_speed_mps'] <= speeds[i] <= RULES['maximum_speed_mps']:
            fallback('planned_speed'); continue
        if np.linalg.norm(accelerations[i]) > RULES['maximum_acceleration_mps2']:
            fallback('planned_acceleration'); continue
        if abs(math.degrees(heading_steps[i]))/.5 > RULES['maximum_turn_rate_deg_s']:
            fallback('planned_turn'); continue
        if abs(math.degrees(delta)) > RULES['maximum_correction_deg']:
            fallback('large_correction'); continue
        if abs(delta) <= 1e-12:
            fallback('already_consistent'); continue
        result[i, :3, :3] = rz(delta) @ p[i, :3, :3]
        validate_pose(result[i])
        audit['correction_deg'][i] = math.degrees(delta)
        audit['corrected'] += 1
    # Preserve planner XYZ and homogeneous row byte-for-byte, not approximately.
    if not np.array_equal(result[:, :, 3], p[:, :, 3]) or not np.array_equal(result[:, 3], p[:, 3]):
        raise RuntimeError('yaw probe changed planner translation')
    return result, audit


def yaw_error_deg(predicted, actual):
    """Scoring only. Never called by tangent_yaw."""
    if len(predicted) != 6 or len(actual) != 6: raise ValueError('six audit poses required')
    return [abs(math.degrees(float(wrap(yaw(p)-yaw(g))))) for p, g in zip(predicted, actual)]
