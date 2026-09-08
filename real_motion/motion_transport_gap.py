"""Dependency-light helpers for P0-F9 v14 motion-transport gap diagnosis."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .metrics.moving_miou_v2 import SPEED_THRESHOLD_MPS


def horizon_seconds(num_horizons: int, frame_dt_s: float) -> np.ndarray:
    return (np.arange(int(num_horizons), dtype=np.float64) + 1.0) * float(frame_dt_s)


def true_moving_observation_mask(
    target_displacement_xy_m: np.ndarray,
    target_valid: np.ndarray,
    *,
    frame_dt_s: float = 0.5,
    speed_threshold_mps: float = SPEED_THRESHOLD_MPS,
) -> np.ndarray:
    """Return the exact center-motion decision used by Moving-mIoU v2."""
    disp = np.asarray(target_displacement_xy_m, dtype=np.float64)
    valid = np.asarray(target_valid, dtype=bool)
    if disp.ndim != 3 or disp.shape[-1] != 2 or valid.shape != disp.shape[:2]:
        raise ValueError("expected displacement [N,H,2] and valid [N,H]")
    dt = horizon_seconds(disp.shape[1], frame_dt_s)[None, :]
    speed = np.linalg.norm(disp, axis=-1) / dt
    return valid & (speed >= float(speed_threshold_mps))


def displacement_errors(
    pred_residual_xy_m: np.ndarray,
    target_residual_xy_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return KTA and learned displacement errors [N,H] in metres."""
    pred = np.asarray(pred_residual_xy_m, dtype=np.float64)
    target = np.asarray(target_residual_xy_m, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError("residual tensors must share shape [N,H,2]")
    return np.linalg.norm(target, axis=-1), np.linalg.norm(pred - target, axis=-1)


def learned_winner_mask(
    pred_residual_xy_m: np.ndarray,
    target_residual_xy_m: np.ndarray,
    target_valid: np.ndarray,
) -> np.ndarray:
    """GT diagnostic selector: use learned only when it is closer than KTA."""
    kta, learned = displacement_errors(pred_residual_xy_m, target_residual_xy_m)
    valid = np.asarray(target_valid, dtype=bool)
    if valid.shape != kta.shape:
        raise ValueError("valid shape mismatch")
    return valid & (learned < kta)


def interpolate_displacement(
    kta_displacement_xy_m: np.ndarray,
    target_displacement_xy_m: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate KTA toward GT displacement for center-error sensitivity."""
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    kta = np.asarray(kta_displacement_xy_m, dtype=np.float64)
    target = np.asarray(target_displacement_xy_m, dtype=np.float64)
    if kta.shape != target.shape:
        raise ValueError("displacement shape mismatch")
    return kta + a * (target - kta)


def masked_error_stats(kta_error: np.ndarray, learned_error: np.ndarray, mask: np.ndarray) -> dict:
    k = np.asarray(kta_error, dtype=np.float64)
    l = np.asarray(learned_error, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if k.shape != l.shape or k.shape != m.shape:
        raise ValueError("error/mask shape mismatch")
    n = int(m.sum())
    if n == 0:
        return {
            "count": 0,
            "kta_ade_m": float("nan"),
            "learned_ade_m": float("nan"),
            "relative_ade_reduction": float("nan"),
            "learned_win_fraction": float("nan"),
        }
    kv = k[m]
    lv = l[m]
    km = float(kv.mean())
    lm = float(lv.mean())
    return {
        "count": n,
        "kta_ade_m": km,
        "learned_ade_m": lm,
        "relative_ade_reduction": (km - lm) / km if km > 1e-12 else float("nan"),
        "learned_win_fraction": float((lv < kv).mean()),
    }


def latest_valid_fde_stats(
    kta_error: np.ndarray,
    learned_error: np.ndarray,
    valid_mask: np.ndarray,
    eligible_mask: np.ndarray | None = None,
) -> dict:
    """FDE at each source's latest eligible valid horizon."""
    k = np.asarray(kta_error, dtype=np.float64)
    l = np.asarray(learned_error, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if k.shape != l.shape or k.shape != valid.shape:
        raise ValueError("shape mismatch")
    eligible = valid if eligible_mask is None else (valid & np.asarray(eligible_mask, dtype=bool))
    ks, ls = [], []
    for i in range(k.shape[0]):
        ids = np.flatnonzero(eligible[i])
        if len(ids):
            h = int(ids[-1])
            ks.append(float(k[i, h]))
            ls.append(float(l[i, h]))
    if not ks:
        return {"count": 0, "kta_fde_m": float("nan"), "learned_fde_m": float("nan")}
    return {
        "count": len(ks),
        "kta_fde_m": float(np.mean(ks)),
        "learned_fde_m": float(np.mean(ls)),
    }


def summarize_rows(rows: list[Mapping]) -> dict:
    if not rows:
        return {
            "count": 0,
            "kta_ade_m": float("nan"),
            "learned_ade_m": float("nan"),
            "relative_ade_reduction": float("nan"),
            "learned_win_fraction": float("nan"),
        }
    k = np.asarray([float(r["kta_error_m"]) for r in rows], dtype=np.float64)
    l = np.asarray([float(r["learned_error_m"]) for r in rows], dtype=np.float64)
    km, lm = float(k.mean()), float(l.mean())
    return {
        "count": len(rows),
        "kta_ade_m": km,
        "learned_ade_m": lm,
        "relative_ade_reduction": (km - lm) / km if km > 1e-12 else float("nan"),
        "learned_win_fraction": float((l < k).mean()),
    }
