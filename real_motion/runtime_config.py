"""Runtime configuration loader for SWFM.

`configs/real_motion_occfm.yaml` is the single experiment source-of-truth.
CLI flags may explicitly override values, but scripts should not duplicate
method hyperparameters as independent Python defaults.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import json
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "real_motion_occfm.yaml"


def _deep_get(data: Mapping[str, Any], path: str, default=None):
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _deep_set(data: dict, path: str, value: Any):
    keys = path.split(".")
    cur = data
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def _parse_scalar(text: str):
    return yaml.safe_load(text)


def load_runtime_config(path: str | Path | None = None, overrides=()) -> dict:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError("runtime YAML root must be a mapping")
    cfg = deepcopy(cfg)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override must be KEY.PATH=value, got: {item}")
        key, value = item.split("=", 1)
        _deep_set(cfg, key.strip(), _parse_scalar(value.strip()))
    validate_runtime_config(cfg)
    cfg.setdefault("RUNTIME", {})
    cfg["RUNTIME"]["SOURCE_CONFIG"] = str(cfg_path.resolve())
    cfg["RUNTIME"]["OVERRIDES"] = list(overrides or ())
    return cfg


def get_cfg(cfg: Mapping[str, Any], path: str, default=None):
    return _deep_get(cfg, path, default)


def validate_runtime_config(cfg: Mapping[str, Any]):
    from .metrics.moving_miou_v2 import (
        PROTOCOL, SPEED_THRESHOLD_MPS, BOX_MARGIN_M,
        REPORT_HORIZONS_S, DYNAMIC_CLASS_IDS,
    )
    checks = {
        "METRIC.PROTOCOL": PROTOCOL,
        "METRIC.SPEED_THRESHOLD_MPS": SPEED_THRESHOLD_MPS,
        "METRIC.BOX_MARGIN_M": BOX_MARGIN_M,
        "METRIC.REPORT_HORIZONS_S": list(REPORT_HORIZONS_S),
        "METRIC.DYNAMIC_CLASS_IDS": list(DYNAMIC_CLASS_IDS),
        "METRIC.AGGREGATION": "per_horizon_miou_then_mean",
    }
    for path, expected in checks.items():
        actual = _deep_get(cfg, path)
        if isinstance(expected, float):
            ok = actual is not None and abs(float(actual) - expected) < 1e-12
        elif isinstance(expected, list):
            ok = list(actual or []) == expected
        else:
            ok = actual == expected
        if not ok:
            raise ValueError(f"Frozen metric contract mismatch at {path}: YAML={actual!r}, code={expected!r}")
    window = _deep_get(cfg, "MODEL.WINDOW_HW")
    if not isinstance(window, list) or len(window) != 2 or min(map(int, window)) <= 0:
        raise ValueError("MODEL.WINDOW_HW must be [H,W] positive integers")
    radii = list(_deep_get(cfg, "MOTION.KTA_TUBE_RADII", []))
    if len(radii) != 6:
        raise ValueError("MOTION.KTA_TUBE_RADII must contain six 0.5s radii")
    if any(int(r) < 0 for r in radii):
        raise ValueError("MOTION.KTA_TUBE_RADII cannot be negative")


def save_resolved_config(cfg: Mapping[str, Any], output: str | Path):
    path = Path(output)
    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        path = path / "resolved_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def add_config_args(parser):
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="runtime YAML source-of-truth")
    parser.add_argument("--override", action="append", default=[], help="explicit YAML override, e.g. MODEL.MAX_WINDOWS=10")
    return parser


def make_prepare_config(cfg):
    from .prepared import PrepareConfig
    from .motion import PersistenceMotionConfig
    from .kta import KTAConfig
    from .geometry import OccupancyGrid
    occ_range=list(get_cfg(cfg,'UPSTREAM.OCC_RANGE',[-40,-40,-1,40,40,5.4]))
    voxel=tuple(float(x) for x in get_cfg(cfg,'UPSTREAM.VOXEL_SIZE',[0.4,0.4,0.4]))
    xmin,ymin,zmin,xmax,ymax,zmax=map(float,occ_range)
    W=round((xmax-xmin)/voxel[0]);H=round((ymax-ymin)/voxel[1]);D=round((zmax-zmin)/voxel[2])
    grid=OccupancyGrid(x_min=xmin,y_min=ymin,z_min=zmin,voxel_size=voxel,shape_hwd=(H,W,D))
    motion=PersistenceMotionConfig(
        free_label=int(get_cfg(cfg,'TARGET.FREE_LABEL',17)),
        static_min_persistence=float(get_cfg(cfg,'MOTION.STATIC_MIN_PERSISTENCE',0.8)),
        moving_max_persistence=float(get_cfg(cfg,'MOTION.MOVING_MAX_PERSISTENCE',0.5)),
        min_observed_frames=2,
        history_dt_s=float(get_cfg(cfg,'MOTION.COMPONENT_HISTORY_DT_S',0.5)),
        voxel_size_xy_m=(voxel[0],voxel[1]),
        use_component_tracks=bool(get_cfg(cfg,'MOTION.USE_COMPONENT_TRACKS',True)),
        component_max_step_m=float(get_cfg(cfg,'MOTION.COMPONENT_MAX_STEP_M',4.0)),
        moving_speed_mps=float(get_cfg(cfg,'MOTION.COMPONENT_MOVING_SPEED_MPS',0.5)),
        static_speed_mps=float(get_cfg(cfg,'MOTION.COMPONENT_STATIC_SPEED_MPS',0.2)),
        min_track_frames=int(get_cfg(cfg,'MOTION.COMPONENT_MIN_TRACK_FRAMES',2)),
    )
    kta=KTAConfig(
        free_label=int(get_cfg(cfg,'TARGET.FREE_LABEL',17)),
        history_dt_s=float(get_cfg(cfg,'MOTION.KTA_HISTORY_DT_S',0.5)),
        max_match_distance_m=float(get_cfg(cfg,'MOTION.KTA_MAX_MATCH_DISTANCE_M',6.0)),
    )
    return PrepareConfig(
        history_frames=6,future_frames=6,
        frame_dt_s=float(get_cfg(cfg,'MOTION.COMPONENT_HISTORY_DT_S',0.5)),
        free_label=int(get_cfg(cfg,'TARGET.FREE_LABEL',17)),
        tube_radii=tuple(int(x) for x in get_cfg(cfg,'MOTION.KTA_TUBE_RADII',[1,2,3,4,5,6])),
        grid=grid,motion=motion,kta=kta,
    )
