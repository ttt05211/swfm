"""Runtime configuration loader for SWFM.

`configs/real_motion_occfm.yaml` is the single experiment source-of-truth.
CLI flags may explicitly override values, but scripts should not duplicate
method hyperparameters as independent Python defaults.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import hashlib
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


def _copy_paths(cfg: Mapping[str, Any], paths):
    out={}
    for path in paths:
        value=_deep_get(cfg,path,None)
        cur=out
        keys=path.split('.')
        for key in keys[:-1]: cur=cur.setdefault(key,{})
        cur[keys[-1]]=deepcopy(value)
    return out


def config_contract(cfg: Mapping[str, Any], kind="cache") -> dict:
    """Return the stable, path-independent subset that defines an experiment asset."""
    cache_paths=(
        "UPSTREAM.COMMIT","UPSTREAM.LATENT_HW","UPSTREAM.LATENT_CHANNELS",
        "UPSTREAM.OCC_RANGE","UPSTREAM.VOXEL_SIZE","UPSTREAM.WM_VARIANT",
        "UPSTREAM.WM_CONFIG","UPSTREAM.WM_CHECKPOINT_REL",
        "MODEL.WINDOW_HW","MODEL.MAX_WINDOWS","MODEL.MIN_WINDOW_COVERAGE",
        "MODEL.TRAJECTORY_LENGTH","MOTION","TARGET","EGO_PROTOCOL",
        "CACHE.VAE_LATENT_MODE","CACHE.PRECOMPUTE_WINDOW_PLAN",
        "CACHE.FILTER_EMPTY_GENERATION_SUPPORT","RUNTIME.VAE_AMP",
    )
    cache=_copy_paths(cfg,cache_paths)
    if kind=="cache": return cache
    if kind=="resume":
        return {
            "cache":cache,
            "model":deepcopy(_deep_get(cfg,"MODEL",{})),
            "optimization_full":deepcopy(_deep_get(cfg,"OPTIMIZATION.FULL",{})),
            "amp":deepcopy(_deep_get(cfg,"RUNTIME.AMP",{})),
            "tf32":deepcopy(_deep_get(cfg,"RUNTIME.TF32",None)),
        }
    raise ValueError(f"unknown config contract kind: {kind}")


def config_fingerprint(cfg: Mapping[str, Any], kind="cache") -> str:
    payload=json.dumps(config_contract(cfg,kind),sort_keys=True,separators=(",",":"),ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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

    # Observation-aware real-motion contract.  Semantic identity gates only
    # whether object-level displacement tracking is meaningful; MOVING still
    # requires measured historical displacement.
    if _deep_get(cfg, "MOTION.OBSERVATION_SOURCE") != "occ3d_mask_lidar":
        raise ValueError("MOTION.OBSERVATION_SOURCE must be occ3d_mask_lidar")
    eligible = tuple(int(x) for x in _deep_get(cfg, "MOTION.MOTION_ELIGIBLE_CLASS_IDS", []))
    if eligible != tuple(DYNAMIC_CLASS_IDS):
        raise ValueError(
            f"MOTION.MOTION_ELIGIBLE_CLASS_IDS must be {tuple(DYNAMIC_CLASS_IDS)}, got {eligible}"
        )
    min_static_obs = int(_deep_get(cfg, "MOTION.MIN_STATIC_OBSERVATIONS", -1))
    if not 1 <= min_static_obs <= 6:
        raise ValueError("MOTION.MIN_STATIC_OBSERVATIONS must be in [1,6]")
    static_p = float(_deep_get(cfg, "MOTION.STATIC_MIN_PERSISTENCE", -1.0))
    if not 0.0 <= static_p <= 1.0:
        raise ValueError("MOTION.STATIC_MIN_PERSISTENCE must be in [0,1]")

    # Main-paper ego protocol is intentionally frozen to the official OccFM-fut
    # model released as epoch=000196.ckpt. The official dataset feeds one-step
    # GT ego offsets for every frame in the 6-history + 6-future window, yielding
    # [12,2], while HIST_LAST=4 zeros the first 2 history trajectory entries.
    expected_ego={
        "NAME":"occfm_fut_12step_v1",
        "FUTURE_POSE_SOURCE":"gt_future_ego",
        "TRAJECTORY_CONDITION_SOURCE":"official_temporal_info_first_step_per_frame",
        "TRAJECTORY_LENGTH":12,
        "HIST_LAST":4,
        "ZERO_PREFIX_STEPS":2,
        "REQUIRE_TEMPORAL_INFO":True,
        "BASELINE_INFORMATION_MATCH":"required",
        "UPSTREAM_INIT_VARIANT":"fut_traj_196",
    }
    ego=_deep_get(cfg,"EGO_PROTOCOL",{}) or {}
    for key,expected in expected_ego.items():
        if ego.get(key)!=expected:
            raise ValueError(f"Frozen OccFM-fut ego contract mismatch at EGO_PROTOCOL.{key}: {ego.get(key)!r} != {expected!r}")
    if int(_deep_get(cfg,"MODEL.TRAJECTORY_LENGTH",-1)) != 12:
        raise ValueError("MODEL.TRAJECTORY_LENGTH must be 12 for OccFM-fut epoch=000196.ckpt")
    if int(ego["ZERO_PREFIX_STEPS"]) != 6-int(ego["HIST_LAST"]):
        raise ValueError("OccFM HIST_LAST contract requires ZERO_PREFIX_STEPS = 6 - HIST_LAST")
    if _deep_get(cfg,"UPSTREAM.WM_VARIANT")!="occfm_fut":
        raise ValueError("UPSTREAM.WM_VARIANT must be occfm_fut")
    if _deep_get(cfg,"UPSTREAM.WM_CONFIG")!="tools/cfgs/occfm_fut.yaml":
        raise ValueError("UPSTREAM.WM_CONFIG must be tools/cfgs/occfm_fut.yaml")


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
        min_static_observations=int(get_cfg(cfg,'MOTION.MIN_STATIC_OBSERVATIONS',3)),
        motion_eligible_class_ids=tuple(int(x) for x in get_cfg(cfg,'MOTION.MOTION_ELIGIBLE_CLASS_IDS',[2,3,4,5,6,7,9,10])),
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
        trajectory_length=int(get_cfg(cfg,'EGO_PROTOCOL.TRAJECTORY_LENGTH',12)),
        trajectory_hist_last=int(get_cfg(cfg,'EGO_PROTOCOL.HIST_LAST',4)),
        trajectory_zero_prefix=int(get_cfg(cfg,'EGO_PROTOCOL.ZERO_PREFIX_STEPS',2)),
        trajectory_protocol=str(get_cfg(cfg,'EGO_PROTOCOL.NAME','occfm_fut_12step_v1')),
        require_temporal_info=bool(get_cfg(cfg,'EGO_PROTOCOL.REQUIRE_TEMPORAL_INFO',True)),
        grid=grid,motion=motion,kta=kta,
    )
