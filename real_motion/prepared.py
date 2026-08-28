"""Raw occupancy-window preparation: geometry -> motion -> SE(3) -> KTA -> GT targets."""
from dataclasses import dataclass
from pathlib import Path
import json
import torch
import numpy as np

from .geometry import (
    OccupancyGrid, ego_compensate_sequence, transport_current_to_future,
    transport_mask_to_future, relative_transform, warp_semantic_grid, warp_mask,
)
from .motion import PersistenceMotionConfig, decompose_masks, split_semantics, MotionMasks
from .kta import KTAConfig, causal_kta
from .support import MotionTubeConfig, build_motion_tube
from .swept_support import swept_support_in_future_ego
from .nuscenes_adapter import (gt_moving_support_for_horizon, gt_moving_only_semantics,
                                causal_dynamic_target_semantics)

PREPARED_VERSION = "real_motion_prepared_v6_hybrid_balanced_r1"


@dataclass(frozen=True)
class PrepareConfig:
    history_frames: int = 6
    future_frames: int = 6
    frame_dt_s: float = 0.5
    free_label: int = 17
    support_geometry: str = "hybrid_endpoint_swept_v1"
    endpoint_tube_radii: tuple = (1, 2, 3, 4, 4, 5)
    swept_tube_radii: tuple = (1, 1, 1, 1, 1, 1)
    uncertain_tube_radii: tuple = (0, 0, 0, 1, 2, 3)
    trajectory_length: int = 12
    trajectory_hist_last: int = 4
    trajectory_zero_prefix: int = 2
    trajectory_protocol: str = "occfm_fut_12step_v1"
    require_temporal_info: bool = True
    grid: OccupancyGrid = OccupancyGrid()
    motion: PersistenceMotionConfig = PersistenceMotionConfig()
    kta: KTAConfig = KTAConfig()

    @property
    def tube_radii(self):
        """Backward-compatible alias for endpoint radii used by old audit code."""
        return self.endpoint_tube_radii


def _masked_semantics(sem, mask, free_label):
    out = np.full_like(sem, free_label)
    out[np.asarray(mask, dtype=bool)] = np.asarray(sem)[np.asarray(mask, dtype=bool)]
    return out


def _early_frame_masks(sem, cfg: PersistenceMotionConfig):
    """Class-route a prefix that is too short for displacement estimation.

    Background can be deterministically transported immediately. Motion-eligible
    things remain UNCERTAIN until enough history exists to prove displacement.
    """
    sem = np.asarray(sem)
    occupied = sem != cfg.free_label
    eligible = occupied & np.isin(
        sem, np.asarray(tuple(int(c) for c in cfg.motion_eligible_class_ids))
    )
    static = occupied & ~eligible
    moving = np.zeros_like(occupied)
    uncertain = eligible
    return MotionMasks(
        static, moving, uncertain, ~occupied,
        np.zeros_like(sem, dtype=np.float32),
    )


def _align_observation_sequence(observed, poses, reference_index, grid):
    """Rigidly align boolean observation masks with the same ego transform as semantics."""
    ref = poses[reference_index]
    return np.stack([
        warp_mask(mask, relative_transform(pose, ref), grid=grid)
        for mask, pose in zip(observed, poses)
    ], axis=0)


def causal_history_split(history_native, history_poses, history_observed, cfg: PrepareConfig):
    """Causally split every history frame in its own ego coordinates."""
    static_frames, moving_frames, candidate_support = [], [], []
    for j in range(len(history_native)):
        if j + 1 < cfg.motion.min_observed_frames:
            masks = _early_frame_masks(history_native[j], cfg.motion)
        else:
            prefix_sem = history_native[:j+1]
            prefix_obs = history_observed[:j+1]
            prefix_pose = history_poses[:j+1]
            aligned = ego_compensate_sequence(prefix_sem, prefix_pose, -1, cfg.grid, cfg.free_label)
            aligned_obs = _align_observation_sequence(prefix_obs, prefix_pose, -1, cfg.grid)
            masks = decompose_masks(
                aligned, cfg.motion, history_observed=aligned_obs
            )
        sta, candidate = split_semantics(history_native[j], masks, cfg.free_label)
        static_frames.append(sta)
        # Legacy cache key name is moving_history_occ, but the formal meaning is
        # generator-visible history = MOVING union UNCERTAIN eligible things.
        moving_frames.append(candidate)
        candidate_support.append(masks.wm_candidate.any(axis=2))
    return np.stack(static_frames), np.stack(moving_frames), np.stack(candidate_support)


def _load_history_semantics_and_observation(source, scene_name, tokens, free_label):
    semantics, observed = [], []
    for token in tokens:
        if hasattr(source, "load_occ3d"):
            sem, obs = source.load_occ3d(scene_name, token, require_lidar_mask=True)
        else:
            # Unit-test/custom-source compatibility only. Formal NuScenesWindowSource
            # always exposes load_occ3d and requires Occ3D mask_lidar.
            sem = source.load_semantics(scene_name, token)
            obs = np.asarray(sem) != free_label
        semantics.append(np.asarray(sem))
        observed.append(np.asarray(obs, dtype=bool))
    return np.stack(semantics), np.stack(observed)


def load_nuscenes_window_raw(source, window, cfg: PrepareConfig = PrepareConfig(), include_gt=True):
    """Load raw arrays/poses once; trajectory matches official OccFM-fut exactly."""
    hist, hist_obs = _load_history_semantics_and_observation(
        source, window.scene_name, window.history_tokens, cfg.free_label
    )
    fut_gt = (np.stack([source.load_semantics(window.scene_name, t) for t in window.future_tokens])
              if include_gt else None)
    trajectory = source.official_trajectory(
        window.history_tokens, window.future_tokens,
        hist_last=cfg.trajectory_hist_last,
        zero_prefix=cfg.trajectory_zero_prefix,
        require_info=cfg.require_temporal_info,
    )
    if trajectory.shape != (cfg.trajectory_length, 2):
        raise ValueError(
            f"{cfg.trajectory_protocol} requires trajectory [{cfg.trajectory_length},2], got {trajectory.shape}"
        )
    if cfg.trajectory_zero_prefix and not np.array_equal(
        trajectory[:cfg.trajectory_zero_prefix],
        np.zeros((cfg.trajectory_zero_prefix,2),dtype=np.float32),
    ):
        raise ValueError("OccFM-fut trajectory prefix masking does not match HIST_LAST contract")
    return {
        "history_occ": hist,
        "history_observed": hist_obs,
        "future_gt_occ": fut_gt,
        "history_poses": [source.pose(t) for t in window.history_tokens],
        "future_poses": [source.pose(t) for t in window.future_tokens],
        "trajectory": trajectory,
    }


def _sample_ann_map(nusc, token):
    sample=nusc.get("sample", token)
    out={}
    for ann_token in sample["anns"]:
        ann=nusc.get("sample_annotation", ann_token)
        out[ann["instance_token"]]=ann
    return out


def _sample_time_s(nusc, token):
    return float(nusc.get("sample", token)["timestamp"]) / 1e6


def _enrich_motion_records(records, source, window, t0_pose, components, horizon_s):
    """Attach history-speed and KTA-center-error diagnostics to metric records."""
    if not records:
        return records
    nusc=source.nusc
    t0_map=_sample_ann_map(nusc, window.t0_token)
    hist_tokens=window.history_tokens[:-1]
    hist_maps=[_sample_ann_map(nusc,t) for t in hist_tokens]
    hist_times=[_sample_time_s(nusc,t) for t in hist_tokens]
    t0_time=_sample_time_s(nusc,window.t0_token)
    inv_t0=np.linalg.inv(t0_pose)
    for rec in records:
        inst=rec["instance_token"]
        ann0=t0_map.get(inst)
        hist_speed=float("nan")
        for hm,ht in zip(hist_maps,hist_times):
            if inst in hm and ann0 is not None and t0_time>ht:
                cp=np.asarray(hm[inst]["translation"],dtype=np.float64)
                c0=np.asarray(ann0["translation"],dtype=np.float64)
                hist_speed=float(np.linalg.norm(c0[:2]-cp[:2])/(t0_time-ht))
                break
        rec["historical_speed_mps"]=hist_speed
        rec["delta_speed_mps"]=(float(rec["speed_mps"])-hist_speed
                                if np.isfinite(hist_speed) else float("nan"))
        c0w=np.asarray(rec["center0_world"],dtype=np.float64)
        chw=np.asarray(rec["centerh_world"],dtype=np.float64)
        c0e=(inv_t0 @ np.r_[c0w,1.0])[:2]
        che=(inv_t0 @ np.r_[chw,1.0])[:2]
        candidates=[c for c in components if int(c.class_id)==int(rec["class_id"])]
        if candidates:
            comp=min(candidates,key=lambda c:float(np.linalg.norm(c.centroid_xy_m-c0e)))
            pred=comp.centroid_xy_m + comp.velocity_xy_mps*float(horizon_s)
            rec["kta_center_error_m"]=float(np.linalg.norm(pred-che))
            rec["kta_component_matched_history"]=bool(comp.matched)
        else:
            rec["kta_center_error_m"]=float("nan")
            rec["kta_component_matched_history"]=False
    return records


def prepare_nuscenes_window(source, window, cfg: PrepareConfig = PrepareConfig(), include_gt=True, raw=None):
    """Prepare one 6+6 window under the frozen hybrid-support OccFM-fut protocol."""
    if len(window.history_tokens) != cfg.history_frames or len(window.future_tokens) != cfg.future_frames:
        raise ValueError("window length does not match PrepareConfig")
    if cfg.trajectory_length != cfg.history_frames + cfg.future_frames:
        raise ValueError("OccFM-fut trajectory length must equal 6-history + 6-future window length")
    if cfg.support_geometry != "hybrid_endpoint_swept_v1":
        raise ValueError(f"unsupported formal support geometry {cfg.support_geometry}")

    raw = load_nuscenes_window_raw(source, window, cfg, include_gt) if raw is None else raw
    hist = np.asarray(raw["history_occ"])
    hist_obs = np.asarray(raw["history_observed"], dtype=bool)
    fut_gt = None if not include_gt else np.asarray(raw["future_gt_occ"])
    hist_poses = list(raw["history_poses"])
    fut_poses = list(raw["future_poses"])
    t0_pose = hist_poses[-1]

    aligned_hist = ego_compensate_sequence(hist, hist_poses, -1, cfg.grid, cfg.free_label)
    aligned_obs = _align_observation_sequence(hist_obs, hist_poses, -1, cfg.grid)
    t0_masks = decompose_masks(
        aligned_hist, cfg.motion, history_observed=aligned_obs
    )
    static_current, _ = split_semantics(hist[-1], t0_masks, cfg.free_label)

    static_future = transport_current_to_future(
        static_current, t0_pose, fut_poses, cfg.grid, cfg.free_label
    )
    protected_future = transport_mask_to_future(
        t0_masks.confident_static, t0_pose, fut_poses, cfg.grid
    )

    horizons = [(i + 1) * cfg.frame_dt_s for i in range(cfg.future_frames)]

    # Routing contract:
    #   MOVING    -> KTA endpoint prior; write support is endpoint tube + thin swept corridor.
    #   UNCERTAIN -> zero object-motion prior; write support expands only by the frozen schedule.
    # Swept support is permission for the learned WM, never a semantic occupancy prediction.
    moving_kta_t0, _, components = causal_kta(
        aligned_hist, t0_masks.moving, horizons, cfg.grid, cfg.kta
    )
    uncertain_current = _masked_semantics(
        hist[-1], t0_masks.uncertain, cfg.free_label
    )

    moving_kta_future = []
    uncertain_zero_future = []
    kta_future = []
    for moving_t0, pose_h in zip(moving_kta_t0, fut_poses):
        rel = relative_transform(t0_pose, pose_h)
        moving_h = warp_semantic_grid(
            moving_t0, rel, cfg.grid, cfg.free_label
        )
        uncertain_h = warp_semantic_grid(
            uncertain_current, rel, cfg.grid, cfg.free_label
        )
        # Explicit motion has priority if a moving extrapolation and a
        # zero-motion uncertain anchor collide after transport.
        combined = uncertain_h.copy()
        write = moving_h != cfg.free_label
        combined[write] = moving_h[write]
        moving_kta_future.append(moving_h)
        uncertain_zero_future.append(uncertain_h)
        kta_future.append(combined)

    moving_kta_future = np.stack(moving_kta_future)
    uncertain_zero_future = np.stack(uncertain_zero_future)
    kta_future = np.stack(kta_future)

    moving_kta_support = (moving_kta_future != cfg.free_label).any(axis=3)
    uncertain_zero_support = (uncertain_zero_future != cfg.free_label).any(axis=3)
    kta_support = moving_kta_support | uncertain_zero_support

    swept_kta_support = swept_support_in_future_ego(
        components, horizons, t0_pose, fut_poses, cfg.grid
    )
    endpoint_generation_support = build_motion_tube(
        torch.from_numpy(moving_kta_support),
        MotionTubeConfig(cfg.endpoint_tube_radii, 0),
    ).cpu().numpy()
    swept_generation_support = build_motion_tube(
        torch.from_numpy(swept_kta_support),
        MotionTubeConfig(cfg.swept_tube_radii, 0),
    ).cpu().numpy()
    uncertain_generation_support = build_motion_tube(
        torch.from_numpy(uncertain_zero_support),
        MotionTubeConfig(cfg.uncertain_tube_radii, 0),
    ).cpu().numpy()
    moving_generation_support = endpoint_generation_support | swept_generation_support
    generation_support = moving_generation_support | uncertain_generation_support

    if include_gt:
        gt_supports, gt_moving, moving_records, excluded = [], [], [], []
        for h_idx, (tok_h, gt_h) in enumerate(zip(window.future_tokens, fut_gt)):
            dt = horizons[h_idx]
            support_h, records_h, excluded_h = gt_moving_support_for_horizon(
                source.nusc, window.t0_token, tok_h, dt, cfg.grid
            )
            records_h=_enrich_motion_records(records_h,source,window,t0_pose,components,dt)
            gt_supports.append(support_h)
            gt_moving.append(gt_moving_only_semantics(gt_h, support_h, cfg.free_label))
            moving_records.append(records_h)
            excluded.append(excluded_h)
        gt_supports = np.stack(gt_supports)
        gt_moving = np.stack(gt_moving)
        future_dynamic_target = np.stack([
            causal_dynamic_target_semantics(gt_h, generation_support[h_idx], cfg.free_label)
            for h_idx, gt_h in enumerate(fut_gt)
        ])
    else:
        gt_supports = gt_moving = future_dynamic_target = moving_records = excluded = None

    static_hist, moving_hist, hist_candidate_support = causal_history_split(
        hist, hist_poses, hist_obs, cfg
    )
    trajectory = np.asarray(raw["trajectory"], dtype=np.float32)
    if trajectory.shape != (cfg.trajectory_length,2):
        raise ValueError(f"prepared trajectory shape {trajectory.shape} violates {cfg.trajectory_protocol}")

    result = {
        "version": PREPARED_VERSION,
        "sample_id": f"{window.scene_name}:{window.t0_token}",
        "scene_name": window.scene_name,
        "history_tokens": list(window.history_tokens),
        "t0_token": window.t0_token,
        "future_tokens": list(window.future_tokens),
        "full_history_occ": hist,
        "history_observation_mask": hist_obs,
        "static_history_occ": static_hist,
        "moving_history_occ": moving_hist,
        "history_candidate_support": hist_candidate_support,
        "static_future_occ": static_future,
        "confident_static_future_mask": protected_future,
        "kta_future_occ": kta_future,
        "moving_kta_future_occ": moving_kta_future,
        "uncertain_zero_future_occ": uncertain_zero_future,
        "kta_support": kta_support,
        "moving_kta_support": moving_kta_support,
        "swept_kta_support": swept_kta_support,
        "uncertain_zero_support": uncertain_zero_support,
        "endpoint_generation_support": endpoint_generation_support,
        "swept_generation_support": swept_generation_support,
        "uncertain_generation_support": uncertain_generation_support,
        "moving_generation_support": moving_generation_support,
        "generation_support_occ": generation_support,
        "support_geometry": cfg.support_geometry,
        "endpoint_tube_radii": np.asarray(cfg.endpoint_tube_radii, dtype=np.int64),
        "swept_tube_radii": np.asarray(cfg.swept_tube_radii, dtype=np.int64),
        "uncertain_tube_radii": np.asarray(cfg.uncertain_tube_radii, dtype=np.int64),
        "trajectory": trajectory,
        "trajectory_protocol": cfg.trajectory_protocol,
        "horizons_s": np.asarray(horizons, dtype=np.float32),
        "t0_confident_static_mask": t0_masks.confident_static,
        "t0_moving_mask": t0_masks.moving,
        "t0_uncertain_mask": t0_masks.uncertain,
        "t0_persistence": t0_masks.persistence,
        "kta_components": [
            {
                "class_id": c.class_id,
                "centroid_xy_m": c.centroid_xy_m.tolist(),
                "velocity_xy_mps": c.velocity_xy_mps.tolist(),
                "matched": bool(c.matched),
                "voxel_count": int(len(c.voxel_indices)),
            }
            for c in components
        ],
    }
    if include_gt:
        result.update({
            "future_gt_occ": fut_gt,
            "future_moving_occ": gt_moving,
            "future_dynamic_target_occ": future_dynamic_target,
            "gt_moving_support": gt_supports,
            "moving_records": moving_records,
            "metric_excluded": excluded,
        })
    return result


def save_prepared_shards(output_dir, samples, shard_size=32, metadata=None):
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    entries, shard = [], []; shard_id = 0
    def flush(items, sid):
        if not items: return
        name = f"shard_{sid:05d}.pt"; torch.save(items, root / name)
        for local_idx, sample in enumerate(items):
            entries.append({"shard": name, "index": local_idx,
                            "sample_id": sample.get("sample_id", f"{sid}:{local_idx}")})
    for sample in samples:
        shard.append(sample)
        if len(shard) >= shard_size:
            flush(shard, shard_id); shard, shard_id = [], shard_id + 1
    flush(shard, shard_id)
    index = {"version": PREPARED_VERSION,"metadata": metadata or {},
             "num_samples": len(entries),"entries": entries}
    (root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


class PreparedShardDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if self.index.get("version") != PREPARED_VERSION:
            raise ValueError(
                f"unsupported prepared version {self.index.get('version')}; "
                "rebuild prepared data for frozen hybrid-balanced-r1 support"
            )
        self.entries = self.index["entries"]
        self._cached_name = None; self._cached_shard = None
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        entry = self.entries[idx]
        if entry["shard"] != self._cached_name:
            self._cached_shard = torch.load(self.root / entry["shard"], map_location="cpu", weights_only=False)
            self._cached_name = entry["shard"]
        return self._cached_shard[entry["index"]]
