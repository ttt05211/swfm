"""Raw occupancy-window preparation: geometry -> motion -> SE(3) -> KTA -> GT targets."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch
import numpy as np

from .geometry import (
    OccupancyGrid, ego_compensate_sequence, transport_current_to_future,
    transport_mask_to_future, relative_transform, warp_semantic_grid,
)
from .motion import PersistenceMotionConfig, decompose_masks, split_semantics, MotionMasks
from .kta import KTAConfig, causal_kta
from .support import MotionTubeConfig, build_motion_tube
from .nuscenes_adapter import (gt_moving_support_for_horizon, gt_moving_only_semantics,
                                causal_dynamic_target_semantics)

PREPARED_VERSION = "real_motion_prepared_v2"


@dataclass(frozen=True)
class PrepareConfig:
    history_frames: int = 6
    future_frames: int = 6
    frame_dt_s: float = 0.5
    free_label: int = 17
    tube_radii: tuple = (1, 2, 3, 4, 5, 6)
    grid: OccupancyGrid = OccupancyGrid()
    motion: PersistenceMotionConfig = PersistenceMotionConfig()
    kta: KTAConfig = KTAConfig()


def _early_frame_masks(sem, free_label):
    occupied = sem != free_label
    zeros = np.zeros_like(occupied)
    # No historical evidence => never hard-freeze; send occupancy to uncertain.
    return MotionMasks(zeros, zeros, occupied, ~occupied,
                       np.zeros_like(sem, dtype=np.float32))


def causal_history_split(history_native, history_poses, cfg: PrepareConfig):
    """Causally split every history frame in its own ego coordinates."""
    static_frames, moving_frames, candidate_support = [], [], []
    for j in range(len(history_native)):
        if j + 1 < cfg.motion.min_observed_frames:
            masks = _early_frame_masks(history_native[j], cfg.free_label)
        else:
            prefix_sem = history_native[:j+1]
            prefix_pose = history_poses[:j+1]
            aligned = ego_compensate_sequence(prefix_sem, prefix_pose, -1, cfg.grid, cfg.free_label)
            masks = decompose_masks(aligned, cfg.motion)
        sta, mov = split_semantics(history_native[j], masks, cfg.free_label)
        static_frames.append(sta)
        moving_frames.append(mov)
        candidate_support.append(masks.wm_candidate.any(axis=2))
    return (
        np.stack(static_frames), np.stack(moving_frames), np.stack(candidate_support)
    )


def load_nuscenes_window_raw(source, window, cfg: PrepareConfig = PrepareConfig(), include_gt=True):
    """Load raw arrays/poses once. Profilers can call this outside the timed region."""
    hist = np.stack([source.load_semantics(window.scene_name, t) for t in window.history_tokens])
    fut_gt = (np.stack([source.load_semantics(window.scene_name, t) for t in window.future_tokens])
              if include_gt else None)
    return {
        "history_occ": hist,
        "future_gt_occ": fut_gt,
        "history_poses": [source.pose(t) for t in window.history_tokens],
        "future_poses": [source.pose(t) for t in window.future_tokens],
        "trajectory": source.official_trajectory(window.t0_token, window.future_tokens),
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
    """Prepare one 6+6 window.

    Future GT is used only for ``future_moving_occ`` / metric supports and P0
    diagnostics. Causal masks, KTA, and generation support depend only on
    history and ego poses.
    """
    if len(window.history_tokens) != cfg.history_frames or len(window.future_tokens) != cfg.future_frames:
        raise ValueError("window length does not match PrepareConfig")

    raw = load_nuscenes_window_raw(source, window, cfg, include_gt) if raw is None else raw
    hist = np.asarray(raw["history_occ"])
    fut_gt = None if not include_gt else np.asarray(raw["future_gt_occ"])
    hist_poses = list(raw["history_poses"])
    fut_poses = list(raw["future_poses"])
    t0_pose = hist_poses[-1]

    # Causal t0 real-motion split uses an ego-compensated copy only for state inference.
    aligned_hist = ego_compensate_sequence(hist, hist_poses, -1, cfg.grid, cfg.free_label)
    t0_masks = decompose_masks(aligned_hist, cfg.motion)
    static_current, candidate_current = split_semantics(hist[-1], t0_masks, cfg.free_label)

    static_future = transport_current_to_future(
        static_current, t0_pose, fut_poses, cfg.grid, cfg.free_label
    )
    protected_future = transport_mask_to_future(
        t0_masks.confident_static, t0_pose, fut_poses, cfg.grid
    )

    horizons = [(i + 1) * cfg.frame_dt_s for i in range(cfg.future_frames)]
    kta_t0, _, components = causal_kta(
        aligned_hist, t0_masks.wm_candidate, horizons, cfg.grid, cfg.kta
    )
    # KTA is predicted in t0 ego coordinates, then converted into each target future ego grid.
    kta_future = []
    for sem_t0, pose_h in zip(kta_t0, fut_poses):
        kta_future.append(warp_semantic_grid(
            sem_t0, relative_transform(t0_pose, pose_h), cfg.grid, cfg.free_label
        ))
    kta_future = np.stack(kta_future)
    kta_support = (kta_future != cfg.free_label).any(axis=3)
    generation_support = build_motion_tube(
        torch.from_numpy(kta_support), MotionTubeConfig(cfg.tube_radii, 0)
    ).cpu().numpy()

    # GT target/evaluation branch. Never consumed by KTA/motion detection.
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
        # Supervision is gated only by the causal generation support. Future GT
        # instance motion is reserved for evaluation and never defines M_gen.
        future_dynamic_target = np.stack([
            causal_dynamic_target_semantics(gt_h, generation_support[h_idx], cfg.free_label)
            for h_idx, gt_h in enumerate(fut_gt)
        ])
    else:
        gt_supports = gt_moving = future_dynamic_target = moving_records = excluded = None

    static_hist, moving_hist, hist_candidate_support = causal_history_split(hist, hist_poses, cfg)
    trajectory = np.asarray(raw["trajectory"], dtype=np.float32)

    result = {
        "version": PREPARED_VERSION,
        "sample_id": f"{window.scene_name}:{window.t0_token}",
        "scene_name": window.scene_name,
        "history_tokens": list(window.history_tokens),
        "t0_token": window.t0_token,
        "future_tokens": list(window.future_tokens),
        "full_history_occ": hist,
        "static_history_occ": static_hist,
        "moving_history_occ": moving_hist,
        "history_candidate_support": hist_candidate_support,
        "static_future_occ": static_future,
        "confident_static_future_mask": protected_future,
        "kta_future_occ": kta_future,
        "kta_support": kta_support,
        "generation_support_occ": generation_support,
        "trajectory": trajectory,
        "horizons_s": np.asarray(horizons, dtype=np.float32),
        "t0_confident_static_mask": t0_masks.confident_static,
        "t0_moving_mask": t0_masks.moving,
        "t0_uncertain_mask": t0_masks.uncertain,
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
            "future_moving_occ": gt_moving,  # metric-only eligible Moving instances
            "future_dynamic_target_occ": future_dynamic_target,  # WM supervision under causal support
            "gt_moving_support": gt_supports,
            "moving_records": moving_records,
            "metric_excluded": excluded,
        })
    return result


def save_prepared_shards(output_dir, samples, shard_size=32, metadata=None):
    """Save large raw prepared windows lazily; avoids one monolithic .pt."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    entries, shard = [], []
    shard_id = 0

    def flush(items, sid):
        if not items:
            return
        name = f"shard_{sid:05d}.pt"
        torch.save(items, root / name)
        for local_idx, sample in enumerate(items):
            entries.append({"shard": name, "index": local_idx,
                            "sample_id": sample.get("sample_id", f"{sid}:{local_idx}")})

    for sample in samples:
        shard.append(sample)
        if len(shard) >= shard_size:
            flush(shard, shard_id)
            shard, shard_id = [], shard_id + 1
    flush(shard, shard_id)
    index = {
        "version": PREPARED_VERSION,
        "metadata": metadata or {},
        "num_samples": len(entries),
        "entries": entries,
    }
    (root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


class PreparedShardDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if self.index.get("version") != PREPARED_VERSION:
            raise ValueError(f"unsupported prepared version {self.index.get('version')}")
        self.entries = self.index["entries"]
        self._cached_name = None
        self._cached_shard = None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        if entry["shard"] != self._cached_name:
            self._cached_shard = torch.load(self.root / entry["shard"], map_location="cpu",
                                            weights_only=False)
            self._cached_name = entry["shard"]
        return self._cached_shard[entry["index"]]
