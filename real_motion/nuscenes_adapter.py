"""nuScenes/Occ3D adapters used by P0 and cache preparation.

No future annotation is exposed to the causal model-preparation path. Functions
whose names contain ``gt_`` / ``metric`` are evaluation-or-target-only.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import math
import numpy as np

from .geometry import OccupancyGrid, pose_matrix, quaternion_yaw
from .metrics.moving_miou_v2 import (
    Box3D, GridSpec, DYNAMIC_CLASS_IDS, SPEED_THRESHOLD_MPS, BOX_MARGIN_M,
    moving_support_from_world_motion,
)


CATEGORY_PREFIX_TO_CLASS = (
    ("vehicle.bicycle", 2),
    ("vehicle.bus", 3),
    ("vehicle.car", 4),
    ("vehicle.construction", 5),
    ("vehicle.motorcycle", 6),
    ("human.pedestrian", 7),
    ("vehicle.trailer", 9),
    ("vehicle.truck", 10),
)


def category_to_dynamic_class(category_name):
    for prefix, cid in CATEGORY_PREFIX_TO_CLASS:
        if category_name.startswith(prefix):
            return cid
    return None


def wrap_angle(x):
    return (float(x) + math.pi) % (2 * math.pi) - math.pi


def sample_ego_to_world(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    return pose_matrix(ego["translation"], ego["rotation"])


def _annotation_map(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    out = {}
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        out[ann["instance_token"]] = ann
    return out


def box3d_to_dict(box: Box3D):
    return {
        "token": box.token, "class_id": int(box.class_id),
        "center_xyz": list(box.center_xyz), "size_lwh": list(box.size_lwh),
        "yaw": float(box.yaw),
    }


def box3d_from_dict(d):
    return Box3D(str(d["token"]), int(d["class_id"]), tuple(d["center_xyz"]),
                 tuple(d["size_lwh"]), float(d["yaw"]))


def _ann_to_future_ego_box(ann, target_ego_to_world, class_id):
    center_world = np.asarray(ann["translation"], dtype=np.float64)
    center_h = np.concatenate([center_world, [1.0]])
    center_ego = (np.linalg.inv(target_ego_to_world) @ center_h)[:3]
    yaw_world = quaternion_yaw(ann["rotation"])
    yaw_ego_world = math.atan2(target_ego_to_world[1, 0], target_ego_to_world[0, 0])
    yaw = wrap_angle(yaw_world - yaw_ego_world)
    # nuScenes size is [width, length, height].
    w, l, h = ann["size"]
    return Box3D(
        token=ann["instance_token"], class_id=int(class_id),
        center_xyz=tuple(float(x) for x in center_ego),
        size_lwh=(float(l), float(w), float(h)), yaw=float(yaw),
    )


def gt_moving_support_for_horizon(nusc, t0_token, th_token, dt_s,
                                  grid=OccupancyGrid(),
                                  speed_threshold=SPEED_THRESHOLD_MPS,
                                  margin=BOX_MARGIN_M):
    """Post-hoc Moving-mIoU v2 support for one target horizon."""
    a0, ah = _annotation_map(nusc, t0_token), _annotation_map(nusc, th_token)
    target_pose = sample_ego_to_world(nusc, th_token)
    metric_grid = GridSpec(grid.x_min, grid.y_min, grid.z_min,
                           grid.voxel_size, grid.shape_hwd)
    support = np.zeros(grid.shape_hwd, dtype=bool)
    moving_records = []

    dyn0 = {k: v for k, v in a0.items() if category_to_dynamic_class(v["category_name"]) is not None}
    dynh = {k: v for k, v in ah.items() if category_to_dynamic_class(v["category_name"]) is not None}
    common = sorted(set(dyn0) & set(dynh))
    for inst in common:
        ann0, annh = dyn0[inst], dynh[inst]
        cid = category_to_dynamic_class(annh["category_name"])
        c0 = np.asarray(ann0["translation"], dtype=np.float64)
        ch = np.asarray(annh["translation"], dtype=np.float64)
        speed = float(np.linalg.norm(ch[:2] - c0[:2]) / float(dt_s))
        if speed < speed_threshold:
            continue
        b0 = _ann_to_future_ego_box(ann0, target_pose, cid)
        bh = _ann_to_future_ego_box(annh, target_pose, cid)
        inst_support = moving_support_from_world_motion(
            c0, ch, b0, bh, dt_s, metric_grid, speed_threshold, margin,
        )
        support |= inst_support
        moving_records.append({
            "instance_token": inst,
            "class_id": int(cid),
            "speed_mps": speed,
            "center0_world": c0.tolist(),
            "centerh_world": ch.tolist(),
            "yaw0_world": quaternion_yaw(ann0["rotation"]),
            "yawh_world": quaternion_yaw(annh["rotation"]),
            "box0_future_ego": box3d_to_dict(b0),
            "boxh_future_ego": box3d_to_dict(bh),
        })

    excluded = {
        "birth_dynamic": len(set(dynh) - set(dyn0)),
        "death_dynamic": len(set(dyn0) - set(dynh)),
        "endpoint_common_dynamic": len(common),
        "moving_eligible": len(moving_records),
    }
    return support, moving_records, excluded


def dynamic_only_semantics(gt_semantics, free_label=17):
    gt=np.asarray(gt_semantics)
    out=np.full_like(gt,free_label)
    keep=np.isin(gt,np.asarray(DYNAMIC_CLASS_IDS))
    out[keep]=gt[keep]
    return out


def gt_moving_only_semantics(gt_semantics, moving_support, free_label=17):
    gt = np.asarray(gt_semantics)
    dyn = np.isin(gt, np.asarray(DYNAMIC_CLASS_IDS))
    out = np.full_like(gt, free_label)
    keep = dyn & moving_support
    out[keep] = gt[keep]
    return out


def causal_dynamic_target_semantics(gt_semantics, causal_bev_support, free_label=17):
    """Training target for the sparse WM under a *causal* support.

    This is deliberately different from ``gt_moving_only_semantics`` used by
    Moving-mIoU v2. The training target must not be defined by future GT
    instance motion. It keeps every dynamic-semantic GT voxel that lies inside
    the causal KTA/generation support, including a currently uncertain object
    that happens to remain stationary. This avoids training the WM to erase
    such objects merely because they fail the post-hoc 0.5 m/s metric test.
    """
    gt = np.asarray(gt_semantics)
    sup = np.asarray(causal_bev_support, dtype=bool)
    if sup.shape == gt.shape[:-1]:
        sup = sup[..., None]
    if sup.shape != gt.shape and sup.shape[:-1] != gt.shape[:-1]:
        raise ValueError("causal support is not aligned with semantic occupancy")
    sup = np.broadcast_to(sup, gt.shape)
    dyn = np.isin(gt, np.asarray(DYNAMIC_CLASS_IDS))
    out = np.full_like(gt, free_label)
    keep = dyn & sup
    out[keep] = gt[keep]
    return out


def gt_true_static_mask(gt_semantics, moving_support, free_label=17):
    gt = np.asarray(gt_semantics)
    return (gt != free_label) & ~moving_support


@dataclass(frozen=True)
class WindowTokens:
    scene_name: str
    history_tokens: tuple
    t0_token: str
    future_tokens: tuple


class NuScenesWindowSource:
    """Chronological keyframe windows matching OccFM's 6+6 protocol."""
    def __init__(self, dataroot, version="v1.0-trainval", info_pkl=None, verbose=False):
        from nuscenes.nuscenes import NuScenes
        self.dataroot = str(Path(dataroot).resolve())
        self.nusc = NuScenes(version=version, dataroot=self.dataroot, verbose=verbose)
        self.allowed_scenes = None
        self.info_by_token = {}
        if info_pkl:
            with open(info_pkl, "rb") as f:
                infos = pickle.load(f)["infos"]
            self.allowed_scenes = set(infos.keys())
            for _, frames in infos.items():
                for rec in frames:
                    self.info_by_token[rec["token"]] = rec

    def scene_tokens(self, scene):
        token = scene["first_sample_token"]
        out = []
        while token:
            out.append(token)
            rec = self.nusc.get("sample", token)
            token = rec["next"]
        return out

    def iter_windows(self, history=6, future=6, stride=1, max_windows=None):
        emitted = 0
        for scene in self.nusc.scene:
            if self.allowed_scenes is not None and scene["name"] not in self.allowed_scenes:
                continue
            tokens = self.scene_tokens(scene)
            for i in range(history - 1, len(tokens) - future, stride):
                yield WindowTokens(
                    scene_name=scene["name"],
                    history_tokens=tuple(tokens[i-history+1:i+1]),
                    t0_token=tokens[i],
                    future_tokens=tuple(tokens[i+1:i+future+1]),
                )
                emitted += 1
                if max_windows is not None and emitted >= max_windows:
                    return

    def load_semantics(self, scene_name, token):
        path = Path(self.dataroot) / "gts" / scene_name / token / "labels.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        return np.load(path)["semantics"]

    def pose(self, token):
        return sample_ego_to_world(self.nusc, token)

    def official_trajectory(self, t0_token, future_tokens=None):
        """Prefer the exact trajectory cached in official nuScenes info files."""
        rec = self.info_by_token.get(t0_token)
        if rec is not None and "gt_ego_fut_trajs" in rec:
            traj = np.asarray(rec["gt_ego_fut_trajs"][0], dtype=np.float32)
            return traj
        if future_tokens is None:
            raise ValueError("future_tokens required when info_pkl trajectory is unavailable")
        pose0 = self.pose(t0_token)
        inv0 = np.linalg.inv(pose0)
        xy = []
        for tok in future_tokens:
            p = self.pose(tok)
            world = np.r_[p[:3, 3], 1.0]
            rel = inv0 @ world
            xy.append(rel[:2])
        return np.asarray(xy, dtype=np.float32)
