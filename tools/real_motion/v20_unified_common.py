"""Shared real-data adapter for V20 unified training and evaluation."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.motion_transport import world_points_to_t0
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    WindowTokens,
    gt_moving_support_sequence,
)
from real_motion.rigid_transport import (
    rasterize_rigid_components_batched,
    rigid_source_points_world,
)
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    extract_instances,
    strong_w2det_sequence,
)
from real_motion.v18_two_wheel_diagnostic import renderer_yaw_delta
from real_motion.v20_history_world import CanonicalLattice, FREE_LABEL
from real_motion.v20_stage1_codec import unpack_bool, unpack_history_semantic
from real_motion.v20_unified_data import (
    CompletionTile,
    CompletionTileQuery,
    UnifiedHistoryInput,
    sample_training_tiles,
)
from real_motion.v20_unified_runtime import (
    completion_support,
    compose_completion_tiles_at_free_logit_offsets,
    dense_geometry_and_transport_condition,
)
STAGE1_PROTOCOL = "p0_f9_v20_stage1_history_cache_v2"


class Stage1RowStore:
    """Lazy shard-backed Stage-1 rows with a small shard LRU."""

    def __init__(self, root: Path, index: dict, max_cached_shards: int = 8):
        self.root = Path(root)
        self.index = index
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.locations: dict[tuple[str, str], tuple[str, int]] = {}
        self.cache: OrderedDict[str, list[dict]] = OrderedDict()
        self.shard_meta: dict[str, dict] = {}
        self.expected_keys: dict[str, tuple[tuple[str, str], ...]] = {}
        self.verified_files: set[str] = set()
        root_resolved = self.root.resolve()
        for shard in index["shards"]:
            file = str(shard["file"])
            path = (self.root / file).resolve()
            if path != root_resolved and root_resolved not in path.parents:
                raise RuntimeError(f"Stage1 shard escapes cache root: {file}")
            if file in self.shard_meta:
                raise RuntimeError(f"duplicate Stage1 shard entry: {file}")
            self.shard_meta[file] = dict(shard)
            keys = shard.get("keys")
            if keys is None:
                obj = torch.load(
                    path, map_location="cpu", weights_only=False
                )
                if obj.get("protocol") != STAGE1_PROTOCOL:
                    raise RuntimeError(f"bad Stage1 shard: {file}")
                rows = list(obj["rows"])
                keys = [[str(r["scene_name"]), str(r["t0_token"])] for r in rows]
                self.cache[file] = rows
            expected_count = int(shard.get("count", len(keys)))
            if len(keys) != expected_count:
                raise RuntimeError(
                    f"Stage1 shard key/count mismatch: {file}: "
                    f"{len(keys)} != {expected_count}"
                )
            normalized_keys = tuple((str(key[0]), str(key[1])) for key in keys)
            self.expected_keys[file] = normalized_keys
            for row_index, key in enumerate(keys):
                pair = (str(key[0]), str(key[1]))
                if pair in self.locations:
                    raise RuntimeError(f"duplicate Stage1 row: {pair}")
                self.locations[pair] = (file, int(row_index))
        declared_windows = index.get("num_windows")
        if (
            declared_windows is not None
            and int(declared_windows) != len(self.locations)
        ):
            raise RuntimeError(
                "Stage1 index window/key count mismatch: "
                f"{declared_windows} != {len(self.locations)}"
            )
        while len(self.cache) > self.max_cached_shards:
            self.cache.popitem(last=False)

    def __contains__(self, key) -> bool:
        return tuple(key) in self.locations

    def __len__(self) -> int:
        return len(self.locations)

    @property
    def ordered_keys(self) -> tuple[tuple[str, str], ...]:
        """Frozen Stage-1 population in index shard/row order."""
        return tuple(self.locations)

    def _rows(self, file: str) -> list[dict]:
        if file in self.cache:
            rows = self.cache.pop(file)
            self.cache[file] = rows
            return rows
        path = (self.root / file).resolve()
        meta = self.shard_meta[file]
        if file not in self.verified_files:
            actual_bytes = int(path.stat().st_size)
            if "bytes" in meta and actual_bytes != int(meta["bytes"]):
                raise RuntimeError(
                    f"Stage1 shard byte-size mismatch: {file}: "
                    f"{actual_bytes} != {meta['bytes']}"
                )
            expected_sha = meta.get("sha256")
            if expected_sha is not None and _file_sha256(path) != str(expected_sha):
                raise RuntimeError(f"Stage1 shard sha256 mismatch: {file}")
            self.verified_files.add(file)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj.get("protocol") != STAGE1_PROTOCOL:
            raise RuntimeError(f"bad Stage1 shard: {file}")
        rows = list(obj["rows"])
        expected = self.expected_keys[file]
        if len(rows) != len(expected):
            raise RuntimeError(
                f"Stage1 shard row/count mismatch: {file}: "
                f"{len(rows)} != {len(expected)}"
            )
        actual = tuple(
            (str(row["scene_name"]), str(row["t0_token"])) for row in rows
        )
        if actual != expected:
            raise RuntimeError(f"Stage1 shard row order/key mismatch: {file}")
        self.cache[file] = rows
        while len(self.cache) > self.max_cached_shards:
            self.cache.popitem(last=False)
        return rows

    def __getitem__(self, key):
        file, row_index = self.locations[tuple(key)]
        row = self._rows(file)[row_index]
        actual = (str(row["scene_name"]), str(row["t0_token"]))
        if actual != tuple(key):
            raise RuntimeError(
                f"Stage1 row identity mismatch: requested={tuple(key)} actual={actual}"
            )
        return row


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage1_manifest_paths(
    prefix: str,
    index_path: str | Path,
    index: dict,
) -> dict[str, Path]:
    """Return the index and every shard for exact resume hashing."""
    index_path = Path(index_path)
    out = {f"{prefix}_index": index_path}
    for shard in index["shards"]:
        file = str(shard["file"])
        out[f"{prefix}_shard:{file}"] = index_path.parent / file
    return out


def _load_unified_raw(source, window, pcfg):
    """Load only arrays unified V20 consumes; skip OccFM trajectory work."""
    history_occ, history_observed = [], []
    for token in window.history_tokens:
        sem, obs = source.load_occ3d(
            str(window.scene_name), str(token), require_lidar_mask=True
        )
        history_occ.append(np.asarray(sem, dtype=np.uint8))
        history_observed.append(np.asarray(obs, dtype=bool))
    future_gt = np.stack(
        [
            np.asarray(
                source.load_semantics(str(window.scene_name), str(token)),
                dtype=np.uint8,
            )
            for token in window.future_tokens
        ]
    )
    return {
        "history_occ": np.stack(history_occ),
        "history_observed": np.stack(history_observed),
        "future_gt_occ": future_gt,
        "history_poses": [source.pose(t) for t in window.history_tokens],
        "future_poses": [source.pose(t) for t in window.future_tokens],
    }


class CachedSource(NuScenesWindowSource):
    @lru_cache(maxsize=768)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=768)
    def load_occ3d(self, scene_name, token, require_lidar_mask=True):
        return super().load_occ3d(
            scene_name, token, require_lidar_mask=require_lidar_mask
        )

    @lru_cache(maxsize=4096)
    def pose(self, token):
        return super().pose(token)


class ComponentLRU:
    def __init__(self, maxsize: int = 1024):
        self.maxsize = max(1, int(maxsize))
        self.data = OrderedDict()

    def get_or_build(self, scene, token, semantics, pose, *, grid, cfg):
        key = (str(scene), str(token))
        if key in self.data:
            value = self.data.pop(key)
            self.data[key] = value
            return value
        value = extract_instances(
            np.asarray(semantics, dtype=np.uint8),
            np.asarray(pose, dtype=np.float64),
            grid=grid,
            cfg=cfg,
        )
        self.data[key] = value
        while len(self.data) > self.maxsize:
            self.data.popitem(last=False)
        return value


def window_from_record(record: dict) -> WindowTokens:
    return WindowTokens(
        scene_name=str(record["scene_name"]),
        history_tokens=tuple(str(x) for x in record["history_tokens"]),
        t0_token=str(record["t0_token"]),
        future_tokens=tuple(str(x) for x in record["future_tokens"]),
    )


def load_v18_cache(path: str | Path) -> tuple[dict, list[dict]]:
    from real_motion.local_st_world_model_v18_se2 import (
        SE2_CACHE_VERSION,
        SE2_TARGET_CONTRACT,
    )

    cache_path = Path(path)
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "v18_cache_load": "start",
                "path": str(cache_path.resolve()),
                "bytes": int(cache_path.stat().st_size),
            }
        ),
        flush=True,
    )
    obj = torch.load(cache_path, map_location="cpu", weights_only=False)
    if obj.get("version") != SE2_CACHE_VERSION:
        raise RuntimeError(f"expected {SE2_CACHE_VERSION}: {path}")
    metadata = obj.get("metadata") or {}
    if metadata.get("se2_target_contract") != SE2_TARGET_CONTRACT:
        raise RuntimeError("SE2 target contract mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("V18 cache has no records")
    print(
        json.dumps(
            {
                "v18_cache_load": "complete",
                "path": str(cache_path.resolve()),
                "records": len(records),
                "elapsed_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    return metadata, records


def align_v18_records_to_stage1(
    records: Sequence[dict],
    stage1_rows: Stage1RowStore,
    *,
    population_name: str,
) -> tuple[list[dict], dict[str, int | str]]:
    """Strictly select/reorder V18 windows by the frozen Stage-1 population."""
    by_key: dict[tuple[str, str], dict] = {}
    duplicate_keys: list[tuple[str, str]] = []
    for record in records:
        try:
            key = (str(record["scene_name"]), str(record["t0_token"]))
        except KeyError as exc:
            raise RuntimeError(
                f"{population_name}: V18 record lacks identity field {exc}"
            ) from exc
        if key in by_key:
            duplicate_keys.append(key)
        else:
            by_key[key] = record
    if duplicate_keys:
        raise RuntimeError(
            f"{population_name}: duplicate V18 identities: "
            f"count={len(duplicate_keys)} examples={duplicate_keys[:5]}"
        )

    frozen_keys = stage1_rows.ordered_keys
    missing = [key for key in frozen_keys if key not in by_key]
    if missing:
        raise RuntimeError(
            f"{population_name}: V18 cache misses frozen Stage1 identities: "
            f"count={len(missing)} examples={missing[:5]}"
        )
    aligned = [by_key[key] for key in frozen_keys]
    actual = tuple(
        (str(record["scene_name"]), str(record["t0_token"]))
        for record in aligned
    )
    if actual != frozen_keys:
        raise RuntimeError(f"{population_name}: aligned V18/Stage1 order mismatch")
    if len(aligned) != len(frozen_keys) or len(set(actual)) != len(frozen_keys):
        raise RuntimeError(
            f"{population_name}: aligned population cardinality mismatch"
        )
    return aligned, {
        "population": str(population_name),
        "v18_source_records": len(records),
        "stage1_frozen_keys": len(frozen_keys),
        "matched": len(aligned),
        "unique": len(set(actual)),
        "missing": 0,
        "duplicate": 0,
    }


def _t0_xy_to_world(xy_t0, source_z_t0, t0_pose):
    point = np.asarray(
        [float(xy_t0[0]), float(xy_t0[1]), float(source_z_t0), 1.0],
        dtype=np.float64,
    )
    return (np.asarray(t0_pose, dtype=np.float64) @ point)[:3]


def _gpu_inputs(record: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": record["features"].float().to(device),
        "tube": record["local_semantic_tube"].to(device),
        "kta": record["kta_displacement_xy_m"].float().to(device),
        "frame_motion": record["frame_motion_features"].float().to(device),
        "source_mask": record["target_source_mask_tube"].to(device),
    }


def lattice_from_dict(value: dict) -> CanonicalLattice:
    return CanonicalLattice(
        tuple(float(x) for x in value["origin_xyz_m"]),
        tuple(float(x) for x in value["voxel_size_xyz_m"]),
        tuple(int(x) for x in value["shape_xyz"]),
    )


def load_stage1_rows(
    path: str | Path, *, max_cached_shards: int = 8
) -> tuple[Path, dict, Stage1RowStore]:
    root = Path(path)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("protocol") != STAGE1_PROTOCOL:
        raise RuntimeError(f"unexpected Stage1 protocol: {index.get('protocol')}")
    return (
        index_path,
        index,
        Stage1RowStore(root, index, max_cached_shards=max_cached_shards),
    )


def decode_stage1_history(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (6,) + tuple(int(x) for x in row["coarse_shape_xyz"])
    observed = unpack_bool(row["history_observed_bits"], shape)
    observed_free = unpack_bool(row["history_observed_free_bits"], shape)
    semantic = unpack_history_semantic(row, observed, observed_free)
    return semantic, observed, observed_free


@dataclass
class PreparedUnifiedWindow:
    record: dict
    row: dict
    raw: dict
    state: dict
    history: UnifiedHistoryInput
    source_anchor_xyz_t0_m: torch.Tensor
    future_semantic: torch.Tensor
    formal_valid: torch.Tensor

    def release(self) -> None:
        self.state["gpu"] = None


def move_prepared_to_device(
    prepared: PreparedUnifiedWindow, device: torch.device
) -> PreparedUnifiedWindow:
    """Materialize a CPU-prepared window on the training device."""
    target = torch.device(device)
    if target.type == "cpu":
        return prepared
    gpu = prepared.state.get("gpu")
    if gpu is None:
        raise RuntimeError("cannot materialize a released prepared window")
    prepared.state["gpu"] = {
        name: value.to(target, non_blocking=False) for name, value in gpu.items()
    }
    history = UnifiedHistoryInput(
        semantic=prepared.history.semantic.to(target, non_blocking=False),
        observed=prepared.history.observed.to(target, non_blocking=False),
        observed_free=prepared.history.observed_free.to(target, non_blocking=False),
        future_ego_to_t0=prepared.history.future_ego_to_t0.to(
            target, non_blocking=False
        ),
    )
    return replace(
        prepared,
        history=history,
        source_anchor_xyz_t0_m=prepared.source_anchor_xyz_t0_m.to(
            target, non_blocking=False
        ),
        future_semantic=prepared.future_semantic.to(target, non_blocking=False),
        formal_valid=prepared.formal_valid.to(target, non_blocking=False),
    )


def prepare_unified_window(
    record: dict,
    row: dict,
    *,
    source,
    pcfg,
    strong_cfg: StrongW2DetConfig,
    component_cache: ComponentLRU,
    device: torch.device,
) -> PreparedUnifiedWindow:
    window = window_from_record(record)
    raw = _load_unified_raw(source, window, pcfg)
    current_semantic = np.asarray(raw["history_occ"][-1], dtype=np.uint8)
    current_pose = np.asarray(raw["history_poses"][-1], dtype=np.float64)
    current = component_cache.get_or_build(
        str(window.scene_name),
        str(window.t0_token),
        current_semantic,
        current_pose,
        grid=pcfg.grid,
        cfg=strong_cfg,
    )
    previous = component_cache.get_or_build(
        str(window.scene_name),
        str(window.history_tokens[-2]),
        np.asarray(raw["history_occ"][-2], dtype=np.uint8),
        np.asarray(raw["history_poses"][-2], dtype=np.float64),
        grid=pcfg.grid,
        cfg=strong_cfg,
    )
    expected_classes = [int(x) for x in record["source_class_id"].tolist()]
    if [int(x["class_id"]) for x in current] != expected_classes:
        raise RuntimeError(f"{record['sample_id']}: Strong/source order mismatch")
    anchors = strong_w2det_sequence(
        np.asarray(raw["history_occ"], dtype=np.uint8),
        np.asarray(raw["history_poses"], dtype=np.float64),
        np.asarray(raw["future_poses"], dtype=np.float64),
        frame_dt_s=float(pcfg.frame_dt_s),
        grid=pcfg.grid,
        cfg=strong_cfg,
        current_instances=current,
        previous_instances=previous,
    )
    centroids_world = np.asarray(
        [comp["centroid_world"] for comp in current], dtype=np.float64
    ).reshape(-1, 3)
    source_z_t0 = (
        world_points_to_t0(centroids_world, current_pose)[:, 2]
        if len(centroids_world)
        else np.empty((0,), dtype=np.float64)
    )
    future_poses = np.asarray(raw["future_poses"], dtype=np.float64)
    future_world_to_ego = [np.linalg.inv(pose) for pose in future_poses]
    source_points_world = [
        rigid_source_points_world(
            comp["voxel_indices"], current_pose, grid=pcfg.grid
        )
        for comp in current
    ]
    source_centers_world = np.asarray(
        [comp["centroid_world"] for comp in current], dtype=np.float64
    ).reshape(-1, 3)
    source_class_ids = np.asarray(
        [int(comp["class_id"]) for comp in current], dtype=np.int64
    )
    anchor_xy = record["anchors_xy_t0_m"].float().cpu().numpy()
    baseline_clear_masks = []
    for horizon in range(6):
        clear = np.zeros(tuple(pcfg.grid.shape_hwd), dtype=bool)
        target_centers = np.asarray(
            [
                _t0_xy_to_world(
                    anchor_xy[source_index, horizon],
                    source_z_t0[source_index],
                    current_pose,
                )
                for source_index in range(len(current))
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        idx, _ = rasterize_rigid_components_batched(
            source_points_world,
            source_centers_world,
            target_centers,
            np.zeros((len(current),), dtype=np.float64),
            future_world_to_ego[horizon],
            grid=pcfg.grid,
        )
        if len(idx):
            clear[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        baseline_clear_masks.append(clear)
    state = {
        "rec": record,
        "window": window,
        "current_pose": current_pose,
        "future_poses": future_poses,
        "future_world_to_ego": future_world_to_ego,
        "current": current,
        "source_points_world": source_points_world,
        "source_centers_world": source_centers_world,
        "source_class_ids": source_class_ids,
        "anchors": anchors,
        "baseline_clear_masks": baseline_clear_masks,
        "source_z_t0": source_z_t0,
        "gpu": _gpu_inputs(record, device),
    }
    semantic, observed, observed_free = decode_stage1_history(row)
    future_pose = torch.as_tensor(row["future_ego_to_t0"], device=device).float().unsqueeze(0)
    history = UnifiedHistoryInput(
        semantic=torch.from_numpy(semantic).to(device).unsqueeze(0),
        observed=torch.from_numpy(observed).to(device).unsqueeze(0),
        observed_free=torch.from_numpy(observed_free).to(device).unsqueeze(0),
        future_ego_to_t0=future_pose,
    )
    n = int(record["features"].shape[0])
    if "source_centroid_xy_t0_m" in record:
        xy = record["source_centroid_xy_t0_m"].float()
    else:
        xy = record["features"][:, :2].float() * 40.0
    z = torch.as_tensor(state["source_z_t0"], dtype=torch.float32).reshape(n, 1)
    anchor = torch.cat((xy.cpu(), z), dim=-1).to(device)
    future_semantic = torch.from_numpy(
        np.asarray(raw["future_gt_occ"], dtype=np.int64)
    ).to(device).unsqueeze(0)
    formal_valid = (future_semantic >= 0) & (future_semantic < 18)
    return PreparedUnifiedWindow(
        record=record,
        row=row,
        raw=raw,
        state=state,
        history=history,
        source_anchor_xyz_t0_m=anchor,
        future_semantic=future_semantic,
        formal_valid=formal_valid,
    )


def first_stage_forward_batch(
    model,
    prepared_windows: Sequence[PreparedUnifiedWindow],
    *,
    adapter_enabled: bool,
):
    if not prepared_windows:
        raise ValueError("first-stage batch cannot be empty")
    gpu_rows = [prepared.state["gpu"] for prepared in prepared_windows]
    history = UnifiedHistoryInput(
        semantic=torch.cat([p.history.semantic for p in prepared_windows], dim=0),
        observed=torch.cat([p.history.observed for p in prepared_windows], dim=0),
        observed_free=torch.cat(
            [p.history.observed_free for p in prepared_windows], dim=0
        ),
        future_ego_to_t0=torch.cat(
            [p.history.future_ego_to_t0 for p in prepared_windows], dim=0
        ),
    )
    source_counts = [int(gpu["features"].shape[0]) for gpu in gpu_rows]
    window_index = torch.cat(
        [
            torch.full(
                (count,),
                window,
                device=gpu_rows[window]["features"].device,
                dtype=torch.long,
            )
            for window, count in enumerate(source_counts)
        ],
        dim=0,
    )
    return model(
        history,
        features=torch.cat([gpu["features"] for gpu in gpu_rows], dim=0),
        local_semantic_tube=torch.cat([gpu["tube"] for gpu in gpu_rows], dim=0),
        kta_displacement_xy_m=torch.cat([gpu["kta"] for gpu in gpu_rows], dim=0),
        frame_motion_features=torch.cat(
            [gpu["frame_motion"] for gpu in gpu_rows], dim=0
        ),
        target_source_mask_tube=torch.cat(
            [gpu["source_mask"] for gpu in gpu_rows], dim=0
        ),
        source_anchor_xyz_t0_m=torch.cat(
            [p.source_anchor_xyz_t0_m for p in prepared_windows], dim=0
        ),
        window_index=window_index,
        adapter_enabled=adapter_enabled,
    )


def first_stage_forward(model, prepared: PreparedUnifiedWindow, *, adapter_enabled: bool):
    return first_stage_forward_batch(
        model, [prepared], adapter_enabled=adapter_enabled
    )


def hard_render_transport(
    model,
    prepared: PreparedUnifiedWindow,
    transport_outputs: dict[str, torch.Tensor],
    *,
    pcfg,
    strong_cfg,
    device,
) -> torch.Tensor:
    residual = transport_outputs["residual_xy_m"].detach().float().cpu().numpy()
    yaw = transport_outputs["yaw_delta_rad"].detach().float().cpu().numpy()
    record = prepared.record
    state = prepared.state
    anchor_xy = record["anchors_xy_t0_m"].float().cpu().numpy()
    rendered = []
    for horizon in range(6):
        count = len(state["current"])
        target_centers = np.asarray(
            [
                _t0_xy_to_world(
                    anchor_xy[source_index, horizon]
                    + residual[source_index, horizon],
                    state["source_z_t0"][source_index],
                    state["current_pose"],
                )
                for source_index in range(count)
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        renderer_yaw = np.asarray(
            [
                renderer_yaw_delta(
                    int(state["source_class_ids"][source_index]),
                    float(yaw[source_index, horizon]),
                    zero_two_wheel_yaw=False,
                )
                for source_index in range(count)
            ],
            dtype=np.float64,
        )
        idx, owners = rasterize_rigid_components_batched(
            state["source_points_world"],
            state["source_centers_world"],
            target_centers,
            renderer_yaw,
            state["future_world_to_ego"][horizon],
            grid=pcfg.grid,
        )
        out = np.asarray(state["anchors"][horizon]).copy()
        clear = state["baseline_clear_masks"][horizon]
        out[clear & np.isin(out, np.asarray(DYNAMIC_CLASS_IDS))] = int(
            pcfg.free_label
        )
        if len(idx):
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = state[
                "source_class_ids"
            ][owners]
        rendered.append(out)
    return torch.from_numpy(np.asarray(rendered, dtype=np.int64)).to(device).unsqueeze(0)


def moving_support_sequence(source, record: dict, *, grid, workers: int):
    window = window_from_record(record)
    rows = gt_moving_support_sequence(
        source.nusc,
        str(window.t0_token),
        window.future_tokens,
        tuple(0.5 * (i + 1) for i in range(6)),
        grid=grid,
        workers=int(workers),
    )
    return np.stack([row[0] for row in rows], axis=0)


def geometry_and_support(
    model,
    prepared: PreparedUnifiedWindow,
    current_transport: torch.Tensor,
    native_grid: dict,
):
    geometry_valid, condition = dense_geometry_and_transport_condition(
        current_transport,
        prepared.history.future_ego_to_t0,
        coarse_lattice=model.coarse_lattice,
        native_origin_xyz_m=native_grid["origin_xyz_m"],
        native_voxel_size_xyz_m=native_grid["voxel_size_xyz_m"],
    )
    return (
        geometry_valid,
        completion_support(current_transport, geometry_valid),
        condition,
    )


def deduplicate_completion_queries(
    queries: Sequence[CompletionTileQuery],
) -> tuple[list[CompletionTileQuery], list[int]]:
    """Reuse exact repeated draws while preserving their loss multiplicity."""
    unique: list[CompletionTileQuery] = []
    inverse: list[int] = []
    lookup: dict[tuple, int] = {}
    for query in queries:
        tile = query.tile
        key = (
            int(tile.window_index),
            int(tile.horizon),
            tuple(tile.core_start_xyz),
            tuple(tile.core_shape_xyz),
            tuple(tile.halo_start_xyz),
            tuple(tile.halo_shape_xyz),
            tuple((sl.start, sl.stop, sl.step) for sl in tile.core_slice_xyz),
            id(query.points_xyz_t0_m),
            id(query.geometry_valid),
            id(query.support),
        )
        unique_index = lookup.get(key)
        if unique_index is None:
            unique_index = len(unique)
            lookup[key] = unique_index
            unique.append(query)
        inverse.append(unique_index)
    return unique, inverse


def training_completion_inputs(
    model,
    prepared: PreparedUnifiedWindow,
    first_stage: dict,
    current_transport: torch.Tensor,
    *,
    native_grid: dict,
    generator: torch.Generator,
    draws_per_horizon: int = 16,
    positive_draws: int = 8,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict]:
    return training_completion_inputs_batch(
        model,
        [prepared],
        first_stage,
        current_transport,
        native_grid=native_grid,
        generator=generator,
        draws_per_horizon=draws_per_horizon,
        positive_draws=positive_draws,
    )


def training_completion_inputs_batch(
    model,
    prepared_windows: Sequence[PreparedUnifiedWindow],
    first_stage: dict,
    current_transport: torch.Tensor,
    *,
    native_grid: dict,
    generator: torch.Generator,
    draws_per_horizon: int = 16,
    positive_draws: int = 8,
    collect_distribution_stats: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict]:
    if not prepared_windows:
        raise ValueError("completion batch cannot be empty")
    future_semantic = torch.cat(
        [prepared.future_semantic for prepared in prepared_windows], dim=0
    )
    formal_valid = torch.cat(
        [prepared.formal_valid for prepared in prepared_windows], dim=0
    )
    future_ego_to_t0 = first_stage["history"].future_ego_to_t0
    geometry_valid, condition = dense_geometry_and_transport_condition(
        current_transport,
        future_ego_to_t0,
        coarse_lattice=model.coarse_lattice,
        native_origin_xyz_m=native_grid["origin_xyz_m"],
        native_voxel_size_xyz_m=native_grid["voxel_size_xyz_m"],
    )
    support = completion_support(current_transport, geometry_valid)
    distribution_stats = None
    if bool(collect_distribution_stats):
        natural_mask = support.bool() & formal_valid.bool()
        natural_labels = future_semantic.masked_select(natural_mask).long()
        natural_class_counts = torch.bincount(
            natural_labels, minlength=18
        )
        natural_positive = natural_mask & (future_semantic != FREE_LABEL)
        windows_with_positive = natural_positive.flatten(1).any(dim=1)
        distribution_stats = {
            "natural_class_counts": natural_class_counts,
            "natural_windows": torch.as_tensor(
                len(prepared_windows), device=support.device, dtype=torch.int64
            ),
            "natural_windows_without_positive": (~windows_with_positive).sum(
                dtype=torch.int64
            ),
        }
    tiles = sample_training_tiles(
        support,
        future_semantic,
        formal_valid,
        draws_per_horizon=draws_per_horizon,
        positive_draws=positive_draws,
        generator=generator,
    )
    queries, runtime_report = model.prepare_runtime_queries(
        current_transport,
        future_ego_to_t0,
        native_origin_xyz_m=native_grid["origin_xyz_m"],
        native_voxel_size_xyz_m=native_grid["voxel_size_xyz_m"],
        tiles=tiles,
        dense_geometry_valid=geometry_valid,
        dense_completion_support=support,
        materialize_report=False,
    )
    unique_queries, inverse = deduplicate_completion_queries(queries)
    unique_logits, scatter_report = model.decode_completion(
        first_stage["history"],
        first_stage["sources"],
        first_stage["fusion"],
        condition,
        unique_queries,
        core_only=True,
        materialize_report=False,
    )
    unique_core_logits: list[torch.Tensor] = []
    unique_targets: list[torch.Tensor] = []
    unique_masks: list[torch.Tensor] = []
    for value, query in zip(unique_logits, unique_queries):
        tile = query.tile
        expected = (*tile.core_shape_xyz, 18)
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"core-only completion logits have shape {tuple(value.shape)}, "
                f"expected {expected}"
            )
        unique_core_logits.append(value)
        start, size = tile.core_start_xyz, tile.core_shape_xyz
        sl = tuple(slice(start[d], start[d] + size[d]) for d in range(3))
        unique_targets.append(
            future_semantic[(tile.window_index, tile.horizon, *sl)]
        )
        unique_masks.append(
            query.support[tile.core_slice_xyz]
            & formal_valid[(tile.window_index, tile.horizon, *sl)]
        )
    core_logits = [unique_core_logits[index] for index in inverse]
    targets = [unique_targets[index] for index in inverse]
    masks = [unique_masks[index] for index in inverse]
    batch_size = len(prepared_windows)
    tiles_by_window = [0] * batch_size
    unique_tiles_by_window = [0] * batch_size
    for tile in tiles:
        tiles_by_window[int(tile.window_index)] += 1
    for query in unique_queries:
        unique_tiles_by_window[int(query.tile.window_index)] += 1
    return core_logits, targets, masks, {
        "tiles": len(tiles),
        "unique_tiles": len(unique_queries),
        "duplicate_tiles_saved": len(tiles) - len(unique_queries),
        "window_indices": [int(query.tile.window_index) for query in queries],
        "tiles_by_window": tiles_by_window,
        "unique_tiles_by_window": unique_tiles_by_window,
        "runtime": runtime_report,
        "scatter": scatter_report,
        "distribution": distribution_stats,
    }


def full_completion_predictions_at_free_logit_offsets(
    model,
    prepared: PreparedUnifiedWindow,
    first_stage: dict,
    current_transport: torch.Tensor,
    *,
    native_grid: dict,
    free_logit_offsets: Sequence[float],
    ablate_source_latents: bool = False,
) -> tuple[dict[float, torch.Tensor], dict]:
    offsets = tuple(dict.fromkeys(float(value) for value in free_logit_offsets))
    if not offsets:
        raise ValueError("free_logit_offsets cannot be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in offsets):
        raise ValueError("free_logit_offsets must be finite and non-negative")
    geometry_valid, support, condition = geometry_and_support(
        model, prepared, current_transport, native_grid
    )
    queries, runtime_report = model.prepare_runtime_queries(
        current_transport,
        prepared.history.future_ego_to_t0,
        native_origin_xyz_m=native_grid["origin_xyz_m"],
        native_voxel_size_xyz_m=native_grid["voxel_size_xyz_m"],
        dense_geometry_valid=geometry_valid,
        dense_completion_support=support,
    )
    completion_sources = first_stage["sources"]
    completion_fusion = first_stage["fusion"]
    if ablate_source_latents:
        completion_sources = replace(
            completion_sources,
            history_source_context=torch.zeros_like(
                completion_sources.history_source_context
            ),
        )
        completion_fusion = replace(
            completion_fusion,
            shared_queries=torch.zeros_like(completion_fusion.shared_queries),
            adapter_delta=torch.zeros_like(completion_fusion.adapter_delta),
        )
    future_features, scatter_report = model.build_future_features(
        first_stage["history"],
        completion_sources,
        completion_fusion,
        condition,
    )
    final_by_offset = {offset: current_transport.clone() for offset in offsets}
    chunk = max(int(model.config.runtime_query_chunk), 1)
    for start in range(0, len(queries), chunk):
        q = queries[start : start + chunk]
        logits = model.decode_completion_from_features(
            first_stage["history"], future_features, q
        )
        final_by_offset = compose_completion_tiles_at_free_logit_offsets(
            final_by_offset, logits, q
        )
    return final_by_offset, {
        "runtime": runtime_report,
        "scatter": scatter_report,
        "support": support,
    }


def full_completion_prediction(
    model,
    prepared: PreparedUnifiedWindow,
    first_stage: dict,
    current_transport: torch.Tensor,
    *,
    native_grid: dict,
    ablate_source_latents: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Run the protocol completion path with its frozen zero-margin argmax."""
    final_by_offset, reports = full_completion_predictions_at_free_logit_offsets(
        model,
        prepared,
        first_stage,
        current_transport,
        native_grid=native_grid,
        free_logit_offsets=(0.0,),
        ablate_source_latents=ablate_source_latents,
    )
    return final_by_offset[0.0], reports
