"""One-window V20 inference pipeline shared by evaluation and benchmarking."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from .v19_scene_memory import build_dynamic_source_memory, prepare_causal_arrays_from_tracks
from .v20_birth import BirthRenderReport, render_birth_queries
from .v20_dormant import DormantRenderReport, render_dormant_sources, split_current_and_dormant_tracks
from .v20_history_world import CanonicalLattice, align_history_once_to_canonical, poses_to_t0_canonical
from .v20_runtime import (
    StaticRuntimeReport,
    decode_static_world_tiled,
    sample_scene_features_at_t0_points,
    v18_birth_condition_from_output,
    v18_source_tokens_from_arrays,
)


@dataclass
class V20WindowPrediction:
    static_future: np.ndarray
    dormant_future: np.ndarray
    birth_future: np.ndarray
    static_report: StaticRuntimeReport | None
    dormant_report: DormantRenderReport | None
    birth_report: BirthRenderReport | None
    birth_outputs: dict[str, torch.Tensor] | None
    dormant_tracks: int
    timings_ms: dict[str, float]
    history_out_of_bounds_observed_samples: int


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _free_future(pcfg) -> np.ndarray:
    return np.full(
        (6,) + tuple(int(x) for x in pcfg.grid.shape_hwd),
        int(pcfg.free_label),
        dtype=np.uint8,
    )


def run_v20_modules(
    model,
    v18,
    record: Mapping[str, object],
    raw: Mapping[str, object],
    *,
    pcfg,
    strong_cfg,
    high_lattice: CanonicalLattice,
    coarse_lattice: CanonicalLattice,
    tile_size_xyz: Sequence[int],
    device: torch.device,
    amp: bool,
    v18_output_with_latents: Mapping[str, torch.Tensor],
    enable_static: bool = True,
    enable_dormant: bool = True,
    enable_birth: bool = True,
    dormant_existence_threshold: float = 0.5,
    birth_existence_threshold: float = 0.5,
    birth_shape_threshold: float = 0.5,
    birth_duplicate_distance_m: float = 2.0,
) -> V20WindowPrediction:
    """Run V20 additions once; future GT never enters this function."""
    timing: dict[str, float] = {}
    history_sem = np.asarray(raw["history_occ"], dtype=np.uint8)
    history_obs = np.asarray(raw["history_observed"], dtype=bool)
    history_pose = np.asarray(raw["history_poses"], dtype=np.float64)
    future_pose = np.asarray(raw["future_poses"], dtype=np.float64)
    t0 = history_pose[-1]
    future_rel = poses_to_t0_canonical(future_pose, t0)

    t = time.perf_counter()
    aligned = align_history_once_to_canonical(
        coarse_lattice,
        history_semantic=history_sem,
        history_observed=history_obs,
        history_ego_to_world=history_pose,
        t0_ego_to_world=t0,
        native_origin_xyz_m=(pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
        native_voxel_size_xyz_m=pcfg.grid.voxel_size,
        free_label=int(pcfg.free_label),
    )
    timing["history_alignment"] = (time.perf_counter() - t) * 1000.0

    sem = torch.from_numpy(aligned.semantic).to(device).unsqueeze(0)
    obs = torch.from_numpy(aligned.observed).to(device).unsqueeze(0)
    obsfree = torch.from_numpy(aligned.observed_free).to(device).unsqueeze(0)
    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if amp and device.type == "cuda" else nullcontext()
    )
    _sync(device)
    t = time.perf_counter()
    with torch.inference_mode(), ctx:
        scene = model.encode_history(sem, obs, obsfree)
    _sync(device)
    timing["v20_encoder"] = (time.perf_counter() - t) * 1000.0

    static_report = None
    static_future = _free_future(pcfg)
    if enable_static:
        _sync(device)
        t = time.perf_counter()
        with torch.inference_mode(), (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp and device.type == "cuda" else nullcontext()
        ):
            static_report = decode_static_world_tiled(
                model,
                scene,
                high_lattice=high_lattice,
                coarse_lattice=coarse_lattice,
                future_ego_to_canonical=future_rel,
                history_observed_coarse=aligned.observed,
                native_shape_xyz=pcfg.grid.shape_hwd,
                native_origin_xyz_m=(pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
                native_voxel_size_xyz_m=pcfg.grid.voxel_size,
                tile_size_xyz=tile_size_xyz,
                free_label=int(pcfg.free_label),
            )
        _sync(device)
        timing["static_network_and_render"] = (time.perf_counter() - t) * 1000.0
        static_future = static_report.future_semantic

    dormant_report = None
    dormant_future = _free_future(pcfg)
    dormant_n = 0
    if enable_dormant:
        t = time.perf_counter()
        source_sem = [
            np.where(history_obs[i], history_sem[i], int(pcfg.free_label)).astype(np.uint8)
            for i in range(6)
        ]
        tracks, _ = build_dynamic_source_memory(
            source_sem,
            history_pose,
            grid=pcfg.grid,
            strong_cfg=strong_cfg,
            frame_dt_s=float(pcfg.frame_dt_s),
        )
        _, dormant = split_current_and_dormant_tracks(tracks)
        dormant = list(dormant)
        dormant_n = len(dormant)
        timing["dormant_prepare"] = (time.perf_counter() - t) * 1000.0
        if dormant:
            arrays = prepare_causal_arrays_from_tracks(
                dormant,
                source_sem,
                history_pose,
                grid=pcfg.grid,
                free_label=int(pcfg.free_label),
                frame_dt_s=float(pcfg.frame_dt_s),
            )
            inv_t0 = np.linalg.inv(t0)
            pts = []
            for tr in dormant:
                pw = tr.anchor_center_world(float(pcfg.frame_dt_s))
                pts.append((inv_t0 @ np.r_[pw, 1.0])[:3])
            p = torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=device)
            _sync(device)
            t = time.perf_counter()
            source_tok = v18_source_tokens_from_arrays(
                v18, arrays, device=device, amp=amp
            )
            local = sample_scene_features_at_t0_points(
                scene, p, coarse_lattice
            )
            with torch.inference_mode(), (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if amp and device.type == "cuda" else nullcontext()
            ):
                dout = model.dormant(source_tok, local)
            _sync(device)
            timing["dormant_network"] = (time.perf_counter() - t) * 1000.0
            t = time.perf_counter()
            dormant_report = render_dormant_sources(
                dout,
                dormant,
                kta_displacement_xy_m=arrays["kta_displacement_xy_m"],
                t0_ego_to_world=t0,
                future_ego_to_world=future_pose,
                grid=pcfg.grid,
                frame_dt_s=float(pcfg.frame_dt_s),
                existence_threshold=float(dormant_existence_threshold),
                free_label=int(pcfg.free_label),
            )
            timing["dormant_render"] = (time.perf_counter() - t) * 1000.0
            dormant_future = dormant_report.future_semantic
        else:
            timing["dormant_network"] = 0.0
            timing["dormant_render"] = 0.0

    birth_report = None
    birth_outputs = None
    birth_future = _free_future(pcfg)
    if enable_birth:
        cond = v18_birth_condition_from_output(
            record,
            v18_output_with_latents,
            device=device,
        )
        _sync(device)
        t = time.perf_counter()
        with torch.inference_mode(), (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp and device.type == "cuda" else nullcontext()
        ):
            bout = model.birth(
                scene,
                current_source_tokens=cond.current_source_tokens,
                current_source_xyz_norm=cond.current_source_xyz_norm,
                future_source_tokens=cond.future_source_tokens,
                future_source_xyz_norm=cond.future_source_xyz_norm,
            )
        _sync(device)
        timing["birth_network"] = (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        birth_report = render_birth_queries(
            bout,
            future_ego_to_canonical=future_rel,
            native_shape_xyz=pcfg.grid.shape_hwd,
            native_origin_xyz_m=(pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
            native_voxel_size_xyz_m=pcfg.grid.voxel_size,
            shape_voxel_size_m=float(model.cfg.birth_shape_voxel_size_m),
            existence_threshold=float(birth_existence_threshold),
            shape_threshold=float(birth_shape_threshold),
            free_label=int(pcfg.free_label),
            current_source_future_xy_t0_m=cond.future_source_xy_t0_m,
            current_source_existence_prob=cond.future_source_existence_prob,
            current_source_class_id=cond.current_source_class_id,
            duplicate_distance_m=float(birth_duplicate_distance_m),
        )
        timing["birth_render"] = (time.perf_counter() - t) * 1000.0
        birth_future = birth_report.future_semantic
        birth_outputs = {k: v.detach().cpu() for k, v in bout.items()}

    return V20WindowPrediction(
        static_future=static_future,
        dormant_future=dormant_future,
        birth_future=birth_future,
        static_report=static_report,
        dormant_report=dormant_report,
        birth_report=birth_report,
        birth_outputs=birth_outputs,
        dormant_tracks=int(dormant_n),
        timings_ms=timing,
        history_out_of_bounds_observed_samples=int(aligned.out_of_bounds_samples),
    )
