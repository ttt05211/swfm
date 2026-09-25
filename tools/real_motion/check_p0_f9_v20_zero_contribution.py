#!/usr/bin/env python3
"""Real-data smoke: fresh V20 must be elementwise identical to frozen V18."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v20_history_world import CanonicalLattice, assert_zero_contribution_identity, protected_add_only
from real_motion.v20_pipeline import run_v20_modules
from real_motion.v20_scene_model import V20HistoryWorldModel, V20SceneConfig
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _model_forward,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import (
    CachedSource,
    _ComponentLRU,
    _prepare_record_from_raw,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

PROTOCOL = "p0_f9_v20_zero_contribution_real_data_v1"


def _lattices(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    lc = cfg["canonical_lattice"]
    if not bool(lc.get("extent_scan_complete", False)):
        raise RuntimeError("zero-contribution smoke requires frozen Omega-max")
    high = CanonicalLattice(
        tuple(float(x) for x in lc["origin_xyz_m"]),
        tuple(float(x) for x in lc["voxel_size_xyz_m"]),
        tuple(int(x) for x in lc["shape_xyz"]),
    )
    fxy = int(cfg["scene_encoder"].get("downsample_xy", 4))
    fz = int(cfg["scene_encoder"].get("downsample_z", fxy))
    hs = np.asarray(high.shape_xyz, dtype=np.int64)
    coarse = CanonicalLattice(
        high.origin_xyz_m,
        (
            high.voxel_size_xyz_m[0] * fxy,
            high.voxel_size_xyz_m[1] * fxy,
            high.voxel_size_xyz_m[2] * fz,
        ),
        (
            int(math.ceil(hs[0] / fxy)),
            int(math.ceil(hs[1] / fxy)),
            int(math.ceil(hs[2] / fz)),
        ),
    )
    return high, coarse


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--v20-config", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--max-windows", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    if int(a.max_windows) <= 0:
        raise ValueError("--max-windows must be positive")
    high, coarse = _lattices(a.v20_config)
    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    _, records = base.load_cache(a.val_cache)
    records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("empty zero-contribution population")
    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    _, v18, _ = full._load_model(a.base_checkpoint, CLEAN_PROTOCOL, device)
    v18.eval()
    model = V20HistoryWorldModel(
        V20SceneConfig(source_dim=int(v18.config.d_model))
    ).to(device).eval()

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = _ComponentLRU(maxsize=128)
    rows = []
    for rec in records:
        w = window_from_record(rec)
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=False)
        state = _prepare_record_from_raw(
            rec, raw, source, pcfg, strong_cfg, device, component_cache
        )
        _stage_gpu_inputs(state, device)
        try:
            with torch.inference_mode():
                v18_out = _model_forward(
                    v18, state["gpu"], device, return_latents=True
                )
            pred = np.asarray(
                _forecast_once(
                    v18, state, pcfg, strong_cfg, device,
                    precomputed_out=v18_out,
                ),
                dtype=np.uint8,
            )
        finally:
            _release_gpu_inputs(state)

        vp = run_v20_modules(
            model,
            v18,
            rec,
            raw,
            pcfg=pcfg,
            strong_cfg=strong_cfg,
            high_lattice=high,
            coarse_lattice=coarse,
            tile_size_xyz=(32, 32, 16),
            device=device,
            amp=amp,
            v18_output_with_latents=v18_out,
            enable_static=True,
            enable_dormant=True,
            enable_birth=True,
        )
        final = protected_add_only(
            torch.from_numpy(pred),
            dormant=torch.from_numpy(vp.dormant_future),
            birth=torch.from_numpy(vp.birth_future),
            static_world=torch.from_numpy(vp.static_future),
            free_label=int(pcfg.free_label),
        )
        assert_zero_contribution_identity(torch.from_numpy(pred), final)
        branch_nonfree = {
            "static": int((vp.static_future != int(pcfg.free_label)).sum()),
            "dormant": int((vp.dormant_future != int(pcfg.free_label)).sum()),
            "birth": int((vp.birth_future != int(pcfg.free_label)).sum()),
        }
        if any(branch_nonfree.values()):
            raise AssertionError(
                f"fresh V20 emitted non-free voxels for {w.t0_token}: "
                f"{branch_nonfree}"
            )
        rows.append({
            "scene": str(w.scene_name),
            "t0_token": str(w.t0_token),
            "elementwise_equal": True,
            "branch_nonfree_voxels": branch_nonfree,
        })

    result = {
        "protocol": PROTOCOL,
        "windows": len(rows),
        "base_checkpoint": str(Path(a.base_checkpoint).resolve()),
        "v20_config": str(Path(a.v20_config).resolve()),
        "all_elementwise_equal": True,
        "rows": rows,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
