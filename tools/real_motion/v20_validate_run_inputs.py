#!/usr/bin/env python3
"""Validate the V20 config/cache/checkpoint chain before GPU experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from real_motion.v20_training import CHECKPOINT_PROTOCOL
from real_motion.v20_static_repair import TRAIN_PROTOCOL as STATIC_REPAIR_PROTOCOL
from tools.real_motion.build_p0_f9_v20_history_cache import PROTOCOL as STAGE1_PROTOCOL


def _cache(path):
    root = Path(path)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != STAGE1_PROTOCOL:
        raise RuntimeError(f"not a V20 Stage-1 cache: {path}")
    if int(idx.get("query_out_of_bounds_voxels", -1)) != 0:
        raise RuntimeError(f"Stage-1 future-query OOB is non-zero: {path}")
    if int(idx.get("history_out_of_bounds_observed_samples", -1)) != 0:
        raise RuntimeError(f"Stage-1 history observed OOB is non-zero: {path}")
    return idx


def _checkpoint(path, stage):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("protocol") != CHECKPOINT_PROTOCOL:
        raise RuntimeError(f"unexpected V20 checkpoint protocol: {path}")
    if str(obj.get("stage")) != str(stage):
        raise RuntimeError(
            f"{path}: expected stage={stage}, got {obj.get('stage')}"
        )
    return obj


def _norm_path(x):
    return str(Path(str(x)).expanduser().resolve())


def _same_lattice(a, b, name):
    if a != b:
        raise RuntimeError(f"{name} lattice mismatch")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v20-config", required=True)
    p.add_argument("--train-stage1-cache", required=True)
    p.add_argument("--dev-stage1-cache", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--static-checkpoint", default="")
    p.add_argument("--dormant-checkpoint", default="")
    p.add_argument("--birth-checkpoint", default="")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.v20_config).read_text(encoding="utf-8"))
    lc = cfg["canonical_lattice"]
    if not bool(lc.get("extent_scan_complete", False)):
        raise RuntimeError("V20 config Omega-max is not frozen")
    for key in (
        "extent_train_future_oob_voxels",
        "extent_dev_future_oob_voxels",
        "extent_train_history_oob_voxels",
        "extent_dev_history_oob_voxels",
    ):
        if int(lc.get(key, -1)) != 0:
            raise RuntimeError(f"V20 frozen config lacks zero-OOB proof: {key}")

    tr = _cache(a.train_stage1_cache)
    dv = _cache(a.dev_stage1_cache)
    overlap = set(tr["scene_names"]) & set(dv["scene_names"])
    if overlap:
        raise RuntimeError(f"train/dev scene overlap: {sorted(overlap)[:5]}")
    _same_lattice(tr["highres_lattice"], dv["highres_lattice"], "train/dev highres")
    _same_lattice(tr["coarse_lattice"], dv["coarse_lattice"], "train/dev coarse")

    expected_high = {
        "origin_xyz_m": [float(x) for x in lc["origin_xyz_m"]],
        "voxel_size_xyz_m": [float(x) for x in lc["voxel_size_xyz_m"]],
        "shape_xyz": [int(x) for x in lc["shape_xyz"]],
    }
    _same_lattice(tr["highres_lattice"], expected_high, "config/cache highres")

    base_path = _norm_path(a.base_checkpoint)
    chain = {}
    static = dormant = birth = None
    if a.static_checkpoint:
        static = _checkpoint(a.static_checkpoint, "static")
        if _norm_path(static["v18_checkpoint"]) != base_path:
            raise RuntimeError("Static checkpoint uses a different V18 checkpoint")
        sx = dict(static.get("extra") or {})
        if sx.get("train_protocol") != STATIC_REPAIR_PROTOCOL:
            raise RuntimeError(
                "Static checkpoint is not the corrected Stage-2 Repair v2 protocol"
            )
        if bool(sx.get("overfit_diagnostic_only", False)):
            raise RuntimeError(
                "overfit diagnostic Static checkpoint cannot enter Stage-3/4/5 chain"
            )
        _same_lattice(sx["highres_lattice"], tr["highres_lattice"], "Static/cache highres")
        _same_lattice(sx["coarse_lattice"], tr["coarse_lattice"], "Static/cache coarse")
        chain["static"] = _norm_path(a.static_checkpoint)

    if a.dormant_checkpoint:
        if static is None:
            raise RuntimeError("--dormant-checkpoint requires --static-checkpoint")
        dormant = _checkpoint(a.dormant_checkpoint, "dormant")
        if _norm_path(dormant["v18_checkpoint"]) != base_path:
            raise RuntimeError("Dormant checkpoint uses a different V18 checkpoint")
        dx = dict(dormant.get("extra") or {})
        if Path(str(dx.get("parent_static_checkpoint", ""))).name != Path(a.static_checkpoint).name:
            raise RuntimeError("Dormant parent Static checkpoint mismatch")
        _same_lattice(dx["highres_lattice"], tr["highres_lattice"], "Dormant/cache highres")
        _same_lattice(dx["coarse_lattice"], tr["coarse_lattice"], "Dormant/cache coarse")
        chain["dormant"] = _norm_path(a.dormant_checkpoint)

    if a.birth_checkpoint:
        if dormant is None:
            raise RuntimeError("--birth-checkpoint requires --dormant-checkpoint")
        birth = _checkpoint(a.birth_checkpoint, "birth")
        if _norm_path(birth["v18_checkpoint"]) != base_path:
            raise RuntimeError("Birth checkpoint uses a different V18 checkpoint")
        bx = dict(birth.get("extra") or {})
        if Path(str(bx.get("parent_dormant_checkpoint", ""))).name != Path(a.dormant_checkpoint).name:
            raise RuntimeError("Birth parent Dormant checkpoint mismatch")
        _same_lattice(bx["highres_lattice"], tr["highres_lattice"], "Birth/cache highres")
        _same_lattice(bx["coarse_lattice"], tr["coarse_lattice"], "Birth/cache coarse")
        chain["birth"] = _norm_path(a.birth_checkpoint)

    report = {
        "ok": True,
        "extent_scan_complete": True,
        "extent_contract": lc.get("extent_contract"),
        "train_windows": int(tr["num_windows"]),
        "train_scenes": int(tr["num_scenes"]),
        "dev_windows": int(dv["num_windows"]),
        "dev_scenes": int(dv["num_scenes"]),
        "scene_overlap": 0,
        "base_checkpoint": base_path,
        "checkpoint_chain": chain,
        "highres_lattice": tr["highres_lattice"],
        "coarse_lattice": tr["coarse_lattice"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
