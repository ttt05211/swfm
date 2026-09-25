#!/usr/bin/env python3
"""Freeze V20 Ωmax only after train and dev OOB audits pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

SCAN_PROTOCOL = "p0_f9_v20_query_extent_scan_v2"
AUDIT_PROTOCOL = "p0_f9_v20_query_extent_audit_v2"


def _load_json(path, protocol):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("protocol") != protocol:
        raise RuntimeError(f"unexpected protocol in {path}: {obj.get('protocol')}")
    return obj


def _same_lattice(scan, audit):
    a = {
        "origin_xyz_m": scan["recommended_origin_xyz_m"],
        "voxel_size_xyz_m": scan["voxel_size_xyz_m"],
        "shape_xyz": scan["recommended_shape_xyz"],
    }
    b = audit["lattice"]
    return (
        [float(x) for x in a["origin_xyz_m"]] == [float(x) for x in b["origin_xyz_m"]]
        and [float(x) for x in a["voxel_size_xyz_m"]] == [float(x) for x in b["voxel_size_xyz_m"]]
        and [int(x) for x in a["shape_xyz"]] == [int(x) for x in b["shape_xyz"]]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v20-config", required=True)
    p.add_argument("--train-scan", required=True)
    p.add_argument("--train-audit", required=True)
    p.add_argument("--dev-audit", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.v20_config).read_text(encoding="utf-8"))
    scan = _load_json(a.train_scan, SCAN_PROTOCOL)
    train = _load_json(a.train_audit, AUDIT_PROTOCOL)
    dev = _load_json(a.dev_audit, AUDIT_PROTOCOL)
    if not _same_lattice(scan, train) or not _same_lattice(scan, dev):
        raise RuntimeError("train/dev audits were not run on the scan-recommended lattice")
    for split_name, report in (("train", train), ("dev", dev)):
        if int(report["future_query"]["out_of_bounds_voxels"]) != 0:
            raise RuntimeError(f"{split_name} future-query OOB is non-zero")
        if int(report["history_view"]["out_of_bounds_voxels"]) != 0:
            raise RuntimeError(f"{split_name} history-view OOB is non-zero")

    lc = cfg["canonical_lattice"]
    lc["origin_xyz_m"] = [float(x) for x in scan["recommended_origin_xyz_m"]]
    lc["voxel_size_xyz_m"] = [float(x) for x in scan["voxel_size_xyz_m"]]
    lc["shape_xyz"] = [int(x) for x in scan["recommended_shape_xyz"]]
    lc["extent_scan_complete"] = True
    lc["extent_scan_windows"] = int(scan["windows"])
    lc["extent_scan_scenes"] = int(scan["scenes"])
    lc["extent_train_future_oob_voxels"] = 0
    lc["extent_dev_future_oob_voxels"] = 0
    lc["extent_train_history_oob_voxels"] = 0
    lc["extent_dev_history_oob_voxels"] = 0
    lc["extent_contract"] = (
        "union_of_6_history_evidence_views_and_6_future_query_views"
    )
    lc["extent_scan_report"] = str(Path(a.train_scan).resolve())
    lc["extent_train_audit_report"] = str(Path(a.train_audit).resolve())
    lc["extent_dev_audit_report"] = str(Path(a.dev_audit).resolve())

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "output": str(op.resolve()),
        "origin_xyz_m": lc["origin_xyz_m"],
        "voxel_size_xyz_m": lc["voxel_size_xyz_m"],
        "shape_xyz": lc["shape_xyz"],
        "extent_scan_complete": True,
    }, indent=2))


if __name__ == "__main__":
    main()
