#!/usr/bin/env python3
"""Freeze deterministic V21 dev64/dev512 identity manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_motion.v21_source_induction import (
    AnchorLattice,
    POPULATION_PROTOCOL,
    PROTOCOL,
    select_scene_balanced_round_robin,
    stable_json_fingerprint,
)
from tools.real_motion.v20_unified_common import STAGE1_PROTOCOL, load_stage1_rows


def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-cache", required=True,
                   help="Frozen V20 Stage-1 dev512 cache whose key/order defines the parent population")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=64,
                   help="64 for Stage-0 smoke; 0 keeps the complete frozen parent population")
    a = p.parse_args()

    if int(a.count)<0:
        raise ValueError("--count must be zero or positive")
    index_path, index, rows = load_stage1_rows(a.stage1_cache, max_cached_shards=1)
    if index.get("protocol") != STAGE1_PROTOCOL:
        raise RuntimeError(f"expected frozen Stage-1 protocol {STAGE1_PROTOCOL!r}")
    parent = tuple((str(s), str(t)) for s, t in rows.ordered_keys)
    if not parent:
        raise RuntimeError("empty parent Stage-1 population")
    if len(parent)!=len(set(parent)):
        raise RuntimeError("parent Stage-1 population contains duplicate identities")
    selected = (parent if int(a.count)==0 or int(a.count)>=len(parent)
                else select_scene_balanced_round_robin(parent,int(a.count)))
    lattice=AnchorLattice.from_stage1_index(index)
    coarse=dict(index.get("coarse_lattice") or {})
    if tuple(float(x) for x in coarse.get("origin_xyz_m",())[:2])!=lattice.origin_xy_m:
        raise RuntimeError("Stage-1 coarse/highres lattice origins differ")
    if tuple(float(x) for x in coarse.get("voxel_size_xyz_m",())[:2])!=(lattice.anchor_resolution_m,)*2:
        raise RuntimeError("Stage-1 coarse lattice is not the frozen 1.6m V21 anchor lattice")
    if tuple(int(x) for x in coarse.get("shape_xyz",())[:2])!=lattice.anchor_shape_xy:
        raise RuntimeError("Stage-1 coarse lattice shape differs from derived V21 anchor lattice")
    payload = {
        "protocol": POPULATION_PROTOCOL,
        "v21_protocol": PROTOCOL,
        "selection_rule": (
            "parent_order_group_by_scene_first_appearance; within_scene_parent_order; "
            "scene_balanced_round_robin; stop_at_count"
        ),
        "parent_stage1_protocol": index.get("protocol"),
        "parent_stage1_index_sha256":_sha256(index_path),
        "frozen_anchor_lattice":lattice.to_dict(),
        "frozen_anchor_lattice_fingerprint":stable_json_fingerprint(lattice.to_dict()),
        "parent_num_windows": len(parent),
        "parent_keys": [list(x) for x in parent],
        "parent_key_fingerprint": stable_json_fingerprint([list(x) for x in parent]),
        "selected_num_windows": len(selected),
        "selected_keys": [list(x) for x in selected],
        "selected_key_fingerprint": stable_json_fingerprint([list(x) for x in selected]),
    }
    payload["manifest_fingerprint"]=stable_json_fingerprint(payload)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(path.resolve()),
        "selected_num_windows": len(selected),
        "parent_key_fingerprint": payload["parent_key_fingerprint"],
        "selected_key_fingerprint": payload["selected_key_fingerprint"],
    }, indent=2))


if __name__ == "__main__":
    main()
