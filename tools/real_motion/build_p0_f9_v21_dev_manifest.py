#!/usr/bin/env python3
"""Freeze deterministic V21 dev64/dev512 identity manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_motion.v21_source_induction import (
    PROTOCOL,
    select_scene_balanced_round_robin,
    stable_json_fingerprint,
)
from tools.real_motion.v20_unified_common import load_stage1_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-cache", required=True,
                   help="Frozen V20 Stage-1 dev512 cache whose key/order defines the parent population")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=64,
                   help="64 for Stage-0 smoke; 0 keeps the complete frozen parent population")
    a = p.parse_args()

    _, index, rows = load_stage1_rows(a.stage1_cache, max_cached_shards=1)
    parent = tuple((str(s), str(t)) for s, t in rows.ordered_keys)
    if not parent:
        raise RuntimeError("empty parent Stage-1 population")
    selected = parent if int(a.count) == 0 else select_scene_balanced_round_robin(parent, int(a.count))
    payload = {
        "protocol": "p0_f9_v21_population_manifest_v1",
        "v21_protocol": PROTOCOL,
        "selection_rule": (
            "parent_order_group_by_scene_first_appearance; within_scene_parent_order; "
            "scene_balanced_round_robin; stop_at_count"
        ),
        "parent_stage1_protocol": index.get("protocol"),
        "parent_num_windows": len(parent),
        "parent_keys": [list(x) for x in parent],
        "parent_key_fingerprint": stable_json_fingerprint([list(x) for x in parent]),
        "selected_num_windows": len(selected),
        "selected_keys": [list(x) for x in selected],
        "selected_key_fingerprint": stable_json_fingerprint([list(x) for x in selected]),
    }
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
