#!/usr/bin/env python3
"""Build the audited full native cache and annotate zero-route sample IDs.

The scientific cache tensors are produced by ``build_p0_f9_full_native_cache``
unchanged.  After that builder finalizes ``index.json``, this wrapper scans the
already-written cache once and adds only routing metadata needed to guarantee
that every DDP rank receives at least one valid Top-2 window per optimizer step.
No latent, route, target, or sample ordering is modified.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from tools.real_motion import build_p0_f9_full_native_cache as base


def _output_arg(argv) -> Path:
    for i, token in enumerate(argv):
        if token == "--output" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser().resolve()
        if token.startswith("--output="):
            return Path(token.split("=", 1)[1]).expanduser().resolve()
    raise RuntimeError("--output is required")


def annotate_zero_routes(root: Path) -> dict:
    ds = MSPWorldModelCacheDataset(root)
    zero_ids = []
    for i, entry in enumerate(ds.entries):
        sample = ds[i]
        if not bool(sample["window_valid"].bool().any()):
            zero_ids.append(str(entry["sample_id"]))
        if i == 0 or (i + 1) % 2048 == 0 or i + 1 == len(ds):
            print(f"zero-route scan {i + 1}/{len(ds)}")

    index_path = root / "index.json"
    obj = json.loads(index_path.read_text(encoding="utf-8"))
    meta = obj.setdefault("metadata", {})
    meta["zero_route_sample_ids"] = zero_ids
    meta["zero_route_count"] = len(zero_ids)
    meta["zero_route_fraction"] = len(zero_ids) / max(len(ds), 1)
    meta["ddp_route_safety_annotation"] = "exact_window_valid_any_scan_v1"
    tmp = index_path.with_name(index_path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(index_path)
    report = {
        "output": str(root),
        "num_samples": len(ds),
        "zero_route_count": len(zero_ids),
        "zero_route_fraction": meta["zero_route_fraction"],
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    output = _output_arg(sys.argv[1:])
    base.main()
    annotate_zero_routes(output)


if __name__ == "__main__":
    main()
