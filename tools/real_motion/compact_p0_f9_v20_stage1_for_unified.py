#!/usr/bin/env python3
"""Strip legacy future static supervision from a Stage-1 cache for unified V20."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

PROTOCOL = "p0_f9_v20_stage1_history_cache_v2"


def _diagnostic_counts(names) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        key = str(name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    src = Path(a.input_dir)
    dst = Path(a.output_dir)
    index = json.loads((src / "index.json").read_text(encoding="utf-8"))
    if index.get("protocol") != PROTOCOL:
        raise RuntimeError(f"unexpected protocol: {index.get('protocol')}")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    before = after = 0
    dynamic_diagnostics_complete = True
    shards = []
    started = time.perf_counter()
    for si, meta in enumerate(index["shards"]):
        file = str(meta["file"])
        obj = torch.load(src / file, map_location="cpu", weights_only=False)
        if obj.get("protocol") != PROTOCOL:
            raise RuntimeError(f"bad shard: {file}")
        rows = []
        for row in obj["rows"]:
            slim = dict(row)
            slim.pop("static_supervision", None)
            dynamic = slim.pop("dynamic_supervision", None)
            if dynamic is not None:
                # Preserve only tiny evaluation-only responsibility counts.
                # Trajectories, identities and box metadata are not model inputs.
                slim["dynamic_diagnostic_counts"] = _diagnostic_counts(
                    item.get("responsibility_name", "IGNORE")
                    for item in dynamic
                )
            elif "dynamic_diagnostic_groups" in slim:
                slim["dynamic_diagnostic_counts"] = _diagnostic_counts(
                    slim.pop("dynamic_diagnostic_groups")
                )
            elif "dynamic_diagnostic_counts" not in slim:
                dynamic_diagnostics_complete = False
            rows.append(slim)
        before += int((src / file).stat().st_size)
        torch.save({"protocol": PROTOCOL, "rows": rows}, dst / file)
        nbytes = int((dst / file).stat().st_size)
        after += nbytes
        shards.append({
            "file": file,
            "count": len(rows),
            "bytes": nbytes,
            "sha256": _file_sha256(dst / file),
            "keys": [
                [str(row["scene_name"]), str(row["t0_token"])]
                for row in rows
            ],
        })
        if si == 0 or (si + 1) % 50 == 0 or si + 1 == len(index["shards"]):
            print(f"compact_stage1 {si+1}/{len(index['shards'])}", flush=True)

    out = dict(index)
    out["consumer_profile"] = "unified_transport_completion"
    out["unified_compact"] = True
    out["dynamic_supervision_stored"] = False
    out["dynamic_diagnostic_groups_stored"] = bool(
        dynamic_diagnostics_complete
    )
    out["dynamic_diagnostic_counts_stored"] = bool(
        dynamic_diagnostics_complete
    )
    layout = dict(out.get("cache_layout") or {})
    layout["static_supervision"] = (
        "omitted by compact conversion; unified reads future GT at runtime"
    )
    layout["dynamic_supervision"] = (
        "identity/trajectory metadata omitted; responsibility-count-only diagnostics retained"
        if dynamic_diagnostics_complete
        else "omitted; dynamic responsibility diagnostics unavailable"
    )
    out["cache_layout"] = layout
    out["compressed_static_semantic_values"] = 0
    out["shards"] = shards
    out["compaction"] = {
        "source": str(src.resolve()),
        "source_bytes": before,
        "compact_bytes": after,
        "bytes_saved": before - after,
        "compression_ratio": after / max(before, 1),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (dst / "index.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out["compaction"], indent=2), flush=True)


if __name__ == "__main__":
    main()
