#!/usr/bin/env python3
"""Freeze sample IDs from an SWFM val128 reference cache."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.external_baselines.geniedrive_contract import build_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-cache", "--prepared", dest="reference_cache", required=True,
        help=(
            "SWFM val128 cache directory. Accepts either the full prepared cache "
            "or a compact P0-F9 validation cache with eval payload."
        ),
    )
    parser.add_argument("--output", required=True, help="manifest JSON path")
    parser.add_argument("--expected-count", type=int, default=128)
    args = parser.parse_args()

    reference = Path(args.reference_cache).resolve()
    index_path = reference / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"reference cache index does not exist: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest = build_manifest(index, expected_count=args.expected_count)
    manifest["reference_cache_root"] = str(reference)
    manifest["reference_cache_version"] = index.get("version")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
