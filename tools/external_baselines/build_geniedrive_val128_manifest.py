#!/usr/bin/env python3
"""Freeze the current SWFM prepared val128 sample IDs for external baselines."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.external_baselines.geniedrive_contract import build_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, help="SWFM prepared val128 directory")
    parser.add_argument("--output", required=True, help="manifest JSON path")
    parser.add_argument("--expected-count", type=int, default=128)
    args = parser.parse_args()

    prepared = Path(args.prepared).resolve()
    index_path = prepared / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"prepared index does not exist: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest = build_manifest(index, expected_count=args.expected_count)
    manifest["prepared_root"] = str(prepared)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
