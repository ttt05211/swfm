#!/usr/bin/env python3
"""Train the v13 displacement-preserving motion-transport head.

The architecture/loss loop is intentionally identical to v12; only the target
contract and checkpoint protocol change.  This keeps the experiment causal and
single-variable while preventing accidental loading of discarded v12 caches or
checkpoints.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from real_motion.motion_transport_v2 import MOTION_TRANSPORT_CACHE_VERSION, TARGET_CONTRACT
from tools.real_motion import train_p0_f9_motion_transport as legacy

PROTOCOL = "p0_f9_v13_learned_motion_transport_v2"


def main():
    legacy.MOTION_TRANSPORT_CACHE_VERSION = MOTION_TRANSPORT_CACHE_VERSION
    legacy.PROTOCOL = PROTOCOL
    original_load_cache = legacy.load_cache

    def load_cache_v2(path: str):
        meta, records = original_load_cache(path)
        if meta.get("target_contract") != TARGET_CONTRACT:
            raise RuntimeError("motion cache target contract mismatch")
        return meta, records

    legacy.load_cache = load_cache_v2
    legacy.main()


if __name__ == "__main__":
    main()
