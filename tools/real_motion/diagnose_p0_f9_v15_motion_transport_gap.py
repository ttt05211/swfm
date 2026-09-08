#!/usr/bin/env python3
"""Run the frozen v14 gap diagnostic on a v15 M/MC checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.real_motion import diagnose_p0_f9_v14_motion_transport_gap as legacy

ALLOWED = {
    "p0_f9_v15_motion_weighted_m",
    "p0_f9_v15_motion_class_balanced_mc",
}


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--checkpoint", required=True)
    known, _ = p.parse_known_args()
    ck = torch.load(known.checkpoint, map_location="cpu", weights_only=False)
    protocol = str(ck.get("protocol", ""))
    if protocol not in ALLOWED:
        raise RuntimeError(f"not a v15 M/MC checkpoint: {protocol}")
    legacy.CHECKPOINT_PROTOCOL = protocol
    legacy.PROTOCOL = f"{protocol}_motion_gap_diagnostic_v1"
    legacy.main()


if __name__ == "__main__":
    main()
