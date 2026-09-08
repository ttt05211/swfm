#!/usr/bin/env python3
"""Evaluate v15 M/MC checkpoints with the frozen v13 rigid-transport geometry."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.real_motion import eval_p0_f9_learned_motion_transport_v2 as legacy

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
    legacy.PROTOCOL = f"{protocol}_rigid_transport_eval_v1"
    legacy.main()


if __name__ == "__main__":
    main()
