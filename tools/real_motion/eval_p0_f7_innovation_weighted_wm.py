#!/usr/bin/env python3
"""Evaluate P0-F7 with the exact P0-F6 deployment/fusion protocol.

Inference is intentionally unchanged from P0-F6.  This wrapper only authorizes
the P0-F7 checkpoint protocol, delegates the frozen evaluator, then annotates the
written JSON with the P0-F7 training objective metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.real_motion import eval_p0_f6_decoder_aware_wm as base
from tools.real_motion.train_p0_f7_innovation_weighted_wm import F7_PROTOCOL


def _arg_value(flag: str):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        return None
    return sys.argv[i + 1]


def main():
    base.F6_PROTOCOL = F7_PROTOCOL
    base.main()

    out = _arg_value("--output")
    ckpt = _arg_value("--sparse-ckpt")
    if not out or not ckpt:
        return
    path = Path(out)
    if not path.is_file():
        return

    import torch

    report = json.loads(path.read_text(encoding="utf-8"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("architecture", {})
    protocol = report.setdefault("protocol", {})
    protocol["name"] = "p0_f7_innovation_weighted_anchor_wm_eval_v1"
    protocol["training_objective"] = (
        "soft innovation-energy weighted FM MSE + gradient-calibrated "
        "decoder-aware 9-way dynamic repair CE"
    )
    protocol["innovation_weight"] = arch.get("innovation_weight")
    protocol["optimizer_groups"] = arch.get("optimizer_groups")
    protocol["train_windows"] = arch.get("train_windows")
    protocol["train_unique_scenes"] = arch.get("train_unique_scenes")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
