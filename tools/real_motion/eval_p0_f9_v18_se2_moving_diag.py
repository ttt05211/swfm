#!/usr/bin/env python3
"""V18 hard-A1 evaluator with a companion micro Moving-IoU diagnostic.

The base hard renderer, A1 composition, frozen Moving-mIoU v2 support, and all
existing metrics are unchanged.  This entrypoint only adds a second accumulator
on the *same* moving support whose class aggregation is micro rather than macro.
It accepts both historical paired Y checkpoints and clean one-stage checkpoints.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from real_motion.metrics.moving_micro_iou import MovingMicroIoUMultiHorizon
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL
from tools.real_motion.train_p0_f9_v18_se2_pair import PROTOCOL as PAIR_PROTOCOL


EVAL_PROTOCOL = "p0_f9_v18_se2_hard_a1_eval_v3_macro_micro_moving"
SUPPORTED_PROTOCOLS = (PAIR_PROTOCOL, CLEAN_PROTOCOL)


def _checkpoint_arg(argv):
    for i, token in enumerate(argv):
        if token == "--checkpoint" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--checkpoint="):
            return token.split("=", 1)[1]
    raise RuntimeError("--checkpoint is required")


def _install_micro_companion():
    original_new = safe._new_metrics
    original_update = safe._update
    original_report = safe._report

    def new_metrics():
        state = original_new()
        state["moving_micro"] = MovingMicroIoUMultiHorizon()
        return state

    def update(state, horizon, pred, gt, moving_support):
        original_update(state, horizon, pred, gt, moving_support)
        state["moving_micro"].update(horizon, pred, gt, moving_support)

    def report(state):
        out = original_report(state)
        out["moving_micro"] = state["moving_micro"].compute()
        return out

    safe._new_metrics = new_metrics
    safe._update = update
    safe._report = report


def main():
    ckpt_path = _checkpoint_arg(sys.argv[1:])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    protocol = str(ck.get("protocol"))
    if protocol not in SUPPORTED_PROTOCOLS:
        raise RuntimeError(
            f"unsupported checkpoint protocol {protocol!r}; expected one of "
            f"{SUPPORTED_PROTOCOLS}"
        )

    # The base evaluator has one protocol guard. Point it at the actual source
    # checkpoint without changing any rendering/evaluation logic.
    base.PROTOCOL = protocol
    base.EVAL_PROTOCOL = EVAL_PROTOCOL
    _install_micro_companion()
    base.main()


if __name__ == "__main__":
    main()
