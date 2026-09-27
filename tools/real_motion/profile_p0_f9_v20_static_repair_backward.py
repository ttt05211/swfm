#!/usr/bin/env python3
"""Operator-level backward profiler for V20 Static Repair.

This wrapper leaves the trainer unchanged.  It temporarily intercepts
Tensor.backward() and profiles only a small configurable number of backward
calls, then delegates all CLI arguments to the normal trainer.

Environment variables:
  V20_PROFILE_BACKWARD_START  1-based backward call to profile (default: 6)
  V20_PROFILE_BACKWARD_COUNT  number of backward calls to profile (default: 2)
  V20_PROFILE_BACKWARD_ROWS   table row limit (default: 30)
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

_START = max(int(os.environ.get("V20_PROFILE_BACKWARD_START", "6")), 1)
_COUNT = max(int(os.environ.get("V20_PROFILE_BACKWARD_COUNT", "2")), 1)
_ROWS = max(int(os.environ.get("V20_PROFILE_BACKWARD_ROWS", "30")), 5)

_ORIGINAL_BACKWARD = torch.Tensor.backward
_backward_calls = 0


def _profiled_backward(self, *args, **kwargs):
    global _backward_calls
    _backward_calls += 1
    active = _START <= _backward_calls < (_START + _COUNT)
    if not active:
        return _ORIGINAL_BACKWARD(self, *args, **kwargs)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        out = _ORIGINAL_BACKWARD(self, *args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(
        f"\n=== V20 Static Repair backward op profile "
        f"call={_backward_calls} ===",
        flush=True,
    )
    try:
        table = prof.key_averages(
            group_by_input_shape=True
        ).table(
            sort_by="self_cuda_time_total",
            row_limit=_ROWS,
        )
    except Exception:
        table = prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=_ROWS,
        )
    print(table, flush=True)
    return out


def main():
    torch.Tensor.backward = _profiled_backward
    try:
        from tools.real_motion.train_p0_f9_v20_static_repair import main as train_main
        train_main()
    finally:
        torch.Tensor.backward = _ORIGINAL_BACKWARD


if __name__ == "__main__":
    main()
