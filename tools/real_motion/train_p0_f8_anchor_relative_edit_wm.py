#!/usr/bin/env python3
"""Public P0-F8 training entrypoint."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

from tools.real_motion.p0_f8_train_impl_v2 import F8_PROTOCOL, main

__all__ = ["F8_PROTOCOL", "main"]


if __name__ == "__main__":
    main()
