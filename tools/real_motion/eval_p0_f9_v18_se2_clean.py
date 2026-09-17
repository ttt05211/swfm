#!/usr/bin/env python3
"""Hard A1 evaluator entrypoint for clean one-stage V18 checkpoints.

The metric/rendering implementation is intentionally identical to
``eval_p0_f9_v18_se2.py``.  Only the accepted checkpoint protocol differs, so a
clean-training comparison cannot silently change evaluation code.
"""
from __future__ import annotations

from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL


if __name__ == "__main__":
    base.PROTOCOL = CLEAN_PROTOCOL
    base.main()
