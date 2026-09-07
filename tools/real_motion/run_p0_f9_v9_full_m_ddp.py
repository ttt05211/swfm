#!/usr/bin/env python3
"""Stable torchrun launcher for the v9 full-M DDP implementation.

The DDP implementation intentionally stores its own training protocol in the
checkpoint architecture.  The reused v9 resume validator keys off the module
``PROTOCOL`` constant, so set that constant to the distributed protocol before
entering the implementation.  This keeps fresh and resumed DDP artifacts under
one exact protocol without changing the legacy single-GPU trainer.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.real_motion import train_p0_f9_v9_full_m_ddp as impl


impl.base.PROTOCOL = impl.PROTOCOL


if __name__ == "__main__":
    impl.main()
