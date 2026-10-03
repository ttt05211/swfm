#!/usr/bin/env python3
"""Explicit compile/ABI/basic buffer check. No data or CUDA required."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
from real_motion.native_column_cpu import prepare_native


def main():
    from real_motion import native_column_cpu
    prepare_native()
    kernel = native_column_cpu._loaded
    shape = (3, 4, 5)
    history = (np.arange(60, dtype=np.uint8) % 18).reshape(shape)
    observed = np.ones(shape, bool)
    ijk = np.column_stack(np.unravel_index(np.arange(60), shape)).astype(np.int64)
    values, flags = kernel.gather(ijk, history, observed)
    if not np.array_equal(values, history.ravel()) or not np.all(flags == 1):
        raise RuntimeError('native CPU gather preflight failed')
    if not np.array_equal(kernel.changed(np.array([[0, 0], [0, 1]], np.int64)), [False, True]):
        raise RuntimeError('native CPU labels preflight failed')
    print(json.dumps(dict(ok=True, **kernel.info()), ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
