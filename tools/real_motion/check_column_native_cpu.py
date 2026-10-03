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
    v, b = kernel.gather_many([ijk, None], np.stack((history, history)),
        np.stack((observed, observed)), [None, None], [None, None], np.array([True, False]), 5)
    if not (np.array_equal(v[:, 0].ravel(), history.ravel()) and np.all(b[:, 0] == 1)
            and np.all(v[:, 1] == 18) and np.all(b[:, 1] == 0)):
        raise RuntimeError('native batch gather preflight failed')
    xy, offsets, bounds = kernel.support_many([np.array([25], np.int64), np.empty(0, np.int64)], shape)
    if not (np.array_equal(xy, [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]])
            and np.array_equal(offsets, [0, 5, 5]) and np.array_equal(bounds, [[0, 0], [5, -1]])):
        raise RuntimeError('native batch support preflight failed')
    free = np.full(shape, 17, np.uint8); owner = np.full(shape, -1, np.int32)
    truth = free.copy(); truth[1, 1, 2] = 11
    args = (np.array([[1, 1]], np.int32), np.array([0], np.uint8), np.array([-3], np.int32),
        np.array([11], np.uint8), np.array([31], np.uint64), free, owner, free, truth)
    active, positive = kernel.compact(*args)
    aa, pp, flat, base, fallback, legal, targets = kernel.compact(*args, materialize=True)
    if not (active.all() and positive.all() and np.array_equal(aa, active) and np.array_equal(pp, positive)
            and np.array_equal(flat, [[25, 26, 27, 28, 29]]) and np.all(base == 17)
            and np.all(fallback == 17) and np.all(legal[..., :2]) and not legal[..., 2].any()
            and np.array_equal(targets, [[0, 0, 1, 0, 0]])):
        raise RuntimeError('native compact candidate preflight failed')
    _, _, counts = kernel.compact(*args, prior_counts=True)
    if not np.array_equal(counts, [4, 1, 0, 0, 0]):
        raise RuntimeError('native complete TRAIN prior preflight failed')
    buckets = kernel.sampling_strata(np.array([0, 0, 1, 1, 1, 1], np.uint8),
        np.array([-3, -3, -2, -2, 0, 0], np.int32), np.array([1, 0, 1, 0, 1, 0], bool))
    if any(not np.array_equal(bucket, [i]) for i, bucket in enumerate(buckets)):
        raise RuntimeError('native original-order TRAIN sampling strata preflight failed')
    print(json.dumps(dict(ok=True, **kernel.info()), ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
