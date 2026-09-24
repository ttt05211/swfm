import numpy as np

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from tools.real_motion.eval_p0_f9_v19_static_new_fov import (
    SEM_CLASSES,
    _new_raw,
    _update,
)


def _reference_update(raw, hi, pred, gt, moving, free_label):
    p = np.asarray(pred)
    g = np.asarray(gt)
    m = np.asarray(moving, dtype=bool)
    po = p != int(free_label)
    go = g != int(free_label)
    raw["occ_inter"][hi] += int((po & go).sum())
    raw["occ_union"][hi] += int((po | go).sum())
    for j, cid in enumerate(SEM_CLASSES):
        pp = p == int(cid)
        gg = g == int(cid)
        raw["sem_inter"][hi, j] += int((pp & gg).sum())
        raw["sem_union"][hi, j] += int((pp | gg).sum())
    for j, cid in enumerate(DYNAMIC_CLASS_IDS):
        pp = (p == int(cid)) & m
        gg = (g == int(cid)) & m
        raw["mov_inter"][hi, j] += int((pp & gg).sum())
        raw["mov_union"][hi, j] += int((pp | gg).sum())


def test_vectorized_v19_metric_update_matches_reference():
    rng = np.random.default_rng(20260924)
    free = 17
    pred = rng.integers(0, 18, size=(9, 7, 5), dtype=np.uint8)
    gt = rng.integers(0, 18, size=(9, 7, 5), dtype=np.uint8)
    moving = rng.random((9, 7, 5)) < 0.37

    got = _new_raw()
    ref = _new_raw()
    _update(got, 3, pred, gt, moving, free)
    _reference_update(ref, 3, pred, gt, moving, free)

    for key in got:
        np.testing.assert_array_equal(got[key], ref[key])
