import numpy as np

from real_motion.metrics.moving_micro_iou import (
    MovingMicroIoUAccumulator,
    MovingMicroIoUMultiHorizon,
)


def test_micro_moving_iou_weights_voxels_not_classes():
    # car: 3/4 IoU, bicycle: 0/1 IoU.
    # macro over the two present classes would be 37.5%, while micro is 3/5=60%.
    gt = np.array([[[4, 4, 4, 4, 2]]], dtype=np.int64)
    pred = np.array([[[4, 4, 4, 17, 17]]], dtype=np.int64)
    support = np.ones_like(gt, dtype=bool)

    acc = MovingMicroIoUAccumulator()
    acc.update(pred, gt, support)
    r = acc.compute()

    assert r["total_intersection"] == 3
    assert r["total_union"] == 5
    assert abs(r["micro_IoU"] - 60.0) < 1e-9


def test_multihorizon_micro_preserves_horizon_first_average():
    metric = MovingMicroIoUMultiHorizon()
    support = np.ones((1, 1, 2), dtype=bool)
    gt = np.array([[[4, 4]]], dtype=np.int64)

    # 1s = 100%, 2s = 0%, 3s = 1/2 = 50% -> horizon-first mean = 50%.
    metric.update(1.0, gt.copy(), gt, support)
    metric.update(2.0, np.array([[[17, 17]]]), gt, support)
    metric.update(3.0, np.array([[[4, 17]]]), gt, support)

    r = metric.compute()
    assert abs(r["micro_IoU"] - 50.0) < 1e-9
    assert abs(r["per_horizon"][1.0]["micro_IoU"] - 100.0) < 1e-9
    assert abs(r["per_horizon"][2.0]["micro_IoU"] - 0.0) < 1e-9
    assert abs(r["per_horizon"][3.0]["micro_IoU"] - 50.0) < 1e-9
    # Pooled: intersections 2+0+1=3, unions 2+2+2=6 -> 50% here as well.
    assert abs(r["pooled_micro_IoU"] - 50.0) < 1e-9
