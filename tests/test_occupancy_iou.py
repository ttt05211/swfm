import numpy as np

from real_motion.metrics.occupancy_iou import OccupancyIoUAccumulator, OccupancyIoUMultiHorizon


def test_binary_occupancy_iou_ignores_semantic_mismatch():
    # Both occupied voxels are geometrically correct even though car/truck labels swap.
    pred = np.array([4, 10, 17, 17], dtype=np.uint8)
    gt = np.array([10, 4, 17, 17], dtype=np.uint8)
    acc = OccupancyIoUAccumulator(free_label=17)
    acc.update(pred, gt)
    report = acc.compute()
    assert report["intersection"] == 2
    assert report["union"] == 2
    assert report["IoU"] == 100.0


def test_binary_occupancy_iou_counts_false_positive_and_false_negative():
    pred = np.array([4, 17], dtype=np.uint8)
    gt = np.array([17, 10], dtype=np.uint8)
    acc = OccupancyIoUAccumulator(free_label=17)
    acc.update(pred, gt)
    report = acc.compute()
    assert report["intersection"] == 0
    assert report["union"] == 2
    assert report["IoU"] == 0.0


def test_binary_occupancy_iou_dataset_accumulates_before_division():
    acc = OccupancyIoUAccumulator(free_label=17)
    acc.update(np.array([4, 17]), np.array([4, 10]))  # inter=1 union=2
    acc.update(np.array([4, 4]), np.array([4, 4]))    # inter=2 union=2
    report = acc.compute()
    assert report["intersection"] == 3
    assert report["union"] == 4
    assert report["IoU"] == 75.0


def test_binary_occupancy_iou_horizon_first_mean():
    metric = OccupancyIoUMultiHorizon(free_label=17)
    metric.update(1.0, np.array([4, 17]), np.array([4, 17]))      # 100
    metric.update(2.0, np.array([4, 17]), np.array([17, 10]))     # 0
    metric.update(3.0, np.array([4, 17]), np.array([4, 10]))      # 50
    report = metric.compute()
    assert report["per_horizon"][1.0]["IoU"] == 100.0
    assert report["per_horizon"][2.0]["IoU"] == 0.0
    assert report["per_horizon"][3.0]["IoU"] == 50.0
    assert report["IoU"] == 50.0
