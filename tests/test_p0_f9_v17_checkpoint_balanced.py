import math

from tools.real_motion.select_p0_f9_v17_checkpoint_balanced import pareto_front, select_balanced


def _row(name, epoch, iou, miou, moving):
    return {
        "name": name,
        "epoch": epoch,
        "occupancy_IoU": iou,
        "semantic_mIoU": miou,
        "moving_mIoU": moving,
    }


def test_current_rl_example_selects_epoch5_balanced_tradeoff():
    rows = [
        _row("ep1", 1, 52.9830, 41.9390, 21.0038),
        _row("ep5", 5, 53.1066, 42.8635, 22.9424),
        _row("ep6", 6, 53.1470, 43.0350, 22.5426),
        _row("ep10", 10, 53.1520, 42.8862, 22.3507),
    ]
    out = select_balanced(rows)
    assert out["selected"]["epoch"] == 5
    assert math.isclose(out["selected"]["max_regret_pp"], 43.0350 - 42.8635, abs_tol=1e-12)


def test_pareto_filter_drops_strictly_dominated_candidate():
    rows = [
        _row("good", 1, 50.0, 40.0, 30.0),
        _row("dominated", 2, 49.0, 39.0, 29.0),
        _row("tradeoff", 3, 51.0, 38.0, 31.0),
    ]
    assert pareto_front(rows) == [True, False, True]


def test_minimax_prefers_balanced_not_best_single_metric():
    rows = [
        _row("iou_best", 1, 60.0, 49.0, 39.0),
        _row("balanced", 2, 59.5, 49.5, 39.5),
        _row("moving_best", 3, 59.0, 49.0, 40.0),
    ]
    out = select_balanced(rows)
    assert out["selected"]["name"] == "balanced"
    assert math.isclose(out["selected"]["max_regret_pp"], 0.5)


def test_equal_metrics_tie_breaks_to_earlier_epoch():
    rows = [
        _row("late", 8, 50.0, 40.0, 30.0),
        _row("early", 4, 50.0, 40.0, 30.0),
    ]
    out = select_balanced(rows)
    assert out["selected"]["epoch"] == 4
