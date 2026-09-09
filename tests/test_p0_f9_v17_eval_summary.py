from __future__ import annotations

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, REPORT_HORIZONS_S
from tools.real_motion.summarize_p0_f9_v17_eval import dynamic_miou_from_report


def test_dynamic_miou_from_saved_overall_per_class_report():
    per_horizon = {}
    expected = {}
    for hi, horizon in enumerate(REPORT_HORIZONS_S, start=1):
        value = float(10 * hi)
        expected[str(float(horizon))] = value
        per_horizon[str(float(horizon))] = {
            "mIoU": 50.0 + hi,
            "per_class": {str(int(c)): value for c in DYNAMIC_CLASS_IDS},
        }
    report = {"overall": {"mIoU": 52.0, "per_horizon": per_horizon}}
    mean, by_h = dynamic_miou_from_report(report)
    assert by_h == expected
    assert mean == sum(expected.values()) / len(expected)
