#!/usr/bin/env python3
"""P0-D: SE(3)-static + GT-moving upper bound under final composition rules."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS, MovingMIoUV2MultiHorizon, SemanticIoUAccumulator,
    REPORT_HORIZONS_S,
)
from real_motion.nuscenes_adapter import gt_true_static_mask, dynamic_only_semantics

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}


def mean_h(acc):
    per = {h: acc[h].compute() for h in REPORT_HORIZONS_S}
    return {"mIoU": float(np.nanmean([per[h]["mIoU"] for h in REPORT_HORIZONS_S])),
            "per_horizon": per}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    ds = PreparedShardDataset(a.prepared)
    n = len(ds) if a.max_windows is None else min(len(ds), a.max_windows)

    oracle_acc = {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S}
    support_oracle_acc = {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S}
    static_acc = {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S}
    moving = MovingMIoUV2MultiHorizon()
    support_moving = MovingMIoUV2MultiHorizon()

    for i in range(n):
        s = ds[i]
        for h, fi in REPORT.items():
            static = torch.from_numpy(s["static_future_occ"][fi])
            gt = s["future_gt_occ"][fi]
            gt_dyn_all = torch.from_numpy(dynamic_only_semantics(gt))
            gt_dyn_causal = torch.from_numpy(s["future_dynamic_target_occ"][fi])
            protected = torch.from_numpy(s["confident_static_future_mask"][fi])
            write = torch.from_numpy(s["generation_support_occ"][fi])
            # Decomposition oracle: a perfect dynamic generator can supply every
            # future dynamic-semantic voxel, while static stays SE(3)-transported.
            oracle = static_protected_compose(
                static, gt_dyn_all, protected, DYNAMIC_CLASS_IDS, write_support=None,
            ).numpy()
            # Causal-support oracle obeys the exact support/target contract used
            # by the trainable sparse WM.
            support_oracle = static_protected_compose(
                static, gt_dyn_causal, protected, DYNAMIC_CLASS_IDS, write_support=write,
            ).numpy()
            oracle_acc[h].update(oracle, gt)
            support_oracle_acc[h].update(support_oracle, gt)
            static_mask = gt_true_static_mask(gt, s["gt_moving_support"][fi])
            static_acc[h].update(s["static_future_occ"][fi], gt, static_mask)
            moving.update(h, oracle, gt, s["gt_moving_support"][fi])
            support_moving.update(h, support_oracle, gt, s["gt_moving_support"][fi])
        if i % 50 == 0:
            print("oracle", i, "/", n)

    report = {
        "num_windows": n,
        "decomposition_oracle_overall": mean_h(oracle_acc),
        "decomposition_oracle_Moving-mIoU_v2": moving.compute(),
        "causal_support_oracle_overall": mean_h(support_oracle_acc),
        "causal_support_oracle_Moving-mIoU_v2": support_moving.compute(),
        "se3_true_static": mean_h(static_acc),
        "note": "decomposition oracle uses all future dynamic semantics with confident-static protection only; causal-support oracle uses dynamic GT only inside the causal KTA tube. Their gap isolates support/reachability headroom.",
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
