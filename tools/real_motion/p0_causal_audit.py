#!/usr/bin/env python3
"""P0-C: stationary->moving and causal-support blind-spot audit."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from real_motion.prepared import PreparedShardDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.metrics.moving_miou_v2 import SPEED_THRESHOLD_MPS

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}


def ann_map(nusc, token):
    sample = nusc.get("sample", token)
    out = {}
    for at in sample["anns"]:
        a = nusc.get("sample_annotation", at)
        out[a["instance_token"]] = a
    return out


def sample_time(nusc, token):
    return float(nusc.get("sample", token)["timestamp"]) / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    ds = PreparedShardDataset(a.prepared)
    n = len(ds) if a.max_windows is None else min(len(ds), a.max_windows)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    nusc = source.nusc
    totals = {h: {"eligible_future_moving":0, "historically_stationary":0,
                  "no_pre_t0_observation":0, "birth_excluded":0, "death_excluded":0}
              for h in REPORT}

    for i in range(n):
        s = ds[i]
        t0_map = ann_map(nusc, s["t0_token"])
        hist_maps = [ann_map(nusc, t) for t in s["history_tokens"][:-1]]
        hist_times = [sample_time(nusc, t) for t in s["history_tokens"][:-1]]
        t0_time = sample_time(nusc, s["t0_token"])
        for h, fi in REPORT.items():
            records = s["moving_records"][fi]
            d = totals[h]
            d["eligible_future_moving"] += len(records)
            excluded = s["metric_excluded"][fi]
            d["birth_excluded"] += int(excluded.get("birth_dynamic", 0))
            d["death_excluded"] += int(excluded.get("death_dynamic", 0))
            for rec in records:
                inst = rec["instance_token"]
                ann0 = t0_map.get(inst)
                if ann0 is None:
                    # Should not occur under Moving-mIoU endpoint contract.
                    d["no_pre_t0_observation"] += 1
                    continue
                prior = None
                # Use the earliest available pre-t0 annotation for a stable
                # interval displacement estimate; fall back to any later prior.
                for hm, ht in zip(hist_maps, hist_times):
                    if inst in hm:
                        prior = (hm[inst], ht)
                        break
                if prior is None:
                    d["no_pre_t0_observation"] += 1
                    continue
                dt = t0_time - prior[1]
                if dt <= 0:
                    continue
                c_prev = np.asarray(prior[0]["translation"], dtype=np.float64)
                c0 = np.asarray(ann0["translation"], dtype=np.float64)
                hist_speed = float(np.linalg.norm(c0[:2] - c_prev[:2]) / dt)
                if hist_speed < SPEED_THRESHOLD_MPS:
                    d["historically_stationary"] += 1
        if i % 50 == 0:
            print("audit", i, "/", n)

    report = {"num_windows": n, "speed_threshold_mps": SPEED_THRESHOLD_MPS, "horizons": {}}
    for h,d in totals.items():
        denom = d["eligible_future_moving"]
        report["horizons"][str(h)] = dict(d)
        report["horizons"][str(h)].update({
            "stationary_to_moving_ratio": d["historically_stationary"] / denom if denom else 0.0,
            "no_pre_t0_ratio": d["no_pre_t0_observation"] / denom if denom else 0.0,
        })
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["horizons"], indent=2))
    print("saved", a.output)


if __name__ == "__main__":
    main()
