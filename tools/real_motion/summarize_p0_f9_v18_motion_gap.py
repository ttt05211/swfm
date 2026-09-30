#!/usr/bin/env python3
"""Compact single-pass motion-gap report; standard library only."""
import argparse
import json
from pathlib import Path


def summarize(data):
    number = lambda x: "NA" if x is None else f"{x:+.6f}"
    rows = ["===== V18 MOTION GAP (诊断，不是 learned gain) =====",
            f"protocol: {data['protocol']}", f"windows: {data['windows']}", f"scenes: {data['scenes']}"]
    for name, r in data["variants"].items():
        d, scene = r["delta_vs_v18_pp"], r["scene_delta"]
        rows.append(f"{name:33s} dMiOU={number(d['mIoU'])} dIoU={number(d['IoU'])} "
                    f"dMovingMicro={number(d['MovingMicro'])} scenes=+{scene['positive']}/0{scene['zero']}/-{scene['negative']}")
    rows.append("===== POSITION / YAW / INTERACTION =====")
    for k, v in data["decomposition"].items(): rows.append(k + ": " + json.dumps(v, ensure_ascii=False))
    rows.append("===== PER HORIZON =====")
    for h in ("1.0", "2.0", "3.0"):
        values = [(k, data["variants"][k]["delta_vs_v18_pp"]["per_horizon"][h]["mIoU"])
                  for k in ("GT_XY_PRED_YAW", "PRED_XY_GT_YAW", "GT_XY_GT_YAW")]
        rows.append(h + "s " + " ".join(k + "=" + number(v) for k, v in values))
    rows.append("===== SOURCE ERROR STRATA (描述统计，非独立样本显著性) =====")
    for group, values in data["motion_errors"].items():
        error, yaw = values["source_center_error_m"], values["yaw_error_deg"]
        rows.append(f"{group}: XY_n={error['count']} mean_m={error['mean']} p90_m={error['p90']} "
                    f"yaw_n={yaw['count']} mean_deg={yaw['mean']}")
    rows.append("all_error_details: " + json.dumps(data["motion_errors"]["all"], ensure_ascii=False))
    rows.append("source_audit: " + json.dumps(data["source_audit"], ensure_ascii=False))
    rows.append("reference_check: " + json.dumps(data["reference_check"], ensure_ascii=False))
    rows.append("performance: " + json.dumps(data["performance"], ensure_ascii=False))
    rows.append("结论边界：只解释 observed-t0 source 的运动缺口；不自动批准新网络或训练，不代表恢复了 oracle 涨点。")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("audit"); a = p.parse_args()
    print(summarize(json.loads(Path(a.audit).read_text(encoding="utf-8"))))


if __name__ == "__main__": main()
