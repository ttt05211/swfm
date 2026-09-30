#!/usr/bin/env python3
"""Print a compact learned-selector result using only the standard library."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def summarize(data):
    def number(value, *, signed=False):
        return "NA" if value is None else format(value, "+.6f" if signed else ".6f")
    r = data["report"]
    d = r["delta_vs_v18_pp"]
    lines = ["===== LEARNED STATIC EVIDENCE SELECTOR ====="]
    for key in ("protocol", "mode", "successful_updates", "train_windows", "dev_windows", "batch_size_patches",
                "bank_mib", "best_update", "elapsed_seconds", "route"):
        if key in data: lines.append(f"{key}: {data[key]}")
    lines.append("===== BEST (固定 threshold=0.5，整体 pp) =====")
    lines.append(f"V18_mIoU={number(r['baseline']['mIoU'])} selected_mIoU={number(r['selected']['mIoU'])} "
                 f"dMiOU={number(d['mIoU'], signed=True)} dIoU={number(d['IoU'], signed=True)} "
                 f"dMovingMicro={number(d['MovingMicro'], signed=True)}")
    for h in ("1.0", "2.0", "3.0"):
        hd = d["per_horizon"][h]
        lines.append(f"{h}s dMiOU={number(hd['mIoU'], signed=True)} dIoU={number(hd['IoU'], signed=True)} "
                     f"dMovingMicro={number(hd['MovingMicro'], signed=True)}")
    lines.append("addition_quality: " + json.dumps(r["quality"], ensure_ascii=False))
    scene = {k: v for k, v in r["scene_delta"].items() if k != "by_scene"}
    lines.append("scene_delta: " + json.dumps(scene, ensure_ascii=False))
    if "last_candidate_report" in data:
        last = data["last_candidate_report"]
        lines.append(f"last_learned_candidate_dMiOU={number(last['delta_vs_v18_pp']['mIoU'], signed=True)} "
                     f"add={last['quality'].get('added', 0)}")
    lines.append("gate: " + json.dumps(data["gate"], ensure_ascii=False))
    if data.get("best_update") == 0:
        lines.append("注意：所有 learned candidates 未优于 V18；best 为零修改基线，不代表方法有效。")
    for key in ("best_checkpoint", "last_checkpoint", "git_commit"):
        if key in data: lines.append(f"{key}: {data[key]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    args = parser.parse_args()
    print(summarize(json.loads(Path(args.summary).read_text(encoding="utf-8"))))


if __name__ == "__main__": main()
