#!/usr/bin/env python3
"""Compact summary for V20 Static Repair training / overfit gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_history(path: Path):
    if path.is_dir():
        hist = path / "history.json"
        if hist.is_file():
            return json.loads(hist.read_text(encoding="utf-8")), str(hist)
        ckpt = path / "latest.pt"
        if ckpt.is_file():
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
            return list((obj.get("extra") or {}).get("history") or []), str(ckpt)
        raise FileNotFoundError(f"no history.json/latest.pt under {path}")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return list((obj.get("extra") or {}).get("history") or []), str(path)


def _phase(row):
    d = dict(row or {})
    diag = dict(d.get("repair_diagnostics") or {})
    base = dict(d.get("full_grid_v18_metrics") or {})
    final = dict(d.get("full_grid_v18_plus_static_metrics") or {})
    delta = dict(d.get("full_grid_delta") or {})
    d123 = dict(delta.get("main_1_2_3s") or {})
    return {
        "loss": d.get("loss"),
        "addP": diag.get("addition_precision"),
        "addR": diag.get("static_positive_recall"),
        "semAcc": diag.get("semantic_accuracy_on_static_positive"),
        "support_mIoU": diag.get("repair_support_semantic_miou"),
        "base_IoU": base.get("IoU"),
        "base_mIoU": base.get("mIoU"),
        "repair_IoU": final.get("IoU"),
        "repair_mIoU": final.get("mIoU"),
        "delta_IoU": delta.get("IoU"),
        "delta_mIoU": delta.get("mIoU"),
        "delta123_IoU": d123.get("IoU"),
        "delta123_mIoU": d123.get("mIoU"),
        "tiles": d.get("mean_tiles_per_window"),
        "seconds": d.get("epoch_elapsed_seconds"),
        "windows": d.get("windows"),
        "added_tp": diag.get("added_tp"),
        "added_fp": diag.get("added_fp"),
        "predicted_add": diag.get("predicted_add_voxels"),
        "target_positive": diag.get("target_static_positive_voxels"),
    }


def _fmt(v, digits=4):
    if v is None:
        return "-"
    if isinstance(v, int):
        return str(v)
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "path",
        help="Static Repair output dir, history.json, or checkpoint",
    )
    p.add_argument(
        "--json-out",
        default="",
        help="Optional compact JSON output path",
    )
    a = p.parse_args()

    history, source = _load_history(Path(a.path))
    if not history:
        raise RuntimeError(f"no epoch history found in {source}")

    compact = []
    print(f"source: {source}")
    print(
        "ep phase  loss     addP    addR   semAcc  "
        "base_mIoU repair_mIoU  dmIoU   d123mIoU  windows"
    )
    print("-" * 100)
    for row in history:
        ep = int(row["epoch"])
        item = {"epoch": ep}
        for phase in ("train", "val"):
            x = _phase(row.get(phase))
            item[phase] = x
            print(
                f"{ep:>2} {phase:<5} "
                f"{_fmt(x['loss'],5):>8} "
                f"{_fmt(x['addP']):>7} "
                f"{_fmt(x['addR']):>7} "
                f"{_fmt(x['semAcc']):>7} "
                f"{_fmt(x['base_mIoU'],2):>9} "
                f"{_fmt(x['repair_mIoU'],2):>11} "
                f"{_fmt(x['delta_mIoU'],2):>7} "
                f"{_fmt(x['delta123_mIoU'],2):>10} "
                f"{_fmt(x['windows'],0):>7}"
            )
        compact.append(item)

    last = compact[-1]
    print("\n=== LAST EPOCH GATE ===")
    for phase in ("train", "val"):
        x = last[phase]
        print(
            f"{phase}: "
            f"base mIoU={_fmt(x['base_mIoU'],2)}, "
            f"repair mIoU={_fmt(x['repair_mIoU'],2)}, "
            f"delta={_fmt(x['delta_mIoU'],2)} pp, "
            f"1/2/3s delta={_fmt(x['delta123_mIoU'],2)} pp, "
            f"addP={_fmt(x['addP'])}, "
            f"addR={_fmt(x['addR'])}, "
            f"semAcc={_fmt(x['semAcc'])}, "
            f"TP/FP={_fmt(x['added_tp'],0)}/{_fmt(x['added_fp'],0)}"
        )

    if a.json_out:
        op = Path(a.json_out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(compact, indent=2), encoding="utf-8")
        print(f"saved compact json: {op}")


if __name__ == "__main__":
    main()
