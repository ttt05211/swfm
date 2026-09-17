#!/usr/bin/env python3
"""Compare two V18 macro/micro moving diagnostic JSON reports.

Primary use: Y600-pred versus Clean-E14.  Prints overall metrics, macro/micro
Moving-IoU by horizon, and the full dynamic-class x horizon Moving-IoU table.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    NUSCENES_LABELS,
    REPORT_HORIZONS_S,
)


def _key(d, value):
    if value in d:
        return value
    s = str(value)
    if s in d:
        return s
    if isinstance(value, float):
        s2 = f"{value:g}"
        if s2 in d:
            return s2
    raise KeyError(value)


def _candidate(x):
    return x["reports"]["candidate"]


def _summary(x):
    r = _candidate(x)
    d = x["diagnostics"]
    mm = r["moving_micro"]
    return {
        "IoU": float(r["occupancy"]["IoU"]),
        "mIoU": float(r["overall"]["mIoU"]),
        "MovingMacro": float(r["moving"]["mIoU"]),
        "MovingMicro": float(mm["micro_IoU"]),
        "MovingMicroPooled": float(mm["pooled_micro_IoU"]),
        "ADE": float(d["learned_ade_m"]),
        "FDE": float(d["learned_fde_m"]),
        "YawMAE": float(d["wrapped_yaw_mae_deg"]),
    }


def _horizon_report(x, h):
    r = _candidate(x)
    macro = r["moving"]["per_horizon"]
    micro = r["moving_micro"]["per_horizon"]
    return macro[_key(macro, h)], micro[_key(micro, h)]


def _class_value(report, class_id):
    pc = report["per_class"]
    return float(pc[_key(pc, class_id)])


def _fmt(v, width=9, digits=4):
    return f"{v:{width}.{digits}f}" if math.isfinite(v) else f"{'nan':>{width}s}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference-name", default="Y600-pred")
    p.add_argument("--candidate-name", default="Clean-E14")
    p.add_argument("--output-json", default="")
    a = p.parse_args()

    ref = json.loads(Path(a.reference).read_text(encoding="utf-8"))
    cand = json.loads(Path(a.candidate).read_text(encoding="utf-8"))
    rs = _summary(ref)
    cs = _summary(cand)

    print("\n=== MOVING MACRO/MICRO SUMMARY ===")
    print(
        f"{'Model':<16}{'IoU':>9}{'mIoU':>9}{'MacroMov':>10}{'MicroMov':>10}"
        f"{'PoolMicro':>11}{'ADE':>9}{'FDE':>9}{'YawMAE':>9}"
    )
    for name, s in ((a.reference_name, rs), (a.candidate_name, cs)):
        print(
            f"{name:<16}{s['IoU']:>9.4f}{s['mIoU']:>9.4f}"
            f"{s['MovingMacro']:>10.4f}{s['MovingMicro']:>10.4f}"
            f"{s['MovingMicroPooled']:>11.4f}{s['ADE']:>9.4f}"
            f"{s['FDE']:>9.4f}{s['YawMAE']:>9.3f}"
        )
    print(
        f"{'Delta C-R':<16}{cs['IoU']-rs['IoU']:>+9.4f}"
        f"{cs['mIoU']-rs['mIoU']:>+9.4f}"
        f"{cs['MovingMacro']-rs['MovingMacro']:>+10.4f}"
        f"{cs['MovingMicro']-rs['MovingMicro']:>+10.4f}"
        f"{cs['MovingMicroPooled']-rs['MovingMicroPooled']:>+11.4f}"
        f"{cs['ADE']-rs['ADE']:>+9.4f}{cs['FDE']-rs['FDE']:>+9.4f}"
        f"{cs['YawMAE']-rs['YawMAE']:>+9.3f}"
    )

    horizon_rows = []
    print("\n=== MOVING BY HORIZON ===")
    print(
        f"{'H':>4}{'RefMacro':>11}{'CandMacro':>11}{'Delta':>10}"
        f"{'RefMicro':>11}{'CandMicro':>11}{'Delta':>10}"
    )
    for h in REPORT_HORIZONS_S:
        rm, rmi = _horizon_report(ref, h)
        cm, cmi = _horizon_report(cand, h)
        row = {
            "horizon_s": h,
            "reference_macro": float(rm["mIoU"]),
            "candidate_macro": float(cm["mIoU"]),
            "reference_micro": float(rmi["micro_IoU"]),
            "candidate_micro": float(cmi["micro_IoU"]),
        }
        horizon_rows.append(row)
        print(
            f"{h:>4.1f}{row['reference_macro']:>11.4f}{row['candidate_macro']:>11.4f}"
            f"{row['candidate_macro']-row['reference_macro']:>+10.4f}"
            f"{row['reference_micro']:>11.4f}{row['candidate_micro']:>11.4f}"
            f"{row['candidate_micro']-row['reference_micro']:>+10.4f}"
        )

    class_rows = []
    print("\n=== PER-CLASS x HORIZON MOVING-IoU ===")
    header = f"{'class':<23}"
    for h in REPORT_HORIZONS_S:
        header += f"{('R'+str(int(h))+'s'):>9}{('C'+str(int(h))+'s'):>9}{'Delta':>9}"
    header += f"{'Rmean':>9}{'Cmean':>9}{'Delta':>9}"
    print(header)

    for c in DYNAMIC_CLASS_IDS:
        rv, cv = [], []
        row = {"class_id": int(c), "class_name": NUSCENES_LABELS[int(c)], "horizons": {}}
        line = f"{c}:{NUSCENES_LABELS[int(c)]:<20}"
        for h in REPORT_HORIZONS_S:
            rm, _ = _horizon_report(ref, h)
            cm, _ = _horizon_report(cand, h)
            r = _class_value(rm, c)
            q = _class_value(cm, c)
            rv.append(r)
            cv.append(q)
            row["horizons"][str(h)] = {
                "reference": r,
                "candidate": q,
                "delta": q - r,
            }
            line += f"{r:>9.3f}{q:>9.3f}{q-r:>+9.3f}"
        rmean = float(np.mean(rv))
        cmean = float(np.mean(cv))
        row["reference_mean"] = rmean
        row["candidate_mean"] = cmean
        row["delta_mean"] = cmean - rmean
        class_rows.append(row)
        line += f"{rmean:>9.3f}{cmean:>9.3f}{cmean-rmean:>+9.3f}"
        print(line)

    ranked = sorted(class_rows, key=lambda z: abs(z["delta_mean"]), reverse=True)
    print("\n=== LARGEST CLASS-MEAN DELTAS |candidate-reference| ===")
    for row in ranked:
        print(
            f"{row['class_id']:>2} {row['class_name']:<22} "
            f"R={row['reference_mean']:.3f} C={row['candidate_mean']:.3f} "
            f"Delta={row['delta_mean']:+.3f}"
        )

    if a.output_json:
        out = {
            "reference": str(Path(a.reference).resolve()),
            "candidate": str(Path(a.candidate).resolve()),
            "reference_name": a.reference_name,
            "candidate_name": a.candidate_name,
            "summary": {
                "reference": rs,
                "candidate": cs,
                "delta_candidate_minus_reference": {
                    k: cs[k] - rs[k] for k in rs
                },
            },
            "per_horizon": horizon_rows,
            "per_class": class_rows,
        }
        op = Path(a.output_json)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
