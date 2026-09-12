#!/usr/bin/env python3
"""Summarize the paired V17 control/backtrace3D fast probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BRANCH = "local_stwm_center_always_source_order"


def _hrow(mapping, h):
    return mapping.get(float(h), mapping.get(str(float(h))))


def _load(path):
    x = json.loads(Path(path).read_text(encoding="utf-8"))
    r = (x.get("reports") or {}).get(BRANCH)
    if r is None:
        raise RuntimeError(f"{path}: missing {BRANCH}")
    mov = r["moving"]
    occ = r["occupancy"]
    return {
        "path": str(path),
        "checkpoint_protocol": x.get("checkpoint_protocol"),
        "IoU": float(occ["IoU"]),
        "mIoU": float(r["overall"]["mIoU"]),
        "Moving": float(mov["mIoU"]),
        "Moving_1s": float(_hrow(mov["per_horizon"], 1.0)["mIoU"]),
        "Moving_2s": float(_hrow(mov["per_horizon"], 2.0)["mIoU"]),
        "Moving_3s": float(_hrow(mov["per_horizon"], 3.0)["mIoU"]),
        "ADE": float((x.get("diagnostics") or {}).get("learned_ade_m", float("nan"))),
        "FDE": float((x.get("diagnostics") or {}).get("learned_fde_m", float("nan"))),
    }


def _delta(t, c):
    return {
        k: float(t[k] - c[k])
        for k in ("IoU", "mIoU", "Moving", "Moving_1s", "Moving_2s", "Moving_3s", "ADE", "FDE")
    }


def _decision(d):
    dm = d["Moving"]
    safe = d["IoU"] >= -0.10 and d["mIoU"] >= -0.10
    horizons_ok = not (d["Moving_2s"] < 0.0 and d["Moving_3s"] < 0.0)
    if dm >= 0.30 and safe and horizons_ok:
        return "GO"
    if 0.15 <= dm < 0.30 and safe and horizons_ok:
        return "BORDERLINE"
    return "STOP"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control-300", required=True)
    p.add_argument("--treatment-300", required=True)
    p.add_argument("--control-600", required=True)
    p.add_argument("--treatment-600", required=True)
    p.add_argument("--output", default="")
    a = p.parse_args()

    rows = {
        "control_step300": _load(a.control_300),
        "treatment_step300": _load(a.treatment_300),
        "control_step600": _load(a.control_600),
        "treatment_step600": _load(a.treatment_600),
    }
    d300 = _delta(rows["treatment_step300"], rows["control_step300"])
    d600 = _delta(rows["treatment_step600"], rows["control_step600"])
    out = {
        "report_branch": BRANCH,
        "rows": rows,
        "treatment_minus_control": {"step300": d300, "step600": d600},
        "decision": {
            "step300": _decision(d300),
            "step600": _decision(d600),
            "primary": "step600",
            "final": _decision(d600),
            "rules": {
                "GO": "Moving >= +0.30 pp; IoU/mIoU >= -0.10 pp; 2s/3s not both negative",
                "BORDERLINE": "+0.15 <= Moving < +0.30 pp with same safeguards",
                "STOP": "Moving < +0.15 pp or safeguard failure",
                "ADE_FDE": "diagnostic only; never overrides Moving-mIoU",
            },
        },
    }

    print("=== V17 BACKTRACE3D FAST PROBE ===")
    print(
        f"{'checkpoint':20s} {'IoU':>8s} {'mIoU':>8s} {'Moving':>8s} "
        f"{'M@1s':>8s} {'M@2s':>8s} {'M@3s':>8s} {'ADE':>8s} {'FDE':>8s}"
    )
    for name, r in rows.items():
        print(
            f"{name:20s} {r['IoU']:8.4f} {r['mIoU']:8.4f} {r['Moving']:8.4f} "
            f"{r['Moving_1s']:8.4f} {r['Moving_2s']:8.4f} {r['Moving_3s']:8.4f} "
            f"{r['ADE']:8.4f} {r['FDE']:8.4f}"
        )

    print("\n=== Treatment - Control ===")
    for key, d in (("step300", d300), ("step600", d600)):
        print(
            f"{key:7s} dIoU={d['IoU']:+.4f} dmIoU={d['mIoU']:+.4f} "
            f"dMoving={d['Moving']:+.4f} dM1={d['Moving_1s']:+.4f} "
            f"dM2={d['Moving_2s']:+.4f} dM3={d['Moving_3s']:+.4f} "
            f"dADE={d['ADE']:+.4f} dFDE={d['FDE']:+.4f} "
            f"decision={out['decision'][key]}"
        )
    print(f"\nFINAL={out['decision']['final']} (primary=fixed step600)")

    if a.output:
        op = Path(a.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved {op}")


if __name__ == "__main__":
    main()
