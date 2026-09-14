#!/usr/bin/env python3
"""Summarize the fixed-time V17 C0-C/C0-S paired experiment.

Primary decision uses C0-S minus C0-C at the same optimizer step.  Historical
B-C is retained only for the already-completed C0 control-fidelity audit and is
never used as the direct C0-S treatment control.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

BRANCH = "local_stwm_center_always_source_order"
BEST_RL_A1 = {"IoU": 53.1066, "mIoU": 42.8865, "Moving": 22.9658}
HORIZONS = (1.0, 2.0, 3.0)


def _ph(row, h):
    x = row.get(float(h))
    if x is None:
        x = row.get(str(float(h)))
    if x is None:
        x = row.get(str(h))
    if x is None:
        raise RuntimeError(f"missing per-horizon row for {h}s")
    return x


def _finite_dict(d, keys):
    bad = [k for k in keys if not math.isfinite(float(d[k]))]
    if bad:
        raise RuntimeError(f"non-finite metrics: {bad}")


def _load(path):
    x = json.loads(Path(path).read_text(encoding="utf-8"))
    r = (x.get("reports") or {}).get(BRANCH)
    if r is None:
        raise RuntimeError(f"{path}: missing {BRANCH}")
    occ = r.get("occupancy") or {}
    moving_ph = (r.get("moving") or {}).get("per_horizon") or {}
    diag = x.get("diagnostics") or {}
    row = {
        "path": str(path),
        "IoU": float(occ["IoU"]),
        "mIoU": float(r["overall"]["mIoU"]),
        "Moving": float(r["moving"]["mIoU"]),
        "Moving_1s": float(_ph(moving_ph, 1.0)["mIoU"]),
        "Moving_2s": float(_ph(moving_ph, 2.0)["mIoU"]),
        "Moving_3s": float(_ph(moving_ph, 3.0)["mIoU"]),
        "ADE": float(diag.get("learned_ade_m", float("nan"))),
        "FDE": float(diag.get("learned_fde_m", float("nan"))),
    }
    _finite_dict(row, ("IoU", "mIoU", "Moving", "Moving_1s", "Moving_2s", "Moving_3s"))
    return row


def _delta(a, b):
    keys = ("IoU", "mIoU", "Moving", "Moving_1s", "Moving_2s", "Moving_3s")
    out = {k: float(a[k] - b[k]) for k in keys}
    for k in ("ADE", "FDE"):
        if math.isfinite(float(a[k])) and math.isfinite(float(b[k])):
            out[k] = float(a[k] - b[k])
        else:
            out[k] = float("nan")
    return out


def _decision(delta):
    dm = float(delta["Moving"])
    di = float(delta["IoU"])
    dmi = float(delta["mIoU"])
    if dm >= 0.30 and di >= -0.02 and dmi >= -0.05:
        return "GO"
    if 0.10 < dm < 0.30 and di >= -0.02 and dmi >= -0.05:
        return "BORDERLINE"
    return "STOP"


def _beats_best(row):
    return {
        "IoU": bool(row["IoU"] > BEST_RL_A1["IoU"]),
        "mIoU": bool(row["mIoU"] > BEST_RL_A1["mIoU"]),
        "Moving": bool(row["Moving"] > BEST_RL_A1["Moving"]),
        "all_three": bool(
            row["IoU"] > BEST_RL_A1["IoU"]
            and row["mIoU"] > BEST_RL_A1["mIoU"]
            and row["Moving"] > BEST_RL_A1["Moving"]
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--historical-control-e8", default="")
    p.add_argument("--c0-control-e8", default="")
    p.add_argument("--control-mid", required=True)
    p.add_argument("--scene-mid", required=True)
    p.add_argument("--control-end", required=True)
    p.add_argument("--scene-end", required=True)
    p.add_argument("--output", default="")
    a = p.parse_args()

    rows = {
        "shared_start": _load(a.start),
        "C0C_step300": _load(a.control_mid),
        "C0S_step300": _load(a.scene_mid),
        "C0C_step600": _load(a.control_end),
        "C0S_step600": _load(a.scene_end),
    }
    if bool(a.historical_control_e8) != bool(a.c0_control_e8):
        raise ValueError("--historical-control-e8 and --c0-control-e8 must be supplied together")
    if a.historical_control_e8:
        rows["historical_BC_e8"] = _load(a.historical_control_e8)
        rows["C0C_e8"] = _load(a.c0_control_e8)

    pair = {
        "step300": _delta(rows["C0S_step300"], rows["C0C_step300"]),
        "step600": _delta(rows["C0S_step600"], rows["C0C_step600"]),
    }
    start = {
        "step300": _delta(rows["C0S_step300"], rows["shared_start"]),
        "step600": _delta(rows["C0S_step600"], rows["shared_start"]),
    }
    out = {
        "report_branch": BRANCH,
        "rows": rows,
        "scene_minus_control": pair,
        "scene_minus_start": start,
        "decision": {
            "step300": _decision(pair["step300"]),
            "step600": _decision(pair["step600"]),
            "primary_fixed_budget": "step600",
            "final": _decision(pair["step600"]),
            "rules": {
                "GO": "Moving >= +0.30 pp, IoU >= -0.02 pp, mIoU >= -0.05 pp",
                "BORDERLINE": "0.10 < Moving < 0.30 pp with IoU/mIoU safeguards",
                "STOP": "Moving <= +0.10 pp or safeguard failure",
            },
        },
        "existing_RL_epoch5_A1_best": BEST_RL_A1,
        "C0S_vs_existing_best": {
            "step300_delta": {k: rows["C0S_step300"][k] - BEST_RL_A1[k] for k in BEST_RL_A1},
            "step600_delta": {k: rows["C0S_step600"][k] - BEST_RL_A1[k] for k in BEST_RL_A1},
            "step300_beats": _beats_best(rows["C0S_step300"]),
            "step600_beats": _beats_best(rows["C0S_step600"]),
        },
    }
    if "historical_BC_e8" in rows:
        out["control_fidelity_C0C_minus_historical_BC_e8"] = _delta(
            rows["C0C_e8"], rows["historical_BC_e8"]
        )

    print("=== V17 C0-S FIXED-TIME PAIRED SUMMARY ===")
    print(f"report_branch={BRANCH}")
    print(
        f"{'checkpoint':18s} {'IoU':>8s} {'mIoU':>8s} {'Moving':>8s} "
        f"{'M@1s':>8s} {'M@2s':>8s} {'M@3s':>8s} {'ADE':>8s} {'FDE':>8s}"
    )
    for name in ("shared_start", "C0C_step300", "C0S_step300", "C0C_step600", "C0S_step600"):
        r = rows[name]
        print(
            f"{name:18s} {r['IoU']:8.4f} {r['mIoU']:8.4f} {r['Moving']:8.4f} "
            f"{r['Moving_1s']:8.4f} {r['Moving_2s']:8.4f} {r['Moving_3s']:8.4f} "
            f"{r['ADE']:8.4f} {r['FDE']:8.4f}"
        )

    print("\n=== C0-S minus same-step C0-C (PRIMARY TREATMENT COMPARISON) ===")
    for key in ("step300", "step600"):
        d = pair[key]
        print(
            f"{key:7s} dIoU={d['IoU']:+.4f} dmIoU={d['mIoU']:+.4f} "
            f"dMoving={d['Moving']:+.4f} dM1={d['Moving_1s']:+.4f} "
            f"dM2={d['Moving_2s']:+.4f} dM3={d['Moving_3s']:+.4f} "
            f"decision={out['decision'][key]}"
        )

    print("\n=== C0-S minus shared RL epoch5 start ===")
    for key in ("step300", "step600"):
        d = start[key]
        print(
            f"{key:7s} dIoU={d['IoU']:+.4f} dmIoU={d['mIoU']:+.4f} "
            f"dMoving={d['Moving']:+.4f}"
        )

    print("\n=== EXISTING RL EPOCH5+A1 BEST ===")
    print(json.dumps(out["C0S_vs_existing_best"], indent=2))
    print(f"\nFINAL={out['decision']['final']} (primary=fixed step600)")

    if a.output:
        op = Path(a.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved {op}")


if __name__ == "__main__":
    main()
