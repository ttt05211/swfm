#!/usr/bin/env python3
"""Fail-closed decision summary for the selective forecasting feasibility test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL_PROTOCOL = "p0_f9_v17_selective_forecast_eval_v1"
LATENCY_PROTOCOL = "p0_f9_v17_selective_learned_stage_latency_v1"
SELECTOR_PROTOCOL = "p0_f9_v17_correction_selector_v1"
PRIMARY_Q = "20"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(cond, msg):
    if not cond:
        raise RuntimeError("FEASIBILITY VALIDATION FAILED: " + msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selector-report", required=True)
    p.add_argument("--eval-report", required=True)
    p.add_argument("--latency-report", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    sel = _load(a.selector_report)
    ev = _load(a.eval_report)
    lat = _load(a.latency_report)

    _require(sel.get("protocol") == SELECTOR_PROTOCOL, "selector protocol mismatch")
    _require(ev.get("protocol") == EVAL_PROTOCOL, "eval protocol mismatch")
    _require(lat.get("protocol") == LATENCY_PROTOCOL, "latency protocol mismatch")
    _require(ev.get("scene_overlap_checked_zero") is True, "selector/eval scene overlap not cleared")
    _require(
        str(Path(ev["selector_checkpoint"]).resolve())
        == str(Path(lat["selector_checkpoint"]).resolve()),
        "eval/latency selector checkpoint differs",
    )
    _require(
        str(Path(ev["v17_cache"]).resolve())
        == str(Path(lat["v17_cache"]).resolve()),
        "eval/latency V17 cache differs",
    )
    ref = ev.get("reference_v17_check")
    _require(isinstance(ref, dict), "missing frozen Q0/Q100 reference check")
    _require(ref.get("pass_1e-6") is True, "Q0/Q100 reference reproduction failed")

    curve = ev["curve"]
    _require(PRIMARY_Q in curve["oracle"], "evaluation lacks Q=20 oracle")
    _require(PRIMARY_Q in curve["selector"], "evaluation lacks Q=20 selector")
    _require(PRIMARY_Q in curve["speed"], "evaluation lacks Q=20 speed")
    _require(PRIMARY_Q in curve["random"], "evaluation lacks Q=20 random")
    _require("20.0" in lat["results"], "latency report lacks Q=20")
    _require("100.0" in lat["results"], "latency report lacks Q=100")

    q0 = curve["oracle"]["0"]
    q100 = curve["oracle"]["100"]
    qo = curve["oracle"][PRIMARY_Q]
    qs = curve["selector"][PRIMARY_Q]
    qspeed = curve["speed"][PRIMARY_Q]
    qrand = curve["random"][PRIMARY_Q]

    kta_m = float(q0["Moving"])
    dense_m = float(q100["Moving"])
    oracle_m = float(qo["Moving"])
    selector_m = float(qs["Moving"])
    baseline_best = max(kta_m, dense_m)

    oracle_sparse_margin = oracle_m - baseline_best
    oracle_gain_vs_kta = oracle_m - kta_m
    selector_gain_vs_kta = selector_m - kta_m
    selector_retention = (
        selector_gain_vs_kta / oracle_gain_vs_kta
        if oracle_gain_vs_kta > 1e-12 else float("nan")
    )
    selector_vs_speed = selector_m - float(qspeed["Moving"])
    selector_vs_random = selector_m - float(qrand["Moving_mean"])
    miou_guard = float(qs["mIoU"]) - float(q0["mIoU"])
    latency_reduction = float(
        lat["results"]["20.0"]["learned_stage_latency_reduction_fraction"]
    )

    checks = {
        "oracle_sparse_headroom": bool(oracle_sparse_margin >= 0.30),
        "selector_beats_speed": bool(selector_vs_speed >= 0.20),
        "selector_beats_random": bool(selector_vs_random >= 0.20),
        "selector_oracle_retention": bool(selector_retention >= 0.50),
        "selector_mIoU_guard": bool(miou_guard >= -0.10),
        "learned_stage_latency_reduction": bool(latency_reduction >= 0.50),
    }
    oracle_go = checks["oracle_sparse_headroom"]
    selector_go = all([
        checks["selector_beats_speed"],
        checks["selector_beats_random"],
        checks["selector_oracle_retention"],
        checks["selector_mIoU_guard"],
    ])
    efficiency_go = checks["learned_stage_latency_reduction"]

    if not oracle_go:
        final = "STOP_SELECTOR_STORY"
        reason = "Q20 oracle does not establish sparse correction headroom over both frozen experts."
    elif selector_go and efficiency_go:
        final = "GO_SELECTIVE_FORECAST_MAINLINE"
        reason = "Sparse oracle, causal selector and measured learned-stage efficiency all pass."
    else:
        final = "ORACLE_ONLY_STOP_AS_MAIN_CONTRIBUTION"
        reason = (
            "Sparse oracle exists, but the cheap causal selector and/or measured efficiency "
            "does not yet retain enough of it."
        )

    report = {
        "protocol": "p0_f9_v17_selective_feasibility_summary_v1",
        "primary_budget_percent": 20.0,
        "frozen_thresholds": {
            "oracle_q20_margin_over_best_expert_pp": 0.30,
            "selector_q20_margin_over_speed_pp": 0.20,
            "selector_q20_margin_over_random_pp": 0.20,
            "selector_q20_oracle_gain_retention": 0.50,
            "selector_q20_mIoU_drop_vs_KTA_pp": -0.10,
            "q20_learned_stage_latency_reduction_fraction": 0.50,
        },
        "observed": {
            "KTA_Moving": kta_m,
            "dense_V17_Moving": dense_m,
            "oracle_Q20_Moving": oracle_m,
            "selector_Q20_Moving": selector_m,
            "oracle_Q20_margin_over_best_expert_pp": oracle_sparse_margin,
            "oracle_Q20_gain_vs_KTA_pp": oracle_gain_vs_kta,
            "selector_Q20_gain_vs_KTA_pp": selector_gain_vs_kta,
            "selector_Q20_oracle_gain_retention": selector_retention,
            "selector_Q20_minus_speed_pp": selector_vs_speed,
            "selector_Q20_minus_random_mean_pp": selector_vs_random,
            "selector_Q20_mIoU_minus_KTA_pp": miou_guard,
            "Q20_learned_stage_latency_reduction_fraction": latency_reduction,
            "Q20_learned_stage_speedup_vs_dense": float(
                lat["results"]["20.0"]["learned_stage_speedup_vs_dense"]
            ),
            "selector_internal_val_spearman": float(sel["best_val_spearman"]),
        },
        "checks": checks,
        "oracle_go": oracle_go,
        "selector_go": selector_go,
        "efficiency_go": efficiency_go,
        "final": final,
        "reason": reason,
        "scope_warning": (
            "This is a val-128 feasibility decision. A GO must be followed by one "
            "frozen clean experiment on an independent/larger final evaluation set."
        ),
    }

    print("=== SELECTIVE FORECAST FEASIBILITY DECISION ===")
    print(json.dumps(report, indent=2))
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
