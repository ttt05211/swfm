#!/usr/bin/env python3
"""Fail-closed summary for the paired V17 control/backtrace3D fast probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BRANCH = "local_stwm_center_always_source_order"
EVAL_PROTOCOL = "p0_f9_v17_local_stwm_rigid_transport_eval_v3"
TRAIN_PROTOCOL = "p0_f9_v17_backtrace3d_paired_probe_train_v1"
CONTROL_CHECKPOINT_PROTOCOL = "p0_f9_v17_local_spatial_temporal_world_model_v1"
TREATMENT_CHECKPOINT_PROTOCOL = "p0_f9_v17_backtrace3d_query_residual_probe_v1"
LOSS_CONTRACT = "L_pos+L_exist+0.25L_overlap"


def _hrow(mapping, h):
    row = mapping.get(float(h), mapping.get(str(float(h))))
    if row is None:
        raise RuntimeError(f"missing horizon {h}")
    return row


def _require(cond, message):
    if not cond:
        raise RuntimeError("PAIRING VALIDATION FAILED: " + message)


def _load(path, *, expected_arm: str, expected_step: int):
    p = Path(path)
    x = json.loads(p.read_text(encoding="utf-8"))
    _require(x.get("protocol") == EVAL_PROTOCOL, f"{p}: evaluator protocol mismatch")
    cont = x.get("fast_probe_continuation")
    _require(isinstance(cont, dict), f"{p}: missing fast_probe_continuation")
    _require(cont.get("protocol") == TRAIN_PROTOCOL, f"{p}: training protocol mismatch")
    _require(cont.get("arm") == expected_arm, f"{p}: arm={cont.get('arm')} expected={expected_arm}")
    _require(int(cont.get("start_epoch", -1)) == 5, f"{p}: start_epoch must be 5")
    _require(int(cont.get("local_step", -1)) == int(expected_step),
             f"{p}: local_step={cont.get('local_step')} expected={expected_step}")
    _require(int(cont.get("source_batch_size", -1)) == 256, f"{p}: source batch must be 256")
    _require(cont.get("loss_contract") == LOSS_CONTRACT, f"{p}: loss contract mismatch")
    _require(cont.get("scene_ce") is False, f"{p}: scene_ce must be false")
    _require(cont.get("msp_routing") is False, f"{p}: msp_routing must be false")
    _require(bool(cont.get("source_batch_contract")), f"{p}: missing source_batch_contract")
    _require(cont.get("paired_shuffle_seed") is not None, f"{p}: missing paired_shuffle_seed")
    _require(bool(cont.get("resume_checkpoint")), f"{p}: missing resume_checkpoint")
    _require(bool(cont.get("train_cache")), f"{p}: missing train_cache")
    _require(cont.get("branch_seed") is not None, f"{p}: missing branch_seed")

    ck_protocol = x.get("checkpoint_protocol")
    expected_ck = (
        CONTROL_CHECKPOINT_PROTOCOL if expected_arm == "control"
        else TREATMENT_CHECKPOINT_PROTOCOL
    )
    _require(ck_protocol == expected_ck, f"{p}: checkpoint protocol {ck_protocol} != {expected_ck}")
    _require(bool(x.get("backtrace3d_enabled")) == (expected_arm == "backtrace3d"),
             f"{p}: backtrace3d_enabled inconsistent with arm")
    _require(str(x.get("variant")) == "RL", f"{p}: variant must be RL")
    _require(bool(x.get("use_representation")), f"{p}: representation must be enabled")
    _require(abs(float(x.get("overlap_weight", -1.0)) - 0.25) <= 1e-12,
             f"{p}: overlap weight must be 0.25")

    r = (x.get("reports") or {}).get(BRANCH)
    _require(r is not None, f"{p}: missing report branch {BRANCH}")
    mov = r["moving"]
    occ = r["occupancy"]
    return {
        "path": str(p),
        "identity": {
            "evaluator_protocol": x.get("protocol"),
            "checkpoint_protocol": ck_protocol,
            "arm": cont.get("arm"),
            "resume_checkpoint": cont.get("resume_checkpoint"),
            "train_cache": cont.get("train_cache"),
            "start_epoch": int(cont.get("start_epoch")),
            "local_step": int(cont.get("local_step")),
            "paired_shuffle_seed": int(cont.get("paired_shuffle_seed")),
            "branch_seed": int(cont.get("branch_seed")),
            "source_batch_size": int(cont.get("source_batch_size")),
            "source_batch_contract": cont.get("source_batch_contract"),
            "loss_contract": cont.get("loss_contract"),
            "branch_gamma_warmup_steps": int(cont.get("branch_gamma_warmup_steps", -1)),
            "local_stwm_cache": x.get("local_stwm_cache"),
            "p0f9_cache": x.get("p0f9_cache"),
            "num_windows": int(x.get("num_windows", -1)),
            "target_contract": x.get("target_contract"),
            "representation_contract": x.get("representation_contract"),
            "occupancy_iou_contract": x.get("occupancy_iou_contract"),
            "a1_write_order_contract": x.get("a1_write_order_contract"),
            "free_label": int(x.get("free_label", -1)),
            "model_config": x.get("model_config"),
        },
        "IoU": float(occ["IoU"]),
        "mIoU": float(r["overall"]["mIoU"]),
        "Moving": float(mov["mIoU"]),
        "Moving_1s": float(_hrow(mov["per_horizon"], 1.0)["mIoU"]),
        "Moving_2s": float(_hrow(mov["per_horizon"], 2.0)["mIoU"]),
        "Moving_3s": float(_hrow(mov["per_horizon"], 3.0)["mIoU"]),
        "ADE": float((x.get("diagnostics") or {}).get("learned_ade_m", float("nan"))),
        "FDE": float((x.get("diagnostics") or {}).get("learned_fde_m", float("nan"))),
    }


def _validate_all(rows):
    keys_same_across_all = (
        "evaluator_protocol",
        "resume_checkpoint",
        "train_cache",
        "start_epoch",
        "paired_shuffle_seed",
        "branch_seed",
        "source_batch_size",
        "source_batch_contract",
        "loss_contract",
        "branch_gamma_warmup_steps",
        "local_stwm_cache",
        "p0f9_cache",
        "num_windows",
        "target_contract",
        "representation_contract",
        "occupancy_iou_contract",
        "a1_write_order_contract",
        "free_label",
        "model_config",
    )
    names = tuple(rows)
    ref = rows[names[0]]["identity"]
    for name in names[1:]:
        cur = rows[name]["identity"]
        for key in keys_same_across_all:
            _require(cur.get(key) == ref.get(key),
                     f"{name}: {key} differs from {names[0]}")

    for step in (300, 600):
        c = rows[f"control_step{step}"]["identity"]
        t = rows[f"treatment_step{step}"]["identity"]
        _require(c["local_step"] == t["local_step"] == step,
                 f"step{step}: local_step mismatch")
        _require(c["arm"] == "control" and t["arm"] == "backtrace3d",
                 f"step{step}: control/treatment arm mismatch")

    _require(
        rows["control_step300"]["identity"]["resume_checkpoint"]
        == rows["control_step600"]["identity"]["resume_checkpoint"],
        "control 300/600 start checkpoint differs",
    )
    _require(
        rows["treatment_step300"]["identity"]["resume_checkpoint"]
        == rows["treatment_step600"]["identity"]["resume_checkpoint"],
        "treatment 300/600 start checkpoint differs",
    )
    return {
        "status": "PASS",
        "common_resume_checkpoint": ref["resume_checkpoint"],
        "train_cache": ref["train_cache"],
        "paired_shuffle_seed": ref["paired_shuffle_seed"],
        "branch_seed": ref["branch_seed"],
        "source_batch_size": ref["source_batch_size"],
        "source_batch_contract": ref["source_batch_contract"],
        "loss_contract": ref["loss_contract"],
        "validation_cache": ref["local_stwm_cache"],
        "p0f9_cache": ref["p0f9_cache"],
        "num_windows": ref["num_windows"],
        "evaluator_protocol": ref["evaluator_protocol"],
    }


def _delta(t, c):
    return {
        k: float(t[k] - c[k])
        for k in (
            "IoU", "mIoU", "Moving", "Moving_1s", "Moving_2s", "Moving_3s",
            "ADE", "FDE",
        )
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
        "control_step300": _load(a.control_300, expected_arm="control", expected_step=300),
        "treatment_step300": _load(a.treatment_300, expected_arm="backtrace3d", expected_step=300),
        "control_step600": _load(a.control_600, expected_arm="control", expected_step=600),
        "treatment_step600": _load(a.treatment_600, expected_arm="backtrace3d", expected_step=600),
    }
    pairing = _validate_all(rows)
    d300 = _delta(rows["treatment_step300"], rows["control_step300"])
    d600 = _delta(rows["treatment_step600"], rows["control_step600"])
    out = {
        "report_branch": BRANCH,
        "pairing_validation": pairing,
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

    print("=== PAIRING VALIDATION ===")
    print(json.dumps(pairing, indent=2))
    print("\n=== V17 BACKTRACE3D FAST PROBE ===")
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
