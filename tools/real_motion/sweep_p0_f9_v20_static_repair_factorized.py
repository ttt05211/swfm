#!/usr/bin/env python3
"""Successive-halving sweep for V20 factorized Static presence weighting.

Default schedule:
  alpha in {0.02, 0.035, 0.05}
  all candidates -> epoch 3
  top 2          -> epoch 6
  top 1          -> epoch 10

Each stage validates only at its endpoint. Surviving candidates resume exact
model+optimizer state; eliminated candidates stop consuming GPU time.
Selection is diagnostic-only and uses composed semantic mIoU on the fixed
overfit population.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _tag(alpha: float) -> str:
    return f"{float(alpha):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _history(path: Path) -> list[dict]:
    hp = path / "history.json"
    if not hp.exists():
        return []
    return json.loads(hp.read_text(encoding="utf-8"))


def _last_epoch(path: Path) -> int:
    h = _history(path)
    return max((int(r["epoch"]) for r in h), default=0)


def _row_at(path: Path, epoch: int) -> dict:
    for row in _history(path):
        if int(row["epoch"]) == int(epoch):
            if row.get("val") is None:
                raise RuntimeError(
                    f"{path}: epoch {epoch} has no validation result"
                )
            return row
    raise RuntimeError(f"{path}: missing epoch {epoch} result")


def _score(row: dict) -> tuple[float, float, float]:
    v = row["val"]
    d = v["repair_diagnostics"]
    x = v["full_grid_delta"]
    return (
        float(x["mIoU"]),
        float(x["main_1_2_3s"]["mIoU"]),
        float(d["addition_precision"]),
    )


def _compact(alpha: float, stage_epoch: int, row: dict) -> dict:
    v = row["val"]
    d = v["repair_diagnostics"]
    x = v["full_grid_delta"]
    return {
        "alpha": float(alpha),
        "epoch": int(stage_epoch),
        "loss": float(v["loss"]),
        "presence_loss": float(v["presence_loss"]),
        "semantic_loss": float(v["semantic_loss"]),
        "addP": float(d["addition_precision"]),
        "addR": float(d["static_positive_recall"]),
        "semAcc": float(d["semantic_accuracy_on_static_positive"]),
        "mIoU": float(v["full_grid_v18_plus_static_metrics"]["mIoU"]),
        "dmIoU": float(x["mIoU"]),
        "d123mIoU": float(x["main_1_2_3s"]["mIoU"]),
    }


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    print(
        " alpha  ep   loss     pres      sem      addP      addR   "
        "semAcc    dmIoU     d123"
    )
    print("-" * 96)
    for r in rows:
        print(
            f"{r['alpha']:>6.3f} "
            f"{r['epoch']:>3d} "
            f"{r['loss']:>8.4f} "
            f"{r['presence_loss']:>8.4f} "
            f"{r['semantic_loss']:>8.4f} "
            f"{r['addP']:>9.4f} "
            f"{r['addR']:>9.4f} "
            f"{r['semAcc']:>8.4f} "
            f"{r['dmIoU']:>8.4f} "
            f"{r['d123mIoU']:>8.4f}"
        )


def _run_to_epoch(a, alpha: float, target_epoch: int) -> Path:
    out = Path(a.output_root) / f"alpha_{_tag(alpha)}"
    current = _last_epoch(out)
    if current >= int(target_epoch):
        print(
            f"[sweep] alpha={alpha:g}: already at epoch {current}; "
            f"skip training to {target_epoch}",
            flush=True,
        )
        return out

    cmd = [
        sys.executable,
        "-u",
        str(
            Path(__file__).with_name(
                "train_p0_f9_v20_static_repair_factorized.py"
            )
        ),
        "--stage1-train-cache", a.stage1_train_cache,
        "--stage1-val-cache", a.stage1_val_cache,
        "--repair-train-cache", a.repair_train_cache,
        "--repair-val-cache", a.repair_val_cache,
        "--v18-checkpoint", a.v18_checkpoint,
        "--dataroot", a.dataroot,
        "--train-info-pkl", a.train_info_pkl,
        "--val-info-pkl", a.val_info_pkl,
        "--output-dir", str(out),
        "--epochs", str(int(target_epoch)),
        "--overfit-windows", str(int(a.overfit_windows)),
        "--lr", str(float(a.lr)),
        "--weight-decay", str(float(a.weight_decay)),
        "--presence-bias-init", str(float(a.presence_bias_init)),
        "--presence-threshold", str(float(a.presence_threshold)),
        "--presence-positive-mass", str(float(alpha)),
        # Validate only at the endpoint of each successive-halving stage.
        "--val-every", "999999",
        "--tile-size", a.tile_size,
        "--tile-batch-size", str(int(a.tile_batch_size)),
        "--val-tile-batch-size", str(int(a.val_tile_batch_size)),
        "--tile-batch-pad-multiple", str(int(a.tile_batch_pad_multiple)),
        "--prep-workers", str(int(a.prep_workers)),
        "--prefetch", str(int(a.prefetch)),
        "--progress-every", str(int(a.progress_every)),
        "--seed", str(int(a.seed)),
        "--device", a.device,
    ]
    if bool(a.no_amp):
        cmd.append("--no-amp")
    if current > 0:
        latest = out / "latest.pt"
        if not latest.exists():
            raise RuntimeError(
                f"{out}: history exists but latest.pt is missing"
            )
        cmd.extend(["--resume", str(latest)])

    print(
        f"\n[sweep] alpha={alpha:g}: epoch {current} -> "
        f"{target_epoch}",
        flush=True,
    )
    subprocess.run(cmd, check=True)
    if _last_epoch(out) < int(target_epoch):
        raise RuntimeError(
            f"alpha={alpha:g} failed to reach epoch {target_epoch}"
        )
    return out


def _rank(candidates: list[float], dirs: dict[float, Path], epoch: int):
    scored = []
    for alpha in candidates:
        row = _row_at(dirs[alpha], epoch)
        scored.append((alpha, row))
    scored.sort(key=lambda x: _score(x[1]), reverse=True)
    return scored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-train-cache", required=True)
    p.add_argument("--stage1-val-cache", required=True)
    p.add_argument("--repair-train-cache", required=True)
    p.add_argument("--repair-val-cache", required=True)
    p.add_argument("--v18-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--train-info-pkl", required=True)
    p.add_argument("--val-info-pkl", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--alphas", default="0.02,0.035,0.05")
    p.add_argument("--stage1-epoch", type=int, default=3)
    p.add_argument("--stage2-epoch", type=int, default=6)
    p.add_argument("--final-epoch", type=int, default=10)
    p.add_argument("--stage1-keep", type=int, default=2)
    p.add_argument("--overfit-windows", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--presence-bias-init", type=float, default=-4.0)
    p.add_argument("--presence-threshold", type=float, default=0.5)
    p.add_argument("--tile-size", default="32,32,16")
    p.add_argument("--tile-batch-size", type=int, default=256)
    p.add_argument("--val-tile-batch-size", type=int, default=256)
    p.add_argument("--tile-batch-pad-multiple", type=int, default=16)
    p.add_argument("--prep-workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=32)
    p.add_argument("--progress-every", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    alphas = [
        float(x.strip())
        for x in str(a.alphas).split(",")
        if x.strip()
    ]
    if len(alphas) < 2 or len(set(alphas)) != len(alphas):
        raise ValueError("--alphas must contain >=2 unique values")
    if not all(0.0 < x < 1.0 for x in alphas):
        raise ValueError("all alpha values must be in (0,1)")
    e1, e2, ef = (
        int(a.stage1_epoch),
        int(a.stage2_epoch),
        int(a.final_epoch),
    )
    if not (1 <= e1 < e2 < ef):
        raise ValueError(
            "require stage1_epoch < stage2_epoch < final_epoch"
        )
    keep = int(a.stage1_keep)
    if not (1 <= keep < len(alphas)):
        raise ValueError("stage1-keep must be in [1, len(alphas)-1]")

    root = Path(a.output_root)
    root.mkdir(parents=True, exist_ok=True)
    dirs: dict[float, Path] = {}

    # Stage 1: all candidates to e1.
    for alpha in alphas:
        dirs[alpha] = _run_to_epoch(a, alpha, e1)
    ranked1 = _rank(alphas, dirs, e1)
    rows1 = [_compact(x, e1, row) for x, row in ranked1]
    _print_table(f"STAGE 1 @ EPOCH {e1}", rows1)
    survivors1 = [x for x, _ in ranked1[:keep]]
    print(f"survivors -> {survivors1}", flush=True)

    # Stage 2: top-K to e2, keep one.
    for alpha in survivors1:
        _run_to_epoch(a, alpha, e2)
    ranked2 = _rank(survivors1, dirs, e2)
    rows2 = [_compact(x, e2, row) for x, row in ranked2]
    _print_table(f"STAGE 2 @ EPOCH {e2}", rows2)
    winner = ranked2[0][0]
    print(f"winner for final stage -> {winner}", flush=True)

    # Final: one candidate to ef.
    _run_to_epoch(a, winner, ef)
    final_row = _row_at(dirs[winner], ef)
    final_compact = _compact(winner, ef, final_row)
    _print_table(f"FINAL @ EPOCH {ef}", [final_compact])

    summary = {
        "protocol": "p0_f9_v20_static_factorized_alpha_sweep_v1",
        "selection_population": (
            f"fixed overfit{int(a.overfit_windows)} diagnostic"
        ),
        "selection_metric": (
            "composed semantic mIoU; tie-break 1/2/3s mIoU then add precision"
        ),
        "alphas": alphas,
        "schedule": {
            "stage1_epoch": e1,
            "stage1_keep": keep,
            "stage2_epoch": e2,
            "stage2_keep": 1,
            "final_epoch": ef,
            "validation": "endpoint-only per stage",
        },
        "stage1": rows1,
        "stage1_survivors": survivors1,
        "stage2": rows2,
        "winner": float(winner),
        "final": final_compact,
        "winner_output_dir": str(dirs[winner].resolve()),
    }
    sp = root / "sweep_summary.json"
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {sp}", flush=True)


if __name__ == "__main__":
    main()
