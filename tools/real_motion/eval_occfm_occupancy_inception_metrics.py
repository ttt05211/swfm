#!/usr/bin/env python3
"""Occupancy FID/KID/FVD for frozen V18 predictions using OccFM's released feature extractors.

The goal is comparison, not a new metric.  We mirror the public OccFM
inception-style feature contract as closely as possible:

FVD:
  - 6-frame semantic occupancy clip [6,200,200,16].
  - OccFM's released temporal 3D-VAE (occfm_3dvae, epoch 40).
  - sampled latent -> adaptive avg pool 5x5 -> flatten all six frames.
  - Frechet distance between prediction and GT clip-feature distributions.

FID/KID:
  - individual future occupancy frames at 1s/2s/3s (indices 1/3/5).
  - OccFM's released single-frame occupancy VAE (occfm_vae, epoch 100).
  - sampled latent -> adaptive avg pool 5x5 -> flatten.
  - FID per horizon and their arithmetic mean.
  - KID is the standard unbiased degree-3 polynomial-kernel subset estimator,
    reported per horizon and averaged.  The exact public OccFM KID estimator is
    not released, so KID is explicitly labeled as our fixed standard estimator.

IMPORTANT public-code audit:
OccFM's released tools/test_fid.py currently overwrites the loaded prediction
with a randomly shuffled GT tensor before feature extraction.  That line is
appropriate only for the paper's "Reorder GT" temporal-consistency sanity
baseline, not for evaluating a model prediction.  This evaluator deliberately
uses the actual exported prediction.  It can optionally compute a correct
time-axis Reorder-GT sanity FVD separately.

For strict published-number comparison, rerun every compared model's occupancy
clips through this same corrected evaluator and the same released feature
extractor checkpoints.  Do not mix our corrected values with values produced by
the public script's current prediction-overwrite behavior.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import scipy.linalg as sla
import torch
import torch.nn.functional as F

PROTOCOL = "p0_f9_occfm_occupancy_inception_metrics_v1"
HORIZONS = ((1.0, 1), (2.0, 3), (3.0, 5))


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_occfm_model(occfm_root: Path, cfg_path: Path, ckpt_path: Path, device):
    root = str(occfm_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from easydict import EasyDict
    from forecast.config import cfg_from_yaml_file
    from forecast.models import build_network

    cfg = EasyDict()
    cfg.ROOT_DIR = occfm_root.resolve()
    cfg.LOCAL_RANK = 0
    cfg_from_yaml_file(str(cfg_path), cfg)
    model = build_network(
        model_cfg=cfg.MODEL,
        loss_cfg=cfg.LOSS,
        cache_mode=cfg.CACHE_MODE,
    ).to(device)
    model.eval()
    status = model.recover_training(str(ckpt_path))
    return model, cfg, status


def _nn_latent(model, semantic_occ: torch.Tensor) -> torch.Tensor:
    ret = model.nn_forward({"semantic_occ": semantic_occ})
    data = ret[0] if isinstance(ret, tuple) else ret
    if "sampled_features" not in data:
        raise RuntimeError("OccFM feature extractor did not return sampled_features")
    return data["sampled_features"]


def _pool_single_frame(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 4:
        raise RuntimeError(f"single-frame latent expected [B,C,H,W], got {tuple(latent.shape)}")
    return F.adaptive_avg_pool2d(latent, (5, 5)).flatten(1)


def _pool_six_frame(latent: torch.Tensor) -> torch.Tensor:
    # OccFM's public test_fid.py feeds [6,H,W,D] as the model batch.  The
    # temporal blocks are configured for six elements and return [6,C,h,w].
    if latent.ndim != 4 or latent.shape[0] != 6:
        raise RuntimeError(f"FVD latent expected [6,C,H,W], got {tuple(latent.shape)}")
    return F.adaptive_avg_pool2d(latent, (5, 5)).reshape(6, -1).flatten()


def _frechet(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"feature shape mismatch: {x.shape} vs {y.shape}")
    if min(len(x), len(y)) < 2:
        raise ValueError("Frechet distance requires at least two samples")
    mx, my = x.mean(axis=0), y.mean(axis=0)
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    covmean, info = sla.sqrtm(cx @ cy, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        off = np.eye(cx.shape[0]) * eps
        covmean, info = sla.sqrtm((cx + off) @ (cy + off), disp=False)
    if np.iscomplexobj(covmean):
        max_imag = float(np.max(np.abs(covmean.imag)))
        if max_imag > 1e-3:
            raise RuntimeError(f"large imaginary sqrtm residual: {max_imag}")
        covmean = covmean.real
    mean_term = float(np.sum((mx - my) ** 2))
    cov_term = float(np.trace(cx + cy - 2.0 * covmean))
    value = mean_term + cov_term
    return {
        "value": float(value),
        "mean_term": mean_term,
        "covariance_term": cov_term,
        "feature_dim": int(x.shape[1]),
        "samples_pred": int(len(x)),
        "samples_gt": int(len(y)),
        "sqrtm_info": float(info),
    }


def _poly_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = float(a.shape[1])
    return (a @ b.T / d + 1.0) ** 3


def _kid_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return float("nan")
    kxx = _poly_kernel(x, x)
    kyy = _poly_kernel(y, y)
    kxy = _poly_kernel(x, y)
    xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    xy = kxy.mean()
    return float(xx + yy - 2.0 * xy)


def _kid_subsets(
    x: np.ndarray,
    y: np.ndarray,
    *,
    subsets: int,
    subset_size: int,
    seed: int,
) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = min(int(subset_size), len(x), len(y))
    if m < 2:
        raise ValueError("KID subset needs >=2 samples")
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(subsets)):
        ix = rng.choice(len(x), size=m, replace=False)
        iy = rng.choice(len(y), size=m, replace=False)
        vals.append(_kid_unbiased(x[ix], y[iy]))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "value": float(arr.mean()),
        "std_over_subsets": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "subsets": int(subsets),
        "subset_size": int(m),
        "feature_dim": int(x.shape[1]),
        "estimator": "unbiased_polynomial_kernel_degree3_gamma_1_over_dim_coef1",
    }


def _clip_files(clip_dir: Path):
    files = sorted(clip_dir.glob("clip_*.npz"))
    if not files:
        raise FileNotFoundError(f"no clip_*.npz under {clip_dir}")
    return files


def _load_clip(path: Path):
    with np.load(path) as x:
        pred = np.asarray(x["pred"], dtype=np.uint8)
        gt = np.asarray(x["gt"], dtype=np.uint8)
    if pred.shape != (6, 200, 200, 16) or gt.shape != pred.shape:
        raise RuntimeError(f"{path}: unexpected pred/gt shapes {pred.shape}/{gt.shape}")
    return pred, gt


def _extract_fvd(
    model,
    files,
    device,
    *,
    reorder_gt_sanity: bool,
    reorder_seed: int,
):
    pred_rows, gt_rows, reorder_rows = [], [], []
    rng = np.random.default_rng(int(reorder_seed))
    started = time.perf_counter()
    with torch.inference_mode():
        for i, path in enumerate(files):
            pred, gt = _load_clip(path)
            # Mirror OccFM public ordering: prediction feature first, then GT.
            pt = torch.from_numpy(pred).to(device)
            gt_t = torch.from_numpy(gt).to(device)
            pf = _pool_six_frame(_nn_latent(model, pt)).float().cpu().numpy()
            gf = _pool_six_frame(_nn_latent(model, gt_t)).float().cpu().numpy()
            pred_rows.append(pf)
            gt_rows.append(gf)
            if reorder_gt_sanity:
                perm = rng.permutation(6)
                rg = torch.from_numpy(gt[perm]).to(device)
                rf = _pool_six_frame(_nn_latent(model, rg)).float().cpu().numpy()
                reorder_rows.append(rf)
            if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(files):
                dt = max(time.perf_counter() - started, 1e-9)
                print(f"FVD features {i+1}/{len(files)} rate={(i+1)/dt:.2f} clips/s", flush=True)
    return (
        np.stack(pred_rows).astype(np.float32),
        np.stack(gt_rows).astype(np.float32),
        np.stack(reorder_rows).astype(np.float32) if reorder_rows else None,
    )


def _extract_fid(
    model,
    files,
    device,
):
    pred_rows = {str(h): [] for h, _ in HORIZONS}
    gt_rows = {str(h): [] for h, _ in HORIZONS}
    started = time.perf_counter()
    with torch.inference_mode():
        for i, path in enumerate(files):
            pred, gt = _load_clip(path)
            # Stack the three paper horizons as a batch; this model has no
            # temporal block, so this is equivalent to three independent calls.
            ids = [idx for _, idx in HORIZONS]
            pt = torch.from_numpy(pred[ids]).to(device)
            gt_t = torch.from_numpy(gt[ids]).to(device)
            pf = _pool_single_frame(_nn_latent(model, pt)).float().cpu().numpy()
            gf = _pool_single_frame(_nn_latent(model, gt_t)).float().cpu().numpy()
            for j, (h, _) in enumerate(HORIZONS):
                pred_rows[str(h)].append(pf[j])
                gt_rows[str(h)].append(gf[j])
            if i == 0 or (i + 1) % 200 == 0 or i + 1 == len(files):
                dt = max(time.perf_counter() - started, 1e-9)
                print(f"FID/KID features {i+1}/{len(files)} rate={(i+1)/dt:.2f} clips/s", flush=True)
    pred_out = {h: np.stack(v).astype(np.float32) for h, v in pred_rows.items()}
    gt_out = {h: np.stack(v).astype(np.float32) for h, v in gt_rows.items()}
    return pred_out, gt_out


def _feature_cache_save(path: Path, fvd_pred, fvd_gt, reorder, fid_pred, fid_gt):
    payload = {
        "fvd_pred": fvd_pred,
        "fvd_gt": fvd_gt,
    }
    if reorder is not None:
        payload["fvd_reorder_gt"] = reorder
    for h, _ in HORIZONS:
        key = str(h).replace(".", "p")
        payload[f"fid_pred_{key}"] = fid_pred[str(h)]
        payload[f"fid_gt_{key}"] = fid_gt[str(h)]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _feature_cache_load(path: Path, need_reorder: bool):
    with np.load(path) as x:
        fvd_pred = np.asarray(x["fvd_pred"])
        fvd_gt = np.asarray(x["fvd_gt"])
        reorder = np.asarray(x["fvd_reorder_gt"]) if "fvd_reorder_gt" in x.files else None
        if need_reorder and reorder is None:
            return None
        fid_pred, fid_gt = {}, {}
        for h, _ in HORIZONS:
            key = str(h).replace(".", "p")
            fid_pred[str(h)] = np.asarray(x[f"fid_pred_{key}"])
            fid_gt[str(h)] = np.asarray(x[f"fid_gt_{key}"])
    return fvd_pred, fvd_gt, reorder, fid_pred, fid_gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--occfm-root", required=True)
    p.add_argument("--clip-dir", required=True)
    p.add_argument("--fvd-cfg", required=True)
    p.add_argument("--fvd-ckpt", required=True)
    p.add_argument("--fid-cfg", required=True)
    p.add_argument("--fid-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--feature-cache", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--kid-subsets", type=int, default=100)
    p.add_argument("--kid-subset-size", type=int, default=1000)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--reorder-gt-sanity", action="store_true")
    a = p.parse_args()

    root = Path(a.occfm_root).resolve()
    clip_dir = Path(a.clip_dir).resolve()
    files = _clip_files(clip_dir)
    if int(a.max_clips) > 0:
        files = files[: min(len(files), int(a.max_clips))]
    if len(files) < 2:
        raise RuntimeError("need at least two occupancy clips")

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _seed_all(int(a.seed))

    cache_path = Path(a.feature_cache).resolve() if a.feature_cache else None
    cached = (
        _feature_cache_load(cache_path, bool(a.reorder_gt_sanity))
        if cache_path is not None and cache_path.exists()
        else None
    )

    if cached is None:
        print("Loading OccFM temporal occupancy feature extractor ...", flush=True)
        fvd_model, fvd_cfg, _ = _load_occfm_model(
            root, Path(a.fvd_cfg).resolve(), Path(a.fvd_ckpt).resolve(), device
        )
        fvd_pred, fvd_gt, reorder = _extract_fvd(
            fvd_model, files, device,
            reorder_gt_sanity=bool(a.reorder_gt_sanity),
            reorder_seed=int(a.seed) + 17,
        )
        del fvd_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Re-seed before loading/extracting the second feature family so the
        # result is deterministic and independent of FVD clip count.
        _seed_all(int(a.seed))
        print("Loading OccFM single-frame occupancy feature extractor ...", flush=True)
        fid_model, fid_cfg, _ = _load_occfm_model(
            root, Path(a.fid_cfg).resolve(), Path(a.fid_ckpt).resolve(), device
        )
        fid_pred, fid_gt = _extract_fid(fid_model, files, device)
        del fid_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if cache_path is not None:
            _feature_cache_save(
                cache_path, fvd_pred, fvd_gt, reorder, fid_pred, fid_gt
            )
            print(f"saved feature cache {cache_path}", flush=True)
    else:
        fvd_pred, fvd_gt, reorder, fid_pred, fid_gt = cached
        print(f"loaded feature cache {cache_path}", flush=True)

    if len(fvd_pred) != len(files):
        raise RuntimeError(
            f"feature cache samples {len(fvd_pred)} != selected clips {len(files)}"
        )

    print("Computing FVD Frechet distance ...", flush=True)
    fvd = _frechet(fvd_pred, fvd_gt)
    fvd["x1e3"] = float(fvd["value"] * 1e3)

    reorder_fvd = None
    if reorder is not None:
        reorder_fvd = _frechet(reorder, fvd_gt)
        reorder_fvd["x1e3"] = float(reorder_fvd["value"] * 1e3)

    fid_report, kid_report = {}, {}
    fid_vals, kid_vals = [], []
    for hi, (h, _) in enumerate(HORIZONS):
        hs = str(h)
        print(f"Computing FID/KID at {h:.1f}s ...", flush=True)
        fr = _frechet(fid_pred[hs], fid_gt[hs])
        kr = _kid_subsets(
            fid_pred[hs],
            fid_gt[hs],
            subsets=int(a.kid_subsets),
            subset_size=int(a.kid_subset_size),
            seed=int(a.seed) + 100 + hi,
        )
        kr["x1e2"] = float(kr["value"] * 1e2)
        fid_report[hs] = fr
        kid_report[hs] = kr
        fid_vals.append(float(fr["value"]))
        kid_vals.append(float(kr["value"]))

    result = {
        "protocol": PROTOCOL,
        "num_clips": int(len(files)),
        "clip_shape": [6, 200, 200, 16],
        "feature_extractors": {
            "fvd": {
                "cfg": str(Path(a.fvd_cfg).resolve()),
                "checkpoint": str(Path(a.fvd_ckpt).resolve()),
                "contract": (
                    "OccFM released temporal occupancy 3D-VAE; sampled latent; "
                    "adaptive_avg_pool2d 5x5; six-frame flatten"
                ),
            },
            "fid_kid": {
                "cfg": str(Path(a.fid_cfg).resolve()),
                "checkpoint": str(Path(a.fid_ckpt).resolve()),
                "contract": (
                    "OccFM released single-frame occupancy VAE; sampled latent; "
                    "adaptive_avg_pool2d 5x5"
                ),
            },
        },
        "seed": int(a.seed),
        "fvd_3s_6frames": fvd,
        "reorder_gt_fvd_3s_6frames": reorder_fvd,
        "fid": {
            "per_horizon": fid_report,
            "average_1s_2s_3s": float(np.mean(fid_vals)),
        },
        "kid": {
            "per_horizon": kid_report,
            "average_1s_2s_3s": float(np.mean(kid_vals)),
            "average_x1e2": float(np.mean(kid_vals) * 1e2),
            "warning": (
                "OccFM does not release its exact KID estimator code. This is a "
                "fixed standard unbiased polynomial-kernel subset KID. Recompute "
                "all compared methods with this evaluator before direct comparison."
            ),
        },
        "public_occfm_script_audit": {
            "file": "tools/test_fid.py",
            "issue": (
                "released script assigns pred = gt[:, indices, ...] before feature "
                "extraction, overwriting the model prediction; this evaluator does "
                "not reproduce that prediction-overwrite line"
            ),
            "fairness_rule": (
                "Use actual predictions for every method and the same released "
                "feature-extractor checkpoints. Published values are contextual "
                "until reproduced with the corrected common evaluator."
            ),
        },
        "feature_cache": str(cache_path) if cache_path is not None else None,
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== OCCUPANCY INCEPTION METRICS ===")
    print(json.dumps(result, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
