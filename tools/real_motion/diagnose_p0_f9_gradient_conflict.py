#!/usr/bin/env python3
"""Measure gradient scale/conflict in the *current* P0-F9 Stage-1 objective.

No optimizer step is taken.  A fresh P0-F9 model is initialized from the exact
released OccFM-Fut checkpoint and the current semantic sidecar.  On identical
batches / flow states the script computes gradients of:

- native absolute-future Flow Matching MSE;
- compact semantic CE;
- compact semantic Lovasz;
- the exact current semantic sum CE + lovasz_weight * Lovasz.

Statistics are restricted to parameters actually reused from the released OccFM
checkpoint, so the result answers the concrete question: does the current
semantic objective dominate or oppose the pretrained world-model objective?
The report also partitions inherited parameters into init/down/mid/up/final and
reports an overlapping attention-only view.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.edit_repair import EditTargetCache, horizon_from_flat_indices, lovasz_softmax_flat
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, MSP_WM_CACHE_VERSION_V2, collate_msp_wm
from real_motion.models.p0_f9 import make_p0_f9_model
from real_motion.native_forecast import (
    NUM_FUTURE_FRAMES,
    class_weights_from_edit_cache,
    collapse_occ_logits_to_dynamic,
    crop_coherent_source_noise,
    semantic_targets_for_sample,
)
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.training_diagnostics import gradient_pair_stats
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL
from tools.real_motion.train_p0_f9_native_sparse_forecast import (
    HIST_LAST,
    _scene_balanced_sampler,
    _validate_semantic_sidecar,
    prepare_batch,
    scatter_absolute_endpoint,
)


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _fixed_noise_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)


def _semantic_components(endpoint_full, *, sample_ids, edit_cache, vae, class_weights):
    """Exact P0-F9 equal-horizon CE and Lovasz tensors before scalar weighting."""
    records = edit_cache.get_batch(sample_ids)
    indices = [semantic_targets_for_sample(rec)[0].to(endpoint_full.device) for rec in records]
    sparse_logits = vae.decode_logits_at_flat_indices(endpoint_full, indices)

    logits_rows, target_rows, horizon_rows = [], [], []
    for logits, rec, idx in zip(sparse_logits, records, indices):
        _, target = semantic_targets_for_sample(rec)
        target = target.to(device=endpoint_full.device, dtype=torch.long)
        if int(logits.shape[0]) != int(target.numel()):
            raise ValueError("semantic logits/target length mismatch")
        if target.numel() == 0:
            continue
        collapsed = collapse_occ_logits_to_dynamic(logits)
        horizon = horizon_from_flat_indices(idx).to(device=endpoint_full.device, dtype=torch.long)
        logits_rows.append(collapsed)
        target_rows.append(target)
        horizon_rows.append(horizon)
    if not logits_rows:
        raise RuntimeError("gradient probe batch has no semantic supervision")

    logits = torch.cat(logits_rows, dim=0)
    target = torch.cat(target_rows, dim=0)
    horizon = torch.cat(horizon_rows, dim=0)
    weight = class_weights.to(device=logits.device, dtype=torch.float32)
    ce_rows, lovasz_rows = [], []
    per_horizon = []
    for h in range(NUM_FUTURE_FRAMES):
        mask = horizon == h
        nh = int(mask.sum().item())
        per_horizon.append(nh)
        if nh == 0:
            continue
        lh = logits[mask]
        th = target[mask]
        ce_rows.append(F.cross_entropy(lh.float(), th, weight=weight))
        lovasz_rows.append(lovasz_softmax_flat(F.softmax(lh.float(), dim=-1), th))
    if not ce_rows:
        raise RuntimeError("gradient probe batch has no valid future horizon")
    return torch.stack(ce_rows).mean(), torch.stack(lovasz_rows).mean(), {
        "num_supervised_voxels": int(target.numel()),
        "per_horizon_voxels": per_horizon,
    }


def _combine_grads(a, b, *, scale_a=1.0, scale_b=1.0):
    if len(a) != len(b):
        raise ValueError("gradient list length mismatch")
    out = []
    for ga, gb in zip(a, b):
        if ga is None and gb is None:
            out.append(None)
        elif ga is None:
            out.append(gb.detach() * float(scale_b))
        elif gb is None:
            out.append(ga.detach() * float(scale_a))
        else:
            out.append(ga.detach() * float(scale_a) + gb.detach() * float(scale_b))
    return out


def _grad_norm(grads) -> float:
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float(g.detach().float().square().sum().cpu())
    return math.sqrt(max(sq, 0.0))


def _module_group(name: str) -> str:
    if name.startswith(("init_conv", "init_temporal_attn")):
        return "init"
    if name.startswith("downs."):
        return "down"
    if name.startswith("mid_"):
        return "mid"
    if name.startswith("ups."):
        return "up"
    if name.startswith("final_conv"):
        return "final"
    if name.startswith(("t_embedder", "traj_encoder", "temp_embed", "pos_embed", "time_rel_pos_bias")):
        return "time_traj_pos"
    return "other_inherited"


def _subset(grads, indices):
    return [grads[i] for i in indices]


def _accumulate_pair(acc: dict, stats: dict) -> None:
    acc["dot"] += float(stats["dot"])
    acc["norm_a_sq"] += float(stats["norm_a"]) ** 2
    acc["norm_b_sq"] += float(stats["norm_b"]) ** 2
    acc["opposite_sign_elements"] += int(stats["opposite_sign_elements"])
    acc["joint_nonzero_elements"] += int(stats["joint_nonzero_elements"])
    acc["total_elements"] += int(stats["total_elements"])
    acc["batches"] += 1


def _finish_pair(acc: dict) -> dict:
    na = math.sqrt(max(float(acc["norm_a_sq"]), 0.0))
    nb = math.sqrt(max(float(acc["norm_b_sq"]), 0.0))
    dot = float(acc["dot"])
    return {
        "norm_fm_concat_batches": na,
        "norm_other_concat_batches": nb,
        "cosine_concat_batches": dot / (na * nb) if na > 0 and nb > 0 else float("nan"),
        "other_over_raw_fm_norm": nb / na if na > 0 else float("inf"),
        "opposite_sign_fraction_on_joint_nonzero": (
            float(acc["opposite_sign_elements"] / acc["joint_nonzero_elements"])
            if acc["joint_nonzero_elements"] > 0 else float("nan")
        ),
        "joint_nonzero_elements": int(acc["joint_nonzero_elements"]),
        "total_elements_across_batches": int(acc["total_elements"]),
        "batches": int(acc["batches"]),
    }


def _new_acc():
    return {
        "dot": 0.0,
        "norm_a_sq": 0.0,
        "norm_b_sq": 0.0,
        "opposite_sign_elements": 0,
        "joint_nonzero_elements": 0,
        "total_elements": 0,
        "batches": 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--train-semantic-targets", required=True)
    p.add_argument("--upstream-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batches", type=int, default=16,
                   help="number of no-update training batches to probe")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--fm-weight", type=float, default=0.1,
                   help="current P0-F9 Stage-1 FM scalar, used only for dominance reporting")
    p.add_argument("--lovasz-weight", type=float, default=1.0)
    p.add_argument("--t", type=float, default=0.5,
                   help="controlled CFM time for the probe; ignored with --random-t")
    p.add_argument("--random-t", action="store_true",
                   help="sample t as training does instead of using a fixed controlled t")
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                   help="default FP32 for diagnostic stability; enable only for a memory-constrained smoke")
    a = p.parse_args()
    if min(a.batches, a.batch_size) <= 0:
        raise ValueError("batches and batch-size must be positive")
    if a.fm_weight <= 0 or a.lovasz_weight < 0:
        raise ValueError("fm-weight must be >0 and lovasz-weight must be >=0")
    if not a.random_t and not 0.0 <= a.t <= 1.0:
        raise ValueError("controlled t must lie in [0,1]")

    _seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = MSPWorldModelCacheDataset(a.train_cache)
    if ds.version != MSP_WM_CACHE_VERSION_V2 or ds.metadata.get("protocol") != P0_F9_CACHE_PROTOCOL:
        raise RuntimeError("gradient diagnostic requires the audited P0-F9 native-future v2 train cache")
    sem = EditTargetCache(a.train_semantic_targets)
    _validate_semantic_sidecar(sem, ds, "train")
    class_weights = class_weights_from_edit_cache(sem)

    sampler = _scene_balanced_sampler(ds, seed=a.seed)
    loader = DataLoader(
        ds,
        batch_size=a.batch_size,
        sampler=sampler,
        num_workers=a.num_workers,
        collate_fn=collate_msp_wm,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = make_p0_f9_model(
        20,
        sample_steps=10,
        unconditional_probability=0.0,
        guidance_scale=1.0,
        hist_last=HIST_LAST,
    ).to(device)
    reuse = load_shape_safe(model.transition, a.upstream_ckpt, verbose=True)
    reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)
    loaded = set(reuse.get("loaded_keys", ()))
    named = [(name, p) for name, p in model.transition.named_parameters() if name in loaded and p.requires_grad]
    if not named:
        raise RuntimeError("no inherited OccFM parameters were identified")
    names = [x[0] for x in named]
    params = [x[1] for x in named]

    module_indices = defaultdict(list)
    for i, name in enumerate(names):
        module_indices[_module_group(name)].append(i)
    scopes = {"all_inherited": list(range(len(names)))}
    for group, idx in sorted(module_indices.items()):
        if idx:
            scopes[group] = idx
    attention_idx = [i for i, name in enumerate(names) if "attn" in name.lower() or "attention" in name.lower()]
    if attention_idx:
        scopes["attention_view"] = attention_idx

    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)

    aggregate = {
        component: {scope: _new_acc() for scope in scopes}
        for component in ("semantic", "ce", "lovasz")
    }
    batch_rows = []
    iterator = iter(loader)
    processed = 0
    skipped = 0

    while processed < a.batches:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        prepared = prepare_batch(batch, device)
        if prepared is None:
            skipped += 1
            continue

        global_noise = _fixed_noise_like(prepared["physics_full"], a.seed + 1000 + processed)
        source_noise = crop_coherent_source_noise(global_noise, prepared["plan"], prepared["effective"])
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(a.amp and device.type == "cuda"),
        ):
            kwargs = {}
            if not a.random_t:
                kwargs["t_override"] = float(a.t)
            fm_loss, info = model.flow_loss(
                prepared["history"],
                prepared["target"],
                prepared["physics"],
                history_context=prepared["context"],
                trajectory=prepared["trajectory"],
                window_origins=prepared["origins"],
                source_noise=source_noise,
                return_endpoint=True,
                force_conditioned=True,
                **kwargs,
            )
            endpoint = scatter_absolute_endpoint(info["predicted_endpoint"], prepared)
            ce_loss, lovasz_loss, sem_info = _semantic_components(
                endpoint,
                sample_ids=prepared["sample_ids"],
                edit_cache=sem,
                vae=vae,
                class_weights=class_weights,
            )
        if not all(torch.isfinite(x) for x in (fm_loss, ce_loss, lovasz_loss)):
            raise RuntimeError("non-finite loss in gradient diagnostic")

        g_fm = torch.autograd.grad(fm_loss, params, retain_graph=True, allow_unused=True)
        g_ce = torch.autograd.grad(ce_loss, params, retain_graph=True, allow_unused=True)
        g_lov = torch.autograd.grad(lovasz_loss, params, retain_graph=False, allow_unused=True)
        g_sem = _combine_grads(g_ce, g_lov, scale_a=1.0, scale_b=a.lovasz_weight)
        g_total = _combine_grads(g_sem, g_fm, scale_a=1.0, scale_b=a.fm_weight)

        scope_rows = {}
        for scope, idx in scopes.items():
            sfm = _subset(g_fm, idx)
            ssem = _subset(g_sem, idx)
            sce = _subset(g_ce, idx)
            slov = _subset(g_lov, idx)
            rows = {
                "semantic": gradient_pair_stats(sfm, ssem),
                "ce": gradient_pair_stats(sfm, sce),
                "lovasz": gradient_pair_stats(sfm, slov),
            }
            for component, row in rows.items():
                _accumulate_pair(aggregate[component][scope], row)
            scope_rows[scope] = {
                "fm_norm": rows["semantic"]["norm_a"],
                "semantic_norm": rows["semantic"]["norm_b"],
                "semantic_over_raw_fm": (
                    rows["semantic"]["norm_b"] / rows["semantic"]["norm_a"]
                    if rows["semantic"]["norm_a"] > 0 else float("inf")
                ),
                "semantic_over_weighted_fm": (
                    rows["semantic"]["norm_b"] / (a.fm_weight * rows["semantic"]["norm_a"])
                    if rows["semantic"]["norm_a"] > 0 else float("inf")
                ),
                "fm_semantic_cosine": rows["semantic"]["cosine"],
                "fm_ce_cosine": rows["ce"]["cosine"],
                "fm_lovasz_cosine": rows["lovasz"]["cosine"],
                "opposite_sign_fraction": rows["semantic"]["opposite_sign_fraction_on_joint_nonzero"],
            }

        batch_rows.append({
            "batch": processed,
            "fm_loss": float(fm_loss.detach().cpu()),
            "semantic_ce": float(ce_loss.detach().cpu()),
            "semantic_lovasz": float(lovasz_loss.detach().cpu()),
            "semantic_total": float((ce_loss + a.lovasz_weight * lovasz_loss).detach().cpu()),
            "current_objective_value": float((ce_loss + a.lovasz_weight * lovasz_loss + a.fm_weight * fm_loss).detach().cpu()),
            "fm_prediction_cosine": float(info["cosine"]),
            "num_supervised_voxels": int(sem_info["num_supervised_voxels"]),
            "per_horizon_voxels": sem_info["per_horizon_voxels"],
            "total_inherited_grad_norm": _grad_norm(g_total),
            "scopes": scope_rows,
        })
        processed += 1
        allrow = scope_rows["all_inherited"]
        print(
            f"batch={processed}/{a.batches} fm={batch_rows[-1]['fm_loss']:.6f} "
            f"sem={batch_rows[-1]['semantic_total']:.6f} "
            f"ratio_sem/(wFM)={allrow['semantic_over_weighted_fm']:.3f} "
            f"cos={allrow['fm_semantic_cosine']:+.4f}"
        )
        del endpoint, fm_loss, ce_loss, lovasz_loss, g_fm, g_ce, g_lov, g_sem, g_total

    summary = {}
    for scope, idx in scopes.items():
        sem_pair = _finish_pair(aggregate["semantic"][scope])
        ce_pair = _finish_pair(aggregate["ce"][scope])
        lov_pair = _finish_pair(aggregate["lovasz"][scope])
        summary[scope] = {
            "num_tensors": len(idx),
            "num_parameters": int(sum(params[i].numel() for i in idx)),
            "fm_vs_semantic": sem_pair,
            "fm_vs_ce": ce_pair,
            "fm_vs_lovasz": lov_pair,
            "semantic_over_current_weighted_fm_norm": (
                sem_pair["other_over_raw_fm_norm"] / float(a.fm_weight)
            ),
        }

    def mean(key):
        return float(np.mean([row[key] for row in batch_rows]))

    report = {
        "protocol": "p0_f9_gradient_conflict_probe_v1",
        "provenance": {
            "train_cache": str(Path(a.train_cache).resolve()),
            "train_cache_index_sha256": file_sha256(Path(a.train_cache) / "index.json"),
            "train_semantic_targets": str(Path(a.train_semantic_targets).resolve()),
            "train_semantic_sha256": file_sha256(a.train_semantic_targets),
            "upstream_checkpoint": str(Path(a.upstream_ckpt).resolve()),
            "upstream_checkpoint_sha256": file_sha256(a.upstream_ckpt),
            "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
            "official_transition_reuse_fraction": float(reuse_fraction),
        },
        "probe_contract": {
            "num_batches": int(a.batches),
            "batch_size": int(a.batch_size),
            "scene_balanced_sampler": True,
            "t_mode": "random_training_t" if a.random_t else "fixed_controlled_t",
            "t": None if a.random_t else float(a.t),
            "source_noise": "deterministic_global_gaussian_then_exact_top2_crop",
            "conditioned": True,
            "amp": bool(a.amp and device.type == "cuda"),
            "fm_weight_current": float(a.fm_weight),
            "lovasz_weight_current": float(a.lovasz_weight),
            "semantic_contract": "current compact background+8-dynamic equal-horizon weighted CE+Lovasz",
            "parameter_scope": "only tensors shape-loaded from released OccFM-Fut checkpoint",
        },
        "loss_means": {
            "fm": mean("fm_loss"),
            "semantic_ce": mean("semantic_ce"),
            "semantic_lovasz": mean("semantic_lovasz"),
            "semantic_total": mean("semantic_total"),
            "current_objective": mean("current_objective_value"),
            "fm_prediction_cosine": mean("fm_prediction_cosine"),
        },
        "semantic_class_weights": class_weights.tolist(),
        "scope_summary": summary,
        "per_batch": batch_rows,
        "skipped_empty_batches": skipped,
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== P0-F9 FM / SEMANTIC GRADIENT CONFLICT ===")
    print(
        f"loss means: FM={report['loss_means']['fm']:.6f} "
        f"CE={report['loss_means']['semantic_ce']:.6f} "
        f"Lovasz={report['loss_means']['semantic_lovasz']:.6f} "
        f"Sem={report['loss_means']['semantic_total']:.6f}"
    )
    print(
        f"{'scope':20s} {'||gFM||':>11s} {'||gSem||':>11s} {'Sem/rawFM':>11s} "
        f"{'Sem/(0.1FM)':>13s} {'cos':>9s} {'signOpp':>9s}"
    )
    for scope, row in summary.items():
        pair = row["fm_vs_semantic"]
        print(
            f"{scope:20s} {pair['norm_fm_concat_batches']:11.4g} "
            f"{pair['norm_other_concat_batches']:11.4g} "
            f"{pair['other_over_raw_fm_norm']:11.3f} "
            f"{row['semantic_over_current_weighted_fm_norm']:13.3f} "
            f"{pair['cosine_concat_batches']:9.4f} "
            f"{pair['opposite_sign_fraction_on_joint_nonzero']:9.4f}"
        )
    print("\n=== CE / LOVASZ CONFLICT WITH FM (ALL INHERITED) ===")
    for key in ("fm_vs_ce", "fm_vs_lovasz"):
        row = summary["all_inherited"][key]
        print(
            f"{key:14s} ratio={row['other_over_raw_fm_norm']:.3f} "
            f"cos={row['cosine_concat_batches']:+.4f} "
            f"signOpp={row['opposite_sign_fraction_on_joint_nonzero']:.4f}"
        )
    print("saved", out)


if __name__ == "__main__":
    main()
