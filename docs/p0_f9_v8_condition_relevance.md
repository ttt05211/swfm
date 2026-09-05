# P0-F9 v8 Condition-Relevance Diagnostic

## Purpose

Before changing the context representation, first test whether the existing learned
conditioning paths actually affect v7 deployment.

This is a no-training ablation on the exact v7 step400 checkpoint:

- `full`: normal v7, context + physics condition;
- `no_context`: remove only the 40x40 history-context condition;
- `no_physics`: zero only the Strong-W2Det latent condition inside the WM.

Everything else is held fixed: validation samples, frozen Top-2 routing, per-sample
Gaussian source seed, NFE=10, CFG=1, VAE decoder, Strong-W2Det fallback outside the
sparse windows, original takeover fusion, and Moving-v2 evaluation support.

The diagnostic reports Overall / Moving mIoU plus the same CLEAR/WRITE physical
statistics used by the training-failure decomposition.

## Interpretation

A positive `full - no_context` delta means the existing context path carries useful
information at deployment. This gives evidence that context representation is worth
improving, but it does **not** prove that the proposed ordered residual context will
necessarily help.

A positive `full - no_physics` delta means the learned physics condition contributes
inside the WM windows. This is separate from the exact Strong-W2Det fallback outside
the windows, which remains unchanged in every variant.

If context is nearly irrelevant, do not spend the next training run on a larger
ordered-context module. If context is materially useful and the motion-FM diagnostic
also supports motion weighting, the next training phase may test both mechanisms,
but a uniform-training control is still needed to separate the gain from extra steps.

## Run

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
TRAINED="$ROOT/outputs/p0_f9_v7_native_fm_only_400/step_0400.pt"
OUT="$ROOT/outputs/p0_f9_v8_condition_relevance_128.json"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_v8_condition_relevance.py" \
  --cache "$VAL" \
  --occfm-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --trained-checkpoint "$TRAINED" \
  --output "$OUT" \
  --use-ema \
  --guidance-scale 1.0 \
  --seed 20260904 \
  --amp
```

For smoke only, add `--max-windows 2` and write to a separate output file.

## What to send back

Send these sections:

```text
=== P0-F9 v8 CONDITION RELEVANCE ===
=== PHYSICAL EDITS @ 1s / 2s / 3s ===
=== CONDITION CONTRIBUTION (full - ablated) ===
```
