# P0-F9 v8 Motion-Associated FM Error Diagnostic

## Goal

Before starting any motion-weighted training, test whether the remaining native
flow-matching error is actually concentrated near true motion.

This diagnostic compares:

- `frozen_sparse`: released OccFM weights in the P0-F9 sparse architecture;
- `v7_step400`: the successful v7 native-FM-only checkpoint, EMA by default.

No training, decoder, NFE rollout, backward pass, semantic loss, or new cache is
used.

## Controlled contract

- validation cache: the existing audited P0-F9 128-window cache;
- same sample IDs and frozen Top-2 WindowPlan;
- same deterministic 50x50 coherent Gaussian source per sample and both models;
- fixed conditioned behavior, eval mode;
- FM times: `t={0.25, 0.50, 0.75}` by default;
- FP32 error accumulation after the normal BF16/FP32 model forward;
- overlapping Top-2 crops remain duplicated, matching the training objective.

## Motion mask

The primary mask is `eval_gt_moving_support`, i.e. the Moving-v2 dual-box
old/future support. It must be described as **motion-associated support**, not as
an exact instance mask.

Mapping:

```text
[T,200,200,Z]
    -> any over Z
[T,200,200]
    -> exact 4x4 block any pooling, no extra dilation
[T,50,50]
    -> same Top-2 WindowPlan crop
[Nvalid,T,20,20]
```

The mask is used only for statistics. It never enters model conditioning or MSP
routing.

## Reported statistics

For motion-associated and non-motion groups:

- velocity MSE;
- NMSE = group squared error / group target-velocity squared energy;
- target/pred RMS;
- cosine, macro-averaged over valid sample-horizon vectors;
- element/cell counts and empty/zero-norm group counts.

The report also includes:

- motion-cell fraction;
- motion squared-error share;
- effective motion-weight mass for the preview value `lambda=2` only;
- exact recomposition check that motion + non-motion MSE recovers native global
  FM MSE;
- all six future frames, with 1s/2s/3s surfaced in the terminal table;
- frozen -> v7 deltas and ratios for MSE/NMSE/cosine.

`lambda=2` here does **not** start weighted training. It only tells us how much
of the future loss mass motion cells would receive if that candidate were used.

## Run

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
TRAINED="$ROOT/outputs/p0_f9_v7_native_fm_only_400/step_0400.pt"
OUT="$ROOT/outputs/p0_f9_v8_motion_fm_error_128.json"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_v8_motion_fm_error.py" \
  --cache "$VAL" \
  --occfm-ckpt "$WM" \
  --trained-checkpoint "$TRAINED" \
  --output "$OUT" \
  --t-values 0.25 0.50 0.75 \
  --motion-weight-lambda 2 \
  --seed 20260904 \
  --use-ema \
  --amp
```

For a quick GPU smoke, use a separate output path and add:

```text
--max-samples 2
```

## How to decide what comes next

The diagnostic does not tune lambda. It only decides whether motion-weighted FM
has direct evidence.

Strong evidence for candidate M:

- motion MSE/NMSE is consistently higher than non-motion across several `t`;
- v7 improvement is disproportionately larger on non-motion cells, leaving a
  clear motion-associated residual gap;
- mask/recomposition checks are sane.

Weak evidence for candidate M:

- moving and non-moving normalized errors are similar across `t`, or motion cells
  already improve as much as non-motion cells.

If evidence for M is weak, do not start M merely because Moving-IoU is low. The
next cheap diagnostic should instead test whether the current 40x40 context and
physics conditions materially affect deployment before changing the temporal
context structure.
