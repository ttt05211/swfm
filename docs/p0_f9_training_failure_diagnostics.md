# P0-F9 Training-Failure Diagnostics

## Why this diagnostic exists

The safe-fusion ablation established two facts that must not be conflated:

1. non-destructive write-only fusion can hide a weak proposal and even recover
   Moving-IoU, but it is not a physically complete motion model: a moving object
   must normally CLEAR its old occupancy and WRITE its new occupancy;
2. the trained P0-F9 proposal is substantially worse than the frozen sparse
   OccFM proposal, so the training failure itself must be diagnosed before a new
   training recipe is proposed.

The GT control already proves that correct CLEAR is necessary: GT takeover is
much stronger than GT write-only. Therefore this diagnostic does **not** promote
write-only to the final method. It asks why the current World Model cannot make
correct paired clear/write decisions.

## Diagnostic A: physical CLEAR / KEEP / WRITE errors

Inside the exact frozen MSP write support, GT future occupancy defines events
relative to Strong-W2Det:

```text
CLEAR: anchor dynamic     -> GT non-dynamic
KEEP:  anchor dynamic     -> GT dynamic
WRITE: anchor non-dynamic -> GT dynamic
```

For each proposal the evaluator reports:

- `clear_recall`: fraction of required old-position clears actually cleared;
- `stale_dynamic_rate`: required clears where the proposal still leaves dynamic
  occupancy -- a voxel-level ghost/stale-object proxy;
- `clear_precision`: among proposal clears, fraction that GT truly requires;
- `keep_presence_recall` and `wrong_clear_rate`;
- `write_recall` / `missed_write_rate`;
- `write_precision` and false-write rate;
- dynamic class accuracy on KEEP and WRITE populations;
- proposal dynamic precision/recall;
- proposal/GT dynamic-volume ratio;
- clear/write count balance.

These are voxel-level occupancy diagnostics. The Occ3D semantic payload has no
persistent instance identity, so `stale_dynamic_rate` must not be described as
an exact duplicate-car count. It measures the physical symptom that would create
ghost/double occupancy when old positions are not cleared coherently.

Metrics are stored for all six future frames and separately surfaced at 1 s,
2 s, and 3 s.

## Diagnostic B: where training damage lives

The step checkpoint is partitioned according to the exact official checkpoint
reuse map, not by guessed module names.

Four variants are evaluated:

```text
frozen_sparse
    released OccFM-loaded tensors
    + fresh P0-F9 nonofficial tensors

trained_full
    actual step checkpoint (EMA by default)

trained_loaded_backbone_only
    only tensors that were successfully loaded from released OccFM
    take their trained values
    + every non-loaded tensor reset to frozen/fresh sparse initialization

trained_nonofficial_only
    released OccFM-loaded tensors reset to frozen values
    + only non-loaded tensors take trained values
```

This gives a fail-closed attribution:

- if `trained_loaded_backbone_only` is already bad, drift in inherited OccFM
  weights is sufficient to damage forecasting;
- if `trained_nonofficial_only` is already bad, the learned new/nonofficial
  branches are sufficient to damage forecasting;
- if both partial variants are much better than `trained_full`, the failure is
  primarily an interaction between backbone drift and new conditions.

The JSON records every nonofficial state key. New condition-module keys
(`prior_proj`, `context_proj`, `physics_fusion`) are separated from any other
nonofficial keys rather than silently calling everything an "adapter".

## Diagnostic C: native FM vs real NFE=10 rollout

Every variant is evaluated on the same native absolute-future CFM task at
`t=0.5`:

```text
z0 = one global Gaussian field -> same Top-2 crop
zt = 0.5 z0 + 0.5 zGT
velocity target = zGT - z0
```

Reported:

- FM MSE;
- velocity cosine;
- target RMS;
- prediction RMS / target RMS.

Then the same variant performs the real NFE=10 rollout, frozen VAE decode, and
legacy takeover fusion for Overall/Moving comparison.

Interpretation:

```text
FM worsens + rollout worsens
    -> finetuning damaged the native CFM task itself

FM improves but rollout/physical edits worsen
    -> fixed-t local FM optimization does not translate to the actual rollout;
       objective/geometry/decode mismatch becomes the main suspect
```

## Run

No new cache and no new training are required.

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
TRAINED="$ROOT/outputs/p0_f9_v2_native_sparse_stage1_4096/step_1200.pt"
OUT="$ROOT/outputs/p0_f9_v5_training_failure_diagnostic_128.json"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_training_failure.py" \
  --cache "$VAL" \
  --occfm-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --trained-sparse-ckpt "$TRAINED" \
  --use-ema \
  --output "$OUT" \
  --seed 20260904 \
  --amp
```

For a GPU smoke first, add:

```text
--max-windows 2
```

and use a separate output file.

## What to send back

The terminal prints three compact sections. Send all three:

```text
=== P0-F9 TRAINING FAILURE DECOMPOSITION ===
=== PHYSICAL EDITS @ 1s / 2s / 3s ===
=== PARAMETER DRIFT ===
```

Do not launch another long training run before this diagnostic is interpreted.
