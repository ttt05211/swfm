# P0-F9 Training Input + Gradient Diagnostics

These diagnostics are the required evidence before another P0-F9/P0-F10
training run.  They do **not** alter the model, loss, routing, checkpoint, or
optimizer and do not launch training.

## A. Training-distribution diagnostic

Purpose: separate three sources of distribution shift that are currently
conflated:

1. native OccFM 6+6 training-window population;
2. MSP selection of hard/true-motion windows;
3. Top-2 spatial crop exposure (including duplicate exposure in overlap).

The report measures the full 18-class Occ3D frequency for:

```text
native_full_future
selected_full_future
selected_top2_union
selected_top2_effective
```

`selected_top2_effective` is the closest semantic population to what the sparse
WM actually sees: if the two routed windows overlap, overlap voxels are counted
twice because they enter two crop slots during training.

The same JSON also records the current P0-F9 compact semantic sidecar
(background + 8 dynamic classes) and the exact inverse-sqrt/clamped weights used
by Stage-1.

### Full run

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
MSP="$ROOT/data/msp_probe_train_4096.pt"
SEM="$ROOT/data/p0_f8_edit_train_4096.pt"
DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
OUT="$ROOT/outputs/p0_f9_v6_training_distribution_4096.json"

"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_train_distribution.py" \
  --train-cache "$TRAIN" \
  --msp-cache "$MSP" \
  --semantic-targets "$SEM" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --output "$OUT"
```

This run is CPU / storage-I/O bound.  It reads each native future frame once and
performs no VAE/WM forward pass.  A smoke can add `--max-selected 32`, but the
native reference is still the full eligible 6+6 window population so that its
denominator remains meaningful.

### What matters

The terminal prints:

```text
=== P0-F9 TRAINING DISTRIBUTION ===
=== TOP-2 OCCUPIED-CLASS ENRICHMENT VS NATIVE ===
=== CURRENT P0-F9 COMPACT SEMANTIC POOL ===
```

Interpretation:

- `selected_full_future / native_full_future` measures selection bias;
- `selected_top2_union / selected_full_future` measures routing bias;
- `selected_top2_effective / selected_top2_union` exposes extra overlap/crop-slot
  exposure;
- large dynamic-class enrichment in Top-2 means class weights must not be copied
  from full-scene Occ3D papers without re-estimation on our effective population.

## B. FM / semantic gradient-conflict diagnostic

Purpose: directly measure which loss currently controls the inherited OccFM
backbone at step 0.

The probe loads the released OccFM-Fut checkpoint and computes, on the exact
current P0-F9 semantic objective:

```text
g_FM      = grad(native absolute-future FM MSE)
g_CE      = grad(current weighted compact semantic CE)
g_Lovasz  = grad(current compact Lovasz)
g_sem     = g_CE + lambda_lovasz * g_Lovasz
```

No optimizer step is executed.  Only tensors successfully shape-loaded from the
released OccFM checkpoint are scored.  New physics/context parameters are not
allowed to hide backbone conflict.

It reports both

```text
||g_sem|| / ||g_FM||
```

and the ratio under the current Stage-1 objective

```text
||g_sem|| / (0.1 * ||g_FM||)
```

plus cosine similarity and elementwise opposite-sign fraction.  The same
statistics are shown for init/down/mid/up/final/time-traj-pos groups and an
overlapping attention-only view.  CE and Lovasz are also separated so we can
identify which semantic term causes conflict.

### Recommended controlled run

Use FP32 and fixed `t=0.5` first.  This removes random-t noise and answers the
loss-scale question cleanly.

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
SEM="$ROOT/data/p0_f8_edit_train_4096.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v6_gradient_conflict_fixed_t05_16b.json"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_gradient_conflict.py" \
  --train-cache "$TRAIN" \
  --train-semantic-targets "$SEM" \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output "$OUT" \
  --batches 16 \
  --batch-size 4 \
  --fm-weight 0.1 \
  --lovasz-weight 1.0 \
  --t 0.5 \
  --no-amp
```

A 2-batch smoke is:

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
SEM="$ROOT/data/p0_f8_edit_train_4096.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v6_gradient_conflict_smoke2.json"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_gradient_conflict.py" \
  --train-cache "$TRAIN" \
  --train-semantic-targets "$SEM" \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output "$OUT" \
  --batches 2 \
  --batch-size 2 \
  --fm-weight 0.1 \
  --lovasz-weight 1.0 \
  --t 0.5 \
  --no-amp
```

Only after the controlled fixed-t probe is understood should `--random-t` be
used as a secondary check of the actual random-t training distribution.

### What matters

The terminal prints:

```text
=== P0-F9 FM / SEMANTIC GRADIENT CONFLICT ===
=== CE / LOVASZ CONFLICT WITH FM (ALL INHERITED) ===
```

Key fields:

- `Sem/(0.1FM) >> 1`: the current semantic objective dominates the inherited
  backbone despite FM being the intended native world-model task;
- negative `cos`: semantic optimization directly opposes native FM locally;
- high `signOpp`: widespread elementwise disagreement even if global cosine is
  mildly positive;
- CE-vs-FM and Lovasz-vs-FM show which auxiliary term is responsible;
- one module group with much worse conflict identifies where fine-tuning first
  damages the pretrained transition.

## Decision rule

Do not choose a new class weight, FM weight, backbone LR, Focal/Lovasz/GeoScal
combination, or replay ratio until both diagnostics are available.  Their job is
to establish the effective semantic population and the actual gradient scale,
so the next training recipe is derived from measured failure modes rather than a
new parameter sweep.
