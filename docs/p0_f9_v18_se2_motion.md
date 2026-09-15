# P0-F9 V18-SE2: source-centred planar rigid-motion continuation

## Scope

This branch performs one controlled change on top of the frozen V17-RL epoch-5
checkpoint:

- keep the historical V17 causal inputs, encoder, future queries, XY residual
  head, existence head and A1 hard scene composition;
- add one scalar relative-yaw prediction per future horizon;
- change the treatment arm's XY supervision to the displacement of the observed
  Strong source centroid under the same GT SE(2) rigid transform;
- train with differentiable source-footprint SE(2) overlap.

It does **not** add selector routing, 3-D crops, voxel flow, scene CE, native
footprint replacement, Lovasz, confidence gating or KTA-relative safety loss.

## Frozen geometry contract

All training targets are expressed in the current (t0) ego frame.

For GT box centres `a0`, `ah`, observed Strong source centroid `cs`, and
GT relative planar rotation `R`:

```text
ds = (ah - a0) + (R - I) (cs - a0)
p' = R (p - cs) + cs + ds
```

This is exactly equivalent to

```text
p' = R (p - a0) + ah
```

without aligning the observed source centroid to the absolute annotation-box
centre.

The treatment XY residual target is

```text
source_residual = source_displacement - KTA_displacement
```

Relative yaw is computed from annotation heading vectors after expressing both
headings in the frozen t0 frame.  The model predicts one scalar angle and the
yaw head is zero-initialized, so treatment step 0 is the old translation-only
V17 behavior.

## First-run yaw classes

Yaw is enabled only for the fixed deployment-known IDs:

```text
2 bicycle
3 bus
4 car
5 construction vehicle
6 motorcycle
9 trailer
10 truck
```

Pedestrian (7) remains translation-only. The class mask is used in both soft
training transport and hard deployment transport. Future yaw-label validity is
supervision-only and never a deployment gate.

## Treatment loss

```text
L_Y =
    SmoothL1(source XY residual)
  + BCE(existence)
  + lambda_yaw * (1 - cos(pred_yaw - gt_yaw))
  + 0.25 * (1 - SoftIoU(predicted SE2 footprint, GT SE2 footprint))
```

`lambda_yaw` is fixed once from a small pre-training loss-scale diagnostic and
is passed explicitly to the trainer.

## Cache augmentation

The SE2 upgrader reuses the existing V17 tensors and reads only nuScenes
annotations/poses to add:

- `target_source_displacement_xy_m`
- `target_source_residual_xy_m`
- `target_yaw_rad`
- `yaw_label_valid`
- `se2_target_valid`
- `yaw_enabled`

It does not rebuild semantic tubes or causal features.

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN_V17="$ROOT/data/p0_f9_v17_local_stwm_train_full_nativefp.pt"
VAL_V17="$ROOT/data/p0_f9_v17_local_stwm_val_128_nativefp.pt"
TRAIN_SE2="$ROOT/data/p0_f9_v18_se2_train_full.pt"
VAL_SE2="$ROOT/data/p0_f9_v18_se2_val_128.pt"

DATAROOT="$OCCFM/data/nuscenes"
TRAIN_INFO="$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
VAL_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"

cd "$ROOT"

"$PY" tools/real_motion/augment_p0_f9_v17_se2_labels.py \
  --input "$TRAIN_V17" --output "$TRAIN_SE2" \
  --dataroot "$DATAROOT" --info-pkl "$TRAIN_INFO"

"$PY" tools/real_motion/augment_p0_f9_v17_se2_labels.py \
  --input "$VAL_V17" --output "$VAL_SE2" \
  --dataroot "$DATAROOT" --info-pkl "$VAL_INFO"
```

## Pre-training checks

```bash
"$PY" -m pytest -q \
  tests/test_p0_f9_v18_se2.py \
  tests/test_p0_f9_v17_local_stwm.py
```

The SE2 tests cover exact box/source pivot equivalence, zero-yaw reduction,
ego-yaw invariance, V17->V18 step-0 identity, differentiable SE(2) overlap,
finite-difference yaw gradients and disabled-class yaw blocking.

## One-time yaw-weight calibration

```bash
E5="$ROOT/outputs/p0_f9_v17_stwm_RL/epoch_0005.pt"
OUT="$ROOT/outputs/p0_f9_v18_se2"

CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  tools/real_motion/train_p0_f9_v18_se2_pair.py \
  --train-cache "$TRAIN_SE2" \
  --val-cache "$VAL_SE2" \
  --start-checkpoint "$E5" \
  --output-dir "$OUT/calibration" \
  --arm Y --calibrate-only
```

Record the single reported `recommended_yaw_weight`; do not sweep it.

## Paired continuation

Control:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  tools/real_motion/train_p0_f9_v18_se2_pair.py \
  --train-cache "$TRAIN_SE2" \
  --val-cache "$VAL_SE2" \
  --start-checkpoint "$E5" \
  --output-dir "$OUT/C" \
  --arm C --steps 600 --save-steps 300,600 --seed 20260915
```

Treatment:

```bash
YAW_WEIGHT=<fixed_calibrated_value>

CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  tools/real_motion/train_p0_f9_v18_se2_pair.py \
  --train-cache "$TRAIN_SE2" \
  --val-cache "$VAL_SE2" \
  --start-checkpoint "$E5" \
  --output-dir "$OUT/Y" \
  --arm Y --steps 600 --save-steps 300,600 \
  --yaw-weight "$YAW_WEIGHT" --seed 20260915
```

Both arms restore the same historical AdamW state, use the original cosine LR
schedule for inherited parameters and consume the same deterministic paired
source sequence. The yaw head is appended at the checkpoint's current LR and
weight decay.

## Hard A1 evaluation

Use:

```bash
tools/real_motion/eval_p0_f9_v18_se2.py
```

for each of C/Y at step 300 and 600 with:

- `--local-stwm-cache "$VAL_SE2"`
- `--p0f9-cache "$ROOT/data/p0_f9_v2_wm_val_top2_128"`
- `--checkpoint <C-or-Y-checkpoint>`
- `--dataroot "$DATAROOT"`
- `--info-pkl "$VAL_INFO"`
- `--output <json>`

The evaluator reports hard A1 IoU/mIoU/Moving-mIoU, source-centre ADE/FDE for Y,
wrapped yaw MAE versus zero-yaw, per-horizon yaw MAE, and fixed diagnostic
turning subsets based on 3-s GT yaw: straight <5 deg, mild 5--15 deg, strong
>=15 deg.

## Decision rule

Always compare both:

```text
Y(step) - C(step)
Y(step) - frozen V17-RL epoch5
```

A lower yaw MAE without occupancy improvement is not sufficient to claim
success. Failure to learn yaw within 600 steps rejects this continuation budget,
not rotation in general.

KTA-relative anti-regret remains deferred until this C/Y deployment result.
