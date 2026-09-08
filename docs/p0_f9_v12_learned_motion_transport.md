# P0-F9 v12 — Learned Center Motion + Rigid 3D Transport

## Motivation

The v11 no-WM rigid oracle established a decisive representation result on the frozen 128-window validation set:

- Strong/KTA: **39.7457 Overall / 21.3872 Moving**
- GT future center + translation-only rigid source-shape transport: **48.9095 / 45.2670**
- GT future center + GT yaw + SE(2) rigid transport: **49.3201 / 46.8943**

Thus future center accounts for most rigid-transport headroom; yaw is secondary. v12 therefore learns only six future center residuals relative to the causal Strong/KTA anchor plus six existence logits. There is no WM/VAE/FM/semantic occupancy loss in this experiment.

## Causal contract

For every Strong-W2Det t0 dynamic component:

1. recover a six-frame backward component track from occupancy only;
2. form normalized current position/velocity, class, size, KTA-match, history offsets/validity and segment velocities;
3. predict `6 x (dx,dy)` residuals relative to the Strong/KTA constant-velocity future centers;
4. predict six future-existence logits;
5. at evaluation, coherently CLEAR the Strong/KTA copy and WRITE the exact observed t0 3D source shape at the learned center.

nuScenes instance annotations are used only to attach train/validation center/existence labels and to form the GT-center oracle. They are never model inputs.

## 1. Build full-train Strong-source motion cache

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

WINDOWS="$ROOT/data/msp_probe_train_full_all_eligible.pt"
DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
OUT="$ROOT/data/p0_f9_v12_motion_transport_train_full.pt"

cd "$ROOT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" "$ROOT/tools/real_motion/build_p0_f9_motion_transport_cache.py" \
  --window-cache "$WINDOWS" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --output "$OUT" \
  --workers 8 \
  --prefetch-windows 32
```

The user's allocation is 10 CPU cores / 80 GB RAM, so 8 workers is the intended default rather than the earlier cluster-wide 32-worker suggestion.

## 2. Build frozen 128-window validation motion cache

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

WINDOWS="$ROOT/data/msp_probe_val_128.pt"
DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"
OUT="$ROOT/data/p0_f9_v12_motion_transport_val_128.pt"

cd "$ROOT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" "$ROOT/tools/real_motion/build_p0_f9_motion_transport_cache.py" \
  --window-cache "$WINDOWS" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --output "$OUT" \
  --workers 8 \
  --prefetch-windows 32
```

## 3. Train the tiny motion head

This model is small enough that single-GPU training is preferable to DDP; the expensive operation is cache construction and deployment occupancy evaluation, not the MLP itself.

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v12_motion_transport_train_full.pt"
VAL="$ROOT/data/p0_f9_v12_motion_transport_val_128.pt"
OUT="$ROOT/outputs/p0_f9_v12_motion_transport"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_motion_transport.py" \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --output-dir "$OUT" \
  --epochs 20 \
  --batch-size 1024 \
  --hidden-dim 128 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 20260908
```

The residual head is zero-initialized, so before learning the center prediction is exactly the KTA center. `best.pt` is selected only by scene-disjoint validation learned ADE, not by the deployment 128-window Moving-IoU.

## 4. Evaluate through real occupancy transport

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

MOTION="$ROOT/data/p0_f9_v12_motion_transport_val_128.pt"
P0F9="$ROOT/data/p0_f9_v2_wm_val_top2_128"
CKPT="$ROOT/outputs/p0_f9_v12_motion_transport/best.pt"
DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"
OUT="$ROOT/outputs/p0_f9_v12_motion_transport_eval_128.json"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/eval_p0_f9_learned_motion_transport.py" \
  --motion-cache "$MOTION" \
  --p0f9-cache "$P0F9" \
  --checkpoint "$CKPT" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --output "$OUT"
```

The evaluator reports:

- `strong_anchor`;
- `learned_center_always` (trajectory-only ablation, ignores existence logits);
- `learned_center_rigid` (causal learned trajectory + learned existence; primary method);
- `gt_center_rigid` (same representation oracle);
- Overall / Moving / 1s / 2s / 3s Moving;
- KTA vs learned ADE/FDE;
- existence precision/recall/F1;
- fraction of the GT-center rigid Moving headroom recovered by the learned causal model.

## Decision rule

The main causal quantity is

`recovery = (Moving_learned - Moving_Strong) / (Moving_GT-center-rigid - Moving_Strong)`.

If learned rigid transport materially exceeds Strong while ADE/FDE improve, center motion prediction is validated as the next main branch. If ADE/FDE improve but Moving does not, inspect occupancy replacement/discretization before adding model complexity. If neither improves, do not add yaw or WM yet; the learned center representation itself has not passed the gate.
