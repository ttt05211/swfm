# P0-F8: True-Motion Anchor-Relative Edit World Model

P0-F8 does **not** add another external router.  Strong W2Det, frozen Real-Motion/MSP Top-2 routing, 15% write support, full 6-frame history, 40×40 context, frozen VAE, and NFE=10 remain unchanged.

The change is the decoder-aware objective and deployment parameterization:

```text
Strong W2Det anchor + full-history World Model
                  ↓
         predicted future latent
                  ↓ frozen VAE
         sparse semantic evidence
                  + exact anchor class
                  + horizon embedding
                  ↓
          KEEP / CLEAR / WRITE
                  ↓
       MSP-protected sparse writeback
```

## Why this is different from P0-F6/F7

P0-F6 predicts absolute dynamic semantics.  Even when the anchor is already correct, the WM must generate the correct class again.  P0-F8 instead predicts the minimal action relative to the exact Strong-W2Det anchor:

```text
anchor car / GT car  -> KEEP
anchor car / GT free -> CLEAR
anchor free / GT car -> WRITE car
anchor car / GT bus  -> WRITE bus
```

The edit head starts with a positive KEEP bias, so untrained behavior is close to exact anchor preservation.

## Class imbalance handling

Every EDIT voxel is always retained.  KEEP examples are subsampled at a fixed ratio (default 1:1) with priority:

1. correct dynamic anchor voxels on true-moving GT support;
2. other correct dynamic anchor voxels;
3. compact background KEEP examples near actual edits, then elsewhere in support.

True-motion GT is used only to prioritize training negatives.  It is never an inference input.

## Loss

The latent flow loss returns to the P0-F6 uniform MSE:

```text
L = L_FM + lambda_edit * L_edit
```

with

```text
L_edit = CE(KEEP/CLEAR/WRITE) + beta * Lovasz(result semantic)
```

Lovasz is not applied to action IDs directly.  Action probabilities are first marginalized through the anchor into the resulting 9-way semantic distribution:

```text
background + 8 dynamic classes
```

so the IoU surrogate is closer to the occupancy that would exist after applying the edit.

`lambda_edit` is gradient-calibrated once and then frozen.  Default `beta=0.5`.

---

## 1. Build the 4096-window training edit sidecar

The expensive WM latent cache is reused.  Only exact Strong-W2Det / GT edit labels are rebuilt.

```bash
cd /root/nas/occ/swfm

python tools/real_motion/build_p0_f8_edit_targets_fast.py \
  --source-cache /root/nas/occ/swfm/data/p0_f7_wm_train_top2_4096 \
  --output /root/nas/occ/swfm/data/p0_f8_edit_train_4096.pt \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_train_4096.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl \
  --workers 16 \
  --prefetch-samples 64 \
  --checkpoint-every 512 \
  --easy-keep-limit 4096
```

If interrupted, append:

```text
--resume
```

This step does **not** rebuild VAE latents and does not retrain MSP.

## 2. Build the validation edit sidecar

The frozen 128-window validation cache already contains GT, exact Strong-W2Det anchor, and true-moving support, so no raw recomputation is needed:

```bash
python tools/real_motion/build_p0_f8_edit_targets_fast.py \
  --source-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --output /root/nas/occ/swfm/data/p0_f8_edit_val.pt \
  --workers 8 \
  --prefetch-samples 32
```

## 3. Train P0-F8

Main run:

```bash
python tools/real_motion/train_p0_f8_anchor_relative_edit_wm.py \
  --train-cache /root/nas/occ/swfm/data/p0_f7_wm_train_top2_4096 \
  --val-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --train-edit-targets /root/nas/occ/swfm/data/p0_f8_edit_train_4096.pt \
  --val-edit-targets /root/nas/occ/swfm/data/p0_f8_edit_val.pt \
  --upstream-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output-dir /root/nas/occ/swfm/outputs/p0_f8_anchor_relative_edit_4096 \
  --steps 1200 \
  --batch-size 8 \
  --num-workers 4 \
  --lr 2e-5 \
  --backbone-lr-scale 1.0 \
  --keep-ratio 1.0 \
  --keep-when-no-edit 64 \
  --keep-bias 2.0 \
  --lovasz-weight 0.5 \
  --edit-grad-ratio 0.5 \
  --min-train-windows 4000 \
  --val-every 200 \
  --sample-steps 10 \
  --amp
```

P0-F8 automatically saves:

```text
step_0200.pt
step_0400.pt
step_0600.pt
step_0800.pt
step_1000.pt
step_1200.pt
latest.pt
best.pt
last.pt
```

No manual checkpoint copy is needed.

Important training logs:

```text
edit_lambda_calibration ...
step=... fm=... edit=... ce=... lovasz=...
edit_acc=...
false_edit=...
E/K=.../...
```

The intended behavior is:

- `edit_accuracy` rises;
- `false_edit_rate` stays low;
- EDIT and KEEP counts remain approximately balanced;
- deployment metrics, not validation objective alone, choose the final checkpoint.

## 4. Evaluate 200–1200

```bash
OUT=/root/nas/occ/swfm/outputs/p0_f8_anchor_relative_edit_4096
VAL=/root/nas/occ/swfm/data/p0_f5_wm_val_top2
VAE=/root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt

for STEP in 200 400 600 800 1000 1200; do
  python tools/real_motion/eval_p0_f8_anchor_relative_edit_wm.py \
    --cache "$VAL" \
    --vae-ckpt "$VAE" \
    --sparse-ckpt "$OUT/step_$(printf '%04d' $STEP).pt" \
    --output "$OUT/eval_${STEP}.json" \
    --amp
done
```

Each report includes:

```text
Overall mIoU / delta Overall
Moving-mIoU / delta Moving
Overall @1s/2s/3s
Moving @1s/2s/3s
action histogram
changed fraction of causal support
same-support GT oracle
```

The success criterion remains:

```text
Delta Overall >= 0
Delta Moving  > 0
```

and the main question is whether explicit KEEP/CLEAR/WRITE increases repair gain without recreating the 1s anchor harm seen in P0-F6/F7.
