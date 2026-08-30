# P0-F4 — Strong W2Det + Full-History Context Sparse World Model

## Frozen method

P0-F4 fixes three concrete P0-F3 problems without reopening the routing study:

1. **Strong causal anchor**: restore the earlier occupancy-only W2Det contract instead of gating KTA by the Real-Motion MOVING mask.
2. **Full historical context**: VAE-encode all six native history occupancy frames. A selected 20×20 future window sees its local 20×20 history plus a surrounding 40×40 full-history context crop.
3. **Protected sparse write-back**: Top-2 decides where expensive future transition runs; horizon-wise MSP support decides where decoded dynamic semantics may modify the strong anchor.

Real-Motion decomposition and the frozen MSP are unchanged. No new router, selector, ABE, occupancy CE, or auxiliary loss is added. The first P0-F4 run still uses only latent flow MSE.

### World-model state

- Source / inference start: `z_anchor = Enc(Strong-W2Det future)`.
- Target: `z_gt = Enc(full future GT)`.
- Local flow state: `x_t = (1-t) z_anchor + t z_gt`.
- Velocity target: `z_gt - z_anchor`.
- Local prediction size: 20×20.
- Surrounding history context: 40×40.
- Full latent grid: 50×50.
- Pretrained transition: official OccFM-Fut epoch 196.
- Source noise: 0.
- NFE: 10.

The 40×40 context branch is intentionally small: temporal mean of the full-history crop followed by one zero-initialized stride-2 3×3 projection. It is conditioning, not a second dense future world model.

## Strong W2Det contract

`real_motion/strong_w2det.py` ports the older occupancy-only baseline:

- dynamic classes `[2,3,4,5,6,7,9,10]`;
- per-class 3D connected components;
- minimum 6 voxels;
- t-1 ↔ t0 same-class mutual-nearest matching;
- 25 m/s match gate;
- causal backward-difference velocity;
- unmatched / filtered dynamic occupancy uses zero object velocity;
- static background uses W1 inverse ego transport + 5×5×1 majority fill;
- future GT semantics / boxes are never used by the anchor.

## New cache contract

Version: `msp_topk_strong_w2det_fullctx_wm_v2`

Stored training fields:

- `full_history_latent`
- `anchor_future_latent`
- `gt_future_latent`
- `window_origins`
- `window_valid`
- `msp_write_support_latent`
- `trajectory`

Validation additionally stores compact semantic GT, strong-anchor semantics, and frozen Moving-mIoU-v2 support.

## Server commands

Update first:

```bash
cd /root/nas/occ/swfm
git pull https://gh-proxy.com/https://github.com/ttt05211/swfm.git main
git log -1 --oneline
```

### 1. Build train cache

```bash
python tools/real_motion/build_p0_f4_cache_direct.py \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_train_1024.pt \
  --msp-checkpoint /root/nas/occ/swfm/outputs/p0_f1_msp_probe/msp_probe_best.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output /root/nas/occ/swfm/data/p0_f4_wm_train_top2 \
  --topk 2 \
  --write-budget-ratio 0.15 \
  --vae-batch-size 4 \
  --shard-size 8
```

If interrupted, rerun the identical command with `--resume`.

### 2. Build validation cache

```bash
python tools/real_motion/build_p0_f4_cache_direct.py \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_val_128.pt \
  --msp-checkpoint /root/nas/occ/swfm/outputs/p0_f1_msp_probe/msp_probe_best.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output /root/nas/occ/swfm/data/p0_f4_wm_val_top2 \
  --topk 2 \
  --write-budget-ratio 0.15 \
  --vae-batch-size 4 \
  --shard-size 4 \
  --include-eval-payload
```

### 3. Mandatory strong-anchor preflight

This is not another model diagnostic. It verifies the baseline contract before training so P0-F4 cannot silently regress to the P0-F3 14.12 Moving anchor.

```bash
python tools/real_motion/eval_p0_f4_anchor.py \
  --cache /root/nas/occ/swfm/data/p0_f4_wm_val_top2 \
  --output /root/nas/occ/swfm/outputs/p0_f4_anchor_preflight.json
```

Read:

- `strong_w2det_anchor`
- `same_support_gt_repair_oracle`
- `oracle_delta_Moving_vs_strong_anchor`

Only after this confirms the restored strong anchor should WM training start.

### 4. Train Top-2

```bash
python tools/real_motion/train_p0_f4_sparse_wm.py \
  --train-cache /root/nas/occ/swfm/data/p0_f4_wm_train_top2 \
  --val-cache /root/nas/occ/swfm/data/p0_f4_wm_val_top2 \
  --upstream-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --output-dir /root/nas/occ/swfm/outputs/p0_f4_top2_sparse_wm \
  --steps 3000 \
  --batch-size 2 \
  --num-workers 4 \
  --lr 2e-5 \
  --val-every 200 \
  --sample-steps 10 \
  --amp
```

Batch size may be increased if GPU memory allows; keep LR unchanged for the first run.

### 5. Real occupancy evaluation

```bash
python tools/real_motion/eval_p0_f4_sparse_wm.py \
  --cache /root/nas/occ/swfm/data/p0_f4_wm_val_top2 \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --sparse-ckpt /root/nas/occ/swfm/outputs/p0_f4_top2_sparse_wm/best.pt \
  --output /root/nas/occ/swfm/outputs/p0_f4_top2_sparse_wm/eval.json \
  --amp
```

Final comparison:

- `strong_w2det_anchor`
- `trained_sparse_wm`
- `same_support_gt_repair_oracle`
- `delta_Moving_vs_strong_anchor`

The GO criterion remains: the trained model must improve the **strong** causal anchor, not the old weak anchor. Top-1 / Top-3 are not trained before Top-2 passes.
