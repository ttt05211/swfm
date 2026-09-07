# P0-F9 v9 Full-M：双卡 DDP 正式流程

## 当前缓存阶段

`msp_probe_train_full_all_eligible.pt` 只代表 **full MSP records 已完成**。它还不是训练直接读取的 latent cache。

训练前还需要：

1. full P0-F9 native latent cache；
2. 与该 full native cache 一一对应的 true-motion mask sidecar。

DDP 使用 `build_p0_f9_full_native_cache_ddp_ready.py`。它先调用原 audited full-native builder，随后只给 `index.json` 增加 `zero_route_sample_ids` 元数据，保证 DDP 每个 rank 的 local batch 都有有效 Top-2 路由；不会修改任何 latent、route、target 或 sample 顺序。

## 1. Full native cache

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

FULL_MSP="$ROOT/data/msp_probe_train_full_all_eligible.pt"
MSP_CKPT="$ROOT/outputs/p0_f1_msp_probe/msp_probe_best.pt"
DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
FULL_TRAIN="$ROOT/data/p0_f9_v9_full_native_train"

cd "$ROOT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/build_p0_f9_full_native_cache_ddp_ready.py" \
  --msp-cache "$FULL_MSP" \
  --msp-checkpoint "$MSP_CKPT" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --vae-ckpt "$VAE" \
  --output "$FULL_TRAIN" \
  --write-budget-ratio 0.15 \
  --route-batch-size 256 \
  --vae-batch-size 24 \
  --prepare-workers 32 \
  --prefetch-windows 128 \
  --shard-size 64 \
  --latent-seed 20260904 \
  --pin-memory
```

如果 24 的 VAE batch OOM，用同一命令增加 `--resume` 并把 `--vae-batch-size 24` 改成 `16`。已经 durable 的 shards 不重做。

结束时必须有：

```text
/root/nas/occ/swfm/data/p0_f9_v9_full_native_train/index.json
```

并记录 `num_samples / zero_route_count / zero_route_fraction`。

## 2. Full true-motion mask sidecar

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

FULL_TRAIN="$ROOT/data/p0_f9_v9_full_native_train"
FULL_MSP="$ROOT/data/msp_probe_train_full_all_eligible.pt"
DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
MASK="$ROOT/data/p0_f9_v9_full_motion_mask.pt"

cd "$ROOT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" "$ROOT/tools/real_motion/build_p0_f9_v8_motion_mask_sidecar.py" \
  --train-cache "$FULL_TRAIN" \
  --msp-cache "$FULL_MSP" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --output "$MASK" \
  --workers 32 \
  --prefetch-samples 128 \
  --motion-weight-lambda 2
```

记录 `full_latent_motion_fraction / routed_top2_motion_fraction / effective_motion_weight_mass`。

## 3. 双卡 DDP：先跑 epoch 1/3/5

正式 DDP 启动入口是 `run_p0_f9_v9_full_m_ddp.py`。

`--batch-size` 在 DDP 版本中是 **global batch size**。因此两张 GPU 配置 `--batch-size 8` 时，每个 rank 实际 batch=4，global batch 仍与原单卡实验的 8 一致。

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v9_full_native_train"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
MASK="$ROOT/data/p0_f9_v9_full_motion_mask.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v9_full_m_official_init_ddp2"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  "$ROOT/tools/real_motion/run_p0_f9_v9_full_m_ddp.py" \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --motion-mask-sidecar "$MASK" \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output-dir "$OUT" \
  --schedule-epochs 10 \
  --run-until-epoch 5 \
  --milestone-epochs 1 3 5 10 \
  --batch-size 8 \
  --num-workers 4 \
  --wm-lr 2e-5 \
  --new-lr 1e-4 \
  --weight-decay 1e-2 \
  --warmup-fraction 0.05 \
  --min-lr-ratio 0.2 \
  --sample-steps 10 \
  --uncond-prob 0 \
  --guidance-scale 1 \
  --ema-decay 0.999 \
  --grad-clip 5 \
  --motion-weight-lambda 2 \
  --seed 20260906 \
  --amp 2>&1 | tee "$ROOT/outputs/p0_f9_v9_full_m_official_init_ddp2.log"
```

DDP checkpoint 仍保存为 **unwrapped P0-F9 state_dict**，所以原来的部署 evaluator 可以直接读取：

```text
epoch_0001.pt
epoch_0003.pt
epoch_0005.pt
latest.pt
last.pt
training_report.json
```

只有 rank0 做 validation / checkpoint 写盘。

## 4. epoch 1/3/5 deployment evaluation

沿用冻结的 128-val / NFE=10 / seed=20260904：

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
DIR="$ROOT/outputs/p0_f9_v9_full_m_official_init_ddp2"

cd "$ROOT"
for EPOCH in 0001 0003 0005; do
  CUDA_VISIBLE_DEVICES=0 \
  "$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_training_failure.py" \
    --cache "$VAL" \
    --occfm-ckpt "$WM" \
    --vae-ckpt "$VAE" \
    --trained-sparse-ckpt "$DIR/epoch_${EPOCH}.pt" \
    --use-ema \
    --output "$ROOT/outputs/p0_f9_v9_full_m_ddp_epoch${EPOCH}_diagnostic_128.json" \
    --seed 20260904 \
    --amp
done
```

只有 epoch 5 的真实 deployment / physical metrics 仍在改善，才从 `epoch_0005.pt` 用同样两卡、world size=2、global batch=8 严格 resume 到 epoch10。
