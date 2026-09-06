# P0-F9 v9：官方 OccFM 初始化的全集 M 训练

## 目的

这次不是把 M-400 / MCTRL-200 继续往后接，而是做一次正式的 full-data scaling 验证：

```text
released OccFM-Fut epoch=000196
        ↓ shape-safe load inherited transition
P0-F9 sparse Top-2 + Strong-W2Det physics + mean context
        ↓
true-motion weighted native FM, lambda=2
        ↓
all eligible train 6+6 windows
```

不带 ordered context，不带 semantic CE/Lovasz，不带 decoder endpoint loss，不带 selector/margin/action head。

第一阶段只跑到 epoch 5；LR schedule 从启动时就冻结为 10 epochs。epoch 5 之后只有部署指标仍明显改善才从同一 `latest.pt` 续到 epoch 10，不能重启 cosine schedule。

---

## 0. 拉取代码

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

cd "$ROOT"
git pull https://gh-proxy.com/https://github.com/ttt05211/swfm.git main
git log -1 --oneline
```

---

## 1. 构建全部 eligible MSP records

这个步骤显式遍历同一 train temporal-info split 中的全部 6-history + 6-future eligible windows，并与 native chronological iterator 的 sample-id 集合做 exact-set 检查。

可复用已有 4096 MSP records；只复用 feature/target/config/seed/stride 完全相同的 sample。

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
REUSE="$ROOT/data/msp_probe_train_4096.pt"
FULL_MSP="$ROOT/data/msp_probe_train_full_all_eligible.pt"

cd "$ROOT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" "$ROOT/tools/real_motion/build_p0_f9_full_msp_cache.py" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --reuse-cache "$REUSE" \
  --output "$FULL_MSP" \
  --workers 16 \
  --prefetch-windows 64
```

结束时记录实际 `num_windows` / `num_scenes`。不要手填旧实验的窗口数。

---

## 2. 直接构建 full P0-F9 native cache

不再先生成 P0-F7 repair cache。Frozen MSP 直接在 full MSP records 上生成 Top-2 route；Strong-W2Det 只作为 physics condition/fallback；VAE 对 history / physics / absolute GT 使用与 P0-F9 一致的 deterministic posterior samples。

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
"$PY" "$ROOT/tools/real_motion/build_p0_f9_full_native_cache.py" \
  --msp-cache "$FULL_MSP" \
  --msp-checkpoint "$MSP_CKPT" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --vae-ckpt "$VAE" \
  --output "$FULL_TRAIN" \
  --write-budget-ratio 0.15 \
  --route-batch-size 128 \
  --vae-batch-size 16 \
  --prepare-workers 16 \
  --prefetch-windows 64 \
  --shard-size 32 \
  --latent-seed 20260904 \
  --pin-memory
```

中断后原命令加 `--resume`。已经 durable 的 shard 不会重做。

---

## 3. 构建全集 true-motion mask sidecar

仍然使用与 v8 U/M 一样的 Moving-v2 dual-box 定义；这是训练标签，不是推理输入。

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
"$PY" "$ROOT/tools/real_motion/build_p0_f9_v8_motion_mask_sidecar.py" \
  --train-cache "$FULL_TRAIN" \
  --msp-cache "$FULL_MSP" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --output "$MASK" \
  --workers 16 \
  --prefetch-samples 64 \
  --motion-weight-lambda 2
```

记录：

```text
full_latent_motion_fraction
routed_top2_motion_fraction
effective_motion_weight_mass
```

如果 routed motion 比例与 4096 版本出现数量级变化，先停止训练检查数据合同。

---

## 4. 正式 Full-M：先跑到 epoch 5

初始化是 **released OccFM-Fut checkpoint**，不是 v7/M/MCTRL checkpoint。

`--schedule-epochs 10` 从一开始锁定完整 cosine schedule；`--run-until-epoch 5` 只是本次预算闸门。

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v9_full_native_train"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
MASK="$ROOT/data/p0_f9_v9_full_motion_mask.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v9_full_m_official_init"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_v9_full_m.py" \
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
  --amp 2>&1 | tee "$ROOT/outputs/p0_f9_v9_full_m_official_init.log"
```

Formal checkpoint policy：

```text
epoch_0001.pt
epoch_0003.pt
epoch_0005.pt
latest.pt
last.pt
training_report.json
```

没有 `best.pt`，因为 fixed-t FM 不能替代 deployment Overall/Moving 做模型选择。

---

## 5. epoch 1 / 3 / 5 正式部署评估

继续用之前已经冻结的 128-val、seed=20260904、NFE=10、takeover diagnostic。

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
DIR="$ROOT/outputs/p0_f9_v9_full_m_official_init"

for EPOCH in 0001 0003 0005; do
  CUDA_VISIBLE_DEVICES=0 \
  "$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_training_failure.py" \
    --cache "$VAL" \
    --occfm-ckpt "$WM" \
    --vae-ckpt "$VAE" \
    --trained-sparse-ckpt "$DIR/epoch_${EPOCH}.pt" \
    --use-ema \
    --output "$ROOT/outputs/p0_f9_v9_full_m_epoch${EPOCH}_diagnostic_128.json" \
    --seed 20260904 \
    --amp
done
```

重点不是只看 FM：

```text
Overall / Moving
1s / 2s / 3s
WRITE precision / recall
CLEAR recall / wrong-clear / stale
dynamic precision / recall / volume
```

如果主要靠 dynamic volume 增大换 recall，同时 precision/stale 持续恶化，停止本轮。

---

## 6. 只有 epoch 5 仍在改善才续到 epoch 10

严格从 epoch5 的 full resume state 继续，同一个 schedule，不重新 warmup/cosine：

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v9_full_native_train"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
MASK="$ROOT/data/p0_f9_v9_full_motion_mask.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v9_full_m_official_init"
RESUME="$OUT/epoch_0005.pt"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_v9_full_m.py" \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --motion-mask-sidecar "$MASK" \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output-dir "$OUT" \
  --schedule-epochs 10 \
  --run-until-epoch 10 \
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
  --resume-from "$RESUME" \
  --amp
```

如果 epoch3-5 已经表现为 FM 下降但部署长期平台，或者 precision/stale 的 trade-off 被继续放大，就不要跑 epoch10，转向显式 motion correspondence / coherent CLEAR+WRITE 方法。
