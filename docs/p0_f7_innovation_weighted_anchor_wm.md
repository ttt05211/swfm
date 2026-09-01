# P0-F7：Innovation-Weighted Anchor WM

P0-F7 的目的不是再加一个 router，也不是继续堆 loss。P0-F6 已经在约 1000 step 显示出一个关键现象：2s/3s Moving 开始超过 Strong W2Det，但 1s 的 anchor harm 仍把平均收益抵消；同时 Overall mIoU 仍低于 Strong W2Det。P0-F7 优先测试 **WM repair gain 是否还没有被充分开发**。

## 冻结不变

- Strong W2Det occupancy-only causal anchor。
- Frozen Real-Motion decomposition + frozen MSP checkpoint。
- Top-2 `20×20` prediction windows。
- 15% causal MSP write support。
- Full 6-frame occupancy history latent。
- `40×40` surrounding history context。
- P0-F5 occupancy repair endpoint `Z_repair = Enc(repair_occ)`。
- P0-F6 decoder-aware 9-way semantic repair CE。
- OccFM-Fut epoch 196 initialization。
- source noise = 0，sampling NFE = 10。
- inference / writeback 完全不变：support 外 exact Strong W2Det；support 内仍使用现有 dynamic repair contract。

## 只改三件事

### 1. Uniform FM → innovation-energy weighted FM

P0-F6 对完整 `20×20` local latent 使用 uniform FM MSE。P0-F7 仍监督所有 latent cell，但把真正 repair energy 大的位置提高权重。

定义：

```text
DeltaZ = Z_repair - Z_anchor
energy_i = RMS_channel(DeltaZ_i)
mean_energy = mean_i energy_i
focus_i = energy_i / (energy_i + mean_energy)
raw_weight_i = 1 + alpha * focus_i
weight_i = raw_weight_i / mean_i raw_weight_i
```

主实验固定：

```text
alpha = 4.0
```

因此：

- 没有 hard latent mask；
- unchanged cell 仍然被监督，继续约束 anchor preservation；
- repair-energy cell 获得更大的梯度份额；
- 每个 sample 的 weight mean = 1，避免因为 reweighting 整体放大 FM loss / LR。

最终仍只有两个 loss：

```text
L = L_weighted_FM + lambda_sem * L_sem
```

`lambda_sem` 仍按首个 semantic-valid batch 的梯度比例自动校准，只是现在相对于 weighted FM 校准。

### 2. 训练窗口扩大到 8192，仍保持 scene-balanced

**不重新训练 MSP。** 仍使用已经冻结的 MSP checkpoint，只扩大用于 Sparse-WM adaptation 的 causal windows。

现有 `p0_msp_build_dataset.py` 的 train selection 本身就是 deterministic scene-balanced round-robin，因此直接把 `--max-windows` 从 1024 提到 8192。

### 3. Pretrained backbone 小 LR，新/未加载参数正常 LR

官方 OccFM-Fut epoch196 shape-safe load 后，根据 `loaded_keys` 自动分 optimizer group：

```text
upstream checkpoint 成功加载的参数：lr = 2e-6
未从 upstream 加载的参数：       lr = 2e-5
```

默认：

```text
base lr = 2e-5
backbone_lr_scale = 0.1
```

这样不需要手工猜哪些 block 属于 pretrained：包括 shape-mismatch 的 local `pos_embed`、P0-F4 新增的 prior/context 参数等，只要没有被 official checkpoint 实际加载，就自动进入正常 LR group。

---

# 0. 先做 2-step 代码 smoke（可选但推荐）

可以先直接复用旧 1024 cache，只验证代码路径，不作为实验结果：

```bash
python tools/real_motion/train_p0_f7_innovation_weighted_wm.py \
  --train-cache /root/nas/occ/swfm/data/p0_f5_wm_train_top2 \
  --val-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --train-semantic-targets /root/nas/occ/swfm/data/p0_f6_semantic_train.pt \
  --val-semantic-targets /root/nas/occ/swfm/data/p0_f6_semantic_val.pt \
  --upstream-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output-dir /root/nas/occ/swfm/outputs/p0_f7_smoke \
  --steps 2 \
  --batch-size 2 \
  --num-workers 4 \
  --lr 2e-5 \
  --backbone-lr-scale 0.1 \
  --innovation-weight-alpha 4.0 \
  --min-train-windows 1 \
  --val-every 2 \
  --sample-steps 10 \
  --semantic-grad-ratio 0.5 \
  --amp
```

重点确认输出包含：

```text
optimizer_groups ...
semantic_lambda_calibration ...
step=1 ... wfm=... ufm=... iw_max=...
validation ...
```

---

# 1. 构建 scene-balanced 8192-window MSP record cache

这一步只生成更大的 causal window / MSP feature records，**不训练新的 MSP**：

```bash
cd /root/nas/occ/swfm

python tools/real_motion/p0_msp_build_dataset.py \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl \
  --mode train \
  --max-windows 8192 \
  --output /root/nas/occ/swfm/data/msp_probe_train_8192.pt
```

最后确认 summary 中：

```text
num_windows = 8192
selection = scene_balanced_round_robin_v1
```

---

# 2. 用 frozen MSP 构建 8192-window P0-F5-compatible WM cache

```bash
python tools/real_motion/build_p0_f5_cache_direct.py \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_train_8192.pt \
  --msp-checkpoint /root/nas/occ/swfm/outputs/p0_f1_msp_probe/msp_probe_best.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output /root/nas/occ/swfm/data/p0_f7_wm_train_top2_8192 \
  --route-batch-size 16 \
  --vae-batch-size 8 \
  --shard-size 16
```

如果中断，原命令追加：

```text
--resume
```

这里仍使用原 frozen MSP checkpoint，因此 routing policy 没有改；只是把它应用到更多 scene-balanced train windows。

---

# 3. 构建 8192-window semantic sidecar

```bash
python tools/real_motion/build_p0_f6_semantic_targets.py \
  --source-cache /root/nas/occ/swfm/data/p0_f7_wm_train_top2_8192 \
  --output /root/nas/occ/swfm/data/p0_f7_semantic_train_8192.pt \
  --msp-cache /root/nas/occ/swfm/data/msp_probe_train_8192.pt \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --batch-size 8
```

Validation protocol 不动，直接复用原来的：

```text
/root/nas/occ/swfm/data/p0_f5_wm_val_top2
/root/nas/occ/swfm/data/p0_f6_semantic_val.pt
```

---

# 4. 正式训练 P0-F7

第一轮先跑约一个 8192-window epoch：batch=2 时约 4096 optimizer steps，因此先固定 4000 steps，不直接做长训练。

```bash
python tools/real_motion/train_p0_f7_innovation_weighted_wm.py \
  --train-cache /root/nas/occ/swfm/data/p0_f7_wm_train_top2_8192 \
  --val-cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --train-semantic-targets /root/nas/occ/swfm/data/p0_f7_semantic_train_8192.pt \
  --val-semantic-targets /root/nas/occ/swfm/data/p0_f6_semantic_val.pt \
  --upstream-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output-dir /root/nas/occ/swfm/outputs/p0_f7_innovation_weighted_8192 \
  --steps 4000 \
  --batch-size 2 \
  --num-workers 4 \
  --lr 2e-5 \
  --backbone-lr-scale 0.1 \
  --innovation-weight-alpha 4.0 \
  --min-train-windows 8000 \
  --val-every 400 \
  --sample-steps 10 \
  --semantic-grad-ratio 0.5 \
  --amp
```

如果中断，可以从 `latest.pt` 完整续训：

```text
--resume-from /root/nas/occ/swfm/outputs/p0_f7_innovation_weighted_8192/latest.pt
```

`--steps 4000` 仍表示最终训练到 step=4000，不是额外再跑 4000。

---

# 5. Occupancy evaluation

Inference/writeback 与 P0-F6 bit-for-bit 同一逻辑：

```bash
python tools/real_motion/eval_p0_f7_innovation_weighted_wm.py \
  --cache /root/nas/occ/swfm/data/p0_f5_wm_val_top2 \
  --vae-ckpt /root/nas/occ/OccFM-NeurIPS2025-main/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --sparse-ckpt /root/nas/occ/swfm/outputs/p0_f7_innovation_weighted_8192/best.pt \
  --output /root/nas/occ/swfm/outputs/p0_f7_innovation_weighted_8192/eval.json \
  --amp
```

## 最终必须同时看 Overall 和 Moving

冻结 Strong W2Det：

```text
Overall mIoU = 39.7457
Moving mIoU  = 21.3872
```

主结果至少报告：

```text
Overall mIoU / delta Overall vs Strong
Moving mIoU  / delta Moving vs Strong
Overall @ 1s / 2s / 3s
Moving  @ 1s / 2s / 3s
```

P0-F7 的真正成功标准不是 loss 下降，而是：

```text
Delta Overall > 0
Delta Moving  > 0
```

同时重点检查 P0-F6 已经出现的 long-horizon gain 是否被放大：2s/3s Moving 应明显优于 Strong W2Det，而不是只靠减少 1s harm 获得很小的平均正收益。
