# P0-F3：Top-2 MSP Sparse World Model

P0-F2 已冻结主路由：**Top-2 × 20×20 latent windows**。本阶段不再改变 MSP、Top-K、window size 或 support 定义，只回答一个问题：

> 在同一 128-scene-disjoint validation 上，OccFM-Fut 初始化的 local World Model 能否把 Top-2 oracle headroom 转化成真实 occupancy 增益？

## 冻结结构

```text
Past occupancy
→ Real-Motion Decomposition
→ frozen MSP (hidden96, 1 layer, K=4)
→ sum 6 future score maps
→ greedy marginal Top-2 × 20×20 windows
→ local OccFM-Fut-196 operator
→ anchor-centered latent flow
→ scatter into full causal anchor latent
→ one frozen VAE decode
→ outside windows keep KTA/zero-motion anchor
```

训练时 selected window 内使用：

```text
source z0 = Enc(causal KTA/zero-motion anchor)
target z1 = Enc(full future GT)
x_t = (1-t) z0 + t z1
v* = z1 - z0
L = ||v_theta - v*||^2
```

第一轮 `source_noise_std=0`，不增加 CE、Lovasz、ABE、preservation、router 或 ranking loss。Window 本身就是唯一 sparse mask。VAE cache 强制 **FP32 posterior mean**。

## 1. 直接构建 Top-2 latent cache（正式路径）

**不要先落盘完整 prepared dataset。** Full prepared 会保存大量 geometry / diagnostics / support arrays，P0-F3 训练并不需要这些中间资产，容易造成几十 GB 的无效磁盘占用。

新的 direct builder 从冻结 MSP cache 中逐条恢复精确 6+6 window，在内存中完成 Real-Motion/KTA/GT preparation、Frozen MSP Top-2 routing 和 VAE encoding，然后立即丢弃 raw prepared。训练只持久化 `moving_history_latent / anchor_future_latent / gt_future_latent / Top-2 route / trajectory`。

Train：

```bash
python tools/real_motion/build_msp_wm_cache_direct.py \
  --msp-cache data/msp_probe_train_1024.pt \
  --msp-checkpoint outputs/p0_f1_msp_probe/msp_probe_best.pt \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_train_temporal_v3_scene.pkl \
  --vae-ckpt /path/to/OccFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output data/msp_wm_train_top2 \
  --topk 2 \
  --vae-batch-size 4 \
  --shard-size 8
```

Validation：

```bash
python tools/real_motion/build_msp_wm_cache_direct.py \
  --msp-cache data/msp_probe_val_128.pt \
  --msp-checkpoint outputs/p0_f1_msp_probe/msp_probe_best.pt \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --vae-ckpt /path/to/OccFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output data/msp_wm_val_top2 \
  --topk 2 \
  --vae-batch-size 4 \
  --shard-size 4 \
  --include-eval-payload
```

Val 的 compact eval payload 只额外保存评测需要的 semantic/mask arrays，并强制为 `uint8/bool`；因此 evaluator 不再需要 full prepared val cache。

Direct builder 具有以下保护：

- exact windows 直接来自 MSP cache，不重新选择；
- 同一窗口集合按 scene/time 重排以提高 NAS locality；
- Occ3D frame / pose 有小型 LRU；
- shard 通过临时文件原子提交；
- 每个 shard 后写 `.index.partial.json`；
- 中断后用原命令追加 `--resume` 即从已提交 sample IDs 继续；
- 完成时生成标准 `index.json`，训练脚本无需改动。

## 2. 训练

```bash
python tools/real_motion/train_msp_sparse_wm.py \
  --train-cache data/msp_wm_train_top2 \
  --val-cache data/msp_wm_val_top2 \
  --upstream-ckpt /path/to/OccFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --output-dir outputs/p0_f3_top2_sparse_wm \
  --steps 3000 \
  --batch-size 2 \
  --val-every 200 \
  --amp
```

训练日志同时报告 loss、velocity cosine、target RMS、prediction RMS。Best checkpoint 按 scene-disjoint val latent loss 保存，但 **latent loss 下降本身不是 GO 标准**。

## 3. 真实 occupancy 评测

新的 val cache 自带 compact eval payload，因此不需要 `--prepared`：

```bash
python tools/real_motion/eval_msp_sparse_wm.py \
  --cache data/msp_wm_val_top2 \
  --vae-ckpt /path/to/OccFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --sparse-ckpt outputs/p0_f3_top2_sparse_wm/best.pt \
  --output outputs/p0_f3_top2_sparse_wm/eval.json \
  --amp
```

Evaluator 同时给出 causal anchor、trained Top-2 Sparse-WM、同一 Top-2 window 的 GT-repair oracle、Overall mIoU、Moving-mIoU v2 + 1/2/3s、真实 slot compute / unique latent ratio。

## 决策

P0-F3 不再新增 support/source 诊断：

- `delta Moving <= 0`：停止扩 Top-1/3；
- `+3 pp` 左右：有学习信号但偏弱，只继续判断 Top-2；
- `+5 pp` 及以上且 2s/3s 不退：Top-2 生成闭环成立；
- Top-2 成立后，才复用同一方法做 Top-1 / Top-3 efficiency-quality variants。
