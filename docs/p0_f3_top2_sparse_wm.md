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

第一轮 `source_noise_std=0`，不增加 CE、Lovasz、ABE、preservation、router 或 ranking loss。Window 本身就是唯一 sparse mask。

VAE cache 强制 **FP32 posterior mean**，避免 anchor 与 GT 的独立 posterior sample noise 被误当成运动 residual。

## 1. 精确构建 MSP 对应 prepared assets

不要用普通 `prepare_nuscenes.py --max-windows 1024/128`：MSP train 是 scene-balanced round-robin，val 是 scene-disjoint midpoint，普通时间顺序不能保证 sample IDs 一致。

从已经冻结的 MSP probe cache 直接读取 `scene/history/t0/future tokens`，只 prepare 同一批 1024 train / 128 val windows：

```bash
python tools/real_motion/prepare_exact_msp_windows.py \
  --msp-cache data/msp_probe_train_1024.pt \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_train_temporal_v3_scene.pkl \
  --output data/prepared_msp_train_1024

python tools/real_motion/prepare_exact_msp_windows.py \
  --msp-cache data/msp_probe_val_128.pt \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --output data/prepared_msp_val_128
```

该 builder：

- 不重新选择窗口；
- 强制复用 MSP cache 内的 `resolved_config`；
- 验证 `sample_id == scene:t0_token`、history 末帧等于 t0、6+6 token 长度；
- 保证最终 prepared sample-ID 集合与 MSP cache 完全一致；
- 为 NAS I/O 按 `scene + t0 timestamp` 重排同一窗口集合；
- 对重复 Occ3D frame / pose 使用小型进程内 LRU cache；
- 已存在 `index.json` 时拒绝覆盖，避免混合旧资产。

## 2. 构建 Top-2 WM cache

Train：

```bash
python tools/real_motion/build_msp_wm_cache.py \
  --prepared data/prepared_msp_train_1024 \
  --msp-cache data/msp_probe_train_1024.pt \
  --msp-checkpoint outputs/p0_f1_msp_probe/msp_probe_best.pt \
  --vae-ckpt /path/to/OccFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output data/msp_wm_train_top2 \
  --topk 2 \
  --vae-batch-size 4
```

Validation：

```bash
python tools/real_motion/build_msp_wm_cache.py \
  --prepared data/prepared_msp_val_128 \
  --msp-cache data/msp_probe_val_128.pt \
  --msp-checkpoint outputs/p0_f1_msp_probe/msp_probe_best.pt \
  --vae-ckpt /path/to/OccFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output data/msp_wm_val_top2 \
  --topk 2 \
  --vae-batch-size 4
```

WM-cache builder 会按 prepared index 重新排序读取，减少 shard/NAS 随机抖动；样本集合不变。`index.json` 必须显示 train/val 都是 `topk=2`、`window_hw=[20,20]`、`vae_mode=mean`，并记录同一个 MSP/VAE SHA256。

## 3. 训练

第一轮固定 3000 steps；这不是 200-epoch full run。

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

训练日志同时报告 loss、velocity cosine、target RMS、prediction RMS。Best checkpoint 仍按 scene-disjoint val latent loss 保存，但 **latent loss 下降本身不是 GO 标准**。

## 4. 真实 occupancy 评测

```bash
python tools/real_motion/eval_msp_sparse_wm.py \
  --cache data/msp_wm_val_top2 \
  --prepared data/prepared_msp_val_128 \
  --vae-ckpt /path/to/OccFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --sparse-ckpt outputs/p0_f3_top2_sparse_wm/best.pt \
  --output outputs/p0_f3_top2_sparse_wm/eval.json \
  --amp
```

Evaluator 同时给出：

- causal KTA/zero-motion anchor；
- trained Top-2 Sparse-WM；
- 同一 Top-2 window 的 GT-repair oracle；
- Overall mIoU；
- Moving-mIoU v2 + 1/2/3s；
- `delta_Moving_vs_anchor`；
- `remaining_Moving_headroom_to_oracle`；
- 实际 slot compute / unique latent ratio。

## 决策

P0-F3 不再新增 support/source 诊断。第一轮只按真实 occupancy 做 GO/STOP：

- `delta Moving <= 0`：Top-2 local WM 没有吃到 headroom，停止扩 Top-1/3；
- `+3 pp` 左右：有学习信号但偏弱，先判断是否值得继续训练 Top-2；
- `+5 pp` 及以上且 2s/3s 不退：Top-2 生成闭环成立；
- Top-2 成立后，才复用同一方法做 Top-1 / Top-3 efficiency-quality variants。

Top-1/Top-3 不在当前训练阶段并行展开。
