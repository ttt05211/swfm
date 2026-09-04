# P0-F9：Physics-Conditioned Native Sparse Future World Model

## 1. 为什么从 P0-F8 转到 P0-F9

P0-F8 已经终止 `KTA anchor -> residual/edit` 路线。Teacher repair endpoint 可以在同一 MSP support 上达到很高的增益，但真实 causal WM endpoint 即使重新训练 fresh edit head，也无法稳定超过 Strong W2Det。结论不是 Flow Matching 的 random-t / NFE 推理机制有问题，而是 **把 pretrained World Model 改造成强 KTA 的错误修补器，没有发挥它原本的 future forecasting 能力**。

P0-F9 因此重新定义 World Model 的任务：

```text
History -> Absolute Future
```

而不是：

```text
Strong KTA -> GT - KTA residual
```

Strong W2Det/KTA 仍然保留，但角色改为：

1. causal physics future condition；
2. MSP 外的 bit-exact deployment fallback；
3. MSP 内 WM 只提供 dynamic future semantics。

论文主张保持不变：

> **Transport what is physically deterministic; generate only what truly moves.**

---

## 2. P0-F9 Stage-1 结构

```text
6-frame occupancy history
          |
          v
   frozen OccFM VAE
          |
      Z_history
          |
          +-------------------------+
          |                         |
          |                    frozen Real-Motion/MSP
          |                         |
          |                    Top-2 20x20 windows
          |                         |
          v                         v
  Native OccFM-style CFM <--- Strong W2Det/KTA future latent
  Gaussian noise -> GT       (condition only, never flow source)
          |
          |  local history 20x20
          |  surrounding history 40x40
          |  ego trajectory
          |  zero-gated physics cross-attention
          v
 absolute future latent windows
          |
          v
 overlap-average scatter into full latent
 (outside windows remains exact Strong W2Det latent)
          |
          v
 frozen VAE decoder
          |
          v
 decoded absolute future occupancy
          |
          +-----------------------------+
          |                             |
 outside MSP write support       inside MSP write support
 exact Strong W2Det              WM dynamic semantics only
          |                             |
          +-------------+---------------+
                        v
                  final future OCC
```

### 关键合同

- Static SE(3)/W1：保留、冻结。
- Dynamic KTA：保留、冻结。
- Strong W2Det：保留、冻结。
- Real-Motion/MSP：保留、冻结。
- Top-2 / 15% write support：保留。
- `anchor_future_latent`：**physics condition + fallback**，不再是 FM source。
- FM source：纯高斯噪声，恢复官方 OccFM native forecasting。
- target：`gt_future_latent`，即 absolute future。
- WM 同时预测六个 0.5s 间隔 future frames；1/2/3s 只用于 benchmark 报告。
- P0-F8 Edit Head / KEEP-CLEAR-WRITE / margin / repair-target latent：全部退出 P0-F9 主方法。

---

## 3. Native Flow Matching

P0-F9 直接复用官方 OccFM 的连续 CFM 语义：

```text
z0 ~ N(0, I)
z1 = absolute GT future latent
zt = (1-t) z0 + t z1
t  = sigmoid(N(0,1))
v* = z1 - z0
```

训练目标的 FM 部分：

```text
L_FM = MSE(v_theta(zt | history, physics, trajectory), z1-z0)
```

Strong W2Det future 只进入 condition：

```text
physics_prior = Enc(Strong-W2Det future)
```

不会进入 `z0`、`z1-z0` 或 residual target。

Stage-1 默认关闭 classifier-free condition dropout：

```text
uncond_prob = 0
guidance_scale = 1
```

原因是当前 **semantic forecasting 是主损失**，第一版不让 20% 样本在语义监督时随机丢失 history/physics condition。代码保留 CFG 能力，以后如果需要可以显式开启。

---

## 4. Physics prior 注入

P0-F9 有两条 condition 路径：

1. inherited zero-init token-wise physics projection；
2. bottleneck lightweight cross-attention。

Cross-attention：

```text
Query = native WM bottleneck token
Key/Value = Strong-W2Det future prior token
```

采用可学习标量 gate：

```text
x' = x + tanh(alpha) * CrossAttn(x, physics)
alpha_init = 0
```

因此初始化时：

```text
x' == x   bit-for-bit at this residual branch
```

随机初始化的 physics module 不会在 step 0 破坏 pretrained OccFM prior。这个 controlled injection 思路来自 GenieDrive/MCA 类条件交互的启发，但实现保持轻量，不复制其 dense 5-layer MCA。

---

## 5. Stage-1 Loss：occupancy semantics 是主任务

P0-F9 不再使用：

```text
FM + small semantic auxiliary
```

而改为：

```text
L = L_semantic + 0.1 * L_FM
```

其中：

```text
L_semantic = weighted CE + Lovasz
```

### Semantic population

复用已经构建好的 P0-F8 sidecar，但**只使用其中的最终 GT result semantics**，不使用 action label：

```text
all previous EDIT locations
+ all correct dynamic KEEP locations
+ bounded hard background KEEP pool
```

18-way frozen VAE decoder logits被概率严格折叠为：

```text
background/non-dynamic + 8 dynamic classes
```

background 使用 non-dynamic 18-way logits 的 `logsumexp`，所以 9-way softmax 与原 18-way probability marginalization 完全一致。

### Mild class balancing

训练 sidecar 上统计 9-way target frequency：

```text
w_c ~ 1 / sqrt(freq_c)
mean(present weights) = 1
clip to [0.5, 2.0]
```

不使用极端 dynamic multiplier。

---

## 6. 从 GenieDrive 借用的训练 trick

第一版直接启用这些低风险基础设施：

### 6.1 Freeze VAE -> 后续再 E2E

Stage-1：VAE encoder/decoder 全冻结。

Stage-2（**本 PR 不启动，只有 Stage-1 确认 WM 出现正系统增益后再实现/运行**）：低 LR 解冻 VAE，让 representation 对 forecasting objective 适配。

### 6.2 EMA

维护 FP32 EMA：

```text
decay = 0.999
current_decay = decay * (1-exp(-updates/2000))
```

deployment evaluator 默认使用 EMA 权重。

### 6.3 Scene-balanced sampling

不是 uniform-over-window，而是：

```text
weight(window_i) = 1 / num_windows(scene_i)
```

使每个 scene 的总采样概率近似相同，避免长 scene/相邻 window 主导训练。

### 6.4 Optimizer / warmup / clipping

```text
AdamW
weight_decay = 1e-2
grad_clip = 5
warmup = first 5%
cosine decay -> min_lr_ratio 0.2
```

差分 LR：

```text
official OccFM loaded params = 2e-5
new physics/context params    = 1e-4
```

### 6.5 Horizon weighting

六个 future frames 第一版统一权重，不复制 GenieDrive 的远期衰减。我们已有证据显示 WM 的边际价值更可能出现在 2-3s，因此不主动压低长 horizon。

### 6.6 Scheduled Sampling

第一版**不使用**。P0-F9 是 six-frame joint CFM，不是逐帧 AR rollout，因此没有直接照搬 GenieDrive 的 scheduled sampling。

---

## 7. Cache upgrade：不重算已有 history/KTA/MSP

现有 P0-F7/P0-F8 v3 cache 已经包含：

```text
full_history_latent
anchor_future_latent
window_origins
window_valid
msp_write_support_latent
trajectory
```

P0-F9 cache builder只新增：

```text
gt_future_latent = VAE_mean(GT future occupancy)
```

并丢弃 `repair_target_latent`。

因此不需要重跑 Strong W2Det、MSP routing 或 history/anchor VAE encoding。

### Train cache

```bash
cd /root/nas/occ/swfm

PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" "$ROOT/tools/real_motion/build_p0_f9_cache_fast.py" \
  --source-cache "$ROOT/data/p0_f7_wm_train_top2_4096" \
  --msp-cache "$ROOT/data/msp_probe_train_4096.pt" \
  --dataroot "$OCCFM/data/nuscenes" \
  --info-pkl "$OCCFM/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --vae-ckpt "$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt" \
  --output "$ROOT/data/p0_f9_wm_train_top2_4096" \
  --vae-batch-size 16 \
  --prepare-workers 16 \
  --prefetch-windows 64 \
  --shard-size 32
```

中断后在相同命令末尾加：

```text
--resume
```

### Val cache

```bash
cd /root/nas/occ/swfm

PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" "$ROOT/tools/real_motion/build_p0_f9_cache_fast.py" \
  --source-cache "$ROOT/data/p0_f5_wm_val_top2" \
  --msp-cache "$ROOT/data/msp_probe_val_128.pt" \
  --dataroot "$OCCFM/data/nuscenes" \
  --info-pkl "$OCCFM/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --vae-ckpt "$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt" \
  --output "$ROOT/data/p0_f9_wm_val_top2_128" \
  --vae-batch-size 16 \
  --prepare-workers 8 \
  --prefetch-windows 32 \
  --shard-size 32
```

Validation source cache 已有 eval payload，因此新 cache 会 bit-for-bit 保留：

```text
eval_future_gt_occ
eval_strong_anchor_occ
eval_gt_moving_support
```

---

## 8. 真实 GPU smoke

先拉目标分支/最终 main 后，确认 cache 和 sidecar 存在。Smoke 只验证真实 BF16、custom FlashAttention backward、frozen decoder sparse supervision、EMA 和 checkpoint 保存。

```bash
cd /root/nas/occ/swfm

PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main
OUT="$ROOT/outputs/p0_f9_stage1_smoke"

rm -rf "$OUT"

"$PY" "$ROOT/tools/real_motion/train_p0_f9_native_sparse_forecast.py" \
  --train-cache "$ROOT/data/p0_f9_wm_train_top2_4096" \
  --val-cache "$ROOT/data/p0_f9_wm_val_top2_128" \
  --train-semantic-targets "$ROOT/data/p0_f8_edit_train_4096.pt" \
  --val-semantic-targets "$ROOT/data/p0_f8_edit_val.pt" \
  --upstream-ckpt "$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt" \
  --vae-ckpt "$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt" \
  --output-dir "$OUT" \
  --steps 2 \
  --batch-size 2 \
  --num-workers 2 \
  --val-every 2 \
  --min-train-windows 4000 \
  --amp
```

---

## 9. Stage-1 正式 4096-window run

```bash
cd /root/nas/occ/swfm

PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main
OUT="$ROOT/outputs/p0_f9_native_sparse_stage1_4096"

"$PY" "$ROOT/tools/real_motion/train_p0_f9_native_sparse_forecast.py" \
  --train-cache "$ROOT/data/p0_f9_wm_train_top2_4096" \
  --val-cache "$ROOT/data/p0_f9_wm_val_top2_128" \
  --train-semantic-targets "$ROOT/data/p0_f8_edit_train_4096.pt" \
  --val-semantic-targets "$ROOT/data/p0_f8_edit_val.pt" \
  --upstream-ckpt "$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt" \
  --vae-ckpt "$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt" \
  --output-dir "$OUT" \
  --steps 1200 \
  --batch-size 8 \
  --num-workers 4 \
  --wm-lr 2e-5 \
  --new-lr 1e-4 \
  --weight-decay 1e-2 \
  --warmup-fraction 0.05 \
  --min-lr-ratio 0.2 \
  --fm-weight 0.1 \
  --lovasz-weight 1.0 \
  --sample-steps 10 \
  --uncond-prob 0.0 \
  --guidance-scale 1.0 \
  --ema-decay 0.999 \
  --val-every 200 \
  --min-train-windows 4000 \
  --amp
```

自动保存：

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
training_report.json
```

`best.pt` 按 EMA validation surrogate 保存；**最终论文 checkpoint 仍必须按真实 deployment Overall + Moving 决定。**

---

## 10. Deployment evaluation

```bash
cd /root/nas/occ/swfm

PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main
OUT="$ROOT/outputs/p0_f9_native_sparse_stage1_4096"
VAL="$ROOT/data/p0_f9_wm_val_top2_128"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"

for STEP in 200 400 600 800 1000 1200; do
  "$PY" "$ROOT/tools/real_motion/eval_p0_f9_native_sparse_forecast.py" \
    --cache "$VAL" \
    --vae-ckpt "$VAE" \
    --sparse-ckpt "$OUT/step_$(printf '%04d' $STEP).pt" \
    --output "$OUT/eval_${STEP}.json" \
    --use-ema \
    --amp
done
```

必须汇总：

```text
Step | Overall | DeltaOverall | Moving | DeltaMoving
     | O@1s/O@2s/O@3s | M@1s/M@2s/M@3s
```

核心成功条件：

```text
DeltaOverall >= 0
DeltaMoving  > 0
```

Strong W2Det 固定 baseline：

```text
Overall = 39.7457
Moving  = 21.3872
```

---

## 11. Official native OccFM baseline

这不是额外诊断，而是论文必须有的基线：在同一 128-window split 上测官方 OccFM native `History -> Future` 能力。

```bash
cd /root/nas/occ/swfm

PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

"$PY" "$ROOT/tools/real_motion/eval_p0_f9_native_occfm_baseline.py" \
  --cache "$ROOT/data/p0_f9_wm_val_top2_128" \
  --occfm-ckpt "$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt" \
  --output "$ROOT/outputs/p0_f9_official_occfm_native_128.json"
```

最终至少比较：

```text
Strong W2Det
Official native OccFM
P0-F9 physics-conditioned sparse WM
Same-support GT oracle
```

---

## 12. Stage-2 E2E 预留合同（当前不启动）

只有 Stage-1 出现真实正系统增益后，再进入 E2E：

```text
WM              train
Physics modules train
VAE decoder     low LR
VAE encoder     lower LR
```

建议第一版 LR：

```text
physics/new = 5e-5 ~ 1e-4
WM          = 1e-5 ~ 2e-5
VAE decoder = 2e-6
VAE encoder = 1e-6
```

Stage-2 必须在线读取 occupancy 并重新 encode；一旦 encoder 更新，Stage-1 latent cache 就不能作为训练输入真值继续使用。

Stage-2 目标仍以 decoded occupancy forecasting 为主，FM 只作小正则，同时加入轻量 GT VAE reconstruction protection。不要把 Stage-2 变回 latent-residual fitting。

---

## 13. 当前明确不要重新引入

P0-F9 不再增加：

```text
KTA residual target
repair_target_latent
innovation magnitude weighting
KEEP/CLEAR/WRITE edit head
KEEP margin sweep
utility router / Need-Score
anchor-error classifier
Flow random-t vs NFE diagnostic
```

如果 P0-F9 Stage-1 仍不能在真实 deployment 上让 WM 产生正边际收益，应重新评估 sparse fusion/task formulation，而不是继续围绕旧 residual 路线调 loss。
