# P0-F9：Physics-Conditioned Native Sparse Future World Model（Audited v2）

## 1. 方法定位

P0-F9 终止 `KTA anchor -> residual/edit` 主路线，把 World Model 恢复为真正的 future forecasting：

```text
History -> Absolute Future
```

Strong W2Det/KTA 仍保留，但只承担：

1. causal physics future condition；
2. MSP 外 bit-exact deployment fallback；
3. MSP 内动态语义写回时的保护基线。

论文主张不变：

> **Transport what is physically deterministic; generate only what truly moves.**

---

## 2. 审计后必须冻结的 native OccFM 合同

第一版 P0-F9 在 CPU CI 通过后又做了一轮与官方 OccFM 代码逐行对齐审计。审计发现两个会真实影响 pretrained WM 能力的隐藏错配，因此 v1 cache / checkpoint 不再使用。

### 2.1 VAE latent 必须是 posterior sample，不是 mean

官方 OccFM 的 VAE cache 保存：

```text
sampled_features = mu + sigma * eps
```

OccFM-Fut 训练读取的就是这些 posterior samples。

因此 P0-F9 v2 重新编码：

```text
history occupancy      -> posterior sample latent
Strong-W2Det occupancy -> posterior sample latent
GT future occupancy    -> posterior sample latent
```

为了可复现，sample 不是每次随机变化，而是按：

```text
SHA256(base_seed, stream, sample_id)
```

固定 per-sample seed。`history / physics / future` 使用不同 stream。

### 2.2 inherited OccFM backbone 必须保持 HIST_LAST=4

官方 `occfm_fut.yaml`：

```text
SEQUENCE_LENGTH = 6
HIST_LAST = 4
```

官方 dataset 在进入 WM 前会把六个历史槽中的前两个置零，并把 trajectory 的前两行同步置零。

P0-F9 v2 因此采用：

```text
loaded native OccFM path:
    [0, 0, h2, h3, h4, h5]

new 40x40 full-history context path:
    [h0, h1, h2, h3, h4, h5]
```

这样既保持 pretrained backbone 的原始输入分布，又保留我们新增的六帧完整历史上下文。

---

## 3. Stage-1 结构

```text
6-frame occupancy history
          |
          v
 frozen OccFM VAE posterior sampling
          |
   Z_history (6 frames)
          |
          +------------------------------+
          |                              |
          |                         frozen MSP
          |                         Top-2 20x20
          v                              |
 native OccFM CFM                        |
 Gaussian noise -> absolute GT future    |
          ^                              |
          | physics condition            |
 Strong W2Det/KTA posterior-sampled latent
          |
          + local native path: HIST_LAST=4
          + 40x40 full-history context: all 6 frames
          + ego trajectory
          + zero-gated physics cross-attention
          v
 absolute future latent windows
          |
          v
 overlap-average scatter into full latent
          |
          v
 frozen VAE decoder
          |
          v
 decoded future occupancy proposal
          |
 outside MSP support: exact Strong W2Det
 inside MSP support : only WM dynamic semantics
```

关键点：

- `anchor_future_latent` 只是 physics condition / latent scatter fallback；
- FM source 永远是 Gaussian noise；
- FM target 是 absolute `gt_future_latent`；
- P0-F8 的 KEEP/CLEAR/WRITE action 不再是模型输出；
- P0-F8 sidecar 仅复用其 sparse GT result semantic population。

---

## 4. Native Flow Matching

```text
z0 ~ N(0, I)
z1 = absolute GT future latent
zt = (1-t) z0 + t z1
t  = sigmoid(N(0,1))
v* = z1 - z0
```

```text
L_FM = MSE(v_theta(zt | history, physics, trajectory), z1-z0)
```

Strong W2Det 不进入：

```text
z0
z1-z0
residual target
```

Stage-1 默认：

```text
uncond_prob = 0
guidance_scale = 1
```

代码中的 `--uncond-prob` 已经过审计：非零时会真实生效；默认 0.0 仍为全条件训练。

---

## 5. Physics condition

两条路径：

1. zero-init token-wise physics projection；
2. bottleneck gated cross-attention。

```text
Query = WM bottleneck
Key/Value = Strong-W2Det future latent
x' = x + tanh(alpha) * CrossAttn(x, physics)
alpha_init = 0
```

审计后所有“无 physics”的历史槽都严格保持 no-op：

- token-wise prior projection 无 bias；
- cross-attention 对原始全零 prior frame 显式 mask；
- 即使 attention bias 后续学成非零，历史 zero-prior frame 也不会变成伪 physics condition。

---

## 6. Stage-1 Loss

```text
L = L_semantic + 0.1 * L_FM
```

```text
L_semantic = weighted CE + Lovasz
```

18-way frozen VAE logits严格折叠为：

```text
background/non-dynamic + 8 dynamic classes
```

background 用所有 non-dynamic logits 的 `logsumexp`，因此 9-way softmax 等价于原 18-way probability marginalization。

### Semantic population

复用 P0-F8 compact sidecar 中：

```text
all EDIT result voxels
+ all correct dynamic KEEP voxels
+ bounded background KEEP pool
```

只使用最终 GT result semantics，不使用 action label。

### Class balancing

```text
w_c ~ 1 / sqrt(freq_c)
normalize present classes
clip [0.5, 2.0]
```

### Horizon weighting

审计后的语义 loss 对六个未来 horizon **显式等权**：

```text
L_sem = mean_h L_sem(h), h=0..5
```

不会再因为某个 horizon 的 sparse target voxel 更多而获得更大梯度权重。

---

## 7. 训练基础设施

Stage-1：

```text
Frozen VAE
Scene-balanced sampling
AdamW
weight_decay = 1e-2
grad_clip = 5
warmup = first 5%
cosine -> min_lr_ratio 0.2
pretrained OccFM LR = 2e-5
new physics/context LR = 1e-4
FP32 EMA decay target = 0.999
```

EMA 使用 ramp：

```text
current_decay = 0.999 * (1-exp(-updates/2000))
```

注意：1200 step 内它是 fast-following EMA，而不是一开始就接近 0.999。这是当前明确保留的设计选择，不是隐藏 bug。

Stage-2 E2E VAE adaptation 暂不启动。只有 Stage-1 出现真实 deployment 正增益后再做。

---

## 8. Cache v2：重新编码 native latent，但不重跑 MSP

旧 P0-F7/F8 v3 cache 继续作为 route provenance：

```text
sample_id / scene
Top-2 origins
window_valid
MSP write support
trajectory
exact eval payload (val)
```

P0-F9 v2 会重新得到 raw occupancy，并重新编码：

```text
full_history_latent
anchor_future_latent
gt_future_latent
```

原因不是重新做科学 routing，而是必须从 mean latent 切回官方 WM 使用的 posterior-sampled latent distribution。

Train 中 Strong-W2Det occupancy 会从 exact raw history 按冻结配置重新计算；不会重新训练/运行 MSP。Val 直接使用已缓存的 exact Strong-W2Det occupancy，并与 raw GT 做一致性检查。

**如果已经生成过 P0-F9 v1 cache，不要 `--resume`。v2 必须使用新的输出目录。**

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
  --output "$ROOT/data/p0_f9_v2_wm_train_top2_4096" \
  --latent-seed 20260904 \
  --vae-batch-size 16 \
  --prepare-workers 16 \
  --prefetch-windows 64 \
  --shard-size 32
```

同一个 v2 output 中断后可以加：

```text
--resume
```

### Val cache

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" "$ROOT/tools/real_motion/build_p0_f9_cache_fast.py" \
  --source-cache "$ROOT/data/p0_f5_wm_val_top2" \
  --msp-cache "$ROOT/data/msp_probe_val_128.pt" \
  --dataroot "$OCCFM/data/nuscenes" \
  --info-pkl "$OCCFM/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --vae-ckpt "$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt" \
  --output "$ROOT/data/p0_f9_v2_wm_val_top2_128" \
  --latent-seed 20260904 \
  --vae-batch-size 16 \
  --prepare-workers 8 \
  --prefetch-windows 32 \
  --shard-size 32
```

Val 会保留：

```text
eval_future_gt_occ
eval_strong_anchor_occ
eval_repair_target_occ
eval_gt_moving_support
```

---

## 9. 真实 GPU smoke（正式训练前必须做一次）

CPU CI 只能验证 Python/import/unit-test，不覆盖 real OccFM CUDA/BF16/custom FlashAttention backward。

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main
OUT="$ROOT/outputs/p0_f9_v2_stage1_smoke"

rm -rf "$OUT"

"$PY" "$ROOT/tools/real_motion/train_p0_f9_native_sparse_forecast.py" \
  --train-cache "$ROOT/data/p0_f9_v2_wm_train_top2_4096" \
  --val-cache "$ROOT/data/p0_f9_v2_wm_val_top2_128" \
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

Smoke 必须至少走通：

```text
posterior-sampled cache load
official checkpoint reuse gate
BF16 forward
FP32 custom FlashAttention backward
frozen VAE sparse semantic gradient
optimizer + EMA
validation
checkpoint save
```

---

## 10. Stage-1 正式训练

```bash
OUT="$ROOT/outputs/p0_f9_v2_native_sparse_stage1_4096"

"$PY" "$ROOT/tools/real_motion/train_p0_f9_native_sparse_forecast.py" \
  --train-cache "$ROOT/data/p0_f9_v2_wm_train_top2_4096" \
  --val-cache "$ROOT/data/p0_f9_v2_wm_val_top2_128" \
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

Resume 是 fail-closed：cache、sidecar、VAE、upstream checkpoint、训练步数、batch size、LR/loss/schedule/seed 等关键合同必须与 checkpoint 一致，否则直接停止。

---

## 11. Deployment evaluation

```bash
OUT="$ROOT/outputs/p0_f9_v2_native_sparse_stage1_4096"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
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

Evaluator 会强制检查：

```text
checkpoint <-> exact val cache SHA
checkpoint <-> VAE SHA
native HIST_LAST=4
posterior-sampled latent contract
same-support GT oracle bit-exact == cached repair target
```

最终必须汇总：

```text
Step | Overall | DeltaOverall | Moving | DeltaMoving
     | O@1s/O@2s/O@3s | M@1s/M@2s/M@3s
```

成功条件仍是：

```text
DeltaOverall >= 0
DeltaMoving  > 0
```

Frozen Strong W2Det reference：

```text
Overall = 39.7457
Moving  = 21.3872
```

---

## 12. Official native OccFM baseline

在相同 v2 val cache 上：

```bash
"$PY" "$ROOT/tools/real_motion/eval_p0_f9_native_occfm_baseline.py" \
  --cache "$ROOT/data/p0_f9_v2_wm_val_top2_128" \
  --occfm-ckpt "$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt" \
  --output "$ROOT/outputs/p0_f9_v2_official_occfm_native_128.json" \
  --seed 20260904
```

该 baseline 使用：

```text
posterior-sampled history latent
official HIST_LAST=4
released OccFM sampler
与 P0-F9 相同的 per-sample forecast seed contract
```

最终论文对比：

```text
Strong W2Det
Official native OccFM
P0-F9 physics-conditioned sparse WM
Same-support GT oracle
```

---

## 13. 这轮审计修掉的隐藏问题

- VAE mean 与官方 OccFM-Fut posterior-sample latent 分布不一致；
- inherited backbone 未保持官方 HIST_LAST=4；
- `--uncond-prob` 被 `force_conditioned=True` 静默绕过；
- semantic loss 实际按 voxel population 加权，而不是六 horizon 等权；
- dynamic/background validation rate 使用了错误的聚合权重；
- zero-aligned physics history frame 可能通过 learned bias 变成伪 condition；
- cache builder 没有严格验证 supplied MSP cache hash；
- resume 对 cache/sidecar/VAE/loss/LR/schedule 等 provenance 校验不足；
- deployment evaluator 没有严格绑定 checkpoint 与 exact val cache；
- same-support oracle bit-exact consistency check 在第一版 evaluator 中丢失。

这些修正属于实现合同修复，不改变 P0-F9 的研究主张。
