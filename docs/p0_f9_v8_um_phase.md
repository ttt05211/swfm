# P0-F9 v8 U/M 实验合同

## 目的

v8 的两项诊断已经给出明确方向：

- motion-associated latent 只占约 6.12%，但承担约 11.32% 的 FM squared error；v7 step400 下 moving NMSE 仍明显高于 non-moving NMSE。因此下一轮只验证 **motion-weighted native FM**。
- 去掉当前 40x40 context 后 Overall/Moving 没有下降，反而轻微上升，因此本轮 **不改 context 结构、不加 ordered context**。
- physics condition 被移除后 Overall/Moving 都下降，因此 physics 保留。

本轮只允许一个训练变量：**FM loss 的空间权重**。

| 项目 | U | M |
|---|---|---|
| 初始化 | 同一 v7 step400 EMA | 同左 |
| architecture / physics / context | 原样保留 | 同左 |
| loss | Uniform native FM | Normalized motion-weighted native FM, lambda=2 |
| 数据 / scene sampling / noise / sampled t | 相同 | 相同 |
| optimizer groups / LR / EMA | 相同 | 相同 |
| 第一阶段预算 | 新增 400 phase steps | 新增 400 phase steps |
| milestone | phase 200 / 400 | phase 200 / 400 |

M 使用

```text
w_i = 1 + lambda * m_i
L_M = sum_i w_i * e_i^2 / sum_i w_i
lambda = 2
```

其中 `m_i` 是与 v8 diagnostic 完全一致的 Moving-v2 dual-box motion-associated latent support。它仅作为训练标签，不进入模型输入，也不参与推理。

## 关键实现合同

1. U/M 都从 **v7 step400 EMA weights** 开始，而不是 raw weights。
2. phase step0 新建 optimizer；phase EMA 也从同一 parent EMA 权重开始。
3. pretrained/new 两组 LR 固定为 v7 step400 末端值：`4e-6 / 2e-5`。phase 内不重新启动 cosine schedule，因此以后从 400 续到 800 不会再次抬高 LR。
4. scene-balanced sampling 改为 **phase-step-indexed deterministic multinomial**；每个 phase step 的 batch、Gaussian source noise、FM sampled t 都由独立固定 seed 决定。U/M 因而逐 step 使用完全相同的训练样本、noise 和 t。
5. `step_0200.pt` / `step_0400.pt` 是完整 resume checkpoint，包含 raw model、EMA、optimizer、RNG state、sampling progress、parent/cache/mask provenance。
6. 实验阶段不生成 `step_0000.pt`、`best.pt`、`latest.pt`、`last.pt`。
7. 以后若 U/M 都需要续到 800，必须从各自 `step_0400.pt` 恢复完整训练状态，不允许重新从 EMA 权重建立 optimizer。

---

## 1. 先生成训练 true-motion mask sidecar

训练 P0-F9 cache 没有保存 evaluation-only moving support，因此先从 frozen MSP exact-window provenance 恢复 nuScenes token，并用与 Moving-mIoU-v2 / v8 diagnostic 相同的定义生成 50x50 latent bool mask。

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
MSP="$ROOT/data/msp_probe_train_4096.pt"
DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl"
MASK="$ROOT/data/p0_f9_v8_motion_mask_train_4096.pt"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/build_p0_f9_v8_motion_mask_sidecar.py" \
  --train-cache "$TRAIN" \
  --msp-cache "$MSP" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --output "$MASK" \
  --workers 8 \
  --prefetch-samples 32 \
  --motion-weight-lambda 2
```

这里会直接打印 **训练集自身** 的 `routed_top2_motion_fraction` 和 lambda=2 后的 effective motion weight mass。不要把验证集的 6.12% 人工写进训练。

---

## 2. U：Uniform FM control，新增 400 phase steps

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
MASK="$ROOT/data/p0_f9_v8_motion_mask_train_4096.pt"
PARENT="$ROOT/outputs/p0_f9_v7_native_fm_only_400/step_0400.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v8_U_phase400"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_v8_um_phase.py" \
  --variant U \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --motion-mask-sidecar "$MASK" \
  --parent-checkpoint "$PARENT" \
  --parent-step 400 \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output-dir "$OUT" \
  --phase-end-step 400 \
  --phase-limit 800 \
  --milestones 200 400 \
  --batch-size 8 \
  --num-workers 4 \
  --wm-lr 4e-6 \
  --new-lr 2e-5 \
  --motion-weight-lambda 2 \
  --seed 20260904 \
  --amp
```

---

## 3. M：Normalized motion-weighted FM，新增 400 phase steps

可与 U 在另一张 GPU 并行跑。

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
MASK="$ROOT/data/p0_f9_v8_motion_mask_train_4096.pt"
PARENT="$ROOT/outputs/p0_f9_v7_native_fm_only_400/step_0400.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v8_M_phase400"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=1 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_v8_um_phase.py" \
  --variant M \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --motion-mask-sidecar "$MASK" \
  --parent-checkpoint "$PARENT" \
  --parent-step 400 \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output-dir "$OUT" \
  --phase-end-step 400 \
  --phase-limit 800 \
  --milestones 200 400 \
  --batch-size 8 \
  --num-workers 4 \
  --wm-lr 4e-6 \
  --new-lr 2e-5 \
  --motion-weight-lambda 2 \
  --seed 20260904 \
  --amp
```

U/M 应只生成：

```text
step_0200.pt
step_0400.pt
training_report.json
```

---

## 4. phase200 / phase400 部署评估

优先使用现有 full training-failure diagnostic，因为它同时给出 NFE=10 rollout 的 Overall/Moving 与 CLEAR/WRITE/stale/dynamic-volume 指标。

U phase400 示例：

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
TRAINED="$ROOT/outputs/p0_f9_v8_U_phase400/step_0400.pt"
OUT="$ROOT/outputs/p0_f9_v8_U_phase400_diagnostic_128.json"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_training_failure.py" \
  --cache "$VAL" \
  --occfm-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --trained-sparse-ckpt "$TRAINED" \
  --use-ema \
  --output "$OUT" \
  --seed 20260904 \
  --amp
```

M 只需把 `TRAINED` / `OUT` 换成 M 的 milestone。

主要比较：

- `M - U` 的 Overall / Moving；
- FM MSE / cosine；
- 1s/2s/3s write recall / precision；
- dynamic recall / precision / volume；
- clear recall / stale / wrong-clear。

若 M 在同预算下有净部署收益，并且没有靠 dynamic flooding 换 recall，则 U/M 一起续到 800。若只改善 moving-FM、没有改善真实 rollout，则保留 U，不立刻放大 lambda。

---

## 5. 400 -> 800 的正确续训方式

下面以 M 为例。必须恢复 `step_0400.pt` 的 raw model + optimizer + EMA + RNG + sampling progress。

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
MASK="$ROOT/data/p0_f9_v8_motion_mask_train_4096.pt"
PARENT="$ROOT/outputs/p0_f9_v7_native_fm_only_400/step_0400.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v8_M_phase400"
RESUME="$OUT/step_0400.pt"

CUDA_VISIBLE_DEVICES=1 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_v8_um_phase.py" \
  --variant M \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --motion-mask-sidecar "$MASK" \
  --parent-checkpoint "$PARENT" \
  --parent-step 400 \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output-dir "$OUT" \
  --phase-end-step 800 \
  --phase-limit 800 \
  --milestones 800 \
  --batch-size 8 \
  --num-workers 4 \
  --wm-lr 4e-6 \
  --new-lr 2e-5 \
  --motion-weight-lambda 2 \
  --seed 20260904 \
  --resume-from "$RESUME" \
  --amp
```

续到 800 后只额外生成 `step_0800.pt`，不会创建 `best/latest/last`。
