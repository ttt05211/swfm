# P0-F9 v8 Ordered Context 小预算实验

## 目的

当前 M-400 已证明 true-motion weighted native FM 能稳定提高 Moving，但伴随 WRITE precision 下降、stale 上升。这个实验只回答一个新问题：

> 在保持 M-400 的 motion-weighted FM、physics、mean context、数据和训练预算完全不变时，额外保留 6 帧历史顺序，是否能提高运动位置预测的准确性？

当前 mean-context 分支是：

```text
H0..H5 -> temporal mean -> Conv(16 -> 128, 3x3, stride2)
```

候选 MT 增加：

```text
C = C_mean + DeltaC_ordered
DeltaC_ordered = Conv(concat(H0,H1,H2,H3,H4,H5))
                 96 -> 128, 3x3, stride2
```

ordered Conv 权重和 bias 全零初始化，因此 MT phase-step0 与 MCTRL phase-step0 数值功能一致。

## 对照合同

| 项目 | MCTRL | MT |
|---|---|---|
| 起点 | 同一 M-400 EMA | 同左 |
| loss | motion-weighted native FM, lambda=2 | 同左 |
| physics | 保留 | 同左 |
| 原 mean context | 保留 | 同左 |
| ordered residual | 存在但关闭、冻结 | 开启、训练 |
| 数据 / batch / noise / sampled t | 相同 | 相同 |
| LR | 4e-6 / 2e-5 固定 | 同左 |
| EMA | 从同一 parent 权重新建 phase EMA | 同左 |
| 第一轮预算 | 200 phase steps | 200 phase steps |
| 保存 | step_0100.pt / step_0200.pt | 同左 |

实验阶段不保存 `step_0000.pt`、`best.pt`、`latest.pt`、`last.pt`。

---

## 1. 拉取最新代码

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

cd "$ROOT"
git pull https://gh-proxy.com/https://github.com/ttt05211/swfm.git main
git log -1 --oneline
```

---

## 2. MCTRL：从 M-400 EMA 再训练 200 steps

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
MASK="$ROOT/data/p0_f9_v8_motion_mask_train_4096.pt"
PARENT="$ROOT/outputs/p0_f9_v8_M_phase400/step_0400.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v8_MCTRL_ordered_phase200"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_v8_ordered_context_phase.py" \
  --variant MCTRL \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --motion-mask-sidecar "$MASK" \
  --parent-checkpoint "$PARENT" \
  --parent-step 400 \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output-dir "$OUT" \
  --phase-end-step 200 \
  --phase-limit 400 \
  --milestones 100 200 \
  --batch-size 8 \
  --num-workers 4 \
  --wm-lr 4e-6 \
  --new-lr 2e-5 \
  --motion-weight-lambda 2 \
  --seed 20260904 \
  --amp 2>&1 | tee "$ROOT/outputs/p0_f9_v8_MCTRL_ordered_phase200.log"
```

---

## 3. MT：同一个 M-400 EMA 起点，开启 ordered residual

可与 MCTRL 在另一张 GPU 同时跑。

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
MASK="$ROOT/data/p0_f9_v8_motion_mask_train_4096.pt"
PARENT="$ROOT/outputs/p0_f9_v8_M_phase400/step_0400.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v8_MT_ordered_phase200"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=1 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_v8_ordered_context_phase.py" \
  --variant MT \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --motion-mask-sidecar "$MASK" \
  --parent-checkpoint "$PARENT" \
  --parent-step 400 \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output-dir "$OUT" \
  --phase-end-step 200 \
  --phase-limit 400 \
  --milestones 100 200 \
  --batch-size 8 \
  --num-workers 4 \
  --wm-lr 4e-6 \
  --new-lr 2e-5 \
  --motion-weight-lambda 2 \
  --seed 20260904 \
  --amp 2>&1 | tee "$ROOT/outputs/p0_f9_v8_MT_ordered_phase200.log"
```

正常情况下每个目录只应有：

```text
step_0100.pt
step_0200.pt
training_report.json
```

MT 日志里的 `ordered_w_rms` 应从 0 开始逐步变成非零；MCTRL 始终为 0。

---

## 4. 同预算部署评估

### MCTRL-100 / MCTRL-200

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
DIR="$ROOT/outputs/p0_f9_v8_MCTRL_ordered_phase200"

for STEP in 0100 0200; do
  CUDA_VISIBLE_DEVICES=0 \
  "$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_ordered_context.py" \
    --cache "$VAL" \
    --occfm-ckpt "$WM" \
    --vae-ckpt "$VAE" \
    --trained-sparse-ckpt "$DIR/step_${STEP}.pt" \
    --use-ema \
    --output "$ROOT/outputs/p0_f9_v8_MCTRL_ordered_phase${STEP}_diagnostic_128.json" \
    --seed 20260904 \
    --amp
done
```

### MT-100 / MT-200

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
DIR="$ROOT/outputs/p0_f9_v8_MT_ordered_phase200"

for STEP in 0100 0200; do
  CUDA_VISIBLE_DEVICES=1 \
  "$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_ordered_context.py" \
    --cache "$VAL" \
    --occfm-ckpt "$WM" \
    --vae-ckpt "$VAE" \
    --trained-sparse-ckpt "$DIR/step_${STEP}.pt" \
    --use-ema \
    --output "$ROOT/outputs/p0_f9_v8_MT_ordered_phase${STEP}_diagnostic_128.json" \
    --seed 20260904 \
    --amp
done
```

核心只比较同预算：`MT100 - MCTRL100`、`MT200 - MCTRL200`。

一个真正有价值的 ordered context 应该至少满足下列之一，同时 Overall 不明显退化：

- Moving 提升；
- WRITE precision 提升；
- stale 降低 / CLEAR recall 改善；
- dynamic precision 提升而不是只靠 volume 增大换 recall。

如果 MT 只是增加 dynamic volume、WRITE recall，而 precision/stale 继续恶化，就停止该候选。

---

## 5. 若 200 steps 明确有效，再一起续到 400

以 MT 为例：

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
MASK="$ROOT/data/p0_f9_v8_motion_mask_train_4096.pt"
PARENT="$ROOT/outputs/p0_f9_v8_M_phase400/step_0400.pt"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v8_MT_ordered_phase200"
RESUME="$OUT/step_0200.pt"

CUDA_VISIBLE_DEVICES=1 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_v8_ordered_context_phase.py" \
  --variant MT \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --motion-mask-sidecar "$MASK" \
  --parent-checkpoint "$PARENT" \
  --parent-step 400 \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output-dir "$OUT" \
  --phase-end-step 400 \
  --phase-limit 400 \
  --milestones 400 \
  --batch-size 8 \
  --num-workers 4 \
  --wm-lr 4e-6 \
  --new-lr 2e-5 \
  --motion-weight-lambda 2 \
  --seed 20260904 \
  --resume-from "$RESUME" \
  --amp
```

MCTRL 若续训必须同步续到 400，保证同预算对照。
