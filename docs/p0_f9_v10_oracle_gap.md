# P0-F9 v10：Oracle Gap Attribution

## 目的

Full-data M 已把 E5/E10 推到约 `38 Overall / 14.7–14.8 Moving`，但 Strong-W2Det 仍约为 `39.75 / 21.39`。这一阶段不再继续堆 epoch、lambda 或 context，而是固定 E5/E10、128-val、NFE=10、Moving-v2 与同一 MSP support，直接测剩余差距究竟来自 CLEAR/KEEP、WRITE、动态语义还是 support/routing。

诊断不训练任何参数。每个 checkpoint 只做一次真实 sparse rollout，后续 oracle 都在 occupancy proposal 上做受控 GT intervention。

## Oracle 定义

- `current`：当前 decoded proposal + audited takeover fusion。
- `oracle_clear_keep`：只在 anchor dynamic voxels 上把“离开/保留”的动态 presence 决策改正确。KEEP 缺失时恢复 anchor class，不直接授予 GT class。
- `oracle_write`：只在 anchor non-dynamic voxels 上把 WRITE/no-WRITE 改正确；required WRITE 使用 GT dynamic class，同时删除 stable non-dynamic 上的 false write。
- `oracle_event_presence`：联合上述 CLEAR/KEEP + WRITE presence oracle。
- `oracle_semantic`：只有 proposal 与 GT 都已经是 dynamic 的位置才改成 GT dynamic class，不改变动态几何/presence。
- `same_support_gt_event`：GT proposal，但仍使用当前 frozen MSP write support。应接近既有 same-support GT takeover ceiling。
- `oracle_support_gt_event`：仍是同一个 GT proposal，只把 support 扩到 Strong anchor 与 GT 确实需要动态出现/消失/重标的 BEV cell。它与 `same_support_gt_event` 的差值专门表示 support/routing headroom。

额外报告 current support 对全局 GT-required CLEAR/WRITE/relabel/event voxels 的覆盖率，以及每个 Moving class 在 1s/2s/3s 的 IoU。

## 正式运行 E5 + E10

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
DIR="$ROOT/outputs/p0_f9_v9_full_m_official_init_ddp2"
E5="$DIR/epoch_0005.pt"
E10="$DIR/epoch_0010.pt"
OUT="$ROOT/outputs/p0_f9_v10_oracle_gap_e5_e10_128.json"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_oracle_gap.py" \
  --cache "$VAL" \
  --occfm-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --checkpoint "E5=$E5" \
  --checkpoint "E10=$E10" \
  --use-ema \
  --seed 20260904 \
  --output "$OUT" \
  --amp
```

Smoke 可加 `--max-windows 2`，但正式结论必须用完整 128 windows。

## 输出重点

终端会打印：

```text
=== ORACLE GAP DECOMPOSITION: E5/E10 ===
=== MOVING BY HORIZON ===
=== CURRENT SUPPORT COVERAGE OF GT-REQUIRED DYNAMIC EDITS ===
=== CURRENT MOVING PER-CLASS DELTA: E10 - E5 ===
```

JSON 还包含每个 oracle 的完整 Overall/Moving per-horizon/per-class、CLEAR/KEEP/WRITE physical metrics，以及：

- `gain_vs_current`
- `strong_gap_recovery`
- `clear_write_interaction_moving`
- `support_increment_over_same_support_gt`
- `residual_between_event_presence_and_same_support_gt`

## 决策规则

这一步只定位，不先承诺新模块：

- `oracle_write` 最大：优先显式 future displacement / shape transport；
- `oracle_clear_keep` 最大：优先 source survival/departure；
- `oracle_event_presence` 比两者单独明显更强：优先联合 source→future correspondence，让 CLEAR(old)+WRITE(new) 成为同一个 motion event；
- `oracle_semantic` 最大：再考虑动态 semantic correction；
- `oracle_support_gt_event - same_support_gt_event` 最大：先修 MSP/support/routing；
- 若 same-support GT 很高但所有单独 oracle 都有限，说明误差高度耦合，应避免再加独立 voxel head。
