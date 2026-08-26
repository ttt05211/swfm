# SWFM 最终实现审查记录

## 当前结论

SWFM 主协议已经从官方 OccFM `hist_traj / epoch=000199.ckpt` 完整迁移到 **OccFM-Fut / epoch=000196.ckpt**。迁移不是只换 checkpoint 路径，而是同步修改了 data → prepared → latent cache → transition architecture → tiny/full training → inference → P0-A → equivalence → L40S tuner → config/documentation 的 trajectory contract。

正式服务器实验仍必须通过：

```text
OccFM-Fut 196 transition equivalence
→ P0-A/B/C/D/E
→ tiny overfit
→ small held-out
→ full training
```

源码审查不能替代 checkpoint + CUDA 的真实 equivalence。

## 1. Frozen Ego / Trajectory Contract

主协议明确允许 **GT future ego**。

### 1.1 Future ego pose

用于：

- confident-static 的 future ego SE(3) transport；
- KTA t0-ego prediction 到各 target future-ego grid 的坐标变换。

### 1.2 OccFM-Fut 12-step trajectory conditioning

严格复现官方 OccFM dataset：

```text
6 history frames + 6 future frames
→ 每个 frame 读取 temporal info: gt_ego_fut_trajs
→ 取该 frame 的第一个 XY step
→ concatenate 为 [12,2]
→ HIST_LAST=4
→ 前 6-4=2 rows 置零
```

冻结值：

```yaml
UPSTREAM.WM_VARIANT: occfm_fut
UPSTREAM.WM_CONFIG: tools/cfgs/occfm_fut.yaml
UPSTREAM.WM_CHECKPOINT_REL: logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt
MODEL.TRAJECTORY_LENGTH: 12
EGO_PROTOCOL.NAME: occfm_fut_12step_v1
EGO_PROTOCOL.HIST_LAST: 4
EGO_PROTOCOL.ZERO_PREFIX_STEPS: 2
EGO_PROTOCOL.UPSTREAM_INIT_VARIANT: fut_traj_196
```

Formal prepare/inference 要求 temporal info 中 12 个 frame 都存在 `gt_ego_fut_trajs`。不允许静默退化回 6-step hist contract。

## 2. 为什么不再用 199 初始化

旧方案用 199 的原因只是 `trajectory_length=6` shape-compatible。现在既然主协议明确使用 GT future ego，196 的 12-step `traj_encoder` 与最终信息协议更匹配，而且论文主 baseline 也应该是 Official OccFM-Fut 196。

正式 SWFM transition 现在 `trajectory_length=12`。加载 checkpoint 时要求 `traj_encoder.0.weight` 真正成功复用；误传 199 会因为 shape mismatch / gate 直接停止，而不是随机初始化 trajectory encoder 后继续训练。

## 3. Fair Baseline Contract

主表必须比较同等信息量：

```text
SWFM + GT future ego + init OccFM-Fut 196
vs
Official dense OccFM-Fut 196
vs
其他获得同等 future ego information 的 baseline
```

199 hist-only 可以做额外 information ablation，但不能作为“同信息量主 baseline”。

## 4. 已完成的 correctness closure

1. `FREE` 与 `confident_static` 独立。
2. `M_gen` 外 latent 每个 ODE step clamp 到 exact `E(empty)`。
3. overlap windows 共享统一 global noise canvas。
4. final composition 仅允许 `dynamic_prediction & write_support & ~confident_static`。
5. raw causal path：ego compensation → real-motion → SE3 → causal KTA → tube。
6. training target = `future_dynamic_target_occ`，不等于 Moving-mIoU 的 `future_moving_occ`。
7. Moving-mIoU v2 冻结为 world-motion interval + future-ego dual-box support + horizon-first aggregation。
8. P0-C 直接检查 actual hard-static mask，margin=0。
9. P0-D 使用 Decomposition Oracle / Causal-Support Oracle。
10. P0-E 同时检查 true-moving、actual WM target、causal sparse canvas。
11. maneuver/KTA-hard subset 只改变 GT instance support，仍调用原 MovingMIoUV2MultiHorizon。
12. evaluator 默认缺 prediction 直接失败。
13. DDP validation no-padding，best.pt 不受 duplicate sample bias。
14. `tools` package shadow 问题已修。

## 5. 196 migration 的资产版本门禁

Prepared version 已升级：

```text
real_motion_prepared_v3_occfm_fut196
```

旧 prepared 必须重建。

Latent cache 明确记录：

```text
trajectory_protocol = occfm_fut_12step_v1
trajectory_length = 12
upstream_wm_variant = occfm_fut
upstream_init_variant = fut_traj_196
```

Formal `train_full.py` fail-closed 检查：

- train/val cache contract fingerprint；
- VAE SHA256 / latent mode / VAE AMP / latent support radius；
- 12-step trajectory protocol；
- `empty_latent.pt` metadata；
- upstream checkpoint reuse fraction；
- exact `traj_encoder.0.weight` reuse；
- resume config/cache/upstream/empty-latent contract。

因此旧 199 cache 或旧 resume 不能静默混入 196 实验。

## 6. P0-A / Transition Equivalence

P0-A frozen dense WM 已改为加载：

```text
tools/cfgs/occfm_fut.yaml
epoch=000196.ckpt
```

并要求 `[12,2]` trajectory。

Transition equivalence 现在比较：

```text
official OccFM-Fut transition
trajectory_length=12
50×50
same 196 checkpoint
same 12-step trajectory

vs

modified SWFM transition
50×50
zero prior
origin=(0,0)
```

`traj_encoder.0.weight` 未成功加载时 equivalence 直接失败。

## 7. L40S 48GB

运行时优化保持不改变科学协议：

- Sparse CFM BF16；
- Frozen VAE 默认 FP32；
- TF32 / cuDNN benchmark / efficient SDPA；
- fused AdamW；
- pinned/persistent DataLoader；
- sharded cache + shard-aware sampler；
- precomputed sparse window plan；
- DDP gradient bucket view / static graph；
- `torch.compile` 默认关闭，实测有效才开；
- tuner 扫 worker + batch size，并且在开始 benchmark 前验证 196 的 12-step trajectory encoder 已加载。

FP8 不进入主论文协议。

## 8. Serialized Asset Trust Boundary

prepared/cache/prediction `.pt` 和 temporal info `.pkl` 只能来自自己生成或可信官方来源。不能直接执行来源不明的 pickle / torch serialized assets。

## 9. 服务器最终验收顺序

```text
0. pytest
1. OccFM-Fut 196 50×50 transition equivalence
2. 16-window prepare smoke；确认每个 trajectory == [12,2] 且前2行0
3. P0-A
4. P0-B
5. P0-C
6. P0-D
7. P0-E
8. tiny latent cache
9. 64/128 overfit
10. small held-out
11. full latent cache
12. L40S worker/batch tuning
13. full training
14. complete inference/evaluation
15. subset / Harm-Repair
16. latency / GFLOPs
```

任何 gate 失败，优先定位 `ego/trajectory protocol → motion/support → geometry/KTA → VAE → WM optimization → composition/evaluation`，不要用 Router / ABE / confidence loss 掩盖基础 contract 问题。
