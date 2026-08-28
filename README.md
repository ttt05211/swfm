# SWFM：Real-Motion Sparse Occupancy World Model

> 基于官方 **OccFM (NeurIPS 2025)**。核心原则：**Transport what is physically deterministic; generate only what truly moves.**

完整方法协议见根目录 `real_motion_sparse_occ_world_model_final_plan.html`。

## 1. 当前主协议：OccFM-Fut 196

SWFM 主线已经从 `hist_traj / epoch=000199.ckpt` 完整迁移到官方 **future-trajectory** 版本：

```text
logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt
```

VAE 仍使用：

```text
logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt
```

主实验信息协议明确允许 **GT future ego**。它用于两处：

1. future ego pose：静态 occupancy 的 deterministic SE(3) transport，以及 KTA 从 t0 ego 到各 future ego grid 的坐标变换；
2. trajectory conditioning：严格复现官方 `occfm_fut.yaml` 的 **12-step GT ego trajectory**。

这不是 future semantic / instance GT 泄漏；但论文主表中的 baseline 必须获得同等 future-ego information。Dense OccFM 主 baseline 应使用官方 `occfm_fut` 196，而不是 hist-only 199。

## 2. 官方 12-step trajectory contract

官方 OccFM-Fut 的 trajectory 不是“t0 后 6 个 future 点”。SWFM 现在严格复现官方 dataset 行为：

```text
6 history frames + 6 future frames = 12 occupancy frames
                ↓
对每一个 frame
读取 temporal info 中 gt_ego_fut_trajs 的第一个 XY step
                ↓
拼成 [12, 2]
                ↓
HIST_LAST = 4
所以最前 6 - 4 = 2 个 trajectory rows 置零
```

冻结配置：

```yaml
UPSTREAM:
  WM_VARIANT: occfm_fut
  WM_CONFIG: tools/cfgs/occfm_fut.yaml
  WM_CHECKPOINT_REL: logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt

MODEL:
  TRAJECTORY_LENGTH: 12

EGO_PROTOCOL:
  NAME: occfm_fut_12step_v1
  FUTURE_POSE_SOURCE: gt_future_ego
  TRAJECTORY_CONDITION_SOURCE: official_temporal_info_first_step_per_frame
  TRAJECTORY_LENGTH: 12
  HIST_LAST: 4
  ZERO_PREFIX_STEPS: 2
  REQUIRE_TEMPORAL_INFO: true
  BASELINE_INFORMATION_MATCH: required
  UPSTREAM_INIT_VARIANT: fut_traj_196
```

Formal prepare/inference 不允许缺失 temporal-info trajectory 后静默退化到别的协议。

## 3. 冻结的 Hybrid-Balanced-R1 support

P0-D2 在 128 个不同 validation scenes 上完成 scene-disjoint 验证后，正式冻结：

```yaml
MOTION:
  SUPPORT_GEOMETRY: hybrid_endpoint_swept_v1
  KTA_ENDPOINT_TUBE_RADII: [1, 2, 3, 4, 4, 5]
  KTA_SWEPT_TUBE_RADII: [1, 1, 1, 1, 1, 1]
  UNCERTAIN_TUBE_RADII: [0, 0, 0, 1, 2, 3]
  LATENT_EXTRA_RADIUS: 0
```

正式 generation support：

```text
MOVING endpoint tube
        ∪
MOVING current→KTA-endpoint thin swept corridor
        ∪
UNCERTAIN zero-motion reachable tube
```

其中 **swept corridor 只定义 World Model 可以写入的位置，不是未来 occupancy 预测本身**。`kta_future_occ` 仍然只保存 KTA endpoint prior + UNCERTAIN zero-motion prior；最终未来 occupancy 由 learned World Model 在 generation support 内预测。

冻结理由：相对 endpoint-only Balanced，128-scene validation 中 Hybrid-Balanced-R1 的 Overall oracle、Moving-mIoU、2s/3s Moving 均提高，而 window slot compute 只小幅增加。`hybrid_compact_r1` 质量更低但计算几乎相同，因此不进入正式训练。

## 4. 方法链路

```text
nuScenes / Occ3D + GT future ego protocol
→ historical ego compensation
→ causal real-motion decomposition
→ confident-static: SE(3) transport
→ MOVING: KTA endpoint prior + endpoint/swept generation support
→ UNCERTAIN: zero-motion prior + reachable support
→ frozen OccFM VAE
→ motion-window sparse CFM initialized from OccFM-Fut 196
→ frozen decoder
→ Static-Protected Motion Composition
→ overall / Dynamic / Moving-mIoU v2
```

`free` 是独立状态，绝不等同 confident-static。KTA 只作为 causal motion prior / support builder，不是最终 predictor。

## 5. WM target 与 Moving metric target 分离

```text
generation_support_occ
  = causal(history, KTA endpoint, swept corridor, uncertain reachability)

future_dynamic_target_occ
  = future dynamic-semantic GT ∩ generation_support_occ
  = actual WM supervision

future_moving_occ / gt_moving_support
  = GT interval-motion instance support
  = Moving-mIoU v2 / P0 only
```

训练 target 统一称 **Future WM-dynamic latent**，不要再称 true-moving-only future latent。

## 6. 安装与数据路径

```bash
git clone --recurse-submodules https://github.com/ttt05211/swfm.git
cd swfm
git submodule update --init --recursive
```

上游固定 commit：

```text
64959840a9a4cb54d5b0f6cd4bc6779bb242a853
```

nuScenes / Occ3D 可以继续放在公共数据目录，通过绝对路径传入。`--info-pkl` 必须使用可信 temporal info，并且 formal run 要保证 window 中 12 个 frame 都有 `gt_ego_fut_trajs`。

> `.pt/.pkl` 只加载自己生成或可信来源的实验资产。

## 7. YAML 是唯一 runtime source-of-truth

```text
configs/real_motion_occfm.yaml
```

所有正式资产都绑定 config fingerprint。改变 199/196、trajectory length、ego protocol、motion/support、VAE convention 等都会使旧 cache/resume contract 失效，要求重建，而不是静默复用。

Hybrid support 已经 freeze-closed：正式 YAML 若把 endpoint/swept/uncertain radius 改回旧值，runtime 会直接报错，避免误训练旧方案。

## 8. 首先跑 transition equivalence

SWFM 检查：

```text
Official OccFM-Fut 196
50×50
12-step trajectory
vs
Modified SWFM transition
50×50 + zero prior + origin=(0,0)
```

命令：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/check_transition_equivalence.py \
  --ckpt logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --device cuda \
  --output outputs/transition_equivalence.json
```

脚本会额外确认 `traj_encoder.0.weight` 真正从 12-step checkpoint 加载。若误传 199，会直接失败。Equivalence 失败时不要训练。

## 9. Prepare

```bash
python tools/real_motion/prepare_nuscenes.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --output data/prepared_val \
  --max-windows 16
```

当前 prepared version：

```text
real_motion_prepared_v6_hybrid_balanced_r1
```

旧 prepared/cache 必须重建。在线 `include_gt=False` 不读取 future semantic / instance GT，但 **会读取协议允许的 GT future ego pose / temporal trajectory**。

## 10. P0 / support freeze

P0-A：Full / causal-static-only / moving+uncertain-only decomposition。

P0-B：Generation reachability 使用 future Moving **arrival occupancy**，不是 dual-box metric support。

P0-C：检查 future Moving GT instance 是否被错误锁入 confident-static。

P0-D / P0-D2：比较 Decomposition Oracle、endpoint support、swept support、hybrid support，并报告 actual future arrival coverage 与 window cost。

最终 support freeze 使用：

```text
scene_disjoint_midpoint_one_window_per_scene_v1
128 validation scenes
seed = 20260828
formal route = hybrid_balanced_r1
```

P0-E：VAE/reconstruction 与 sparse target canvas 检查。

## 11. Build latent cache

```bash
python tools/real_motion/build_latent_cache.py \
  --prepared data/prepared_train \
  --vae-ckpt logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --output data/latent_train \
  --empty-latent data/empty_latent.pt
```

当前 latent cache version：

```text
real_motion_v6_hybrid_balanced_r1
```

cache metadata 会显式保存 trajectory / VAE / config fingerprint；Hybrid support 的 geometry/radii 已进入 config fingerprint。formal trainer 会 fail-closed 检查 train/val cache、empty latent、VAE SHA256、config fingerprint、resume contract 和 upstream checkpoint reuse。

## 12. Tiny / full training

先做 Tiny smoke，不要直接启动 200-epoch full run：

```bash
python tools/real_motion/train_sparse.py \
  --cache data/latent_tiny \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --output logs/swfm/tiny.pt \
  --steps 200 \
  --amp
```

Tiny trainer 也会检查 cache fingerprint 和 empty-latent contract，旧 v5 endpoint-only cache 不能用于正式 smoke。

Full 单张 L40S：

```bash
python tools/real_motion/train_full.py \
  --train-cache data/latent_train \
  --val-cache data/latent_val \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --output-dir logs/swfm/full
```

多 GPU：

```bash
torchrun --nproc_per_node=<GPU_NUM> tools/real_motion/train_full.py \
  --train-cache data/latent_train \
  --val-cache data/latent_val \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --output-dir logs/swfm/full
```

## 13. L40S tuning

```bash
python tools/real_motion/tune_l40s.py \
  --cache data/latent_train \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt \
  --output outputs/l40s_tune.json
```

默认保持 Sparse CFM BF16、VAE FP32、TF32、fused AdamW、shard-aware I/O，并实测 workers / batch size。FP8 不进入主协议。

## 14. Inference / evaluation

```bash
python tools/real_motion/infer_nuscenes.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/val_temporal_info.pkl \
  --vae-ckpt logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt \
  --sparse-ckpt logs/swfm/full/ckpt/best.pt \
  --output outputs/predictions
```

正式 evaluator 默认要求 prediction set 完整；只有显式 `--allow-missing-predictions` 才允许诊断性子集评测。

## 15. Frozen Moving-mIoU v2

```text
protocol: interval_displacement_v2
speed threshold: 0.5 m/s
box margin: 0.5 m
horizons: 1s / 2s / 3s
classes: [2,3,4,5,6,7,9,10]
aggregation: per-horizon class mIoU → mean over horizons
```

Maneuver / KTA-hard 只改变选入 support 的 GT instances，metric accumulator 保持同一 `MovingMIoUV2MultiHorizon`。

## 16. 当前正式执行顺序

```text
pytest
→ OccFM-Fut 196 transition equivalence
→ rebuild Hybrid-v6 prepared smoke
→ rebuild Hybrid-v6 latent smoke
→ tiny 200-step smoke
→ tiny/held-out quality check
→ full train/val prepared assets
→ full latent cache
→ L40S tuning
→ full training
→ inference / complete evaluation
→ subset / Harm-Repair
→ latency / GFLOPs
```

Support geometry 已冻结，不再继续调 endpoint/swept radius。若训练后 quality 不足，优先判断是 **模型没有吃到现有 oracle headroom**，还是 **固定 causal support 本身成为 ceiling**；不要继续无上限扩大规则 support，也不要直接增加 Router / ABE / confidence loss。
