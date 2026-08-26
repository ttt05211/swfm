# SWFM：Real-Motion Sparse Occupancy World Model

> 基于官方 **OccFM (NeurIPS 2025)** 的研究实现。核心原则：**Transport what is physically deterministic; generate only what truly moves.**

官方 OccFM 固定在 `upstream_occfm` submodule；SWFM 自己的代码位于 `real_motion/` 与 `tools/real_motion/`。完整方法说明见根目录 `real_motion_sparse_occ_world_model_final_plan.html`。

## 1. 当前实现状态

当前源码链路已经闭环：

```text
nuScenes / Occ3D
→ ego compensation
→ causal real-motion decomposition
→ SE(3) static transport
→ causal KTA + horizon-dependent motion tube
→ frozen OccFM VAE / sharded latent cache
→ motion-window sparse CFM
→ frozen decoder
→ Static-Protected Motion Composition
→ overall / Dynamic / Moving-mIoU v2
→ P0 / subset / Harm-Repair / latency / GFLOPs
```

代码就绪不等于真实实验已通过。第一次在服务器上运行，必须先做 official transition equivalence，再按 P0-A→B→C→D→E 验收，不能直接 full train。

## 2. 安装与路径

```bash
git clone --recurse-submodules https://github.com/ttt05211/swfm.git
cd swfm
git submodule update --init --recursive
```

上游固定 commit：

```text
64959840a9a4cb54d5b0f6cd4bc6779bb242a853
```

推荐 checkpoint 仍沿用 OccFM 官方布局：

```text
logs/occfm/2s_3s_nusc_hist_traj/ckpt/epoch=000199.ckpt
logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt
```

nuScenes / Occ3D 原始数据无需复制进仓库，命令直接传绝对路径。

## 3. YAML 是唯一 runtime source-of-truth

正式配置：

```text
configs/real_motion_occfm.yaml
```

核心脚本都支持：

```bash
--config configs/real_motion_occfm.yaml
--override MODEL.MAX_WINDOWS=10
```

每次 prepare / cache / P0 / train / infer / eval / profile 都应保存 resolved config。Moving-mIoU v2 的 frozen contract 会和代码常量互相校验，禁止 YAML 静默改 benchmark 定义。

## 4. 最终方法 contract

### 4.1 Real-motion decomposition

当前 occupied voxel 被分成：

```text
confident_static / moving / uncertain
```

`free` 是第四种独立状态，绝不等于 confident-static：

```python
confident_static = occupied & (state == STATIC)
```

默认 causal component-track hysteresis：

- `speed >= 0.5 m/s` → moving
- `speed <= 0.2 m/s` 且 persistence 足够高 → confident-static
- 其他 / track 证据不足 → uncertain

### 4.2 Static / KTA

- confident-static 只做 benchmark 允许的 future ego SE(3) transport；
- KTA 只做 causal motion prior + generation-support builder；
- KTA 不是最终 predictor，也不作为 residual-repair target。

### 4.3 Training target 与 metric target 必须分开

```text
generation_support_occ
  = history + KTA 得到的 causal support

future_dynamic_target_occ
  = future dynamic semantic GT ∩ causal generation support
  = WM training target

future_moving_occ / gt_moving_support
  = future GT instance 按 Moving-mIoU v2 判定
  = metric / P0 only
```

训练 target 统一称 **Future WM-dynamic latent / future dynamic-semantic latent inside causal support**，不要再叫 true-moving-only future latent。

## 5. 为什么是 motion-window sparse

官方 OccFM transition 包含 Conv3d、temporal attention、U-Net down/up 和 DiT block，因此 v1 在 50×50 latent grid 上选择少量固定 motion windows（默认 20×20），而不是随意 gather 任意 token。

Window contract：

- `generation_support`：required future support，也是唯一能创建 window 的信号与 future loss mask；
- `planning_support`：history/KTA context，只做相同 future coverage 下的 tie-break；
- 不创建 history-only window，因为不同 sparse windows 不通信；
- overlap windows 共享同一 global 50×50 noise canvas；
- `M_gen` 外每一步 ODE 都 clamp 到 exact `E(empty)`；
- full cache 默认预计算 `window_origins/window_valid`，正式训练不再每 step 运行 CPU planner。

最终 composition：

```python
writable = dynamic_prediction & write_support & ~confident_static
```

所以 `M_gen` 控制哪里计算，不是 wholesale overwrite mask。

## 6. 正式实验前的实现验收

```bash
PYTHONPATH=$PWD pytest -q tests
```

随后必须跑 official 50×50 / zero-prior equivalence：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/check_transition_equivalence.py \
  --ckpt logs/occfm/2s_3s_nusc_hist_traj/ckpt/epoch=000199.ckpt \
  --device cuda \
  --output outputs/transition_equivalence.json
```

失败时不要训练。

## 7. Prepare

```bash
python tools/real_motion/prepare_nuscenes.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --output data/prepared_val \
  --max-windows 16
```

先 smoke；正式生成去掉 `--max-windows`。在线 `include_gt=False` 路径不读取 future semantic / instance GT。

## 8. P0 顺序

### P0-A — Frozen causal real-motion decomposition

```bash
python tools/real_motion/p0_true_motion_decomposition.py \
  --prepared data/prepared_val \
  --vae-ckpt /path/to/vae.ckpt \
  --wm-ckpt /path/to/occfm.ckpt \
  --output outputs/p0_a.json
```

比较 `Full / causal-static-only / moving+uncertain-only`。

### P0-B — Arrival coverage / sparsity / history connectivity

```bash
python tools/real_motion/p0_support_stats.py \
  --prepared data/prepared_val \
  --output outputs/p0_b.json
```

Coverage 使用 future Moving **arrival occupancy**，不是 Moving-mIoU 的 dual-box support。报告 BEV/latent coverage、support ratio、window slot compute ratio，以及 future windows/cells 是否真的拥有 historical evidence。

### P0-C — Actual hard-static blind spot

```bash
python tools/real_motion/p0_causal_audit.py \
  --prepared data/prepared_val \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/info.pkl \
  --output outputs/p0_c.json
```

核心指标直接检查 future Moving GT instance 在 t0 的真实 occupied voxels 与 `t0_confident_static_mask` 的交集。**这里 GT box rasterization 使用 `margin=0`**；Moving-mIoU 的 0.5m margin 只属于 evaluation dual-box support，不能混进 hard-static audit。

报告：any / ≥50% / ≥80% hard-static instance ratio 与 hard-static future-moving voxel ratio。

### P0-D — Two Oracle bounds

1. Decomposition Oracle = SE3-static + all future GT dynamic semantics
2. Causal-Support Oracle = SE3-static + GT dynamic semantics inside causal support

前者→后者差值 = support/reachability loss；后者→learned WM 差值 = learning headroom。

### P0-E — Frozen VAE sanity

```bash
python tools/real_motion/p0_vae_sanity.py \
  --prepared data/prepared_val \
  --vae-ckpt /path/to/vae.ckpt \
  --output outputs/p0_e.json
```

## 9. Frozen Moving-mIoU v2

```text
protocol: interval_displacement_v2
speed threshold: 0.5 m/s
box margin: 0.5 m
horizons: 1s / 2s / 3s
classes: [2,3,4,5,6,7,9,10]
aggregation: per-horizon class mIoU -> arithmetic mean over 1/2/3s
```

Motion 判定使用 GT world XY；support 在 target future ego grid rasterize：`Box(t0→h) ∪ Box(th→h)`。

困难机动 / KTA-hard 只改变被选入 support 的 GT instance；随后仍调用原始 `MovingMIoUV2MultiHorizon`，不允许 per-instance micro IoU 冒充 subset Moving-mIoU。

## 10. L40S 48GB 优化

YAML 默认按 **NVIDIA L40S / Ada** 设置安全高吞吐 profile：

- Sparse CFM：BF16 autocast；BF16 不使用 GradScaler；
- FP32 matmul/cudNN：TF32 + `matmul_precision=high`；
- `cudnn.benchmark=True`；
- Flash / memory-efficient SDPA backend；
- CUDA fused AdamW，不可用自动 fallback；
- pinned memory + persistent workers + prefetch；
- DDP `gradient_as_bucket_view=True`、`static_graph=True`、bucket tuning；
- sharded cache 使用 shard-local / distributed shard-aware sampler，避免随机跨 shard `torch.load`；
- full latent cache 预计算 sparse window plan；
- plan 只做一次 H2D，然后复用于所有 crop。

### 为什么 frozen VAE 默认 FP32

VAE latent 是复用 pretrained OccFM 的表示基础。默认保持 FP32，避免为了 cache 速度无意改变 latent convention。只有 reconstruction/parity 验证通过后才能：

```bash
--override RUNTIME.VAE_AMP.ENABLED=true
```

### 为什么不默认 FP8

L40S 支持更低精度 Tensor Core，但官方 OccFM 没有 Transformer Engine/FP8 recipe。FP8 会引入新的数值/依赖/checkpoint contract，不应偷偷混进主实验。正式主线默认 BF16。

### torch.compile

默认关闭，因为 active-window batch size 是 data-dependent。需要分别 benchmark：

```bash
--override RUNTIME.COMPILE.TRAIN=true
--override RUNTIME.COMPILE.INFERENCE=true
```

只有真实 L40S 上无明显 graph-break/recompile 且吞吐变好才冻结。

## 11. L40S batch tuning

正式 full run 前先测真实 latent cache：

```bash
python tools/real_motion/tune_l40s.py \
  --cache data/latent_train \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt /path/to/occfm.ckpt \
  --output outputs/l40s_batch_tune.json
```

默认扫描 `4,6,8,10,12,16,20,24` samples/GPU，真实执行 BF16 forward/backward + AdamW + EMA，报告 samples/s、transition windows/s、step latency、peak allocated/reserved VRAM。推荐配置默认保留至少 3GB 显存 headroom。

最终把选定 batch 固定到 resolved config，例如：

```bash
--override OPTIMIZATION.FULL.BATCH_SIZE_PER_GPU=8
```

`train_full.py` 使用 `effective LR = BASE_LR × batch_per_gpu × world_size`，因此 batch 改动必须记录。

## 12. L40S 加速 latent cache

```bash
python tools/real_motion/build_latent_cache.py \
  --prepared data/prepared_train \
  --vae-ckpt /path/to/vae.ckpt \
  --output data/latent_train \
  --empty-latent data/empty_latent.pt
```

默认 `VAE_BATCH_SIZE=4`，可在 L40S 上测试 6/8。builder 同时预计算 sparse window plan，并把 VAE SHA256、latent mode、实际 batch size、window plan contract、resolved config 写入 metadata。

full cache 默认过滤完全没有 generation support 的 pure-static samples，避免 DDP collective 不对称。

## 13. Training

Tiny/small diagnosis：

```bash
python tools/real_motion/train_sparse.py \
  --cache data/latent_tiny \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt /path/to/occfm.ckpt \
  --output logs/swfm/tiny.pt \
  --amp
```

正式单张 L40S：

```bash
python tools/real_motion/train_full.py \
  --train-cache data/latent_train \
  --val-cache data/latent_val \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt /path/to/occfm.ckpt \
  --output-dir logs/swfm/full
```

多张 L40S：

```bash
torchrun --nproc_per_node=<GPU数量> tools/real_motion/train_full.py \
  --train-cache data/latent_train \
  --val-cache data/latent_val \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt /path/to/occfm.ckpt \
  --output-dir logs/swfm/full
```

`train_full.py` 支持 distributed shard-aware sampler、official-style split-decay AdamW、fused AdamW、effective LR scaling、warmup+cosine、EMA=0.9999、resume、validation、best/last/periodic checkpoints。

## 14. Inference / evaluation / efficiency

```bash
python tools/real_motion/infer_nuscenes.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/val_info.pkl \
  --vae-ckpt /path/to/vae.ckpt \
  --sparse-ckpt logs/swfm/full/ckpt/best.pt \
  --output outputs/predictions
```

```bash
python tools/real_motion/evaluate_predictions.py \
  --prepared data/prepared_val \
  --pred-dir outputs/predictions \
  --output outputs/eval.json
```

论文效率不要混成一个 Active Token Ratio，分别报告：

```text
Generation-support ratio
Window slot compute ratio
GFLOPs
End-to-end latency / FPS
```

Latency：

```bash
python tools/real_motion/profile_pipeline.py ... --output outputs/profile_l40s.json
```

GFLOPs：

```bash
python tools/real_motion/profile_gflops.py ... --output outputs/gflops.json
```

GFLOPs 只覆盖 `torch.profiler(with_flops=True)` 支持的算子；baseline 必须在同一软件/硬件协议下统计。

## 15. 最终执行顺序

```text
0. pytest
1. official transition equivalence
2. prepare smoke (16 windows)
3. P0-A
4. P0-B
5. P0-C
6. P0-D
7. P0-E + profiler smoke
8. tiny latent cache
9. 64/128 tiny overfit
10. small held-out
11. full latent cache + precomputed window plan
12. L40S batch / compile tuning
13. full training
14. overall / Moving / subset / Harm-Repair
15. L40S latency + GFLOPs
```

任何 gate 明显失败，优先定位 `motion/support → geometry/KTA → VAE → WM optimization → composition/evaluation`，不要直接堆 Router / ABE / confidence loss。

## 16. 关键文件

```text
configs/real_motion_occfm.yaml             runtime source-of-truth
real_motion/runtime_config.py              config + frozen-contract checks
real_motion/perf.py                        L40S BF16/TF32/loader/compile helpers
real_motion/dataset.py                     shard-aware samplers
real_motion/motion.py                      causal real-motion decomposition
real_motion/geometry.py                    ego compensation / SE3
real_motion/kta.py                         causal KTA
real_motion/prepared.py                    raw preparation
real_motion/cache.py                       sharded latent cache
real_motion/windows.py                     sparse window planning / vectorized IO
real_motion/models/cfm.py                  masked CFM inpainting
real_motion/models/transition.py           OccFM-compatible transition
real_motion/composition.py                 static-protected composition
real_motion/metrics/moving_miou_v2.py      frozen Moving-mIoU v2
tools/real_motion/build_latent_cache.py    batched VAE cache + precomputed plans
tools/real_motion/train_sparse.py          tiny/small trainer
tools/real_motion/train_full.py            formal L40S/DDP trainer
tools/real_motion/tune_l40s.py             throughput/VRAM sweep
tools/real_motion/profile_pipeline.py      end-to-end latency
tools/real_motion/profile_gflops.py        supported-op FLOPs
```

数据、官方 checkpoints、prepared/latent shards、trained checkpoints 和 predictions 都放服务器大盘，不提交 GitHub。
