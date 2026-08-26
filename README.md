# SWFM：Real-Motion Sparse Occupancy World Model

> 基于官方 **OccFM (NeurIPS 2025)** 的研究实现。核心原则：**Transport what is physically deterministic; generate only what truly moves.**

本仓库固定官方 OccFM 为 `upstream_occfm` submodule，不直接魔改上游源码。我们自己的实现集中在 `real_motion/` 与 `tools/real_motion/`。

完整方法说明同时见根目录：`real_motion_sparse_occ_world_model_final_plan.html`。

---

## 1. 当前实现状态

当前代码已经闭环：

```text
nuScenes / Occ3D
  -> ego compensation
  -> causal real-motion decomposition
  -> SE(3) static transport
  -> causal occupancy KTA
  -> horizon-dependent motion tube
  -> frozen OccFM VAE / sharded latent cache
  -> motion-window sparse CFM
  -> frozen decoder
  -> Static-Protected Motion Composition
  -> overall / Dynamic / Moving-mIoU v2
  -> P0 / subset / Harm-Repair / efficiency analysis
```

代码就绪不等于服务器实验已经通过。第一次在真实数据 + 官方 checkpoint + GPU 上运行时，必须先做 transition equivalence，再按 P0-A→B→C→D→E 顺序验收，不能直接 full train。

---

## 2. 安装与目录

```bash
git clone --recurse-submodules https://github.com/ttt05211/swfm.git
cd swfm
git submodule update --init --recursive
```

官方上游固定 commit：

```text
64959840a9a4cb54d5b0f6cd4bc6779bb242a853
```

环境优先沿用官方 OccFM；额外需要 `nuscenes_devkit numpy torch einops einops_exts easydict pyyaml pytest`。

推荐 checkpoint 路径沿用 OccFM：

```text
logs/occfm/2s_3s_nusc_hist_traj/ckpt/epoch=000199.ckpt
logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt
```

nuScenes / Occ3D 原始数据不需要复制进仓库，命令里传绝对路径即可。

---

## 3. YAML 是唯一 runtime source-of-truth

正式配置：

```text
configs/real_motion_occfm.yaml
```

它统一冻结：

- `WINDOW_HW / MAX_WINDOWS / MIN_WINDOW_COVERAGE`
- real-motion hysteresis
- KTA 参数与六个 horizon 的 tube radius
- latent support radius / VAE mode
- WM target contract
- Moving-mIoU v2 contract
- tiny/full optimization
- maneuver analysis thresholds

核心脚本均接受：

```bash
--config configs/real_motion_occfm.yaml
--override MODEL.MAX_WINDOWS=10
```

CLI override 必须是显式的。每个 prepare/P0/train/infer/eval/profile 输出都会保存 resolved config。

`runtime_config.py` 还会把 YAML 中的 Moving-mIoU v2 定义与代码冻结常量逐项核对，禁止配置文件静默改变 benchmark contract。

---

## 4. 最终方法 contract

### 4.1 Real-motion decomposition

当前 occupied voxel 被分成：

```text
confident_static / moving / uncertain
```

free 是第四种独立状态，绝不等于 confident-static：

```python
confident_static = occupied & (state == STATIC)
```

默认 detector 不是纯 voxel persistence，而是 causal component-track hysteresis：

- `speed >= 0.5 m/s`：moving
- `speed <= 0.2 m/s` 且 persistence 足够高：confident-static
- 其他：uncertain

新出现或跟踪证据不足的 component 进入 uncertain，不 hard-freeze。

### 4.2 Static branch

`confident_static` 只做 benchmark 允许的 future ego `SE(3)` transport，不经过 generative WM。

### 4.3 KTA 的角色

KTA 只承担：

1. causal motion prior
2. future generation-support builder

不是最终 predictor，也不作为 residual repair target。

### 4.4 训练 target 与 metric target 必须分开

这是最终冻结的术语：

```text
generation_support_occ
  = history + KTA 得到的 causal support

future_dynamic_target_occ
  = future dynamic semantic GT ∩ causal generation support
  = WM training target

future_moving_occ / gt_moving_support
  = future GT instance 按 Moving-mIoU v2 判定后得到
  = metric / P0 only
```

因此不要再把训练 target 叫 `true-moving future latent`。正确名称是：

> **Future WM-dynamic latent / future dynamic-semantic latent inside causal support**

一个历史 uncertain、未来恰好停住的车仍然属于 WM 的责任对象；不会因为未来速度低于 0.5 m/s 而被训练成 empty。

---

## 5. 为什么是 motion-window sparse

官方 OccFM transition 是 `Conv3d + temporal attention + U-Net down/up + DiT blocks`，不是任意长度 token Transformer。为了最大复用官方 checkpoint，v1 在 50×50 latent grid 上只选择少量固定 window（默认 20×20）。

因此论文应写：

> **motion-guided sparse windows / motion-block sparse world model**

而不是“任意 active tokens”。

Window contract：

- `generation_support`：required future support，唯一能创建 window 的信号，也是 future loss mask；
- `planning_support`：history/KTA context，只做同等 future coverage 下的 tie-break；
- 不创建 history-only window，因为不同 sparse windows 之间没有 communication。

window crop/scatter 已 vectorize；重叠 window 共享同一 global 50×50 noise canvas。

---

## 6. Sparse CFM correctness

### support 外不是未监督随机 latent

训练与每一步 ODE sampling 都执行 masked latent inpainting：

```text
M_gen 内   -> noise / denoise
M_gen 外   -> clamp to exact E(empty)
```

### Static-Protected Motion Composition

最终权限分开：

```text
generation support -> 哪里计算
write support      -> 动态 prediction 哪里允许写
confident static   -> 哪里绝对保护
```

最终只写：

```python
writable = dynamic_prediction & write_support & ~confident_static
```

所以 `M_gen != wholesale overwrite mask`。

---

## 7. 正式实验前：先做实现验收

### Unit tests

```bash
PYTHONPATH=$PWD pytest -q tests
```

上一版 dependency-light baseline 为 `31 passed, 1 skipped`；本版新增 runtime-config 与 strict-subset tests。请以最终 checkout 在服务器跑出的完整结果为准。唯一预期 skip 是 official checkpoint + CUDA 的 transition equivalence integration test。

### Official transition equivalence

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/check_transition_equivalence.py \
  --ckpt logs/occfm/2s_3s_nusc_hist_traj/ckpt/epoch=000199.ckpt \
  --device cuda \
  --output outputs/transition_equivalence.json
```

使用 `50×50 / zero prior / origin=(0,0) / same input+timestep+trajectory+checkpoint` 比较官方 transition 与修改版。失败时不要训练。

---

## 8. 生成 prepared raw shards

先 smoke：

```bash
PYTHONPATH=$PWD python tools/real_motion/prepare_nuscenes.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --output data/prepared_val \
  --max-windows 16
```

正式生成时去掉 `--max-windows`。

在线 `include_gt=False` 路径不触碰 future semantic/instance GT。

---

## 9. P0 顺序

### P0-A：Frozen true-motion decomposition（第一个方法实验）

```bash
python tools/real_motion/p0_true_motion_decomposition.py \
  --prepared data/prepared_val \
  --vae-ckpt /path/to/vae.ckpt \
  --wm-ckpt /path/to/occfm.ckpt \
  --output outputs/p0_a.json
```

同一 frozen OccFM 比较：

```text
Full / Causal-static-only / Moving+uncertain-only
```

重点看 static-only 对 true-static 的保持，以及 moving-only 对 Moving-mIoU v2 的保持。

### P0-B：arrival coverage / sparsity / window connectivity

```bash
python tools/real_motion/p0_support_stats.py \
  --prepared data/prepared_val \
  --output outputs/p0_b.json
```

Coverage 使用 **future moving arrival occupancy**，不是 Moving-mIoU dual-box old+new support。报告：

- coverage-BEV / active-BEV
- coverage-latent / active-latent
- true-moving sparsity
- slot compute ratio
- future windows with historical evidence ratio
- future required cells inside history-connected windows ratio

### P0-C：真正的 hard-static blind spot

```bash
python tools/real_motion/p0_causal_audit.py \
  --prepared data/prepared_val \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/info.pkl \
  --output outputs/p0_c.json
```

核心不再是 `hist_speed < 0.5`，而是对每个 future Moving GT instance 直接检查其 t0 occupied voxels 与真实 `t0_confident_static_mask` 的交集，报告：

- any-overlap instance ratio
- >=50% hard-static instance ratio
- >=80% hard-static instance ratio
- hard-static → future-moving voxel ratio

historical speed 只作为辅助解释；no-history/birth 作为 innovation 原因。

### P0-D：两个 Oracle

```bash
python tools/real_motion/p0_oracle.py \
  --prepared data/prepared_val \
  --output outputs/p0_d.json
```

1. **Decomposition Oracle** = SE3-static + all future GT dynamic semantics
2. **Causal-Support Oracle** = SE3-static + GT dynamic semantics inside causal support

两者差值是 support/reachability loss；Causal-Support Oracle 与 learned WM 的差值才是 learning headroom。

### P0-E：Frozen VAE sanity

```bash
python tools/real_motion/p0_vae_sanity.py \
  --prepared data/prepared_val \
  --vae-ckpt /path/to/vae.ckpt \
  --output outputs/p0_e.json
```

---

## 10. Frozen Moving-mIoU v2

主 protocol 永远保持：

```text
protocol: interval_displacement_v2
speed threshold: 0.5 m/s
box margin: 0.5 m
horizons: 1s / 2s / 3s
classes: [2,3,4,5,6,7,9,10]
aggregation: per-horizon class mIoU -> arithmetic mean over 1/2/3s
```

Motion 判定使用 GT instance world XY；support rasterization 使用 target future ego grid；support 是 `Box(t0→h) ∪ Box(th→h)`，同时惩罚 trailing ghost 与 missed arrival。

---

## 11. Latent cache

```bash
python tools/real_motion/build_latent_cache.py \
  --prepared data/prepared_train \
  --vae-ckpt /path/to/vae.ckpt \
  --output data/latent_train \
  --empty-latent data/empty_latent.pt
```

cache 为 sharded lazy format，并记录：

- VAE SHA256
- latent mode
- latent support radius
- resolved config

full-training cache 默认过滤完全没有 generation support 的 pure-static samples，避免 DDP 某些 rank 无 forward 导致 collective 不对称。

---

## 12. Tiny training 与 full training 分离

### Tiny / small diagnosis

```bash
python tools/real_motion/train_sparse.py \
  --cache data/latent_tiny \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt /path/to/occfm.ckpt \
  --output logs/swfm/tiny.pt \
  --amp
```

只用于 64/128 overfit 与 small held-out。

### Full training

```bash
torchrun --nproc_per_node=4 tools/real_motion/train_full.py \
  --train-cache data/latent_train \
  --val-cache data/latent_val \
  --empty-latent data/empty_latent.pt \
  --upstream-ckpt /path/to/occfm.ckpt \
  --output-dir logs/swfm/full \
  --amp
```

`train_full.py` 支持：

- DDP
- 与官方 OccFM 对齐的 split-decay AdamW
- effective LR = `BASE_LR × batch_per_gpu × world_size`
- warmup + cosine
- EMA=0.9999
- resume
- validation
- best / last / periodic checkpoint

默认 optimization 来自 YAML，不在 Python 脚本里另维护一套实验超参。

---

## 13. Inference / evaluation

```bash
python tools/real_motion/infer_nuscenes.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/val_info.pkl \
  --vae-ckpt /path/to/vae.ckpt \
  --sparse-ckpt logs/swfm/full/ckpt/best.pt \
  --output outputs/predictions
```

checkpoint 中的 resolved config、VAE mode、VAE fingerprint、latent support 和 exact empty latent 都会用于一致性检查。

总体评估：

```bash
python tools/real_motion/evaluate_predictions.py \
  --prepared data/prepared_val \
  --pred-dir outputs/predictions \
  --output outputs/eval.json
```

---

## 14. 困难机动 / KTA-hard：严格重算 Moving-mIoU v2

先在 train/calibration split 冻结 KTA cuts：

```bash
python tools/real_motion/analyze_motion_subsets.py \
  --prepared data/prepared_calib \
  --output outputs/subset_config.json
```

测试时：

```bash
python tools/real_motion/evaluate_predictions.py \
  --prepared data/prepared_val \
  --pred-dir outputs/predictions \
  --subset-config outputs/subset_config.json \
  --output outputs/eval_subsets.json
```

现在 subset **不会**再做 per-instance micro IoU。流程是：

```text
select Turning / Accel / KTA-Hard GT instances
 -> union their original frozen dual-box supports at each horizon
 -> unchanged MovingMIoUV2MultiHorizon
 -> per-class mIoU @ 1s/2s/3s
 -> mean across horizons
```

因此结果才能严格写作 `Moving-mIoU v2 on Turning`、`Moving-mIoU v2 on KTA-Hard`。

---

## 15. Harm / Repair + Oracle selector

最终 evaluator 已闭环：

- moving-support voxel micro harm/repair
- instance/tube macro harm/repair
- Oracle KTA-vs-WM selector Moving-mIoU v2
- Oracle selector headroom（pp）

这些都是 post-training diagnosis，不进入 inference。只有 oracle headroom 明显时才考虑 lightweight gate；CFG 放最后。

---

## 16. 效率必须分四个概念

不要再用一个 Active Token Ratio 代替全部效率：

```text
Generation-support ratio
Window slot compute ratio
GFLOPs
End-to-end latency / FPS
```

Latency/FPS：

```bash
python tools/real_motion/profile_pipeline.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/info.pkl \
  --vae-ckpt /path/to/vae.ckpt \
  --sparse-ckpt logs/swfm/full/ckpt/best.pt \
  --output outputs/profile.json
```

GFLOPs：

```bash
python tools/real_motion/profile_gflops.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/info.pkl \
  --vae-ckpt /path/to/vae.ckpt \
  --sparse-ckpt logs/swfm/full/ckpt/best.pt \
  --output outputs/gflops.json
```

GFLOPs 使用 `torch.profiler(with_flops=True)`，只覆盖其支持的算子；所有 baseline 必须在同一软件/hardware 统计协议下比较，并同时报告真实 latency。

---

## 17. 推荐执行顺序

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
11. full latent cache
12. train_full.py
13. overall / Moving / subset / Harm-Repair analysis
14. latency + GFLOPs
```

任何一步失败，都先判断失败来自：

```text
motion/support
geometry/KTA
VAE representation
WM optimization
composition/evaluation
```

不要直接堆 Router / ABE / confidence loss。

---

## 18. 关键文件

```text
configs/real_motion_occfm.yaml             runtime source-of-truth
real_motion/runtime_config.py              config loader + frozen-contract checks
real_motion/motion.py                      causal real-motion decomposition
real_motion/geometry.py                    ego compensation / SE3
real_motion/kta.py                         causal reference KTA
real_motion/prepared.py                    raw preparation
real_motion/cache.py                       sharded latent cache
real_motion/windows.py                     sparse window planning / IO
real_motion/models/cfm.py                  masked CFM inpainting
real_motion/models/transition.py           OccFM-compatible transition
real_motion/composition.py                 static-protected composition
real_motion/metrics/moving_miou_v2.py      frozen Moving-mIoU v2
real_motion/metrics/diagnostics.py          Harm/Repair / oracle selector
tools/real_motion/train_sparse.py          tiny/small trainer
tools/real_motion/train_full.py            formal DDP trainer
tools/real_motion/evaluate_predictions.py  final evaluator
tools/real_motion/profile_pipeline.py      end-to-end latency
tools/real_motion/profile_gflops.py        supported-op FLOPs
```

---

## 19. 不提交 GitHub 的资产

nuScenes / Occ3D 数据、官方 VAE/OccFM checkpoint、prepared shards、latent shards、trained checkpoints 和 predictions 都应放服务器大盘，不提交本仓库。

上游：`Orbis36/OccFM-NeurIPS2025`，*Towards foundational LiDAR world models with efficient latent flow matching*, NeurIPS 2025。
