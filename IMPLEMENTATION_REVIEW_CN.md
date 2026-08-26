# SWFM 最终实现审查记录

## 结论

仓库已经从 Sparse-WM prototype 补成 protocol-synchronized 的 P0 / tiny / full-training 实现，并针对 NVIDIA L40S 48GB 做了不改变科学协议的运行时优化。源码审查通过不等于真实实验通过；服务器仍必须按 transition equivalence → P0-A/B/C/D/E → tiny overfit → full train 顺序验收。

## 已修 correctness 问题

1. `FREE` 与 `confident_static` 独立；hard protection 始终与 current occupied 相交。
2. window 内 `M_gen` 外 latent 每个 ODE step clamp 到 exact `E(empty)`。
3. overlapping windows 从统一 50×50 global noise canvas crop 初始噪声。
4. final composition 仅允许 `dynamic_prediction & write_support & ~confident_static` 写入。
5. raw pipeline 闭环：ego compensation → real-motion → SE3 → causal KTA → tube → target branch。
6. latent cache 使用 sharded lazy format，full cache 默认过滤 empty-generation-support samples。
7. Moving-mIoU v2 严格区分 world-motion 判定与 future-ego support rasterization，并固定 horizon-first aggregation。
8. 50×50 zero-prior official-vs-modified transition equivalence test 已提供。
9. P0-A/B/C/D/E 都有 executable。
10. crop/scatter 已 vectorize；formal latent cache 可预计算 `window_origins/window_valid`。

## 最终协议同步

### A. WM target 不等于 Moving-mIoU target

- `generation_support_occ`：history + KTA causal support；
- `future_dynamic_target_occ / latent`：causal support 内 future dynamic semantic GT，作为 WM supervision；
- `future_moving_occ / gt_moving_support`：只用于 Moving-mIoU v2 / P0 metric。

### B. P0-B 使用 arrival coverage

Generation reachability 使用 future Moving arrival occupancy，而不是 dual-box old+new metric support。额外报告 window slot compute ratio 与 history-connected future window/cell 指标。

### C. P0-C 直接审计真实 hard-static mask

对 future Moving GT instance 的 t0 真实 object occupancy 与 `t0_confident_static_mask` 求交。这里 GT box rasterization 使用 **margin=0**，再与 t0 同类 semantic occupancy 相交；Moving-mIoU 的 0.5m margin 只属于 evaluation support，不能混入 hard-static audit。

### D. P0-D 两个 Oracle

- Decomposition Oracle：SE3-static + all GT dynamic semantics
- Causal-Support Oracle：SE3-static + GT dynamic semantics inside causal support

### E. Subset analysis 仍严格复用 Moving-mIoU v2

Calibration 冻结 maneuver/KTA cuts；test 只筛 GT instance 并 union 原 dual-box support，然后调用同一个 `MovingMIoUV2MultiHorizon`。不再使用 per-instance micro IoU 冒充 subset Moving-mIoU。

### F. Harm/Repair diagnosis

最终 evaluator 支持 voxel micro、instance/tube macro 与 Oracle KTA-vs-WM selector headroom；这些只用于 post-training diagnosis，不进入 inference。

### G. YAML source-of-truth

`configs/real_motion_occfm.yaml` + `real_motion/runtime_config.py` 统一控制 method/runtime/optimization。每次 run 保存 resolved config；Frozen Moving-mIoU contract 会和代码常量互相校验。

## L40S 48GB 最终优化

主线不默认 FP8；保持可复现实验协议：

- Sparse CFM：BF16 autocast；BF16 不使用 GradScaler；
- TF32 + `torch.set_float32_matmul_precision("high")`；
- cudNN benchmark；
- Flash / memory-efficient SDPA backend；
- CUDA fused AdamW，失败 fallback；
- pinned memory + persistent workers + prefetch；
- Frozen OccFM VAE 默认 FP32，只有 parity 通过后才显式启用 VAE BF16；
- full cache batched VAE encoding，并预计算 sparse window plan；
- `DistributedShardSampler` 避免普通 DistributedSampler 全局 shuffle 导致 shard I/O thrashing；
- plan 只做一次 H2D 后复用于所有 crop；
- DDP `gradient_as_bucket_view=True`、`static_graph=True`、bucket tuning；
- `torch.compile` 默认关闭，只有真实 L40S benchmark 证明有效时开启；
- `tune_l40s.py` 扫描 4/6/8/10/12/16/20/24 samples/GPU，报告 samples/s、transition windows/s、step latency、peak allocated/reserved VRAM，并默认保留 3GB memory headroom。

## 本轮静态验收

- 新增/修改 Python 文件再次通过 `python -m py_compile`；
- dependency-light L40S runtime helper tests：PASS；
- distributed shard sampler tests：PASS；
- 之前 protocol closure baseline 已通过完整 dependency-light tests；
- official checkpoint + CUDA 的 transition equivalence 必须在服务器真实执行，不能由静态审查替代。

## 服务器仍必须真实验证

1. official 50×50 transition equivalence RMS/max_abs；
2. nuScenes/Occ3D 路径与 temporal info pickle；
3. VAE / WM checkpoint state dict；
4. P0-A real-motion functional separability；
5. P0-B coverage-sparsity/history connectivity；
6. P0-C hard-static blind spot；
7. P0-D Oracle headroom；
8. P0-E VAE sparse-canvas stability；
9. 64/128 tiny overfit；
10. L40S batch / compile tuning；
11. full convergence；
12. end-to-end latency 与 GFLOPs。

任何 gate 失败，优先定位 `motion/support → geometry/KTA → VAE → WM optimization → composition/evaluation`，不要直接增加 Router / ABE / confidence loss。
