# SWFM 最终实现审查记录

## 结论

仓库已经从 Sparse-WM prototype 补成 protocol-synchronized 的 P0 / tiny / full-training 实现。这里记录最终仍然必须由服务器真实实验验证的边界，避免把静态代码审查误写成实验通过。

## 已修 correctness 问题

1. `FREE` 与 `confident_static` 独立；hard protection 始终与 current occupied 相交。
2. window 内 `M_gen` 外 latent 每个 ODE step clamp 到 exact `E(empty)`。
3. overlapping windows 从统一 50×50 global noise canvas crop 初始噪声。
4. final composition 仅允许 `dynamic_prediction & write_support & ~confident_static` 写入。
5. raw pipeline 已闭环：ego compensation → real-motion → SE3 → causal KTA → tube → target branch。
6. latent cache 使用 sharded lazy format；full cache 默认过滤 empty-generation-support samples。
7. Moving-mIoU v2 严格区分 world-motion 判定与 future-ego support rasterization，并固定 horizon-first aggregation。
8. 50×50 zero-prior official-vs-modified transition equivalence test 已提供。
9. P0-A/B/C/D/E 都有 executable。

## 最终协议同步修复

### A. WM target 不等于 Moving-mIoU moving target

最终训练字段：`future_dynamic_target_occ / future_dynamic_target_latent`，定义为 causal generation support 内的 future dynamic semantic GT。`future_moving_occ / gt_moving_support` 只用于 Moving-mIoU v2 和 P0 metric。

### B. P0-B 使用 arrival coverage

KTA tube reachability 用 future moving arrival occupancy，而不是 dual-box old+new metric support；同时报告 window slot compute ratio 与 history-connected future window 指标。

### C. P0-C 直接审计真实 hard-static mask

对 future Moving GT instance 的 t0 occupied voxels 与 `t0_confident_static_mask` 求交，报告 any / >=50% / >=80% instance overlap 与 voxel overlap。`hist_speed<0.5` 只作为辅助统计。

### D. P0-D 拆成两个 Oracle

- Decomposition Oracle：SE3-static + all GT dynamic semantics
- Causal-Support Oracle：SE3-static + GT dynamic semantics inside causal support

前者到后者的差值是 support/reachability loss；后者到 learned WM 的差值是 learning headroom。

### E. Subset analysis 重新严格复用 Moving-mIoU v2

不再汇总 per-instance intersection/union。Calibration 只冻结 maneuver threshold / KTA-error cuts；test evaluator 选择 instance 后 union 原 dual-box support，并调用同一个 `MovingMIoUV2MultiHorizon`。

### F. Harm/Repair diagnosis 闭环

最终 evaluator 同时报告 voxel micro、instance/tube macro、Oracle KTA-vs-WM selector Moving-mIoU v2 与 selector headroom。

### G. YAML 成为唯一 runtime source-of-truth

新增 `real_motion/runtime_config.py`。核心 prepare/cache/P0/train/infer/eval/profile 脚本读取 `configs/real_motion_occfm.yaml`，CLI 只做显式 override，并保存 resolved config。Frozen Moving-mIoU v2 的 YAML 与代码常量会互相校验。

### H. Tiny trainer 与 full trainer 分离

`train_sparse.py` 只用于 tiny/small diagnosis。`train_full.py` 支持 DDP、official-style split-decay AdamW、effective LR scaling、warmup+cosine、EMA=0.9999、resume、validation、best/last/periodic checkpoints。

### I. GFLOPs 与 latency 分开

`profile_pipeline.py` 统计在线 latency/FPS；`profile_gflops.py` 使用 `torch.profiler(with_flops=True)` 统计支持算子的 FLOPs。论文同时报告 generation-support ratio、window slot compute ratio、GFLOPs 与真实 latency/FPS。

## 服务器上仍必须真实验证

1. official 50×50 transition equivalence 的 RMS/max_abs；
2. nuScenes/Occ3D 路径与 official temporal info pickle；
3. VAE / WM checkpoint state dict；
4. P0-A real-motion functional separability；
5. P0-B coverage-sparsity 与 history connectivity；
6. P0-C hard-static blind spot；
7. P0-D Oracle headroom；
8. P0-E VAE sparse-canvas stability；
9. tiny overfit；
10. full DDP convergence；
11. end-to-end latency 与 GFLOPs。

代码审查不能替代上述实验。任何一项失败，优先定位对应层，不直接增加 Router / ABE / confidence loss。
