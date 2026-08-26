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
11. SWFM `tools` 已显式 package 化，复用 sibling entrypoint 的脚本把 repo `ROOT` 放在 upstream OccFM 前，避免 upstream `tools` 遮蔽 `tools.real_motion`。
12. 正式 evaluator 默认要求 prediction set 完整；缺失任何 prepared sample 都 fail，只有显式 `--allow-missing-predictions` 才允许诊断性子集评测。

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

### E. P0-E 同时检查 metric target 与实际 WM target

P0-E 现在同时报告：

- true-moving reconstruction（`future_moving_occ`）；
- WM-target reconstruction（`future_dynamic_target_occ`）；
- actual WM target on causal sparse `E(empty)` canvas；
- sparse canvas 的 Moving-mIoU v2 projection。

### F. Subset analysis 仍严格复用 Moving-mIoU v2

Calibration 冻结 maneuver/KTA cuts；test 只筛 GT instance 并 union 原 dual-box support，然后调用同一个 `MovingMIoUV2MultiHorizon`。不再使用 per-instance micro IoU 冒充 subset Moving-mIoU。

### G. Harm/Repair diagnosis

最终 evaluator 支持 voxel micro、instance/tube macro 与 Oracle KTA-vs-WM selector headroom；这些只用于 post-training diagnosis，不进入 inference。

### H. YAML source-of-truth 与 formal asset gate

`configs/real_motion_occfm.yaml` + `real_motion/runtime_config.py` 统一控制 method/runtime/optimization，并生成稳定的 cache/resume contract fingerprint。正式 `train_full.py` 默认 fail-closed：

- train/val cache 必须使用同一 VAE SHA256、latent mode、VAE AMP convention、latent support radius、motion/support/target contract；
- `empty_latent.pt` 必须携带并匹配 VAE/cache fingerprint；
- upstream OccFM transition 的 shape-safe reuse 必须达到 `MODEL.MIN_UPSTREAM_REUSE_FRACTION`，且关键 backbone/trajectory blocks 必须实际加载；
- resume checkpoint 必须匹配当前 config contract、cache contract、upstream checkpoint SHA256 和 exact empty latent。

这些 gate 用来防止“训练几天后才发现资产混用”。

### I. DDP validation 不重复 padding sample

Training 的 distributed shard sampler 为 collective 对齐可以 rank-local padding；validation 使用 exact no-padding sampler。EMA validation 各 rank 可以处理不同样本数，最后只 all-reduce loss numerator/denominator，因此 `best.pt` 不受重复样本 bias。

### J. GT future ego protocol 已显式冻结

SWFM 主协议明确使用 **GT future ego pose / GT ego trajectory**：

- future ego poses 用于 deterministic SE(3) static transport；
- trajectory conditioning 优先读取 official info 中的 `gt_ego_fut_trajs`；
- 这是信息协议选择，不是 future semantic/instance GT 泄漏。

官方 OccFM 同时发布了 “with future trajectory” 和 “without future trajectory” 两套 forecasting protocol，因此 GT future trajectory 本身是官方支持的评测设置。SWFM 使用 `occfm.yaml` hist-trajectory checkpoint 仅作为权重初始化；论文主表若给 SWFM 使用 GT future ego，则 dense OccFM 等 baseline 也必须使用相同 future-ego information（对官方 OccFM 应使用/对齐 future-trajectory variant），不能拿 hist-only baseline 直接做不等信息量比较。

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
- `tune_l40s.py` 扫描 workers 与 4/6/8/10/12/16/20/24 samples/GPU，报告 data wait、samples/s、transition windows/s、step latency、peak allocated/reserved VRAM，并默认保留 3GB memory headroom。

## Serialized asset trust boundary

仓库中的 prepared/cache/prediction `.pt` 以及 nuScenes/official-info `.pkl` 属于实验资产。部分数据结构需要 `torch.load(weights_only=False)` / `pickle.load` 才能读取，因此只能加载**自己生成或可信来源**的文件；不要直接运行来源不明的 `.pt/.pkl`。模型权重能使用 `weights_only=True` 的路径优先使用安全加载。

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
