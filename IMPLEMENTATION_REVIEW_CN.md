# SWFM P0-ready Closure：实现与代码审查记录

## 结论

这次修复的目标不是宣称“模型已经在 nuScenes 上训练成功”，而是把仓库从 **Sparse-WM backend prototype** 补成一个可以在服务器上按 P0 顺序真实验收的闭环实现。

当前本地 dependency-light 验收：

```text
python -m py_compile: PASS
pytest：31 passed，1 skipped（integration equivalence）
```

skip 项是必须在服务器用 official checkpoint + GPU 执行的 50×50 transition equivalence。

## 已修 correctness 问题

### 1. free 不再等于 confident-static

旧：

```python
state[~occupied] = STATIC
```

风险：cache builder 若 `state == STATIC`，当前 free / future arrival cell 会被永久保护。

新：

```text
STATIC / MOVING / UNCERTAIN / FREE
```

公开 API 直接返回 `confident_static = occupied & static`。

### 2. window margin 不再是未监督随机输出

旧训练只有 `M_gen` 内 loss，采样却生成整个 window 并 scatter。

新 CFM 是 masked inpainting：

```text
M_gen 内：noise / denoise
M_gen 外：每个 ODE step clamp 到 E(empty)
```

scatter 前再 safety clamp。

### 3. overlap window 使用统一 global z0

旧：每个 flattened window 独立 `randn`。

新：先构造 `[B,F,C,50,50]` global noise canvas，再 crop 给 window。共享 cell 的初始 noise 一致。

### 4. decoder spillover 也受控

`M_gen` 仍不是“整块 overwrite”。最终 composition：

```text
dynamic prediction AND write_support AND NOT confident_static
```

只有动态 semantic voxel 能写，不会因为 support 存在就把道路/建筑整块覆盖。

### 5. 前半段 pipeline 已补

新增：

- `geometry.py`: ego compensation / SE3
- `kta.py`: causal occupancy KTA baseline
- `nuscenes_adapter.py`: poses / annotations / Moving-mIoU endpoint adapter
- `prepared.py`: raw history → split → SE3/KTA/support → GT target branch
- `prepare_nuscenes.py`: sharded prepared dataset CLI

### 6. VAE cache builder 已补，且 full dataset 不再一个大 `.pt`

`real_motion_v3` latent cache 使用 shard + `index.json` + lazy one-shard loading。

### 7. Moving-mIoU v2 坐标 contract 闭环

Motion 判定：world XY。

Support rasterization：target future ego grid。

动态类固定为 `(2,3,4,5,6,7,9,10)`。

1s/2s/3s 单独 mIoU 后算术平均。

### 8. P0 executables 已补

- A: `p0_true_motion_decomposition.py`
- B: `p0_support_stats.py`（per-horizon BEV + latent）
- C: `p0_causal_audit.py`
- D: `p0_oracle.py`（decomposition oracle + causal-support oracle）
- E: `p0_vae_sanity.py`
- F: `profile_pipeline.py`

### 9. 50×50 zero-prior equivalence test 已补

`check_transition_equivalence.py` 比较 official `FLOW_MATCHING_DOWN_X4_DiT` 与修改版。

必须在真正训练前用 official ckpt + fp32/GPU 跑通过。

### 10. crop/scatter GPU sync 已处理

原 B×K Python crop/scatter 改成 vectorized gather/scatter_add。

planner 在 CPU 上用 integral image 穷举 50×50 grid 的全部合法 top-left；future required coverage 为第一目标，context 只做 tie-break。crop/scatter 本身已 vectorize，避免 GPU 循环同步。

## 仍需服务器真实验证（不是源码静态审查能够替代的）

1. official transition equivalence 的实际数值误差；
2. `NuScenesWindowSource` 与本地数据路径/info pickle 是否完全一致；
3. official VAE / WM checkpoint 名称与 state dict 是否吻合；
4. P0-A real-motion separability 是否成立；
5. KTA tube coverage vs active ratio；
6. 64/128 tiny overfit；
7. end-to-end latency 是否真的因为 window sparse 下降。

## Go / No-Go

### 在 tiny training 前必须全部满足

- transition equivalence PASS；
- P0-A moving-only / static-only 没有明显 representation collapse；
- P0-B coverage/active-ratio 有合理 sparse trade-off；
- P0-C blind-spot 不足以推翻 confident-static assumption；
- P0-D Oracle 有明显 headroom；
- P0-E frozen VAE moving-only reconstruction 可接受。

如果其中任何一个失败，优先修对应层，不直接增加 Router / ABE / confidence loss。


## 合并前第三轮审查新增修复

### 11. 训练 target 与 Moving-mIoU GT support 已彻底解耦
第三轮复核发现，旧草稿把 `future_moving_occ`（由未来 GT instance speed>=0.5m/s 定义）直接拿去做 WM target。这个会让 causal support 内“未来没有达到 Moving 指标阈值”的动态语义对象被错误训练成 empty。

最终 contract：
- `generation_support_occ`：只由 history + KTA 决定；
- `future_dynamic_target_occ`：future GT 中位于 causal support 内的动态 semantic voxel，用于训练；
- `future_moving_occ` / `gt_moving_support`：只用于 Moving-mIoU v2 / P0 evaluation。

因此 future GT 不参与“哪里算/哪里生成”的决策，只提供 causal support 内的监督标签。

### 12. P0-B coverage 改为 arrival coverage
Moving-mIoU dual support 同时包含旧位置与新位置，旧位置用于惩罚 trailing ghost；它不是 generator 必须覆盖的区域。P0-B 现在用 eligible moving instance 在 future GT 的真实动态 arrival voxels 计算 KTA tube coverage，避免为了覆盖旧位置把 tube 无意义扩宽。

### 13. Window planner 改为 exhaustive future-first + context tie-break
50x50 latent map 上直接用 integral-image 对所有合法 top-left 做精确 window sum。Future required coverage 永远第一优先级，history/KTA context 只在相同 future coverage 下决定窗口向哪边平移。

### 14. VAE latent mode / empty latent 训练推理一致性
训练 cache metadata 记录 `latent_mode`。Sparse checkpoint 额外保存训练时 exact `empty_latent`。在线 inference/profiler 默认 `--vae-mode auto`，与训练 mode 不一致会直接报错，并复用 checkpoint 内的 exact empty latent。

### 15. Final evaluator 闭环
新增 `evaluate_predictions.py`：sample_id 对齐 prepared GT 与 online prediction，报告 overall、Dynamic、Moving-mIoU v2、同 composition 规则 KTA baseline 与 Moving-support harm/repair。


### 16. Real-motion detector 从 voxel persistence 升级为 component-track hysteresis
纯 voxel persistence 会把一个平移但连续帧仍大量重叠的汽车拆成“静态内部 + 运动边缘”，这会直接污染 confident-static protection。最终默认对当前同语义 connected component 做 causal backward association，用历史 centroid speed + persistence 整体判定 moving/static/uncertain；track 不完整的新出现 component 进入 uncertain，不 hard-freeze。新增单测覆盖“2×2 汽车连续平移仍整车 moving”。

### 17. Static/KTA prior 数值尺度与官方 OccFM 对齐
官方 OccFM 在 transition 前把 latent 乘 `RESCALE_FACTOR=10`。第三轮后又检查到新 prior adapter 若直接读取未缩放 VAE latent，条件幅值会与 denoising state 的数值 convention 不一致。现在 future-only prior 先严格补齐 history zero prior，再统一乘 10；不再通过时间长度猜测 prior 是否已对齐。新增单测检查 prefix + scale。

### 18. P0-D Oracle 定义修正
Decomposition Oracle 现在使用 **全部 future dynamic semantics** + SE3 static，只受 confident-static protection；Causal-Support Oracle 才额外限制在 causal generation support 内。两者差值才真正对应 support/reachability headroom。

### 19. Profiler 不再把磁盘 I/O 算成模型 FPS
raw labels / pose / info 表查询在 timing loop 外预加载；计时从已经拿到 raw arrays/poses 开始，包含 real-motion/SE3/KTA、VAE、planning、NFE、scatter、decoder、composition。这样既不拿 cache-only backend 冒充端到端，也不把文件系统吞吐混进模型 latency。

### 20. 困难机动 / KTA-hard 分层已与 final evaluator 接通
`evaluate_predictions.py --subset-records` 会输出每个 Moving instance 的 speed、historical speed、speed change、heading change、KTA center error、support IoU；`analyze_motion_subsets.py` 再按 calibration-frozen 阈值汇总 Uniform/Accel/Turning 与 KTA Easy/Medium/Hard。主 Moving-mIoU v2 contract 不变。

### 21. 内置 KTA 的定位
`real_motion/kta.py` 是 dependency-free reference baseline，目的是让 raw→P0→cache→inference 闭环可独立运行。正式论文优先接入已经验证过的 stronger KTA；接口只要求 causal history 输入并产出 future semantic prior/support，不能让 future GT 进入。


### 22. Sharded cache 随机训练不再反复 thrash 磁盘
如果 sharded dataset 仍使用全局 `shuffle=True`，lazy loader 只缓存一个 shard 时会频繁跨 shard，几乎每个样本都可能重新加载整个 `.pt`。新增 `ShardShuffleSampler`：每轮随机 shard 顺序，并在 shard 内随机样本，但一个 shard 连续消费，保留随机性同时避免 I/O thrashing。

### 23. VAE / support contract 增加运行时一致性保护
latent cache metadata 保存 VAE SHA256、latent mode、latent extra radius；sparse checkpoint 继承这些 metadata 并保存 exact empty latent。Online inference/profiler 自动恢复 window/support/latent mode，并拒绝 VAE fingerprint mismatch。support radius 只有显式 ablation flag 才允许改。

### 24. P0-B 增加“同一 future window 是否真的看到 history”检查
全局 planning/context coverage 可能掩盖一个问题：不同 sparse window 不通信。现在 P0-B 额外报告 future window 中含历史 motion evidence 的比例，以及 required future cells 被“含 history 的 window”覆盖的比例。这个统计比单纯 context union coverage 更接近当前 backend 的真实因果信息可达性。

### 25. Stratified test 不允许在 test 上重新拟合 KTA-hard cut
`analyze_motion_subsets.py` 现在强制二选一：calibration 用 `--fit-kta-cuts`，test 用显式 `--kta-cuts c1,c2`。Turning 也改为按 turn-rate (deg/s) 而不是 raw heading angle，避免 1s/3s horizon 因时间长度不同产生不一致阈值。


### 26. Cache target 字段去掉 ambiguous `future_moving_latent`
既然训练 target 已与 Moving-mIoU metric-only GT 解耦，继续把字段叫 `future_moving_latent` 很容易让后续实验重新混淆。v3 canonical key 改为 `future_dynamic_target_latent`；v1/v2 cache 仍可读取，并在 loader 内只做 key alias 升级。
