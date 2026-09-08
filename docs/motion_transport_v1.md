# motion_transport_v1 — MT-V1-SPEC-2 运行与验收说明

本分支实现独立的 occupancy-space future transport 路线，不加载 VAE、OccFM 或 FM 权重。它复用 `real_motion/strong_w2det.py` 作为 Strong-W2Det/KTA 几何基准，冻结现有 MSP activation checkpoint，仅对路由选中的 Strong-W2Det source 运行 STPN。

## 运行顺序

```bash
CFG=configs/real_motion/motion_transport_v1.yaml
python tools/real_motion/build_motion_transport_v1_manifest.py --config "$CFG" --output data/motion_transport_v1/manifest.json
CUDA_VISIBLE_DEVICES=0 python tools/real_motion/preflight_motion_transport_v1.py --config "$CFG" --output outputs/motion_transport_v1/preflight.json --samples 16
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/real_motion/profile_motion_transport_v1.py --config "$CFG" --output-dir outputs/motion_transport_v1/profile
LOCKED=outputs/motion_transport_v1/profile/resolved_profile_config.yaml
OUT=outputs/motion_transport_v1/run_seed3407
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/real_motion/train_motion_transport_v1.py --config "$LOCKED" --output-dir "$OUT"
```

恢复训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/real_motion/train_motion_transport_v1.py --config "$OUT/resolved_train_config.yaml" --output-dir "$OUT" --resume "$OUT/latest.pt"
```

EMA hard 主评估、bootstrap 与 Q 曲线：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/real_motion/eval_motion_transport_v1.py --config "$OUT/resolved_train_config.yaml" --checkpoint "$OUT/best.pt" --output-dir "$OUT/eval_dev" --split dev --weights ema --budgets 0,4,8,16,32,all --strategies msp,uniform,speed
```

GT future-moving routing 只允许作为诊断：

```bash
python tools/real_motion/eval_motion_transport_v1.py --config "$OUT/resolved_train_config.yaml" --checkpoint "$OUT/best.pt" --output-dir "$OUT/eval_gt_route_diag" --split dev --weights ema --budgets 16 --strategies gt_moving
```

Latency：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/real_motion/eval_motion_transport_v1.py --config "$OUT/resolved_train_config.yaml" --checkpoint "$OUT/best.pt" --output-dir "$OUT/eval_latency" --split dev --weights ema --budgets 0,4,8,16,32,all --latency --latency-warmup 100 --latency-windows 500
```

## Soft renderer 与监督域

正式部署与 checkpoint selection 始终使用 hard forward-floor compositor。soft renderer 只提供训练梯度，**不参与 hard 指标**。

PyTorch 2.6 在恰好落于 trilinear lattice knot 时会选取单边导数；当 CE 目标概率接近 epsilon 时，这会在 zero-initialized motion head 的第一步制造远大于局部有限差分斜率的梯度，并污染 Adam 二阶矩。MT-V1 因此把训练 soft occupancy 定义为一个真正 forward/backward 一致的 anti-aliased surrogate：对每个 XY voxel footprint 采用四个 `(+/-0.25, +/-0.25)` voxel 的对称 quadrature，四个 trilinear probability 的平均值同时作为 forward probability 与 autograd 对象。它不是 straight-through gradient replacement。`compose_hard()` 完全不使用该 surrogate，所以 zero-delta hard KTA identity、部署与指标口径保持不变。

对任一 selected source，soft 查询/替换域 `U` 至少是 **当前预测 support AABB 与该 source 原 KTA support 的并集**。因此 source 大位移或完全移出网格时，旧 KTA 占据仍处于替换域中，会先恢复 background，再按 Strong-W2Det 原 writer 顺序重合成所有 source 与 rest；不会在旧位置留下 ghost。代码同时保留 `full_grid_reference=True` 作为合成回归对照，不用于正式训练。

历史 `mask_lidar` 仍是 causal motion decomposition 的真实观测 mask。未来 CE 的 supervision validity **不再由 `mask_lidar` 决定**：当前 Occ3D 合同下，合法 semantic label `0..17` 为可监督域；future `mask_lidar` 只作为 `future_observed` audit 字段，用于统计 observed/unobserved、free/non-free 与 Moving support 覆盖。冻结的 Overall / Moving-mIoU v2 口径不修改。

## Formal gradient calibration

`profile_motion_transport_v1.py` 固定运行 8 个 calibration batch，并调用与正式代码相同的 `engine.calibrate_lambda()`：

1. STPN 仍从 zero-initialized output 开始；校准图使用 GT-independent、direction-neutral 的 `1e-3` probe family，覆盖 `(±dx, ±dy, ±yaw)` 全 8 个符号象限，避免单一 checkerboard probe 偶然采到 benign octant。
2. 分别计算 occupancy 与 GT rigid-motion auxiliary 对共享 final head 的梯度，以及对六 horizon `(dx,dy,yaw)` 输出的梯度。
3. head 级 recovery-dominance 约束为 `||g_occ|| <= 0.5 * ||lambda * g_motion||`。
4. 另外逐输出坐标检查 gradient conflict；若全局 cosine 掩盖局部反向冲突，用 `output_gradient_lambda_floor` 抬高安全下界。
5. 每个 calibration batch 取 8 个 probe 的安全 envelope；最终 `lambda_ref = max(median(head safe ratios), max(output conflict floors))`，写入 locked config。formal train 不重新校准。
6. 正式 `lambda_at()` schedule 保持原合同：前 10% 使用 `1.0 * lambda_ref`，10%–30% 线性衰减到 `0.25 * lambda_ref`，30% 后保持 `0.25 * lambda_ref`。

profile JSON 会记录 head/output gradient cosine、p50/p95/p99/max、probe index 与每 batch safe ratio，便于复查优化路径。

## Formal profile 与 4 小时预算

50 warmup + 200 measured microsteps 的计时从 **dataset 取样之前**开始，因此包含磁盘读取、Strong-W2Det/source decomposition、MSP candidate/mapping、future target 准备、crop、STPN、renderer、backward 和 optimizer step。DDP 用跨 rank 的最慢 wall time。

profile 另外使用真实 `save_checkpoint()` payload 测完整 optimizer/EMA/RNG/provenance 保存时间，并把 quarter saves、epoch dev、最终 raw/dev/save 预算计入 `epochs_locked`。locked config 写入 `wall_clock_final_reserve_seconds`、`wall_clock_next_group_guard_seconds` 与实测 `runtime.profile_dev_s`。

formal train 在每个 accumulation group 的一致边界检查跨 rank 最大 wall time。如果下一 group 会侵犯 4 小时预算，所有 rank 停在同一边界，写 `phase=budget_stop` 的可恢复 `latest.pt`。epoch train groups 完成后，在启动 dev 前还会单独检查 `estimated_dev + final_reserve`；若预算不足，停在 `(epoch, groups, phase=dev_pending)`，不会先进入不可中断 dev 再超预算。若首轮 profile 本身 OOM，明确失败并要求所有 rank 使用 `--force-checkpointing on` 重跑，不把单 rank OOM 当成可自动继续的状态。

## DDP、EMA 与恢复合同

每卡 1 scene、4 micro-step 累积，nominal global scene batch=8。DDP 会平均 rank 梯度，因此每个 rank 使用 `world_size * local_numerator / global_denominator`，不再额外除 accumulation steps。空 source scene 仍走 parameter-connected zero，保持 collective 节奏。

前 20% progress 使用 all sources；之后每 scene 50% all / 50% fixed-seed uniform Q16。随机 budget 不依赖 MSP/GT。镜像只改变 F 局部表示，raw source/MSP/ego/GT 不变。

EMA 在 5% LR warmup 后复制 raw，之后只在 successful optimizer step 更新。`latest.pt` 保存完整 optimizer/EMA/RNG/provenance。epoch 最后一组完成后先写 `phase=dev_pending`；只有 dev evaluation 与 best selection 完成后才原子刷新为 `phase=epoch_complete`。中途 wall-clock budget stop 时，`last.pt` 不再把 `next_group` 重置为 0，而是保留与 `latest.pt` 相同的真实 `(epoch,next_group,phase,global_step)` cursor；从二者恢复必须产生相同剩余样本顺序、global_step 与最终参数。

## 评估

部署指标始终来自 hard compositor。已有 Overall / Moving-mIoU v2 不修改；`stationary_movable` 与 motion groups 为附加 target-only 诊断。未匹配 moving instance 被拆成 `ambiguous-causal-source` 与真正 `no-causal-source`，避免把 association 不确定误解释成 causal representation 缺失。

Q16 没有选中 source 时，soft diagnostic 退化为相同的 KTA 结果，并仍计入同一窗口集合；输出显式记录 `hard_windows` 与 `soft_main_windows`。完整 dev 的 Q16 vs KTA 使用 scene bootstrap 2000 次，`mean_pp/ci95_pp` 使用百分点单位，例如 50%→60% 为 `+10 pp`。

## Preflight 与回归测试

普通依赖轻回归：

```bash
PYTHONPATH="$PWD:$PWD/upstream_occfm" pytest -q tests/real_motion/test_motion_transport_v1_*.py
```

PyTorch 2.6 CUDA 正式验收必须在真实 CUDA runner 上显式开启 fail-fast，不能把 CPU skip 当作通过：

```bash
# 环境必须实际安装 PyTorch 2.6.x CUDA build，例如官方 cu124 wheel。
export MT_V1_REQUIRE_CUDA=1
PYTHONPATH="$PWD:$PWD/upstream_occfm" \
pytest -q tests/real_motion/test_motion_transport_v1_cuda_acceptance.py::test_pytorch26_cuda_complete_acceptance
```

该 CUDA gate 会先要求 `torch.cuda.is_available()==True`，正式模式还要求 `torch.__version__` 以 `2.6.0` 开头；随后在 CUDA 上执行 soft probability 范围/归一化、zero-point CE autograd vs central finite difference、query-chunk probability/gradient 等价、hard zero-selected-delta KTA exact identity，以及完整 8-direction + ±pure-yaw + actual-λ-schedule hard reachability。没有 CUDA 时普通 CI 显式 `skip`；设置 `MT_V1_REQUIRE_CUDA=1` 时则直接失败。

preflight 不再用自报 source 数作为稀疏执行证据：它对 STPN 注册真实 forward hook，要求 Q0 不调用 heavy network，Q16 的实际 source forward 数与 selected sources 一致。CE 与 motion 分别报告 output/head 梯度；联合梯度还必须保持 motion-directed recovery。

回归覆盖包括：

- Strong-W2Det zero-delta exact identity；empty/no-dynamic/small/unmatched/conflict source。
- F/world/ego geometry roundtrip、STPN chunk/empty/mirror/checkpoint gradient。
- MSP↔source explicit overlap mapping、random budget 与 future-GT isolation。
- soft probability、zero/nonzero finite difference、局部 U vs full-grid reference、4 m 大位移与 20 m 完全出界旧 support 清除。
- hard `selected + delta=0` 与 KTA 逐体素 exact identity。
- 原验收 `dx=0.35 m, yaw=0.08 rad` 的 48-error synthetic case、完整 8-direction reachability matrix、±0.8 rad pure-yaw，以及 actual λ schedule，均要求 hard error 严格下降，不能只看 soft loss。
- future `mask_lidar` audit-only supervision regression。
- formal `engine.calibrate_lambda()` 的 8-probe/head/output gradient 路径。
- profile 故意慢 data preparation 必须计入 wall time；epoch dev 前预算 guard 必须阻止预算不足时进入 dev。
- mid-epoch budget-stop 的 `latest.pt` / `last.pt` 必须保留同一真实 cursor，分别恢复后剩余样本顺序/global_step/最终参数一致。
- `10→12→11` best-selection 的 dev 前/后 resume 回归。
- bootstrap `+10 pp` 单位与 Q16 empty-source soft sample-set 一致性。

## 仍需目标服务器完成的训练准入

依赖轻 CI 不能代替真实 nuScenes/Occ3D、冻结 MSP checkpoint、NCCL/BF16 和 2×L40S。正式长训练前仍必须在目标服务器完成：

1. PyTorch 2.6 CUDA F1 strict gate（上面的 `MT_V1_REQUIRE_CUDA=1` 命令）；
2. real-data preflight 与 Strong-W2Det identity/provenance scan；
3. true 2-GPU one-step 等价检查，覆盖不同 source 数、不同 motion-valid 数、empty-source rank 与 tail accumulation；
4. 2×L40S 50+200 formal profile、显存/checkpointing 和完整 dev timing；
5. profile 生成 locked `lambda_reference`、`epochs_locked` 与 wall-clock reserve 后，才允许启动 formal training。

在这些服务器检查完成之前，代码回归通过只代表 **实现准入条件已修复**，不代表模型已经证明超过 KTA，也不声称真实 Moving-mIoU 改善。

## 工程适配/偏差

1. 不修改旧 `MSPCandidate` dataclass；新 adapter 重建原候选与 19-d feature，同时保存 `comp.voxel_indices`，Strong source 与 MSP candidate 通过同类 t0 3D voxel overlap 显式映射。
2. 使用独立 config loader，因为旧 `runtime_config.py` 强制 OccFM/VAE/WM 合同，与本路线冲突。
3. `latest.pt` 是完整可恢复 checkpoint，不是裁剪 optimizer/EMA/RNG 的轻量版本；优先满足 SPEC-2 resume 完整性。
4. 原 SPEC-2 对 soft `U` 的文字没有明确写出旧 KTA support，并已在本实现合同中澄清为 predicted support 与 old KTA support 的并集。
5. soft renderer 的 quarter-voxel anti-alias 仅为训练 surrogate 的数值稳定化；hard compositor 与 headline metric 路径不改变。
