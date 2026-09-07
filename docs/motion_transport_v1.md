# motion_transport_v1 — MT-V1-SPEC-2 运行说明

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

EMA hard 主评估与 Q 曲线：

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

## Formal profile 合同

`profile_motion_transport_v1.py` 固定使用 50 warmup + 200 measured microsteps；先以 8 个 batch 计算 `lambda_ref=median(G_occ/G_mot)`，再测显存、吞吐和完整 dev。若代表性峰值超过 40 GiB，`auto` 重跑 non-reentrant activation checkpointing。只有完整 dev profile 才写入 `loss.lambda_reference` 和 `training.epochs_locked`；formal train 不会重新校准。

## 训练与恢复合同

每卡 1 scene、4 micro-step 累积，nominal global scene batch=8。DDP 对 rank 梯度做平均，因此每个 rank 使用 `world_size*local_numerator/global_denominator`，不再除 accumulation steps。空 source scene 仍走 parameter-connected zero，保持相同 collective 节奏。前 20% progress 全 source；之后每 scene 50% all / 50% fixed-seed uniform Q16。随机 budget 不依赖 MSP/GT。镜像只改变 F 局部表示，raw source/MSP/ego/GT 不变。

EMA 在 5% LR warmup 后复制 raw 开始，随后只按 successful optimizer step 更新。`latest.pt` 每约 0.25 epoch 原子覆盖；它保存完整 optimizer/EMA/RNG/provenance，以保证精确 resume，因此比规格里“lightweight”一词更重，但不改变训练合同。checkpoint 含 config/manifest/MSP hash，resume 要求同一 WORLD_SIZE。

## 评估

部署结果始终是 hard forward-floor compositor。已有 Overall/Moving-mIoU v2 口径不修改；`stationary_movable` 和 observed-moving/dormant-to-moving/stopping/no-causal-source 是附加 target-only 诊断。完整 dev 的 Q16 vs KTA 进行 2000 次 scene bootstrap。训练 CE 的 valid domain 使用 Occ3D `mask_lidar`，因为当前正式数据适配器提供的显式 supervision validity 就是该 mask；headline Overall/Moving 保持仓库冻结指标口径，不据此改写。

## 测试

```bash
PYTHONPATH="$PWD:$PWD/upstream_occfm" pytest -q tests/real_motion/test_motion_transport_v1_*.py
```

依赖轻测试覆盖 Strong-W2Det zero-delta identity、F/world 坐标、future-GT 字段隔离、MSP/source 显式 overlap mapping、随机 budget、STPN chunk/empty/mirror/checkpoint gradient、soft probability/gradient/query chunk/full-scene CE/reachability、EMA、DDP 归一化代数和 checkpoint resume。

真实 2×GPU optimizer-step 对照、L40S BF16/checkpointing、nuScenes provenance 和最终 hard improvement 必须在目标服务器执行 preflight/profile/train/eval；CI 不能替代这些数据与硬件实验，因此在完成服务器运行前不得声称性能改善。

## 工程适配/偏差

1. 不修改旧 `MSPCandidate` dataclass，而由新 adapter 重建原候选和 19-d feature，同时保存 `comp.voxel_indices`；Strong source 与 MSP candidate 用同类 t0 3D voxel overlap 显式映射。
2. 使用独立 config loader，因为旧 `runtime_config.py` 强制 OccFM/VAE/WM 合同，与本路线冲突。
3. `latest.pt` 为可精确恢复的完整 checkpoint，不是删减 optimizer/EMA/RNG 的轻量文件；这是为了优先满足 SPEC-2 resume 完整性。
4. soft inverse-trilinear renderer 只提供训练梯度，与 hard forward-floor 本来就不要求逐点数学等价；模型有效性只由 hard 指标判断。
