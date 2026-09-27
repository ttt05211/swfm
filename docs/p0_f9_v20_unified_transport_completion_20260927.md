# V20 统一运输补全实现与运行交接（2026-09-27）

协议：`p0_f9_v20_unified_transport_completion_v1`

本文只记录已实现的代码合同与可复现实验命令。代码或测试存在不等于真实数据实验已经完成；在得到 smoke/screen 日志以前，不报告方法收益。

## 1. 已实现边界

- V18 仍是唯一当前源运输路径。新增纯 `decode_transport_queries` 接口，V20 每个窗口只编码一次 V18 source/history token。
- 六帧历史语义、observed 和 observed-free 在固定 t0 canonical 3D 网格编码，unknown 与 observed-free 不混合。
- source adapter 融合 V18 query/history token、当前位置和 KTA 位置的场景采样、全局场景、位置及时间；最后一层严格零初始化。
- warmup 显式 bypass adapter，`Qshared = Q0`；joint 才启用 `Qshared = Q0 + delta`。
- source token 以当前 transport 预测中心（KTA + current residual）做 trilinear normalized accumulation scatter；scatter 坐标 detach，token 梯度保留。物理 OOB 点被计数且完全不写入。
- transport condition 是对齐到 t0 canonical coarse grid 的空间 19 通道场（18 类比例 + coverage），不再使用整帧全局直方图。
- completion 是一个跨 horizon 共享的 18 类头。Static、Dormant、Birth 只作为评测诊断统计，不参与推理路由。
- completion support 固定为 `geometry_query_valid & (current_transport == 17)`；最终合成只允许在该 support 内将 free 改成非 free。
- 训练 GT 只进入 tile 抽样与 CE 监督，不进入模型 forward、运输条件或 runtime query。
- 每 horizon 默认 16 个有放回 tile draw：8 个 uniform-support 和 8 个 positive-containing；无 positive 时全部 uniform，无 support 时不产生 completion loss。
- core tile 为 `32x32x16`，两层 `3x3x3` completion trunk 使用 halo=2，只有 core 写回或参与监督。
- 原始 V18 Clean-E14 loss 保持为 `L_trans + L_exist + 19*L_yaw + 0.25*L_shape_SE2`；completion 使用单一 18-way CE。
- optimizer 参数组只创建一次。warmup 的 V18 LR 为 0 且 `eval/requires_grad=False`；joint 设置 V18 LR 为 `2e-5`，新模块 LR 为 `2e-4`。
- checkpoint 保存 model、optimizer、scheduler、scaler、attempted/successful update、phase、全局 RNG、tile RNG、manifest/checkpoint/grid hash、配置和 Git SHA。

## 2. 入口

- 训练：`tools/real_motion/train_p0_f9_v20_unified.py`
- 三变体评测：`tools/real_motion/eval_p0_f9_v20_unified.py`
- 核心模型：`real_motion/v20_unified_model.py`
- 数据/抽样合同：`real_motion/v20_unified_data.py`
- 损失：`real_motion/v20_unified_loss.py`
- runtime/合成：`real_motion/v20_unified_runtime.py`
- checkpoint/phase：`real_motion/v20_unified_training.py`

评测始终同时报告：

1. `frozen_v18_reference`
2. `current_transport_only`
3. `transport_plus_completion`

## 3. 真实数据 smoke

以下变量必须指向已经审计且互相配对的 Clean-E14/V18 cache、Stage-1 v2 history cache 和 nuScenes 数据。Stage-1 index 中的 frozen coarse lattice 是本实现唯一接受的运行网格。

```bash
PY=/path/to/conda/env/bin/python
RUNTIME_CONFIG=configs/real_motion_occfm.yaml
BASE_CKPT=/path/to/clean_e14/epoch_0014.pt
TRAIN_V18=/path/to/train_v18_se2_cache.pt
DEV_V18=/path/to/dev_v18_se2_cache.pt
TRAIN_STAGE1=/path/to/v20_stage1_train
DEV_STAGE1=/path/to/v20_stage1_dev
DATAROOT=/path/to/nuscenes
TRAIN_INFO=/path/to/nuscenes_infos_train.pkl
DEV_INFO=/path/to/nuscenes_infos_val.pkl

CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/real_motion/train_p0_f9_v20_unified.py \
  --config "$RUNTIME_CONFIG" \
  --train-cache "$TRAIN_V18" \
  --stage1-cache "$TRAIN_STAGE1" \
  --dev-cache "$DEV_V18" \
  --dev-stage1-cache "$DEV_STAGE1" \
  --base-checkpoint "$BASE_CKPT" \
  --dataroot "$DATAROOT" \
  --info-pkl "$TRAIN_INFO" \
  --dev-info-pkl "$DEV_INFO" \
  --out-dir outputs/v20_unified_smoke \
  --smoke
```

`--smoke` 固定执行 1 次 warmup update + 1 次 joint update，并在两个 update 后各跑 1 个 full-support dev 窗口；因此真实覆盖 V18 冻结与解冻两种状态。验收日志至少应包含非空 completion voxel count、有限 loss/grad norm、OOB 统计、三变体指标和 checkpoint。

## 4. 默认训练与精确续训

不传覆盖参数时采用规范默认值：seed `20260927`、batch size `1`、gradient accumulation `4`、warmup `128` successful updates、最大 `1024` successful updates、每 `128` updates 监控 `128` 个 dev 窗口。

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/real_motion/train_p0_f9_v20_unified.py \
  --config "$RUNTIME_CONFIG" \
  --train-cache "$TRAIN_V18" \
  --stage1-cache "$TRAIN_STAGE1" \
  --dev-cache "$DEV_V18" \
  --dev-stage1-cache "$DEV_STAGE1" \
  --base-checkpoint "$BASE_CKPT" \
  --dataroot "$DATAROOT" \
  --info-pkl "$TRAIN_INFO" \
  --dev-info-pkl "$DEV_INFO" \
  --out-dir outputs/v20_unified_main
```

续训不重建 optimizer：

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/real_motion/train_p0_f9_v20_unified.py \
  ...同一组数据与基础 checkpoint 参数... \
  --out-dir outputs/v20_unified_main \
  --resume outputs/v20_unified_main/update_0128.pt
```

base checkpoint hash 不一致时入口会拒绝运行。

## 5. compact cache、screen、dev512 与正式评测

Unified 路径不使用旧 Stage-1 的 native `static_supervision`，也不使用 GT dynamic identity/trajectory metadata。compact 构建会跳过这两类监督的计算与存储，只保留 history evidence、future pose 和必要索引。新建 cache 时建议：

```bash
python tools/real_motion/build_p0_f9_v20_history_cache.py ... \
  --unified-compact --shard-size 128
```

已有 Stage-1 v2 cache 不需要重算几何，可直接瘦身：

```bash
python tools/real_motion/compact_p0_f9_v20_stage1_for_unified.py \
  --input-dir "$TRAIN_STAGE1_OLD" \
  --output-dir "$TRAIN_STAGE1"
```

确认 compact cache 后再删除旧 cache，避免长期保留两份。新 cache 的 shard metadata 带 row keys，unified loader 使用小型 shard LRU，不再把全部 Stage-1 row 常驻 RAM。默认 spatial transport mapping 使用更大的 64×64×32 chunk 降低小 kernel 数；completion tile 仍保持 32×32×16，不改变训练口径。训练默认只保留最近 3 个 update checkpoint，可用 `--keep-checkpoints 0` 关闭轮转。训练内的固定 dev monitor 第一次计算 frozen V18 reference 后只复用其 raw intersection/union counts，后续 checkpoint 不再重复跑冻结 V18 forward/render；这不改变任何指标口径，也不生成额外的大型预测 cache。

真正的 train-screen1024 是固定 1024 个训练窗口做 1024 successful updates（grad_accum=4，约四遍暴露）：

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/real_motion/train_p0_f9_v20_unified.py \
  ...同一组数据与基础 checkpoint 参数... \
  --out-dir outputs/v20_unified_screen1024 \
  --screen1024
```

1024-update checkpoint 的 dev 评测：

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/real_motion/eval_p0_f9_v20_unified.py \
  --config "$RUNTIME_CONFIG" \
  --val-cache "$DEV_V18" \
  --stage1-cache "$DEV_STAGE1" \
  --base-checkpoint "$BASE_CKPT" \
  --checkpoint outputs/v20_unified_main/update_1024.pt \
  --dataroot "$DATAROOT" \
  --info-pkl "$DEV_INFO" \
  --max-windows 1024 \
  --output outputs/v20_unified_main/screen1024.json
```

将 `--max-windows` 改为 `128` 或 `512` 可运行对应 dev 检查。只有 loss、semantic mIoU 和 Moving 指标显示稳定正趋势后，才去掉 `--max-windows` 运行完整正式集合。

主方案有效后再运行 completion source-latent ablation；该开关不改变 current transport：

```bash
... eval_p0_f9_v20_unified.py 的同一命令 ... \
  --ablate-completion-source-latents \
  --output outputs/v20_unified_main/screen1024_no_source_latents.json
```

## 6. 本地验证状态

本提交在 Windows Anaconda base（PyTorch `2.3.1+cu118`）完成了语法、CLI import 和相关单元/回归测试。真实 nuScenes cache、Clean-E14 checkpoint 与数据根目录不在当前工作区，因此未伪造 smoke、screen 或正式实验结果。
