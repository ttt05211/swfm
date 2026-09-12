# motion_transport_v1 — MT-V1-SPEC-2 运行与验收说明

本分支实现独立的 occupancy-space future transport 路线，不加载 VAE、OccFM 或 FM 权重。它复用 `real_motion/strong_w2det.py` 作为 Strong-W2Det/KTA 几何基准，冻结现有 MSP activation checkpoint，仅对路由选中的 Strong-W2Det source 运行 STPN。

## 硬件与时间约定

- **调试、缓存、preflight、普通评估默认单 GPU**。没有必要为了这些步骤占两张卡。
- 正式训练支持 **1 GPU 或 2 GPU**。如果单卡显存和吞吐足够，直接单卡训练；只有决定使用双卡正式训练时，才需要跑 true 2-GPU DDP acceptance。
- formal profile 必须和最终训练的 GPU 数一致：单卡训练就用单卡 formal profile；双卡训练前重新用双卡 formal profile。这样显存、吞吐、checkpointing 和 epoch 时间估计才有效。
- `training.gpu_type` 只是运行记录/建议，不再作为固定 L40S 型号门槛。
- `training.max_hours` 是**可选** wall-clock 预算。默认 `null` 表示不启用按小时强制停止；显式设置任意正数（如 4、6、10）后，才启用 F2/F3 那套安全停机与 reserve 逻辑。不存在“必须 4 小时”的方法合同。

## 推荐运行顺序

```bash
CFG=configs/real_motion/motion_transport_v1.yaml

# 1) manifest / cache / debug：单卡即可
python tools/real_motion/build_motion_transport_v1_manifest.py \
  --config "$CFG" \
  --output data/motion_transport_v1/manifest.json

# 2) real-data preflight：单卡 CUDA
CUDA_VISIBLE_DEVICES=0 python tools/real_motion/preflight_motion_transport_v1.py \
  --config "$CFG" \
  --output outputs/motion_transport_v1/preflight.json \
  --samples 16 \
  --device cuda \
  --require-cuda \
  --identity-scan all
```

如果最终准备**单卡正式训练**：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/real_motion/profile_motion_transport_v1.py \
  --config "$CFG" \
  --output-dir outputs/motion_transport_v1/profile_1gpu \
  --formal-server-gate

LOCKED=outputs/motion_transport_v1/profile_1gpu/resolved_profile_config.yaml
OUT=outputs/motion_transport_v1/run_seed3407

CUDA_VISIBLE_DEVICES=0 python tools/real_motion/train_motion_transport_v1.py \
  --config "$LOCKED" \
  --output-dir "$OUT"
```

如果最终准备**双卡正式训练**，先额外跑 DDP acceptance，再用同样两张卡 profile：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  tools/real_motion/accept_motion_transport_v1_ddp.py \
  --config "$CFG" \
  --output outputs/motion_transport_v1/ddp_acceptance.json \
  --scan-windows 256

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  tools/real_motion/profile_motion_transport_v1.py \
  --config "$CFG" \
  --output-dir outputs/motion_transport_v1/profile_2gpu \
  --formal-server-gate

LOCKED=outputs/motion_transport_v1/profile_2gpu/resolved_profile_config.yaml
OUT=outputs/motion_transport_v1/run_seed3407

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  tools/real_motion/train_motion_transport_v1.py \
  --config "$LOCKED" \
  --output-dir "$OUT"
```

恢复训练时必须使用与原 profile/checkpoint 相同的 `WORLD_SIZE`：

```bash
# 单卡示例
CUDA_VISIBLE_DEVICES=0 python tools/real_motion/train_motion_transport_v1.py \
  --config "$OUT/resolved_train_config.yaml" \
  --output-dir "$OUT" \
  --resume "$OUT/latest.pt"
```

## 方法结构

```text
history occupancy (6 frames)
        ↓
causal Strong-W2Det / KTA source decomposition
        ↓
frozen MSP candidates + explicit same-class t0 voxel-overlap mapping
        ↓
budget routing (route before heavy network)
        ↓
only selected sources → KTA-backtraced 64×64×16 local history crops
        ↓
STPN
        ↓
6-horizon cumulative residual (dx, dy, dyaw) relative to KTA
        ↓
transport raw 3D source shapes
        ↓
hard compositor for deployment/eval
soft anti-aliased compositor only for training gradients
        ↓
full 18-class future occupancy
```

不使用 VAE、FM 或 OccFM pretrained world model。未来 semantic、future box、future instance ID 不进入 causal inference；future ego pose 仅按数据协议使用。

## Soft renderer 与监督域

正式部署与 checkpoint selection 始终使用 hard forward-floor compositor。soft renderer 只提供训练梯度，不参与 hard 指标。

PyTorch 2.6 在 trilinear lattice knot 上的单边导数曾导致 zero-initialized motion head 出现异常 CE 梯度并污染 Adam 二阶矩。当前训练 soft occupancy 使用 forward/backward 一致的 quarter-voxel anti-alias surrogate：对 XY footprint 的 `(+/-0.25, +/-0.25)` 四个位置做 trilinear probability quadrature，四个 probability 的平均值同时作为 forward probability 与 autograd 对象。它不是 straight-through gradient replacement。

`compose_hard()` 完全不使用该 surrogate，因此 `selected + delta=0` 与 Strong/KTA hard identity 保持逐体素一致。

soft 查询/替换域 `U` 至少包含 predicted support AABB 与 old KTA support 的并集，因此 source 大位移或移出网格后不会在旧位置留下 ghost。代码保留 full-grid soft reference 作为回归对照。

历史 `mask_lidar` 仅用于 causal motion decomposition 的 observation。future CE supervision validity 使用合法 semantic label `0..17`；future `mask_lidar` 仅作为 observed/unobserved audit，不决定 CE 是否监督。冻结的 Overall / Moving-mIoU v2 口径不修改。

## Loss 与 gradient calibration

训练目标保持简单：

```text
L = L_occ + lambda(t) * L_motion
```

- `L_occ`：完整有效 future occupancy categorical CE。
- `L_motion`：GT rigid source-point XY SmoothL1，仅作为训练辅助监督。
- 不启用 FM / overlap / temporal smoothness / volume / moving extra weight。

formal profile 固定调用与正式代码相同的 `engine.calibrate_lambda()`，用 8 个 GT-independent `(±dx, ±dy, ±yaw)` probe envelope 检查 occupancy 与 motion 对 final head 和输出坐标的梯度，得到 `lambda_reference`。正式 schedule 仍为前 10% `1.0×lambda_ref`，10%–30% 衰减到 `0.25×lambda_ref`，之后保持 `0.25×lambda_ref`。

F1 已在 PyTorch 2.6 CPU 和真实 PyTorch 2.6 CUDA strict gate 中验证：完整 8-direction、±pure-yaw、actual lambda schedule、soft probability/gradient finite-difference consistency、query chunk consistency、hard zero-delta KTA identity 均保留原严格断言。

## Formal profile 与可选 wall-clock 预算

formal profile 仍使用 50 warmup + 200 measured microsteps，计时从 dataset 取样前开始，包含：disk/data loading、Strong-W2Det/source decomposition、MSP candidate/mapping、future target preparation、crop、STPN BF16 forward、FP32 renderer、backward/optimizer、peak memory、full checkpoint save 与 full dev timing。

`--formal-server-gate` 现在表示：**对本次实际 launch 的 1 或 2 张 CUDA/BF16 GPU 做正式 full-dev profile**。它不再要求固定 2×L40S。profile 会把实际 `profile_world_size` 与 GPU 名称写入 locked config；正式训练必须使用相同 `WORLD_SIZE`。

epoch lock 的来源按以下优先级：

1. 如果显式给 `training.fixed_epoch_cap`，直接使用它；
2. 否则如果显式给 `training.max_hours`，根据实测 train/dev/checkpoint 时间估算可完成 epoch 数；
3. 否则使用 `training.initial_epoch_estimate` 作为初始正式训练 epoch 数。

当 `max_hours: null` 时，训练完全不做 wall-clock stop；F2/F3 的 `dev_pending / budget_stop / latest.pt / last.pt` 安全恢复代码仍保留，并在用户显式设置时间预算时启用。

## 单卡与双卡合同

单卡训练直接使用 raw module，不经过 DDP collective；其它 loss、EMA、checkpoint、hard eval 合同完全相同。

双卡训练时 DDP 使用 `world_size * local_numerator / global_denominator`。因为 PyTorch DDP 会平均 rank gradient，所以不能额外按 accumulation steps 再除一次。empty-source rank 使用 parameter-connected zero 保持 collective 节奏。

`accept_motion_transport_v1_ddp.py` 是**双卡正式训练的条件性准入**，不是所有调试流程的前置条件。只有最终决定用两张卡训练时才跑。它比较 manual globally-summed reference 与真实 DDP optimizer step，覆盖 heterogeneous source/motion count、empty-source rank、tail accumulation 和 activation checkpointing，要求 gradient/model/optimizer state allclose。

## EMA、checkpoint 与恢复

EMA 在 warmup 后启动，之后只在 successful optimizer step 更新。`latest.pt` 保存完整 raw/EMA/optimizer/RNG/provenance。

checkpoint phase：`train / dev_pending / epoch_complete / budget_stop`。

F2/F3 回归仍保留：时间预算启用时，进入 dev 前会检查剩余时间；budget stop 时 `latest.pt` 与 `last.pt` 保留同一真实 `(epoch,next_group,phase,global_step)` cursor，从二者恢复必须产生相同剩余样本顺序、global step 和最终参数。

## 评估

部署指标始终来自 hard compositor。已有 Overall / Moving-mIoU v2 不修改；`stationary_movable` 与 motion groups 只是附加 target-only 诊断。

评估预算：`Q=0/4/8/16/32/all`。Q16 没选中 source 时 soft diagnostic 退化为同一 KTA 结果但仍计入相同窗口集合。完整 dev Q16 vs KTA 使用 scene bootstrap 2000 次；delta 使用 percentage points，例如 50%→60% 为 `+10 pp`。

## 当前训练前准入

已完成：R1–R7 代码级修复与回归、F1 PyTorch 2.6 CPU/CUDA、F2/F3 budget/resume、hard-zero KTA identity synthetic regression、dependency-light CI 与 PyTorch 2.6 CI。

仍需真实服务器完成，但按实际卡数选择：

1. **单卡** real-data preflight + full Strong/KTA identity/provenance scan；
2. 如果最终双卡训练，再跑 **2-GPU DDP acceptance**；如果单卡训练则跳过；
3. 在最终训练准备使用的 **1 或 2 张 GPU** 上跑 formal 50+200 full-dev profile；
4. 检查 profile 生成的 lambda、显存、checkpointing、epoch lock 与可选时间预算；
5. 这些通过后才启动完整训练。

在真实服务器准入完成前，代码回归通过只代表实现已经准备好，不代表模型已经证明超过 KTA，也不声称真实 Moving-mIoU 改善。
