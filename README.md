# SWFM：基于真实运动分解的稀疏 Occupancy World Model

> 基于官方 **OccFM (NeurIPS 2025)** 的实验实现。核心原则：**transport what is physically deterministic; generate only what truly moves**。

## 0. 仓库结构与上游版本

官方 OccFM 以 git submodule 固定在：

```text
upstream_occfm -> Orbis36/OccFM-NeurIPS2025
                  commit 64959840a9a4cb54d5b0f6cd4bc6779bb242a853
```

本仓库不直接魔改官方目录。我们的实现都在 `real_motion/` 和 `tools/real_motion/`，这样可以：
1. 明确区分官方 baseline 与我们的代码；
2. 最大化复用官方 checkpoint；
3. 后续升级官方代码时容易做 diff。

首次克隆：

```bash
git clone --recurse-submodules https://github.com/ttt05211/swfm.git
cd swfm
git checkout feat/real-motion-occfm
git submodule update --init --recursive
```

## 1. 方法总览

历史 occupancy 先做 ego compensation，再按**真实历史运动**分为 `confident_static / moving / uncertain`。

- `confident_static`：不进入生成模型，直接用 future ego `SE(3)` 刚体传播；
- `moving + uncertain`：用 causal KTA 外推；
- KTA **不是最终预测器**，只负责 motion prior 和 future motion tube；
- VAE 完全冻结，训练前离线缓存 static/KTA/moving/target latent；
- Sparse WM 直接预测 future moving latent，**不预测 KTA residual**。

### 为什么采用 motion-window sparse backend

官方 OccFM transition backbone 是 `Conv3d + temporal attention + U-Net down/up + DiT block`，而不是任意长度 token Transformer，官方 latent 为 `50x50`。如果直接改成 packed tokens，会丢掉大量卷积/down-up checkpoint。

v1 因此使用：
- 在 `50x50` latent support 上选择少量固定 `20x20` motion windows；
- 只对这些 windows 执行 OccFM transition model；
- 重叠 window 用平均 scatter；
- 外部 cache/support API 保持独立，未来可以替换 packed-token backend。

这是真实减少 WM 计算，不是在 dense 50x50 上简单乘 mask。

## 2. 环境

```bash
conda create -n swfm python=3.10 -y
conda activate swfm
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install nuscenes_devkit matplotlib==3.10.3 einops einops_exts \
  pyyaml easydict wandb rich pytest
```

验证上游：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python -c "from forecast.models.worldmodels.occfm import OccFM; print('OccFM import OK')"
```

## 3. 正式训练前先做 P0；不要直接训练

### P0-A（第一个必须做）：Frozen True-Motion Decomposition

目标：把你之前 semantic split 的实验换成 real-motion split。

分别构造：
1. Full
2. true-static-only
3. true-moving-only

三者全部用**同一个 frozen OccFM checkpoint**。

重点比较：
- Full vs static-only 的 true-static IoU；
- Full vs moving-only 的 `Moving-mIoU v2`；
- mIoU / Dynamic-mIoU 作为辅助。

**Go：** 对应分支指标接近 Full。  
**No-Go：** true-moving-only 明显崩，则不要训练 sparse moving-only WM，优先回到 full-latent masked formulation。

> P0-A 需要复用你已有的 real-motion mask/KTA 数据生成代码；本仓库不会用 future GT 冒充 causal motion mask。

### P0-B（必须）：稀疏率 + KTA Tube Coverage

输入：

```python
{
    "kta_support": Tensor[F,H,W],        # causal
    "gt_moving_support": Tensor[F,H,W],  # 仅统计
}
```

执行：

```bash
PYTHONPATH=$PWD \
python tools/real_motion/p0_support_stats.py \
  --input data/p0_support.pt \
  --radii 0,1,2,3,4,5
```

输出 `radius,coverage,active_ratio`。关键图：`GT Moving Coverage vs Active Token Ratio`。

### P0-C（必须）：Causal Blind-Spot Audit

按 1s/2s/3s 统计：
- 历史静止、future 启动的 instance 比例；
- future moving 中历史没有 causal support 的 innovation / entering-FOV 比例。

GT 只能用于 audit。占比高时扩大 `uncertain`，不要把所有 stationary hard-freeze。

### P0-D（必须）：SE(3)-Static + GT-Moving Oracle

```text
Oracle = SE3(true-static) + GT(true-moving)
```

报告 mIoU、Moving-mIoU v2，以及 1s/2s/3s static branch accuracy。Oracle 本身没有足够 headroom 时先停，不烧 full training。

### P0-E / P0-F（低成本建议）

- P0-E：frozen VAE true-moving reconstruction + empty-latent/scatter sanity check；
- P0-F：profile 原 OccFM 的 VAE encoder / DiT / decoder / other latency。

## 4. Frozen Moving-mIoU v2

主协议冻结：

```text
protocol: interval_displacement_v2
speed threshold: 0.5 m/s
box margin: 0.5 m
report horizons: 1s / 2s / 3s
```

定义：
1. 用 GT instance token 匹配 `t0` 与目标 future 时刻的同一实例；
2. 世界坐标 XY interval speed >= `0.5 m/s` 时定义为 Moving；
3. 在 future ego grid 中 rasterize：

```text
Support_h = GT Box(t0 -> h) UNION GT Box(th -> h)
```

两个 box 各加 `0.5 m` margin；
4. 只在该 support 内统计**动态 semantic classes** 的 IoU，不统计 free/static；
5. 动态类取 mIoU，再平均 1s/2s/3s。

双 support 同时惩罚：
- trailing ghost：旧位置还残留；
- missed arrival：新位置没预测到。

future GT **只用于 post-hoc metric**；模型/KTA/WM 推理完全看不到。实例要求 t0 与目标 future 端点都存在；出生、消失、端点缺失需排除并记录。

实现：`real_motion/metrics/moving_miou_v2.py`

## 5. 生成 real-motion latent cache

cache 协议：`real_motion/cache.py`

每个 sample 必须包含：

```python
moving_history_latent  # [H,16,50,50]
future_moving_latent   # [F,16,50,50]
static_future_latent   # [F,16,50,50]
kta_future_latent      # [F,16,50,50]
generation_support     # [F,50,50] bool，控制未来 active loss
planning_support       # [H+F,50,50] bool，可选但强烈建议
```

可选：`trajectory / confident_static_mask / gt_moving_support / sample_id`。

训练时不重复跑 VAE。先验证 cache：

```bash
PYTHONPATH=$PWD \
python tools/real_motion/validate_cache.py --cache data/real_motion_train.pt
```

你现有 KTA pipeline 需要提供 occupancy-space：
- historical true-moving OCC；
- future SE3 static OCC；
- future KTA OCC；
- causal generation support；
- future GT moving OCC（训练 target）。

然后统一用官方 frozen VAE encoder 编码并缓存。**不要再训练小 encoder。**

## 6. Window planner

默认：

```yaml
WINDOW_HW: [20,20]
MAX_WINDOWS: 8
```

流程：
1. 优先用 `planning_support = historical moving support UNION future KTA tube` 选窗口；旧 cache 没有时才退回 `generation_support`；
2. greedy 选择固定 window，优先覆盖未覆盖 active cells；
3. crop history / GT target / static prior / KTA prior / mask；
4. 只把 valid windows 展平进 batch；
5. 重叠 window scatter 时平均，避免覆盖顺序依赖。

`planning_support` 与 `generation_support` **不能混为一个东西**：前者负责选窗口，后者负责未来计算/flow loss。只看 future tube 选窗口会把远期窗口中的历史运动上下文裁掉。

代码：`real_motion/windows.py`、`real_motion/support.py`。

## 7. 条件注入

不使用 global cross-attention。

static/KTA latent 与 moving target 已经空间对齐，v1 直接：

```text
prior = concat(static_latent, kta_latent)  # 32 channels
              |
        zero-init Linear
              |
       token-wise AdaLN
              |
      每个 spatial/temporal DiT block
```

官方 OccFM DiT 本身使用 AdaLN-Zero。本实现保持相同模块字段名，将 condition 扩展为 `[B,D]` 或 `[B,N,D]`。新增 prior projection 为 zero-init，因此训练第 0 步 prior 的增量为 0。

## 8. checkpoint 初始化

不能简单 `strict=False`。

50x50 改成 20x20 window 后固定 position embedding shape 会变化，PyTorch 对 shape mismatch 即使 `strict=False` 也会报错。

使用：`real_motion/checkpoint.py::load_shape_safe`

它只加载 key 和 shape 都匹配的权重，并打印复用率与跳过原因。位置编码按新 window size 重新生成，这是预期行为。

## 9. Tiny-set overfit：第一场训练

先准备 64/128 windows cache：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/train_sparse.py \
  --cache data/real_motion_tiny.pt \
  --upstream-ckpt logs/occfm/2s_3s_nusc_hist_traj/ckpt/epoch=000199.ckpt \
  --output logs/real_motion/tiny_overfit.pt \
  --window 20 --max-windows 8 \
  --batch-size 2 --steps 2000 --lr 2e-5 --amp
```

检查：
1. masked flow loss 持续下降；
2. train Moving-mIoU v2 明显逼近 oracle；
3. active ratio 与预处理一致；
4. window 边界无明显 artifact。

训练集都学不上去时**不要 full train**。

## 10. 正式训练

顺序：tiny overfit → small held-out → 冻结 window/radius → full training。

第一版 loss 只有 `masked CFM / flow MSE`。不加 ABE、router、repair、preservation、confidence loss。模型直接预测 future moving latent，不预测 KTA residual。

### 10.1 稀疏采样与 moving latent canvas

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/sample_sparse_latent.py \
  --cache data/real_motion_val.pt \
  --ckpt logs/real_motion/model.pt \
  --empty-latent data/empty_latent.pt \
  --output outputs/future_moving_latent.pt \
  --window 20 --max-windows 8
```

`empty_latent.pt` 必须来自同一个 frozen VAE 的 `E(empty occupancy)`，不要用数值 0 冒充 empty latent。脚本对重叠 windows 做 latent averaging。随后用官方 frozen decoder 解码为 moving occupancy，再做 composition。

## 11. Static-Protected Motion Composition

`M_gen` 是 **computation mask，不是 overwrite mask**。

最终：
1. `SE3-static` 是 base；
2. `confident_static` 永远保护；
3. WM 只在非 confident-static 区域写动态 semantic prediction；
4. dilation support 内的 empty prediction 不允许擦掉 road/building。

实现：`real_motion/composition.py`；命令工具：`tools/real_motion/compose_occ.py`。

## 12. 第一版为什么不加 confidence gate

KTA 是 prior，不是可靠 fallback；转弯/加速时本来就需要 WM 推翻 KTA。

训练后先统计：
- support 内 voxel-level harm / repair；
- instance / motion-tube harm / repair；
- oracle KTA-vs-WM selector headroom。

只有 oracle selector headroom 很大才研究 gate。优先级：

```text
single-pass lightweight gate > 多采样 > CFG
```

CFG 最后考虑，因为它增加每个 ODE step 的 forward 次数，会破坏效率主线。

## 13. 困难机动分析

不改 Moving-mIoU v2，只做 subset analysis：
- Uniform / Easy
- Accel / Decel
- Turning
- Turning + Speed Change

建议用 interval quantities：`Δv_h = |v_h-v_0|`、`Δψ_h = |wrap(ψ_h-ψ_0)|`。

再按 post-hoc KTA center/yaw error 分 KTA-Easy / Medium / Hard。阈值只在 train/calibration side 冻结。最关键的证据是提升集中在 Turning / Accel / KTA-Hard，而不是只提高匀速样本。

## 14. 效率报告

至少报告：
- mIoU
- Moving-mIoU v2
- Active Token Ratio
- GFLOPs
- end-to-end latency / FPS
- 可选 peak GPU memory

训练 cache 可以排除 VAE 重复开销；但**论文端到端在线 FPS 必须把推理时一次性的 VAE condition encoding 算进去**，不能拿 cache-only inference 冒充端到端 FPS。

后端微基准：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/profile_backend.py --window 20 --batch 8
```

## 15. 单元测试

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD pytest -q tests
```

当前测试覆盖：motion tube、coverage/active ratio、window crop/scatter、planning/loss support 分离、static protection、Moving-mIoU v2 速度边界、双 box support 对 trailing ghost/missed arrival 的惩罚。

## 16. 开发纪律

1. **第一个实验永远先做 P0-A frozen true-motion decomposition。**
2. future GT 不能进入 motion detector / KTA / M_gen。
3. `M_gen != overwrite mask`。
4. KTA 是 prior，不是最终答案。
5. 第一版不做 residual repair。
6. 第一版不加 Router/CFG/gate。
7. tiny-set 不能 overfit 时绝不 full train。
8. 效率同时看 FLOPs 和真实 latency。
9. Moving-mIoU v2 contract 不随结果修改。
10. 每次失败先定位 support / VAE / WM / composition，再改代码。

## 17. 数据资产

以下大文件不提交 GitHub：nuScenes、OccFM checkpoint、KTA cache、real-motion latent cache。

如果暂时用 GT historical track 做 P0，请明确标记 **P0-only diagnostic**，不能当最终 causal inference。

## 18. 上游致谢

本项目基于 `Orbis36/OccFM-NeurIPS2025`：*Towards foundational LiDAR world models with efficient latent flow matching*, NeurIPS 2025。请保留官方论文引用。
