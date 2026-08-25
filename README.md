# SWFM：基于真实运动分解的稀疏 Occupancy World Model

> 基于官方 **OccFM (NeurIPS 2025)** 的研究实现。核心原则：**transport what is physically deterministic; generate only what truly moves**。

## 0. 仓库结构

官方 OccFM 固定为 submodule：

```text
upstream_occfm -> Orbis36/OccFM-NeurIPS2025
                  commit 64959840a9a4cb54d5b0f6cd4bc6779bb242a853
```

我们的代码放在 `real_motion/`、`tools/real_motion/`、`tests/`，不直接魔改官方源码。

首次使用：

```bash
git clone --recurse-submodules https://github.com/ttt05211/swfm.git
cd swfm
git submodule update --init --recursive
```

已有仓库：

```bash
git pull
git submodule update --init --recursive
```

## 1. 方法定义

历史 occupancy 先做 ego compensation，再按真实历史运动划分：

- `confident_static`
- `moving`
- `uncertain`

处理方式：

1. `confident_static`：不用生成模型，直接通过 benchmark 允许的 future ego `SE(3)` transport；
2. `moving + uncertain`：由 causal KTA 外推；
3. KTA **不是最终预测器**，只提供 future motion prior 和 sparse computation support；
4. VAE 完全冻结；训练前缓存 moving/static/KTA/target latent；
5. Sparse WM 直接预测 future moving latent，**不预测 KTA residual**；
6. 最终使用 Static-Protected Motion Composition，`M_gen` 只是 compute mask，不是 overwrite mask。

### 为什么 v1 是 motion-window sparse backend

官方 OccFM transition 是 `Conv3d + temporal attention + U-Net down/up + DiT blocks`，latent spatial size 为 `50x50`。直接改成任意长度 packed tokens 会丢掉大量 checkpoint 可复用结构。

因此 v1 在 50x50 latent grid 上选少量固定窗口（默认 20x20），只对这些窗口运行 transition model，再 scatter 回完整 latent canvas。

## 2. 环境

```bash
conda create -n swfm python=3.10 -y
conda activate swfm
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install nuscenes_devkit matplotlib==3.10.3 einops einops_exts \
  pyyaml easydict wandb rich pytest
```

检查上游：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python -c "from forecast.models.worldmodels.occfm import OccFM; print('OccFM import OK')"
```

## 3. 正式训练前必须先做 P0

### P0-A：Frozen True-Motion Decomposition（第一个做）

同一个 frozen OccFM checkpoint 分别跑：

```text
Full
True-static-only
True-moving-only
```

重点比较：

- Full vs true-static-only 的 true-static IoU；
- Full vs true-moving-only 的 Moving-mIoU v2；
- mIoU / Dynamic-mIoU 作为辅助。

如果 true-moving-only 明显崩，先停，不直接训练 sparse moving-only WM。

### P0-B：Sparsity + KTA Tube Coverage

输入样本：

```python
{"kta_support": Tensor[F,H,W], "gt_moving_support": Tensor[F,H,W]}
```

运行：

```bash
PYTHONPATH=$PWD python tools/real_motion/p0_support_stats.py \
  --input data/p0_support.pt --radii 0,1,2,3,4,5
```

核心看 `GT Moving Coverage vs Active Token Ratio`。

### P0-C：Causal Blind-Spot Audit

按 1s/2s/3s 统计：

- historical stationary -> future moving；
- future moving 中 historical side 没有 causal support 的 innovation / entering-FOV。

比例高时扩大 `uncertain`，不要把所有 stationary hard-freeze。

### P0-D：SE(3)-Static + GT-Moving Oracle

```text
Oracle = SE3(true-static) + GT(true-moving)
```

报告 mIoU、Moving-mIoU v2 和 1s/2s/3s static accuracy。Oracle 没有足够 headroom 时不进入 full training。

P0-E/P0-F 可顺手做 frozen-VAE true-moving reconstruction 和原 OccFM runtime breakdown。

## 4. Frozen Moving-mIoU v2

冻结 contract：

```text
protocol: interval_displacement_v2
speed threshold: 0.5 m/s
box margin: 0.5 m
report horizons: 1s / 2s / 3s
```

定义：

1. GT instance token 匹配 `t0` 与目标 horizon；
2. 若世界坐标 XY interval speed >= 0.5 m/s，则为 Moving instance；
3. future ego grid 中：

```text
Support_h = GT Box(t0 -> h) UNION GT Box(th -> h)
```

两个 box 均扩 0.5 m；
4. 仅在 support 内统计动态 semantic classes，不统计 free/static；
5. **先分别计算 mIoU@1s、mIoU@2s、mIoU@3s，再对三个 mIoU 做算术平均。禁止把三个 horizon 的 voxel/support 合并后做 micro mIoU。**

双 support 同时惩罚 trailing ghost 和 missed arrival。future GT 只用于 post-hoc metric，模型/KTA/WM inference 完全看不到。

核心实现：`real_motion/metrics/moving_miou_v2.py`。

## 5. Frozen VAE latent cache

每个训练 sample：

```python
moving_history_latent  # [H,16,50,50]
future_moving_latent   # [F,16,50,50]
static_future_latent   # [F,16,50,50]
kta_future_latent      # [F,16,50,50]
generation_support     # [F,50,50] required future support
planning_support       # [T,50,50] optional context support
```

训练不重复运行 VAE。先检查：

```bash
PYTHONPATH=$PWD python tools/real_motion/validate_cache.py \
  --cache data/real_motion_train.pt
```

## 6. Window planner：最终 contract

默认：

```yaml
WINDOW_HW: [20,20]
MAX_WINDOWS: 8
```

必须区分：

- `generation_support`：**required future support**，决定哪些窗口必须打开，同时作为 future flow-loss mask；
- `planning_support`：historical moving + future KTA tube 等上下文，只做候选窗口的 context tie-break，**不能创建 history-only window**。

原因：当前不同 window 之间没有跨窗口 attention。单独开的 history-only window 不能给另一个 future window 提供历史信息，只会浪费 `MAX_WINDOWS`。

trainer/sampler 默认要求 future `window_coverage >= 95%`，否则 hard fail；只有诊断时允许 `--allow-low-coverage`。

> v1 仍有需要数据验证的边界：如果高速目标的历史位置与 future target 间跨度超过单个 window，context tie-break 也不能跨窗口传信息。P0/tiny 阶段必须检查 history-context coverage；不足时先增大 window 或调整 backend，不要直接 full train。

## 7. Spatially Aligned Conditioning

不用 global cross-attention：

```text
prior = concat(static_latent, kta_latent)  # 32 channels
              |
        zero-init Linear
              |
       token-wise AdaLN
              |
   spatial / temporal DiT blocks
```

`prior_proj` zero-init。每个 sparse window 使用其在原 50x50 grid 中的绝对位置。

### Position convention

官方 OccFM 对 2500 spatial tokens 实际使用 `grid_size=int(sqrt(N))+1=51` 的 sin-cos grid，再截断前 2500 token。当前实现严格复现这一 token 顺序，再按 `window_origin` 裁剪，避免无意更换 pretrained positional convention。

## 8. 官方 checkpoint 初始化

使用：

```text
real_motion/checkpoint.py::load_shape_safe
```

只加载 key 和 shape 均匹配的 tensor，并打印复用/跳过统计。第一次服务器 tiny-run 必须检查 reuse report；除预期的新 prior/position 项外若大量 skip，立即停止训练排查。

## 9. Tiny-set overfit

先做 64/128 windows：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/train_sparse.py \
  --cache data/real_motion_tiny.pt \
  --upstream-ckpt logs/occfm/2s_3s_nusc_hist_traj/ckpt/epoch=000199.ckpt \
  --output logs/real_motion/tiny_overfit.pt \
  --window 20 --max-windows 8 --batch-size 2 \
  --steps 2000 --lr 2e-5 --amp
```

先确认 masked flow loss 能下降、train Moving-mIoU v2 能明显提高、window coverage 足够。tiny-set 都学不上去，不进入 full training。

第一版 loss 只有 masked CFM / flow MSE，不加 ABE、Router、repair、preservation、confidence loss。

## 10. 稀疏采样

`empty_latent.pt` 必须来自同一个 frozen VAE 的 `E(empty occupancy)`，不能拿数值 0 代替。

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/sample_sparse_latent.py \
  --cache data/real_motion_val.pt \
  --ckpt logs/real_motion/model.pt \
  --empty-latent data/empty_latent.pt \
  --output outputs/future_moving_latent.pt \
  --window 20 --max-windows 8
```

重叠 windows 在 latent 空间平均 scatter，再走官方 frozen decoder。

## 11. Static-Protected Motion Composition

`M_gen` 是 computation mask，不是 overwrite mask。

最终：

1. `SE3-static` 为 base；
2. `confident_static` 永远保护；
3. WM 只在非 confident-static 区域写动态 semantic prediction；
4. dilation support 内的 empty/error 不允许擦掉 road/building。

实现：`real_motion/composition.py`。

## 12. 第一版不加 confidence gate

训练后先统计：

- voxel-level harm/repair；
- instance / motion-tube harm/repair；
- oracle KTA-vs-WM selector headroom。

只有 oracle selector headroom 很大才研究 gate。优先级：

```text
single-pass lightweight gate > multi-sampling > CFG
```

## 13. 困难机动分析

Moving-mIoU v2 主定义不变，只做 subset analysis：

- Uniform / Easy
- Accel / Decel
- Turning
- Turning + Speed Change
- KTA-Easy / Medium / Hard

阈值只在 train/calibration side 冻结。

## 14. 效率报告

至少报告：

- mIoU
- Moving-mIoU v2
- Active Token / Window Ratio
- GFLOPs
- end-to-end latency / FPS
- 可选 peak GPU memory

论文在线 FPS 必须包含 real-motion/KTA preprocessing、condition VAE encoding、window planning/crop、NFE sparse WM、scatter、decoder 和 composition；不能拿 cache-only 时间冒充端到端 FPS。

## 15. 单元测试

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD pytest -q tests
```

当前 merge-review：`python -m py_compile` PASS，`pytest` **8 passed**。覆盖 motion tube、coverage、crop/scatter、future-first window planning、truncation、static protection、0.5m/s metric 边界、dual-box support 和 horizon-first aggregation。

## 16. 开发纪律

1. 第一个实验永远先做 P0-A；
2. future GT 不得进入 motion detector / KTA / M_gen；
3. `generation_support != overwrite mask`；
4. KTA 是 prior，不是最终答案；
5. 第一版不做 residual repair；
6. 第一版不加 Router/CFG/gate；
7. tiny-set 不能 overfit 时不 full train；
8. 效率同时看 FLOPs 与真实 latency；
9. Moving-mIoU v2 contract 不随实验结果修改；
10. 失败先定位 support / VAE / WM / composition，再改模型。

## 17. 不提交 GitHub 的数据

- nuScenes
- OccFM VAE / world-model checkpoints
- KTA cache
- real-motion latent cache

## 18. 上游致谢

本项目基于 `Orbis36/OccFM-NeurIPS2025`：*Towards foundational LiDAR world models with efficient latent flow matching*, NeurIPS 2025。请保留官方论文引用。

更完整的合并前代码审查与服务器验收项见 `IMPLEMENTATION_REVIEW_CN.md`。
