# SWFM：基于真实运动分解的稀疏 Occupancy World Model

> 基于官方 **OccFM (NeurIPS 2025)** 的研究实现。核心原则：**transport what is physically deterministic; generate only what truly moves**。

## 0. 仓库结构

官方 OccFM 作为 submodule 固定在：

```text
upstream_occfm -> Orbis36/OccFM-NeurIPS2025
                  commit 64959840a9a4cb54d5b0f6cd4bc6779bb242a853
```

我们的实现放在 `real_motion/`、`tools/real_motion/`，不直接魔改官方目录，便于复用 checkpoint、对照 baseline 和以后同步上游。

首次使用：

```bash
git clone --recurse-submodules https://github.com/ttt05211/swfm.git
cd swfm
git checkout feat/real-motion-occfm
git submodule update --init --recursive
```

---

## 1. 方法最终定义

历史 occupancy 先 ego-compensate，再按**真实历史运动**划分：

- `confident_static`
- `moving`
- `uncertain`

处理方式：

1. `confident_static`：不用生成模型，按 benchmark 允许的 future ego pose 做 `SE(3)` transport；
2. `moving + uncertain`：用 causal KTA 外推；
3. KTA **不是最终预测器**，只负责：
   - future motion prior；
   - future motion tube / sparse computation support；
4. VAE 完全冻结；static/KTA/moving/target latent 均可在训练前离线缓存；
5. Sparse WM 直接预测 future moving latent，**不预测 KTA residual**；
6. 最终使用 `Static-Protected Motion Composition`，`M_gen` 只决定哪里计算，不等于哪里覆盖 static。

### 为什么不是 packed-token DiT v1

官方 OccFM transition 不是纯 token Transformer，而是：

```text
Conv3d + temporal attention + U-Net down/up + DiT blocks
```

官方 latent grid 是 `50x50`。直接改成任意长度 token 会让大量 Conv/U-Net checkpoint 无法复用。

因此当前 v1 使用 **motion-window sparse backend**：

- 在 50x50 latent support 上选择少量固定窗口，默认 20x20；
- 只对这些窗口运行 OccFM transition backbone；
- overlapping windows 在 latent 空间平均 scatter；
- support/cache API 与 backend 解耦，以后可替换 packed-token backend。

这是真正减少 world-model 计算，而不是在 dense 50x50 输出上乘一个 mask。

---

## 2. 环境

先按官方 OccFM 环境：

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

---

## 3. 正式训练前：必须先做 P0

### P0-A：Frozen True-Motion Decomposition（第一个做）

把你之前 semantic split 的实验换成 real-motion split：

1. Full history
2. true-static-only history
3. true-moving-only history

全部使用**同一个 frozen OccFM checkpoint**。

重点比较：

- Full vs true-static-only 的 true-static IoU；
- Full vs true-moving-only 的 `Moving-mIoU v2`；
- mIoU / Dynamic-mIoU 作为辅助。

判定：

- 对应分支几乎不掉：继续 moving-only sparse WM；
- true-moving-only 明显崩：先回到 full-latent masked formulation，不直接烧 full training。

> P0-A 应复用你已有的 real-motion mask/KTA 生成代码。本仓库不会用 future GT 冒充 causal mask。

### P0-B：Sparsity + KTA Tube Coverage

样本：

```python
{
    "kta_support": Tensor[F,H,W],        # causal
    "gt_moving_support": Tensor[F,H,W],  # 仅用于统计
}
```

运行：

```bash
PYTHONPATH=$PWD \
python tools/real_motion/p0_support_stats.py \
  --input data/p0_support.pt \
  --radii 0,1,2,3,4,5
```

输出：

```text
radius,coverage,active_ratio
```

核心图：`GT Moving Coverage vs Active Token Ratio`。

### P0-C：Causal Blind-Spot Audit

按 1s/2s/3s 统计：

- historical stationary -> future moving 比例；
- future moving 中 historical side 没有 causal support 的 innovation / entering-FOV 比例。

比例高时扩大 `uncertain`，不要把所有 stationary hard-freeze。

### P0-D：SE(3)-Static + GT-Moving Oracle

```text
Oracle = SE3(true-static) + GT(true-moving)
```

报告：

- mIoU
- Moving-mIoU v2
- 1s / 2s / 3s static branch accuracy

Oracle 没有足够 headroom 时，先改 decomposition/fusion，不进入 full training。

### P0-E / P0-F（低成本建议）

- P0-E：frozen VAE true-moving reconstruction + `E(empty)` scatter sanity；
- P0-F：profile 原 OccFM 的 VAE encoder / transition model / decoder / other latency。

---

## 4. Frozen Moving-mIoU v2

冻结 contract：

```text
protocol: interval_displacement_v2
speed threshold: 0.5 m/s
box margin: 0.5 m
report horizons: 1s / 2s / 3s
```

定义：

1. 用 GT instance token 匹配 `t0` 和目标 future horizon 的同一实例；
2. 世界坐标 XY interval speed：

```text
||c_GT(th,xy)-c_GT(t0,xy)|| / (th-t0) >= 0.5 m/s
```

则该实例为 Moving；
3. 在目标 future ego grid 中：

```text
Support_h = GT Box(t0 -> h) UNION GT Box(th -> h)
```

两个 box 都扩 0.5m；
4. 只在 support 内统计动态 semantic classes，不统计 free/static；
5. 动态类取 mIoU，再平均 1s/2s/3s。

双 support 同时惩罚：

- trailing ghost：旧位置残留；
- missed arrival：新位置缺失。

future GT **只用于 post-hoc metric**。模型、KTA、WM inference 完全看不到 future GT。出生/消失/端点缺失实例排除并记录。

核心实现：`real_motion/metrics/moving_miou_v2.py`。

---

## 5. Frozen VAE 与 real-motion latent cache

训练时不重复跑 VAE。每个 sample 的 cache contract：

```python
moving_history_latent  # [H,16,50,50]
future_moving_latent   # [F,16,50,50]，训练 target
static_future_latent   # [F,16,50,50]
kta_future_latent      # [F,16,50,50]
generation_support     # [F,50,50] bool：未来 compute/loss support
planning_support       # [H+F,50,50] bool：建议提供，负责选窗口
```

可选：

```python
trajectory
confident_static_mask
gt_moving_support
sample_id
```

检查：

```bash
PYTHONPATH=$PWD \
python tools/real_motion/validate_cache.py --cache data/real_motion_train.pt
```

你现有 preprocessing/KTA pipeline 需要先产生 occupancy-space：

- historical true-moving OCC；
- future SE3 static OCC；
- future KTA OCC；
- causal generation support；
- future GT moving OCC（只作为训练 target）。

再统一通过同一个官方 frozen VAE 编码并缓存。**不需要再训练小 encoder。**

---

## 6. Motion-window planner

默认：

```yaml
WINDOW_HW: [20,20]
MAX_WINDOWS: 8
```

两个 support 必须分开：

- `planning_support`：选窗口，应该覆盖 historical moving 到 future KTA tube 的时空路径；
- `generation_support`：未来 flow loss / compute mask。

如果只拿 future tube 选窗口，远期窗口可能把历史位置裁掉。

实际流程：

1. 对 `planning_support` 沿时间取 union；
2. 50x50 greedy planner 在 CPU 做一次，避免 Python `.item()` 对 CUDA 反复同步；
3. origins 传回 GPU；
4. crop history / target / static / KTA / loss mask；
5. valid windows 展平到 batch；
6. overlap scatter 做平均；
7. trainer 计算 `window_coverage`，低于 95% 会报警，避免 `MAX_WINDOWS` 静默漏 support。

代码：`real_motion/windows.py`。

---

## 7. Spatially Aligned Conditioning

不用 global cross-attention。

```text
prior = concat(static_latent, kta_latent)  # 32 channels
              |
        zero-init Linear
              |
       token-wise AdaLN
              |
   spatial / temporal DiT blocks
```

官方 DiTBlock 的参数结构被保留，因此 attention/MLP/AdaLN 权重可以从官方 checkpoint 复用。

`prior_proj` zero-init：训练开始时新 prior 分支增量为 0。

### 绝对位置不能丢

不同 window 不能都用同一套“局部 20x20 位置”。当前实现会根据每个 `window_origin`，从完整 `50x50` absolute sin-cos grid 中**向量化裁剪**对应 positional embedding，并在每个 window forward 使用。

这样同时解决：

- sparse crop 后丢失全局位置；
- 每个 NFE 对 GPU origins `.tolist()` 造成 CPU-GPU 同步。

---

## 8. 官方 checkpoint 初始化

不能直接 `strict=False`。

50x50 -> 20x20 后，官方固定 `pos_embed` shape 不同；PyTorch 即使 `strict=False` 也不会自动忽略 tensor shape mismatch。

使用：

```text
real_motion/checkpoint.py::load_shape_safe
```

只加载：

- key 匹配；
- shape 完全匹配。

并打印 `loaded / target_total / skipped`。

官方局部 `pos_embed` 因 shape 不同会跳过；实际运行的位置由上面的 50x50 absolute positional crop 提供。

---

## 9. 第一场训练：Tiny-set Overfit

先用 64/128 windows：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/train_sparse.py \
  --cache data/real_motion_tiny.pt \
  --upstream-ckpt logs/occfm/2s_3s_nusc_hist_traj/ckpt/epoch=000199.ckpt \
  --output logs/real_motion/tiny_overfit.pt \
  --window 20 \
  --max-windows 8 \
  --batch-size 2 \
  --steps 2000 \
  --lr 2e-5 \
  --amp
```

先看：

1. masked flow loss 是否持续下降；
2. train Moving-mIoU v2 是否明显逼近 oracle；
3. `window_coverage` 是否足够；
4. active ratio 是否与 P0-B 一致；
5. window boundary 是否出现 artifact。

**tiny-set 都学不上去，不进入 full training。**

第一版 loss 只有 masked CFM / flow MSE。不加 ABE、Router、repair、preservation、confidence loss。

---

## 10. Small held-out -> Full training

顺序固定：

```text
P0 -> tiny overfit -> small held-out -> freeze hyperparameters -> full training
```

不要绕过 small held-out 直接 full train。

---

## 11. 稀疏采样

先准备同一 frozen VAE 的：

```text
empty_latent = E(empty occupancy)
```

不能把数值 0 当成 empty latent。

采样：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/sample_sparse_latent.py \
  --cache data/real_motion_val.pt \
  --ckpt logs/real_motion/model.pt \
  --empty-latent data/empty_latent.pt \
  --output outputs/future_moving_latent.pt \
  --window 20 --max-windows 8
```

没有 active window 的样本直接返回 empty latent canvas，不会把 0-batch 喂给 transition。

重叠 windows 在 latent 上平均，然后用官方 frozen decoder 解出 moving occupancy。

---

## 12. Static-Protected Motion Composition

`M_gen` 是 **computation mask，不是 overwrite mask**。

最终规则：

1. `SE3-static` 是 base；
2. `confident_static` 永远保护；
3. WM 只在非 confident-static 区域写动态 semantic prediction；
4. dilation support 内的 empty/error 不允许擦掉 road/building。

代码：`real_motion/composition.py`

命令工具：

```bash
PYTHONPATH=$PWD \
python tools/real_motion/compose_occ.py \
  --static-occ outputs/static.pt \
  --wm-occ outputs/wm_dynamic.pt \
  --conf-static outputs/conf_static.pt \
  --dynamic-classes 1,2,3,4,5,6,7,8 \
  --output outputs/final_occ.pt
```

动态类别 ID 请按你实际 nuScenes occupancy label contract 替换，不能照抄示例。

---

## 13. 第一版不加 confidence gate

KTA 只是 prior，不是天然可靠 fallback。转弯/加速时需要 WM 推翻 KTA。

训练后再统计：

- voxel-level harm / repair；
- instance / motion-tube harm / repair；
- oracle KTA-vs-WM selector headroom。

只有 oracle selector headroom 很大才研究 gate。

优先级：

```text
single-pass lightweight gate > multi-sampling > CFG
```

CFG 最后考虑，因为会增加每个 ODE step 的 forward 次数，破坏效率主线。

诊断代码：`real_motion/metrics/diagnostics.py`。

---

## 14. 困难机动分析

不创造新主指标，只把 frozen Moving-mIoU v2 按 instance subset 重算。

Motion subsets：

- Uniform / Easy
- Accel / Decel
- Turning
- Turning + Speed Change

建议用 interval quantities：

```text
Delta v_h   = |v_h - v_0|
Delta psi_h = |wrap(psi_h - psi_0)|
```

再按 post-hoc KTA center/yaw error 分：

- KTA-Easy
- KTA-Medium
- KTA-Hard

阈值只在 train/calibration side 冻结。

---

## 15. 效率报告

至少：

- mIoU
- Moving-mIoU v2
- Active Token / Window Ratio
- GFLOPs
- end-to-end latency / FPS
- 可选 peak GPU memory

训练使用 latent cache 可以省重复 VAE；但**论文的在线端到端 FPS 必须把推理时一次性的 VAE condition encoding 算进去**。

微基准：

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD \
python tools/real_motion/profile_backend.py --window 20 --batch 8
```

---

## 16. 单元测试

```bash
PYTHONPATH=$PWD/upstream_occfm:$PWD pytest -q tests
```

当前测试覆盖：

- horizon-dependent motion tube；
- coverage / active ratio；
- window crop/scatter；
- planning support 与 loss support 分离；
- MAX_WINDOWS 截断 coverage；
- static protection；
- Moving-mIoU v2 的 0.5m/s 边界；
- 双 box support 对 trailing ghost / missed arrival 的惩罚。

---

## 17. 开发纪律

1. **第一个实验永远先做 P0-A。**
2. future GT 不得进入 motion detector / KTA / M_gen。
3. `planning_support != generation_support`。
4. `M_gen != overwrite mask`。
5. KTA 是 prior，不是最终答案。
6. 第一版不做 residual repair。
7. 第一版不加 Router/CFG/gate。
8. tiny-set 不能 overfit 时不 full train。
9. 效率同时看 FLOPs 与真实 latency。
10. Moving-mIoU v2 contract 不随实验结果修改。
11. 失败时先定位 support / VAE / WM / composition，再改模型。

---

## 18. 不提交 GitHub 的数据

- nuScenes
- OccFM VAE / world-model checkpoints
- KTA cache
- real-motion latent cache

如果 P0 临时使用 GT historical track，请明确标记为 **P0-only diagnostic**，不能作为最终 causal inference。

---

## 19. 上游致谢

本项目基于 `Orbis36/OccFM-NeurIPS2025`：

*Towards foundational LiDAR world models with efficient latent flow matching*, NeurIPS 2025。

请保留官方论文引用。
