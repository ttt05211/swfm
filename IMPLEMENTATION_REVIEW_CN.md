# SWFM 实现审查报告（合并 main 前最终版）

本文档记录 `Real-Motion Sparse OccFM` 在合并到 `main` 前完成的源码级审查、已修复问题，以及仍必须在实际 nuScenes / GPU / OccFM checkpoint 环境中完成的验收项。

> 当前结论：代码已经达到“可以进入 P0 数据验证与 tiny-set 集成”的状态；但由于当前执行环境没有用户的 nuScenes、KTA cache、OccFM `.ckpt` 和目标 GPU，**不能把本次审查等价成端到端实测通过**。

## 1. 上游固定版本

```text
Orbis36/OccFM-NeurIPS2025
commit: 64959840a9a4cb54d5b0f6cd4bc6779bb242a853
```

以 `upstream_occfm/` git submodule 固定，不直接复制魔改官方源码。

## 2. 与官方 OccFM 的接口核对

官方 OccFM 已确认：
- CFM 训练路径支持 cached latent；
- transition model 是 `Conv3d + temporal attention + U-Net down/up + DiT`，不是纯 token Transformer；
- latent transition 输入为 16 channels、50×50；
- flow matching 使用 `zt=t*z1+(1-t)*z0`，target 为 `z1-z0`；
- sampling 使用 shifted ODE；
- official DiT 使用 AdaLN-Zero；
- official `DiTAttention` 支持 `attention_mode='flash'`；
- official spatial `pos_embed` 对 2500 tokens 使用 `grid_size=int(sqrt(N))+1=51` 后截断前 2500 token。

因此 v1 使用固定 motion-window sparse backend，而不是强行改成任意长度 packed tokens。

## 3. 主要代码映射

| 功能 | 文件 |
|---|---|
| causal motion 三状态 contract | `real_motion/motion.py` |
| KTA tube / support | `real_motion/support.py` |
| latent cache contract | `real_motion/cache.py` |
| cached dataset | `real_motion/dataset.py` |
| sparse window planner / crop / scatter | `real_motion/windows.py` |
| token-wise AdaLN | `real_motion/models/blocks.py` |
| OccFM window transition | `real_motion/models/transition.py` |
| masked CFM | `real_motion/models/cfm.py` |
| checkpoint safe loading | `real_motion/checkpoint.py` |
| sparse train/sample | `tools/real_motion/` |
| static-protected composition | `real_motion/composition.py` |
| Moving-mIoU v2 | `real_motion/metrics/moving_miou_v2.py` |
| harm/repair/oracle selector | `real_motion/metrics/diagnostics.py` |

## 4. 合并前已修复的问题

### Bug 1：future-only window 可能看不到历史运动
最初只用 future KTA tube 规划窗口，会让远期 target window 看不到历史位置。

### Bug 2：history-only window 会浪费 MAX_WINDOWS
反向把 `historical moving + future KTA` 全部当 required support 后，会产生 history-only window；但不同 window 之间没有跨窗口通信，因此它并不能帮助另一个 future window。

最终 contract：

```text
generation_support -> required future support，决定哪些窗口必须打开
planning_support   -> context tie-break，只影响候选窗口位置
```

每一个 selected window 都必须由未覆盖的 future target 触发。

### Bug 3：window 丢失 50×50 全局位置
现在保存 `window_origin`，从完整 absolute position table 中裁剪对应位置。

### Bug 4：position table convention 与官方不一致
官方不是简单 50×50 sin-cos，而是 51×51 后截断前 2500 token。当前实现严格复现这一顺序后再 crop，尽量保持 pretrained positional convention。

### Bug 5：position crop 每个 NFE CPU/GPU 同步
移除逐 window `.tolist()`，改为 GPU tensor vectorized indexing。

### Bug 6：greedy planner 在 GPU 上 Python 同步
50×50 planner 固定在 CPU 一次运行，origins 再传回 device。

### Bug 7：MAX_WINDOWS 静默截断
新增 `window_coverage()`；trainer/sampler 默认要求 future generation-support coverage >= 95%，否则 hard fail。只有诊断时允许 `--allow-low-coverage`。

### Bug 8：tiny cache 可能死循环
- `drop_last=False`
- empty dataset 直接报错
- 整个 epoch 没有 active future window 直接报错

### Bug 9：无 active sample 导致 0-batch
采样时直接返回 `E(empty)` canvas，不进入 transition。

### Bug 10：overlap window 写入顺序依赖
改为 latent average scatter。

### Bug 11：50×50 -> 20×20 checkpoint shape mismatch
`load_shape_safe()` 只加载 key 和 shape 都匹配的 tensor，并显式报告 skip。

### Bug 12：M_gen 被错误当 overwrite mask
最终使用 Static-Protected Motion Composition：

```text
M_gen               -> computation mask
confident_static     -> write protection
WM dynamic semantics -> actual dynamic writes
```

### Bug 13：Moving-mIoU v2 跨 horizon 错误 micro 聚合
新增 `MovingMIoUV2MultiHorizon`，强制：

```text
mIoU@1s
mIoU@2s
mIoU@3s
-> arithmetic mean
```

而不是把三个 horizon 的 voxel/support 混在一起算一次。

## 5. 已完成测试

本地 merge-review 最终结果：

```text
python -m py_compile: PASS
pytest: 8 passed
```

覆盖：
1. horizon-dependent motion tube；
2. GT coverage / active ratio；
3. window crop/scatter；
4. history context 不单独创建无用 window；
5. MAX_WINDOWS truncation detection；
6. static-protected composition；
7. Moving-mIoU v2 0.5m/s 等号边界；
8. dual-box support trailing ghost / missed arrival；
9. frozen Moving-mIoU v2 horizon-first aggregation。

## 6. 当前已知但必须用数据验证的 v1 边界

固定 motion-window backend 没有跨窗口通信。如果高速目标从历史位置到 future target 的位移跨度超过单个 window 可覆盖范围，即使 `planning_support` 做 context tie-break，也不能让两个不同窗口互相传递历史信息。

因此 P0/tiny 阶段需要额外观察 future-target window 的 historical-moving context coverage。若明显不足，应优先增大 window、改变窗口组织方式或升级 backend，而不是直接进入 full training。

## 7. 服务器端仍必须验收

### P0-A（第一项）
同一 frozen OccFM：

```text
Full
True-static-only
True-moving-only
```

不过就不训 sparse WM。

### P0-B/C/D
- sparsity + KTA tube coverage；
- stationary→moving / innovation；
- SE3-static + GT-moving oracle headroom。

### Checkpoint integration
第一次 tiny run 必须记录：

```text
loaded tensors / target tensors
shape-skipped
missing
```

如果除新 prior / expected positional item 外还有大量 skip，先停。

### Tiny overfit
64/128 windows。训练集都学不上去，不进入 full training。

### End-to-end
必须在服务器真实跑：

```text
preprocess/KTA
-> frozen VAE condition encoding
-> sparse WM
-> scatter
-> official decoder
-> static-protected composition
-> Moving-mIoU v2
-> latency/FPS
```

## 8. 合并结论

源码级没有再发现阻止进入 P0/tiny integration 的明显问题。当前适合合并到 `main` 作为正式实验代码基线，但不应在 README/论文中宣称 nuScenes 端到端、官方 checkpoint GPU forward 或 FPS 已经验证完成。
