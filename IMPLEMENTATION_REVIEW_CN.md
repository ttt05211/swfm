# SWFM 实现审查报告（中文版）

本文档记录 `feat/real-motion-occfm` 分支相对于官方 OccFM 的实现映射、已完成的代码审查、已经修复的问题，以及仍必须在实际 nuScenes / GPU / checkpoint 环境中完成的验收项。

> 结论先行：当前代码已经达到“可以开始 P0 数据验证与 tiny-set 集成实验”的状态，但**不能宣称已经完成 nuScenes 端到端实测**。本地完成的是源码级审查、接口审查、公式/shape 审查和 dependency-light 单元测试；真正的官方 checkpoint GPU forward、实际 KTA cache、nuScenes P0-A/C/D、端到端 decoder/FPS 必须在你的服务器环境继续验证。

---

## 1. 上游版本与兼容策略

官方代码固定为：

```text
Orbis36/OccFM-NeurIPS2025
commit: 64959840a9a4cb54d5b0f6cd4bc6779bb242a853
```

以 git submodule 方式放在 `upstream_occfm/`。

没有直接复制并修改官方源码，原因：

1. 保持 baseline 可复现；
2. 后续能明确区分官方实现与 SWFM 新代码；
3. 可以最大化复用官方 checkpoint；
4. 如果官方代码更新，可以独立同步 submodule，而不是人工 merge 一份复制品。

---

## 2. 官方 OccFM 代码审查结论

### 2.1 Latent cache 路径与我们的方案兼容

官方 OccFM 本身支持 cache mode，并提供 VAE latent cache 工具。CFM 训练可以直接读取 cached latent，不需要训练时重复运行 VAE。

因此本方案采用：

```text
Frozen VAE
    -> offline latent cache
    -> sparse transition-model training
```

而没有新增小型 encoder。

### 2.2 官方 transition model 不是纯 token Transformer

官方 `FLOW_MATCHING_DOWN_X4_DiT` 包含：

```text
Conv3d
+ temporal attention
+ U-Net style down/up sampling
+ DiT blocks
```

官方 latent 输入为 16 channels、50x50 spatial grid。

因此 v1 没有强行改成任意长度 packed tokens，而采用固定 motion windows。这样才能保留 Conv/U-Net/attention 的大部分 checkpoint。

### 2.3 CFM 方向已核对

实现保持官方逻辑：

```text
z0 = Gaussian noise
z1 = future latent
zt = t * z1 + (1-t) * z0
target velocity = z1 - z0
```

本方案唯一核心变化是：

- target 变成 future moving latent；
- loss 只在 future generation support 内计算；
- KTA/static 作为 condition，而不是 residual target。

### 2.4 时序长度约束已核对

官方 transition：

```text
temp_embed length = 12
trajectory_length = 6
```

当前实现保留该限制，并在 sequence length 超出时直接报错，避免 silent shape error。

---

## 3. 方法设计与代码一一映射

| 方法组件 | 实现文件 | 审查状态 |
|---|---|---|
| causal real-motion 三状态接口 | `real_motion/motion.py` | 已实现 baseline contract；最终可替换成现有 KTA/tracker |
| KTA tube / dilation / support | `real_motion/support.py` | 已实现并测试 |
| frozen latent cache contract | `real_motion/cache.py` | 已实现 versioned validation |
| cached dataset | `real_motion/dataset.py` | 已实现 |
| motion-window planner | `real_motion/windows.py` | 已实现；CPU planner + coverage audit |
| token-wise spatial conditioning | `real_motion/models/blocks.py` | 已实现 |
| window OccFM transition | `real_motion/models/transition.py` | 已实现；保留官方 backbone 权重结构 |
| masked CFM | `real_motion/models/cfm.py` | 已实现 |
| checkpoint shape-safe loader | `real_motion/checkpoint.py` | 已实现 |
| sparse train | `tools/real_motion/train_sparse.py` | 已实现 |
| sparse sample / scatter | `tools/real_motion/sample_sparse_latent.py` | 已实现 |
| static-protected composition | `real_motion/composition.py` | 已实现 |
| Moving-mIoU v2 core | `real_motion/metrics/moving_miou_v2.py` | 已实现 |
| harm / repair / oracle selector | `real_motion/metrics/diagnostics.py` | 已实现 core |
| P0-B coverage stats | `tools/real_motion/p0_support_stats.py` | 已实现 |
| runtime micro-profiler | `tools/real_motion/profile_backend.py` | 已实现 |
| Chinese workflow README | `README.md` | 已完成 |

---

## 4. 代码审查中发现并修复的问题

### Bug 1：future-only window planning 会裁掉历史运动上下文

问题：如果 window 只根据 future KTA tube 选择，远期目标窗口可能看不到物体在历史帧中的位置。

修复：区分两个 contract：

```text
planning_support   -> 只负责选窗口
                    historical moving + future KTA tube

generation_support -> 只负责 future compute / flow loss
```

这样为了保留 history 扩大的窗口，不会错误扩大 future supervision。

### Bug 2：20x20 sparse window 丢失 50x50 绝对位置

问题：不同位置的 20x20 window 如果共享同一个局部 position embedding，模型无法区分它们位于原始 latent map 的哪里。

修复：

- 保存 `window_origin`；
- 构造完整 50x50 absolute sin-cos positional grid；
- 根据 origin 为每个 window 裁剪绝对位置；
- 训练和采样均显式传入 origins。

### Bug 3：绝对位置初版会在每个 NFE 触发 CPU-GPU 同步

问题：逐 window `.tolist()` 会在多步采样时不断同步 GPU -> CPU，直接损害 FPS。

修复：absolute position crop 改为 GPU tensor vectorized indexing。

### Bug 4：greedy planner 如果在 CUDA tensor 上做 Python `.item()` 会频繁同步

修复：50x50 support planner 明确只在 CPU 执行一次，再把 `origins/valid` 传回原 device。

这是一次性规划成本，不随 NFE 重复。

### Bug 5：`MAX_WINDOWS` 可能静默截断 moving support

问题：如果 active regions 太分散，最大窗口数用完后仍有 support 未覆盖，训练看起来可以正常进行，但其实一部分 moving target 永远没进入 WM。

修复：新增 `window_coverage()`；trainer 当 min coverage < 95% 时显式报警。

full training 前必须根据数据决定 window size / max windows，使 coverage 达到冻结要求。

### Bug 6：无 moving sample 会产生 0-batch transition

修复：采样阶段如果没有任何 valid window，直接返回 frozen VAE 的 `E(empty occupancy)` latent canvas，不进入 transition model。

### Bug 7：overlap window 写入存在顺序依赖

修复：overlap latent 使用平均 scatter，而不是后窗口覆盖前窗口。

后续如果 boundary artifact 明显，再考虑中心加权；v1 不提前复杂化。

### Bug 8：官方 50x50 checkpoint position embedding shape mismatch

问题：PyTorch `strict=False` 并不会忽略同名 tensor 的 shape mismatch。

修复：`load_shape_safe()` 只有 key 和 tensor shape 都匹配才加载，并输出加载/跳过统计。

官方 50x50 `pos_embed` 会被预期跳过；运行时用前述 absolute 50x50 positional crop。

### Bug 9：`M_gen` 被误当 overwrite mask 会擦坏 static occupancy

修复：实现 `Static-Protected Motion Composition`：

```text
M_gen               -> computation mask
confident_static     -> write protection mask
WM dynamic semantics -> actual dynamic writes
```

WM 的 empty prediction 不允许因为 dilation support 扩大而擦掉 road/building。

---

## 5. 已执行测试

在独立实现环境完成：

```text
python -m py_compile: PASS
pytest: 7 passed
```

当前测试覆盖：

1. horizon-dependent motion tube；
2. GT coverage / active ratio；
3. window crop / scatter round-trip；
4. planning support 与 future loss support 解耦；
5. MAX_WINDOWS coverage truncation detection；
6. static-protected composition；
7. Moving-mIoU v2：0.5 m/s 等号边界；
8. dual GT box support 对 trailing ghost / missed arrival 的惩罚。

> 这些是 dependency-light 测试，不等于官方 checkpoint GPU integration test。

---

## 6. 当前没有伪装成“已完成”的部分

### 6.1 官方 OccFM checkpoint 的真实加载复用比例

已经实现 shape-safe loader，但当前环境没有你的实际 `.ckpt` 文件，因此**必须在服务器第一次运行 tiny-set 时记录**：

```text
loaded tensors / target tensors
shape-skipped tensors
missing tensors
```

如果除 position embedding 和新增 prior adapter 外仍大量 skip，必须先停下来检查 key mapping。

### 6.2 P0-A / P0-C / P0-D 依赖你的实际 nuScenes + KTA 数据

当前没有用 future GT 伪造 causal input，因此这几项不会在无数据环境中假装跑通。

第一件事仍然是 P0-A：

```text
Full
True-static-only
True-moving-only
```

使用同一个 frozen OccFM ckpt。

### 6.3 raw nuScenes -> Moving-mIoU v2 adapter

当前实现的是指标核心：interval speed、双 oriented-box support、dynamic-class mIoU accumulator。

它要求传入的两个 box 已经被转换到目标 future ego grid。你现有 nuScenes evaluation pipeline 需要负责：

```text
world box / t0 box
    -> target future ego coordinate
    -> Box3D contract
```

这样可以把数据集坐标变换与指标数学定义解耦，也避免 metric 内部重复实现一套 nuScenes pose logic。

### 6.4 Official frozen decoder 没有复制到本仓库

这是刻意设计，不是遗漏：

`sparse_sample` 输出完整 50x50 future moving latent canvas，后续直接走 `upstream_occfm` 官方 frozen decoder。

这样不会维护第二份 VAE decoder。

真正服务器端验收时必须做：

```text
sparse latent
 -> official frozen decoder
 -> moving OCC
 -> static-protected composition
 -> metrics
```

### 6.5 端到端 FPS 尚未实测

当前只提供 backend profiler。

最终论文的在线 FPS 必须包括：

- real-motion/KTA preprocessing；
- condition VAE encoding；
- window planning/crop；
- NFE sparse WM；
- scatter；
- decoder；
- composition。

latent cache 时间只能用于训练/诊断，不得冒充端到端 FPS。

---

## 7. 服务器端验收顺序

### Step 1：P0-A

Frozen true-motion decomposition。

不过就停，不训 WM。

### Step 2：P0-B/C/D

确认：

- moving 足够 sparse；
- KTA tube coverage 足够高；
- stationary->moving / innovation blind spot 可控；
- SE3-static + GT-moving oracle 有足够 headroom。

### Step 3：生成 tiny cache

64/128 windows。

先运行：

```bash
python tools/real_motion/validate_cache.py --cache ...
```

### Step 4：检查 checkpoint reuse

第一次 `train_sparse.py` 启动时保存 loader report。

如果复用异常，不继续训练。

### Step 5：tiny-set overfit

验证训练集能力，而不是先看 val。

### Step 6：small held-out

固定：

- window size；
- max windows；
- tube radius；
- uncertainty policy。

### Step 7：full training

full training 后再做：

- Moving-mIoU v2；
- harm/repair；
- oracle selector headroom；
- maneuver / KTA-hard subset；
- end-to-end latency/FPS。

---

## 8. 最终验收结论

当前 branch 已完成：

- 方法主干代码化；
- 官方 OccFM 版本固定；
- sparse window backend；
- cached prior conditioning；
- CFM train/sample；
- absolute position preservation；
- support/window safety；
- static-protected composition；
- Moving-mIoU v2 core；
- diagnostics core；
- 中文完整执行 README；
- dependency-light unit tests；
- 多轮自审与 bug 修复。

当前 branch 尚需在真实环境证明：

- causal real-motion 数据协议正确；
- official ckpt 实际复用率；
- VAE moving-only / empty canvas 成立；
- tiny-set 能 overfit；
- small-held-out 能泛化；
- sparse backend 在 GPU 上真正比 dense OccFM 快；
- final Moving-mIoU / mIoU 达到目标。

因此，本分支当前适合保持 **Draft PR** 状态，直到 P0 和 tiny-set 验收完成后再考虑 merge。
