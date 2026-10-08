# 最终论文框架、统一数据流与 FPS 计时协议（2026-10-06）

> 状态：**框架层级 / 数据接口 / FPS 边界冻结；Point CCR 的具体支持域与精度尚未冻结。**
>
> 目标：借数据流清理把“代码模块、论文 Fig.2、正式 FPS 口径”统一起来。当前阶段只做等价重构与测速协议收口，**不修改网络结构、已训练权重、阈值、支持域、监督目标或评价口径**。

---

## 1. 当前基线与冻结项

开发线：

- repository: `ttt05211/swfm`
- branch: `feature/v22-causal-emergence-tokens`
- 本文档基于 2026-10-06 的 V22 Point CCR / epoch19 transport 路径整理。

科学参考：

- 冻结 Clean-E14 / V18：`freeze/v18-main-final-20260918`
- 当前质量参考：**epoch19 Local joint model**
- 当前高效候选：**V18 transport + Point CCR**
- Point CCR 目前仍未通过最终精度门槛，因此本文档只冻结其**模块职责和接口**，不宣称当前具体 Point CCR support 为最终方法。

本轮清理禁止：

- 改 V18 / epoch19 权重；
- 改 ADD / REMOVE 阈值；
- 改 candidate / support；
- 改 source identity / ownership / fallback 语义；
- 改 loss、sampling target 或未来 GT 的使用边界；
- 用 cached Strong/KTA 结果作为正式 FPS；
- 为了“论文好看”而改变模型真实数据依赖。

---

# 2. 最终论文 Fig.2：三个主模块 + 一个薄融合层

最终方法按论文层级划分为：

1. **Causal History Representation**
2. **Source-centric Motion & Transport**
3. **Canonical Causal Repair**
4. 薄层：**Constrained Composition**

推荐主拓扑：

```text
Historical Occupancy / Visibility / Historical Poses
                         |
                         v
          +--------------------------------+
          | Causal History Representation |
          +---------------+----------------+
                          / \
                         /   \
              Source States   Canonical Evidence
                   |                |
                   v                v
        +-------------------+  +-------------------------+
        | Source-centric    |  | Canonical Causal Repair |
        | Motion Prediction |  | shared encode once      |
        +---------+---------+  | + horizon readout x6    |
                  |            +------------+------------+
          dxy / dyaw / exist                |
                  v                         | ADD / REMOVE
          +---------------+                 |
          | SE(2) Source  |---- dynamic ----+
          | Transport     |     context
          +-------+-------+
                  |
          Transported Occupancy
                  |                         |
                  +------------+------------+
                               v
                    ----------------------
                     Constrained Composition
                    ----------------------
                               |
                               v
                     Future Occupancy x6
```

**Future Ego Poses / Future Ego Condition** 从图上方进入：

- future-frame projection / rendering；
- CCR horizon context；

不要画成当前 Source-centric Motion Predictor 的直接 learned input，除非未来代码真实修改为该接口。

---

## 3. Module I — Causal History Representation

### 3.1 名称与定位

正式名称建议使用：

> **Causal History Representation**

而不是默认使用 `Tokenizer`。

原因：当前实现主要是**确定性的结构化历史编码、source 提取/关联和几何配准**，不是 VQ tokenizer，也不是离散量化器。正文可将其描述为“将历史观测转换为 model-ready structured representation”。

### 3.2 输入

论文图使用一般形式：

[
\{\mathcal O_{t-H+1:t},\;\mathcal V_{t-H+1:t},\;\mathbf P_{t-H+1:t}\}.
]

其中包括：

- historical semantic occupancy；
- visibility / observation mask；
- historical ego poses。

当前 V22 epoch19 / Point CCR 路径的 active history contract 为 4 帧；图中不必把历史帧数硬编码进模块名称。

### 3.3 输出

输出分成两路。

#### A. Source States

代表可搬运 source 的结构化状态，包括实现所需的：

- source identity / semantic class；
- observed source geometry；
- causal historical association / registration；
- local historical semantic / motion context；
- transport network 所需的 source-level features。

图上统一写 **Source States**，不要展开 connected components、matching、tube crop 等工程步骤。

#### B. Canonical Evidence

代表可复用的历史几何证据。

当前实现的真实语义是：

- dynamic evidence：按 source association / registration 对齐到当前 source 参考；
- static road / sidewalk evidence：按历史 ego pose 对齐到当前场景参考；
- 保留可用于未来 projection 的 metric/world geometry；
- support 包括当前实现定义的 t0 source、可见注册历史 source、观测 road/sidewalk 与局部 halo。

因此论文中不要声称“所有 evidence 都进入同一个统一物体坐标系”。推荐描述为：

> **source-aligned dynamic evidence and ego-aligned static evidence**

### 3.4 图中不展开的实现细节

以下内容属于该模块内部实现，不作为 Fig.2 独立大框：

- component extraction；
- source matching；
- registration；
- lookup/hash/cache；
- native CPU kernel；
- thread pool；
- fixed descriptor construction；
- neighbor lookup。

---

# 4. Module II — Source-centric Motion & Transport

该模块不能只画成“速度外推 + warp”。当前 V18 包含真实的 learned spatiotemporal predictor。

推荐内部只保留两层：

```text
Source States
     |
     v
Source-centric Spatiotemporal Prediction
     |
     |  dxy, dyaw, existence
     v
SE(2) Source Transport
```

### 4.1 Source-centric Spatiotemporal Prediction

概念上包含：

- historical semantic / spatial encoding；
- temporal source context；
- causal motion prior；
- six future source queries；
- residual XY、relative yaw、existence prediction。

图中允许单独画一个很小的 **Causal Motion Prior** 输入，但不要把 Strong、KTA、majority vote、tube schedule 等实现名称逐个展开。

### 4.2 SE(2) Source Transport

核心原则：

> 已观测 source 的几何不重新生成，只预测其未来 source-centric rigid motion 并搬运。

输出：

- six-horizon transported source states；
- transported dense occupancy / ownership state；
- dynamic source context 供 CCR 使用。

---

# 5. Module III — Canonical Causal Repair

正式名称：

> **Canonical Causal Repair**

推荐核心视觉信息：

> **Encode once -> horizon-conditioned readout x6**

### 5.1 Shared evidence encoding

历史 canonical evidence 每个点只做一次共享编码，而不是对 6 个未来时刻重复做完整 patch encoder。

### 5.2 与 Source Transport 的真实耦合

当前实现中，**dynamic evidence** 额外使用：

- `history_source_context`；
- `future_transport_queries[h]`。

因此 Source Transport -> CCR 的箭头推荐标：

> **Dynamic Source Context**

不要暗示 static road/sidewalk evidence 也依赖 V18 source query。

### 5.3 Horizon-conditioned context

六个 horizon 各自条件化的信息包括当前实现中的：

- projected future position；
- horizon/time；
- future ego relative transform；
- predicted dynamic-source yaw；
- transported baseline semantic；
- fallback semantic；
- ownership / legality context。

因此“共享”的准确含义是：

> **shared canonical evidence encoding, independent horizon-conditioned decisions**

而不是六个未来时刻共享完全相同的 repair result。

### 5.4 Repair 职责

不要写成“修复所有 uncertain boundaries”，因为当前没有独立 uncertainty predictor。

推荐表述：

> **Predict horizon-specific ADD / REMOVE corrections over a causally supported repair domain.**

中文：

> **在因果历史证据支持的候选域内预测各未来时刻的 ADD / REMOVE 修正。**

当前阶段不能声称：

- arbitrary future object generation；
- never-seen dynamic object generation；
- unrestricted dense completion。

---

# 6. Constrained Composition

该部分保留为 Fig.2 底部的**薄融合层**，不作为第四个等大的主模块。

建议图中只写：

> **ownership-aware · occupancy-preserving · fallback-safe**

正文必须明确：

1. ADD 不覆盖原 transport 已占据 voxel；
2. REMOVE 只能作用于对应 source / visible owner；
3. REMOVE 恢复保存的 lower-layer / fallback 内容，而不是全局删除；
4. static semantic conflict fail-closed；
5. zero repair action 必须严格返回 transport baseline。

这些约束定义了 repair 的编辑权限，属于方法本身，不应全部隐藏在附录。

---

# 7. 论文 Fig.1 / Fig.2 分工

## Fig.1：Why

只解释核心动机：

- regenerate everything：浪费计算，也可能破坏已有几何；
- ours：**transport observed geometry + repair causally supported missing geometry**。

Fig.1 不重复完整网络。

## Fig.2：How

只展示：

> Causal History Representation -> Source-centric Motion & Transport + Canonical Causal Repair -> Constrained Composition

不要展示 cache、hash、thread pool、native kernel、具体 Strong/KTA 规则树等工程细节。

---

# 8. 数据流清理后的正式代码接口

本轮数据流清理目标是让**论文模块、代码 API、FPS 边界一致**。

推荐统一为：

```python
history_state = prepare_history(...)
future = forecast_six(history_state, future_ego_condition)
```

概念对应：

```text
prepare_history()
    -> Causal History Representation
    -> immutable CausalHistoryState

forecast_six()
    -> Causal Motion Prior / Strong-KTA
    -> Source-centric Motion Prediction
    -> SE(2) Source Transport
    -> Canonical Causal Repair
    -> Constrained Composition
    -> six dense occupancy frames
```

## 8.1 CausalHistoryState 允许包含

只允许依赖已观测历史、且与 learned future prediction 无关的确定性信息：

- historical occupancy / visibility / historical poses；
- source identity / class / observed geometry；
- historical source association / registration；
- fixed canonical evidence / descriptors；
- fixed history-only neighbor/index structure；
- 一次生成的 immutable content fingerprint。

## 8.2 CausalHistoryState 禁止包含

不能把 forecast 工作提前缓存进 history representation：

- future GT；
- repair target / label；
- learned CCR activations；
- V18 future motion prediction；
- future transport queries；
- future source targets；
- future ownership / fallback；
- horizon-specific repair logits / probabilities；
- cached Strong/KTA future prior 结果。

---

# 9. 训练数据流也必须同时清理

训练和推理不得保留两套逐渐分叉的 geometry/preparation 协议。

Point CCR 当前应向已有 full-joint batching 方式收敛：

```text
batch CausalHistoryState
        |
        v
ONE batched frozen V18 motion forward
        |
        v
split per window/source
        |
        v
live horizon projection / legality
        |
        v
GT repair targets (training only)
        |
        v
batched CCR head
        |
        v
loss / backward
```

目标：

- batch4/source128 等现有 batch contract 下，只做一次 V18 batched motion forward；
- fixed history evidence 每 window 只构建一次；
- six horizons 共享一次 canonical point encoding；
- GT 只在训练 target 阶段进入，不影响 candidate/support/history representation；
- 不缓存 learned activations 跨 optimizer step；
- 不建立几十/几百 GB 的全量 learned-feature cache。

建议加入构建次数/调用次数断言，主动抓重复计算：

- fixed history representation：1 次 / window；
- V18 motion：1 次 / batch；
- shared CCR point encoding：1 次 / window forward；
- horizon readout：6 个条件读出，不允许 6 次完整 history encoder。

---

# 10. 唯一正式 FPS 协议：Dense Forecast FPS

不再设置两个“正式 FPS 口径”。

最终论文只保留：

> **Dense Forecast FPS**

定义：

> 从已经完成因果历史表示的 `CausalHistoryState` 开始，到 6 帧 dense semantic future occupancy 全部计算完成为止。

这与“history representation / tokenizer 可预先形成，forecast-dependent computation 必须计时”的公开 world-model 实践相容，同时不会把本方法的 Strong/KTA、motion、repair 或 dense composition 偷移到 timer 外。

## 10.1 正式 timer 边界

```text
Raw history
    |
    v
prepare_history()
    |
    v
CausalHistoryState
    |
    |   EXCLUDED FROM OFFICIAL FPS
    |
============ CUDA sync / START TIMER ============

Causal Motion Prior / Strong-KTA
             |
Source-centric Motion Prediction
             |
SE(2) Source Transport
             |
future-frame projection / ownership
             |
Canonical Causal Repair
             |
Constrained Composition
             |
six FINISHED dense semantic occupancy grids

============= CUDA sync / STOP TIMER =============
```

## 10.2 正式 FPS 必须计入

- fresh Strong / KTA / causal motion prior；
- frozen V18 motion forward；
- future transport query decoding；
- source-centred SE(2) transport；
- future ego projection；
- ownership / fallback / legality；
- canonical evidence 的 live future projection；
- CCR shared learned encoding；
- 6 horizon-conditioned readouts；
- ADD / REMOVE action application；
- final dense semantic composition。

**cached Strong/KTA 只能作为 profiler / ablation，不能作为论文正式 FPS。**

## 10.3 正式 FPS 不计入

- checkpoint loading；
- CUDA / native compilation；
- warm-up；
- disk I/O / dataset loading；
- historical occupancy 文件读取；
- deterministic history-only source extraction / association / registration；
- fixed history-only canonical representation construction；
- GT；
- metric calculation；
- correctness hash；
- visualization；
- 保存 `.npy/.npz`；
- dense output 的 GPU->CPU copy（只要正式模型输出已在设备上完成并在 stop 前 CUDA synchronize）。

## 10.4 计时实现

由于本方法同时包含 CPU geometry 和 GPU network，正式计时统一使用：

```python
torch.cuda.synchronize()
start = time.perf_counter()

dense_future = forecast_six(history_state, ...)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

不使用未同步的 `time.time()`；也不只用 CUDA Event 忽略 CPU wall time。

## 10.5 FPS 定义

若一次 forward 完成 6 个 future frames：

[
\mathrm{Dense\ Forecast\ FPS}
=
\frac{6N}{\sum_{i=1}^{N} T_i}.
]

即：

- 不先算每个 window 的 FPS 再平均；
- 用**总 future frame 数 / 总同步 wall-clock 时间**。

同时报告同一口径下的：

- mean six-frame latency；
- P50 latency；
- P90 latency。

这些是同一计时边界的 latency 描述，不是第二套 FPS。

## 10.6 正式 benchmark population

继续使用固定、可复现、非 cherry-pick 的窗口：

- batch size = 1；
- 20 个固定窗口；
- 18 个 scene-balanced；
- 2 个 high-source-count stress；
- 不按 GT / error / speed 选窗口；
- warm-up 后正式计时；
- 推荐 3 次 interleaved repeat；
- 同一 population 同时跑 reference 与 optimized implementation。

---

# 11. 与公开方法的 FPS 审计结论

公开实现没有统一计时边界，因此不能直接把论文 FPS 当作严格 apples-to-apples：

- I2-World：公开 timer 主要覆盖 latent autoregressive world prediction，历史 tokenizer 在外，dense occupancy decoder 也在 timer 外；
- GenieDrive：E2E 路径可包含 VAE encode，但公开 timer 停在 latent prediction 后，occupancy decoder 在外，而且同步语句被注释；
- OccFM：提供显式 CUDA-synchronized timer，cached latent 后的 flow sampling 与 occupancy decoding 可被计入；
- OccWorld：可计入 occupancy encode + autoregressive token prediction，但 dense occupancy decoder 位于其主要 timer 之后；
- SparseWorld-TC：FPS 专用 online 路径显式复用历史 image backbone feature，未发现与上述方法统一的完整 timer 边界。

因此本文不继承任一单独工作的私有计时边界，而采用上面定义的 **Dense Forecast FPS**。

原则是：

> **history-only deterministic representation may be prepared before the timer; all forecast-dependent computation and dense future output generation must be timed.**

---

# 12. 数据流清理后是否需要重新训练？

## 12.1 结论

**纯数据流 / 执行路径清理不要求重新训练。**

如果本轮只做：

- 去除重复计算；
- 抽出 `prepare_history()`；
- 建立 immutable `CausalHistoryState`；
- 将逐-window V18 forward 改成等价 batched forward；
- 复用固定 history-only descriptors；
- 合并相同 geometry construction；
- 规范 timer；
- 不改变 feature、support、weights、thresholds、loss、sampling 与模型数学函数；

则已有 checkpoint 本身仍然有效，不应该为了“适配重构代码”而重新训练。

## 12.2 清理完成后的等价性 Gate

在宣布“不需重训”前，至少验证：

1. 同一 checkpoint、同一输入 window；
2. source population / ordering 一致；
3. Strong/KTA 输出一致；
4. V18 residual XY / yaw / existence 与 latent context 一致；
5. transported dense baseline / owners / fallbacks 一致；
6. canonical evidence population / class / actor / coordinates 一致；
7. repair plan 的 flat/base/fallback/legal/context 一致；
8. Point CCR logits/probabilities 一致（若仅因合法的 batched GEMM/BF16 kernel 出现不可避免的浮点差异，必须单独记录，不得默认为“新模型”）；
9. six dense outputs 一致；
10. dev64 / dev512 的最终指标不因重构发生变化。

只要这些 Gate 通过，**不重训**。

## 12.3 哪些情况才需要重新训练

只有当后续真正改变模型语义，例如：

- 改 CCR support / candidate domain；
- 改 history representation 的 feature definition；
- 增加/删除 learned layer；
- 改 Source Transport -> CCR 的条件信息；
- 改 sampling / loss / target；
- 改阈值或 compositor 的动作权限；

才把它视作新的科学模型，需要训练并重新做精度比较。

## 12.4 当前 Point CCR 后续训练的含义

当前 Point CCR 仍低于 epoch19 Local 质量参考。若数据流清理和 restricted oracle 之后决定做完整 TRAIN20430 x 3 epochs warm-start，这属于：

> **“当前 CCR 本身可能训练不足，因此继续训练”**

而不是：

> **“因为数据流清理，所以必须重新训练”**。

两件事必须严格区分。

---

# 13. 本轮完成条件

本轮“数据流 + FPS 收口”完成的最低标准：

- [ ] `CausalHistoryState` 的字段合同冻结；
- [ ] `prepare_history()` 唯一历史准备入口；
- [ ] `forecast_six()` 唯一正式推理入口；
- [ ] 训练复用相同 history representation；
- [ ] 删除/绕过重复 source extraction、registration、canonical history construction；
- [ ] batched frozen V18 motion 正式用于 Point CCR training；
- [ ] 共享 CCR encoding 不在六 horizon 重复；
- [ ] exact/equivalent parity Gate 通过；
- [ ] 单一 Dense Forecast FPS runner；
- [ ] fresh Strong/KTA 纳入正式 timer；
- [ ] history-only preparation、GT、metrics、I/O 从正式 timer 排除；
- [ ] 固定 20-window benchmark 完成；
- [ ] 输出 mean/P50/P90 latency + Dense Forecast FPS；
- [ ] 不因本轮等价清理启动不必要的 retraining。

---

## 一句话方法定义

> **We organize causal history into transportable source states and reusable canonical evidence, predict source-centric motion to transport observed geometry, and produce horizon-specific repairs from a shared causal representation.**

中文：

> **将因果历史组织为可搬运的 source 状态与可复用的 canonical 几何证据；预测 source-centric 运动以保留已有几何，再从共享因果表示中生成六个未来时刻各自的占据修正。**


---

# 13.1 正式 FPS 冻结结果（L40S，2026-10-06）

固定 20 个窗口（18 scene-balanced + 2 high-source stress），batch=1，3 次重复，共 60 个 synchronized samples / 360 个未来帧：

- **Dense Forecast FPS = 47.9285**
- mean six-frame latency = **125.186 ms**
- P50 = **130.548 ms**
- P90 = **145.928 ms**
- legacy/new parity = **20 / 20 PASS**
- status = **complete**

正式计时范围严格遵守本文件第 10 节：

- timer 内：live KTA/Strong、V18 motion、SE(2) transport、future projection/ownership、CCR shared encoding + six readouts、constrained dense composition；
- timer 外：history-only representation、disk I/O、GT/metrics、checkpoint loading、compile/warmup。

阶段均值（仅诊断，不替代 uninterrupted synchronized wall-clock 总时间）：

| Stage | Mean ms |
|---|---:|
| causal motion prior | 0.405 |
| Strong prior | **83.214** |
| Strong clear index | 0.503 |
| live KTA staging | 0.075 |
| V18 motion forward | 6.456 |
| source transport + layers | 7.841 |
| six projection / ownership / legality | 13.083 |
| shared CCR encode + six readouts | 8.964 |
| constrained dense composition | 3.692 |

当前最大 runtime bottleneck 是 Strong prior（约 66.5% 正式六帧时延）。**该结果现阶段冻结，不继续做 FPS micro-optimization。**

因为 20/20 同窗 old/new motion/probability/dense byte parity 全部通过，本轮数据流清理判定为 execution-equivalent refactor：

> **不需要因为数据流清理而重新训练 epoch19 V18 或当前 Point CCR checkpoint。**

下一阶段优先追当前 Point CCR 与 epoch19 Local 的质量差距，而不是继续优化 FPS。

---

# 14. 实现状态（2026-10-06 晚）

当前实现分支：

- `feature/v22-final-dataflow-fps`
- Draft PR: #78

已实现：

- `real_motion/final_dataflow.py`
  - `CausalHistoryState`
  - `prepare_history()`
  - `build_causal_motion_prior()`
  - `forecast_six()`
  - `batch_frozen_motion()`
- `tools/real_motion/benchmark_p0_f9_dense_forecast_fps.py`
  - 唯一正式 `Dense Forecast FPS` 入口；
  - history-only representation 在 timer 外；
  - KTA 与 Strong 在 timer 内重新构建；
  - V18 motion / SE(2) transport / CCR / composition 全部计时；
  - batch=1；
  - CUDA 前后同步；
  - 固定 20-window population；
  - 默认 3 repeats；
  - 旧路径与新路径同窗 byte-parity gate。
- `tools/real_motion/ccr_screen_common.py`
  - 新增可选 `--ccr-batched-motion`；
  - 一个 packed window batch 只执行一次 frozen V18 motion forward；
  - 默认关闭以保持旧 checkpoint / training execution contract；
  - 只有显式开启时才形成新的训练执行合同。
- `tests/test_final_dataflow.py`
  - live KTA 重建；
  - history state 不携带 KTA / future GT / target；
  - batched frozen motion 与逐窗 CPU reference 等价。

## 14.1 正式服务器 FPS 命令

沿用之前 Point CCR FPS 实验的真实路径，只把脚本换成：

```bash
python tools/real_motion/benchmark_p0_f9_dense_forecast_fps.py \
  --config "$CONFIG" \
  --checkpoint "$EPOCH19" \
  --ccr-checkpoint "$POINT_CCR" \
  --base-checkpoint "$CLEAN_E14" \
  --dev-cache "$DEV_CACHE" \
  --population-manifest "$POP_MANIFEST" \
  --dataroot "$DATAROOT" \
  --dev-info "$DEV_INFO" \
  --out-dir "$OUT/dense_forecast_fps_final" \
  --device cuda \
  --windows 20 \
  --stress-windows 2 \
  --repeats 3 \
  --cpu-workers 8 \
  --ccr-cpu-workers 4 \
  --parity-windows 20
```

如果配置需要 override，继续使用现有 `add_config_args` 支持的 `--override` 参数。

## 14.2 服务器验收顺序

1. dependency-light CI / py_compile / unit tests；
2. 1-window GPU smoke：`--windows 1 --stress-windows 0 --parity-windows 1`；
3. 4-window parity smoke；
4. 正式 20-window × 3 repeats；
5. 只有 old/new parity 全过后，才接受新的 Dense Forecast FPS；
6. FPS 验收不会改变 Point CCR 的精度状态。

## 14.3 训练数据流开关

本轮只把 batched frozen V18 motion 做成**显式 opt-in**：

```text
--ccr-batched-motion
```

在真实 GPU 上完成 batched-vs-per-window latent/output parity 前，不把它静默写进旧训练合同，也不要求因本次数据流清理重训现有 checkpoint。
