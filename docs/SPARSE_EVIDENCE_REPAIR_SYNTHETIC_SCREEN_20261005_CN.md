# Refine 结构替代：本地模拟筛选（非真实数据验收）

## 做了什么

本次不修改 Local 网络、epoch19 权重、阈值或服务器实验。新增独立实验模块
`real_motion/sparse_evidence_repair.py` 和一次运行入口
`tools/real_motion/benchmark_sparse_repair_synthetic.py`。

目的不是把当前逐列网络再加速 10%，而是比较数量级不同的计算结构，先在本地
排除明显昂贵或明显缺乏表达能力的候选，再决定是否值得移植到真实数据。

默认严格 **4 历史 → 6 未来**。GT 仅生成训练标签与评估数字；历史候选、输入特征、
邻域一致性、source context 均从四帧观测产生，不给网络 primitive 的真实尺寸。
模拟 context 只是从历史占据点统计得到的 128 维代理，不是已训练 V18 的真实 latent。

## 一次比较的候选

| 候选 | 计算结构 | 要验证的限制 |
|---|---|---|
| `column_196` | 当前 LinkedColumns，完整 4×7×7 memory，两个 cross-attention | 旧逐列计算成本；本地从随机初始化训练，非 epoch19 |
| `column_36` | 相同完整 patch CNN，reader 仅取 4×3×3 token | 减少 token 但仍重复 patch 编码，收益是否足够 |
| `once` | 每个 canonical evidence 点输出一个 ADD score，重复六帧 | 刚体适合；时变形状有不可消除的上限 |
| `actor_gate` | 点编码一次＋六个 source 标量 gate | 不能独立控制同一 source 两侧的相反变化 |
| `cached_future` | 点编码一次＋六个轻量 point/time 交互头 | 保留时变修复，但未显式保留空间证据一致性 |
| `local_consensus` | 同上，增加一次计算的六邻域四帧占据一致性 | 避免孤立历史误关联；推荐的下一候选 |
| `repeated_future` | 和 `cached_future` 相同参数/计算，但点编码重复六遍 | 用数值一致性检查分离复用收益与结构变化 |

推荐结构不是全场景 dense shared encoder：

```text
4 帧注册后的 source/static 历史证据
        ↓ 仅历史实际占据过的点，保留 source/class/XYZ/visibility
canonical sparse memory + 一次邻域一致性
        ↓ 与已有 V18 history context 融合，小 MLP 编码一次
        ↓ 已有六个 future queries，六个极小 point/time 头
六组 ADD confidence
        ↓ 沿用预测 SE(2)，继承 class，原占据区域保护
六帧输出
```

静态证据单独以 actor=-2 输入，source context 为零，不强行送入动态 motion。
本地实验没有 generation 或 REMOVE。**这不等于决定删除正式模型的这两个能力**，
也不等于证明静态只需要 road/sidewalk；实验保留其他静态类别和完整 Z。

## 模拟数据与检查

- 随机表面、遮挡后的 t0 形状、历史孤立假点/同类误关联、静态多类别与高度。
- 刚体场景和未来两侧交替出现的非刚体反例；未知/观测 free 不成为候选。
- 16 个 TRAIN 场景、8 个不同 seed 的 TEST 场景；各架构同一批 point/horizon 标签，
  默认 384 更新×32 点，普通 BCE，固定 threshold=0.5，无调参或 best checkpoint。
- 旧网络会将同一 column/horizon 的多个 Z 监督点去重编码，不故意重复放大训练成本。
- `cached_future` 与 `repeated_future` 相同权重的概率 allclose 检查。
- 真正执行 forward、BCE、backward、AdamW；检查到已有 source latent 的梯度。
- 当前已经出现的点不由 repair 重写；原 V18 occupied 不覆盖；source ADD 冲突遵循
  source 顺序；静态类别冲突 fail-closed；存在性关闭、旋转/ego 变换、OOB、空集合测试。
- 六帧 planar transport 不平移 world Z。注册使用已知的模拟历史变换，**没有证明真实
  registration 误差下同样有效**。

## 测速边界（不能误读为方法 FPS）

1. `head_and_input_pack`：已有 canonical memory/maps → 模型输入与修复概率。
2. `registered_input_pipeline`：注册后的四帧观测 → 新建 memory/固定邻域 → 模型 →
   六次 SE(2) scatter → 六个 dense 输出副本。
3. `training_cost`：相同 256 point/horizon 对的 forward/BCE/backward/AdamW，**不含 V18
   motion 本身的训练**。也不把它叫正式联合训练速度。

旧输入采样在 canonical 坐标系中进行，省掉真实逐体素 SE3 逆采样；这是偏向旧结构
的简化。所有 timed pipeline 用相同的已接受 ADD 工作量，防止“一个网络不预测任何点
所以渲染更快”这种假收益。质量评估则用每个模型自己的真实概率，不使用这个固定 mask。

计时不含：raw I/O、source extraction、association/registration 估计、V18/Strong/KTA、
generation、真实 GT 指标。canvas 尺寸、source 数、候选数均在 result.json 内。
本地 CPU 的倍数 **不能直接换算成 L40S FPS**；网络使用 FP32，也没有套用旧 CUDA graph。

## 仍需真实数据回答的两个问题

1. epoch19 修复收益中 dynamic/static × ADD/REMOVE 的实际贡献是多少？不能从
   “98.9% 修改是 ADD”推断大部分 mIoU 收益一定来自动态 source。
2. 新的 evidence-only 候选域能覆盖多少 teacher 的有效修复？旧网络还可以修复邻域
   扩展点，union-only 域可能丢失收益；必须把域损失与 head 学习损失分别报告。

本地 `once` 的 GT oracle 反例用于提前显示固定形状上限；history 从未观测过的点也
另外报告不可恢复质量。**不能靠模拟中的低 unseen 比例推断真实数据的 coverage。**

真实数据入口优先比较 teacher 与推荐的 local_consensus；纯 once 已在模拟反例中
暴露时变形状上限，可留作后续结构消融。同一缓存/人口/renderer，同一趟完成
domain oracle、分支贡献、真实六帧延迟和小规模迁移。没有 gate 前不替换
epoch19，不自动重训 full20，不先砍 V18 或生成分支。

## 本地复现

Windows 只使用已验证的项目虚拟环境，绝不执行 Anaconda base：

```powershell
cd F:\Desktop\intern\compression\stochoccdiagnosis\v3\stochocc-diagnosis\swfm-v20utc
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& .\.venv-selector-check\Scripts\python.exe -u tools/real_motion/benchmark_sparse_repair_synthetic.py `
  --out-dir outputs/synthetic_sparse_repair_new_run --steps 384 --repeats 2 --threads 1
```

最终应只引用 `outputs/synthetic_sparse_repair_20261005_causal_final` 的报告。
之前 v1/v2 是构造与输入审计过程的诊断产物，不作为最终方案质量证据。
正式报告记录 Python/Torch/device/threads、实现文件 SHA256、每次计时样本、数据 seed 与所有边界。

## 最终实测（2026-10-05，本地 CPU 单线程 FP32）

最终运行 `causal_final` 共 270.24 秒，384 个训练更新，每种模型用相同采样点与标签。
这里的 IoU 是 held-out 模拟数据的二元修复动作 IoU，**不是 occupancy semantic mIoU**。

| 结构 | 模拟动作 IoU | Precision | Recall | 混合窗口修复六帧流程 |
|---|---:|---:|---:|---:|
| 旧 column196 | 0.9196 | 0.9731 | 0.9436 | 9566.79 ms |
| 旧 column36 | 0.9127 | 0.9609 | 0.9479 | 7099.86 ms |
| source once | 0.8777 | 0.9188 | 0.9516 | 58.51 ms |
| actor gate | 0.8819 | 0.9186 | 0.9567 | 55.89 ms |
| cached future | 0.8796 | 0.9207 | 0.9517 | 58.59 ms |
| local consensus + cached future | **0.9450** | 0.9603 | **0.9835** | **76.70 ms** |

混合窗口包含 20 个动态 source、400 个静态输入列、4212 个 evidence 点；旧路径六帧
共 11754 个列查询，canvas 为 120×64×16。local consensus 修复流程相对旧196约
124.72×；小窗口约75.68×、静态重窗口约144.09×。这是 CPU 结构筛选结果，不是
“L40S 将快 125 倍”。两次计时中位数样本全部保存，短任务/系统负载波动仍在。

同一批256个point/horizon对的真实修复头训练步骤：旧196 **615.51ms**、旧36
**464.77ms**、local consensus **2.87ms**。均观察到 live source 梯度；新头10346参数，
旧头182680参数。**该测量不含 V18 motion forward/loss/backward，也不含训练时在线
候选准备，不能用214×推断正式联合训练加速214×。**

推荐版的优势不是“未来形状固定”，而是将昂贵计算共享一次：邻域证据＋point编码共享，
廉价的每点时间交互保留。旧patch CNN换成MLP而不保留邻域一致性时，模拟IoU落到
0.88左右；补上廉价历史邻域统计后达到0.945。新头Precision略低于旧196，并非每个
指标都提高；仍不能据此保证真实mIoU不降。

刚体模拟里 once 的理论固定形状上限没有损失；非刚体混合例子里固定形状上限已经
低于逐horizon输出。单个actor gate也不能改变同一source两个点的相对置信度排序，
因此不建议将这两个极简版直接作为正式结构。

额外的独立cProfile诊断（非上表稳态测速）确认新结构主要CPU开销来自
`build_memory/searchsorted`、一次邻域lookup和六帧raster，而不是MLP。该诊断cold调用
约113ms，包含不同accepted-mask负载，**不能与上述76.70ms直接相加或当作回退**。
稳态独立阶段测量的memory build约56.7ms、common scatter约28.1ms，阶段样本单独
测量，不能要求它们相加等于pipeline的另一次中位数。

验证：新模拟/安全测试与现有column/shared-evidence回归一起 **73 passed / 2 skipped**；
跳过的CUDA检查未在本地伪造。保留旧实验，原正式模型没有被替换。

结论：**选择 `local_consensus + cached_future` 进入真实数据候选，不选择无邻域MLP、
纯 once 或继续给逐列管线减 token。**下一次真实验证必须同时检查teacher有效修复域
覆盖和静态/动态贡献；本地不能解决这个科学有效性问题。
