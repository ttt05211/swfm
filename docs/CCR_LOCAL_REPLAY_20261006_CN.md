# CCR：共享源坐标证据、时序条件化修复的本地实验

本文件记录实现合同；实际结果由独立输出目录的 `pilot.json` / `summary.txt` 给出。
不是旧 checkpoint 的等价执行后端，不自动部署，不改服务器训练，不宣称任何精度保证。

## 方法

```text
4 帧 occupancy / visibility / ego pose
                │
       已有因果 source 关联与注册
                │
     dynamic source-local / static t0-reference evidence
                │
  一次局部几何编码：语义、完整 XYZ、四帧可见性、邻域、age
                │                     V18 history source context
                ├─────────────────────────────┘
                ↓
   六个时间 / future query / 当前组合状态的轻量 readout
                ↓
          六组 masked ADD / REMOVE
                ↓
     原 SE(2) / ego 投影 + visible-owner fallback 组合
                ↓
             六帧 dense occupancy
```

共享的是历史编码，不强制六帧修复形状相同。动态和静态共享主要 encoder/readout，
但有明确 entity/class embedding，动态保留 live source context 和六个 future queries。
这是源坐标几何与时序修复的因子化，不把 C++/缓存当论文方法。

初版点编码 MLP 为 128→64，readout 为 64→32→2；四帧语义分别嵌入。
完整高度、source-local XYZ、presence/visibility/inside、六面邻域和逐帧邻域密度均保留。
不建立完整 scene learned field，不使用六个逐列 patch reader。

## 因果与几何合同

- 当前所有 t0 sources 都保留；没有 past association 的 source 不被悄悄删掉。
- past dynamic evidence 只用已接受因果注册且观测有效的 occupancy。
- static 限定历史观测 road(11)/sidewalk(13)，不扩大到全静态类别。
- 当前与历史 union 外加一层六面 halo，不预测从未观测的任意新动态实体。
- 历史落在 t0 网格外的点不裁剪；未来进入网格仍可用。
- entity 内局部整数 lattice 聚合、查邻域，避免全局 `(actor,class,xyz)` 排序。
- 临时 dense lattice 过大则使用 entity-local scalar keys，不截断 source/candidates。
- quantization 只做 lookup。原始注册 world points 保留，t0 integer IDs 不通过浮点 floor 回收。
- 不缓存 learned features、当前预测 trajectory、owner、GT labels 跨训练更新。
- 六帧投影使用现有 planar yaw/XY、world Z 保持、future ego SE(3)。
- ADD 只能写原 baseline-free voxel；静态跨类别冲突 fail-closed。
- REMOVE 只能改原可见 owner。动态恢复底层 source/background，不全局清空。
- static REMOVE 必须是无动态 owner 的本类别位置。
- 空动作/全部 KEEP 与 V18 六帧 baseline 逐 voxel 完全相同。

## 监督与训练

原附件的“GT-motion 下六帧 canonical 标签”没有唯一冲突归并规则；此处不做 union/vote。
直接对各 horizon 的实际预测投影分别构造 edit-outcome 标签：

- ADD 正例：合法 baseline-free，GT 等于候选继承语义。
- REMOVE 正例：合法 visible owner，恢复后的 fallback 等于 GT，且不同于 baseline。
- GT unknown 不作为 free/负例。
- GT 只进入 `repair_targets` 和指标；未来 annotation/GT pose 不进入证据或候选。

这会使监督与部署的实际操作一致，而不是仅在 GT 轨迹下修好形状。
motion 仍由原 motion loss 监督；离散 registration / rasterization 不假装可微。
live context 和 future queries 保留修复损失回传路径。

ADD/REMOVE 两个 masked BCE，static/dynamic 分别在合法支持域归一化。
TRAIN-only role/action 正负频率决定有界 `sqrt(neg/pos)` 正例权重；推理减 `log(weight)`
纠正目标先验倾斜。此修正不是一般神经网络校准保证，不做 DEV 阈值搜索。
训练采样保留 inverse-probability 权重，不用平衡采样冒充自然 prior。
REMOVE 相对权重 0.25 是本次固定实验超参数，不是已证明最优的正式合同。
阈值固定 ADD=0.5 / REMOVE=0.95。没有 teacher logits、KD、AE 预训练。

## 一趟本地验收与真实边界

用户提供的 replay：TRAIN16（12常规+4压力）mini-fit；DEV12常规与4压力单独报告。
64 passes / 1024 updates，仅训练新头，epoch19 motion 冻结；最终固定预算权重，不选 DEV best。
这是 developer screen，不是完整训练集或独立验证；不能证明最终收敛上限。

同一入口完成：

1. 构建完整支持域、empty/KEEP exactness、旧 head 正确 ADD/REMOVE coverage。
2. GT-support oracle，检查仅支持域能保留多少修复收益。
3. GT-only mini-fit、DEV12与压力评估、static/dynamic 分解。
4. old/CCR 同卡交错预热、两次完整六帧推理；常规/压力样本分别报告。
5. 新旧克隆 actual motion+head backward/AdamW，八个同窗口 batch4。
6. 独立分段同步计时、CPU profile，不混入正常 FPS。
7. 原 checkpoint/config SHA 再验证。所有科学原文件保持不变。

六帧 FPS = 6 / mean(six-frame latency)，不是平均各样本 FPS。
计时含 fresh Strong/KTA prior、live V18 motion、fresh canonical union/visibility/neighbours、
六帧 readout/projection/ownership/dense output。
排除磁盘 I/O、最初 source extraction/registration、GT/metrics、warmup/hash；
明确不是 raw-input E2E FPS。3050 结果不能冒充 L40S。

mini-fit 可以缓存因果数据用于小样本拟合，但该时间不是完整训练速度。
actual joint probe 每步重新构造支持域、标签和特征，并真正更新可丢弃克隆。
新旧任务/采样单位不同，数值不等价，测速只用于判断该实现是否有实际潜力。
本轮不保留旧 frontier GEN 分支：其功能改成历史证据/halo completion，scope 改变单列声明。

## 本地入口

使用已验证项目环境（不是 Anaconda base）：

```powershell
& ./tools/real_motion/run_local_cuda.ps1 -PythonArgs @(
  'tools/real_motion/pilot_p0_f9_canonical_causal_repair.py',
  '--bundle', 'D:/tian/Documents/replay.zip',
  '--out-dir', 'outputs/ccr_local_pilot_20261006_v1', '--passes', '64'
)
```

该数据路径是用户明确提供的本机路径。目录必须全新，不覆盖旧结果。
Codex 的 Windows CUDA 命令按 `windows-python-env-guard` 使用沙盒外已验证 wrapper。

## 本轮执行优化

- 同一 canonical 点仅编码一次，六帧分别读取时序条件，允许不同 ADD/REMOVE。
- source context / future query 投影按 actor/horizon 共享，不在每个点上重复计算。
- 六帧 ego 投影和 owner/fallback gather 对整个人口批量处理；source 的 float64
  SE(2) 算术仍保持原实现。全部 32 个真实 replay 窗口与逐 entity 参考映射逐元素一致。
- TRAIN 仍构造完整候选域、六帧合法性和 GT 标签；只为已抽中的训练点实例化网络输入。
  与 eager 版本的 features/labels 逐元素一致。推理仍读取完整候选人口，不截断。
- 几何邻域使用有界 entity-local 整数索引；大跨度实体走 sparse fallback。
  不持久化 learned features，也不缓存当前预测运动或监督。

## 实测结果：2026-10-06

实际设备为 RTX 3050 Laptop GPU。TRAIN16 × 64 passes / 1024 updates，
新头随机初始化、旧 epoch19 motion 冻结。最终固定预算权重在 DEV12 上评估，
另有 4 个压力窗口；没有从 DEV 选择 checkpoint。该小样本结果不能代表服务器完整训练。

| DEV12 指标 | 原 Local joint | CCR | 差值（pp） |
| --- | ---: | ---: | ---: |
| mIoU | 40.577554 | 39.738593 | -0.838961 |
| MovingMicro | 31.846816 | 30.706335 | -1.140480 |

CCR 相对相同 frozen transport 的 mIoU 增益仅约 +0.0883 pp；旧头约 +0.9273 pp。
CCR 实际发生 ADD/REMOVE，并非仅输出 KEEP。历史支持对旧头正确静态/动态 ADD
的覆盖分别约 91.1% / 75.2%，说明支持不为空，但不能由此认定预测已学会。
旧 frontier GEN 的正确 ADD 仅覆盖约 0.76%；本版本不是完整生成能力的等价替代。

六帧计时为同卡交错测试，4 个分层窗口，每个模式重复 2 次。FPS 使用
`6 / mean(six-frame latency)`，不是对逐窗口 FPS 取平均。

| 六帧推理人口 | 原 Local joint 延迟 | CCR 延迟 | 延迟比 | CCR FPS |
| --- | ---: | ---: | ---: | ---: |
| 常规（2 窗口） | 2.588732 s | 0.339696 s | 7.6207× | 17.6628 |
| 高 source 压力（2 窗口） | 3.693157 s | 0.443193 s | 8.3331× | 13.5381 |

这包含候选/邻域构造、live motion、六帧 heads 和 dense composition，但排除最初
source extraction/registration、磁盘、GT/metrics。人口很小，任务 scope 也不同；
只能证明当前实现的局部推理成本降低，不能宣称等质量提速或推算 L40S FPS。

真实联合反传 probe 使用相同 8 个不同 TRAIN 常规窗口、batch4，包含 live motion、
完整支持域/标签、采样、head、backward、clip 和 AdamW；不保存科学训练更新。

| 实际联合训练 | 秒/窗口 | peak allocated |
| --- | ---: | ---: |
| 原 Local joint | 0.133184 | 1220.69 MiB |
| CCR | 0.152301 | 854.64 MiB |

CCR 仍慢约 14.4%，不宣称训练提速。只实例化抽样点特征的优化将 CCR 从最初
0.183007 降至 0.152301 秒/窗口（约降低 16.8%，短程测速仍有波动）。
当前支持域/投影/监督准备为 0.099770 秒/窗口，占新训练步约 65.5%；
采样与 head/loss 为 0.022838，反传/clip/optimizer 为 0.024960。
缓存 mini-fit 的 20.75 秒不是在线联合训练耗时，不用于提速结论。

验收：74 tests passed，包含实际 CUDA replay、GT 隔离、六帧 KEEP exactness、
owner/fallback、逐帧不同输出、source/query 梯度与 lazy feature 精确性。
原 checkpoint/config 的 SHA 在诊断后复核不变。

结论：推理 ≥3× 的测速门槛通过，但 mIoU / MovingMicro 保留门槛均失败。
`candidate.pt` 为本地失败候选、不可部署；不自动服务器训练、重训或调整阈值。
这不能证明整个 CCR 思路无效，但当前轻量点级修复头没有保留旧头的有效修复能力。
若继续，必须同时处理表达/学习与在线监督几何成本，不能仅凭 FPS 提升扩大训练。

本地报告：`outputs/ccr_local_pilot_20261006_v1/summary.txt` 和 `pilot.json`。

训练侧后续已经实现“因果先采样、后实时投影与标注”和共享残差邻域层。
新的 RAM/磁盘反传测速有改善，但新头仍未满足小幅质量损失门槛。
这不修改上述 v1 数字；后续全部对照见 `CCR_SPATIAL_TRAINING_REVISION_20261006_CN.md`。
