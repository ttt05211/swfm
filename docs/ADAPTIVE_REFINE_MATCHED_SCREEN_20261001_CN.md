# Refine 自适应因果上下文：一次同预算对照

## 动机与边界

现版 Column 对每个候选读取六帧 7×7 full-Z 局部 patch。新模块不改变候选、归属、KEEP/ADD/REMOVE、渲染器、生成 head 或损失，只在 refine 的局部 query 后添加 gated context。

这不是已证实的涨点方案。尤其 V18 的 FutureQueryBlock 已有六帧 self-attention；新增 temporal readout 只是 refine 的显式整段读取，不是补一个原来不存在的时序模块。新上下文仍来自历史输入及既定未来 ego pose，不允许读取 future occupancy / GT annotation / GT motion。

## 结构

`V18 → 原 hard transport → 原候选及局部 Column encoder → refine query + gated context → 原动作 head → 原 compositor`

- 静态 refine：从既有六帧历史融合静态 memory 构造 stride=4 的八通道 BEV 描述场（道路/人行道密度、各自平均高度、其他静态密度、历史 grid coverage、占据 column 与地面 column 比例）。它是固定证据场，不是新的全分辨率可学习 BEV backbone，也不应宣称完全没有 dense field。每个 horizon 一份，不能每 query 复制。UNKNOWN / 历史 grid 外为零；coverage 不是 lidar visibility。
- 局部 query 预测八个二维 offset，tanh 限制在每轴 ±8m，初始点在半径 4m 圆周。FP32 bilinear lookup，屏蔽原 grid 外、padding 和无历史覆盖点，再加权聚合。grid_sample 的 W/H 是 Y/X，坐标采用 align_corners=False 的 voxel center。
- 动态 refine：只 gather **相同 actor** 的六帧 live `future_transport_queries`，小型 attention 聚合。不跨 source，不读取 GT existence，不 detach。原当前 horizon source projection 保留。不能把这叫做生成未见过的动态 source。
- `q_refine = q_local + sigmoid(gate([q_local,context])) * context`。新增输出 projection 零初始化、gate bias=-2，初始化时包括已有非零输出 head 都与现版逐元素一致。此初始化不保证训练后无退化。
- Generation readout 仍用 `q_local`；联合训练下共享参数与 transport 仍可能间接受新 refine 梯度影响。因此“生成不变”指结构/候选/loss 未改，不是训练完生成预测必然相同。

## 唯一小试预算

沿用一阶段 screen，而不是 full15：scene/time-balanced TRAIN1024，1024 个窗口尝试，单 window/update，约 **一遍数据**（不是 1024 轮）。每窗口最多 256 个 Column 查询，同 seed=20261002、相同 window order、初始化共享参数、query 采样 RNG、原 motion/column losses 与全周期余弦 LR。

两组：现版 local 与 adaptive refine。每组仍有同初始化 V18-only 对照，E14 仅评估参考。最终同一冻结 dev512，dev64 是同次结果子集；固定 dev64 中途监控不选 best。不跑 full4369，不扫 offset、点数、半径、阈值。

阈值各自只在同一 held-out TRAIN64、同一冻结网格上校准，不在 dev 选。报告保留 calibrated 与固定 0.5 的 diagnostic 两套输出。比較工具分别打印：JOINT 绝对性能差、transport 差、refine 相对各自 transport 的增益差，防止把 transport 变化全归给 refine。零修改模型不算成功。

新 checkpoint protocol 为 `p0_f9_joint_adaptive_refine_screen_v1`，与原 screen/full protocol 隔离。原模型构造、默认参数、full15 方法及 resume 不变；full15 不能恢复 adaptive checkpoint。

## 速度与空间

两组顺序运行，同一实验目录复用已有经过完整性检查的历史固定几何缓存（上限 4GiB，RAM 512MiB），不缓存当前模型输出、GT、学习后的特征或概率。CPU 预取固定历史工作；冷 Strong 仍走原设备，不能把 GPU Strong 换成 CPU。

Static atlas 每个 window/horizon 构建一次，存于该 PreparedColumns 的短生命周期固定 evidence cache。此处不保存可学习 encoder 输出，避免参数更新后复用旧特征。每个 horizon 一张小场，不对每个 query 复制大 grid。两个 arm 的 wall time 因 cache cold/warm 不可直接当成网络速度对比；分别报告 stage time、cache 和参数量。

脚本检测到 joint 训练在运行时拒绝启动，不自动停止 full15、不覆盖旧实验。

## 服务器执行

在 OccFM 环境：

```bash
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
conda activate OccFM
bash tools/real_motion/run_p0_f9_adaptive_refine_pair.sh screen
```

一次命令完成测试、两组同预算训练/校准/最终 dev512 和合并报告。发回打印的 `report/combined_summary.txt`。无需额外 oracle / prototype / full eval。

判定：先看 adaptive 相比 local 是否实际提高且通过旧安全 gate，以及非零修改和分 horizon 指标。单遍 screen 只回答“这个预算下是否有改进迹象”，不能回答收敛上限或正式全数据优越性。若失败，不自动重训/扩大规模。
