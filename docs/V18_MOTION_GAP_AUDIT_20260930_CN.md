# V18 motion gap：一趟 dev64 归因，不改模型

## 目的和结论边界

此前同一 dev64 的 `T0_GT_MOTION` 为 +6.125720 pp mIoU、
+31.333160 pp Moving-Micro。这是 **GT 位置和 yaw 同时替换** 的结果，
不是可以直接实现的收益，更不能据此认定改位置或改 yaw 一项就足够。

这次只拆清该缺口，不重新开展 birth、shape generation、static selector
实验，不训练、不修改 Clean-E14，也不自动启动下一轮实验。
以下新增条件的真实数据结果尚待服务器运行；单元测试不是 nuScenes 实验结果。

## 固定比较：共用一次 forward、一次数据准备

| 条件 | source destination XY | yaw |
| --- | --- | --- |
| V18_BASE | 原 V18 | 原 V18 |
| KTA_XY_ZERO_YAW | 原 KTA anchor | 0 |
| GT_XY_PRED_YAW | GT source destination | 原 V18 |
| PRED_XY_GT_YAW | 原 V18 | GT yaw |
| GT_XY_GT_YAW | GT source destination | GT yaw |
| GT_XY_GT_YAW_SUPERVISED_ONLY | 仅原 `supervised_source` 替换 GT | 同左 |

六个条件共用 frozen dev64 identity/order、原 t0 Strong sources 的几何、
class、顺序、source-centred pivot、Z、原 A1 CLEAR/WRITE 与 frozen metrics。
只报告原 1/2/3s 指标；误差统计同时包含全部六个预测时刻。
不是 V21 的 free-only add compositor，不引入 GT source 或未来 shape。

GT XY 只用于 `se2_target_valid` 的 source-horizon；GT yaw 还要求
`yaw_label_valid` 和冻结 semantic yaw enable rule。
缺失标签保持 V18，**不删除 source、不使用 GT survival、不改变部署 gate**。
`supervised_source` 只限制最后一个诊断条件的干预范围，不改变预测源集合。

source destination 指真实 observed source 中心的 GT 终点，不是 GT box
中心；使用原 `anchors + target_source_residual` 合同。
额外检查 KTA anchor/pivot、displacement/residual 等价性以及实际 Strong
source 与 cache 中心/order。零干预必须逐 voxel 等于正式 V18 forecast。

## 一次运行得到什么

1. 位置、yaw 单独修正和共同修正的 mIoU/Moving-Micro 增益；按 horizon 和 scene 输出。
2. 2×2 交互 `joint - xy_only - yaw_only`；两个主效应不能直接相加。
   Shapley 值只作为该四条件下的描述性分摊，不是物理独立因果或可学习收益证明。
3. 原训练监督范围能覆盖多少 joint headroom；额外未监督 source 的增量。
4. source-centre 误差、yaw 误差、KTA/zero-yaw 对照；按 class、历史有效帧数、
   原监督状态、GT 转弯幅度分层。GT 分层仅在诊断中使用。
5. 速度/加速度误差，只对连续有效 GT 区间计算，绝不对缺失 GT 零填充求导。
   acceleration 大不等于模型必然错误，需与对应 GT acceleration 比较。

这些是 overlapping windows 中 source-horizon **occurrences** 的描述统计，
不是独立物体样本，也不进行未经聚类处理的显著性检验。
不会因发现某个分层差就自动增设 gate、loss 或重训。

如果旧 source-evidence audit 仍在，会核对 checkpoint SHA、selection
fingerprint、配置 SHA/overrides，并要求 baseline 和联合 oracle 四项主指标
在 `1e-8` 内复现。默认旧文件已删时明确标记 `checked=false`；用户显式指定
但不存在的 reference 路径则报错。

## 方法主线，不伪装生成能力

合理主线是 **历史证据保持 source 身份 → 时空特征推断运动 → source-centred
SE(2) 搬运保留形状 → 组合未来 occupancy**。
核心研究问题是在几何来源固定的条件下，如何学习更准确的联合运动过程，
而不是把一个失败的静态 selector 或象征性 birth token 拼回主方法。

V18 的 future-query decoder 已有 query self-attention，因此不能把
“加未来时序注意力”包装成它没有的模块。若诊断确实定位到联合动力学，
再设计有实质区别的运动参数化/监督；不能仅凭这个 oracle 承诺有效。

这种主线只覆盖 observed sources。若论文仍将 never-seen/new-source
生成作为核心贡献，单靠运动改进并不能使那个故事完整，需明确收缩声明。

## 服务器入口

先在 `/root/nas/occ/swfm` 更新当前 `feature/v22-causal-emergence-tokens`，
激活 `OccFM`，再以子进程运行，**不要 source**：

```bash
bash tools/real_motion/run_p0_f9_v18_motion_gap_dev64.sh
```

runner 使用已确认的 Clean-E14、full4369 V18 cache、冻结 dev64 manifest、
nuScenes dataroot/info、tracked runtime config；不重建 cache/prototype bank。
默认 8 个 CPU IO/support workers，每个 Torch/BLAS worker 单线程，CUDA 0。
所有条件复用数据和 V18 outputs；输出仅 JSON/JSONL/日志，不保存 dense
predictions 或 checkpoints。每个窗口打印进度，避免长时间无反馈。

新输出目录：`outputs/p0_f9_v18_motion_gap/dev64_<time>_<sha>/`。
返回其中的 `summary.txt` 即可；`result/audit.json` 保存完整明细。
可通过 `V18_MOTION_CPU_WORKERS` 调整 workers、`V18_MOTION_OUT` 指定全新
输出目录、`V18_MOTION_REFERENCE_AUDIT` 指定实际旧 audit 路径。
已有输出目录会拒绝覆盖。默认 reference 的真实路径来自用户提供的 provenance：

```text
/root/nas/occ/swfm/outputs/p0_f9_source_evidence_audit/dev64_20260930_122107_575150d/audit.json
```

本地新增测试覆盖干预隔离、缺失 GT、pedestrian yaw rule、pivot/target
合同、无输入 mutation、空 source、连续性 mask、完美预测、非独立交互、
bf16 outputs，以及实际 renderer/metric/audit CLI/reference guard 的端到端
synthetic 路径。真实数据 IO 和 teacher 在该端到端测试中是替身，
不能据此宣称真实 dev64 数值已验证。
