# 最终论文长时预测指标：Surface CCR + Static Carry（2026-10-09）

## 冻结结论

用户已确认将本次完整验证集的 **static_carry** 结果作为最终论文 **nuScenes 1–6s 长时预测实验**指标。
保留原 reconciled rollout 为 baseline 消融，不加入 `se2_carry` 或 `combined`，不再为本结果训练或调阈值。
这是对现有服务器实验结果的归档，不是本地重新运行的评估。

完整人口为 **2,569 个窗口、150 个场景**，由固定 GenieDrive 官方公开 metadata 的 4历史+20未来定义起点；
本模型仅预测12张未来 occupancy（名义2Hz，0.5–6s）。不能把本表称为4369窗口主表或独立测试集结果。

以下均为 **标准 IoU / mIoU，单位 %**。各时距先累计全人口整数交并计数，再计算指标；
Avg. 是指定时距指标的算术平均，不是逐窗口平均，也不是把所有时距交并计数合并后计算。
标准 mIoU 保留 union 有效但 IoU 为零的类别；GenieDrive 公开代码的零类排除/round2兼容口径另存服务器JSON，未用于替换本表。

## 最终论文表（保留两位小数）

| 方法 | IoU 4s | IoU 5s | IoU 6s | IoU Avg. | mIoU 4s | mIoU 5s | mIoU 6s | mIoU Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Surface CCR / reconciled baseline | 42.03 | 38.33 | 35.32 | 38.56 | 30.57 | 27.10 | 24.49 | 27.39 |
| **Surface CCR + Static Carry（最终）** | **44.77** | **41.08** | **37.80** | **41.22** | **32.35** | **28.88** | **26.13** | **29.12** |
| 增益（百分点） | +2.74 | +2.75 | +2.48 | **+2.66** | +1.78 | +1.78 | +1.64 | **+1.73** |

增益先按原始未展示精度计算，再四舍五入，不能先减两位小数的表格值。

## 全时距精度存档

第一段六张预测由两路共享；1/2/3s 输出不变。

| 名义时距 | Baseline IoU | Static Carry IoU | Baseline mIoU | Static Carry mIoU |
| --- | ---: | ---: | ---: | ---: |
| 1s | 60.275510 | 60.275510 | 52.882357 | 52.882357 |
| 2s | 54.248655 | 54.248655 | 43.136524 | 43.136524 |
| 3s | 49.940189 | 49.940189 | 37.103890 | 37.103890 |
| 4s | 42.032229 | 44.768673 | 30.572013 | 32.350774 |
| 5s | 38.331931 | 41.083833 | 27.095679 | 28.876602 |
| 6s | 35.324381 | 37.801952 | 24.488694 | 26.126216 |
| Avg. 1/2/3s | 54.821451 | 54.821451 | 44.374257 | 44.374257 |
| Avg. 4/5/6s | 38.562847 | **41.218153** | 27.385462 | **29.117864** |

最终 avg4–6 相对 baseline：**IoU +2.655306pp，mIoU +1.732402pp**。
原始摘要的最后一位可能因隐藏小数而与六位显示数直接相减差1e-6，应保留原摘要增益。

## 冻结方法与条件

- 模型：干净随机联合训练的 V18 Transport + Surface-aware CCR；固定 epoch **5/6/8/12/14** 可学习参数等权平均。
  推理时运行一个平均网络，不是五网络 ensemble；没有新的6秒训练、权重更新或阈值校准。
- 每段4张历史（含当前帧）预测6张未来；第二段的预测历史取第一段的1.5/2/2.5/3s输出。
- `static_carry` 保留原始观测 t0 静态几何，直接投影到第二段未来网格，避免中间3s网格的重复体素化；
  同时保留第一段末帧CCR自预测的静态新增。第二段仍使用原 motion forward 和 reconciled 身份/速度接续。
- ADD 使用 weighted raw sigmoid@0.5；REMOVE-off；不修改学习模型或阈值。
- 初始仅四张真实历史；不输入未来 GT occupancy、mask、物体 annotation、GT物体运动或GT身份。
  未来 occupancy 仅在预测完成后用于计分。**未来6秒 GT ego poses 是显式预测条件**，不能称未知自车轨迹预测。
- 接续是 **stateful causal rollout**，携带原始静态几何及自预测状态；不能称仅依赖最后四张dense grid的无状态递推。
- 静态接续可能替换/删除第二段旧静态背景，因此属于改变输出的推理方法，而不是此前“逐字节无损执行提速”。
  动态搬运前景在背景替换时受保护，但不能据此保证最终 Moving 指标完全不变。

## 选择来源、指标范围与比较边界

四路 TRAIN64+dev64 screen 的原规则要求平均 MovingMicro 不下降；static_carry 当时下降约0.02pp，**原规则未通过**。
用户在查看screen后明确接受小幅下降，固定static_carry并授权本次全集复测；选择是研究阶段的显式用户决定，
不是原预设TRAIN gate通过、预注册选择或独立测试选择。旧screen报告和门槛保持不变。

本次服务器结果明确记录：

```text
selection_policy: explicit_user_static_carry_accepts_screen_small_Moving_decline_NOT_TRAIN_gate_pass
metric_scope: IoU_mIoU_only
```

**本次全集未评估 MovingMacro/MovingMicro，不借用64窗口的变化作为全集 Moving 数字。**
也没有重新测本接续方法的长时预测FPS；此前54.945 FPS的六帧执行边界不能冒充本次12帧stateful端到端速度。

该人口对齐 GenieDrive 官方公开代码及固定 metadata，但论文 Table 2 的实际起点集合尚未独立确认。
可给出“公开报告数值比较”，不能据此直接宣布严格同协议的 SOTA。
GenieDrive v1 [Table 2](https://arxiv.org/html/2512.12751v1#S4.T2) 报告的4/5/6s：

| 方法/来源 | IoU 4s | IoU 5s | IoU 6s | IoU Avg. | mIoU 4s | mIoU 5s | mIoU 6s | mIoU Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GenieDrive（论文公开值，非本仓库复跑） | 42.81 | 39.00 | 35.60 | 39.14 | 31.16 | 27.17 | 23.66 | 27.33 |
| 本方法（本次服务器标准指标） | 44.77 | 41.08 | 37.80 | 41.22 | 32.35 | 28.88 | 26.13 | 29.12 |

数值上本方法平均 IoU 高约2.08pp、mIoU高约1.79pp；这不消除上面的协议/状态接口比较边界。

## 不替换原1–3s主表

原 **full4369 / 150场景** 的3秒主结果继续使用此前冻结均值评估：IoU **55.211054**、mIoU **44.153241**。
本次长序列人口的1–3s平均为54.821451 / 44.374257；由于人口不同，不能替换原主表，
也不能声称静态接续把1–3s指标提高到本表数值。

## 来源与复现

本档案数据来源：用户于2026-10-09粘贴的完整服务器 `FROZEN SURFACE STATIC CARRY / FULL 1--6s` 摘要。
原文保存在 [原始摘要](results/surface_static_carry_long6s_20261009_summary.txt)，
结构化转录保存在 [JSON记录](results/surface_static_carry_long6s_20261009.json)。JSON是摘要转录，**不是服务器原 evaluation.json**。

入口实现至少包含 `af6d68d`；该提交加入显式用户选择、IoU/mIoU-only、两路共享第一段和避免未用SE2计算。
实际服务器运行HEAD、输出目录、snapshot SHA、完整counts/key-order/contract/时间审计未在本条摘要中提供，
本地未下载或独立校验这些文件，记录为未知，不猜测具体路径或伪造指纹。
服务器原 `evaluation.json`、`contract.json`、`bundle.json`、`timestamp_audit.json`、`checkpoint_snapshot.pt` 应保留作复现依据；
补充归档它们不要求再跑实验。

固定公开 metadata：`world-nuscenes_infos_val.pkl`，HF revision `17e37acfff5b10517393a669ecf471f75f34d43f`，
SHA256 `0426072260a908260625c6dd91b9f06919726f5265c10849b87156d282547ded`；
参考官方代码 `da48a529ffbe14136688e9b7a56f5d1061c366c5`。

复现参数（新输出目录，不能覆盖或混续旧实验）：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export SURFACE_CARRY_APPROVED_ROUTE=static_carry
export SURFACE_CARRY_METRICS_ONLY=1
export CUDA_VISIBLE_DEVICES=0
OUT="$PWD/outputs/p0_f9_joint_surface_ccr/static_carry_all_$(date +%Y%m%d_%H%M%S)"
bash tools/real_motion/run_p0_f9_surface_geometry_carry.sh all "$OUT"
```

本次记录不修改训练、权重、阈值、默认旧推理入口或已完成服务器实验。
