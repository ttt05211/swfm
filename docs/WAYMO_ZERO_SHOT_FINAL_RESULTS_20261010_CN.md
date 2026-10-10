# Surface CCR：Waymo 2Hz / 10Hz 完整 zero-shot 结果（2026-10-10）

## 完成状态和主要结论

两套协议均已完成全部官方数据锚点，不再需要恢复未完成的10Hz任务。
固定同一nuScenes干净联合训练的epoch5/6/8/12/14单一平均模型，无Waymo训练、校准、阈值搜索或时间重缩放。
这里归档用户的服务器结果，不是本地重新评估，也不是I²-World模型的复跑分数。

| 协议 | 完成窗口 | 名义评分时距 | Transport IoU | Joint IoU | ΔIoU pp | Transport mIoU | Joint mIoU | ΔmIoU pp |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Hz / stride5 | 7,998 / 7,998 | 1 / 2 / 3s | 60.251916 | **61.209875** | +0.957959 | 51.734899 | **52.152906** | +0.418007 |
| 10Hz / stride1 | 39,987 / 39,987 | 0.2 / 0.4 / 0.6s | 74.762743 | **75.100701** | +0.337958 | **66.111438** | 65.997882 | −0.113556 |

2Hz的CCR在三个报告时距IoU和mIoU均有正收益，支持修复头在该跨域验证人口上的增益。
10Hz的平均occupancy IoU提升，但semantic mIoU轻微下降；最短0.2s下降较明显，0.4s几乎持平，0.6s两项均提高。
不能包装成“两协议全面提升”，也不能拿10Hz约66的mIoU直接证明其优于2Hz约52：预测时距、历史覆盖和锚点人口均不同。

以下数值单位均为%，差值单位为百分点；Avg.为三个horizon指标的算术平均，不是逐窗口平均，也不是合并horizon整数交并之后计算。
本次标准mIoU与I²-World排除恰为零IoU类别的口径数值恰好相同，不代表两定义永远相同。

## 2Hz 完整结果

| 名义时距 | Transport IoU | Joint IoU | ΔIoU pp | Transport mIoU | Joint mIoU | ΔmIoU pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1s | 67.552118 | 68.209418 | +0.657300 | 59.487389 | 59.602591 | +0.115202 |
| 2s | 59.659763 | 60.669601 | +1.009838 | 50.866556 | 51.381103 | +0.514547 |
| 3s | 53.543867 | 54.750606 | +1.206739 | 44.850752 | 45.475025 | +0.624273 |
| Avg. | 60.251916 | 61.209875 | +0.957959 | 51.734899 | 52.152906 | +0.418007 |

全局timestamp排序后stride5，7,998锚点；场景边界重复最近有效历史/未来帧、pose和target，不丢窗。
7,796同场景相邻链接中15个超出0.35–0.65s，最大1.199943s；保留真实pose/timestamp，不重新采样或插值。
1/2/3s是index-based名义时距，实际目标跨度不保证逐窗恰为1/2/3s；本条摘要未提供完整目标跨度统计，不猜测。

## 10Hz 完整结果

| eval_time / 原生未来帧 | 名义时距 | Transport IoU | Joint IoU | ΔIoU pp | Transport mIoU | Joint mIoU | ΔmIoU pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / +2 | 0.2s | 78.856326 | 78.768551 | −0.087775 | 71.631700 | 71.074143 | −0.557557 |
| 3 / +4 | 0.4s | 74.238185 | 74.689365 | +0.451180 | 65.532694 | 65.526547 | −0.006147 |
| 5 / +6 | 0.6s | 71.193718 | 71.844188 | +0.650470 | 61.169922 | 61.392958 | +0.223036 |
| Avg. | 三时距平均 | 74.762743 | 75.100701 | +0.337958 | 66.111438 | 65.997882 | −0.113556 |

固定I²-World公开代码`II-World@661d830f9b34ee03ce368db164a72753ab8764a3`的下标协议：
`load_interval=1`、`eval_metric=miou`、六未来零基下标`eval_time=1/3/5`。
这对应原生未来第2/4/6帧，**不是物理1/2/3秒预测**。不根据配置注释错误地给10Hz表加长时距标签。
一趟六帧预测累计三个独立horizon指标，无需重复三次模型推理。保留训练时0.5s slot clock/嵌入，不重训或插值适配10Hz。

| eval_time | 名义秒数 | 实际目标跨度均值s | 最小s | 最大s | 实际零跨度target数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2 | 0.19893245067146823 | 0.0 | 0.699997 | 202 |
| 3 | 0.4 | 0.3958439374796809 | 0.0 | 1.099955 | 202 |
| 5 | 0.6 | 0.5907344247880562 | 0.0 | 1.299967 | 202 |

39,785同场景相邻原生链接中82个超出0.07–0.13s（约0.2061%）；frame step全部为1。
无重采样、丢窗；实际跨度和场景尾端重复目标已披露，不将名义时距伪装成逐窗固定物理时间。

## 方法和比较边界

- 同一固定均值模型，四张总历史含t0→六未来；weighted ADD raw sigmoid@0.5，REMOVE-off。
- 历史occupancy为官方dense语义输入；不额外读取camera/lidar observation mask，评分不加visibility mask。
- Waymo标签映射到模型的nuScenes标签空间；2Hz摘要明确raw free23，10Hz使用同份数据与既有冻结协议。
- 未来ego poses是显式给定条件，不声称预测未知ego trajectory；未来GT occupancy仅在六个预测完成后读取。
- 不使用Waymo训练集、标签微调、阈值校准，不复用nuScenes几何缓存预测Waymo。
- 本模型四总历史预算与I²-World temporal-tokenizer previous/current预算不可直接等同；复现其数据/评分规则不等于架构或输入预算全同。
- 本协议未评估MovingMacro/MovingMicro，也未测Waymo正式Dense Forecast FPS；不能从多进程eval吞吐填论文FPS列。
- 尚未汇入其他方法严格同协议结果、每场景统计或置信区间；不据此宣布SOTA、所有场景均提升或完全匹配物理时距的优越性。
- 本次只归档，不根据10Hz观察改阈值、重训或筛选新checkpoint。

## 执行验收与恢复记录

10Hz最终目录是`waymo10_parallel4_20261010_055012`，同卡4进程×1工作线程、连续8窗chunk。
此前同64窗×2布局对照：2×2为0.176829s/窗，4×1为0.127069s/窗，报告1.3916×；六帧/概率/运动/整数指标一致性通过。
这些是短窗实际eval吞吐，不是正式单窗口FPS或整个39,987人口的累计速度。
用户日志在接近35,936窗时发生`BrokenProcessPool`；同目录严格resume，最终摘要确认复用35,959个保存窗口，并完成39,987/39,987。
最终invocation worker startup=22.514327936805785s；摘要未提供跨全部invocation累计wall time，不推算或伪造总耗时。
子进程异常退出原因未确认，不擅自记录为CPU内存OOM/GPU OOM。恢复后全人口完成，但原始整数counts/contract尚未下载到本地独立检查。

## 来源、仓库档案与服务器路径

原始分数来源是用户粘贴的服务器摘要；本地未复跑完整Waymo模型。

- [2Hz摘要转录](results/surface_ccr_waymo_2hz_20261010_summary.txt)：来自此前聊天正文，保留服务器摘要，省略终端prompt。
- [10Hz附件原文](results/surface_ccr_waymo_10hz_20261010_summary.txt)：本次附件文本归档，仅规范为LF换行。
- [两协议结构化转录](results/surface_ccr_waymo_2hz_10hz_20261010.json)：含原始展示分数、增益、协议与时间审计；**不是服务器原`waymo_validation.json`**。

服务器原结果（已完成，只读保留，不需重跑）：

```text
2Hz:
/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/waymo_i2world_2hz_20261009_225100_83458/summary.txt

10Hz:
/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/waymo10_parallel4_20261010_055012/summary.txt
```

对应各目录的`waymo_validation.json`、`contract.json`、`state.json`及原前缀快照/progress日志应保留，以便后续复现检查。
实际服务器Git HEAD、均值checkpoint SHA、metadata SHA、完整per-class/counts及跨invocationwall time本条摘要未提供，结构化档案标为未知，不伪造。
附加下一步是收集已有原JSON/contract作复现归档，不重新评估、不覆盖原实验，不修改nuScenes主表或已冻结6秒结果。
