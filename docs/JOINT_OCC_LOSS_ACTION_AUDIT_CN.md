# 新占据试训：WM / CCR 一趟集中诊断

## 背景，不混训练预算

用户提供的新 GT-only 试训结果：随机初始化，TRAIN20% / 4086 窗 ×3轮，3088更新，DEV512。

| 硬输出 | mIoU | IoU | MovingMicro |
| --- | ---: | ---: | ---: |
| Transport | 38.472382 | 50.435229 | 26.235745 |
| ADD-only | 37.889726 | 50.679728 | 28.263353 |
| ADD+REMOVE | 37.908709 | 50.697061 | 28.136075 |

ADD 对 mIoU 为 −0.582656pp、IoU +0.244499pp；REMOVE 增量 +0.018983 / +0.017333pp，MovingMicro −0.127278pp。
不是“完全没有作用”，但新 CCR 对最终 mIoU 净贡献为负。
之前完整 TRAIN×20 的固定平均在同 DEV512 上 CCR 增益 +1.069630pp；旧动作监督/正权重和新最终占据监督、预算都不同，不能据此宣称损失已被证明有害。
上述为用户摘要转录；没有假装读取服务器原始累计计数。

## 已确认的实现边界

- 新头的两通道都保留 sigmoid，执行后端不会清零 REMOVE。
- REMOVE 只针对原 Transport 基线的合法 owner，恢复固定下层语义；不能任意改类，也不能撤销同次 ADD 新写的错误占据。
- 训练是连续概率编辑/插值，推理固定 ADD0.5 / REMOVE0.95；没有直接动作 BCE 或拟合正权重。动作值未证明校准。
- WM 同时接收最终全网格 CE+Lovász 和直接 source SE(2)/existence 监督；GT 配置的 ego 辅助只更新独立 ego 头。
- 新试训存在软硬差距；此前 FIRST8 的 +1.697028pp 不能全部归给 CCR、不能代表 DEV512。

## 这次只做什么

独立 `tools/joint_occ_experiments/diagnostics/`，不改旧 trainer/model/缓存指纹或训练恢复契约。

1. 从原优化 TRAIN 子集固定 scene-balanced 选16窗，保持原 batch/source 分组。逐窗口训练模式一次前向，全六未来、原归一化和权重。分离最终占据梯度与 source 几何+existence 梯度，报告 WM 全体/共享历史/decoder/XY/yaw/existence 参数的范数和余弦。
2. 原 DEV64 同人口报告硬/软 Transport、ADD-only、ADD+REMOVE 六路；硬输出用原 chunk/graph/BF16 路径，软读出用训练连续 context。仅评分1/2/3s，梯度检查仍覆盖全部六帧。
3. 固定阈值的 ADD/REMOVE 合法行数、唯一体素、实际触发、可纠正覆盖率、correcting/damaging/wrong-to-wrong 分数直方图，以及实际改对/损坏。直方图是候选行加权、包含重复；真实编辑和候选覆盖是每窗/时距唯一体素数，累计跨窗/时距求和。
4. 模型/Adam/RNG/原数据、原缓存只读；不创建优化器、不 clip、不 step、不做阈值选择或 GT oracle 部署。不重建 bank/缓存，不自动追加对照训练。

梯度统计是原权重下的局部、一阶、pre-clip 诊断。零范数的 cosine/ratio 为 null。平均 batch cosine 不是整个 TRAIN 的聚合梯度，也不能证明哪种 loss 泛化更好。
诊断专用 RNG 按固定 seed/batch 设置并保留调用者 RNG；不是重放源训练的历史随机轨迹。
未来标签用于监督/审计，不进入候选或预测；Moving 支持仍按原 t0 评估口径。

## 服务器命令

更新到含本入口的代码后：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
bash tools/joint_occ_experiments/diagnostics/run.sh
```

自动在 `outputs/p0_f9_joint_occ_ego/gt3_*/last.pt` 中查找已完成的 GT-only 三轮断点，打印实际 source/update/epoch。
不会按修改时间误选部分断点或 paired 模型。若有多个完整任务，优先更新数更大、再较新保存的文件；明确指定优先：

```bash
DIAG_CHECKPOINT=/实际/已完成GT目录/last.pt \
bash tools/joint_occ_experiments/diagnostics/run.sh
```

输出新 `ccr_loss_audit_日期时间`，最终只需发 `summary.txt`。
源文件快照使用一次打开的 inode；即使另一个训练进程原子发布新断点，本诊断固定 snapshot，不修改源训练。
Ctrl-C 在完整 TRAIN batch / DEV window 边界保存；继续同一个诊断而非训练：

```bash
DIAG_OUT=/实际/ccr_loss_audit_目录 DIAG_RESUME=1 \
bash tools/joint_occ_experiments/diagnostics/run.sh
```

恢复固定原 snapshot，不重新选择较新模型；源代码、数据、人口、权重或阈值改变拒绝混拼。
详细整数计数与梯度在 `evaluation.json` / `state.json`。wall time 不是正式 FPS。
本地没有完整 nuScenes/L40S 数据，只验证真实小网格计算、梯度与只读恢复，服务器需运行上述一趟。
