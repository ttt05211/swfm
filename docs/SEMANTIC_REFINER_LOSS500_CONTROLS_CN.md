# Semantic Refiner：500步直接辅助损失对照

目的：判断 source motion 几何监督、existence BCE 是否在目前全网格 CE+Lovász 下仍有帮助。先不做完整收敛、结构消融、Pred 升级或自动删 loss。

## 三路固定比较

| 路径 | CE + Lovász | ego R10 | source motion | existence |
|---|---|---|---|---|
| full | 保留 | 保留 | 保留 | 保留 |
| no_motion | 保留 | 保留 | 删除 | 保留 |
| no_existence | 保留 | 保留 | 保留 | 删除 |

每路相同 RANDOM 起点、seed=20261011、相同 AdamW/初始LR、TRAIN4086 的首500个原分组、全6时距、batch≤4/source≤128。500步统一全周期余弦，不从已经受 motion 监督的 epoch2 权重微调，避免掩盖运动监督从头训练的作用。实际窗口数由原 source budget 分组决定，约2000，不是完整TRAIN20%一轮。原 trainer 的 epoch cursor=1 仅表示这个500步周期结束，结果明确 complete_dataset_epochs=0。

各路完成500步后固定 DEV512 一次，同时报告 Transport/Refiner 的 mIoU、IoU、MovingMicro、六时距 ego 误差。比较冻结的最终步，不能按 dev 选迭代。分数相同/略好不能证明完整收敛后该损失无用；需要关注 Transport 是否退化、Refiner 是否以修复代替正确运动，且最终结论仍需多seed/更长预算。

## motion 实际约束什么

不是新的占据 CE。对每个有监督且标签有效的 source，取其**当前观测**点云中32个固定索引点，以 source 当前中心为原点，分别施加预测与GT的位移/旋转，用 SmoothL1 比较变换后的二维几何。预测位移是 KTA 位移加网络残差。GT 位移延续 V18 的 source/annotation offset 修正；无效 yaw 标签只监督平移。未来GT仅作为训练标签，不作为网络输入，不把未来GT物体形状送入预测。

它直接要求物体未来“在哪里、朝向如何”。占据 CE+Lovász 也可通过连续 Transport 和 aligned memory 更新 XY/yaw，但这是整个场景的间接监督，Refiner 可能用语义残差补偿错误运动，所以不预设 motion 无用。

existence 在当前版本不控制 Transport/Memory 是否写入，仅 BCE 辅助监督 STWM。它仍可能通过共享上下文起正则作用。no_existence 删除该图连接，未使用的 existence 参数不会通过零梯度 AdamW 权重衰减偷偷更新；网络结构保留以便严格比较。raw_source_geometry/raw_existence 仍只读计算、记录用于诊断，不参加被关闭分支的反传，三路额外诊断计算一致。

GT 配置中的 ego 头仍只有 R10 几何监督，GT未来姿态作为场景条件；不能声称此实验已经验证 Pred 配置或占据梯度回传到ego。

## 执行和恢复

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export RUN_DIR="$PWD/outputs/p0_f9_shared_history/loss500_$(date +%Y%m%d_%H%M%S)"
bash tools/shared_history_experiments/loss_controls/run.sh
cat "$RUN_DIR/summary.txt"
```

顺序执行三路，不同时挤占GPU。既有几何缓存只读复用，不建学习特征bank。Ctrl-C保存当前路完成更新/评估窗口；恢复使用同一根目录：

```bash
export RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/loss500_实际时间
LOSS500_RESUME=1 bash tools/shared_history_experiments/loss_controls/run.sh
cat "$RUN_DIR/summary.txt"
```

完成的路不重复，未完成的路恢复其 Adam/RNG/500步余弦和评估整数计数。协议、arm、预算、源码、数据或参数改变会拒绝恢复。原 Refiner 源码/epoch2/训练指纹不修改，新入口放在 sibling loss_controls/。不自动扩大预算、选头、推广或修改原 loss 默认值。

按之前约0.4秒/窗估计，三路纯训练约40分钟，另加三次DEV512和初始化；仅估算，不是本地模拟得到的L40S吞吐。
