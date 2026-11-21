# Refiner 前方静态遮挡训练

从随机初始化训练原版 Forward AdaLN + FPN + U-Net。沿用原始 TRAIN4086 的窗口、全局 B4/source128 分组、1031 次更新、AdamW、余弦学习率和六帧 CE＋Lovász。使用既有一轮模型作为评估控制。该控制为单卡训练，本候选为双卡，比较采用相同样本及预算，GPU 分工有别。

每个训练窗以 scene、t0 和原种子固定抽取。预计约半数保持正常输入，其余在 t0 坐标系遮住前方 8、12 或 16 米。遮挡界面按位姿映射到四帧历史和六帧未来，跨时刻保持在同一个世界位置。Prior 在原静态地图和 Strong 回填上模拟 free 边界。Memory 从遮挡后的静态输入重新编码历史，同时去掉遮挡位置的静态融合权重。原始动态运动和移动物体搬运仍走已有路径。训练目标使用完整未来占据标签。

在原固定 DEV64 上以同一个评估器复测原版和候选，OCC/STC 分别统计。

- 真实输入下的全网格 mIoU、IoU，按 1、2、3 秒报告及平均。
- 真实输入下 3 秒前方、四帧历史 XY 网格并集之外的体素。分别统计道路、人行道、地形、建筑和植被的三维 IoU 与 TP、FP、FN，以及 GT 空位上的占据误报。
- 人为遮挡的 3 秒区域，限定在原 t0 网格内且越过该样本的固定遮挡界面。两模型输入同一人工缺口，统计相同的指标。
- OCC、STC 各固定前四窗的 3 秒俯视对比。图中线条标出对应评分区。

模型与评估分别保存训练更新游标、评估窗口游标和整数混淆矩阵。只写两份滚动训练断点、一份轮末模型权重和小型评估结果。稠密预测及 Memory 逐窗在内存中处理。

在服务器的 `OccFM` 环境和仓库根目录运行以下入口。默认双卡训练、单卡评估。

```bash
bash tools/refiner_masked_completion/workflow.sh
```

`PILOT_GPUS` 和 `PILOT_EVAL_GPU` 指定卡。默认数据及基准目录与 coverage 小实验相同，可用 `DATAROOT`、`TRAIN_CACHE`、`DEV_CACHE`、`STC_ROOT` 和 `MASKED_REFINER_CONTROL` 覆盖路径。可执行 `workflow.sh train` 或 `workflow.sh eval`。单独评估时设置 `MASKED_REFINER_DIR`，续训另设 `MASKED_REFINER_RESUME=1`。

本地验证使用真实训练窗 scene-0429。原生 200×200×16 网格六帧前向有效，3 秒人工缺口包含 76,268 个体素，其中 3,956 个体素的非空 Prior 概率相对原路径下降超过 0.5。运动位移与 yaw 和原路径一致。完整六帧 CE＋Lovász 反传及一次 AdamW 更新通过，Refiner 输出层发生更新，监督标签保持一致。完整训练及 DEV64 指标待运行。
