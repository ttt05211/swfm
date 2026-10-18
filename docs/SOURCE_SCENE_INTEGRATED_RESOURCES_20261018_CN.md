# 最新 KTA Query / Scene–Temporal + AdaGN 资源对比

## 对象与边界

只读刚完成的两组 RANDOM TRAIN4086×1：`tools/source_scene_kta_query` 和 `tools/source_scene_integrated`。不比较旧 Surface、clean1000/norm192 或无 Query 的 Stage2；不重新训练、不评分、不自动启动全量训练。

整合版用户回传：1194146 参数（控制1573314），同64最终 OCC 46.546783/60.318127/Moving31.508839，STC17.075513/27.842079/20.507685；相对控制 mIoU −.031579/−.027393、Moving −.431229/−.134926。支持继续资源验证，不代表已胜出或完整训练稳定持平。

新增独立 `tools/source_scene_integrated_speed`，旧训练/模型/测速源码完全不改，避免破坏原断点的实现哈希。严格只接受完成的同预算两组、原结果绑定的控制 SHA、相同 train/eval keys/known/损失/日程/数据/world_size/执行配置，checkpoint 与结果 contract/cursor/参数量一致；旧源码和输入文件校验。多候选要求显式目录，不按分数/时间选模型。

## 报告什么

- 参数：总量、共享 Encoder、其余 STWM、Refiner。
- 训练：从原 `progress.jsonl` 提取各自正确 event 的真实 GLOBAL-B4 DDP 秒/窗，排前16更新，另列末256更新；按实际窗口数加权。日志游标完整检查；恢复导致的重复 update 保留最后写入，不选最快记录。包含输入等待/梯度归约，不含 checkpoint/评估，不冒充单卡训练速度。
- 推理：一张 GPU、一个模型进程、一次仅一个模型驻留 GPU；两轮顺序控制→整合、整合→控制。GT/STC 同窗交替。默认原 remaining64 的固定前32窗（可显式64），原 BF16/几何精度、H6、相同 CPU4 native majority。
- 每次完整输出六帧 CPU semantics，统计总时间比率得到 output_frame_fps=6×window_fps、六帧均值/P90、allocated/reserved 峰值。额外前4窗分段 profile 不加入 FPS。

计时内：live 原两帧 KTA、fresh Strong/CLEAR、Encoder/STWM/lift、Transport、**无 ICP 的 direct-history memory（含学习历史 yaw）**、原或 AdaGN Refiner、六帧 argmax+D2H。

计时外：磁盘 I/O、历史 CC/共享身份/read-coordinate 构造、初始 H2D、预热、权重上传、串行 profile、日志/结果写盘。没有调用 ICP；不能将“去 ICP 的准备收益”算入此 FPS，不是相机端到端 FPS。每次 F/Strong/未来 prior/memory/预测均重新计算，不跨调用缓存学习输出。没有导入旧 packed/切片/age/support/calibration 后端。

参考为现有 KTA-query 完整前向（不能误用 Stage2 no-query 的 `DirectPredictor.forward`，后者传入 None KTA）。本次只复用其历史 prepare，显式调用 query adapter forward。

原路径自身重复漂移、跨计时入口的六帧 byte 差异各自保留。所有已计时样本保留，不因 FAIL/慢样本筛选；严格 byte FAIL 仍 FAIL，不放宽阈值、不称无损。失败运行只留原始已完成 samples，不假装完整 FPS。

## 服务器命令

测速时不要并行运行其他 GPU 训练或评估。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/shared-history-execution-speed-v2
git merge --ff-only FETCH_HEAD

export SCENE_SPEED_GPU=0
export SCENE_RESOURCE_WINDOWS=32
export SCENE_RESOURCE_DIR="$PWD/outputs/p0_f9_shared_history/scene_integrated_speed_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/source_scene_integrated_speed/run.sh > "${SCENE_RESOURCE_DIR}.log" 2>&1 &
tail -F "${SCENE_RESOURCE_DIR}.log"
```

默认识别唯一完成的 `scene_integrated_1ep_*`、`scene_stage2_kta_query_1ep_*`。如果不唯一，明确设置 `INTEGRATED_DIR` 和 `SCENE_KTA_CONTROL_DIR`；不会静默改模型。所有输出写新 `SCENE_RESOURCE_DIR`，原权重/结果/日志/缓存只读，完成再核原文件 SHA。

完成后：`cat "$SCENE_RESOURCE_DIR/summary.txt"`，原始样本在 `samples.jsonl`。中断后保留旧目录，用新资源目录重测；不将不完整窗口拼成完整两轮 FPS。

## 测完之后

若加载、完整六帧、资源/速度无异常，可进入本次整合架构的完整 TRAIN20430×2、双卡、每2000 global step 保存及同64评估的下一阶段质量验证。不是自动选胜者，速度不一定必须提高才有研究价值；Moving 的下降仍需完整训练关注。不能误运行旧 `tools/source_scene_full` 来训练此结构，该入口对应早期 Stage2 no-query。新全量入口需显式接本次 IntegratedBody，不能通过旧 loader 或旧断点静默切换。

本地 33 项新旧测试通过（22.90 秒），含实际小网格 CUDA 完整两轮/双模态测速入口、非零预测头 CPU 严格同入口输出、无 ICP trap、fresh Strong 每次调用、源文件只读、控制绑定/完整日志/输入库存/多目录拒绝。CLI help 和 Bash 语法检查通过；旧 Query/Integrated/Stage2/原测速源码零 diff。

本地小网格 CPU/CUDA 测试仅验证实现/只读边界，不是 L40S 性能或正式尺寸训练显存结果；没有服务器执行连接，需运行以上命令获得真实资源表。
