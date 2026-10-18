# 上一版 dense/shared-pinned epoch2：同64窗运动贡献

独立 `tools/source_motion_baseline_audit/run.sh`，只诊断原始 SharedTransport/ICP + SemanticRefiner 的 dense-valid/shared-pinned 全 TRAIN 两轮 epoch2/update10292，不加载第一阶段 Reader、Stage2 或更早 clean1000/norm192。没有新训练、optimizer、参数合并、阈值或模式推广。原四个干预与标准全网格/框条件位移统计复用已验证的诊断定义。

## 干预与可比边界

- learned 路线直接调用原 `shared_source_track.adapter.forward_conditioned`，fresh Strong/CLEAR，六帧读出；额外重复原路径报告 byte 漂移，不豁免或筛掉窗口。
- KTA_future：只把未来 XY/yaw residual 清零；XY_only/yaw_only：分别保留一种未来修正。
- F、lifted history、H/Q、KTA→Query 条件、ICP/registration_valid、source 身份和背景固定。未来运动改变后同时重建几何 prior 与配准历史的 feature memory。
- 旧模型没有学习的历史 yaw，该项显示 NA，不伪造为0°；原 ICP 保留。Stage2 比较中其历史 learned yaw 也原样保留。
- 四条干预×两模态先完成所有六帧预测，再用 GT 未来占据、moving support 和标注框打分/分组。GT future ego 是明确条件，非导航输入或 Pred 实验。
- 标准1/2/3s全网格、逐类/区域混淆矩阵、source位移误差与correct/damage均存入JSON。框归属/碎片/低速分组限制与原诊断相同；不是完美点运动GT。

## 固定对照与契约

读取已完成的 Stage2 诊断 `stage2_motion_audit_20261018_144119/results.json`，不重跑第二阶段。核对64窗顺序、四干预、输入文件统计、六份nuScenes元数据、Torch/CUDA/GPU与CPU线程预算相同；任一不符拒绝比较。原checkpoint自身训练协议/人口/架构和代码/数据哈希也必须通过原loader检查。这里只比较同人口，不承诺原路径逐体素完全一致。

自动定位仅允许一个 `dense_valid_shared_full2_ddp_*/epoch_0002.pt`；多份需显式设 `BASELINE_MOTION_CKPT` 或 `DENSE_DDP_DIR`，不按指标/mtime静默选择。输出目录不能放进源训练目录。旧源码一律不改，避免破坏已完成 Stage2 audit 的 fingerprint 或续跑。

## 运行与查看

```bash
export BASELINE_MOTION_GPU=0
export STAGE2_AUDIT_DIR="$PWD/outputs/p0_f9_shared_history/stage2_motion_audit_20261018_144119"
export BASELINE_MOTION_DIR="$PWD/outputs/p0_f9_shared_history/baseline_motion_audit_$(date +%Y%m%d_%H%M%S)"
export BASELINE_MOTION_RESUME=0
nohup bash tools/source_motion_baseline_audit/run.sh > "${BASELINE_MOTION_DIR}.log" 2>&1 &
tail -F "${BASELINE_MOTION_DIR}.log"
```

完成后 `cat "$BASELINE_MOTION_DIR/comparison.txt"`：只有12行核心数据，分别列出OCC/STC、Transport/Refiner、mIoU/IoU/MovingMicro的旧KTA/旧learned/旧增益与新KTA/新learned/新增益。先看两版KTA Transport底稿是否一致，再比较运动修正收益，不把训练期间涨分冒充相对KTA的收益。

`summary.txt`含旧版精简诊断和对照；`results.json`与`state.json`保存完整统计。每8个完整窗口持久化，SIGINT/SIGTERM在完整窗后保存，续跑同目录设置 `BASELINE_MOTION_RESUME=1`。不读取逐窗旧日志来拼分，也不改源结果。仅报告完整同64窗对照，运行中不把旧版部分窗口与新版完整64窗当成可比结果。

这不是 KTA Query 单变量实验：旧/新还同时涉及 reader、时序结构、历史配准和有效性等变化。结果只能先回答“旧版未来运动修正有没有用、其收益是否在新版保留”，不能直接宣布原因或恢复/推广某个模块。该DEV已反复查看，不是独立测试或FPS。
