# 当前任务摘要（2026-10-09）

## 当前目标

Clean Joint Surface CCR 已完整随机联合训练 20 轮。现在比较优秀单轮和固定权重平均，选出下一次质量验证候选；不重启失败结构、不继续堆修正器。
用户同意按 DEV64 Joint mIoU top5，第 5/6/8/12/14 轮整网等权平均，并与第 6/8/12 轮在 DEV512 一次性对照。
第 20 轮复用已经完成的 DEV512 结果。不要额外试“末五轮平均”或扫描融合系数。
DEV512 对照已完成，用户现在明确批准只评估固定平均模型的 full4369；不重跑其他单轮、不重新平均、不自动增加 FPS/训练。

## 核心决策与约束

- 4 历史→6 未来。固定 weighted ADD raw sigmoid@0.5，REMOVE-off；没有 KD/AE。
- Surface 描述进入 CCR 编码，实时投影高度/相位进入静态条件读出；不是外挂后处理修正器。
- 正式模型应运动/静态/动态一起干净随机初始化训练；当前 20 轮正是这一版，不能混称早先冻结验证。
- 原 checkpoint、optimizer/RNG、缓存、实验目录保留。平均文件只用于评估，不能拿它当 resume。
- 同训练契约、同模型配置、同 TRAIN 正权重；只平均可学习参数，固定 buffers 不平均。
- 评估共享确定性历史/几何和 Moving 区域，不能共享候选之间 learned poses/features/输出。
- 固定 Dense Forecast FPS 边界：CausalHistoryState→fresh Strong/KTA+live motion+CCR+实时投影+六帧 dense。历史表示准备另报，不称 raw-input E2E。
- 新平均权重的 FPS 尚未测；不能沿用旧冻结 Surface 或 Frozen B 的 FPS。
- DEV64/DEV512 和 VAL 扩大人口均已参与研究，不称独立测试；本次已授权固定平均的 full4369，不自动重训/部署。

## 当前进度与服务器锚点

完整 TRAIN20430×20，102893 updates，whole-cycle cosine floor0.1，无 tail。所有 `epoch_XXXX.pt` 保留在原及续训目录。
当前完成锚点（文件在 run 根目录，不是 model 子目录）：

```
/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787
  last.pt
  training.json
  epoch_XXXX.pt
  summary.txt
```

其他轮次按相同 contract 在 `/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr` 下查找。
VAL 只读缓存：`/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_val_v1`。
TRAIN 缓存：`/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_v1`。
单卡 L40S，10 核、80 GiB；不再自动尝试多卡/改变 batch4/source128。

DEV64 关键 Joint 结果：

| epoch | IoU | mIoU | MovingMacro | MovingMicro |
| --- | --- | --- | --- | --- |
| 6 | 53.476777 | 41.666093 | 33.921350 | 28.234393 |
| 8 | 53.531044 | 41.574074 | 33.866847 | 28.646939 |
| 12 | 53.574938 | 41.553765 | 34.735137 | 30.010121 |
| 20 | 53.499071 | 41.172214 | 33.323547 | 27.574565 |

第 6 轮最高 mIoU；第 12 轮最高 IoU/Macro/Micro。不应只按最后一轮判断效果。
按 mIoU top5 的平均来源固定为 5/6/8/12/14，跨较大训练间隔，不保证平均更好。

原第 20 轮 DEV512：

| branch | IoU | mIoU | MovingMacro | MovingMicro |
| --- | --- | --- | --- | --- |
| static_repair | 52.541410 | 39.619555 | 23.362541 | 28.837789 |
| dynamic_repair | 50.628250 | 39.324149 | 24.779588 | 30.681102 |
| joint | 52.543060 | 40.043394 | 24.779588 | 30.681102 |

## 最新已完成 DEV512 对照与冻结选择

| candidate | IoU | mIoU | MovingMacro | MovingMicro |
| --- | --- | --- | --- | --- |
| epoch6 | 52.451547 | 40.248853 | 24.912767 | 30.599135 |
| epoch8 | 52.488365 | 40.300115 | 25.243657 | 30.535807 |
| epoch12 | 52.506929 | 40.199564 | 25.281199 | 30.875808 |
| mean5/6/8/12/14 | 52.372722 | 40.432276 | 25.718056 | 31.486519 |

平均的三个时距 mIoU/Macro/Micro均最高，但IoU略低，不称四项全面提升。
收益主要体现于Transport；平均Transport mIoU39.362646 / Micro30.014088，CCR再增+1.069630 / +1.472431 pp。
四候选512窗口合计705.52s，这是质量eval耗时，不是正式FPS。
新平均 full4369 已完成，正式 FPS 正在补测，不能沿用旧冻结模型的 FPS。

2026-10-09 服务器全集报告：固定平均（5/6/8/12/14）在 full4369、150 场景上
IoU55.211054 / mIoU44.153241 / MovingMacro28.256901 / MovingMicro32.214410。
同模型 Transport：53.631093 / 43.218661 / 26.889216 / 30.729273；CCR 增益分别
+1.579960 / +0.934580 / +1.367684 / +1.485137 pp。
相对旧 Local full4369：IoU-0.347217 / mIoU+0.124314 / Macro+0.968314 / Micro+0.974187 pp。
耗时1649.02s是带GT/整数指标的质量评估，不是FPS；仍不称独立测试。
结果：`outputs/p0_f9_joint_surface_ccr/mean_full4369_20261009_104036_848/full_validation.json`（服务器）。

## 本轮实现与验收

独立分支 `feature/v22-surface-aware-ccr`，工作副本 `swfm-surface-ccr`。原 `swfm-v20utc` 的用户未提交改动不动。
用户已授权该新分支提交/推送；不能推断新服务器训练/部署授权。

新增选择/平均模块、共享历史批量评估模块、CLI 与 Bash wrapper。
跨续训目录严格校验 lineage、完成 epoch、报告、源 SHA、原 TRAIN 权重，缺文件/重复冲突停止。
新平均权重标记 evaluation_only，不含原 optimizer/RNG。
四候选共用一次窗口准备，分别实时运行模型；每 8 个完整四模型窗口保存整数状态，严格 --resume。
既有 trainer/model/cache 依赖文件未改，避免破坏原训练 resume 或大缓存 namespace。
120 项相关本地回归通过，含真实 CPU/CUDA 单路径 vs 批量计数一致、中断恢复、平均/源文件只读与空类别指标，以及原联合恢复/表面读出回归。
增加只读缓存 namespace 预检和明确四窗口 CPU 预取后，10 项专项复查也通过；新文件 AST 和 Bash 语法检查通过。未声称全仓 CI 或真实 nuScenes 质量通过，本地没有完整服务器数据。

操作与风险见 `docs/JOINT_SURFACE_CHECKPOINT_AVERAGING_CN.md`。
服务器运行 `bash tools/real_motion/run_p0_f9_joint_surface_checkpoint_comparison.sh`；结果在新 comparison 目录，不覆盖训练。
中断后指定原 comparison 输出目录和 `SURFACE_COMPARE_RESUME=1`，不是恢复训练。
本次增加 `eval_p0_f9_joint_surface_mean_full.py` 与 wrapper，复用同一评估引擎，仅加载已完成对照中的平均文件。
只读已选来源，完整VAL4369不做DEV512筛选、不导入epoch20子集报告；新结果写full_validation.json，严格同目录整数resume。
操作见 `docs/JOINT_SURFACE_MEAN_FULL_VALIDATION_CN.md`。121 项相关本地回归通过，包含完整人口、来源冻结/只读、真实共享CPU/CUDA路径和CLI续评等价；AST/Bash语法通过。模型/trainer/缓存依赖未改。服务器未运行，不冒充全集结果或完整CI。

新增平均权重正式FPS入口；复用原 `paired_speed`（旧20×3默认兼容），不改训练/model/cache依赖。
19项相关本地回归通过，含真实CPU/CUDA读出及六帧合成一致、均值FPS算法、权重只读、缺帧拒绝、计时内建图拒绝、旧入口回归；CLI help/Bash语法通过。
这不是服务器实测FPS或全仓CI。服务器命令：`bash tools/real_motion/run_p0_f9_joint_surface_mean_fps.sh`；输出新 `mean_fps_*` 目录。
可用 `SURFACE_MEAN_FPS_WINDOWS=512` 扩大人口；中断保留partial日志，不改断点，重测使用新输出，不混拼前次计时。

## 未解决问题与下一步

1. 用户已授权新平均的正式 FPS，要求更多窗口平均。新入口 `run_p0_f9_joint_surface_mean_fps.sh`：默认256场景均衡随机窗口×3，不按GT/错误/耗时选样，不重训或重跑全集质量。
2. 只加载已完成DEV512对照中同一固定平均文件；原训练/权重/缓存不动。沿用正式边界，fresh Strong/KTA和六帧CCR全部实时计算；history/descriptor准备、预热/图捕获/一致性核对不计入FPS且单独报告。
3. 普通及融合执行完整概率/六帧dense逐字节核对；正式固定fused_graph（完整8192块建图、尾块eager），禁止计时内建图。FPS=6/平均六帧延迟，P90/分段/显存同报，不按最小延迟挑结果。
4. 本地回归只证明实现正确，实际L40S FPS仍待服务器运行。256随机窗口与旧20压力人口不同，不据跨次数字宣称成对提速。
5. 全集已追回mIoU和Moving，IoU仍较旧Local略低；暂不改结构/阈值，不重启已淘汰支线。

## 仅按需检索的历史

冻结 Surface CCR 完整 TRAIN×3 DEV512 mIoU40.418031/Micro31.052754；扩大 VAL4369 mIoU43.991199/Micro32.043717。
Frozen B 与旧 Local 的静态差距集中 road11/sidewalk13；Surface 内部几何条件读出已追回大部分冻结静态差距。
这些是早先冻结实验，不是新联合精度。旧阈值、halo、共享场、点 CCR 支线不重启。
详细历史仍保留在 `docs/CLEAN_JOINT_SURFACE_CCR_CN.md`、`docs/SURFACE_CONSISTENT_CCR_FULL_VALIDATION_CN.md`、`docs/SURFACE_CCR_EXPANDED_VALIDATION_CN.md`，需要时检索，不复制旧日志。
