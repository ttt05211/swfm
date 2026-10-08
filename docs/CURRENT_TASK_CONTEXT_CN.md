# 当前任务摘要（2026-10-08）

## 目标与决定

追回 Frozen B 的路面/人行道精度，同时保留固定 Dense Forecast FPS≥40 的速度目标；不在 Transport→CCR 后再外挂修正器。以 CCR 内部的表面几何表示和条件读出做一次受控验证，有效后才用同一结构从头完整联合训练。

当前分支 `feature/v22-surface-aware-ccr`，独立工作副本 `swfm-surface-ccr`，基于别人最新实现 `feature/v22-final-dataflow-fps@b1507ed`。原 `swfm-v20utc` 有用户未提交改动，未动。用户已明确允许提交并推送新分支。

## 当前运行问题与修复（2026-10-08）

用户已停止扩大评估 `expanded_20261008_150128_838`。最近125窗口surface_probability2629ms/B8.35ms，输入等待1.41ms；不是输入等待造成。代码确定存在变长尾块按shape捕获CUDA Graph的单遍退化；本地合成变长6窗口head-only复现旧策略64.0ms/eager7.27ms，概率逐字节一致，但未证明服务器2629ms全部由此造成。

修复：质量eval默认eager，与预热FPS后端选择分离；可选fixed-full-chunk graph，只捕获8192静态整块，尾块eager。正式FPS口径/模型/权重/候选/阈值均不变。逐窗口记录缓存命中、输入等待、points及graph upload/capture/forward/readback统计；JSON运行中也保存缓存/执行统计。仅allow-list原36714f1指纹且head文件SHA不变时允许整数eval断点迁移，其余科学合同继续严格检查；完成的FPS复用，旧prefix计时不能混算paired提速。相关本地回归65 passed，含真实CUDA变长窗口、不变动态/cache/dataflow及原实现指纹断点兼容；不是全仓CI或L40S实测。

下一步先从 `/root/nas/occ/swfm/outputs/p0_f9_surface_ccr/expanded_20261008_150128_838/evaluation_progress.pt` 接续，读取新readout子计时确认异常是否消除。不重训，不重建缓存、不从头评估；Ctrl+C边界/kill-9周期区别保持。之后才按full质量/FPS决定干净联合训练。

## 已知结果（用户服务器报告）

Surface-consistent CCR 已完成完整 TRAIN20430×3 冻结验证，第三轮 DEV512 mIoU40.418031 / MovingMicro31.052754；vs Frozen B +0.380824 / 0 pp，vs Old Local REMOVE-off +0.041886 / +0.941836 pp。三个时距 road11/sidewalk13 都改善。训练步0.053211秒/窗口（运动/动态冻结，不能称完整联合训练提速）。同20窗口×3正式对照：B49.696705 FPS/120.732ms，新CCR40.102844 FPS/149.615ms，达到40但余量很薄。

服务器新候选：`/root/nas/occ/swfm/outputs/p0_f9_surface_ccr/full20430x3_20261008_131503_838/last.pt`。固定第三轮，不再追加轮数/调阈值。目前用户同意继续：一次扩大验证并尝试输出一致的执行优化，仍不自动启动随机联合训练。

Frozen B：4历史→6未来，epoch19 V18固定 + full TRAIN Point CCR epoch2；weighted ADD raw sigmoid@0.5，REMOVE-off。full4369 vs Transport mIoU +0.5576 / MovingMicro +1.5020 pp；vs Old Local −0.4056 / +0.8035 pp。正式 FPS47.522，六帧126.256ms。

静态差距集中于 road11/sidewalk13。DEV512 B-only FP中 Direct约68%、Halo约32%，97.7% GT-free；不能仅据此确定几何错位的内因。七种轻量规则只追回+0.0326mIoU，不再继续阈值/halo小修补。正式 Frozen B 未被升级。

## 核心约束

- 无 GT 输入/候选筛选、KD、AE、新 Transformer、六个重空间编码器。
- 保留全部候选、采样、合成/ownership、动态路径；不悄悄改 batch 或训练预算。
- v2 TRAIN/VAL 缓存 namespace 的七个依赖文件保持原样；require只读，禁止自动重建/缓存 learned features、future phase或未来GT。
- 固定 FPS边界：CausalHistoryState→fresh Strong/KTA+live motion+learned CCR+未来投影/读出+六帧dense。确定性历史表面表示单列准备时间；不得称原始输入端到端FPS。
- 暂时冻结运动/动态做验证，最终干净联合训练仍待验证结果，不能宣称已完成或自动启动。

## 已实现 / 验证状态

`real_motion/surface_canonical_repair.py`：12维局部表面描述进入 CCR 编码，6维 live投影高度/相位进入替换后的静态条件读出。候选不变，动态读出保留，无外挂logit修正。

`surface_ccr_screen_common.py` + `train_p0_f9_surface_ccr.py` + `run_p0_f9_surface_ccr_full.sh`：完整TRAIN20430×3，新静态optimizer/LR3e-4全周期余弦，复用B权重/正权重，dev64监控、final dev512、20×3 paired正式FPS，严格resume。

首次实现集中本地回归124 passed/1 skipped；601个Python文件AST语法检查及Bash语法检查通过。覆盖CPU/CUDA实际更新、动态逐字节不变、cache/full/sample几何一致、无未来GT、next-update恢复、final-dev后中断只续做FPS、纯静态/动态/混合读出路由。修复旧测试mock的live-preflight、CUDA设备接口，测试入口提前设置cuBLAS确定性环境，不放宽容差。全仓库1437项重回归中途停止，未宣称完整CI通过。本地没有完整nuScenes缓存。

本次新增：`surface_ccr_execution.py`仅 pure-static chunk 的 CUDA Graph 执行，保留原8192分块/BF16/候选与算术，实际捕获后概率字节核对；动态/混合仍eager，最多4图，权重变化拒绝复用，不持久缓存 learned feature。`validate_p0_f9_surface_ccr_expanded.py` + `surface_ccr_validation_common.py` + `run_p0_f9_surface_ccr_expanded.sh` 一次运行固定20×3 paired B/eager/graph正式FPS，然后固定 full4369 验证。图模式通过字节检查且至少快2%才用于质量评估（按速度选执行，不按精度选方法）。报告full、DEV512、窗口/场景之外子集，所有类/时距IoU、MovingMacro/Micro、路面/人行道TP/FP/FN、场景变化、静态/动态ADD P/R。整数计数只计算一次供多个子集复用；Ctrl+C窗口边界恢复，kill-9恢复32窗口周期，完成FPS不会重测。参考/候选输入先快照，原optimizer/权重/缓存不写。

本次相关本地回归81 passed/1 skipped，606个Python AST和Bash语法检查通过；包含真实CUDA图、非零几何权重/变化输入、8192+尾块/不同source数/road-sidewalk字节一致、图淘汰与拒绝回退、实际head/投影/合成/整数统计的CLI中断恢复（外部nuScenes IO和FPS mock）、原缓存/正式数据流回归。缓存namespace七个依赖文件仍与b1507ed逐字节相同。不是全仓CI。

增加纯静态/动态分块的读出路由，保持批次大小和概率逐字节不变；冻结验证不保留无用旧读出的backward图。原Frozen B输入先快照；原文件不写。实现提交 `15a479a` 已推送到新分支 `feature/v22-surface-aware-ccr`，原分支未改动。

本地RTX3050合成48000候选、5遍head-only：B81.28ms，新head未路由120.90ms、路由91.77ms，初始化概率字节一致。这不是正式FPS、联合训练耗时或精度验证。

## 精确服务器路径

```
B=/root/nas/occ/swfm/outputs/p0_f9_point_ccr/full20430_cache_lr2e3_epoch2_20261007_194545/last.pt
B_SHA256=fb6f7bdfa9e7ebca8f479d7005372ad328e09a2ed44e84514cfb180722942204
TRAIN_CACHE=/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_v1
VAL_CACHE=/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_val_v1
E19=/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt
```

OccFM服务器单L40S，10核/80GiB。旧cached完整轮十几分钟为用户经验，不是新模型实测承诺。原checkpoint和optimizer不得覆盖。

## 下一步与未解决问题

1. 服务器拉取新分支，运行 `bash tools/real_motion/run_p0_f9_surface_ccr_expanded.sh`。不重训/重建缓存；已有第三轮结果不得重新反复筛选。
2. 读取 expanded_validation.json/summary.txt 的 full4369、DEV512之外窗口/场景、road/sidewalk TP/FP/FN及同口径FPS。所有这些VAL数据已参与研究，不称独立测试。是否值得保留graph由L40S同窗口字节/速度检查决定，失败/不快保留eager。
3. 扩大验证保持收益、速度不退后，再实现相同结构的干净随机初始化完整联合训练；当前入口不会启动。

具体设计、风险与命令见 `docs/SURFACE_CONSISTENT_CCR_FULL_VALIDATION_CN.md`、`docs/SURFACE_CCR_EXPANDED_VALIDATION_CN.md`。历史失败方案不重启，不复刻重复日志，必要时按旧文档索引检索。
