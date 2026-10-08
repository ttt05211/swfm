# 当前任务摘要（2026-10-08）

## 目标与决定

追回 Frozen B 的路面/人行道精度，同时保留固定 Dense Forecast FPS≥40 的速度目标；不在 Transport→CCR 后再外挂修正器。以 CCR 内部的表面几何表示和条件读出做一次受控验证，有效后才用同一结构从头完整联合训练。

当前分支 `feature/v22-surface-aware-ccr`，独立工作副本 `swfm-surface-ccr`，基于别人最新实现 `feature/v22-final-dataflow-fps@b1507ed`。原 `swfm-v20utc` 有用户未提交改动，未动。用户已明确允许提交并推送新分支。本轮用户批准保持结果的速度优化，并实现干净联合训练与resume；没有自动启动服务器训练。

## 最新扩大验证：已完成（2026-10-08）

用户从原128窗口整数断点接续后，`expanded_20261008_153254_838` 已完成full4369/150场景。Surface CCR IoU55.273305 / mIoU43.991199 / MovingMicro32.043717；vs Transport +1.668354 / +0.925487 / +1.502008 pp；vs Frozen B +1.052876 / +0.367860 / 0 pp；vs同transport Old Local REMOVE-off −0.284966 / −0.037729 / +0.803494 pp。追回B→Old mIoU差距约90.7%，不宣称已证明统计等价。

DEV512 mIoU40.418031/Moving31.052754，与原三轮验证一致。Outside DEV512 3857窗口：vsB mIoU+0.369121/Moving0；outside132场景3851窗口：+0.368612/0，静态收益稳定。这些VAL人口已用于研究，不称全新独立测试。

正式固定20窗口×3同口径：B48.7055 FPS/123.1893ms；Surface eager39.6339 FPS/151.3857ms；预热graph40.4534 FPS/148.3188ms，P90 177.96ms。graph刚越40，余量约1.1%，不称稳定保证。Strong约82ms，readout含live phase约34ms，其中phase约26.8ms；单独历史surface描述准备122.66ms不含于冻结Dense Forecast FPS，不称原始历史输入端到端。

续评VAL缓存hits4241/misses0/writes0，加之前128窗口，共4369。修复后用户日志readout13–15ms/capture0。累计performance含旧慢prefix：old Local2195s、描述630s、指标514s；不能将混合prefix的surface累计均值当修复版独立速度。本次完整结果JSON位于 `/root/nas/occ/swfm/outputs/p0_f9_surface_ccr/expanded_20261008_153254_838/expanded_validation.json`。

已推送执行修复 `6ef4246`：质量eval默认eager，与预热FPS后端分离；可选fixed-full-chunk graph，仅捕获8192整块、尾块eager。模型/阈值/原缓存namespace不变，已知旧指纹可严格迁移整数进度；相关65项本地回归通过。没有自动开启新训练或升级部署候选。

## 已知结果（用户服务器报告）

Surface-consistent CCR 已完成完整 TRAIN20430×3 冻结验证，第三轮 DEV512 mIoU40.418031 / MovingMicro31.052754；vs Frozen B +0.380824 / 0 pp，vs Old Local REMOVE-off +0.041886 / +0.941836 pp。三个时距 road11/sidewalk13 都改善。训练步0.053211秒/窗口（运动/动态冻结，不能称完整联合训练提速）。同20窗口×3正式对照：B49.696705 FPS/120.732ms，新CCR40.102844 FPS/149.615ms，达到40但余量很薄。

服务器新候选：`/root/nas/occ/swfm/outputs/p0_f9_surface_ccr/full20430x3_20261008_131503_838/last.pt`。固定第三轮，不再追加轮数/调阈值。扩大验证已完成，当前准备执行优化与新的随机联合训练入口，未自动启动训练。

Frozen B：4历史→6未来，epoch19 V18固定 + full TRAIN Point CCR epoch2；weighted ADD raw sigmoid@0.5，REMOVE-off。full4369 vs Transport mIoU +0.5576 / MovingMicro +1.5020 pp；vs Old Local −0.4056 / +0.8035 pp。正式 FPS47.522，六帧126.256ms。

静态差距集中于 road11/sidewalk13。DEV512 B-only FP中 Direct约68%、Halo约32%，97.7% GT-free；不能仅据此确定几何错位的内因。七种轻量规则只追回+0.0326mIoU，不再继续阈值/halo小修补。正式 Frozen B 未被升级。

## 核心约束

- 无 GT 输入/候选筛选、KD、AE、新 Transformer、六个重空间编码器。
- 保留全部候选、采样、合成/ownership、动态路径；不悄悄改 batch 或训练预算。
- v2 TRAIN/VAL 缓存 namespace 的七个依赖文件保持原样；require只读，禁止自动重建/缓存 learned features、future phase或未来GT。
- 固定 FPS边界：CausalHistoryState→fresh Strong/KTA+live motion+learned CCR+未来投影/读出+六帧dense。确定性历史表面表示单列准备时间；不得称原始输入端到端FPS。
- 原冻结验证已完成；干净联合训练入口已实现，尚未在完整服务器数据上训练，不能把冻结验证的精度/耗时当联合结果。

## 已实现 / 验证状态

`real_motion/surface_canonical_repair.py`：12维局部表面描述进入 CCR 编码，6维 live投影高度/相位进入替换后的静态条件读出。候选不变，动态读出保留，无外挂logit修正。

`surface_ccr_screen_common.py` + `train_p0_f9_surface_ccr.py` + `run_p0_f9_surface_ccr_full.sh`：完整TRAIN20430×3，新静态optimizer/LR3e-4全周期余弦，复用B权重/正权重，dev64监控、final dev512、20×3 paired正式FPS，严格resume。

首次实现集中本地回归124 passed/1 skipped；601个Python文件AST语法检查及Bash语法检查通过。覆盖CPU/CUDA实际更新、动态逐字节不变、cache/full/sample几何一致、无未来GT、next-update恢复、final-dev后中断只续做FPS、纯静态/动态/混合读出路由。修复旧测试mock的live-preflight、CUDA设备接口，测试入口提前设置cuBLAS确定性环境，不放宽容差。全仓库1437项重回归中途停止，未宣称完整CI通过。本地没有完整nuScenes缓存。

本次新增：`surface_ccr_execution.py`仅 pure-static chunk 的 CUDA Graph 执行，保留原8192分块/BF16/候选与算术，实际捕获后概率字节核对；动态/混合仍eager，最多4图，权重变化拒绝复用，不持久缓存 learned feature。`validate_p0_f9_surface_ccr_expanded.py` + `surface_ccr_validation_common.py` + `run_p0_f9_surface_ccr_expanded.sh` 已完成固定20×3 paired B/eager/graph正式FPS和full4369验证。质量eval默认eager，不再由预热FPS自动选择变长图。报告full、DEV512、窗口/场景之外子集，所有类/时距IoU、MovingMacro/Micro、路面/人行道TP/FP/FN、场景变化、静态/动态ADD P/R。整数计数一次供多个子集复用，参考输入快照，原optimizer/权重/缓存不写。

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

1. 扩大验证已结束，不重跑、不继续阈值/支线结构搜索。静态表示设计得到支持；Moving提升继承Frozen B，不归因于本次静态训练。
2. 已实现 `surface_projection_execution.py`：本次forward复用 float64 phase/Z/目的体素，保留原整数kernel；`SurfaceAtlas`最多4线程query，复用缩放坐标。原namespace七文件未改，不保存learned内容。参考/优化概率+六帧dense字节核对；初版中间数组过多反而变慢，已弃。合成60000点5遍CPU微基准投影109.01→86.56ms、描述324.60→268.07ms，不是整网/server速度。`run_p0_f9_surface_ccr_execution_benchmark.sh` 同20×3对比B/旧surface eager+graph/复用eager+graph，准备时间单列，未在服务器运行。
3. 已实现 `JointSurfaceCCR` + `train_p0_f9_joint_surface_ccr.py` + wrapper：完整TRAIN20430，4→6，运动/静态/动态全部随机初始化和训练，无teacher/KD/AE；默认20全周期余弦、batch4/source128、LR5e-4/3e-4、原V18 loss+现有weighted ADD/raw0.5/REMOVEoff，TRAIN256 unsampled先验，dev64逐轮/final dev512，禁止自动full/DEVbest/提升部署。硬renderer detach副本，live sourcequeries保留CCR梯度；只读现有TRAIN/VAL缓存。
4. 严格新协议resume：atomic last+previous、每128步+每轮snapshot、全模型/optimizer/所有RNG/cursor/完成评估阶段恢复；失败半步不写污染checkpoint。旧冻结/Local ckpt不能resume为clean联合训练，预算或实现变更拒绝。只读 `--evaluate-only` 可评估dev64/dev512/显式full4369，不改源optimizer/RNG。CPU/CUDA next-update、CLI中断恢复、TRAIN完整count、empty support已有实际测试。
   本轮相关回归191项通过，补测新增16个真实native/并行投影组合及旧整数断点迁移（43项补测全部通过）；合计207个相关测试均通过，615个Python AST、两新wrapper Bash语法通过。不是完整CI，不是nuScenes真实训练验证；保留旧空Moving denominator的nanmean warnings。
5. 后续先读取一次服务器execution benchmark结果，再按用户选择启动clean训练；新联合质量/完整训练吞吐与最终权重FPS尚未验证，不把旧冻结分数冒充。可以初次设 stop-after-epoch2，但保持20轮LR周期；resume保持同一契约。新入口直接VAL缓存评估本身与transport，不重跑旧Local。

具体当前命令/风险/恢复说明见 `docs/CLEAN_JOINT_SURFACE_CCR_CN.md`。先前科学验证见 `docs/SURFACE_CONSISTENT_CCR_FULL_VALIDATION_CN.md`、`docs/SURFACE_CCR_EXPANDED_VALIDATION_CN.md`，仅按需读取。历史失败方案不重启，不复刻重复日志。
