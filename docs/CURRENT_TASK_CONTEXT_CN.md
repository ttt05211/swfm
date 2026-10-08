# 当前任务摘要（2026-10-08）

## 目标与决定

追回 Frozen B 的路面/人行道精度，同时保留固定 Dense Forecast FPS≥40 的速度目标；不在 Transport→CCR 后再外挂修正器。以 CCR 内部的表面几何表示和条件读出做一次受控验证，有效后才用同一结构从头完整联合训练。

当前分支 `feature/v22-surface-aware-ccr`，独立工作副本 `swfm-surface-ccr`，基于别人最新实现 `feature/v22-final-dataflow-fps@b1507ed`。原 `swfm-v20utc` 有用户未提交改动，未动。用户已明确允许提交并推送新分支。

## 已知结果（用户服务器报告）

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

最终集中本地回归124 passed/1 skipped；601个Python文件AST语法检查及Bash语法检查通过。覆盖CPU/CUDA实际更新、动态逐字节不变、cache/full/sample几何一致、无未来GT、next-update恢复、final-dev后中断只续做FPS、纯静态/动态/混合读出路由。修复旧测试mock的live-preflight、CUDA设备接口，测试入口提前设置cuBLAS确定性环境，不放宽容差。全仓库1437项重回归中途停止，未宣称完整CI通过。服务器训练尚未启动，本地没有完整nuScenes缓存。

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

1. 服务器拉取新分支，运行 `bash tools/real_motion/run_p0_f9_surface_ccr_full.sh`。本地合成head开销不是服务器FPS/精度证据。
2. 一次跑完完整TRAIN×3、final dev512及同口径FPS，确认静态增益、Moving不退、≥40FPS；目前服务器训练未启动。
3. 只有通过，才设计并启动同一结构的随机初始化完整联合训练；不能从冻结验证直接推断联合效果。

具体设计、风险与命令见 `docs/SURFACE_CONSISTENT_CCR_FULL_VALIDATION_CN.md`。历史失败方案不重启，不复刻重复日志，必要时按旧文档索引检索。
