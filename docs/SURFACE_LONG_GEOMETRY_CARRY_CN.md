# 冻结 Surface CCR：6 秒因果几何接续对照

## 目标与边界

当前同一个 5/6/8/12/14 均值网络，在 GenieDrive 公开代码人口的 2569 个窗口、150 场景上：
4/5/6 秒 mIoU 为 30.572013 / 27.095679 / 24.488694，平均 27.385462。
已有 first/second block exactness 通过。这不是论文 Table-2 精确人口的独立复现，不能直接宣称 SOTA。

本实验尝试减少递推时重复体素化和历史形状对齐的误差，不重训、不改网络、权重、阈值、未来轨迹预测或旧入口。
这属于**推理算法候选**，会改变第二段输出，不属于此前逐字节无损的执行后端优化。
局部几何例子可以证明重复体素化有误差，但不能据此声称数据集指标一定提升。

初始仍只有四张真实历史。第二段保留由这些观测和首段预测产生的几何状态，严格不新增真实未来观测。
因此应描述为四历史输入的 **stateful causal rollout**，不能声称与“只携带最后四张 dense grid、不携带状态”的接口完全相同。
未来 6 秒 GT ego poses 仍是显式条件；未来 occupancy、mask、annotation 只在四路预测完成后进入指标。
Moving 支持来自原始 t0，而不是第二段 t0。

## 一趟四路

| 路线 | 第二段静态背景 | 第二段 CCR 的过去动态形状对齐 |
| --- | --- | --- |
| baseline | 原 reconciled 递推 | 原 ICP |
| static_carry | 原始 t0 静态直接投影到第二段未来 | 原 ICP |
| se2_carry | 原背景 | 可靠身份的首段预测 SE(2) |
| combined | static_carry | se2_carry |

共享第一段六帧预测和原 reconciled 的第二段 motion forward；第二段每路独立实时 evidence/projection/Surface 读出与合成。
1–3 秒所有路线完全相同，不把同一结果当作四次独立证据。

静态候选沿用 Strong 的静态类别划分、nearest inverse warp 和 majority fill；原始 t0 静态只体素化一次。
保留首段最后一张预测相对 Transport 的 CCR 静态新增，动态前景及重叠动态 fallback 不改。
它可能删除或重标原递推的静态背景，故**相对 baseline 不属于 free-only**；各路 CCR 本身仍 ADD@0.5、REMOVE-off。
相对 baseline 的 changed/corrected/damaged/added/removed 都报告，不能用 CCR 的 removed=0 掩盖背景替换损伤。

SE(2) 候选只用首段已预测的世界中心和实际 renderer yaw，将过去预测组件对齐到预测 3 秒姿态。
只接受现有 handoff 的可靠身份；历史组件 split/merge、语义不符、缺身份、对齐质心偏移超过既有 4m 安全门槛，均回退原 ICP。
不复活丢失 source，不预测新 birth，不平滑速度，不把首段 yaw 重复施加到第二段未来。
第二段未来位置/yaw 仍由原网络给出。当前来源、输入特征、输出 Tensor 顺序不变。

## 固定人口与选择

`screen` 一趟跑 TRAIN64 + dev64：TRAIN 从完整 train20430 的完整 6 秒窗口中确定性 scene-balanced 取64；
dev64 使用现有冻结 dev512 的完整6秒 population 规则。各自保存实际 key/order、未来 token 与时间审计。
TRAIN 和 DEV 场景必须不相交。无需重建 prototype 或大几何缓存。

只在 TRAIN64 的三个候选上采用以下预先固定规则：

- 4–6 秒平均 mIoU 至少增加 0.05 pp。
- 平均 IoU、MovingMicro 不下降。
- 4/5/6 秒每个时距 mIoU 不下降。

通过者中取 TRAIN 平均 mIoU 最高者；同分保持上述路线顺序。没有通过者就停止，不自动调阈值或重试。
TRAIN64 是 in-sample 诊断，不能证明泛化。dev64 全部报告，但不用于选择路线。
`all` 只接受同权重、同代码、完整校验通过的 screen evaluation.json，跑冻结的 TRAIN 候选和 baseline。
人口由官方固定 GenieDrive metadata 决定（当前2569、150场景），仍报告标准指标与公开代码兼容指标，不混换口径。

## 服务器运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm

git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-surface-aware-ccr
git merge --ff-only FETCH_HEAD

OUT="$PWD/outputs/p0_f9_joint_surface_ccr/geometry_carry_screen_$(date +%Y%m%d_%H%M%S)"
bash tools/real_motion/run_p0_f9_surface_geometry_carry.sh screen "$OUT"
cat "$OUT/summary.txt"
```

自动复用已完成 DEV512 对照中的同一均值文件；来源不唯一或缺文件会停，不重新平均、不猜 checkpoint 名字。
`SURFACE_MEAN_SOURCE` 可显式指向真实 comparison 目录；`SURFACE_COMPARE_RUN` 默认已确认的 full20 完成锚点。
默认已有 VAL 几何缓存只读，TRAIN64 历史实时准备；`SURFACE_LONG_VAL_CACHE=off` 可显式禁用。
采用有界 native fused projection/graph 与4线程无损多数投票；资源上限沿用10核，缓存不写盘。
这是带 GT 指标的质量对照耗时，不称正式 FPS。

需要中断用一次 Ctrl-C 或 SIGTERM，等待完整窗口保存。继续同一输出目录：

```bash
bash tools/real_motion/run_p0_f9_surface_geometry_carry.sh screen "$OUT" --resume
```

每8个完整四路窗口保存整数计数；SIGKILL/崩溃仅回到最后一次保存，不承诺尚未保存窗口。
resume 锁定来源 SHA、输入 SHA、key/order、实现、执行开关，修改代码后不得混续旧计数。
旧模型、训练断点、缓存和原实验目录不改。

TRAIN 门槛通过并检查 dev64 报告后，再显式运行（不会自动启动）：

```bash
export SURFACE_CARRY_SELECTION="$OUT/evaluation.json"
bash tools/real_motion/run_p0_f9_surface_geometry_carry.sh all
```

汇总 `summary.txt`；原始整数指标、类别4–6秒IoU、场景delta、候选保护/回退计数、损伤、分段计时在 `evaluation.json`。
实际服务器收益和 GPU 路径需这次运行确认，本地 CPU 单元结果不能代替。
