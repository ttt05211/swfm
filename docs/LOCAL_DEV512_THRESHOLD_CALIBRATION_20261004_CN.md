# 固定 Local checkpoint 的 dev512 阈值校准

目的：仅调整 generation ADD、refine ADD、refine REMOVE 的推理门槛，不训练或改动权重。三个门槛都是经 TRAIN prior 校正后的 action probability，**不是 mIoU 增益门槛**；动作仍须严格胜过 KEEP 并符合原合法区域。ADD 不能覆盖原占据标签；REMOVE 保留原 source-owner fallback 语义。

## 冻结范围与速度

- 固定已选中的 epoch19 snapshot；冻结 dev512 的512个 identity/order，从 full4369 V18 cache 严格对齐，禁止取前512条。
- 每个门槛预定扫描 `0.5 / 0.75 / 0.95 / off`，共64组 joint、4组 generation-only、16组 refine-only。`off` 真正禁止该动作，即使预测概率为1。
- 一个窗口只做一次 current transport、一次冻结 E14 reference、三个 report horizon 的 probability prediction。不同门槛共享概率，不重复运行64遍网络；64组 joint 复用4份生成和16份 refine 合成。指标仍逐组合准确计数，并不是零成本；未承诺实测服务器耗时。
- 同一 horizon 中产生完全相同 voxel 输出的组合复用整数指标计数；哈希命中后仍检查逐元素一致，source-layer REMOVE 决策计数独立保留。只保留当前 horizon 的有限缓存，不增加大型持久缓存。
- 首个窗口逐元素验证64组因式分解合成等于原 renderer，并检查代表组合的 sparse/full integer counts；优化概率路由也保留首用 exactness gate。GT 仅进入统计，不输入模型、候选或合成决策。

## 选择规则

固定对照 `(0.5, 0.5, REMOVE-off)`。在整体 mIoU 不降低且**整体及1/2/3s MovingMicro均不低于固定对照**的组合中选择最高 mIoU。另列无 Moving 保护的最高 mIoU 组合，仅供诊断。没有安全的 mIoU 增益就保留固定对照；不会偷偷放松约束，也不会自动启动新的训练或 full 评估。

dev512 已经用于选 checkpoint，现在又用于调阈值，输出明确是 **selection/calibration 分数，不是独立测试**。full4369 包含 dev512，也不能称为独立测试。正在运行的 fixed-threshold full 结果不改动、不重新解释；新门槛仅写入独立 calibration artifact，后续是否重跑 full 另行决定。

## 服务器入口

在 OccFM 环境、更新仓库后运行：

```bash
bash tools/real_motion/run_p0_f9_joint_threshold_calibration.sh \
  /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt
```

脚本默认等待本仓库已运行的训练/评估退出，不发终止信号；可在另一终端排队。只新增代码入口，没有修改现有 full evaluator，因此更新代码不会改变该任务正在使用的实现。输出新目录的 `summary.txt`（也打印到终端）、`calibration.json`（所有组合与 provenance）、`progress.jsonl`。

允许与 full 同时运行时，在另一终端显式关闭等待，并降低校准 worker 预算：

```bash
JOINT_THRESHOLD_WAIT=0 JOINT_THRESHOLD_CPU_WORKERS=2 \
nice -n 5 bash tools/real_motion/run_p0_f9_joint_threshold_calibration.sh \
  /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt
```

两项任务使用独立进程、checkpoint snapshot 和输出目录；不修改正在运行的 full 阈值或权重。CPU/GPU争用可能使两项任务分别变慢，低显存占用不能保证并行更快；主存也需容纳两项任务。worker 数是线程池预算，并非进程的硬 CPU 核数限制。`nice` 只降低校准进程的 CPU 调度优先级，不限制 GPU 占用。无需停止 full。若之前启动了等待中的校准 wrapper，先在该等待终端 Ctrl+C，再启动并行命令，避免重复启动。

SIGINT/SIGTERM 在完整窗口边界保存；每8个完整窗口另存一次。强制 kill 最多丢失未保存的窗口统计，不写源 checkpoint 或 optimizer/RNG。恢复必须指定原校准输出目录：

```bash
JOINT_THRESHOLD_OUT=/实际校准输出目录 \
JOINT_THRESHOLD_RESUME=1 \
bash tools/real_motion/run_p0_f9_joint_threshold_calibration.sh \
  /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt
```

恢复校验 snapshot/source SHA、原配置、缓存/info/manifest provenance、ordered population、扫描网格和计数 fingerprint；独立输出 lease 拒绝同目录并发恢复。完成后不允许覆盖重跑。原 full 评估目录与训练文件始终只读、不清理。

并行模式以2 workers启动的校准，恢复时也须设置 `JOINT_THRESHOLD_CPU_WORKERS=2`；CPU预算是恢复合同的一部分，禁止静默切换。若仍需并行，另设置 `JOINT_THRESHOLD_WAIT=0`。等待开关和 nice 优先级不改变统计合同。
