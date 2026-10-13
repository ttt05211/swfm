# 服务器：固定 B4 Refiner 合批＋连续切片，只读测速

用户允许浮点近似执行的测速；**不再把严格数值FAIL当作禁止测速度的理由，但原FAIL、容差和基线重复漂移完整保留**。不自动采纳任何候选，不更新正式训练checkpoint。

## 四个候选

| 名称 | 执行 |
| --- | --- |
| reference | 当前GT完整训练原 sequential B4/H6 |
| slice_queries | 原B4/H6，prior高级索引换连续切片 |
| packed_reference | 4窗Refiner合为最多24个horizon，原grid_sample读出 |
| packed_slice | 同一Refiner合批＋连续切片读出 |

上采样分支仍只保留在本地实验，不在本次服务器测速中运行。不是全局batch16/24，不改变source128或训练步数。每窗六时距的CE＋Lovász及其归一化不变，无ego/motion/existence辅助loss；Encoder/STWM/Transport prior/memory仍独立逐窗，packed一次multi-root backward。

临时instance forward钩子在正常/异常路径都恢复，无正式trainer源码修改。

## 服务器操作

先在原训练终端按一次Ctrl-C，等完整update保存及返回提示符；不要同时有其它GPU任务。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/shared-history-execution-speed-v2
git merge --ff-only origin/feature/shared-history-execution-speed-v2

unset GT_CKPT
export SPEED_RUN_DIR="$PWD/outputs/p0_f9_shared_history/refiner_b4_slice_speed_$(date +%Y%m%d_%H%M%S)"
bash tools/shared_history_experiments/refiner_batch_readout/run.sh \
  && cat "$SPEED_RUN_DIR/summary.txt"
```

默认只扫描 `outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_*/train/last.pt`，取最近且协议正确的一份，打印 `AUTO_FULL_GT` / `FROZEN_SOURCE`。不会选旧20%或内部Pred模型。
如有多个全量GT任务，明确设定 `GT_CKPT=/实际原训练目录/train/last.pt`，不要执行上面的unset。
`SPEED_RUN_DIR`独立于原训练 `RUN_DIR`，所以测速不会把原续训目录变量改成测速目录。

默认8个真实TRAIN batch，实际window数以打印为准，受source128约束可少于32。四候选各两遍AB/BA，加载CPU输入在计时外，同一组记录/源权重/Adam/RNG/余弦位置；目标准备、所有几何/网络、全6loss、反传/clip/Adam及GPU完成等待均计时。各候选额外一轮数值audit预热不计时；不是FPS，不运行DEV512质量评估。
可追加 `--probe-steps 4` 缩短（仍两遍），或 `--profile` 单独计时外profiling。

## 只读与报告

源last.pt复制到新 `frozen/source.pt`；允许中期epoch0/1或完成epoch2，始终核对完整GT协议、数据/执行/初始化/目标契约。数据不匹配明确拒绝，不改旧契约来接线。
全部临时Adam更新都丢弃。原权重/Adam/RNG/缓存文件、训练summary不写；源文件与快照SHA在结束后校验，不保存新的训练last.pt。

新 `speed.json` 包含四路完整每遍秒/窗、阶段、reserved显存和free余量、单步CE/Lovász/梯度/全部窗六帧logits与argmax差异、多步model+Adam差异、原版重复检查。
`fastest_timed_backend`只表示耗时最短，**不是精度通过**。`speed_only_candidate`需两遍都更快且显存留10%；`pass_gate`保留原严格数值判定。`quality_verified=False`，`automatic_promotion=False`。OOM明确记失败/速度null，不偷换回原版；非OOM的形状/NaN/协议异常仍终止。

中断测速不保存临时优化状态，重测需新SPEED_RUN_DIR。原完整训练仍用原RUN_DIR：

```bash
unset RESUME
FULL_GT_RESUME=1 bash tools/shared_history_experiments/full_gt/run.sh
```

若另开终端，先设置RUN_DIR为原完整训练流程根目录（不是train或speed目录）。本轮只推送入口，不代启动服务器。

## 本地边界

本地新服务器专项＋候选＋旧GT/Refiner回归31passed in31.43s；补TRAIN元数据仅加载一次、简洁摘要后专项6passed in24.36s，Bash/CLI帮助检查通过。通用驱动的整套TRAIN记录直接复用，不在测速回调重复读取一份全量缓存。

真实CPU/CUDA小图B4前后向、四路hook/Adam/RNG恢复、来源协议及自动选择、source文件只读CLI、OOM记录和中断恢复已针对性验证；模拟数据集仅用于验证完整20430人口入口接线。没有本地nuScenes或L40S，服务器收益待回传，不沿用本地1.4×或旧模型FPS。
