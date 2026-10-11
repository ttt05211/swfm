# GT-only 占据联合训练：同目标 loss 执行提速

这次只优化执行，不改网络、六帧全网格监督、CE/Lovász 权重、batch4/source128、Adam 或余弦调度。
旧 GT 训练源码保持不动；新增工具全部在 `tools/joint_occ_experiments/performance/` 子目录，不进入旧 trainer 的源码契约指纹。
因此 c57bd48 的 GT-only `last.pt` 仍可用原命令续训。

## 为什么优化这里

用户 L40S 的新版 GT 8 步实测：每窗计算 0.933024 秒，含输入 0.947234 秒。
其中 soft render + full-grid CE/Lovász 0.416069 秒，占据 backward 0.297100 秒，占计算时间约 76%。
新的候选不去改已经优化好的硬推理，主要减少 loss 中重复构造的完整概率梯度和 Lovász 排序。

- `prepared_combined`：提前在 CPU 整理合法标签/出现类别，合并 CE 与 Lovász 一阶反传；只分配一次完整概率梯度。合法网格位置全部参与。
- `prepared_sparse`：在上一项基础上，只压缩 Lovász 中数学上零贡献的背景排序尾部。
  原概率恰为零的 GT 前景仍保留最大误差和梯度；非零概率仍稳定排序，用完整 GT 前景总数计算 Jaccard 权重。
  没有 voxel sampling/top-k/近似 loss，也不会丢掉漏检的监督。
  当概率不满足证明条件或较密集时退回完整排序；检测本身有开销。

这是同一目标的一阶梯度实现，不支持二阶导数。浮点加总顺序可能不同，数值一致不声称逐字节一致。
GT 标签仅用于 loss，不进入模型/候选/owner 构造；训练连续合成、推理硬合成仍有原有差异。

## 已做的本地实验

Windows 项目 CPU 环境，单线程，18 类 × 640000 体素，真实 loss 前向与反传；候选计时包含标签准备。
同概率/标签比较 loss 及每个体素梯度，交替执行顺序，暖机不计时。

| 合成概率分布 | 当前 fused | combined | sparse | sparse 加速 |
| --- | ---: | ---: | ---: | ---: |
| 每体素两个非零类别，四遍交替 | 0.944153s | 0.754193s | 0.349099s | 2.705× |
| 全类别密集随机概率，两遍交替 | 1.739012s | 1.532281s | 1.669256s | 1.042× |

密集负对照下 combined 约 1.135×，sparse 因退回检测反而弱于 combined。
这些仅是 CPU synthetic loss 测速，不能外推整轮训练、真实 L40S 提速、预测精度或推理 FPS。
本地真实小网格 GT/Pred 六帧更新也检查了 loss、所有参数/Adam moments、XY/yaw/action 梯度链；服务器全网格 CUDA 尚未验证。

## 服务器一趟 A/B/C

先在训练终端按一次 Ctrl-C，等安全保存结束。新终端激活 OccFM 并设 `GT_RUN_DIR` 为已有 GT 目录，而非旧 paired 目录。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
export GT_RUN_DIR=/实际GT训练目录
export GT_SPEED_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt_speed_probe_$(date +%Y%m%d_%H%M%S)
bash tools/joint_occ_experiments/performance/run_probe.sh
cat "$GT_SPEED_OUT/summary.txt"
```

若还在原终端且 `$RUN_DIR` 保留：`export GT_RUN_DIR="$RUN_DIR"`。
若旧启动使用了 `DATAROOT/E14/TRAIN_CACHE/DEV_CACHE/CCR_HISTORY_CACHE/CCR_VAL_CACHE`，此处保留相同变量。

工具读取原 `last.pt` 并写入新目录的不可变快照，用相同权重/Adam/RNG/原断点后八批做三路真实训练计算；每路重置，两遍反向次序。
检查损失、梯度链和最终所有参数/Adam 的数值一致性，报告秒/窗、CUDA allocated/reserved 峰值。
标签准备计入计算；共享输入 I/O、首次使用/暖机、状态克隆和保存不计入吞吐。这不是 raw-input E2E 或正式 FPS。
源 checkpoint、cursor、训练 RNG 和几何缓存只读；临时优化步不保存 `last.pt` 或成为科学训练更新。
候选门槛为数值一致且两遍都更快、平均至少 10% 提速；不根据 DEV 选后端，不自动改变训练配置。
数值超标不再中断其余测速：按原 rtol2e-4/atol2e-5 报告失败参数路径、数量和最大误差，并令候选 gate=False。
第一批另对真实1s概率图检查 loss 数值及逐体素梯度，隔离 loss 内核与BF16网络/Adam累计舍入。
第二遍原版重复也核查最终状态；原版本身不可复现时不能给候选通过判定。
逐阶段结果写新 `progress.jsonl`，结构、数据、非有限loss等运行错误仍报错，不绕过检查。

服务器第一次测试曾在 combined 连续八步后超标退出：首个报告的128维张量中1个元素最大绝对差2.2003544e-5，允许2e-5；其余张量是否超标当时未完整统计，只有原版0.636530s/窗结果。
代码检查发现候选将 CE/Lovász 导数分别除归一化常数再相加，而原图是先相加再统一除。
已恢复原图运算顺序；不修改任何数值预算，不据此宣称真实CUDA已通过或错误已全部排除。
本地单独概率梯度增加非单位upstream的精确一致测试，真实小网格及数值超标仍继续测速、绝不标通过的回归保留。

得到服务器结果之前，继续训练仍用旧兼容命令：

```bash
RUN_DIR="$GT_RUN_DIR" RESUME="$GT_RUN_DIR/last.pt" bash tools/joint_occ_experiments/run_gt.sh
```

上面的 resume **不会自动使用新 loss**。新后端需要服务器测速有效后另行显式接入，并记录执行 provenance；不能把 synthetic 收益当作已经优化了真实训练。
