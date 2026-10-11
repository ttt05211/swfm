# GT-only：只合并两张未来帧，不合批网络

先前 packed_h2 服务器0.703916→0.475507s/窗，实际1.48035倍，但warm/八步模型+Adam数值检查失败；原版自身repeat亦失败。保留FAIL，不采用、不放宽容差，也不能据此断言最终精度下降或无害。

本次只保留有希望的未来帧分组，收回网络执行变化：

- WM、ego、CCR仍按原单窗口形状前向；每窗口完整shared backward后才运行下一窗口，不跨窗口拼网络/图。
- 单窗依次渲染0.5/1s、1.5/2s、2.5/3s，两张未来帧并行完整soft rendering、CE/Lovasz和occupancy backward。
- 每帧独立有效体素CE均值与present-class Lovasz均值，各自除以原窗口数×6。保留source几何/existence/R10 ego、clip5和一次全局Adam。
- 全局batch4/source128、原数据顺序/步数/余弦/RNG/游标不变。旧history cache只读，无新bank或缓存构建。
- 原top-level trainer和快速硬推理/评估未改；新代码只用于训练。不声称字节无损，也不改/借用正式FPS。

## 先测，不自动训练

只跑current_fused vs sequential_h2两路线，同原断点/八批/Adam/RNG，AB/BA两遍。预热、实际首窗renderer概率/loss/边界梯度及warm更新审计在计时外；计时含目标准备，共享输入I/O不计。

原模型/Adam预算rtol2e-4/atol2e-5不变；renderer概率/loss/梯度预算沿用上一轮。需warm和最终数值、原版repeat、两遍均快、平均≥1.1倍及reserved显存≤90%全部通过。FAIL仍FAIL，不自动推广或处理掉失败张量。OOM/非有限/结构错误停止，不半步重试。

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch origin feature/v22-surface-aware-ccr
git merge --ff-only FETCH_HEAD
export GT_RUN_DIR=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt3_fast_20261011_123726
export HORIZON_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt_horizon_probe_$(date +%Y%m%d_%H%M%S)
unset RESUME HORIZON_PROOF
bash tools/joint_occ_experiments/horizons/run.sh probe
cat "$HORIZON_OUT/summary.txt"
```

不重新跑失败的packed或prepared/sparse。源last.pt取实际已保存cursor，测速不发布科学更新，也不修改原训练文件。

## 通过后显式接续

仅当前窄版成功speed.json可以授权导入；旧packed_h2的报告不能使用。成功前不要运行以下续训。

```bash
export HORIZON_PROOF="$HORIZON_OUT/speed.json"
export HORIZON_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_occ_ego/gt_horizon_resume_$(date +%Y%m%d_%H%M%S)
bash tools/joint_occ_experiments/horizons/run.sh resume
```

新目录导入原weights/Adam/RNG/epoch/batch/LR/科学契约，只有implementation明确变更，记录horizon_import.json。source SHA变了需重新检查，不能使用过期报告。Ctrl-C/SIGTERM完成当前Adam步后保存，勿连续中断或kill -9。

以后同一窄版目录恢复不再需要proof：保留或重新设置HORIZON_OUT为该实际训练目录，再执行：

```bash
export RESUME="$HORIZON_OUT/last.pt"
bash tools/joint_occ_experiments/horizons/run.sh resume
```

若新候选失败，原训练仍可继续：

```bash
export RUN_DIR="$GT_RUN_DIR"
export RESUME="$RUN_DIR/last.pt"
bash tools/joint_occ_experiments/run_gt.sh
```

本地7项窄版+60项原相关CPU回归共67 passed；覆盖原前后向交替、空窗口、两次Adam数值/损失/梯度链接、两路线只读reset、失败receipt拒绝、旧导入及新同目录恢复。CLI help/Bash/diff通过。不是完整CI，未代跑nuScenes/L40S，CPU通过不保证BF16/CUDA通过或提速。
