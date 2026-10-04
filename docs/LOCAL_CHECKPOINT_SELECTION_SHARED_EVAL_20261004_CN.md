# Local 多目录 checkpoint 筛选与共享评估

本次不改网络、loss、batch/source、学习率、TRAIN采样，也不回改已经完成的15/20轮实验。
仅审计和优化评估；不删除目录，不自动部署，不用完整验证集挑轮次。

## 一次执行

先等20轮训练 **以及最后的TRAIN校准/dev512评估** 正常结束。激活 `OccFM` 后：

```bash
bash tools/real_motion/run_p0_f9_joint_checkpoint_selection.sh \
  /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full20_history4_extend_20261004_155940_1023 8
```

脚本沿 `execution_contract.json -> arguments.resume` 追溯真实来源；不是按目录名、修改时间或“latest”猜来源。
继承的每轮dev64记录全部列出，四/六历史和非本续训链目录不能混入同一排名。
epoch_snapshot 优先；同轮、同update的candidate/last按权重指纹去重，权重不同直接报错。
丢失的早期权重明确列在 `missing_weight_epochs`：不能保证选到已经删除的全局最优轮次，也不自动重训来补权重。

默认最多8个候选：保留最终轮/原15轮、dev64整体高分与Moving高分，再按整体分数补足。
**已有dev64分数不重复跑**，这些候选一次共享dev512评估。
所有候选固定 `0.5/0.5/REMOVE-off`，不将TRAIN校准candidate与未校准epoch的不同阈值混排名。

## 共享与不共享

共享一个当前/一个下一窗口的原始数据和固定因果准备：历史关联、固定Strong状态、静态证据、历史inverse/membership。
Moving support只用于统计，独立于网络输入。冻结E14 counts只算一次，同窗口其余模型复用。
不同权重的XY/yaw、搬运、owner/fallback、candidate、特征和概率 **全部重算**。
候选顺序、FP64变换、原推理batch256、完整population、1/2/3s与Moving协议不变。
不创建共享磁盘几何缓存，也不把模型预测、GT特征、标签或learned pose写入训练缓存。

CPU下一小批特征与GPU当前网络计算重叠。logits/校准权重有限检查在GPU累计，到horizon返回前检查；默认训练检查不变。
每个模型首窗三个report horizon都和原CPU路径 **逐项比较概率**，失败即停止；不会放宽阈值继续选模型。
可用 `--reference-inference` 关闭新的CPU prefetch/deferred checks。无法在本地CPU测试中宣称真实CUDA提速多少。

## 断点和输出

`selection.txt/json`：全轮dev64分数、真实续训链、可用/缺失权重、筛选原因。
`summary.txt/comparison.json`：dev512统一阈值四路结果、按mIoU排名、Moving安全候选、stage计时。
`evaluation_state.json`：最多每8窗口原子保存小的指标计数；不保存raw/GT体积或features。
Ctrl-C可能丢失最后不足8窗口的工作，但不会混入不同模型的半窗口计数。恢复用本次实际OUT：

```bash
PY="$(command -v python)"
"$PY" -u tools/real_motion/eval_p0_f9_joint_checkpoints.py \
  --run-dir "$RUN20" --runs-root /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns \
  --population dev512 --out-dir "$OUT" --cpu-workers 8 --resume
```

RUN20/OUT是实际目录，不猜时间戳；恢复核对原snapshot、selection、population顺序、阈值和计数fingerprint。
中断状态不会生成完成排名；完成后拒绝重复恢复。
默认Moving安全条件是aggregate MovingMicro相对E14非负；若无候选通过，就返回null，而不是偷偷放宽条件。
最高整体mIoU和Moving安全最优分别报告，不自动复制best.pt，不自动把“排名第一”等同论文有效。
E14仍是legacy六历史参考，不是同预算四历史消融。
先固定权重/阈值，再单独做一次full4369；full4369包含selection dev512，不能冒称独立test。

单模型亦支持新CPU预取/概率gate：

```bash
"$PY" -u tools/real_motion/eval_p0_f9_joint_causal_columns.py \
  ... --column-feature-backend cpu --optimized-inference --fixed-monitor-thresholds
```

原单模型包装脚本也支持 `FULL_JOINT_EVAL_OPTIMIZED=1` 和
`FULL_JOINT_EVAL_FIXED_MONITOR=1` 两个显式开关。前者启用首窗概率检查后的新推理，
后者保证与本次筛选相同的阈值。先确定checkpoint，再指定 `full4369`；默认仍不自动跑完整集。

已被拒绝的 `SWFM_LOCAL_FAST_SUPERVISION` / `SWFM_LOCAL_STATIC_ROI` 默认关闭，
评估脚本也强制关闭。这只是防止旧shell遗留实验开关进入正式路径，不重新训练或改写权重。

这次主要减少 **多个checkpoint** 的重复准备/参考统计，单个full4369的网络预测不能凭空省略。
计时注明本次与累计耗时、复用前缀、模型窗口数；没有服务器测量之前不承诺分钟数或倍数。
