# 现有三模型的模拟污染域只读诊断

不再训练。读取已完成的 clean、Refiner-only empirical、full_empirical 各500步 checkpoint，重建原TRAIN20%分组的前500组，排除它们实际用过的**整个场景**，从其余TRAIN场景固定均衡取64窗。若不足64窗直接拒绝，不偷偷降为仅窗口不重叠。GT主体预训练已用过完整TRAIN，所以只是适配阶段场景留出，不是独立测试。

每窗 clean_dense / empirical_mix 两种历史均 dense-valid；后者保持原seed与原50%随机选窗及噪声配方，三个模型共享同一个噪声实现。额外列出实际抽中污染的 corrupted_only 子集；不是调噪声强度或选择分数好看的窗口。重新构造污染后的CC/匹配/注册/KTA/Strong，学习特征分别实时算，不跨模型复用。未来GT占据仅评分，未来GT ego为明确条件。预测完整六帧，报告1/2/3s标准全网格IoU/mIoU，不计算Moving或FPS。

原checkpoint、Adam/RNG和纯历史缓存只读。新目录 contract/state/results/summary，按完整窗口保存整数混淆并可恢复。不会写模型、训练器、学习特征bank；首次改变历史仍核六帧Strong执行一致性。自动定位 full run 仅接受唯一的匹配原workflow且完成的任务，不按分数选；多匹配用FULL_NOISE_DIR显式指定。

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only
export REFERENCE_DIR="$PWD/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954"
# 若仍保留上次FULL_NOISE_DIR，会直接使用；否则只接受唯一匹配的完成任务。
export SYNTH_EVAL_DIR="$PWD/outputs/p0_f9_shared_history/stc_synthetic_holdout64_$(date +%Y%m%d_%H%M%S)"
unset SYNTH_EVAL_RESUME
bash tools/stc_noise_holdout/run.sh && cat "$SYNTH_EVAL_DIR/summary.txt"
```

中断后设置原SYNTH_EVAL_DIR和SYNTH_EVAL_RESUME=1，运行同入口，不重新训练。

重点看 corrupted_only 的 empirical/full_empirical 相对clean差：模拟域明显改善但真实STC仍无收益，支持模拟—真实迁移不足；模拟域也无收益，说明当前适配未展示留出纠错能力，预算/优化/噪声信号仍待区分。不能凭这一次诊断唯一归因、保证延长训练有效或将分数作为真实STC正式结果。full vs clean同时改变污染和解冻范围，不是完整的全解冻clean控制。
