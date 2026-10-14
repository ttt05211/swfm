# 现有 Refiner500 污染对照：执行提速后续训

## 来自服务器的瓶颈

原任务 `stc_empirical_refiner500_20261014_194954`：clean 最后64更新约 .269632 compute + .006891 input 秒/窗；empirical 42–105约 .869472 compute + 1.382326 simulation/rebuild 等待 = 2.2518秒/窗，其中 fresh Strong .616620，剩余网络/loss/反传约 .252852。CPU simulation1.020461 / geometry1.858543 是并发窗口的累计CPU时间，不能再加到上述墙时。

问题不是没有历史缓存或网络反传慢：污染输入必须重建几何，原包装器只在当前batch内并行，然后训练线程等待；Strong 又重复提取上一帧组件并重新匹配，而这些已在 `Predictor.prepare` 完成。

## 新入口做什么

只新增 sibling `tools/stc_noise_execution/`，旧 `stc_noise_adaptation/` / 模型 / trainer / cache namespaces 不修改。

1. 有界跨 batch CPU lookahead：全局最多4个窗口工作线程，队列2组＋生产者1组，最多3个B4组在消费批之前准备（加消费批最多4组）；已有原始数据预取也保持有界。保持原group/window顺序，未来窗口准备与当前GPU训练重叠。只CPU工作在producer/worker，GPU Strong和模型/Adam仍主线程执行。
2. fresh Strong 复用污染历史刚构建的 `previous/velocities/current/source_world_points`，不重复上一帧CC和匹配；原FP64位姿、CUDA逆变换、native majority、CLEAR和六帧合成不变。第一个真实改变的TRAIN窗口额外对照原Strong六帧byte，失败不更新该步、不放宽预算或静默fallback。
3. 跳过后续未读取的 wrong-class 随机场的 Gaussian filter / rank sort；保留完全相同 `rng.standard_normal(shape)` 消耗，后续概率/随机数/污染体素不变。固定旧augmentation规范化文本SHA，源码改动拒绝套此优化。12个并发固定seed、selected/unselected两类逐字节污染与原始计数一致。

第三项本地200×200×16合成网格3窗仅模拟CPU计时：原1.622056、新1.624574秒/选中窗（0.99845倍），byte/audit一致；没有稳定单项提速，不将其包装成收益。主要期望来自流水重叠与重复CC省略，真实L40S倍率以续训日志为准。

消费端 `simulation_and_rebuild_wait` 只记实际等待，producer原input等待不重复算墙时；原augmentation累计CPU counters继续保留。可能启动时需等首批，首次Strong一致性检查也会额外重算一次；不能用第一步外推稳态，也不把detail或累计CPU时间重复相加。Ctrl-C关闭有界队列/取消尚未开始的任务、等待正在运行的CPU任务，可能稍等再保存；不读整个epoch到内存、不新增磁盘几何或学习特征bank。

## 原目录 / Adam / 科研设置保持

新workflow从原 `workflow.json` 重放全部旧参数，仍经过原workflow输入/数据/源码/STC inventory/初始化检查。已完成clean自动跳过；empirical恢复原 `train/last.pt` 的权重、Adam、RNG、update/batch cursor和原500余弦，不额外训500步。

没有迁移/伪造旧checkpoint contract、改旧fingerprint、增加window batch、改LR或污染配方。新增执行源码SHA、恢复来源SHA/update和六帧Strong gate单独存每路 `execution_speed_resume.json`，旧科学核心代码contract保持而执行变化明确记账。CPU/CUDA训练原有非确定性仍存在，不承诺最终权重位级重现或精度不变；优化不改变前向数学、样本与loss。

噪声/评估科研边界同原实验：VAL拟合统计用于TRAIN模拟，不是严格无目标域信息zero-shot；固定remaining64与旧DEV64不同，不能直接拼旧分数。本次没有重训/扩大正式实验。

## 用户这次实际目录续训命令

先原终端Ctrl-C一次，等保存并返回提示符，再运行：

```bash
cd /root/nas/occ/swfm
conda activate OccFM
git pull --ff-only
export CALIB_RUN_DIR="$PWD/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954"
bash tools/stc_noise_execution/run.sh && cat "$CALIB_RUN_DIR/summary.txt"
```

不需要再次设置NOISE_MODEL/GT_CKPT/CPU_WORKERS/CALIB_RESUME：从原workflow参数读入；不能趁续训改科学参数。若再中断，仍同一命令继续。已全部完成则不会训练。可选 `--max-updates 128` 指**总更新停止位置**，不是增加128步；默认继续到原500。

回传新的最近64更新 compute/input/simulation_and_rebuild_wait/fresh_Strong，才能确认真实收益。没有访问服务器或代启动训练。

本地专项及旧污染/历史重建共30项通过（18.07s）：CPU/CUDA真实小图、污染bytes/整数计数、FIFO预取与线程错误/取消清理、真正原版runner第1步checkpoint到新入口第2步Adam/RNG恢复、clean结果和workflow元数据字节不改。Bash语法通过。测试不代表真实L40S速度或最终精度。
