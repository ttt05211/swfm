# 当前平均模型：full4369 只读评估

用户已批准评估全集。DEV512 对照选择的是第 5/6/8/12/14 轮整网等权平均：
IoU52.372722 / mIoU40.432276 / MovingMacro25.718056 / MovingMicro31.486519。
这些不是全集分数。旧 Local full4369 mIoU44.028927 不能与 DEV512 的40.432276直接比较。

本入口只加载已完成 DEV512 对照中的平均文件，不重新平均，也不重新跑其他单轮。
4历史→6未来，weighted ADD raw0.5 / REMOVE-off；网络、support、权重与阈值不改。
报告全集 Transport、静态/动态/Joint 四指标、1/2/3s、类别、场景和编辑质量。
原第20轮 DEV512 报告不混入全集；full4369包含研究开发子集，不称独立测试。

## 运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-surface-aware-ccr &&
git merge --ff-only FETCH_HEAD &&
bash tools/real_motion/run_p0_f9_joint_surface_mean_full.sh
```

默认在原 runs root 自动查找已完成的同训练锚点 DEV512 comparison。
同权重的多个完整副本仅按时间挑来源，不看分数；不同权重冲突则停止。
需要显式来源时：

```bash
SURFACE_MEAN_SOURCE=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/checkpoint_comparison_实际时间_PID \
bash tools/real_motion/run_p0_f9_joint_surface_mean_full.sh
```

默认锚点仍为 `full20_nohup_resume_20261008_190410_787`，可用 `SURFACE_COMPARE_RUN` 修改。
VAL require缓存仍为 `/root/nas/occ/swfm/cache/p0_f9_ccr_history_geometry_val_v1`，可用 `CCR_VAL_CACHE` 修改。
缺文件/namespace不一致立即停止，不重建大缓存。源 checkpoint、optimizer/RNG、缓存和已完成 comparison 全部只读。
新输出 `mean_full4369_时间_PID` 中只有评估进度和报告；原平均文件留在 comparison 目录，不复制或改写。

## 中断与恢复

四窗口有界历史预取；每8个完整窗口保存整数统计，已保存窗口不重算。
Ctrl+C/SIGTERM等当前完整窗口结束后停止；kill-9只能回到最近周期保存处。
恢复同一个**全集评估输出**，不是训练/DEV512 comparison目录：

```bash
SURFACE_MEAN_FULL_OUT=/root/nas/occ/swfm/outputs/p0_f9_joint_surface_ccr/mean_full4369_实际时间_PID \
SURFACE_MEAN_FULL_RESUME=1 bash tools/real_motion/run_p0_f9_joint_surface_mean_full.sh
```

权重、source comparison、配置、人口、实现指纹、VAL namespace任一变化都拒绝混合进度。
完成后拒绝静默重跑或覆盖。

`summary.txt` 可直接贴回；完整信息在 `full_validation.json`。
这是质量评估计时，不是正式 FPS；本次不额外启动 FPS、训练或部署。
