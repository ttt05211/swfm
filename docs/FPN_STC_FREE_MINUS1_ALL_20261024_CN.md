# 固定 FREE−1：STC 全 VAL4369

用户明确“直接把之前那个−1测一下全集”。独立入口 `tools/fpn_stc_bias_all/run.sh`；不接入任何后续失败的embedding、norm、末层校准、滑窗或其他候选。旧448及原全集入口不改。

保持完整TRAIN20430×2最终epoch2（2520442参数，SHA `cedcf7c35123489b0b7e3e0fdca1367241c3baf28c18fb1b8db6e0ff6bef138f`）、STC age0.5、GT future ego、四历史六未来、dense-valid、全网格1/2/3s。同原全集4369窗/150场景，不排除原64、不裁剪缺STC，不使用外部planner的人口。

每窗只运行一次网络，通过与448入口完全相同的FP32读出得到原版与FREE−1两路，再读未来GT和真实Moving标注评分。只输出空类logit减1，其余17类相对次序不变，不改权重或网络。平均/逐时距 mIoU、IoU、Moving及差值都报告，整数counts累加后计算，不平均子集得分。

这是前32验证标签选定参数的**有标签目标域输出校准**；全集包含这些场景，不称独立测试或zero-calibration成绩。此次没有任何拟合、选择或继续调参，也不保证达标。

源权重、缓存与默认模型只读，独立源快照/输出合同；每8窗和退出保存完整窗口状态。`STC_BIAS_ALL_DIR`独立、`STC_BIAS_ALL_GPU=0`单卡、`STC_BIAS_ALL_RESUME=0/1`。恢复必须用同一目录与配置，不重开任务。没有复用448的累积counts混入新全集；新全集两路同窗同前向完整评估。

## 后台启动

```bash
cd /root/nas/occ/swfm
conda activate OccFM
export FPN_UNET_FULL_DIR=/root/nas/occ/swfm/outputs/p0_f9_shared_history/fpn_unet_full_2ep_20261021_112202
export STC_BIAS_ALL_GPU=0 STC_BIAS_ALL_RESUME=0
export STC_BIAS_ALL_DIR="$PWD/outputs/p0_f9_shared_history/stc_free_minus1_all4369_$(date +%Y%m%d_%H%M%S)"
nohup bash tools/fpn_stc_bias_all/run.sh >"${STC_BIAS_ALL_DIR}.log" 2>&1 &
echo "PID=$! OUTPUT=$STC_BIAS_ALL_DIR LOG=${STC_BIAS_ALL_DIR}.log"
```

`tail -f "${STC_BIAS_ALL_DIR}.log"`查看进度，Ctrl-C只停止日志查看；`cat "$STC_BIAS_ALL_DIR/progress.json"`查看已完成窗数，完成后`cat "$STC_BIAS_ALL_DIR/summary.txt"`。终端重开后需恢复原目录变量，不要生成新时间戳覆盖概念。

中断后设原`STC_BIAS_ALL_DIR`与`STC_BIAS_ALL_RESUME=1`，同样nohup命令续跑。正常SIGINT/SIGTERM会在当前完整窗口结束后保存；不强杀或干预其他任务。评估耗时不是FPS。
