# Local eval 同窗口速度对比

这不是重新训练、阈值/权重筛选或 full4369 重跑。使用已冻结的 epoch19 快照及 frozen dev64 的确定性 scene-balanced 32 窗口，固定 4 历史、6 未来、报告 1/2/3 秒、网络 batch256 和阈值 0.5/0.5/REMOVE-off。

## 一趟对比

`bash tools/real_motion/run_p0_f9_joint_eval_speed.sh`

默认服务器 checkpoint 为已确认的 `/root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt`，也可显式传入 checkpoint。输出为独立的 `outputs/p0_f9_joint_causal_columns/eval_speed_*`。

- serial_raw：单窗口预取，旧逐字段上传/逐 batch 回传。
- parallel_raw：四窗口 CPU 预取，旧传输方式。
- parallel_buffered：四窗口预取、CPU 字段字节打包单次上传、8MiB 有界概率汇总回传、Local source finite 检查整 horizon 汇总。

两轮按正序/反序测试。每次相同的两窗口 warmup，不计入吞吐；各 trial 重置相同 256MiB 帧缓存。首次 Strong all-six renderer 和 probability byte exactness 检查仍运行。每种模式的完整整数指标、addition/quality、置信度累计和 source audit 状态必须一致，否则不生成成功测速摘要。旧模式和新模式均包含本次减少重复 previous Strong 提取的 CPU 改进，因此三模式的 speedup **不包含**这一项。

`summary.txt` 打印各模式的秒/窗口、速度比和最大主线程阶段。`speed.json` 含每轮耗时及 raw I/O / Strong / history-static CPU worker 阶段细节，`progress.jsonl` 含逐窗口记录。所有结果标记 speed-only，不改原 checkpoint、阈值，不进行选择/部署。

## 不变与安全

- 不改 source/候选数量、网络 batch/shape、FP64 几何、floor、模型权重、概率计算、ADD/REMOVE 和 Moving 指标。
- 概率缓存只包含最终 FP32 输出；8MiB 缓存 + 至多 8MiB 拼接暂存；不缓存 logits、learned poses/features，CPU 候选仍不接触未来 GT。
- 输入打包只复制字节，保留每字段 dtype、shape、对齐和顺序；CUDA feature 已驻留的数据不回传。
- 训练/direct forward 默认有限值检查不变；仅明确优化推理会延期检查，并在返回整个 horizon 前 fail-closed。
- 只复用同窗口已算过的 previous Strong 实例；不跨训练步骤保存 learned 数据。
- Ctrl+C/TERM 在窗口边界退出，没有正式完成指标。检测到其他训练/评估/校准进程时测速拒绝启动，不给其发信号。
- 小样本及 OS 页缓存仍影响吞吐，不将小样本速度直接宣称 full 数据的测量结果。

如实测 IO 合并不快，可运行正式 evaluator 时设置 `FULL_JOINT_EVAL_IO_OPTIMIZED=0`，保留四窗口预取但回退旧上传/回传。整个预取也可用 `FULL_JOINT_EVAL_RAW_WORKERS=1 FULL_JOINT_EVAL_RAW_DEPTH=1` 回退。

本地没有真实 nuScenes 文件及 CUDA。CPU 合成算子（200×200×16、12 个 source、4 历史）旧/复用 previous 的历史注册约 59.5/56.0ms，约 6% 改善，只是该算子，不是服务器整条 eval 的吞吐。真实数据速度以服务器上述一趟对比为准。
