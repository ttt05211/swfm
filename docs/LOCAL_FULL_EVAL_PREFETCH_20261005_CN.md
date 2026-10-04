# full Local 评估：补齐指标与有界 CPU 窗口预取

## 证据与修改

epoch19 full4369 原结果耗时5556.64s，约92.61分钟、1.272s/window。末两条日志 main compute 为0.536/0.507s，但 input_wait 为1.088/1.282s，分别占总耗时67%/72%；CPU下一窗口 fixed geometry/raw准备供给不足，而不是仅因显存未占满。末两条不是全量统计，读取原 progress 的摘要工具会计算全量平均。

- 原 IoU/MovingMacro/MovingMicro 已保存到 `evaluation.json`，只是旧 `summary.txt` 少打印。新摘要输出全部四项整体及1/2/3s指标，同时明确 scene delta 的参考是 current transport，非 E14。
- 新只读摘要工具不加载 Torch/checkpoint、不使用GPU、不修改已有文件，不需要重跑来获取 IoU 或全量耗时统计。
- 新 full wrapper 默认最多4个 CPU preparation workers、4个 NEXT窗口排队；按原 identity/order消费结果，错误在原窗口位置传播，关闭/中断会回收 workers。每个几何任务缩小内层 worker预算，降低嵌套线程过多的风险。若争用或内存紧张，可显式降到2或1。
- 模型前向/ CUDA仍只在调用线程；learned trajectory、owner/fallback、候选与预测不跨模型缓存。网络batch仍为256，6个Strong horizon和FP64几何算法不变，保留首次概率/renderer/整数指标exactness检查；没有通过减少窗口、候选、指标或帧数提速。
- CLI与训练/阈值扫描/多checkpoint调用默认仍为原单窗口预取，只有 full wrapper 显式启用4窗口。训练优化步数、RNG、LR与checkpoint格式不变。
- 新 progress 增加 raw worker（可重叠）的 I/O、Strong、background、history/static stages 与串行 candidate wait/probability/metric/moving耗时。worker和 inclusive stage不可直接相加当wall time。

没有在本地GPU或真实nuScenes测量新版吞吐，不承诺具体加速倍率。CPU/GPU竞争、磁盘、GIL、内存带宽都可能限制并行效果。local CPU exactness/ordering tests与首用真实数据gate只能证明一致性，不是服务器测速结果。

## 已完成结果：仅补打印，不重跑

```bash
conda activate OccFM
cd /root/nas/occ/swfm
"$(command -v python)" tools/real_motion/summarize_p0_f9_joint_evaluation.py \
  /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full20_history4_extend_20261004_155940_1023/eval_full4369_20261004_224716_75424
```

## 下一次评估

更新代码后沿用 `run_p0_f9_joint_interim_eval.sh`，默认开启 byte-exact optimized inference 与4-worker有界 raw预取。不自动启动新full，也不重算原结果。

```bash
FULL_JOINT_EVAL_RAW_WORKERS=4 FULL_JOINT_EVAL_RAW_DEPTH=4 \
FULL_JOINT_EVAL_OPTIMIZED=1 FULL_JOINT_EVAL_FIXED_MONITOR=1 \
bash tools/real_motion/run_p0_f9_joint_interim_eval.sh \
  /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/full20_history4_extend_20261004_155940_1023 \
  full4369 \
  /root/nas/occ/swfm/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt
```

这是未来需要完整验证时的入口，不要求立刻重跑本次full。与阈值校准并行时应降低其中一项的worker预算，避免CPU配额竞争；两个进程的同时elapsed不能与单独运行直接比较。
