# Local 执行优化、eval 吞吐与完整六帧 FPS

## 边界与不做的事情

- 不改训练、checkpoint 参数/键、阈值、候选集合、原 decoder batch256、BF16 或 source 几何。
- 不把 patch CNN 换成全图 CNN：局部 patch 的 padding 与全图 padding 不等价。
- 不复用 query/context/source/预测；不跨窗口缓存历史编码。
- 不重试失败方案、不自动晋升后端、不跑 full4369。

## 三项可选优化

1. `PatchMemory`：完整 history+flags **字节相等**才可共享历史 embedding/局部 CNN 编码。
   Python bytes 键在 hash 冲突时仍比较完整内容；不是用坐标近似或 GT 识别相同 patch。
   每 window LRU 有效载荷上限64MiB，包含键及 GPU memory/mask（另有 Python 元数据和当前 batch 暂存）。
   同一窗口不同 horizon 的完整历史字节若相同也可复用，窗口结束强制释放；没有跨窗口/跨 checkpoint 复用。
   GPU memory 用有界 slab，两次批量 index_copy 写入，避免每条 miss 独立 clone 带来的上千次调度。
   当前 query 的 context/source/attention/action head 仍逐行执行，输出顺序保持原样。
   只相邻但不完全相同的 patch 不直接复用；记录真实命中率，不预设收益。
   缩小 encoder batch 可能改变浮点内核选择，所以必须通过实际 CUDA 概率字节门槛；
   数字变化即拒绝该模式，不能因最终 argmax 没变而声称严格等价。
2. `ColumnExecution`：原 forward 或 memory 后的 decoder+概率校准由有界 CUDA Graph 重放；
   固定 shape，最多两个图，尾 batch 未缓存时原 eager 执行，不填充/截断 query。
   无法 capture/OOM 时记录原因并使用原数学路径；不是隐藏的加速成功。
   capture 内关闭 autocast 权重缓存，防止图引用到逐块 autocast context 退出后释放的临时 cast。
   仅 eval+inference_mode；参数和 calibration buffer version 变化立即报错；session 结束释放图。
3. `AsyncProbabilityReadback`：只优化 D2H，不重新捆绑实测较慢的打包上传。
   有界 pinned 输出8MiB，独立 copy stream，horizon 末一次等待。
   Graph 重用输出 buffer 前必须 clone；`record_stream` 本身不能防止 graph 覆写。
   CPU/超预算保持原回传路径；异常也等待 copy stream，避免生命周期问题。

## 一次测速命令

```bash
conda activate OccFM
cd /root/nas/occ/swfm
git fetch https://ghfast.top/https://github.com/ttt05211/swfm.git feature/v22-causal-emergence-tokens
git merge --ff-only FETCH_HEAD
bash tools/real_motion/run_p0_f9_joint_execution_speed_fps.sh
```

默认 epoch19，固定已选候选阈值 `(0.5,0.5,0.95)`。不搜索阈值、不改训练文件。
eval 用冻结 dev64 中确定性 scene-balanced32；四种模式正反序各一次：
`parallel_raw / async_readback / graph_async / reuse_graph_async`。
预热时逐元素概率检查，计时窗口也检查概率 SHA 和完整整数指标状态。
实际数据不等价的模式明确 rejected，其余模式继续完成。

同一命令另在18个窗口测完整0.5/1/1.5/2/2.5/3s预测：

```
FPS = 6 / 单窗口生成六个完整 joint occupancy 的秒数
```

计时包括 Strong/KTA 六帧 prior 重建、live transport、全部六个 horizon 的候选/特征/column
网络、动作决策及六帧 dense composition；不包含磁盘/GT/指标/额外E14对照。
输入为已准备的因果 source tensors、注册历史、静态 memory；**不是 raw-sensor 端到端 FPS**。
几何输入准备（有 I/O 和旧 prior prefill）单独报告，不能与 generation 相加冒充无重复的 E2E。
Graph warmup/capture 在稳态计时外单独列出；内存编码 cache 每次六帧窗口预测重建，不以第二遍偷用 learned cache。
同卡/同窗口另外测原 E14 的六帧 generation。E14六历史、新Local四历史，明确不是匹配预算对照。
检查所有六帧 dense bytes **以及所有 horizon probability bytes**，不可把只修复1/2/3s当成六帧FPS。

另外用最多6个相同窗口实测 baseline 与已验证最快模式的 **strict loaded-history FPS**：
计时从内存中已加载的4帧 occupancy/visibility/pose 和 resident source tensors 开始，
清除 prepared causal state，重新执行 source 提取/匹配、历史注册、静态证据、prior、网络和六帧合成。
这是真实独立计时，不是把重叠 worker 阶段时间相加；不含磁盘/GT/指标。
cached source tensor 的提取仍在边界外，故也不宣称 raw-sensor 端到端FPS。
这样可看到新增固定几何处理的代价，而不是只报告较乐观的 prepared generation FPS。

结果为新目录 `summary.txt`、`speed.json`、`execution_progress.jsonl` 和只读 checkpoint snapshot。
summary 分别打印 eval 秒/窗口、六帧延迟与帧FPS、patch命中率、graph重放/失败及 CPU host 分段耗时。
host 调度时间不是 CUDA 活跃利用率；execution 的 network_forward_host 包含 history_encoding
以及 graph 内校准，不要将包含项重复相加。短样本/OS缓存波动仍存在，不承诺 full ETA。

通过本次真实数据字节/整数门槛后，独立 evaluator 可显式指定
`--optimized-inference --legacy-chunk-io --column-feature-backend cpu --execution-backend graph_async`
（或实际通过且更快的 `async_readback`/`reuse_graph_async`）。默认仍为 `eager`，不会自动切换。
现有 `run_p0_f9_joint_interim_eval.sh` 也可显式设置
`FULL_JOINT_EVAL_EXECUTION_BACKEND=graph_async`；不设置时原流程完全不变。
首次三个 horizon 仍与原概率逐元素核对；compact patch encoder 的矩阵批量形状会改变，
即使一次测速通过也不等于对所有未来输入的浮点字节相等做了数学证明，因此 reuse 仍属实验选项。

## 本地与服务器验证

Windows 必须用项目安全 Python；不得运行损坏的 Anaconda base。
本地 CPU 测试覆盖原训练反传/状态键、非恒定概率/整数指标、4/6历史、不同 query
共用相同 patch、缓存上限、非有限输入、权重变动保护、六帧输出和一次完整 suite。
CUDA graph/pinned-copy tests 在 CPU 环境 skip；真实 CUDA 的速度和字节门槛由上述服务器命令验证。
没有服务器真实数据时不得写入虚构 FPS/提升比例。
