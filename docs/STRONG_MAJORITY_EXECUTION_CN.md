# Strong 多数投票无损执行候选

## 目标与冻结约束

2026-10-09 服务器同256窗口×3成对测速：buffered逆变换只使整网151.236→149.391ms（1.0124×）。
Strong的多数投票约66ms，逆变换约12ms；下一步只优化多数投票，不改模型、权重、阈值、几何精度或六帧输出。
正式Dense Forecast边界不变，仍实时重建Strong/KTA；不能缓存未来预测来提升FPS。

## 本次实现

- 独立`ParallelNativeMajority`，默认不启用，最多8个CPU线程；本轮固定4线程候选。
- X方向分块，每块携带2行邻域，内部5×5×1整数投票仍调用原native函数；全体体素都处理，不截断候选。
- 临界值/平票坐标按原全网格顺序合并；仍用原SciPy float32滤波、0.3门槛和升序类别决胜。
- 不再为少量临界点排序全网格类别。只取回退邻域里真正出现的类别；缺席类别得分恒为0，不可能胜出正的0.3门槛，因此不影响赢家。
- 紧凑提取临界坐标，去掉重复全网格扫描；池只保留线程，不存前次窗口的输入/预测。
- ContextVar作用域显式开关、异常恢复；旧训练/推理默认路径不变。
- 7个历史缓存namespace文件、10个训练指纹文件和native C++/ABI未改，原cache/checkpoint继续复用。

## 本地结果及限制

现有RTX3050笔记本环境，只读用户真实`replay.zip`，6个窗口、每个6张200×200×16网格，轮换次序重复7遍：

| 多数投票执行 | 六帧平均耗时 | 相对原版 |
| --- | ---: | ---: |
| 原native+原SciPy回退 | 59.873ms | 1.000× |
| 单线程紧凑回退 | 29.879ms | 2.004× |
| 2线程 | 23.049ms | 2.598× |
| 4线程 | 19.949ms | 3.001× |
| 6线程 | 18.997ms | 3.152× |

原native、完整SciPy参考、新路径逐字节一致；回放warp在内部投票计时之外。
93项本地CPU/CUDA相关回归通过，含分块边缘、平票/3/10临界值、非连续输入、异常恢复、Strong六帧anchor/components/CLEAR、原FPS与新增6秒递推；CLI help、Bash语法和diff检查通过。不称全仓CI通过。
这是本地CPU多数投票阶段，不是训练提速、整网FPS或L40S预测。6线程相对4线程收益较小，优先4线程。
本地结果：`outputs/local_strong_majority_20261009_v3/speed.json`（ignored，不提交大回放）。

## 服务器一次性成对验收

更新`feature/v22-surface-aware-ccr`后：

```bash
conda activate OccFM
cd /root/nas/occ/swfm
SURFACE_MEAN_FPS_COMPARE_MAJORITY=1 \
  bash tools/real_motion/run_p0_f9_joint_surface_mean_fps.sh
```

与前次同seed1729、256窗口/150场景×3，原`surface_fused_graph`和新`surface_fused_majority_graph`轮换。
两臂逆变换仍为reference，隔离多数投票收益；不要同时设`SURFACE_MEAN_FPS_COMPARE_STRONG=1`。
可选`SURFACE_MEAN_FPS_MAJORITY_WORKERS=2/4/6`，不自动扫描或按时延选模型。
预热/图捕获/一致性检查在计时外；每个窗口的概率和完整六帧逐字节一致才报告。
CLI额外确认候选在每次完整六帧forecast都实际调用投票，而不是空分支测速。
summary直接包含整网FPS、P90、Strong四段耗时和投票调用统计，不需再提取日志或重跑精度。
原目录/权重/缓存不改，不自动采用新后端，不启动训练、阈值搜索或full4369。

本地复现（使用健康的项目Python）入口：`benchmark_p0_f9_strong_majority_local.py --replay ... --out-dir NEW_DIRECTORY`。
