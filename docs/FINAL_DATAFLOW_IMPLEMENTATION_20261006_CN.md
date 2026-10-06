# 终版数据流 / Dense Forecast FPS 实现状态（2026-10-06）

开发分支：

`feature/v22-final-dataflow-fps-20261006`

Draft PR：

`#77 V22: unify final dataflow and Dense Forecast FPS`

设计合同：

`docs/FINAL_METHOD_ARCHITECTURE_AND_FPS_PROTOCOL_20261006_CN.md`

## 已实现

### 1. 单一历史表示入口

新增：

`tools/real_motion/final_dataflow.py`

核心接口：

```python
history_state, future_ego = prepare_history(...)
result = forecast_six(history_state, future_ego, ...)
```

`CausalHistoryState` 只允许包含历史观测决定的信息：

- 4 帧历史 semantic occupancy / visibility / poses；
- current / previous source extraction；
- source identity/order/class；
- causal association / registration；
- observed source world geometry；
- history-only canonical evidence；
- history-derived V18 inputs；
- history content fingerprint。

明确不保存：

- future GT；
- repair target；
- learned CCR activation；
- future motion prediction；
- future transport query；
- horizon ownership/fallback；
- Strong/KTA future prior。

Future ego poses 使用独立 `FutureEgoCondition` 传入，不混进 history representation。

### 2. 单一正式 forecast 路径

`forecast_six()` 的同步计时范围内依次执行：

1. fresh causal KTA displacement；
2. fresh all-six Strong/KTA prior；
3. KTA H2D；
4. frozen epoch19 V18 motion forward；
5. source-centred SE(2) transport + layered occupancy/owner/fallback；
6. canonical evidence -> six future projection / ownership legality；
7. Point CCR shared point encoding + six horizon-conditioned readouts；
8. constrained ADD/REMOVE composition；
9. 六帧 dense semantic occupancy 完成后 CUDA synchronize。

history-only canonical evidence在 `prepare_history()` 中只构建一次，六个 horizon 不重复读取/编码四帧历史。

### 3. 训练侧第一轮清理

`tools/real_motion/ccr_screen_common.py::train_step` 已去掉 batch 内逐 window 的 frozen V18 forward。

旧路径：

```text
window1 -> V18
window2 -> V18
window3 -> V18
window4 -> V18
```

新路径：

```text
pack sources of batch
      -> ONE frozen V18 forward
      -> split by source population
      -> per-window hard geometry / GT target / CCR loss
```

不改变 Point CCR support、GT target、loss、阈值或 optimizer 定义。

固定几何/descriptor cache 暂时保留；本轮不为了“代码统一”删除已经有效的训练缓存。后续只有在正式 parity/profile 证明存在重复构建时再继续收敛。

### 4. 唯一正式 FPS runner

新增：

- `tools/real_motion/benchmark_p0_f9_dense_forecast_fps.py`
- `tools/real_motion/run_p0_f9_dense_forecast_fps.sh`

正式结果只叫：

> **Dense Forecast FPS**

不再把 fresh/cached Strong 两套数字都当正式 FPS。

定义：

[
FPS = \frac{\text{total generated future frames}}{\text{total synchronized wall-clock seconds}}.
]

6 future frames/window；batch=1；固定 20 窗口（18 scene-balanced + 2 high-source stress），3 次重复。

同时报告：

- Dense Forecast FPS；
- mean six-frame latency；
- P50；
- P90；
- host-side stage profile（仅诊断）。

### 5. 正式 FPS 边界

计入：

- fresh KTA；
- fresh Strong；
- V18 motion；
- SE(2) transport；
- future projection；
- ownership / fallback / legality；
- CCR；
- dense composition。

不计：

- disk / dataset I/O；
- history-only source extraction；
- history association / registration；
- history-only canonical evidence；
- checkpoint load；
- native/CUDA compile；
- warmup；
- GT / metrics；
- correctness hash；
- visualization / saving。

`cached Strong` 只允许作为 profiler，不允许进入论文主 FPS。

## 正确性 Gate

正式计时前，每个固定窗口先跑一次旧路径和新路径。

要求 exact signature 一致：

- motion outputs / latent context；
- CCR probabilities；
- 六帧 dense occupancy。

任一窗口失败立即停止，不输出“正式 FPS”，不 silent fallback，也不通过重训来掩盖数据流错误。

## 是否需要重训

本轮属于 execution/dataflow refactor。

只要 old/new parity 和 dev 指标一致：

> **不需要重新训练。**

若 parity 不通过，第一动作是定位重构错误，而不是重训。

只有后续改变 support / feature / learned module / loss / target / sampling / composition semantics，才构成新科学模型并需要重新训练。

## 服务器正式运行

```bash
conda activate OccFM
cd /root/nas/occ/swfm

git fetch origin feature/v22-final-dataflow-fps-20261006
git switch feature/v22-final-dataflow-fps-20261006
git pull --ff-only

bash tools/real_motion/run_p0_f9_dense_forecast_fps.sh
```

默认读取已经确认的：

- epoch19 Local transport；
- Clean-E14；
- Point CCR 20% TRAIN x 3 pass checkpoint；
- dev4369 cache；
- frozen dev64/dev512 manifest；
- nuScenes temporal-v3 validation metadata。

输出目录：

`outputs/p0_f9_point_ccr/dense_forecast_fps_YYYYMMDD_HHMMSS_PID/`

关键文件：

- `summary.txt`
- `dense_forecast_fps.json`
- `fps_manifest.json`
- `progress.jsonl`

## 当前验收状态

GitHub dependency-light CI 用于：

- 全 Python `py_compile`；
- `pytest -m 'not integration'`。

真实 CUDA / L40S 验收仍必须在服务器完成，尤其包括：

1. 20/20 window old/new exact parity；
2. 正式 Dense Forecast FPS；
3. mean/P50/P90；
4. stage profile；
5. checkpoint SHA 在只读测速前后不变。

在上述真实 CUDA Gate 通过前，PR 保持 Draft，不合并。
