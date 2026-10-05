# Local column probability 推理提速

## 已有服务器证据

epoch19 / 同一 scene-balanced dev64 的32窗口 / batch256 / 正反序各一次：

- `serial_raw`：1.762262 秒/窗口。
- `parallel_raw`：0.857911 秒/窗口，其中 `column_probability_seconds` 0.586060 秒/窗口。
- `parallel_buffered`：0.944768 秒/窗口，比 `parallel_raw` 慢。不能把 batch 上传/读回 bundle 宣称为有效提速。

因此保留四窗口有界 CPU raw 预取，独立 evaluator 和 wrapper 默认恢复逐字段上传、逐 chunk 读回。显式 `--buffered-chunk-io` / `FULL_JOINT_EVAL_IO_OPTIMIZED=1` 仍可诊断旧 bundle，但不默认启用。

## 本次新增路径

仅 inference opt-in，不改训练 forward、checkpoint、阈值、精度、候选 population、batch256 或4→6设定。

1. 编译 CPU `patch_rows`：从已有 F×X×Y×Z byte inverse map 直接写入原 N×F×P×P×Z batch；合并静态 class membership bit。只做整数索引/拷贝，不动原 NumPy float64 逆变换、floor、矩阵乘法顺序。Native ABI4，按源码/编译器指纹生成新的 library，不覆盖旧 library。
2. 每 horizon 上传一次 base/fallback/context/kind/classes/legal，按原 chunk 切片。source query 也只按原 actor 索引取一次；source projection/网络仍逐原 batch 前向。临时 GPU 输入预算64MiB，超预算或自定义网络自动保持原逐 chunk 路径。
3. 连续 source finite 检查独立于已经回退的上传 bundle；仍检查所有使用到的 source、两类 logits 和校准权重，任何 nonfinite 都不得返回有效概率。训练/direct forward 的检查完全不变。
4. 可选一个 NEXT horizon 的 CPU inverse-map 预取，在当前 horizon CUDA 推理时准备下一张图。至多 CURRENT+NEXT 两份，每份 map 上限64MiB；frame mapping 限制为两线程，避免 raw4 × map6 的嵌套争抢。未来准备只读取当前模型预测下的因果 geometry，不读取 future GT。窗口结束/异常关闭线程，绝不跨 learned update 缓存 map、source query 或特征。

## 一趟验证

服务器 `OccFM` 激活并更新当前分支后：

```bash
JOINT_EVAL_SPEED_COLUMN_SUITE=1 \
  bash tools/real_motion/run_p0_f9_joint_eval_speed.sh
```

脚本使用已确认 epoch19 snapshot 和上次相同32个 key，不跑 full，不改变 optimizer/RNG，不自动部署。三种模式正反序各测一次：

- `parallel_raw`：已验证更快的旧参考路径。
- `parallel_columns`：编译 patch + horizon inputs/source hoist，map 仍串行。
- `parallel_columns_prefetch`：再开启 NEXT map 重叠。

first-use 三个 horizon 逐字节概率检查、Strong exactness 在每 trial 的两窗口 warmup 中完成，不算吞吐。全部 timed windows 每个 horizon 的概率数组 SHA256，以及 frozen integer counts、confidence 与 quality fingerprint 必须完全一致，才能产出 complete speed.json。任一 exactness 失败直接报错，不产生被接受的速度结论。timed records/order/network batch/threshold 相同；OS cache 无法真的 flush，不能承诺 full 绝对耗时。

`summary.txt` 同时输出 end-to-end 与 column probability 的 speedup、实际最快模式，以及对应可显式设置的 evaluator 环境变量。`progress.jsonl` / `speed.json` 已拆开 map wait、map worker、patch worker、采样等待、固定输入准备、输入上传、source 准备、network forward host dispatch、calibration host、probability readback、finite check wait。CPU worker 耗时重叠，不得与串行 wall time相加；CUDA host dispatch 不是 GPU kernel 活跃时间，读回可包含等待之前 GPU 工作。

新优化默认为关闭，直到真实 L40S 对照通过。若最快模式不是新路径，保留 `parallel_raw`；不以显存/利用率上涨作为提速证据，也不根据速度改科学配置。

## 本地测量边界

Windows CPU synthetic：4 history，20,100 queries，原 batch256，map已建立，仅 patch-copy/sampling 循环，四次正反序中位数。NumPy 路径0.125485秒，编译直接写入0.082121秒，约1.528倍，字节完全一致。这不是 nuScenes 数据，也不包含 CUDA 网络、raw I/O 或完整 eval，不能作为服务器端到端提速结论。
