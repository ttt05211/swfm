# V20 V3：推理对齐的 factorized completion gate

## V2 screen1024 结论

V2 `a7c5f49` 解决了 completion 全部预测 free 的塌缩，但没有通过效果门槛：

- positive semantic CE 从 `2.397` 降到 `1.132`；
- 真阳性 addition 上的 semantic accuracy 达到 `93%+`；
- update1024 写入 `429902` 个体素，occupancy precision 只有 `20.044%`；
- completion delta mIoU 为 `-1.094067`。

因此 V2 学会了条件语义，却把 occupied/free 门控训练成了过度补全。不得用 V2
checkpoint 进入正式训练或 full4369 报告。

## 根因

V2 presence loss 使用：

```text
logsumexp(non-free logits) - free logit
```

正式推理使用：

```text
max(non-free logits) > free logit
```

两者门槛不一致；同时 positive-only semantic CE 和 presence 共用 18-way head，语义梯度
可以改变正式 occupied/free 门槛。训练后期 semantic/free 梯度比持续上升，与 dev false
positive 同步增长。

## V3 模型与损失

completion trunk 后使用两个输出：

- 一个 scalar presence logit `g`；
- 17 类 conditional semantic logits `s`。

对外仍合成为 18 类 logits：

```text
z_nonfree = g + s - max(s)
z_free = 0
```

因此有严格恒等式：

```text
max(z_nonfree) - z_free = g
argmax(z) is non-free  <=>  g >= 0       # 精确相等事件除外
```

在输出投影层，presence BCE 不再更新 semantic head，positive semantic CE 也不再更新
presence head；两者仍共同训练 shared completion trunk，而 presence loss 会持续直接约束正式
门槛。presence bias 以冻结先验 `p=0.02` 初始化，即 `logit(p)≈-3.892`。presence 默认使用
普通 BCE（`gamma=0`），保留真实稀有事件先验，不再用 gamma=2 抑制大量 free 监督。

协议更新为：

```text
model: p0_f9_v20_unified_transport_completion_v2
train: p0_f9_v20_unified_transport_completion_train_v3
objective: argmax_aligned_bce_presence_positive_semantic_v2
head: factorized_presence_semantic_v1
```

V1/V2 checkpoint 仍可加载做 evaluator/诊断，但不能续训成 V3。

## 快速止损

`--stop-after-updates N` 只改变本次进程的停止点，不改变 `max_updates=1024` 的学习率
调度、训练 population、随机顺序或 resume contract。第一次 V3 screen 使用 `N=256`；如果
update128/256 没有形成非零、高精度 addition 或 completion delta 没有改善，则停止，不再
浪费完整 screen 时间。通过后可从 update256 原样 resume 到1024。

## signed margin 诊断

evaluator 的 `--diagnostic-free-logit-offsets` 现在接受 signed nonzero 值：

- 正数：从 free logit 减去 offset，门槛更松；
- 负数：等价于提高 free logit，门槛更严。

所有 offset 共用同一次 completion forward。它只用于判断已有 checkpoint 的排序/校准，
不是正式 `offset=0` 指标，也不能替代 V3 验收。
