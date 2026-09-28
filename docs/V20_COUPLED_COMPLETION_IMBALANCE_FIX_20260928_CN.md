# V20 耦合 Completion 类别不均衡修复

## 诊断

dev512 的 completion support 中约 97.93% 是 free。旧训练目标对所有采样体素执行无权重
18 类交叉熵，并按体素总数归一化。positive tile 只保证 tile 内至少出现一个非 free
体素，因此 tile enrichment 并没有消除体素级 free 主导。

margin 诊断同时表明：提高 free-logit offset 后，新增体素的 occupancy precision 可达到
48.281%，但 semantic accuracy 只有 1.536%。这说明模型已包含占用位置排序信号，主要缺口是
free/non-free 优化不平衡以及正体素语义梯度不足，不能通过正式推理阈值修复。

## V2 训练目标

模型、18 类 logits、add-only 合成和正式 `offset=0` 推理协议均保持不变。对同一组 logits
定义：

```text
g = logsumexp(z[0:17]) - z[17]
L_presence = binary_focal_with_logits(g, target != 17; gamma=2)
L_semantic = CE(z[0:17], target), only target != 17
L_completion = L_presence + lambda_semantic * L_semantic
```

默认 `gamma=2`、`lambda_semantic=1`。presence 对所有有效采样体素归一化；semantic 只对真实
非 free 体素归一化。如果一个窗口没有正体素，其 semantic 项是与图连接的零，不会产生 NaN。
重复采样 tile 仍按原 draw multiplicity 精确计权。

这不是独立 binary presence head。presence energy 始终由17个语义类别 logits 与 free logit
共同派生，因而保留类别条件证据以及原始推理接口。

## 协议和续训

- 新 checkpoint：`p0_f9_v20_unified_transport_completion_train_v2`。
- V1 checkpoint 仍可被 evaluator 和诊断工具读取。
- V1 checkpoint 不允许作为 V2 训练断点继续；resume contract 包含 objective protocol、
  focal gamma 和 semantic weight，任何不一致都会直接报错。

## 诊断输出

训练控制台每次成功 update 只打印关键项：presence loss、positive semantic CE、采样正例率，
以及 completion head 的 non-free/free 梯度范数。第一次成功 update 额外打印自然 support 正例率。

完整诊断写入 `<out-dir>/train_diagnostics.jsonl`，包括：

- 自然 support 与采样后的18类计数；
- 自然/采样正例率、无正例窗口数；
- presence、semantic 分项 loss；
- completion head 的逐类梯度范数和 non-free/free 梯度比。

这些诊断不改变采样顺序、模型输入、推理结果或正式指标口径。

## 验证门槛

1. smoke 必须完成 warmup 和 joint update，loss/gradient 全部有限。
2. sampled positive rate 必须显著高于 natural support positive rate。
3. screen1024 必须在正式 `offset=0` 下产生非零 additions。
4. completion semantic accuracy 和 completion delta mIoU 必须在连续 checkpoint 上改善，
   不能只依赖 occupancy IoU 上升。
5. screen 通过后才运行 dev512；不得用 margin sweep 结果替代正式结果。
