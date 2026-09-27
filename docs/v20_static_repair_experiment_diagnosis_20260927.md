# V20 Static Repair 实验演化与失败原因诊断

> 日期：2026-09-27  
> 分支：`feature/v20-3d-history-world-model`  
> 目标：记录 V20 Static Repair 从 coupled softmax 到 independent factorized presence 的实验过程，避免后续混淆 overfit / formal / threshold / loss-weight 结果，并明确当前已经被实验排除的假设。

---

## 1. 任务与固定协议

Static Repair 的目标不是重新预测完整 occupancy，而是在 **冻结 V18 当前预测为 free 的位置** 上，利用六帧历史证据补回未来静态 occupancy。

正式 Static Repair v2 的固定监督协议：

- support：full future grid ∩ frozen V18 predicts free；
- dynamic GT：映射为 free / no-add，不允许 Static 分支新增动态类；
- static positive：9 个静态语义类；
- query mask：严格使用 runtime 的未来 query union；
- horizon：6 个未来时距均保留，重复映射不去重；
- composition：add-only，Static 只允许覆盖 V18-free support；
- 正式模型选择指标：scene-disjoint dev 上的 composed semantic mIoU，而不是 binary IoU、训练 loss 或 overfit32。

Static support 极度稀疏。在 formal dev512 上：

- support voxels：1,874,038,008
- static-positive voxels：36,420,607
- positive fraction：约 **1.94%**
- no-add / free：约 **98.06%**

这一本身决定了 add/no-add 学习是主要难点。

---

## 2. 实验口径说明

### 2.1 overfit32

固定 32 个训练窗口，同时用同一 32 个窗口做 validation。

用途仅为：

- capacity check；
- 判断模型是否能拟合 repair 信号；
- 比较不同 head / loss 的优化行为。

**不能作为泛化结果，也不能与 scene-disjoint formal dev 指标直接等价比较。**

### 2.2 formal e1

- train：20,430 windows
- dev：512 scene-disjoint windows
- 每个训练窗口在 epoch 1 基本只见一次。

这是当前 Static Repair 的正式泛化证据。

### 2.3 fixed train512 diagnostic

使用 formal epoch1 checkpoint，在固定 train512 上只做 inference，用于区分：

- train / dev generalization gap；
- 还是 objective 本身导致 train 和 dev 都低 recall。

该集合不用于超参数选择。

---

# 3. 实验 A：原 coupled 10-class softmax overfit32

结构：

```
shared 3D encoder
      ↓
shared tile feature
      ↓
9 static logits + 1 free logit
      ↓
10-class softmax
```

即每一个 static class 都直接和 free 竞争。

历史 overfit32 结果显示：

- 前几轮几乎也是 no-add；
- 训练很久后可以达到明显正收益；
- epoch 93 峰值约 **+2.34 pp mIoU**；
- epoch 100 约 **+2.02 pp mIoU**；
- 后期 recall 可到约 30% 左右。

### 结论

原 coupled softmax **不是没有容量**。

它至少证明：

> 在固定 32-window population 上，当前 historical 3D encoder + tile feature 中存在足以学习 Static Repair 的信息。

但该结果是同集 overfit，不能证明 formal 泛化，只能证明 capacity / learnability。

---

# 4. 实验 B：原 coupled softmax formal epoch 1

路径：

```
outputs/p0_f9_v20_static_repair_formal_e1/history.json
```

结果：

| metric | value |
|---|---:|
| train loss | 0.06878 |
| val loss | 0.07723 |
| addition precision | **49.44%** |
| static-positive recall | **0.204%** |
| semantic accuracy on positive | **97.41%** |
| predicted additions | 150,539 |
| target static-positive voxels | 36,420,607 |
| V18 baseline mIoU | 42.0539 |
| V18 + Static mIoU | 42.0610 |
| ΔmIoU | **+0.00715 pp** |
| 1/2/3s ΔmIoU | **+0.00755 pp** |

六个 horizon 的 mIoU delta 均为很小的正值。

### 关键现象

模型不是“乱加”：

- add precision 约 49%；
- 一旦位置真的是 static positive，semantic accuracy 高达 97.4%。

真正的问题是：

> **几乎不加。**

predicted additions 只有 150k，而 positive target 有 36.4M；recall 仅 0.2%。

### 初步原因

普通 full-support CE 面对约：

```
1.94% positive
98.06% free
```

很容易获得一个低 loss 的保守解：

```
大多数 support → free
仅极少数非常确定的位置 → static
```

因此 formal epoch1 暴露的是 **free/no-add bias**，而不是 semantic classification failure。

---

# 5. 实验 C：formal epoch1 softmax logit-margin threshold sweep

定义：

```
score = max_static_logit - free_logit
```

默认 argmax 等价于：

```
tau = 0
```

dev512 主要结果：

| tau | addP | addR | ΔmIoU | 1/2/3s ΔmIoU |
|---:|---:|---:|---:|---:|
| 0.00 | 49.44% | 0.204% | +0.0071 | +0.0075 |
| -0.25 | 45.02% | 0.65% | +0.0169 | +0.0180 |
| **-0.50** | **40.78%** | **1.50%** | **+0.0241** | **+0.0272** |
| -0.75 | 37.53% | 2.70% | +0.0185 | +0.0237 |
| -1.00 | 33.96% | 4.64% | -0.0274 | -0.0146 |
| -1.50 | 25.20% | 14.06% | -0.6877 | -0.6014 |

最佳 composed semantic mIoU：

```
tau = -0.5
ΔmIoU = +0.0241 pp
```

fixed train512：

- tau=0：addP 58.44%，addR 0.267%，ΔmIoU +0.0123
- tau=-0.5：addP 44.81%，addR 1.31%，ΔmIoU +0.0305

### 结论 1：不是主要的 generalization gap

train512 与 dev512 的 recall 同样极低：

```
train tau=0 recall = 0.267%
dev   tau=0 recall = 0.204%
```

因此不能解释成：

> train 学得很好，只是 dev 泛化失败。

相反，更符合：

> **同一个 loss / free prior 在 train 和 dev 上都把模型压成低-recall 解。**

### 结论 2：threshold calibration 只能小幅救回

tau 从 0 降到 -0.5 可以把 recall 提高到 1.5%，说明 softmax logits 中存在一定 ranking 信息。

但收益仅：

```
+0.007 → +0.024 mIoU
```

仍远小于 overfit32 的 +2.x。

继续降低 threshold 后 precision 很快跌破可盈利区域，mIoU 转负。

因此：

> **softmax epoch1 不是“模型已经学好了，只是 argmax threshold 太保守”。**

threshold 只能恢复很小一部分潜在收益。

---

# 6. 实验 D：independent factorized presence + semantic

为避免 98% free 在同一 10-class softmax 中淹没 positive，尝试结构：

```
shared 3D encoder
      ↓
shared tile feature
      ├─ presence head: add / no-add
      └─ semantic head: 9 static classes
```

初始 presence objective：

```
L_presence
= 0.5 * mean_positive_BCE
+ 0.5 * mean_negative_BCE

L_semantic
= CE only on GT static-positive

L
= L_presence + L_semantic
```

presence bias 初始化为 -4.0，以保持 fresh checkpoint 默认 no-add。

## 6.1 overfit32，alpha=0.5

主要曲线：

| epoch | addP | addR | semAcc | ΔmIoU |
|---:|---:|---:|---:|---:|
| 1 | 3.33% | 0.00% | 100% | ~0 |
| 2 | 6.47% | 0.43% | 60.35% | -0.206 |
| 3 | 8.36% | 2.43% | 62.48% | -0.713 |
| 4 | 8.97% | 7.79% | 73.80% | -1.630 |
| 5 | 10.65% | 16.19% | 88.26% | -2.537 |
| 6 | 10.42% | 26.49% | 89.88% | -3.709 |
| 10 | 9.94% | 58.27% | 92.86% | -6.847 |
| 15 | 9.43% | 78.07% | 94.53% | -8.614 |
| 20 | 9.76% | 83.74% | 94.74% | -8.743 |

### 结论

Factorized head 成功解决了：

> recall 起不来的问题。

但同时制造了相反的问题：

> **presence precision collapse。**

后期：

- recall 可到 80%+
- semantic accuracy 94%+
- add precision 只有约 9%–11%
- composed mIoU 严重下降

这证明：

1. encoder / semantic feature 并非完全无效；
2. positive static 的语义类别很好学；
3. **真正失败的是 independent binary presence ranking。**

---

# 7. 实验 E：factorized presence loss alpha sweep

将：

```
L_presence =
alpha * mean_pos_BCE
+ (1-alpha) * mean_neg_BCE
```

从过于激进的 alpha=0.5 调回保守区域。

原因：

alpha=0.5 在 1.94% positive 数据下，相当于单 positive 对单 negative 的有效权重约 50×。

因此测试：

```
alpha = 0.02
alpha = 0.035
alpha = 0.05
```

使用 successive halving：

- 全部到 epoch3；
- 最佳 2 个续到 epoch6；
- 最佳 1 个续到 epoch10。

路径：

```
outputs/p0_f9_v20_static_factorized_alpha_sweep32/sweep_summary.json
```

## 7.1 epoch3

| alpha | addP | addR | semAcc | ΔmIoU |
|---:|---:|---:|---:|---:|
| **0.020** | **10.06%** | 0.071% | 99.82% | **-0.0203** |
| 0.035 | 6.91% | 0.417% | 99.77% | -0.1798 |
| 0.050 | 6.71% | 0.734% | 98.28% | -0.3200 |

保留 0.02 / 0.035。

## 7.2 epoch6

| alpha | addP | addR | semAcc | ΔmIoU |
|---:|---:|---:|---:|---:|
| **0.020** | **8.56%** | 0.489% | 98.81% | **-0.1744** |
| 0.035 | 6.32% | 1.046% | 98.08% | -0.5119 |

保留 0.02。

## 7.3 alpha=0.02, epoch10

| metric | value |
|---|---:|
| addP | **11.23%** |
| addR | **0.587%** |
| semAcc | **98.87%** |
| ΔmIoU | **-0.1453 pp** |
| 1/2/3s ΔmIoU | **-0.1369 pp** |

### 结论

alpha=0.02 已经非常接近原始 empirical class frequency。

按 train positive fraction ~1.94% 估算，其单样本正/负有效权重约接近 1:1，而不是 alpha=0.5 时的约 50:1。

但即使如此仍然：

- semantic 非常准确；
- addition precision 很低；
- composed mIoU 为负。

因此排除：

> “Factorized 失败只是因为 alpha=0.5 的 positive weighting 太大。”

继续微调 alpha=0.015 / 0.025 / 0.03 的信息价值很低。

---

# 8. 实验 F：alpha=0.02 epoch10 raw presence threshold sweep

这是对 factorized independent presence 的最终零训练检查。

checkpoint：

```
outputs/p0_f9_v20_static_factorized_alpha_sweep32/
alpha_0p02/epoch_0010.pt
```

直接使用 raw presence logit 扫 threshold。

结果：

| tau | additions | addP | addR | ΔmIoU |
|---:|---:|---:|---:|---:|
| 0.00 | 80,567 | 11.23% | 0.587% | -0.1453 |
| 0.50 | 40,613 | 12.07% | 0.32% | -0.0724 |
| 1.00 | 20,607 | 13.38% | 0.18% | -0.0351 |
| 1.50 | 9,307 | 14.82% | 0.09% | -0.0150 |
| 2.00 | 3,940 | 16.60% | 0.04% | -0.0058 |
| 2.50 | 1,581 | 20.18% | 0.02% | -0.0019 |
| 3.00 | 567 | 20.99% | 0.01% | -0.0007 |
| 4.00 | 81 | 22.22% | ~0 | -0.0001 |
| 5.00 | 10 | 30.00% | ~0 | ~0 negative |
| >=6.00 | 0 | 0 | 0 | **0** |

最终最佳：

```
tau >= 6
prediction = no-add everywhere
ΔmIoU = 0
```

tau=0 reproduction check：

```
checked = true
passed  = true
```

说明诊断与训练 validation 口径严格一致。

### 最终结论

**所有非零 addition threshold 都使 composed mIoU 下降。**

这意味着 factorized presence 并不是：

> ranking 已经正确，只是 default threshold 错。

而是：

> **presence score 本身没有形成一个有用的 TP-before-FP 排序。**

即使只保留最高置信度 tail：

- tau=5
- 仅 10 个 additions
- precision 仍只有 30%
- 仍没有正收益

因此 independent factorized presence 应正式停止。

---

# 9. 为什么 semantic 很准，但 presence 仍失败？

这是本轮实验最重要的结构性认识。

Factorized head 学的是两个不同问题：

```
presence:
“这里是不是任意一种 static？”

semantic:
“已知这里是 static，它属于哪一种？”
```

实验显示：

```
semantic accuracy ≈ 95%–99%
presence ranking 失败
```

这并不矛盾。

9 个 static class 的 appearance / history evidence 很可能是多模态的：

- barrier
- driveable surface
- sidewalk
- terrain
- manmade
- vegetation
- ...

一个独立 scalar presence head 被要求把所有这些类别压缩到同一个：

```
static vs free
```

方向上。

但不同 static class 可能需要不同的 class-specific evidence。

因此，一个统一 scalar presence direction 会丢掉原 coupled softmax 中很重要的结构：

> **每一种 static class 分别与 free 竞争。**

---

# 10. 为什么原 coupled softmax 反而有 overfit 正证据？

原 softmax：

```
z_static_1
z_static_2
...
z_static_9
z_free
```

每一种 static class 都有自己的 evidence direction。

是否 add 并不是由一个独立 scalar presence head 决定，而是：

```
max static evidence vs free evidence
```

因此 class identity 本身参与了 presence decision。

这与当前实验非常一致：

- independent semantic head 能准确识别类别；
- independent presence head 却不能区分 TP / FP；
- 原 coupled softmax 在 overfit32 上最终可以获得 +2.x mIoU。

### 注意

这并不证明 coupled softmax 已解决 formal 泛化。

它只证明：

> **保留 class-conditioned static-vs-free competition 比压成统一 binary presence 更有依据。**

---

# 11. 标准 coupled softmax 的更有用解释

令：

```
S = logsumexp(z_static_1 ... z_static_9)
g = S - z_free
```

则：

```
g
```

可以视为 class-conditioned presence energy。

对于 positive static class c：

```
10-class CE
=
presence loss(g, add)
+
semantic CE(z_static, c)
```

对于 free：

```
10-class CE
=
presence loss(g, no-add)
```

因此原 10-class softmax 本身已经隐式包含：

1. presence；
2. semantic；

区别是 presence 不是独立 scalar head，而是由 **9 个 static class evidence 聚合后与 free 比较**。

这可能正是当前 independent factorized head 丢掉的部分。

---

# 12. 当前被实验排除的解释

截至目前，可以明确排除：

### 排除 1：只是 semantic head 不会分类

否。

positive 上 semantic accuracy 多次达到 95%–99%。

---

### 排除 2：formal e1 主要是 train/dev 泛化问题

不支持。

softmax formal checkpoint：

```
train512 recall ≈ 0.267%
dev512   recall ≈ 0.204%
```

两边都低。

---

### 排除 3：softmax 只需要简单 threshold calibration

不支持。

dev512 最佳 tau=-0.5 仅：

```
ΔmIoU +0.024 pp
```

不能恢复主要 headroom。

---

### 排除 4：factorized 只是 alpha=0.5 太激进

否。

alpha 从：

```
0.50
↓
0.05
0.035
0.020
```

仍然没有得到正收益。

---

### 排除 5：factorized 只是 inference threshold 不对

否。

alpha=0.02 epoch10 raw-logit threshold sweep：

```
任何非零 addition → ΔmIoU < 0
最佳策略 → 0-add
```

---

# 13. 当前最可能的原因

现有证据最支持以下组合，而不是单一原因。

## 原因 A：任务本身极度稀疏

positive only ~1.94%。

普通 coupled CE 很容易首先学成：

```
free everywhere
```

因此 formal epoch1 recall 极低。

---

## 原因 B：Static positive 是多类、多模态集合

“static”并不是一种统一 appearance。

把 9 个不同 static class 强行压成一个 binary presence scalar，会损失 class-conditioned evidence。

这是 independent factorized presence 失败的最合理结构性解释。

---

## 原因 C：semantic supervision 与 presence ranking 的难度不同

实验反复出现：

```
semAcc very high
addP very low
```

说明：

> “positive 已知后是什么类”远比“当前位置到底值不值得 add”容易。

因此不能用 semantic accuracy 证明 repair localization 已经学好。

---

## 原因 D：threshold 不能修复缺乏 score separation 的模型

Threshold calibration 只有在：

```
TP score > FP score
```

具有足够排序能力时才有效。

Factorized 最终 sweep 表明高 score tail 仍然 FP 占多数。

因此 threshold 只能逐渐减少错误数量，不能制造正收益。

---

# 14. 当前决策

## 停止

以下方向不再继续：

- independent binary presence head；
- alpha sweep；
- factorized presence threshold sweep；
- 用 factorized checkpoint 跑 formal 20,430；
- 在 alpha=0.02/0.035/0.05 周围继续微调小数点。

这些已经有足够负证据。

## 保留

保留：

- HistoricalEvidence3DEncoder；
- shared tile feature；
- V18-free add-only composition；
- formal full-grid supervision；
- 9 static class semantic modeling；
- coupled class-vs-free competition。

---

# 15. 下一步建议

当前信息价值最高、成本最低的一步是回到原 coupled softmax formal checkpoint，做 **epoch 2**。

原因：

- coupled softmax 是目前唯一在 overfit32 有明显正收益证据的结构；
- formal epoch1 仍然是正 delta，只是 recall 极低；
- overfit32 早期同样经历长时间 no-add，最终才出现明显收益；
- independent factorized 已被系统排除，因此没有理由继续在它上面花 formal 训练预算。

建议 gate：

### 若 formal epoch2

出现：

```
addP 仍保持较高
addR 明显从 0.2% 上升到 1%+
ΔmIoU 同步明显扩大
```

则继续 coupled softmax。

### 若 formal epoch2

仍然：

```
addR ~0.2%–0.5%
ΔmIoU ~0
```

则停止“只是训练轮数不足”的解释。

下一版本应优先考虑：

> **保留 coupled static-class logits，并显式使用 class-conditioned presence energy**
>
> `g = logsumexp(z_static) - z_free`

而不是重新引入独立 scalar presence head。

---

# 16. 一句话总结

目前实验不是证明“3D history world model 没用”，而是把问题定位到了更具体的位置：

> **历史 3D 表征和 static semantic 本身具有学习能力；正式瓶颈是极稀疏 add-only 任务中的 repair localization。原 coupled softmax 太保守，但独立 binary presence 又破坏了 class-conditioned static-vs-free 排序。下一步应保留 coupled class competition，而不是继续调 independent presence。**
