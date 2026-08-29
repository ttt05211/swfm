# P0-F：轻量 Motion Support Proposal 可行性验证

P0-F **不是正式 World Model 训练**。目的只有一个：

> 在固定约 10–15% latent 计算预算下，一个极轻量、因果的 MSP 是否能明显提高 future-motion support oracle？

如果不能，就停止这条支线；不要通过继续加模块或扩大规则 support 来“救”结果。

## 1. 三层职责

```text
Real-Motion Decomposition
  → 只回答：当前谁已经真实运动 / 谁只是 dormant motion-capable？

MSP Probe
  → 只回答：哪些候选未来需要激活？未来大概去哪里？

Sparse World Model
  → 以后才回答：这些 sparse regions 里未来到底是什么 occupancy？
```

本 probe 中 **不训练 World Model**。

## 2. MSP 输入严格因果

第一版只使用 occupancy component 特征：

- t0 component center `(x,y)`；
- 历史 occupancy 得到的 KTA last-step `(vx,vy)`；
- component BEV extent / voxel count；
- KTA 是否成功匹配历史 component；
- `OBSERVED_MOVING` / `DORMANT` state；
- motion-capable semantic class one-hot。

nuScenes GT `instance_token`、future box、future center **只用于训练标签和诊断，不进入 feature tensor**。

第一版故意不加 image / HD map / OccFM latent，先判断“learned where-to-compute”本身有没有信号。

## 3. P0-F0：A/B/C miss attribution

定义：

- **A / OBSERVED_MOVING**：future-moving GT instance 在 t0 能匹配到真实历史运动 component；
- **B / DORMANT**：future-moving GT instance 在 t0 能匹配到 motion-capable 但历史未动 component；
- **C / NO_CAUSAL_SOURCE**：没有可匹配的 t0 occupancy component。

C 不是简单的 nuScenes birth count；它表示 object-centric MSP 在当前 occupancy decomposition 下没有 causal source。

```bash
python tools/real_motion/p0_msp_attribution.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --max-windows 128 \
  --output outputs/p0_f0_msp_attribution_128.json
```

重点看：

```text
aggregate.A_observed_moving.missed_voxels
aggregate.B_dormant.missed_voxels
aggregate.C_no_causal_source.missed_voxels
```

如果 A+B 占绝大多数 miss，object-centric MSP 值得继续；如果 C 很高，应该优先考虑 global sparse queries，而不是把 object head 做得更重。

## 4. 构建小样本 probe cache

推荐第一轮：

```text
train: 1024 windows，scene-balanced round-robin
val:   128 windows，最多一个 midpoint window / scene
seed:  20260829
```

Train：

```bash
python tools/real_motion/p0_msp_build_dataset.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_train_temporal_v3_scene.pkl \
  --mode train \
  --max-windows 1024 \
  --output data/msp_probe_train_1024.pt
```

Validation：

```bash
python tools/real_motion/p0_msp_build_dataset.py \
  --dataroot /path/to/nuscenes \
  --info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --mode val \
  --max-windows 128 \
  --output data/msp_probe_val_128.pt
```

Builder 会报告：

- candidate 数量；
- OBSERVED_MOVING / DORMANT 比例；
- GT instance matching ratio；
- future activation positive ratio。

## 5. Tiny MSP

默认：

```text
feature dim = 19
hidden = 96
object attention = 1 layer / 4 heads
future modes K = 4
future frames = 6
loss = activation BCE + multimodal Gaussian NLL
```

训练：

```bash
python tools/real_motion/p0_msp_train_probe.py \
  --train-cache data/msp_probe_train_1024.pt \
  --val-cache data/msp_probe_val_128.pt \
  --dataroot /path/to/nuscenes \
  --val-info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --steps 2000 \
  --batch-size 8 \
  --budgets 0.10,0.125,0.15 \
  --output-dir outputs/p0_f1_msp_probe
```

Trainer 会 fail-closed 检查 train/val scene overlap；validation cache 必须是 scene-disjoint midpoint 协议。

## 6. 真正的 gate：不要只看 NLL

输出：

```text
outputs/p0_f1_msp_probe/msp_probe_report.json
```

重点比较：

```text
oracle_curve.decomposition
oracle_curve.frozen_hybrid_v6
oracle_curve.learned_msp_by_budget.0.1
oracle_curve.learned_msp_by_budget.0.125
oracle_curve.learned_msp_by_budget.0.15
```

每个 learned budget 都报告：

- GT-filled overall oracle；
- GT-filled Moving-mIoU v2 oracle；
- `delta_Moving_vs_rule`；
- future moving arrival recall；
- active latent ratio；
- window count / slot compute ratio / coverage。

建议第一轮成功标准：

```text
15% budget 下 Moving oracle 至少比 frozen Hybrid-v6 高 +5 pp；
2s / 3s 均不退化；
window compute 仍明显低于 dense；
A/B/C 诊断说明主要 miss 确实有 causal source。
```

这是 feasibility gate，不是论文最终阈值。

## 7. 结果解释

### 明显成功

例如：

```text
Hybrid-v6 Moving oracle ≈ 52
MSP @ 12.5%            ≈ 65+
MSP @ 15%              ≈ 70+
```

再考虑把 MSP 接入正式 sparse-WM pipeline。

### 只有很小提升

例如 `52 → 54`：停止 MSP，不训练正式 WM，不继续堆 attention / loss。

### C-source 很高

说明 object-centric proposal 有结构性上限，下一步应研究 global sparse support query，而不是把 KTA radius 继续放大。

## 8. 当前实现边界

- MSP 是 probe-only，不修改 formal Hybrid-v6 cache / trainer；
- learned latent support 按 50×50 cell budget top-k，然后以对应 4×4 BEV block 做 GT-filled oracle；
- future ego transform 只用于把 t0-frame proposal 映射到 future ego grid，和现有 OccFM-Fut 信息协议一致，不作为网络 feature；
- 第一版不加 frozen OccFM latent context，避免在 feasibility 阶段混入额外变量。
