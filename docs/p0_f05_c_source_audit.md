# P0-F0.5：C-source 二次审计

这一步只分析 P0-F0 的 `C_no_causal_source`，不训练 MSP，也不修改 Hybrid-v6 / World Model 训练链路。

## 为什么要做

P0-F0 的 C 只表示：future-moving GT instance 没有被当前 GT→occupancy-component matcher 关联到 MSP candidate。

它 **不等于 future birth**。当前 frozen Moving-mIoU v2 的 `moving_records` 只包含 t0 与 future 都存在的 common instance；因此真正的 birth 不会进入这个 C 集合。

本审计把 C 分成两组互补诊断：

1. **matching mechanics**
   - `C1_no_same_class_candidate`
   - `C2_candidate_outside_match_gate`
   - `C3_candidate_within_gate_but_unmatched`
2. **t0 occupancy evidence**
   - exact GT box 内有同类 occupancy
   - 只在 +0.5m margin 后有同类 occupancy
   - +0.5m 后仍无同类 occupancy
   - t0 box 完全落在 occupancy grid 外

输出还包含 `mechanism_x_occupancy_evidence` 交叉表，以及 XY grid boundary 统计。

## 运行

```bash
python tools/real_motion/p0_msp_c_source_audit.py \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl \
  --max-windows 128 \
  --output outputs/p0_f05_c_source_audit_128.json
```

默认保持与 P0-F0 一致：

```text
scene seed = 20260828
GT↔component match gate = 4.0 m
one window per validation scene
```

## 重点看什么

优先看：

```text
summary
aggregate.mechanism
aggregate.occupancy_evidence
aggregate.mechanism_x_occupancy_evidence
aggregate.boundary
```

解释规则：

- `C3_candidate_within_gate_but_unmatched`：4m 内确实存在同类 candidate，但被另一个 GT instance 占用。按照 frozen greedy matcher，这是**确定的 supervision matching artifact**。
- `C2_candidate_outside_match_gate`：存在同类 candidate，但超过 4m。它可能是 component merge / centroid 偏移，也可能只是另一辆同类物体，不能自动视为 recoverable；必须结合 GT-box occupancy evidence。
- `C1_no_same_class_candidate + no_same_class_occ_in_0p5m_box`：更接近真正的“当前 occupancy 没有 object-centric source”。
- `C1_no_same_class_candidate + same_class_occ_in_exact_t0_box`：说明 candidate extraction 与当前 occupancy 表示不一致，应先查实现，而不是增加 global query。
- `touches_xy_boundary` 高：优先考虑 FOV/grid truncation，而不是把它叫作 future birth。

脚本会 fail-closed 检查两个不变量：

1. Moving-v2 C record 缺少 t0 annotation → 直接报错；
2. C instance 在 4m 内还有未被占用的 candidate → 直接报 matcher invariant broken。

只有这个审计确认 C 主要不是当前 occupancy source 缺失后，才进入 1024-window MSP 小训练。
