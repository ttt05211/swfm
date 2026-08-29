# P0-F0.6：最后一次 Historical Source Audit

这是 MSP feasibility training 前的**最后一个诊断**。本实验结束后不再继续增加 P0-F0.x。

## 唯一问题

P0-F0.5 已经确认 C 的主要部分是：future-moving instance 在 t0 的 GT box 位于当前 occupancy grid 外。

P0-F0.6 只回答：

> 这些 t0-outside-grid instance 在过去 5 个 pre-t0 history frames 中，是否曾经在 occupancy 输入里出现过同类 voxel？

- `history_seen`：至少一个过去帧中，同一 instance 的 GT box（允许 0.5m margin）内存在同类 occupancy voxel。
- `history_never_seen`：过去 5 帧都没有上述 causal occupancy source。

GT instance identity / box 只用于诊断，不是 MSP 输入。

## 坐标合同

每个历史帧都在**该帧自己的 ego frame**中检查：

1. 用该历史帧 `ego_to_world` 把 GT annotation world box 转回该帧 ego；
2. 与该帧原始 `history_occ[i]` 比较；
3. t0 明确排除，只检查 `t-2.5, -2.0, -1.5, -1.0, -0.5s`。

因此不存在把 t0-aligned occupancy 和 history-native GT box 混用的问题。

## 运行

```bash
python tools/real_motion/p0_msp_history_source_audit.py \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --info-pkl /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl \
  --max-windows 128 \
  --output outputs/p0_f06_history_source_audit_128.json
```

## 只看这些输出

- `summary.history_seen_share`
- `summary.history_never_seen_share`
- `unique_instances`
- `aggregate.last_seen_age_s`
- `per_horizon.*.source_category`

`summary.target_t0_outside_C_missed_voxels` 应与同协议 P0-F0.5 的 `t0_box_outside_grid` missed-voxel 口径一致；当前 128-scene reference run 的预期参考值是 41,798，但代码不硬编码这个数。

## 冻结后的决策

无论结果如何，P0-F0.6 后都直接进入 MSP feasibility training：

- `history_seen` 占比明显：MSP candidate 加入 causal last-seen temporal tracks；
- `history_never_seen` 占比较高：这部分视为 occupancy-only causal input ceiling，不增加 global query / birth head；
- 然后固定 1024 train windows + 128 scene-disjoint val windows，训练 tiny MSP，评估 `Moving Support Oracle @ 10/12.5/15% budget`。

除非发现明确代码 bug、GT leakage 或坐标错误，不再增加新的 P0 诊断。
