# P0-F2：MSP 与真实 World Model window 对齐

P0-F1 已证明 tiny MSP 在 15% latent-cell budget 下能把 Moving support oracle 从约 52.7 提升到约 63.2，但 cell support 被 20×20 `WindowPlanner` 覆盖后需要约 90% slot compute。因此 P0-F2 不再训练 MSP，而是把 selection unit 直接改成当前 sparse World Model 的真实执行单位：20×20 latent window。

## 冻结协议

- 复用 P0-F1 `msp_probe_best.pt`，不重新训练；
- MSP 仍输出 6 个 future horizon 的 50×50 score map；
- 6 个 horizon score 相加，得到一个 spatial score map；
- 6 个 future steps 共用同一组 spatial windows；
- 每次选一个 20×20 window，随后把该 window 中 score 置零，再选下一窗，形成 marginal-score Top-K；
- 评估 `K=1,2,3`；
- 一个 20×20 window / 50×50 latent map 对应 16% slot area，因此 K=1/2/3 的上限分别约为 16%/32%/48% slot compute；
- 不增加 image/map/latent feature，不改 MSP loss，不加新诊断。

## 两种 oracle

### 1. Support-only oracle

与 P0-F1 的 learned support oracle 和 frozen Hybrid-v6 support oracle保持同一含义：仅在选中 window 内写入 GT dynamic occupancy，用来回答“window support 位置是否选得对”。

### 2. Anchor-preserving repair oracle

更贴近最终方法：

```text
selected window     -> GT dynamic repair（oracle only）
unselected region   -> 保留 causal KTA / dormant zero-motion anchor
static background   -> 原 deterministic transport
```

它只作为“如果 sparse WM 在选中 window 内能完美修复，最终 anchor-preserving 结构有多少 headroom”的上限，不是实际模型结果。

## 运行

```bash
python tools/real_motion/p0_msp_eval_window_budget.py \
  --checkpoint outputs/p0_f1_msp_probe/msp_probe_best.pt \
  --val-cache data/msp_probe_val_128.pt \
  --dataroot /path/to/nuscenes \
  --val-info-pkl /path/to/nuscenes_infos_val_temporal_v3_scene.pkl \
  --topk 1,2,3 \
  --probe-report outputs/p0_f1_msp_probe/msp_probe_report.json \
  --output outputs/p0_f2_window_aligned_msp/report.json
```

重点看：

- `window_support_oracle_by_topk.K.oracle_Moving-mIoU_v2`
- `delta_Moving_vs_hybrid_support`
- `future_arrival_recall_per_horizon`
- `window_backend.mean_slot_compute_ratio`
- `window_backend.mean_unique_latent_ratio`
- `window_backend.mean_score_capture_ratio`
- `anchor_preserving_repair_oracle_by_topk.K`

P0-F2 是 deployment alignment，不是新的诊断树。结果用于直接选择后续正式 sparse-WM 的 window budget。
