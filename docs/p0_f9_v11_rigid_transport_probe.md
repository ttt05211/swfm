# P0-F9 v11：No-WM Rigid Transport Probe

## 为什么先做这个 probe

v10 oracle-gap 已经把当前 Full-M E10 的主要剩余误差定位到：

- `oracle_write`: Moving `+28.59`；
- `oracle_clear_keep`: Moving `+10.54`；
- `oracle_event_presence`: Moving `+41.57`；
- `oracle_semantic`: Moving 只有 `+2.66`；
- 当前 MSP 对 GT-required WRITE 的覆盖约 `48.42%`。

因此下一步不应先给 OccFM 继续叠 context/semantic head，而应先验证一个更基础的问题：**如果我们已经知道物体未来去哪，直接把 t0 的原始 3D occupancy shape 刚体搬过去，本身能做到多好？**

本 probe 完全不加载 World Model / VAE，也不训练任何参数。

## 三个 transport 版本

所有版本都从 Strong-W2Det 的 occupancy-only t0 动态 connected components 出发。每个 component 保存的是当前真实观测到的 source voxel cloud，而不是 box 模板。

1. `kta_rigid_replay`
   - 只用历史 occupancy；
   - 使用 Strong/KTA 的 backward-difference velocity；
   - `yaw_delta=0`；
   - 目的主要是 sanity check：用同一 source shape 的 rigid transport 应尽量重放 Strong anchor。

2. `gt_center_rigid`
   - source shape 仍来自 causal t0 occupancy；
   - 未来 center / existence 用 GT annotation，只作为 oracle；
   - 不旋转 source shape；
   - 回答“如果只把未来平移位置预测对，刚体搬运能到什么水平”。

3. `gt_pose_rigid`
   - source shape 仍来自 causal t0 occupancy；
   - 未来 center / existence / yaw 用 GT annotation，只作为 oracle；
   - source shape 做 object-centric planar SE(2)；
   - 回答“若未来 pose 完全正确，source shape rigid transport 的结构上限是多少”。

这里的 GT future annotation **绝不作为部署输入**；它只用于验证这种表示是否值得训练一个 motion predictor。

## Coherent CLEAR + WRITE

不能把 transport shape 简单加在 Strong anchor 上，否则会同时保留 KTA 旧预测和新 transport 位置，形成重复车辆。

probe 对每个被替换的 source object 做：

```text
Strong/KTA predicted footprint -> CLEAR
transported source footprint   -> WRITE
```

先统一清除 selected sources 的 Strong/KTA copy，再统一写入 transport targets。其他没有被匹配到 causal source 的动态物体、静态场景和背景都保持 Strong anchor 不变。

因此这是一次 object/event-level replacement，而不是 write-only 叠加。

## 为什么还要报告 MatchedM

全局 Moving-mIoU 同时受两件事限制：

1. 有没有 causal source component；
2. 已有 source 时 rigid transport 是否足够准确。

所以脚本除了正式 Full Moving-v2，还额外构建只属于“成功匹配到 causal t0 source”的 GT Moving-v2 support，并报告 `MatchedM`。

如果出现：

```text
gt_pose_rigid Full Moving 不高
但 MatchedM 很高
```

说明 rigid transport 本身可行，主要瓶颈是 source discovery / reachability。

如果连 `gt_pose_rigid MatchedM` 也很低，则说明 source shape 的刚体假设、可见性变化或 non-rigid deformation 本身就是主要问题，不能急着训练 motion head。

## 正式 128-val 运行

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

MSP="$ROOT/data/msp_probe_val_128.pt"
P0F9="$ROOT/data/p0_f9_v2_wm_val_top2_128"
DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"
OUT="$ROOT/outputs/p0_f9_v11_rigid_transport_probe_128.json"
LOG="$ROOT/outputs/p0_f9_v11_rigid_transport_probe_128.log"

cd "$ROOT"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_rigid_transport.py" \
  --msp-cache "$MSP" \
  --p0f9-cache "$P0F9" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --output "$OUT" \
  --match-max-distance-m 4.0 \
  2>&1 | tee "$LOG"
```

这是 CPU/raw-data diagnostic，不需要占 GPU。10 核机器上建议内部 BLAS 固定 1，脚本当前是串行 128-window，以保证 source matching / component replacement 的可审计性。

Smoke：

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

MSP="$ROOT/data/msp_probe_val_128.pt"
P0F9="$ROOT/data/p0_f9_v2_wm_val_top2_128"
DATAROOT="$OCCFM/data/nuscenes"
INFO="$OCCFM/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl"
OUT="$ROOT/outputs/p0_f9_v11_rigid_transport_probe_smoke.json"

cd "$ROOT"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$PY" "$ROOT/tools/real_motion/diagnose_p0_f9_rigid_transport.py" \
  --msp-cache "$MSP" \
  --p0f9-cache "$P0F9" \
  --dataroot "$DATAROOT" \
  --info-pkl "$INFO" \
  --output "$OUT" \
  --max-windows 2 \
  --verify-strong-anchor
```

`--verify-strong-anchor` 会从 raw history 重新生成 Strong-W2Det sequence 并要求与 cache bit-exact；正式 128-window 默认关闭，避免重复做一遍昂贵 anchor 构建。

## 输出重点

终端输出：

```text
=== P0-F9 NO-WM RIGID TRANSPORT PROBE ===
variant                  Overall    Moving   dOverall   dMoving   MatchedM
strong_anchor
kta_rigid_replay
gt_center_rigid
gt_pose_rigid

=== MOVING BY HORIZON (FULL / MATCHED-SOURCE SUPPORT) ===
=== CAUSAL SOURCE MATCH / MOVING-SUPPORT COVERAGE ===
=== KTA RIGID REPLAY SANITY ===
```

重点按以下顺序解释：

1. `kta_rigid_replay` 应接近 `strong_anchor`。若差很多，先修 transport/raster/fusion 实现，不做任何方法结论。
2. `gt_center_rigid - kta_rigid_replay`：未来中心/trajectory error 的潜在收益。
3. `gt_pose_rigid - gt_center_rigid`：yaw/rotation 的额外收益。
4. `gt_pose_rigid MatchedM`：在“已经有 causal source”条件下，rigid source-shape transport 本身的上限。
5. `matched_source_support_coverage`：全 Moving-v2 区域中有多少属于可追溯 causal source；用于区分 source discovery 问题和 transport representation 问题。

## 下一步决策

- 若 `gt_pose_rigid MatchedM` 很高，且 Full Moving 明显超过 Strong / Full-M：开始训练轻量 motion correction head，输出六个 horizon 的 `(dx,dy,dyaw,existence)`，再显式 transport source shape。
- 若 `gt_center_rigid` 已接近 `gt_pose_rigid`：第一版 motion head 不必优先预测 yaw，可先只做 center/existence。
- 若 `gt_pose_rigid MatchedM` 很高但 Full Moving 仍低：优先补 source discovery / trajectory-generated support，不加 WM。
- 若 `gt_pose_rigid MatchedM` 仍明显低：rigid transport 本身不够，应继续拆 source-shape incompleteness / visibility / deformation，再决定 WM residual 应承担什么。
