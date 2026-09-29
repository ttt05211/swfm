# V21 Stage 0：Causal Source Induction and Transport

状态：**只实现 Stage-0 upper-bound audit**。没有 Transformer、loss、训练脚本，也不加载或调用任何 V20 completion checkpoint/logit。

## 冻结合同

- 科学基线：Clean-E14 / V18 source-centred SE(2)
- frozen commit：`ccf7d77e65e9773f441b35083d625b06791bfeaa`
- V21 工程分支基于：`feature/v20-unified-transport-completion@22b07d0`
- target：`DORMANT_ANCESTRAL + BIRTH`；保留六帧 annotation existence，但正式 source activation 使用第一次进入冻结 Ωmax 且存在无歧义 occupancy component 的 query-entry onset
- 正式 horizon：1 / 2 / 3 s；完整 state/onset：0.5 / 1 / 1.5 / 2 / 2.5 / 3 s
- Historical memory：完整过去 2.5 s，25 m/s causal association
- Frontier：从 Stage-1 index 继承冻结 Ωmax（0.4m canonical BEV boundary）→ 同 origin 的 1.6m anchor lattice；禁止使用 tracked placeholder extent
- compositor：V18 occupied 永远保护；Historical > Frontier；同类型 anchor ID 升序，first-writer wins
- frozen Moving-mIoU v2 不修改

## 代码入口

- `real_motion/v21_source_induction.py`：target、anchors、一对一 coverage、shape/prototype、add-only compositor、dev64 selector。
- `tools/real_motion/build_p0_f9_v21_dev_manifest.py`：冻结 dev64/dev512 identity/order。
- `tools/real_motion/build_p0_f9_v21_prototype_bank.py`：构建 K=1/4/8/16 train-only prototype。
- `tools/real_motion/eval_p0_f9_v21_stage0_upper_bounds.py`：运行 UB-0/UB-1/UB-2 和 Stage-0B gate。

## 1. 冻结 population

```bash
"$PY" -u tools/real_motion/build_p0_f9_v21_dev_manifest.py \
  --stage1-cache "$V20_STAGE1_DEV512" \
  --output data/p0_f9_v21_dev64_manifest.json --count 64

"$PY" -u tools/real_motion/build_p0_f9_v21_dev_manifest.py \
  --stage1-cache "$V20_STAGE1_DEV512" \
  --output data/p0_f9_v21_dev512_manifest.json --count 0
```

dev64 严格采用 parent dev512 原始 key/order 的 scene-balanced round-robin，不使用 GT positive 选样；manifest 保存 parent/selected fingerprint、Stage-1 index SHA256、冻结 highres/coarse lattice 和整份 manifest fingerprint。Evaluator 会重新验证所有字段。

## 2. 构建 train-only prototype

```bash
"$PY" -u tools/real_motion/build_p0_f9_v21_prototype_bank.py \
  --config "$RUNTIME_CONFIG" --train-cache "$TRAIN_V18" \
  --dataroot "$DATAROOT" --info-pkl "$TRAIN_INFO" \
  --output-dir outputs/p0_f9_v21_prototypes --k 1 4 8 16
```

唯一 observation key 为 `(sample_token, instance_token)`；对 train windows 的 history/future sample union 去重，不按 overlapping window 重复加权。同一 instance 在不同 sample 可各计一次。每个 sample 只运行一次 frozen Strong component extraction，归因和 shape 构建直接复用同一组 component/match，不再重复做 connected-component。shape 只做 GT center/yaw 的 offline canonicalization，不做 box-size scaling，保留真实尺度；距离为 `1-binary IoU`。某类样本少于 K 时不复制 medoid。

每类 observation 数 `<=512` 时运行 exact PAM；更大 population 使用冻结的 deterministic CLARA（sample size 256，5 trials），避免全量 `N×N` 距离矩阵。bank 保存 train-cache/info SHA256、完整 population 标志、shape fingerprint、算法参数和 medoid 内容 fingerprint。`--max-windows` 生成的只是 diagnostic bank，且窗口按训练缓存中的 scene first-appearance order 做 deterministic scene-balanced round-robin，不再取有顺序偏差的前 N 条；正式 evaluator 默认拒绝不完整 bank，只有显式 `--allow-incomplete-prototype-bank` 才能用于调试。evaluator 另报告 causal-covered BIRTH 的 prototype/exact canonical shape IoU 与体素量比，用于区分训练 population 不充分和 prototype 表示本身不足。

构建器默认使用至多 8 个线程并发完成独立 sample 的 Occ3D 读取、Strong component extraction 和 shape attribution（`--workers 0` 自动选择，服务器可显式使用 `--workers 8`）。unique sample 只访问一次，因此不再使用会保留数 GB 三维数组但没有复用收益的 occupancy LRU。结果仍由主线程按冻结 sample order 汇总，所以 worker 完成顺序不会改变 population、medoid 或 fingerprint。

## 3. dev64 Stage-0 smoke

Stage-0 evaluator 的 `--cpu-workers 0` 默认选择至多 8 个线程，并行加载同一窗口的帧、提取六帧历史/未来 component 和计算 Moving support。global/scene 指标共用一次 confusion-matrix 计数，替代对每个类别重复扫描全体素；输出 `performance.seconds_by_stage` 便于继续定位瓶颈。这里 GPU 只负责冻结 V18 forward/warp，Stage-0 的 GT attribution、component、renderer 与 metric 仍主要是 CPU 工作，因此低显存和间歇性 GPU utilization 本身不是停滞。

若普通 prototype 的 UB2/UB1 retention 不足，evaluator 同次额外报告 `UB2_FACTORIZED_ORACLE_EXTENT`：保持同一个 train-only prototype 形状码，只把 query-entry GT occupancy 的三轴 extent 作为 oracle 连续尺度施加到 prototype。它只回答“shape code × continuous extent”是否有表示上限，不复制目标 voxel pattern，不进入原 `stage0b_gate`，也不得作为 deployable 结果；其独立门槛位于 `factorized_oracle_extent_gate`。若该门槛仍失败，停止 prototype 路线；若通过，后续模型才增加一个 causal extent prediction head。

```bash
for K in 1 4 8 16; do
  for R in 0.8 1.6 3.2; do
    "$PY" -u tools/real_motion/eval_p0_f9_v21_stage0_upper_bounds.py \
      --config "$RUNTIME_CONFIG" --val-cache "$DEV_V18" \
      --population-manifest data/p0_f9_v21_dev64_manifest.json \
      --checkpoint "$V18_CLEAN_E14" \
      --expected-checkpoint-sha256 "$BASE_CKPT_SHA256" \
      --prototype-bank "outputs/p0_f9_v21_prototypes/prototype_bank_k$K.pt" \
      --dataroot "$DATAROOT" --info-pkl "$DEV_INFO" \
      --coverage-radius-m "$R" \
      --output "outputs/v21_stage0_dev64_k$K_r$R.json"
  done
done
```

只允许用 dev64 选择一次最小可用 K 和 coverage radius，随后冻结。

## 4. dev512 Stage-0B

```bash
"$PY" -u tools/real_motion/eval_p0_f9_v21_stage0_upper_bounds.py \
  --config "$RUNTIME_CONFIG" --val-cache "$DEV_V18" \
  --population-manifest data/p0_f9_v21_dev512_manifest.json \
  --checkpoint "$V18_CLEAN_E14" --prototype-bank "$SELECTED_PROTOTYPE_BANK" \
  --expected-checkpoint-sha256 "$BASE_CKPT_SHA256" \
  --dataroot "$DATAROOT" --info-pkl "$DEV_INFO" \
  --coverage-radius-m "$SELECTED_RADIUS" \
  --output outputs/v21_stage0b_dev512.json --enforce-stage0b-gate
```

硬门槛：deployable UB2 `ΔmIoU >= +0.50 pp`；保留 UB1 mIoU headroom >=70%；causal component coverage >=70%；mean legal candidates/positive <=10。Moving-Micro 按冻结协议报告，但不是对所有 target 的无条件否决门槛。

## 五条 oracle

1. `UB0_EXACT`：所有 report-horizon component target + query-entry exact shape + GT state。
2. `UB1_CAUSAL_EXACT`：只保留一对一 causal-covered target，shape/state 与 UB0 相同。
3. `UB2_DORMANT_CAUSAL_SHAPE`：DORMANT → last-observed real shape；BIRTH 仍 exact。
4. `UB2_BIRTH_PROTOTYPE`：BIRTH → oracle-best train prototype；DORMANT 仍 exact。
5. `UB2_DEPLOYABLE_REPRESENTATION`：DORMANT last-observed shape + BIRTH prototype。

UB0→UB1 只测 candidate coverage；两个 UB2 diagnostic 分别隔离 historical shape aging 与 prototype quantization；deployable UB2 用于最终 Stage-0B gate。全局 annotation 首次存在时间不等于 query-entry onset：例如物体 0.5s 已在 Ωmax 外存在、2.0s 才进入 occupancy/query domain，则保留完整 existence/state，但以 2.0s 的中心、frontier eligibility 和 exact shape 做 source activation。域外且所有 report horizon 都没有 occupancy component 的 annotation 不进入 component coverage 分母，仍单独报告 identity coverage。

## 输出审计

报告包括：IoU/mIoU/Moving Macro/Moving Micro、per-horizon/per-class delta、all V21 / Moving-eligible target mass、component/voxel coverage、history/frontier/uncovered、candidate budget、non-report-horizon-only、last-seen age source/voxel histogram、age-stratum UB、shape unresolved/ambiguous、addition precision/target recall、collision、blocked-by-V18、OOB 和 scene delta。额外的 `coverage_strata_diagnostic` 分别统计 annotation-onset shape、任意 future shape 和 report-horizon component：既报告它们在正式 all-target assignment 下的覆盖，也报告只在该 stratum 内重新一对一匹配的几何覆盖上限；该诊断不会修改正式 UB、compositor 或 Stage-0B gate。

## V18 exactness

Evaluator 默认至少检查第一个窗口：

1. 当前 V18 default forward 与逐行复刻 frozen `ccf7d77` forward elementwise 比较；
2. 复用 `benchmark_p0_f9_v18_runtime._exactness_check` 验证 Strong、SE(2) raster 和 A1 compositor；
3. 每个窗口验证空 V21 proposals 与 V18 逐 voxel 相等。
4. 每个 resolved future-onset exact shape 在原 horizon/pose 下 round-trip 后必须与 attributed Occ3D component 逐 voxel 完全一致。

任一失败立即停止。

正式运行前冻结 checkpoint 内容指纹：

```bash
export V18_CLEAN_E14="$BASE_CKPT"
export BASE_CKPT_SHA256="$(sha256sum "$V18_CLEAN_E14" | awk '{print $1}')"
```

Evaluator 同时强制 `training_mode=clean_one_stage_from_scratch_v1_tail_continuation`、`epoch=14` 和 SHA256 完全一致，不再只凭通用 protocol 接受任意 Clean epoch。

## 明确未实现

Source Induction Transformer、Event/State/Render loss、Positive32、Train1024/4096/full、learned V21 checkpoint、dense completion、global Null query、Autoencoder、4–6s rollout均未实现。只有 dev64 和 dev512 Stage-0B 支持这条表示路线后才进入 learned stage。
