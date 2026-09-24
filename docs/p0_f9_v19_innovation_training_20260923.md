# P0-F9 V19 Innovation training — frozen-base phase

Date: 2026-09-23

## Status

The 64-window ancestry-v2 gate passed.

Key group perfect-add headroom:

- `core_innovation`: share 53.449%, delta mIoU +12.810, delta Moving-Micro +11.043;
- `memory_addressable`: share 38.261%, +7.265 mIoU, +0.307 Moving-Micro;
- `known_ancestor_model_miss`: share 8.274%, +10.645 mIoU, +26.992 Moving-Micro;
- `ambiguous`: share 0.017%, +0.104 mIoU, +0.393 Moving-Micro.

`dynamic_other_ambiguous` is only 494 voxels / 16 components in this smoke;
all are class-3 components whose nearest same-class annotation centre lies in
the 4--6 m bin. It is no longer a blocker and remains excluded from training.

The first learned phase is therefore approved:

    frozen Clean-E14/V18 Transport
    + deterministic Static Memory
    + Residual Innovation Head (only trainable module)

Dynamic Source Memory / reconciliation remains the lifecycle layer for
multi-block rollout. The observed 1--3 s training cache does not reintroduce
the previously negative naive dormant-KTA branch.

## Frozen Innovation responsibility contract

Positive categories:

- `future_birth_dynamic`
- `source_shape_innovation`
- `never_seen_static`

Ignored rather than treated as negative:

- `history_source_recoverable`
- `history_static_recoverable`
- `t0_unrepresented_dynamic`
- `current_source_transportable_miss`
- `history_static_seen_mismatch`
- `dynamic_other_ambiguous`
- `static_other_ambiguous`

If one BEV column contains both innovation-positive and excluded-responsibility
voxels, the whole column is ignored for the column-level presence/semantic
heads.

## Implemented files

- `real_motion/v19_innovation_training.py`
- `tools/real_motion/build_p0_f9_v19_innovation_cache.py`
- `tools/real_motion/train_p0_f9_v19_innovation.py`
- `tools/real_motion/eval_p0_f9_v19_innovation.py`
- `tests/test_p0_f9_v19_innovation_training.py`

The cache stores future-aligned historical BEV tensors, the frozen explained
BEV state, and supervision masks. Geometry is uint8-quantized and 16-bin
vertical occupancy targets are uint16 bit-packed.

The trainer checkpoint contains only `innovation_state_dict` for the learned
module. Clean-E14 is not placed in the optimizer.

## First smoke

Use the same server variables as the V19 handoff:

```bash
ROOT=/root/nas/occ/swfm
PY=/root/miniconda/envs/OccFM/bin/python
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main
DATAROOT="$OCCFM/data/nuscenes"
VAL_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
TRAIN_INFO="$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
V18_ALL="$ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
CLEAN_CKPT="$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
cd "$ROOT"
```

Recover the exact V18 train cache used by Clean-E14 from the checkpoint:

```bash
TRAIN_V18=$("$PY" - "$CLEAN_CKPT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(ck["args"]["train_cache"])
PY
)
echo "$TRAIN_V18"
```

Build a small train cache:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  tools/real_motion/build_p0_f9_v19_innovation_cache.py \
  --source-cache "$TRAIN_V18" \
  --checkpoint "$CLEAN_CKPT" \
  --dataroot "$DATAROOT" \
  --info-pkl "$TRAIN_INFO" \
  --output-dir "$ROOT/data/p0_f9_v19_innovation_train_smoke128" \
  --max-windows 128 \
  --shard-size 8 \
  --device cuda
```

Build the matched validation cache:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  tools/real_motion/build_p0_f9_v19_innovation_cache.py \
  --source-cache "$V18_ALL" \
  --checkpoint "$CLEAN_CKPT" \
  --dataroot "$DATAROOT" \
  --info-pkl "$VAL_INFO" \
  --output-dir "$ROOT/data/p0_f9_v19_innovation_val_smoke64" \
  --max-windows 64 \
  --shard-size 8 \
  --device cuda
```

Train only the Innovation Head:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  tools/real_motion/train_p0_f9_v19_innovation.py \
  --train-cache "$ROOT/data/p0_f9_v19_innovation_train_smoke128" \
  --val-cache "$ROOT/data/p0_f9_v19_innovation_val_smoke64" \
  --output-dir "$ROOT/outputs/p0_f9_v19_innovation_smoke128" \
  --epochs 2 \
  --batch-size 2 \
  --device cuda
```

Evaluate actual occupancy metrics:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  tools/real_motion/eval_p0_f9_v19_innovation.py \
  --val-cache "$V18_ALL" \
  --base-checkpoint "$CLEAN_CKPT" \
  --innovation-checkpoint "$ROOT/outputs/p0_f9_v19_innovation_smoke128/best.pt" \
  --dataroot "$DATAROOT" \
  --info-pkl "$VAL_INFO" \
  --max-windows 64 \
  --device cuda \
  --output "$ROOT/outputs/p0_f9_v19_innovation_smoke128_eval64.json"
```

The decisive comparison is:

    v18_static_innovation - v18_static

Loss reduction by itself is not a success criterion.

## Gate before full training

Proceed to the full train cache only if the smoke shows:

1. non-zero but not explosive additions;
2. validation presence recall/precision move away from the zero-initialized state;
3. `v18_static_innovation` improves matched mIoU over `v18_static`;
4. Moving metrics do not materially regress;
5. no GT/future identity enters evaluator inference input.

## Later clean joint-training phase

Do not start this before the frozen-base Innovation gate succeeds.

If Innovation is useful, the next experiment can train a clean unified learned
checkpoint from scratch containing both the V18 Transport predictor and the
Innovation Head. Static Memory and Source Reconciliation remain deterministic
system logic and therefore do not need trainable weights.

That later phase is for packaging/training cleanliness, not for changing the
Transport / Memory / Innovation responsibility definitions.


## 2026-09-23 update: dynamic-only v3 and fast-paths

The first dynamic-only balanced-v2 smoke learned the semantic target well but
over-generated badly: validation presence recall rose to 0.733 while precision
fell to 0.127; vertical recall was about 0.935 with precision about 0.227. The
64-window occupancy evaluation therefore regressed by -1.598 mIoU and -11.924
Moving-Micro relative to the static-memory base.

The next objective is intentionally conservative:

- positives remain only `future_birth_dynamic + source_shape_innovation`;
- presence uses all positives plus the hardest 4x negatives, with no large
  positive BCE weight;
- vertical occupancy uses symmetric BCE (`pos_weight=1`) by default;
- `best.pt` is selected by validation presence F0.5 (precision-favouring),
  while `best_loss.pt` is also retained.

Trainer:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -u \
  tools/real_motion/train_p0_f9_v19_innovation_hardneg.py \
  --train-cache "$ROOT/data/p0_f9_v19_innovation_dynamic_train_smoke128" \
  --val-cache "$ROOT/data/p0_f9_v19_innovation_dynamic_val_smoke64" \
  --output-dir "$ROOT/outputs/p0_f9_v19_innovation_dynamic_hardneg_smoke128" \
  --epochs 5 \
  --batch-size 2 \
  --hard-negative-ratio 4 \
  --vertical-positive-weight 1 \
  --device cuda
```

### Cache/eval acceleration

Single-process geometry was reduced without changing the prediction contract:

1. mask warps now use direct OR-scatter rather than semantic collision sorting;
2. semantic + observation support are warped with one coordinate transform;
3. cache generation reuses the already aligned 3D history coverage instead of
   running a second 6x6 mask-warp pass.

Exactness tests compare the fast mask/fused paths against the historical
semantic-warp implementation.

For large jobs, cache and evaluation also support deterministic window
sharding. Example for 4 GPUs:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i "$PY" -u \
    tools/real_motion/build_p0_f9_v19_innovation_cache.py \
    --source-cache "$TRAIN_V18" \
    --checkpoint "$CLEAN_CKPT" \
    --dataroot "$DATAROOT" \
    --info-pkl "$TRAIN_INFO" \
    --output-dir "$ROOT/data/p0_f9_v19_innovation_dynamic_train_full_part$i" \
    --positive-mode dynamic \
    --num-shards 4 \
    --shard-index $i \
    --shard-size 8 \
    --device cuda > "$ROOT/outputs/v19_cache_part$i.log" 2>&1 &
done
wait

"$PY" tools/real_motion/merge_p0_f9_v19_innovation_cache_shards.py \
  --inputs \
    "$ROOT/data/p0_f9_v19_innovation_dynamic_train_full_part0" \
    "$ROOT/data/p0_f9_v19_innovation_dynamic_train_full_part1" \
    "$ROOT/data/p0_f9_v19_innovation_dynamic_train_full_part2" \
    "$ROOT/data/p0_f9_v19_innovation_dynamic_train_full_part3" \
  --output-dir "$ROOT/data/p0_f9_v19_innovation_dynamic_train_full"
```

Evaluation uses the same `--num-shards/--shard-index` partition and can be
merged with `merge_p0_f9_v19_innovation_eval_shards.py`. Because each process
has its own model and NuScenes reader, this is intended for one process per GPU.
