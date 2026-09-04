# P0-F9 Frozen/Trained Sparse OccFM + Safe-Fusion Diagnostic

## Purpose

P0-F9 Stage-1 failed the deployment criterion badly. The controlled diagnostic localized the loss into destructive takeover fusion, sparse 20x20 adaptation, CFG, and finetuning; the dominant Moving loss came from takeover fusion.

This no-training ablation asks:

> Do the same dense, frozen-sparse, or trained P0-F9 proposals contain useful motion innovation if they are forbidden from erasing Strong-W2Det dynamic predictions?

No model is retrained. The evaluator only replays existing checkpoints and changes the occupancy-space fusion rule.

## Proposal sources

`tools/real_motion/eval_p0_f9_frozen_sparse_occfm.py` evaluates:

1. `strong_anchor`: frozen Strong-W2Det;
2. `dense_official_raw`: released dense OccFM prediction;
3. released dense OccFM as a proposal inside the frozen MSP support;
4. frozen Top-2 20x20 OccFM with released CFG=2;
5. frozen Top-2 20x20 OccFM with P0-F9 CFG=1;
6. optional trained P0-F9 Stage-1 checkpoint, replayed with its exact physics/context/cache provenance and EMA by default;
7. GT as the proposal, to measure fusion-specific oracle headroom.

The frozen sparse proposal branches use released weights only and disable new physics/context condition paths. The trained branch instead replays the actual P0-F9 Stage-1 architecture, including full-history context and Strong-W2Det physics conditioning.

## Three fusion rules

For every proposal source, the exact same frozen MSP support is fused in three ways.

### 1. `takeover`

Legacy P0-F9 behavior:

```text
inside support:
    clear every anchor dynamic voxel
    write every proposal dynamic voxel
```

This is the existing `apply_dynamic_repair` contract and remains unchanged for provenance compatibility.

### 2. `write_only`

Most conservative innovation rule:

```text
inside support:
    if anchor is dynamic:
        keep anchor bit-exact
    elif proposal is dynamic:
        write proposal dynamic
    else:
        keep anchor
```

The proposal can add missing dynamic evidence, but cannot delete or relabel an existing Strong-W2Det dynamic prediction.

### 3. `dynamic_union`

Non-destructive dynamic union:

```text
inside support:
    if proposal is dynamic:
        write proposal dynamic
    else:
        keep anchor
```

Proposal dynamic semantics may add a missing dynamic voxel or relabel an existing dynamic class, but proposal free/background/non-dynamic semantics can never erase anchor dynamic presence.

## Why both safe variants are needed

- `write_only` measures whether missing dynamic evidence alone is useful.
- `dynamic_union` additionally permits dynamic-class correction.

Their difference is diagnostic:

- `write_only > dynamic_union`: proposal class corrections are harmful; Strong dynamic semantics should remain authoritative;
- `dynamic_union > write_only`: proposal contributes useful class corrections in addition to new dynamic evidence.

GT is also evaluated under all three rules, so we know whether a non-destructive innovation design still has theoretical headroom before any new training.

## Full run, including trained step1200

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main
TRAINED="$ROOT/outputs/p0_f9_v2_native_sparse_stage1_4096/step_1200.pt"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/eval_p0_f9_frozen_sparse_occfm.py" \
  --cache "$ROOT/data/p0_f9_v2_wm_val_top2_128" \
  --occfm-ckpt "$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt" \
  --vae-ckpt "$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt" \
  --dense-baseline-json "$ROOT/outputs/p0_f9_v2_official_occfm_native_128.json" \
  --trained-sparse-ckpt "$TRAINED" \
  --use-ema \
  --output "$ROOT/outputs/p0_f9_v4_safe_fusion_ablation_128.json" \
  --seed 20260904 \
  --amp
```

For a GPU smoke first, add `--max-windows 2` and use a separate output path. The existing 128-window dense baseline is not compared against a partial smoke run.

## Main output table

The terminal prints:

```text
strong_anchor

dense_official_raw
dense_official_same_support_fusion       # takeover
dense_official_write_only
dense_official_dynamic_union

frozen_sparse_official_cfg               # takeover, CFG=2
frozen_sparse_official_cfg_write_only
frozen_sparse_official_cfg_dynamic_union

frozen_sparse_p0f9_cfg                    # takeover, CFG=1
frozen_sparse_p0f9_cfg_write_only
frozen_sparse_p0f9_cfg_dynamic_union

trained_p0f9_takeover
trained_p0f9_write_only
trained_p0f9_dynamic_union

same_support_gt_oracle                    # takeover GT oracle
same_support_gt_write_only_oracle
same_support_gt_dynamic_union_oracle
```

The legacy takeover state names for dense/frozen branches are intentionally preserved, so the new result remains directly comparable with the previous P0-F9 diagnostic.

## Key deltas

For every proposal source the JSON records:

```text
takeover_minus_strong
write_only_minus_strong
dynamic_union_minus_strong
write_only_minus_takeover
dynamic_union_minus_takeover
union_minus_write_only
```

Interpretation:

- real proposal `write_only` or `dynamic_union` > Strong: the WM contains useful sparse innovation once destructive CLEAR authority is removed;
- GT safe oracle > Strong but real safe fusion <= Strong: the non-destructive fusion principle has headroom, but proposal quality is insufficient;
- even GT safe oracle <= Strong: purely additive innovation is insufficient; future work needs selective learned clear/correction rather than unconditional takeover;
- `write_only > dynamic_union`: preserve Strong dynamic class semantics;
- `dynamic_union > write_only`: proposal contributes useful dynamic-class correction;
- trained safe fusion substantially above trained takeover: Stage-1 may have learned useful evidence that the old fusion was discarding;
- trained safe fusion still far below frozen safe fusion: Stage-1 objective itself remains damaging even after fixing fusion.

Do not launch another long training run until this result is known.
