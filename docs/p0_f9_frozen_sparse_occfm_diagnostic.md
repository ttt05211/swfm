# P0-F9 Frozen Sparse OccFM + Safe-Fusion Diagnostic

## Purpose

P0-F9 Stage-1 fails the deployment criterion badly. The first controlled diagnostic already localized the loss into destructive takeover fusion, sparse 20x20 adaptation, CFG, and finetuning. The dominant Moving loss came from the takeover fusion itself.

The next no-training ablation therefore asks a narrower question:

> Does the same dense/frozen-sparse OccFM proposal contain useful motion innovation if it is forbidden from erasing Strong-W2Det dynamic predictions?

No new model is trained. The exact same proposal predictions and MSP support are evaluated under multiple fusion contracts.

## Proposal sources

`tools/real_motion/eval_p0_f9_frozen_sparse_occfm.py` evaluates:

1. `strong_anchor`: frozen Strong-W2Det;
2. `dense_official_raw`: released dense OccFM prediction;
3. released dense OccFM as a proposal inside the P0-F9 MSP support;
4. frozen Top-2 20x20 OccFM with released CFG=2;
5. frozen Top-2 20x20 OccFM with P0-F9 CFG=1;
6. GT as the proposal, to measure fusion-specific oracle headroom.

The sparse proposal branches still use released weights only: no P0-F9 trained checkpoint, no full-history context branch, and no physics-condition branch.

## Three fusion rules

For every proposal source, the exact same support is fused in three ways.

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

The learned model can add missing dynamic evidence, but it cannot delete or relabel any Strong-W2Det dynamic prediction.

### 3. `dynamic_union`

Non-destructive dynamic union:

```text
inside support:
    if proposal is dynamic:
        write proposal dynamic
    else:
        keep anchor
```

Therefore proposal dynamic semantics may add a missing dynamic voxel or relabel an existing dynamic class, but proposal free/background/non-dynamic semantics can never erase anchor dynamic presence.

## Why both safe variants are needed

`write_only` answers whether missing dynamic evidence alone is useful.

`dynamic_union` additionally tests whether the proposal has useful dynamic-class corrections. Their difference is diagnostic:

- `write_only > dynamic_union`: proposal class corrections are harmful; Strong dynamic semantics should remain authoritative;
- `dynamic_union > write_only`: proposal contributes useful class corrections in addition to new dynamic evidence.

GT is evaluated with all three rules as well. This tells us whether a non-destructive innovation design still has enough theoretical headroom before any new model is trained.

## Full run

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/eval_p0_f9_frozen_sparse_occfm.py" \
  --cache "$ROOT/data/p0_f9_v2_wm_val_top2_128" \
  --occfm-ckpt "$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt" \
  --vae-ckpt "$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt" \
  --dense-baseline-json "$ROOT/outputs/p0_f9_v2_official_occfm_native_128.json" \
  --output "$ROOT/outputs/p0_f9_v3_safe_fusion_ablation_128.json" \
  --seed 20260904 \
  --amp
```

For a GPU execution smoke first, add `--max-windows 2` and use a separate output path.

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

same_support_gt_oracle                    # takeover GT oracle
same_support_gt_write_only_oracle
same_support_gt_dynamic_union_oracle
```

The old takeover state names are intentionally preserved so the result remains directly comparable with the previous P0-F9 diagnostic.

## Key deltas

For each proposal source the report records:

```text
takeover_minus_strong
write_only_minus_strong
dynamic_union_minus_strong
write_only_minus_takeover
dynamic_union_minus_takeover
```

Interpretation:

- real proposal `write_only` or `dynamic_union` > Strong: learned WM contains useful sparse innovation once destructive clear authority is removed;
- GT safe oracle > Strong but real safe fusion <= Strong: the fusion principle is viable, but proposal quality is still insufficient;
- even GT safe oracle <= Strong: purely additive/non-destructive innovation is insufficient, so future work needs selective learned clear/correction authority rather than unconditional takeover;
- `write_only > dynamic_union`: preserve Strong dynamic class semantics;
- `dynamic_union > write_only`: proposal contributes useful dynamic class correction.

Do not launch another long training run until this safe-fusion result is known.
