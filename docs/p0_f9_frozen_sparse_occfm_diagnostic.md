# P0-F9 Frozen Sparse OccFM Diagnostic

## Purpose

P0-F9 Stage-1 fails the deployment criterion badly, while the released dense OccFM baseline is also substantially weaker than Strong-W2Det on the frozen 128-window split. Before designing another training objective, run one no-training diagnostic to locate where native OccFM capability is lost.

The diagnostic asks:

> Does the current Top-2 20x20 sparse adaptation already lose the released OccFM-Fut function before finetuning, or does the major loss appear after P0-F9 Stage-1 training?

## Frozen protocol

`tools/real_motion/eval_p0_f9_frozen_sparse_occfm.py`:

- loads only the released OccFM-Fut epoch=000196 transition weights;
- never loads a P0-F9 trained checkpoint;
- uses the audited P0-F9 v2 validation cache;
- keeps official `HIST_LAST=4`;
- keeps the native Gaussian -> absolute-future flow;
- uses a single global 50x50 Gaussian source field and crops it through the frozen Top-2 plan;
- keeps absolute 50x50 coordinates for each 20x20 window;
- disables the new full-context path;
- passes an all-zero physics prior and verifies the token prior projection, context projection, and physics gate are exact no-ops;
- scatters sparse predictions back into the Strong-W2Det latent fallback;
- applies the exact same dynamic-only deployment fusion and same-support GT-oracle check as P0-F9 evaluation.

The same frozen weights/noise are evaluated twice:

1. `frozen_sparse_official_cfg`: guidance scale `2.0`, matching released OccFM (`UNCOND_P=0.2`, `UNCOND_SCALE=2`);
2. `frozen_sparse_p0f9_cfg`: guidance scale `1.0`, matching P0-F9 Stage-1 deployment.

This separation is important: if CFG=2 preserves dense OccFM but CFG=1 drops, the inference-policy change is material and the result should not be blamed on sparse cropping alone.

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
  --output "$ROOT/outputs/p0_f9_v2_frozen_sparse_occfm_128.json" \
  --seed 20260904 \
  --amp
```

For a quick execution smoke first, add `--max-windows 2` and use a separate output path. Dense-baseline comparison is intentionally skipped for partial runs.

## Readout

The report contains:

- Strong-W2Det Overall / Moving;
- frozen sparse official-CFG Overall / Moving;
- frozen sparse P0-F9-CFG Overall / Moving;
- same-support GT oracle;
- delta of each sparse variant versus Strong;
- when the full dense-baseline JSON is supplied, delta of each sparse variant versus the exact dense official OccFM run;
- `CFG1 - CFG2` deltas.

Interpretation:

- large negative `frozen_sparse_official_cfg - dense_official`: sparse geometry/adaptation already loses native OccFM capability before finetuning;
- official-CFG near dense, but P0-F9-CFG much worse: guidance-policy change is a material source of loss;
- both frozen sparse variants near dense, while trained P0-F9 is much worse: Stage-1 finetuning/objective is the primary failure source.

Do not launch another long training run before this diagnostic is resolved.
