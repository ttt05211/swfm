# P0-F9 Frozen Sparse OccFM Diagnostic

## Purpose

P0-F9 Stage-1 fails the deployment criterion badly. Before designing another training objective, this no-training diagnostic locates where the loss is introduced.

A direct comparison between raw dense OccFM and a sparse P0-F9 deployment is not sufficient, because the sparse result uses Strong-W2Det fallback plus dynamic-only fusion. The controlled diagnostic therefore evaluates the dense OccFM proposal under the exact same MSP support and fusion rule before comparing it to the sparse crop.

## Controlled states

`tools/real_motion/eval_p0_f9_frozen_sparse_occfm.py` reports six states on the exact same validation split:

1. `strong_anchor`: frozen Strong-W2Det;
2. `dense_official_raw`: released dense OccFM prediction;
3. `dense_official_same_support_fusion`: released dense OccFM used only as the proposal inside the exact P0-F9 MSP write support and dynamic-only fusion;
4. `frozen_sparse_official_cfg`: released OccFM transition weights in the current Top-2 20x20 sparse geometry, no new condition paths, CFG=2;
5. `frozen_sparse_p0f9_cfg`: the same frozen sparse model/noise, but CFG=1;
6. `same_support_gt_oracle`: exact GT proposal under the same support/fusion.

No P0-F9 trained checkpoint is loaded.

## Frozen sparse contract

The sparse branches:

- use the audited P0-F9 v2 validation cache;
- keep official `HIST_LAST=4`;
- keep native Gaussian -> absolute-future flow;
- use one global 50x50 Gaussian source field and crop it through the frozen Top-2 plan;
- keep absolute 50x50 coordinates for each 20x20 window;
- disable the new full-history context path;
- pass an all-zero physics prior;
- fail if token-prior projection, context projection, or the physics gate is not an exact no-op;
- load only shape-compatible released OccFM transition weights and keep the checkpoint-reuse gate;
- scatter sparse predictions back into the Strong-W2Det latent fallback;
- decode with the frozen official VAE;
- apply the same dynamic-only deployment fusion as P0-F9;
- verify the same-support GT oracle remains bit-exact with the cached repair target.

The two sparse variants use the same source noise. `CFG=2` matches the released OccFM config (`UNCOND_P=0.2`, `UNCOND_SCALE=2`); `CFG=1` matches the P0-F9 Stage-1 deployment choice.

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

For a GPU execution smoke first, add `--max-windows 2` and use a separate output path. The existing 128-window dense-baseline JSON is not compared to a partial smoke run.

## Four controlled deltas

The key outputs are:

```text
fusion_effect_dense_same_support_minus_strong
    = dense OccFM proposal under the exact P0-F9 fusion - Strong

sparse_geometry_effect_cfg2_minus_dense_same_support
    = frozen 20x20 sparse CFG2 - dense OccFM under the same fusion

guidance_effect_cfg1_minus_cfg2
    = frozen sparse CFG1 - frozen sparse CFG2

frozen_sparse_cfg1_minus_strong
    = the complete frozen P0-F9-style sparse deployment - Strong
```

The script also reports `oracle_headroom_minus_strong`.

## Interpretation

- If `dense_official_same_support_fusion` is already clearly below Strong, the current dynamic takeover fusion is unsafe even with a full dense OccFM proposal. The next method should not simply let a weaker WM erase Strong dynamics inside MSP.
- If dense same-support is acceptable but `frozen_sparse_official_cfg` drops strongly, the Top-2 20x20 sparse geometry/adaptation is losing native OccFM capability before any finetuning.
- If CFG2 is acceptable but CFG1 drops, changing the released OccFM inference policy contributes materially.
- If both frozen sparse variants are reasonable but trained P0-F9 is much worse, Stage-1 finetuning/objective is the main additional failure source.
- Compare trained P0-F9 step1200 against `frozen_sparse_p0f9_cfg` to isolate the finetuning contribution directly.

Do not launch another long training run before this controlled diagnostic is resolved.
