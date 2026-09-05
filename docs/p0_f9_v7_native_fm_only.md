# P0-F9 v7 Native-FM-Only Causal Control

## Why this version exists

The P0-F9 training-failure diagnostics showed that the inherited OccFM backbone is functionally damaged by the Stage-1 objective even though its parameter drift is small. The 16-batch gradient probe localized the dominant optimization authority to the decoded compact semantic CE:

- raw semantic / raw FM gradient norm ratio on inherited OccFM parameters: ~1232x;
- semantic / current `0.1 * FM` authority: ~12320x;
- CE / FM ratio: ~1224x;
- Lovasz / FM ratio: ~21.5x.

The training-distribution diagnostic separately showed that selecting 4096 MSP windows does not materially shift the full-window semantic distribution relative to the native train population, while Top-2 routing only produces a moderate motion enrichment. Zero-route samples are only 27/4096 (0.659%). Therefore v7 changes only the training objective and keeps the rest of P0-F9 fixed.

## Training contract

`tools/real_motion/train_p0_f9_v7_native_fm_only.py` uses the native OccFM conditional-flow-matching objective only:

```text
z_t = (1 - t) * z0 + t * z1
v*  = z1 - z0
L   = mean((v_theta(z_t, t, condition) - v*) ** 2)
```

The implementation deliberately keeps:

- the audited P0-F9 absolute-future cache;
- Gaussian flow source and absolute GT future latent target;
- one coherent full-grid Gaussian source field cropped into the two routed windows;
- frozen Top-2 `20x20` prediction windows and `40x40` context windows;
- inherited OccFM `HIST_LAST=4` contract;
- Strong-W2Det only as a physics condition and deployment fallback;
- the current pretrained/new differential optimizer groups;
- scene-balanced sampling;
- EMA;
- grad clip 5;
- NFE=10 deployment sampling.

The implementation deliberately removes from the world-model training graph:

- semantic sidecars;
- compact 9-class CE;
- Lovasz;
- one-step endpoint reconstruction;
- VAE decoder calls.

`--vae-ckpt` remains required only for provenance. Its SHA must match the latent cache metadata so the existing deployment evaluator can fail-close on the same representation.

## Formal 400-step causal run

Run this only after pulling the commit containing v7. The command is self-contained.

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main
TRAIN="$ROOT/data/p0_f9_v2_wm_train_top2_4096"
VAL="$ROOT/data/p0_f9_v2_wm_val_top2_128"
WM="$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt"
VAE="$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt"
OUT="$ROOT/outputs/p0_f9_v7_native_fm_only_400"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/train_p0_f9_v7_native_fm_only.py" \
  --train-cache "$TRAIN" \
  --val-cache "$VAL" \
  --upstream-ckpt "$WM" \
  --vae-ckpt "$VAE" \
  --output-dir "$OUT" \
  --steps 400 \
  --batch-size 8 \
  --num-workers 4 \
  --wm-lr 2e-5 \
  --new-lr 1e-4 \
  --weight-decay 1e-2 \
  --warmup-fraction 0.05 \
  --min-lr-ratio 0.2 \
  --sample-steps 10 \
  --uncond-prob 0.0 \
  --guidance-scale 1.0 \
  --ema-decay 0.999 \
  --grad-clip 5.0 \
  --val-every 100 \
  --seed 20260904 \
  --amp
```

Fresh training first evaluates and saves `step_0000.pt`, then saves `step_0100.pt`, `step_0200.pt`, `step_0300.pt`, and `step_0400.pt`. `best.pt` is selected by the fixed-`t=0.5` EMA validation FM MSE only. The deployment decision must not be made from this surrogate validation metric alone.

## Deployment evaluation

The existing evaluator accepts the v7 checkpoint because the P0-F9 architecture/cache/provenance contract is unchanged. Run the same 128-window evaluation separately for the checkpoints of interest.

Example for step 400:

```bash
PY=/root/miniconda/envs/OccFM/bin/python
ROOT=/root/nas/occ/swfm
OCCFM=/root/nas/occ/OccFM-NeurIPS2025-main
TRAINED="$ROOT/outputs/p0_f9_v7_native_fm_only_400/step_0400.pt"
OUT="$ROOT/outputs/p0_f9_v7_native_fm_only_step0400_eval_128.json"

CUDA_VISIBLE_DEVICES=0 \
"$PY" "$ROOT/tools/real_motion/eval_p0_f9_frozen_sparse_occfm.py" \
  --cache "$ROOT/data/p0_f9_v2_wm_val_top2_128" \
  --occfm-ckpt "$OCCFM/logs/occfm_fut/2s_3s_nusc_fut_traj/ckpt/epoch=000196.ckpt" \
  --vae-ckpt "$OCCFM/logs/occfm_vae/100ep_3docc_sem_voxel/ckpt/epoch=000100.ckpt" \
  --dense-baseline-json "$ROOT/outputs/p0_f9_v2_official_occfm_native_128.json" \
  --trained-sparse-ckpt "$TRAINED" \
  --use-ema \
  --output "$OUT" \
  --seed 20260904 \
  --amp
```

For the causal comparison, evaluate at least `step_0000`, `step_0100`, `step_0200`, and `step_0400` with separate output JSONs.

## Decision criteria

The first question is not whether 400-step v7 immediately beats Strong-W2Det. It is whether removing semantic-gradient domination restores a stable pretrained world-model optimization regime.

Compare against the frozen and failed P0-F9 references:

```text
frozen sparse:
FM MSE    ~0.085
FM cosine ~0.959
Overall    35.06
Moving      8.13

failed trained P0-F9:
FM MSE    ~0.621
FM cosine ~0.766
Overall    26.57
Moving      6.61
```

Primary v7 checks:

1. FM MSE should stay near the frozen scale or improve instead of exploding toward ~0.6.
2. FM cosine should not collapse.
3. Overall/Moving should no longer show the catastrophic Stage-1 degradation.
4. Dynamic-volume flooding should disappear or shrink sharply relative to the failed trained model.
5. Clear/stale/write diagnostics should be inspected before any new loss or routing change.

If v7 is stable but Moving remains insufficient, the next experiment should remain inside the native FM family, e.g. a carefully normalized true-motion-weighted FM objective. Do not reintroduce decoded semantic CE/Lovasz before v7 establishes the clean FM baseline.
