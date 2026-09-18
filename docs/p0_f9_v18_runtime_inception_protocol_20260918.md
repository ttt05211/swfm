# V18 runtime + occupancy inception metric protocol — 2026-09-18

This document adds measurement infrastructure after the main Clean-E14 method/training/validation freeze. None of these scripts changes the frozen model, loss, checkpoint, source set, compositor, or main IoU/mIoU/Moving-IoU results.

## 1. Why validation wall-clock is not FPS

The 4,369-window full evaluator reconstructs Strong/KTA, evaluates multiple variants, builds GT moving support, rasterizes rigid components, accumulates dense metrics and runs scene bootstrap. Its wall-clock must never be converted into method FPS.

OccFM's released implementation measures `cfm_eval` with CUDA events. That timer starts after cached latent preparation and includes flow sampling plus the occupancy decoder; GT loading and metric accumulation happen outside the timer.

## 2. Frozen runtime benchmark

Script: `tools/real_motion/benchmark_p0_f9_v18_runtime.py`

The formal checkpoint is the exact Clean-E14 path stored in the frozen
provenance.  Its metadata mode is
`clean_one_stage_from_scratch_v1_tail_continuation`: it is the same
from-scratch clean V18 objective/model continued with restored optimizer state
and fixed tail LR, not a Balanced or V17-warm-start checkpoint.

Report these boundaries:
- `neural_model`: cached source tensors on GPU -> six motion outputs.
- `cached_representation_forecast_6frames`: main OccFM-comparable boundary. Frozen cached causal source representation and precomputed deterministic Strong/KTA prior -> six dense future occupancy grids. Includes Clean forward, predicted SE(2) rigid transport and hard-A1 composition.
- `strong_kta_prior_rebuild_6frames`: deterministic prior reconstruction.
- `full_causal_forecast_in_memory_6frames`: prior rebuild + Clean + render + composition.
- `source_extract_match`: t-1/t0 connected components and matching.
- `occworld_style_fps`: repository-native diagnostic that mechanically follows
  the historical OccWorld expression `per_frame = encode + autoreg / N_future`.
  It is **not** the paper main-table FPS.
- paper main-table FPS: six-frame-amortized generation throughput,
  `FPS = N_future / T_generate_N_future`. For the frozen v3 result this is
  `6 / 100.746246 ms = 59.5556 FPS`.

A direct public-code audit shows that I2-World and GenieDrive store test time as
`(end_time - start_time) / test_future_frame / bs`, i.e. they amortize the
timed multi-frame generation block over the generated future frames. OccFM's
maintainer likewise clarified in issue #2 that the reported FPS is the
six-frame generation speed amortized to a single future frame. Therefore
`occworld_style_fps=20.6193` is retained only for diagnostic compatibility
with the historical OccWorld repository formula and must not be placed in the
paper main efficiency table.

The main V18 generation timer includes Strong/KTA reconstruction, Clean
inference, predicted SE(2) rendering and final dense A1 composition. A stricter
transparency value may additionally charge the separately measured source
extraction/matching stage; that value is not the primary generation FPS.

All timings exclude disk I/O, GT and metric computation. The paper main efficiency row uses the full causal six-frame generation latency and reports both latency and `6 / latency` FPS. Keep source-extraction timing and cached-representation timing as transparency rows. Hardware must be stated, and published RTX-4090 FPS should not be treated as a hardware-matched speedup against an L40S result.

FLOPs are optional. The script first tries PyTorch `FlopCounterMode` and falls back to profiler-supported FLOPs. This applies to the learned Clean forward only, not deterministic geometry.

## 3. FID/FVD decision

FID/FVD are not universal requirements for occupancy forecasting; IoU/mIoU remain the primary forecasting metrics. They are nevertheless worth adding because OccFM is a principal comparison and explicitly reports occupancy-space FID/KID and six-frame FVD.

Do not use RGB ImageNet Inception or RGB-video I3D FVD. The closest protocol is OccFM's released occupancy VAE feature space.

- FVD: useful secondary/main-table metric for 3 s temporal consistency; use all six future occupancy frames.
- FID: useful secondary metric at 1 s / 2 s / 3 s.
- KID: useful companion, but label our estimator explicitly because OccFM does not release its exact KID implementation.
- F3D/MMD from UniScene is generation-oriented and is not required unless a later experiment directly targets their occupancy generation setting.

Because frozen V18 uses future ego poses, the fair OccFM reference is the future-trajectory-conditioned OccFM row, not the history-only row.

## 4. Frozen occupancy clip export

Script: `tools/real_motion/export_p0_f9_v18_occfm_inception_cache.py`

Default output is compressed `clip_XXXXX.npz` per validation window. `pred` and `gt` are uint8 `[6,200,200,16]` at 0.5/1.0/1.5/2.0/2.5/3.0 s. Prediction is completed before future GT is loaded. Optional `--official-npy-dir` additionally writes `pred_i.npy` / `gt_i.npy` in OccFM's public loader format, but this is storage-expensive.

## 5. OccFM-aligned evaluator

Script: `tools/real_motion/eval_occfm_occupancy_inception_metrics.py`

FVD uses OccFM's released temporal occupancy 3D-VAE (`tools/cfgs/occfm_3dvae.yaml`, epoch 40). It follows the public feature contract: six-frame clip -> `sampled_features` -> adaptive average pool 5x5 -> flatten six frames -> Fréchet distance.

FID uses OccFM's released single-frame occupancy VAE (`tools/cfgs/occfm_vae.yaml`, epoch 100) at future indices 1/3/5 = 1/2/3 s, with the same 5x5 pooled latent feature construction.

KID uses those single-frame features with a fixed standard unbiased degree-3 polynomial-kernel MMD (gamma=1/d, coefficient 1), 100 subsets, subset size 1,000 by default. The exact OccFM KID implementation is not released, so direct KID comparison requires recomputing compared methods with this evaluator.

## 6. Public OccFM script audit

The current released `tools/test_fid.py` loads GT and prediction but then contains `pred = gt[:, indices, ...]` before feature extraction. That overwrites the model prediction and is consistent with a Reorder-GT diagnostic rather than ordinary model evaluation. After DataLoader batching, the documented clip is `[1,6,200,200,16]`, so axis 1 is the six-frame time axis. The overwrite still makes the released line a Reorder-GT-style diagnostic rather than ordinary prediction FVD.

Our evaluator follows the intended released feature-extractor contract but does not reproduce that overwrite. Optional `--reorder-gt-sanity` instead correctly shuffles the six-frame time axis.

Fairness rules:
1. actual prediction is used for model FID/FVD;
2. same released feature-extractor checkpoints for every method;
3. same validation windows and horizon convention;
4. one prediction per input for deterministic methods; no oracle selection;
5. published OccFM/OccWorld/DOME inception numbers are contextual until verified with the corrected common evaluator.

## 7. Paper reporting recommendation

Keep IoU / mIoU / Macro Moving / Micro Moving as the frozen main accuracy table. Add Params and hardware-matched latency/FPS as efficiency results. Put occupancy FVD in the main comparison if space permits; put FID/KID at 1/2/3 s in a secondary table or appendix unless they become especially important.

Do not add RGB FID/FVD, downstream video-generation FVD, or a new feature extractor trained specifically for V18. Using OccFM's released occupancy extractors avoids post-hoc metric selection that could favor our method.