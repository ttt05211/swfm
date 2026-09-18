# P0-F9 V18 paper main-table freeze — 2026-09-18

This file freezes the data/checkpoint/evaluation/runtime configuration intended
for the paper main tables.  It does not change the frozen Clean-E14 model or any
accuracy result.

## 1. Main checkpoint and data population

Primary model: **Clean-E14**.

Checkpoint:
`outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt`

Checkpoint metadata:
- protocol: `p0_f9_v18_se2_clean_train_v1`
- training mode: `clean_one_stage_from_scratch_v1_tail_continuation`
- epoch: 14
- global step: 18,410

Training population:
- nuScenes official temporal train split;
- 700 scenes;
- 20,430 unique eligible stride-1 windows;
- six history frames + six future frames;
- cache: `data/p0_f9_v18_se2_train_full.pt`.

Main validation population:
- nuScenes official temporal validation split;
- 150 scenes;
- 4,369 eligible stride-1 windows;
- every eligible overlapping window is evaluated;
- cache: `data/p0_f9_v18_se2_val_all_4369.pt`.

Prediction horizons are 0.5/1.0/1.5/2.0/2.5/3.0 s.  Accuracy tables report
1.0/2.0/3.0 s aggregates under the frozen evaluator.

Future ego poses are part of the frozen input protocol for deterministic
ego-motion transport/KTA and the matched future-trajectory-conditioned OccFM
comparison.  Future semantic/instance GT is never used for prediction.

## 2. Frozen main accuracy row

Full 150-scene / 4,369-window validation:

| Method | IoU | mIoU | Macro Moving | Micro Moving |
|---|---:|---:|---:|---:|
| Clean-E14 | 53.6142 | 43.1178 | 27.2753 | 30.9144 |

References on the same population:

| Method | IoU | mIoU | Macro Moving | Micro Moving |
|---|---:|---:|---:|---:|
| Strong/KTA | 53.0900 | 40.0515 | 25.8641 | 27.8619 |
| Y600-pred | 53.5783 | 42.9892 | 26.8855 | 30.0794 |

The paired scene-bootstrap Clean-E14 minus Y600-pred confidence intervals remain
those frozen in `docs/p0_f9_v18_main_experiment_freeze_20260917.md`.

## 3. Main-table efficiency protocol

Formal runtime script:
`tools/real_motion/benchmark_p0_f9_v18_runtime.py`

Formal runtime configuration:
- GPU: NVIDIA L40S
- AMP: BF16
- warmup windows: 20
- measured windows: 200
- exactness windows: 8
- selection seed: 20260918
- future frames per forecast: 6
- exactness gate: reference components + all-six Strong anchors + per-source
  KTA baseline footprints/CLEAR + cached target centres + vectorized SE(2)
  raster + hard A1.
- all disk I/O, GT loading and metric accumulation are outside the timer.

Frozen v3 runtime result:

| Quantity | Value |
|---|---:|
| Parameters | 2.073220 M |
| Learned forward FLOPs | 28.570391 GFLOPs |
| Peak CUDA memory | 213,963,776 bytes (~204.1 MiB) |
| Neural forward | 7.4003 ms |
| Strong/KTA six-frame prior | 86.3618 ms |
| Full causal six-frame forecast | **100.7462 ms** |
| Full causal six-frame windows/s | 9.9259 |
| **Six-frame-amortized FPS** | **59.5556 FPS** |
| Source extraction + matching | 31.7073 ms |
| Strict source-extract + full-forecast amortized FPS | 45.2989 FPS |

### Main FPS definition

The paper efficiency row uses:

```
FPS = N_future / T_generate_N_future
    = 6 / T_generate_6_frames
```

For Clean-E14:

```
T_generate_6_frames = 100.746246 ms
FPS = 6 / 0.100746246 = 59.5556
```

The timed generation block starts from the prepared causal source
representation and includes deterministic Strong/KTA future reconstruction,
Clean prediction, predicted source-centred SE(2) rendering and final hard-A1
dense occupancy composition.  It therefore does **not** use the much narrower
`cached_representation_forecast_6frames` value as the paper FPS.

This six-frame amortization matches the public timing arithmetic in the
I2-World and GenieDrive forecasting code:
- I2-World:
  `mmdet3d/models/ii_world/world_model/ii_world.py` stores
  `(end_time - start_time) / test_future_frame / bs`.
- GenieDrive:
  `occ_gen/mmdet3d/models/ee_world/world_model/ee_world.py` uses the same
  per-future-frame division for both naive and end-to-end test paths.
- OccFM's maintainer also clarified in public issue #2 that its reported FPS is
  the six-frame generation speed amortized to a single future frame and that
  compared methods were re-measured under the same convention.

The historical OccWorld repository expression
`encode + autoreg / N_future` is retained only as a repository-native
diagnostic/compatibility value.  The benchmark's
`occworld_style_fps=20.6193` must **not** be used as the paper main-table FPS.

For transparency, a stricter value that additionally charges the separately
measured source extraction/matching stage is:

```
6 / ((100.746246 + 31.707275) ms) = 45.2989 FPS
```

This can be reported in an appendix/runtime breakdown but is not the main
forecasting-generation FPS.

Hardware must always be stated.  Do not claim a direct speedup over published
RTX-4090 numbers as if they were hardware matched; use same-card reruns when
making a hardware-normalized speedup claim.

## 4. Inception-style metrics pending/final protocol

Occupancy FVD/FID/KID use the frozen Clean-E14 predictions on the same 4,369
validation windows.

- FVD: all six future frames; OccFM released `occfm_3dvae`,
  `epoch=000040.ckpt`.
- FID: future 1/2/3 s frames; OccFM released single-frame `occfm_vae`,
  `epoch=000100.ckpt`; report per horizon and arithmetic mean.
- KID: the same single-frame features, unbiased degree-3 polynomial-kernel
  estimator, 100 subsets, subset size 1,000, seed 1000; report per horizon and
  arithmetic mean.
- The public OccFM `tools/test_fid.py` prediction-overwrite line is not used.
  Actual model predictions are evaluated.
- Published FID/FVD/KID values from other papers are contextual unless their
  predictions are recomputed through the same corrected feature pipeline.

Scripts:
- `tools/real_motion/export_p0_f9_v18_occfm_inception_cache.py`
- `tools/real_motion/eval_occfm_occupancy_inception_metrics.py`

Once FID/FVD/KID finish, append their frozen values here without changing the
checkpoint, validation population, feature extractors or estimator settings
above.
