# V18 temporal-consistency / FVD diagnosis — 2026-09-18

This is a **diagnostic-only** protocol.  The frozen Clean-E14 main result is not
changed by this work.

## Goal

Explain why the frozen model can have strong IoU/mIoU/Moving-IoU while its
corrected-public-code occupancy FVD is less competitive.

The diagnosis compares the exact same 4,369-window validation population under:

1. Strong/KTA deterministic anchor;
2. historical Y600-pred;
3. frozen Clean-E14.

All three are evaluated with the same corrected OccFM temporal 3D-VAE FVD
extractor.  Clean-E14 reuses the already exported clips.

In parallel, source-level temporal diagnostics measure:
- velocity error;
- acceleration error;
- jerk error;
- predicted acceleration / jerk magnitude versus GT;
- yaw-rate error;
- yaw-acceleration error;
- semantic frame-change rate and excess versus GT;
- occupancy flip rate and excess versus GT;
- change-mask IoU against GT.

## Scripts

- `tools/real_motion/diagnose_p0_f9_v18_temporal_consistency.py`
- `tools/real_motion/eval_occfm_occupancy_fvd_only.py`

The first script exports Strong/KTA and Y600 clips and writes the motion/change
diagnostics.  The second computes only FVD and skips unnecessary FID/KID work.

## Decision rule after the diagnosis

### A. Clean FVD < Y600 FVD < Strong/KTA FVD

The learned SE(2) model already improves temporal quality.  Do **not** retrain
the main model for FVD.  Keep Clean-E14 frozen and report FVD as a secondary
diagnostic.  The residual gap is then mainly representation/compositor/protocol
limited rather than evidence that Clean harmed temporal consistency.

### B. Clean FVD < Strong/KTA FVD but Y600/Clean ordering is mixed

The learned correction helps over the deterministic prior, but the last-stage
SE(2) objective is not the dominant FVD bottleneck.  Inspect trajectory and
change-mask diagnostics.  Only continue if one clear error mode (for example
yaw-rate error or excess CLEAR/WRITE frame changes) is substantially worse in
Clean.

### C. Clean FVD > Strong/KTA FVD

The learned correction improves overlap metrics while damaging temporal
consistency.  Use the source-level diagnostics to choose exactly one controlled
follow-up:
- high acceleration/jerk error -> trajectory-consistency regularization;
- high yaw-rate/yaw-acceleration error -> yaw temporal regularization;
- normal source trajectories but high semantic/occupancy flip excess -> A1 /
  raster/composition temporal discontinuity, so do not add a trajectory loss.

### D. Strong/KTA, Y600 and Clean FVD are all similar

FVD is dominated by deterministic transport, feature-extractor sensitivity, or
the evaluation protocol rather than the learned head.  Stop optimizing FVD.
The paper should prioritize IoU/mIoU/Moving-IoU and runtime.

## Allowed follow-up if training is justified

Any follow-up is a separate ablation/extension, not a replacement for the
frozen main checkpoint.

The first training intervention is intentionally minimal.  If XY trajectory
jerk is the identified failure, add one weak finite-difference consistency term
on the six predicted source displacements:

`L = L_clean + lambda_temp * L_delta`

where `L_delta` compares adjacent predicted displacement increments with GT
increments.  Start with `lambda_temp in {0.05, 0.1}`; do not add VAE/FVD
perceptual losses or several new losses at once.

The candidate is accepted only if:
1. full-validation IoU/mIoU are not materially degraded;
2. Moving-Micro does not regress;
3. corrected same-protocol FVD improves;
4. the diagnosed temporal error (jerk/yaw/change excess) improves in the
   expected direction.

Otherwise retain frozen Clean-E14.
