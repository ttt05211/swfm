# P0-F9 V18 main experiment freeze — 2026-09-17

This document freezes the main-method implementation, training protocol, validation protocol and headline results before paper writing and ablation-only work begins.

## 1. Frozen implementation

- Repository: `ttt05211/swfm`
- Branch: `feature/v17-se2-motion`
- Frozen implementation baseline commit: `f379739cd074496a747851d683dbc691ce3be697`
- Main model: clean one-stage V18 source-centred SE(2) motion predictor.
- Main training script: `tools/real_motion/train_p0_f9_v18_se2_clean.py`
- Formal all-window evaluator: `tools/real_motion/eval_p0_f9_v18_all_window_validation_fast_strict.py`
- The optimized evaluator is required to preserve the historical per-window BF16 model-forward path and exact Strong/KTA, rigid-raster, hard-A1 and raw-count contracts. The built-in first-window exactness checks must pass.

From this point, paper ablations must not silently modify the frozen main model, main loss, data population, hard compositor, evaluation support, or metric aggregation. Any intentional change is an ablation/new branch and must be labeled as such.

## 2. Frozen method contract

The method predicts future motion for causal Strong occupancy sources. Static/background transport remains deterministic. Motion-capable source geometry is transported rather than regenerated densely.

V18 extends the historical planar-translation predictor with source-centred SE(2):

- XY source-centre displacement correction relative to KTA.
- One relative-yaw scalar per horizon for yaw-enabled classes.
- Exact observed source 3D geometry is rigidly transported to future ego coordinates.
- Yaw-enabled semantic classes: bicycle, bus, car, construction vehicle, motorcycle, trailer and truck.
- Pedestrian remains translation-only.
- Deployment uses predicted yaw; no GT yaw is used at inference.
- Hard A1 compositor uses the frozen legacy CLEAR rule and original Strong source write order.

Geometry contract:

`d_s = (a_h - a_0) + (R - I)(c_s - a_0)`

with transported source point

`p' = R(p - c_s) + c_s + d_s`,

which is equivalent to the GT box-centred planar rigid transform while preserving the observed source geometry.

## 3. Frozen clean training protocol

Training mode: `clean_one_stage_from_scratch_v1`.

The model is trained from scratch with zero-initialized XY/yaw output heads. There is no V17->V18 continuation in the main method.

Objective from step 1 to the end:

`L = L_trans + L_exist + 19.0 * L_yaw + 0.25 * L_shape_SE2`

where:

- `L_trans`: SmoothL1 on source-centre XY residual.
- `L_exist`: BCE on source existence.
- `L_yaw`: periodic yaw loss.
- `L_shape_SE2`: differentiable GT-relative SE(2) source-footprint overlap loss.

Optimizer/schedule are the frozen Clean protocol in `train_p0_f9_v18_se2_clean.py`; no long-tail sampler, anti-regret loss, confidence gate, extra scene CE, Lovasz, native-footprint replacement objective, voxel flow, or post-hoc two-wheel weighting belongs to the main method.

### Training population

Official nuScenes temporal train split:

- 700 scenes.
- 28,130 raw frames.
- 20,430 eligible stride-1 6-history + 6-future windows.
- Actual V18 train cache: 20,430 unique windows, 700 unique scenes, no duplicate windows.
- Cache: `data/p0_f9_v18_se2_train_full.pt`.
- Cache lineage is the audited full-all-eligible train path.

The train cache contains all eligible windows; small 1,024/4,096-window development subsets are not the formal training population.

## 4. Frozen validation protocol

Final/main validation is the complete official temporal validation population, not the historical 128-midpoint diagnostic subset.

Official nuScenes temporal validation split:

- 150 scenes.
- 6,019 raw frames.
- 4,369 eligible stride-1 6-history + 6-future windows.
- Every eligible overlapping window is evaluated.

Formal validation cache:

`data/p0_f9_v18_se2_val_all_4369.pt`

The old `data/p0_f9_v18_se2_val_128.pt` remains a development/diagnostic subset only and must not be used as the final paper performance table.

### Frozen metrics

Report horizons: 1.0 s, 2.0 s, 3.0 s.

Report:

- IoU.
- mIoU.
- original Macro Moving-mIoU.
- horizon-first Micro Moving-IoU.
- per moving class x horizon breakdown when diagnosing behavior.

Moving support: frozen interval-displacement Moving-mIoU-v2 GT support.

Statistical comparison: paired scene-level bootstrap. For each bootstrap draw, all overlapping windows inside sampled scenes are re-aggregated as raw intersections/unions before metrics are recomputed. Window-level resampling and averaging scene IoU are not permitted.

## 5. Frozen main checkpoints / references

Primary model: **Clean-E14**.

References:

- **Strong/KTA anchor**: deterministic causal anchor.
- **Y600-pred**: historical continuation model used as the principal learned reference.

The exact Y600 and Clean checkpoint paths are the frozen provenance entries in:

`outputs/p0_f9_v18_two_wheel_diag/two_wheel_diag.json`

and are also recorded in the completed full-validation result:

`outputs/p0_f9_v18_all_window_validation_fast_strict.json`.

Do not substitute Balanced-E10, Clean-E10, or later exploratory checkpoints into the main table.

## 6. Frozen full-validation headline results

150 scenes / 4,369 windows:

| Variant | IoU | mIoU | Macro Moving | Micro Moving |
|---|---:|---:|---:|---:|
| Strong/KTA | 53.0900 | 40.0515 | 25.8641 | 27.8619 |
| Y600-pred | 53.5783 | 42.9892 | 26.8855 | 30.0794 |
| Clean-E14 | **53.6142** | **43.1178** | **27.2753** | **30.9144** |

Clean-E14 minus Y600-pred, paired scene bootstrap:

| Metric | Delta (pp) | 95% CI |
|---|---:|---:|
| IoU | +0.0359 | [+0.0221, +0.0505] |
| mIoU | +0.1285 | [+0.0255, +0.2380] |
| Macro Moving | +0.3898 | [-0.2069, +1.0794] |
| Micro Moving | +0.8349 | [+0.4704, +1.1872] |

Clean-E14 minus Strong/KTA on the same full validation population:

- IoU: +0.5242 pp.
- mIoU: +3.0663 pp.
- Macro Moving: +1.4112 pp.
- Micro Moving: +3.0525 pp.

The final two-wheel diagnosis on all windows does **not** support a stable systematic motorcycle regression. Motorcycle deltas are small and all scene-bootstrap CIs cross zero. Bicycle remains a mild negative trend, but all reported CIs also cross zero. Therefore no two-wheel-specific loss/sampler/gate is part of the frozen main method.

## 7. Negative / diagnostic branches excluded from the main method

The following remain useful negative evidence or diagnostics but are not part of the final method:

- two-wheel yaw-zero intervention;
- class-aware x motion-density long-tail resampling;
- Balanced-E10;
- Moving-Safe / Need-Score / prior Router experiments;
- extra anti-regret/safety losses;
- per-class hand weighting.

In particular, the long-tail sampler is frozen as a negative result: poor prediction on hard/rare examples does not imply that oversampling them improves the final predictor.

## 8. Runtime reporting is intentionally separate from validation wall-clock

The 4,369-window evaluator performs much more than model inference: raw occupancy/pose access, Strong/KTA reconstruction, connected components and matching, GT moving-support construction, per-source rigid rasterization for multiple variants/horizons, hard compositing, full-grid semantic/moving counts and scene-bootstrap bookkeeping. Its wall-clock time must not be reported as model FPS.

The paper runtime/FPS number must be measured separately with a dedicated inference benchmark and a clearly frozen timing boundary. Runtime benchmarking is the only remaining main-table infrastructure item; it may not change model outputs or the frozen main-method protocol.

## 9. Phase transition

As of this freeze, the main experiment is complete. Subsequent work is limited to:

1. paper writing/figures/tables;
2. controlled ablations of frozen components;
3. runtime/FPS/GFLOPs/parameter reporting under an explicitly defined timing protocol;
4. diagnostics needed to explain results, without retroactively changing the main method.
