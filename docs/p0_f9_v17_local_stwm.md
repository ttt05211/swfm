# P0-F9 V17 Local-STWM: representation and transport-aware objective

V17 belongs **only** to the V16-STWM/main route. It does not touch or depend on
`feature/motion_transport_v1-spec2` / MT-V1-STPN.

## Why V17 exists

V16 produced lower center error than the v13 MLP on all-valid and true-moving
sources, including voxel-weighted and cross-track errors, yet its Moving-mIoU was
worse. This rules out simple under-capacity and shows two remaining mismatches:

1. the source-centered tube keeps local context but visually recenters the target
   source every frame, so per-frame motion is available mainly through the old
   flattened 46-D kinematic vector;
2. SmoothL1 center residual is not the same objective as overlap after rigid
   source-shape transport.

V17 keeps one learned STWM, one KTA-residual output, and one rigid-transport
path. There is no router, MLP/STWM ensemble, or multi-branch fusion.

## Controlled variants

- **R**: representation only. V16 loss is unchanged. Each history frame receives
  an explicit `[offset_x, offset_y, velocity_x, velocity_y, valid]` embedding and
  a causal target-source mask channel derived from the already cached local tube.
- **L**: loss only. `use_representation=False` calls the V16 forward path exactly;
  training adds a differentiable transport-overlap Soft-IoU auxiliary term.
- **RL**: both changes.

The overlap term translates the current source BEV footprint by the displacement
**error** `pred_residual - target_residual`. GT stays centered. Therefore no
future occupancy is read, no absolute future-motion canvas is needed, and
SmoothL1 remains the long-range attraction term when the two shapes do not
intersect.

Default objective for L/RL:

`L = SmoothL1(residual) + BCE(existence) + 0.25 * (1 - SoftIoU(transported footprint))`

The overlap weight is exposed as `--overlap-weight`, but the first controlled
run should keep the fixed default 0.25 rather than tune it.

## Safety contracts

- target remains `GT displacement - KTA displacement`;
- fresh residual head is zero, therefore fresh V17 == KTA;
- V17-L disabled representation uses the exact V16 forward path;
- source mask and frame-motion features are derived only from causal cached
  history; no future occupancy or future annotation is added;
- rigid transport still preserves the observed source-shape offset;
- V16 cache is reused and upgraded without rerunning nuScenes preprocessing.

## Cheap frozen-tube diagnostic

Before retraining, run `tools/real_motion/diagnose_p0_f9_v16_tube_ablation.py`.
It holds all kinematic inputs fixed and compares the original tube against:
`repeat_t0`, `reverse`, and `background`. This directly measures whether the
frozen V16 uses temporal visual changes versus only spatial context.

## Cache upgrade

Use `tools/real_motion/upgrade_p0_f9_v16_local_stwm_cache_v17.py` on both the
existing full-train and 128-window val V16 caches. It adds only:

- `frame_motion_features [N,6,5]`;
- `target_source_mask_tube [N,6,H,W]`.

The summary reports mask availability on valid tracked frames and at t0.

## Checkpoint selection

V17 saves both:

- `best.pt`: minimum validation **V17 objective**;
- `best_ade.pt`: minimum validation center ADE.

This is deliberate because V16 showed that the ADE-optimal checkpoint need not
be the occupancy-optimal checkpoint. Periodic epoch 1/5/10/final snapshots are
also retained.

## Decision protocol

Run R and L first. Only run RL after both single-variable results are known.

- If R improves 2s/3s true-moving trajectory and Moving-mIoU, the centered-tube
  representation was limiting the model.
- If L improves Moving-mIoU without necessarily improving ADE, the objective
  mismatch is confirmed.
- If both help independently, RL tests complementarity.
- Do not enlarge the Transformer or tune overlap weights before these controlled
  comparisons are complete.
