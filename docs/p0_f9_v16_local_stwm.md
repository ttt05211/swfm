# P0-F9 v16 — Local Spatial-Temporal World Model

## Purpose

v16 replaces the successful v13 two-layer MLP probe with one object-centric local
spatial-temporal world model.  It does **not** stack a second model on top of the
MLP and it does **not** restore OccFM/VAE/FM.

What is inherited from the successful v13 experiment is the *prediction
contract*, not the MLP weights:

1. causal Strong-W2Det source decomposition;
2. six-frame backward occupancy-only source history;
3. Strong/KTA constant-velocity displacement as the physics prior;
4. target = GT object displacement - KTA displacement;
5. the observed t0 3D source shape is preserved and rigidly transported;
6. CLEAR old + WRITE transported source is one coherent replacement event.

The fresh v16 residual head is exactly zero.  Therefore step-0 predicted motion
is exactly KTA.  This is enforced both in tests and at trainer startup.

## Local semantic tube

For every Strong source and each of the six history frames:

- history occupancy is ego-compensated into the current t0 frame;
- the crop center follows the causally backtracked Strong source center;
- a 16 m x 16 m local patch is extracted;
- 3D semantics are collapsed to a top-surface semantic BEV so vehicles and
  pedestrians are not hidden by road voxels below them;
- deterministic 2x2 pooling gives a 20x20 grid at 0.8 m resolution, with dynamic
  semantics taking priority in pooling collisions.

No future occupancy and no future annotations are read by this tube builder.
Future boxes remain training/evaluation supervision only through the already
frozen v13 displacement targets.

## Network

Input streams enter the same learned model:

- local semantic tube: `[B, 6, 20, 20]` semantic IDs;
- the v13 causal kinematic feature vector (class, size, KTA velocity, six-frame
  offsets/validity, segment velocities, etc.);
- six KTA future displacements.

The model uses:

1. semantic embedding + 2D convolutional spatial stem;
2. four factorized spatial-temporal blocks;
   - ConvNeXt-style depthwise spatial mixing per history frame;
   - temporal self-attention over the six frames at each source-relative cell;
3. six future queries conditioned on KTA displacement and the source kinematics;
4. two query decoder blocks with self-attention and cross-attention to the local
   spatio-temporal history;
5. six `(dx,dy)` KTA residuals and six existence logits.

The first experiment deliberately excludes yaw, confidence routing, class
balancing, true-motion weighting, occupancy CE/Lovasz, VAE, FM and any dense WM.
This keeps the comparison against v13 attributable to one change: replacing the
flattened MLP with explicit local spatial-temporal modeling.

## Safety / invariants

A v16 run is invalid if any of the following changes:

- target is anything other than GT displacement - KTA displacement;
- the source component is aligned to a GT box center instead of translated by
  object displacement;
- fresh residual output is non-zero;
- future occupancy is used as an input;
- Strong source ordering differs from the cached source ordering;
- GT-center rigid oracle no longer matches the v11/v13 rigid-transport contract.

## Expected experiment

First compare on the fixed 128-window scene-disjoint validation set:

- Strong/KTA: Overall 39.7457, Moving 21.3872;
- v13 MLP: Overall 42.2504, Moving 22.4781;
- GT-center rigid oracle: Overall about 48.91, Moving about 45.27;
- v16 Local-STWM: measured by the new evaluator.

The primary question is whether explicit spatial-temporal modeling improves
2 s / 3 s motion without sacrificing the KTA-residual / rigid-transport gains
already established by v13.
