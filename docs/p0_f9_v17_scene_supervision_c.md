# P0-F9 V17-RL C: full-scene supervision with sparse computation

## Status before C

- A1 source-order WRITE is a small compositor improvement and is retained as the
  common hard evaluation compositor for C.
- Exact native footprint supervision (B-S) did not improve fixed-time occupancy
  metrics and is stopped.
- H0 found no evidence that late horizons are under-driven: residual-output
  gradient energy is approximately balanced across the six horizons.  C does not
  change horizon weights.

## Scientific question

Starting from the same V17-RL epoch-5 model, does adding final semantic scene
supervision improve the same decoder/head beyond the original motion objective?

The paired objectives are

```text
C-C: L = L_pos + L_exist + 0.25 L_overlap
C-S: L = L_pos + L_exist + 0.25 L_overlap + alpha(t) L_scene
```

No existing term is removed in the treatment arm.

## Scene-loss definition

`L_scene` is **full-scene 18-way CE**.  It is not an ROI-averaged CE and it does
not use a GT-moving filter.

For efficiency, differentiable probabilities are only materialized on

```text
all Strong/KTA source destination cells
union
current predicted source AABBs + interpolation halo
```

GT support is not included in the query-domain definition.  Outside this domain
the scene is the constant Strong/KTA anchor.  For each horizon the implementation
starts from the anchor, applies the already-audited legacy CLEAR rule at the KTA
source positions, then alpha-composites every predicted source in the original
Strong source order.  The resulting CE is divided by the complete `X*Y*Z` scene
voxel count before averaging horizons.  Thus sparse computation changes runtime,
not the scene-loss sampling or normalization rule.

This follows the useful full-scene-loss idea already present in MT-V1-STPN, but
C is implemented independently on the V17-STWM line and does not import the STPN
backbone or its training protocol.

## Common hard compositor and starting point

Full occupancy evaluation uses

```text
local_stwm_center_always_source_order
```

for both arms and the shared start.  This is A1: the historical CLEAR contract
plus replacement WRITE in original Strong source order.  The frozen RL epoch-5
A1 start was already measured before C, so any A1 compositor delta is not counted
as a C training gain.

## Parameters that train

The historical encoder is frozen.  The trainable future side is:

- `future_query` and `future_time_embedding`;
- `kta_future_proj`;
- future-query decoder blocks;
- residual head and existence head.

The shared `kinematic_proj` is frozen because it is part of the historical
representation as well as a decoder input.  Both C-C and C-S use exactly this
same freeze contract.

## Scene cache

A bounded scene-balanced subset of the existing V17 train windows is cached once
(default 1024 windows).  Selection is round-robin over train scenes and never
uses future GT content.  The cache stores exact Strong/KTA future anchors, future
semantic GT, original-order source voxel indices, and poses.  Model inputs and
trajectory targets continue to come from the existing V17 train cache.

The scene cache is intentionally shared by both arms.  There is no GT
true-moving source selection.

## Alpha calibration

Before either arm trains, run `--calibrate-only` once on a fixed training subset.
It records separate gradient norms for:

- position SmoothL1;
- weighted overlap (`0.25 * L_overlap`);
- their combined motion objective;
- unit-weight full-scene CE.

It also records position/overlap and scene/motion gradient cosines.  A single
fixed target alpha is then computed so the median unit scene gradient is a
pre-declared fraction (default 0.25) of the median combined-motion gradient.
This is a one-time scale calibration, not a hyperparameter sweep or dynamic
weighting scheme.  During C-S, alpha ramps linearly from zero to that fixed value
over the first 20% of continuation steps and then stays fixed.

Both paired runs are required to use the exact calibration result.  The control
still computes scene CE for matched data/runtime diagnostics but multiplies it
by zero.

## Short paired continuation

Default short-run contract:

```text
shared start      = V17-RL epoch 5
continuation      = 600 optimizer updates
midpoint          = step 300
endpoint          = step 600
scene batch       = 4 windows
paired seed       = 20260910
```

The historical AdamW state and original cosine LR schedule are restored.  The
same scene-shard order is used in both arms.  Frozen encoder tensors are checked
for bitwise identity at midpoint and endpoint.

The short experiment is judged only at fixed matched training times.  It does
not use SoftCH or balanced checkpoint selection to decide whether the new loss
works.

## Decision output

Run the normal V17 full evaluator on the A1 branch at the shared start, midpoint,
and endpoint.  Report both

```text
C-S minus C-C
C-S minus shared RL-epoch5+A1 start
```

for:

- occupancy IoU;
- semantic mIoU;
- Moving-mIoU.

A treatment gain must be interpreted jointly across these three metrics.  Loss,
ADE, or the calibrated scene CE alone cannot establish success.
