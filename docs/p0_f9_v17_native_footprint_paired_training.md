# P0-F9 V17-RL native-footprint paired short training

This experiment follows the scoped B0 result.  It does not add the full ordered
compositor, horizon reweighting, or scene CE.

## Question

Does replacing the legacy center-gated 0.8 m overlap template with the exact
Strong t0 source XY footprint improve the same V17-RL model when overlap-gradient
scale is approximately matched?

## Frozen starting point

Both arms resume the same V17-RL epoch-5 checkpoint.  The model architecture,
V17 representation input (including the historical target-source mask), AdamW
state, batch size, remaining cosine schedule, source set, overlap eligibility,
and paired shuffle seed are identical.

The cache is augmented once with an additional loss-only field:

- `native_source_footprint_mask`: exact Strong source voxels projected to XY and
  cropped to the same 16 m source-centered local frame at native 0.4 m
  resolution.

No future occupancy or future annotation is read to build this field.

## Arms

- **B-C (control):** legacy V17 t0 target-source footprint at 0.8 m,
  `lambda_overlap = 0.25`.
- **B-S (source-exact):** exact Strong source footprint at 0.4 m,
  `lambda_overlap = 0.175`.

The B-S coefficient is fixed before training from the B0 observation that the
native overlap residual-output gradient is approximately 1.4--1.5x the legacy
one.  This is a gradient-scale calibration, not a hyperparameter sweep.

Both arms use `target_valid AND legacy_t0_mask_present` for overlap eligibility.
Therefore B-S does not gain extra labels merely because the exact footprint is
present more often.

## Continuation contract

The official short run starts at global epoch 5 and continues through global
epoch 10.  The trainer restores the epoch-5 optimizer state and verifies:

- V17-RL protocol and representation are unchanged;
- resume overlap weight is exactly 0.25;
- flattened train/val scene sets remain disjoint;
- optimizer step count equals `5 * len(train_loader)`;
- checkpoint LR equals the original 10-epoch cosine schedule at that step;
- initial validation ADE reproduces the frozen checkpoint within tolerance.

Every continuation epoch is saved.  The paired shuffle sequence is deterministic
and identical between B-C and B-S.  It is a new paired continuation sequence; the
experiment does not claim to reproduce the historical epoch-6-to-10 minibatch
order.

## Decision rule

This short experiment is judged at fixed matched training times, not by best
checkpoint selection or SoftCH screening:

- shared start: epoch 5;
- midpoint: epoch 8;
- endpoint: epoch 10.

At epoch 8 and epoch 10, run the same full V17 evaluator and compare B-S minus
B-C directly on:

- occupancy IoU;
- semantic mIoU;
- Moving-mIoU.

The historical A0 compositor branch `local_stwm_center_always` is the primary
comparison branch so the earlier A1 WRITE-order change does not enter the B
causal comparison.  Balanced checkpoint selection remains a final delivery tool,
not the decision rule for this paired short experiment.
