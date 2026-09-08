# P0-F9 v15: true-motion-aware trajectory objective

## Motivation

The converged v13 rigid-transport model improved all-valid center ADE by about
41%, but v14 showed only about 28% ADE reduction on observations that are truly
moving under the frozen Moving-mIoU v2 rule.  True-moving observations were only
about one third of valid trajectory labels.  V14 also showed severe class-count
imbalance while Moving-mIoU macro-averages the eight dynamic classes.

V15 changes only the trajectory loss weights.  It does **not** change the v13
Strong source decomposition, six-frame causal features, MLP architecture,
displacement target, KTA anchor, existence BCE, optimizer, schedule, seed,
training budget, or rigid-transport evaluator.

## Variants

### M: true-motion weighting

For every valid source/horizon observation, true motion is defined exactly as in
Moving-mIoU v2: GT interval center speed from t0 to that horizon is at least
0.5 m/s.  This GT flag is training/validation supervision only.

- non-true-moving valid trajectory observation weight = 1
- true-moving valid trajectory observation weight = 2

With a raw true-moving fraction near one third this makes roughly half of the
effective trajectory-loss mass come from true-moving observations without
throwing away the rest of the data.

### MC: M + macro class redistribution

MC keeps the same total true-moving weight mass as M and only redistributes it
across dynamic classes.  On the training set, for each class c with n_c
true-moving observations,

`class_factor_c = N_true_moving / (K_present * n_c)`.

Therefore every class present in training contributes the same total
true-moving mass and the mean class factor across true-moving observations is
exactly one.  Non-moving observations stay weight 1.  No clipping or extra
class-balance hyperparameter is introduced.

## Checkpoint selection

For direct comparability with v13, formal `best.pt` is still selected by
scene-disjoint **all-valid learned ADE**.  The trainer also saves
`best_true_moving.pt` as a secondary diagnostic only.  Occupancy comparisons
should use `best.pt` first.

## Run contract

Use the existing v13/v2 full train and 128-window validation caches.  No cache
rebuild is required.

Train M and MC from the same seed and identical hyperparameters.  Because the
model is tiny, one GPU is sufficient.

After training, evaluate each `best.pt` with
`eval_p0_f9_v15_motion_transport.py`.  Compare against the frozen v13 result:
Overall 42.2504 / Moving 22.4781 and Strong/KTA 39.7457 / 21.3872.

Interpretation order:
1. If M improves true-moving ADE and Moving-IoU, the objective mismatch was real.
2. If MC further improves macro Moving-IoU, retain class redistribution.
3. Run the v15 gap wrapper on the better of M/MC.  If the KTA/learned selector
   oracle still has a large gap, the next addition is a causal correction gate.
4. Only if true-moving ADE remains the dominant bottleneck after objective
   alignment should model capacity be increased (e.g. STPN-style temporal
   encoder).
