# P0-F9 V17-RL H0 horizon-gradient attribution

This is a no-training diagnostic performed after the native-footprint B-S arm
failed to improve fixed-time occupancy metrics.  It does not enable horizon
weighting and does not add scene loss.

## Question

Does the frozen V17-RL epoch-5 residual objective systematically under-drive
later future horizons?  If so, is the imbalance explained by:

- fewer valid labels;
- the per-label SmoothL1 error distribution;
- the transport-overlap term;
- or cancellation/amplification after the six future queries pass through the
  shared residual head?

## Frozen checkpoint and data

H0 hard-requires the historical V17-RL epoch-5 checkpoint with
`overlap_weight=0.25` and the existing V17 validation cache.  No NuScenes future
occupancy or annotation access is required.

The exact motion objective being decomposed is:

```text
L_motion = L_position + 0.25 * L_overlap
```

`L_position` keeps the historical global micro reduction over every valid XY
coordinate. `L_overlap` keeps the historical global micro reduction over every
eligible source/horizon footprint pair.  Each horizon contribution uses the same
global denominator as the original loss, so all six contributions sum back to
the frozen objective.  This deliberately preserves any label-count imbalance.

Existence BCE is outside H0 because it has no direct computational path to
`residual_xy_m` or `residual_head`.  H0 is specifically a diagnostic for deciding
whether the residual motion loss needs horizon normalization.

## Reported attribution

For each 0.5/1.0/1.5/2.0/2.5/3.0 s horizon the script reports:

- valid, true-moving, and overlap-eligible label counts;
- within-horizon SmoothL1 mean and true-moving ADE;
- transport Soft-IoU;
- exact share of the frozen position, overlap, and combined motion losses;
- direct residual-output gradient RMS and squared-energy share;
- residual-head gradient L2 norm;
- residual-head norm share;
- cosine to the total shared-head update;
- signed projection share onto the total shared-head update.

Direct output slots are disjoint across horizons, so their squared gradient
energy shares have an exact interpretation and sum to one.  The residual head is
shared, so parameter-gradient norms do not add linearly.  For that reason H0 also
reports

```text
projection_share_h = dot(g_h, g_total) / ||g_total||^2
```

whose six values sum to one and may be negative if a horizon opposes the final
shared-head update.

## Decision discipline

H0 is diagnostic only.  Do not infer that later horizons should be reweighted
merely because their raw label counts are smaller.  A horizon-normalized
SmoothL1 experiment is justified only if the *actual combined residual gradient*
shows a meaningful systematic short-to-long imbalance.  If gradients are already
reasonably balanced, skip horizon weighting and move to the later scene-level
supervision question instead.
