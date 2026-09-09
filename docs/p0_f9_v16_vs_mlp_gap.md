# P0-F9 V16-STWM vs v13 MLP geometric gap diagnostic

This is a **diagnostic only**. It performs no training and belongs exclusively to the **V16-STWM / main** route. It does not read or modify `feature/motion_transport_v1-spec2` or MT-V1-STPN artifacts.

## Question

V16-STWM lowers center-trajectory ADE/FDE relative to the successful v13 MLP, but its Moving-mIoU is worse. The diagnostic determines whether the discrepancy comes from:

1. cross-track versus along-track geometry;
2. large/voxel-heavy sources being worse despite lower micro ADE;
3. specific horizons/classes/source sizes;
4. complementary motion regimes where KTA, MLP, and STWM each win on different source/horizon observations.

## Frozen contracts

- Strong causal source decomposition and ordering are unchanged.
- The v13 target remains `GT displacement - KTA displacement`.
- Both learned models predict the same six `(dx,dy)` KTA residuals.
- The same rigid source-shape transport and CLEAR/WRITE composition are used.
- True-moving observations use the exact Moving-mIoU-v2 speed threshold (`0.5 m/s`).
- No existence gate is used in the standalone MLP/STWM occupancy variants, so the comparison isolates center motion.

## Outputs

`diagnose_p0_f9_v16_vs_mlp_gap.py` reports:

- ordinary and source-voxel-weighted ADE;
- absolute along-track and cross-track error relative to the GT motion direction;
- true-moving FDE;
- 1s/2s/3s, class, and source-size decompositions;
- GT-only center-error selector oracles for `KTA/MLP`, `KTA/STWM`, `MLP/STWM`, and `KTA/MLP/STWM`;
- the unchanged GT-center rigid upper bound.

Selector oracles are **diagnostic upper bounds**, not deployable routing results. They use future GT only to reveal whether the predictors are complementary.
