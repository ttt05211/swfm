# P0-F9 v14 motion-transport gap diagnostic

This is a no-training diagnostic run on the frozen v13 learned displacement model.
Its purpose is to explain why v13 reduces center ADE/FDE strongly while improving
Moving-mIoU by only about +1.09 pp.

## Questions answered

1. **True-moving trajectory error**
   - The source/horizon observation is called true-moving iff its GT annotation
     center interval speed from t0 is at least 0.5 m/s, exactly matching the
     motion decision in Moving-mIoU v2.
   - ADE/FDE, relative reduction and learned-vs-KTA win fraction are reported for
     all valid observations and for true-moving observations separately.
   - True-moving results are further split by report horizon, dynamic class and
     source voxel-count tertile.

2. **KTA / learned selector oracle**
   - This is diagnostic-only and uses GT displacement error.
   - For each source/horizon, KTA is retained unless the learned corrected center
     is strictly closer to the correct displacement.
   - If this oracle is much better than both KTA and always-learned transport,
     the next method should focus on a causal confidence/router rather than simply
     enlarging the motion predictor.

3. **Center-error sensitivity**
   - For matched sources, KTA displacement is interpolated 25/50/75/100% toward
     GT displacement before rigid transport.
   - alpha=0 is Strong/KTA; alpha=1 is the frozen v11-compatible GT-center rigid
     oracle.
   - A strongly nonlinear Moving-mIoU curve means the current ~0.68 m learned
     center error may still be too large for voxel IoU even though ADE improves a
     lot; this would support a stronger temporal motion model.

4. **Class/size localization**
   - Reports true-moving trajectory error by the eight frozen Moving-v2 classes.
   - Reports small/medium/large source buckets using tertiles of voxel count among
     true-moving source/horizon observations.
   - Also reports mean per-class Moving-IoU across 1/2/3 s for Strong, learned,
     selector oracle and GT-center oracle.

## Decision contract

After convergence has already been checked, interpret in this order:

- If true-moving ADE reduction is much weaker than all-source reduction, use
  true-motion-aware weighting/sampling before increasing model capacity.
- If selector-oracle Moving is substantially above always-learned Moving, learn a
  causal correction-confidence/router so easy KTA cases remain untouched.
- If true-moving ADE reduction is strong but the alpha sensitivity curve rises
  sharply only near GT, the dominant problem is center precision; then test a
  stronger temporal predictor (e.g. STPN-style history encoder).
- If one class or object-size bucket dominates residual error, target that failure
  mode rather than adding a generic module.

No VAE, OccFM or new training is involved.
