# KTA / V17 selective forecasting — small feasibility protocol

This experiment exists to make one paper-level decision and then stop expanding
routing diagnostics.

## Paper hypothesis

**Semantic dynamic class != actual physical motion != need for learned
forecasting correction.**

The frozen expert hierarchy is:

1. physically static / ego-compensated content: deterministic transport;
2. simple moving sources: KTA constant-velocity transport;
3. sources for which KTA is insufficient: frozen V17-RL correction.

The feasibility question is narrower than building the final method:

> Is the benefit of V17 over KTA concentrated in a small set of sources, and can
> a cheap causal selector predict those sources better than random/speed rules?

## Audit of the earlier MSP / Need-style routing work

The committed MSP probe (`real_motion/msp.py`, `p0_msp_build_dataset.py`,
`p0_msp_train_probe.py`) solves a materially different task:

- **unit:** causal moving/dormant occupancy components, then rasterized latent
  cells / sparse windows;
- **label:** GT future *motion activation and future location*;
- **input:** current component position, KTA velocity/speed, size, class and
  moving/dormant state;
- **objective:** activation BCE + future-location likelihood;
- **evaluation:** GT-filled support/window oracle, i.e. whether the proposed
  support reaches future motion, not whether an actual learned expert should
  replace KTA;
- **failure mode relevant here:** good motion-support evidence does not imply
  positive end-to-end repair utility. Historical Need-Score experiments also
  showed very low retained headroom, so the new selector is not assumed to work.

The present experiment therefore changes the target rather than renaming the
old router:

- **unit:** the exact Strong source order used by the final V17 A1 compositor;
- **experts:** final KTA vs frozen `V17-RL epoch5 + A1`;
- **GT label:** one-source counterfactual change in correctly classified voxels
  on the final GT Moving support when KTA is replaced by V17;
- **selector input:** only causal V17/KTA history scalars, explicitly excluding
  V17 output and all future GT;
- **evaluation:** exact final occupancy composition and Moving-mIoU for every
  budget point.

The GT utility ranking is called **GT-assisted**, not an exact combinatorial
oracle: source interactions can make utilities non-additive. Each curve point,
however, is composed and evaluated exactly under A1.

## Frozen feasibility sequence

### A. GT-assisted expert budget curve — no selector training

Use the final V17 checkpoint and final A1 compositor. Evaluate per-window source
budgets:

`Q = 0%, 10%, 20%, 40%, 100%`

- `Q=0`: all KTA;
- `Q=100`: all V17;
- intermediate Q: sources ranked by GT one-source V17-vs-KTA utility.

Also report equal-budget speed and deterministic-random ranking. This establishes
whether useful correction is sparse at all.

### B. Tiny causal selector — one training job

Build labels from TRAIN scenes only. The selector input is:

- frozen V17 causal motion feature vector;
- six-frame causal frame-motion scalars;
- six KTA future displacements.

No V17 output, future annotation, future occupancy or oracle score appears in an
input feature. A two-layer MLP regresses GT utility. Checkpoint selection uses an
internal scene-disjoint split of TRAIN scenes; val-128 is diagnostic only.

### C. Equal-budget final comparison

For `Q = 0/10/20/40/100%`, compare:

- GT-assisted utility rank;
- learned selector;
- speed rule;
- deterministic random rule.

Report final IoU, mIoU, Moving-mIoU and 1s/2s/3s Moving. Optional latency mode
measures the selector forward plus the V17 forward on selected sources only;
accuracy always uses one common full V17 forward so strategy comparisons are
numerically paired.

## Decision rule

Do not continue routing just because the GT-assisted curve is attractive.
Two separate facts are required for a routing mainline:

1. **sparse headroom exists:** a small Q obtains a meaningful part of the best
   KTA/V17 mixture benefit, or approximately preserves dense V17 Moving while
   invoking V17 on far fewer sources;
2. **the benefit is causally predictable:** at equal Q the learned selector is
   consistently better than random and the speed rule, with a real V17-forward
   latency reduction.

If (1) fails, stop immediately: there is no sparse allocation structure to
learn. If (1) passes but (2) fails, record the oracle observation but do not use
learned routing as a main paper contribution.

`val-128` is only a rapid feasibility screen. Any accepted paper claim must be
rerun on a frozen independent/full validation protocol with the same selector
checkpoint and no retuning.

## D. Final loss-alignment probe: utility-weighted ranking

The regression selector passed the basic causal-predictability gate but showed a
specific failure mode on val-128: it captured most positive utility mass while
also routing a large fraction of harmful sources. A historical KTA-error probe
did not separate positive from negative utility, so no new feature family is
introduced.

The one allowed follow-up changes **only the selector objective**, not its
architecture or causal inputs.

For each window, define:

- positive sources: one-source counterfactual utility u_i > 0;
- zero sources: u_i = 0;
- harmful sources: u_i < 0.

The ranking objective enforces positive > zero > harmful, with the strongest
utility-weighted pairwise term on positive-vs-harmful pairs. Positive-vs-zero
and zero-vs-harmful pairs receive a fixed weight of 0.25. The tiny 96-dim MLP,
optimizer scale, causal feature contract and 2000-step budget remain unchanged.

Checkpoint selection is also aligned to deployment: maximize **internal-dev Q20
net selected utility**, not regression loss. Val-128 remains diagnostic only
during training, and final acceptance is still decided by exact A1-composed
Moving-mIoU at the frozen Q=0/10/20/40/100% budgets.

Ranking logits are intentionally *not* calibrated utility values. Therefore
the earlier predicted-utility-positive abstention rule is invalid for this
model and is disabled by the evaluator.

This is the final selector-training variant. If it does not materially reduce
harmful-source capture and improve exact Moving-mIoU over the regression
selector, stop selector optimization rather than adding a larger router,
additional feature families or reinforcement learning.
