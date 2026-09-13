# P0-F9 / V17 Selective Forecasting Feasibility

## Goal

This phase is a **small decision experiment**, not another model-development line.

Paper hypothesis:

> **Semantic dynamics != actual motion != need for learned correction.**

Frozen experts:

- physics expert: Strong-W2Det / KTA;
- learned expert: V17-RL epoch-5 with the A1 source-order compositor.

The only new trainable module is a tiny source-level selector. V17, KTA, the
Moving-mIoU-v2 metric and the compositor remain frozen.

The experiment answers two questions:

1. Is the benefit of V17 over KTA concentrated in a small fraction of sources?
2. Can that useful subset be ranked from causal history better than random or
   a trivial current-speed rule?

A negative answer stops the selector story. No router/module sweep follows.

---

## Why historical Need-Score results are not reused as formal evidence

Historical router/Need-Score experiments used different experts, different
targets or different routing granularity. The current `main` branch does not
contain a directly reusable artifact named Need-Score with enough provenance to
certify all of:

- KTA vs **V17-RL epoch-5 + A1**;
- final Moving-mIoU-v2 as the utility;
- source-level decisions;
- the current V17 cache/Strong source ordering.

Therefore this phase recomputes the supervision against the final frozen
experts. Old router results remain motivation/negative history only.

---

## Exact selector supervision

For every Strong source `i`, the builder first evaluates the pure KTA anchor on
the complete chosen dataset and stores the Moving-mIoU-v2 class/horizon
intersection and union counts.

Then, for source `i` only, it applies the exact one-source A1 operation:

```text
KTA anchor
  -> CLEAR this source's KTA landing mask where anchor semantics are dynamic
  -> WRITE this source's V17 transported source shape
```

No other source is changed.

Only voxels in the union of the KTA and V17 component masks can change, so the
builder computes the exact sparse change in the Moving-mIoU counts rather than
copying a full 200x200x16 scene for every source.

The source target is:

```text
u_i =
Moving-mIoU(dataset with source i switched KTA -> V17)
-
Moving-mIoU(pure KTA dataset)
```

This is an exact **single-source marginal** for the frozen final metric. It is
not ADE/FDE and does not require GT instance matching.

Important: ranking sources by these marginals is a GT-assisted **marginal
oracle ranking**, not a claim of solving the combinatorial optimal subset. The
actual Q-budget curve is always evaluated by composing the selected sources
together and recomputing the final occupancy metrics.

GT future occupancy and Moving support are used only to generate training/oracle
labels. Selector inputs contain only the frozen 46-D causal V17
`motion_transport.FEATURE_NAMES`.

---

## Train/eval data separation

Quick feasibility protocol:

```text
selector-label pool:
    P0-F9 train sample IDs
    intersect V17 train-full nativefp
    fixed random 1024 windows by seed 20260913

selector training:
    scene-disjoint 90/10 split inside those train windows

final quick screen:
    independent nuScenes val-128 V17/P0-F9 cache
```

The evaluator fails closed if any selector train/internal-validation scene
overlaps an evaluation scene.

P0-F9 train caches commonly do not contain compact evaluation occupancy. The
label builder therefore reconstructs the required report-horizon targets from
raw Occ3D:

```text
future GT               <- raw Occ3D labels
Strong/KTA anchor       <- frozen w2det_predict from t-1/t0
Moving support          <- frozen Moving-mIoU-v2 GT annotation protocol
```

Only the 1s / 2s / 3s report frames are reconstructed.

When a P0-F9 validation cache contains the compact eval payload, the first
`--audit-raw-payload-windows` windows are reconstructed from raw data and must
match the cached GT / Strong anchor / Moving support exactly at the report
horizons.

---

## Selector

Architecture:

```text
46 causal source features
 -> Linear 64
 -> GELU + LayerNorm
 -> Linear 32
 -> GELU
 -> scalar utility score
```

Training target is standardized exact marginal utility. The model uses
Smooth-L1 regression, while checkpoint selection uses scene-disjoint validation
Spearman because deployment only needs a ranking.

No occupancy crop, latent feature, GT motion label, future box, V17 output or
future GT is a selector input.

---

## Equal-budget evaluation

Primary budgets:

```text
Q = 0 / 10 / 20 / 40 / 100 %
```

Selection is per-window so each policy receives the same source-count budget.

Policies:

```text
oracle   = GT-assisted exact single-source marginal ranking
selector = causal tiny MLP
speed    = current source speed
random   = five deterministic random rankings
```

Interpretation:

```text
Q=0   = pure Strong/KTA
Q=100 = dense V17-RL + A1
```

The evaluator can take a separately generated frozen V17 eval JSON. It then
requires Q=0 to reproduce `strong_anchor` and Q=100 to reproduce
`local_stwm_center_always_source_order` to within `1e-6`; otherwise evaluation
aborts.

---

## Frozen feasibility decision

`Q=20%` is the primary operating point. The full budget curve is explanatory.

The rules are fixed before results:

```text
Oracle feasibility:
    Oracle@20 Moving
    >= max(KTA, Dense-V17) + 0.30 pp

Learned selector:
    Selector@20 >= Speed@20 + 0.20 pp
    Selector@20 >= Random@20(mean) + 0.20 pp
    selector retains >= 50% of Oracle@20 gain over KTA
    Selector@20 mIoU - KTA mIoU >= -0.10 pp

Efficiency:
    incremental learned-stage latency at Q20
    reduced >= 50% relative to router-free dense V17
```

Possible final decisions:

```text
GO_SELECTIVE_FORECAST_MAINLINE

ORACLE_ONLY_STOP_AS_MAIN_CONTRIBUTION
    sparse GT headroom exists, but causal selector/efficiency is insufficient

STOP_SELECTOR_STORY
    even the GT-assisted Q20 marginal oracle does not establish sparse headroom
```

A GO on val-128 is only permission to freeze the method and run one clean,
larger/independent final experiment. It is not itself the paper's final result.

---

## Latency scope

The profiler measures:

```text
selector + routed V17 forward
```

including the routed tensor transfer used by the probe.

It deliberately excludes common operations:

```text
Strong/W2Det + KTA
raw occupancy IO
rigid rasterization
final composition
```

Therefore it is reported as **incremental learned-stage latency**, not
end-to-end FPS. A paper-level end-to-end latency table should be measured only
after the feasibility phase passes and the complete method is frozen.
