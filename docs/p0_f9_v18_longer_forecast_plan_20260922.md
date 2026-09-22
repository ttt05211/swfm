# P0-F9 V18 longer-forecast plan — 2026-09-22

This note separates two questions:

1. **Representational headroom:** how much 1--6 s future occupancy can be
   explained by transporting geometry already observed at t0?
2. **Zero-shot long forecasting:** can frozen Clean-E14 be reused beyond its
   trained 0.5--3.0 s output horizon without additional optimization?

The frozen 3 s main checkpoint and main-table protocol are not changed.

## 1. GenieDrive audit

GenieDrive's released long config
`occ_gen/configs/world_model/vae_e2e_long.py` keeps
`train_load_future_frame_number=6` (3 s) but sets
`test_load_future_frame_number=20` (10 s).  Its world model uses a shared
autoregressive transition inside a loop over `predict_future_frame`.  At test
time, the predicted latent and predicted ego state are fed back into the next
step, so extending the loop does not require a new occupancy-world-model
checkpoint.

This is why the paper can report 4/5/6 s forecasting "without additional
training".  The released inference path still consumes future control/ego
conditioning from metadata (including `curr_to_future_ego_rt` and
`gt_ego_fut_cmd`), so the result should not be described as unconditional
future-ego prediction.

## 2. Why Clean-E14 cannot simply change 6 -> 12

V18 is not a one-step recurrent model.  It has exactly six learned future
queries / future-time embeddings and emits the six relative horizons
0.5--3.0 s in parallel.  Therefore changing a constant from six to twelve would
introduce untrained queries and is not a valid no-training long-horizon test.

The valid zero-shot extension is **block-autoregressive reuse**:

    observed history [-2.5, ..., 0.0]
        -> frozen Clean-E14
        -> predicted [0.5, ..., 3.0]

    predicted history [0.5, ..., 3.0]
        -> rebuild the same causal Strong sources / tracks / local tubes
        -> the same frozen Clean-E14
        -> predicted [3.5, ..., 6.0]

The second block uses exactly the trained relative-horizon contract again.  No
weights or objectives change.

Because the model was trained with real occupancy histories rather than its own
generated histories, the second block is a genuine zero-shot distribution-shift
test and may accumulate source merge/split, trajectory and composition errors.
That distinction from GenieDrive must be stated if the result is reported.

## 3. Future-ego protocol

The main V18 protocol already permits GT future ego poses and compares against
future-trajectory-conditioned baselines.  A 6 s zero-shot experiment should
keep that same information contract and provide ego poses through 6 s to both
deterministic transport and any matched baselines.  Do not silently switch to a
predicted-ego or history-only protocol.

## 4. First diagnostic: observed-geometry transportability

Before implementing block rollout, run:

`tools/real_motion/diagnose_p0_f9_long_horizon_transportability.py`

It uses one common population requiring a full 6-history + 12-future sequence
and reports 1/2/3/4/5/6 s:

- occupied-voxel recall of an offline observed-geometry transport oracle;
- unexplained occupied fraction;
- static transport recall;
- dynamic transport recall;
- semantic mIoU of the transport oracle;
- dynamic GT occupancy lying inside boxes of:
  - t0 instances represented by an extracted source;
  - t0 instances missed by source extraction;
  - future dynamic instances absent at t0.

The oracle uses future GT motion only for analysis.  "Unexplained" is broader
than "new object": it includes truly new content, newly revealed geometry,
source-extraction misses and rigid-shape mismatch.

## 5. Decision after the diagnostic

- If transport occupied recall remains high at 4--6 s and the birth/new-area
  fraction remains small, proceed with frozen Clean-E14 block rollout.
- If total recall stays high but dynamic recall drops, the long-horizon problem
  is mainly source motion / source persistence; block rollout is still worth
  testing.
- If static recall drops sharply, scene reveal / new-area completion is a
  substantial limitation of the current representation.
- If birth/unrepresented dynamic occupancy grows sharply, a transport-only
  model has a structural long-horizon ceiling; do not add a completion branch
  before quantifying whether the 4--6 s result is still competitive.

## 6. Longer-forecast reporting

If zero-shot block rollout is implemented, use one complete-window population
and report exactly 4 s, 5 s and 6 s IoU/mIoU to match the useful GenieDrive
comparison.  Also retain Moving-Micro / Moving-Macro at those horizons when
annotation support is available, because they directly test the paper's motion
claim.

The longer-forecast table should be labeled **zero-shot block rollout** and
must not replace the frozen 1--3 s main table.
