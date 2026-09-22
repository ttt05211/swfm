# V19-MI: Memory–Transport–Innovation implementation

Date: 2026-09-22

## 1. Scope

V19 is an additive extension of the frozen Clean-E14/V18 source-motion
forecaster. The frozen main result is not overwritten.

The method contract is:

**Remember observed scene -> transport known sources -> generate only
unexplained innovation.**

The implementation is deliberately staged. Expensive learned modules are not
allowed to proceed before the corresponding zero-training/perfect-oracle gate
shows meaningful headroom.

## 2. Implemented modules

### 2.1 Causal source memory

File: real_motion/v19_scene_memory.py

build_dynamic_source_memory() builds six-frame annotation-free Strong
tracklets. Its output order is frozen:

1. t0-current sources in the exact Strong component order;
2. recently missing history tracks appended afterwards.

Each SourceTrack stores class, source-relative observed geometry, world-centre
history, validity, velocity, source age/confidence and provenance.

Dormant tracks use a causal pseudo-anchor

\[
\tilde c_0=c_{last}+v_{last}\Delta t.
\]

prepare_causal_arrays_from_tracks() constructs the existing 46-d V18 source
feature contract, local semantic tube, source mask, frame motion and KTA prior
without requiring a component at the current block anchor.

persistent_tracks_from_v18_predictions() promotes one six-query V18 block
directly into the next block's source memory. The next rollout block therefore
does not need to rediscover known sources by connected-component extraction.

### 2.2 Static world memory

File: real_motion/v19_scene_memory.py

StaticWorldMemory accumulates only Occ3D mask_lidar-observed,
non-motion-capable cells in world coordinates.

Rules:

- observed dynamic classes are ignored;
- observed static occupancy is inserted/updated;
- genuinely observed free cells invalidate stale map occupancy;
- conflict resolution is effectively most-recent valid observation;
- future rendering is deterministic world -> future-ego geometry;
- composition is add-only.

Free semantic labels are never treated as equivalent to unknown observation.

### 2.3 Gated memory adapter

File: real_motion/v19_memory_adapter.py

MemoryAdaptedV18SE2 inherits the frozen V18 architecture but adds only
memory-specific modules:

- status projection;
- memory XY delta;
- memory yaw delta;
- memory survival head.

History status is

\[
s=[is_{observed@anchor}, age_{real}, confidence, is_{predicted-birth}].
\]

The adapter gate is

\[
g=1-is_{observed@anchor}.
\]

Therefore every ordinary current source has g=0. Its residual XY,
existence and yaw outputs are mathematically identical to the loaded Clean-E14
core regardless of learned memory-adapter weights.

freeze_clean_core() makes only the four new memory modules trainable.

### 2.4 Residual innovation head

File: real_motion/v19_innovation.py

The innovation branch works in the target future coordinate frame. Each of the
six historical Occ3D grids and lidar observation masks is explicitly warped to
each future ego frame before the network is called.

Per history frame it receives:

- top semantic label;
- lidar coverage;
- top occupied height;
- bottom occupied height;
- occupied vertical count.

A small shared stride-2 frame stem plus depthwise temporal 3D convolution
produces the future-aligned history representation. The base explained BEV
mask and learned relative-time embedding are then fused before one lightweight
upsampling decoder.

Outputs per future horizon:

- add-presence logit;
- 17-class occupied semantic logits;
- 16-bin vertical occupancy logits.

The final head is conservatively initialized: add-presence bias is -4, so a
fresh innovation branch predicts no additions.

The composer is protected add-only: it cannot delete or overwrite any occupied
base voxel.

## 3. Implemented gates and diagnostics

### 3.1 Innovation decomposition + perfect-add oracle

File:
tools/real_motion/diagnose_p0_f9_v19_innovation_decomposition.py

On frozen Clean-E14 predictions, only GT-occupied voxels for which V18 predicts
free are considered addable. They are partitioned into disjoint categories:

- history_source_recoverable
- future_birth_dynamic
- source_shape_innovation
- history_static_recoverable
- never_seen_static
- other_ambiguous

For each category the script creates a perfect add-only oracle by filling only
that category with the GT semantic label. It reports exact dataset-level
IoU/mIoU/Moving deltas.

Future GT and instance identity are diagnostic-only.

### 3.2 Zero-training 1--3 s memory ablation

File:
tools/real_motion/eval_p0_f9_v19_memory_ablation.py

Variants:

- frozen Clean-E14;
- + dormant source KTA add-only;
- + static world memory add-only;
- + both.

No V19 weight is trained.

### 3.3 Zero-training 3--6 s persistent-memory rollout

Files:

- tools/real_motion/eval_p0_f9_v19_memory_rollout.py
- tools/real_motion/merge_p0_f9_v19_memory_rollout_shards.py

The first 0--3 s block is the exact frozen Clean-E14 path. For 3--6 s,
predicted per-source trajectories are promoted directly into persistent source
memory and the same frozen model is reused. This removes full-scene
component re-detection/identity re-association for known sources.

A second variant adds static world memory.

## 4. Safety / non-interference invariants

The implementation is required to preserve all of the following:

1. Existing Clean-E14 files/checkpoints are not modified.
2. Current-source order remains the frozen Strong t0 source order.
3. Memory-only tracks are appended after current sources.
4. Current-source adapter gate is exactly zero.
5. New trainable memory heads are zero initialized.
6. Static memory uses mask_lidar; free is not used as an unknown proxy.
7. Dynamic observations do not enter or erase static memory.
8. Dormant/static/innovation proposals are add-only.
9. Innovation works in future ego coordinates after explicit deterministic
   ego warping.
10. Future GT is never consumed by a deployable prediction path.

## 5. Development decision gates

Run in this order:

1. innovation decomposition + perfect-add oracle;
2. zero-training dormant/static memory ablation;
3. zero-training persistent source rollout;
4. only if memory headroom is real: train memory adapter + survival;
5. only if future_birth_dynamic + never_seen_static + shape innovation
   perfect-add headroom is meaningful: train the residual innovation head;
6. only after innovation quality is established: promote stable generated
   dynamic births into SourceMemory.

Do not train the full V19-MIR stack before gates 1--3 have been measured.


## 6. Source reconciliation: detection remains authoritative

The first persistent-source rollout ablation showed that replacing second-block
Strong redetection with pure propagated source memory is not the intended final
design. On the 32-window smoke, persistent memory changed average 4--6 s mIoU
only slightly but reduced Moving-Macro/Micro relative to the matched V18
redetection path.

V19 therefore uses source memory as a lifecycle layer, not as a replacement for
the current source state:

    current/generated occupancy
        -> frozen Strong source redetection
        -> reconcile with SourceMemory
             matched detected source -> exact frozen V18 path
             unmatched detected      -> exact frozen V18 path + new identity
             unmatched memory        -> confidence-gated add-only recovery

Implementation:

- real_motion/v19_source_reconciliation.py
- tools/real_motion/eval_p0_f9_v19_memory_rollout.py

The association is deterministic same-class nearest one-to-one matching in
world XY. The frozen detected-source order is never changed.

The memory recovery branch is evaluated separately from the frozen path.
Therefore adding memory-only sources cannot numerically perturb current-source
Clean-E14 predictions through batching or shared composition. Recovery output
is restricted to dynamic semantics and is composed only into free voxels.

Source state now separates two concepts:

- detected_at_anchor: whether the current scene state has a source component;
- age_since_real_observation: elapsed time since a real causal sensor
  observation.

This matters in open-loop rollout: a source can be re-detected from generated
occupancy while its real-observation age continues to grow.

Matched tracks preserve their stable track IDs. Newly detected sources receive
fresh IDs. This provides the lifecycle contract required for future
innovation-to-memory promotion without changing any V18 prediction tensor.

## 7. Current evidence and design status

Completed smoke evidence before the reconciliation implementation:

- zero-training dormant KTA on 64 windows:
  mIoU -0.069 pp, Moving-Micro -0.029 pp;
- zero-training static memory on the same 64 windows:
  IoU +0.545 pp, mIoU +0.195 pp, Moving unchanged;
- on a matched 32-window 3--6 s rollout, replacing redetection with pure
  persistent source memory changed average mIoU from 30.847 to 30.902 but
  Moving-Micro from 10.079 to 8.924;
- adding static memory on top of the persistent path increased average 4--6 s
  mIoU from 30.902 to 34.304 while leaving its Moving metrics unchanged.

Interpretation:

1. Static memory has direct zero-training evidence and remains part of the
   primary V19 hypothesis.
2. Pure persistent dynamic replacement is a negative ablation.
3. Dynamic source memory remains useful for lifecycle/recovery, but only through
   reconciliation with the detected V18 state.
4. Innovation remains justified by large perfect-add static/shape headroom, but
   it is still gated by corrected decomposition diagnostics before training.

The decomposition diagnostic was subsequently corrected to use the frozen
Moving-IoU 0.5 m box margin for dynamic attribution. Dynamic category numbers
from earlier smoke output must not be frozen until this corrected diagnostic is
rerun.
