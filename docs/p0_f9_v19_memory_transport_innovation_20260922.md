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
