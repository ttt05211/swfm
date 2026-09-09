# P0-F9 V17-RL: scoped A1 + B0 validation

This change intentionally implements only the first no-training validation step.
It does **not** migrate the full MT-V1-STPN compositor, run a multi-checkpoint
proxy matrix, or add scene CE.

## A1: WRITE-order-only compositor check

`eval_p0_f9_v17_local_stwm.py` keeps the historical
`local_stwm_center_always` branch unchanged and adds:

- `local_stwm_center_always_source_order`

Both branches use the same Strong anchor, CLEAR masks, source set, predicted
centers and rasterized replacement components.  A1 changes only replacement
WRITE ordering: from the historical `(-source_voxel_count, class_id)` ordering to
the original Strong source order.

A difference is therefore an ordering-policy effect, not evidence that the old
implementation was a bug.

## B0: single-checkpoint native-footprint audit

`diagnose_p0_f9_v17_native_footprint.py` uses one V17-RL checkpoint and the
frozen V17 validation cache.  It reconstructs exact Strong t0 source voxels from
history occupancy and compares the exact source XY support with the legacy V17
mask in common 0.8 m source-centered coordinates.

Reported summaries include:

- overall / per-class / source-size-tertile support IoU, precision and recall;
- exact native-footprint crop coverage and source-centroid alignment checks;
- per-horizon target-valid / true-moving / overlap-eligible counts;
- legacy-vs-native overlap Soft-IoU and residual-output gradient RMS/nonzero
  fraction at every 0.5 s future step.

For the loss comparison both masks use exactly the same eligibility set:
`target_valid AND legacy_t0_mask_present`.  This prevents B0 from silently
increasing supervision coverage while changing footprint geometry.

B0 does not read future occupancy and is not a checkpoint-selection experiment.
Its purpose is only to verify that the native footprint is a real, correctly
aligned supervision change before paired short training.
