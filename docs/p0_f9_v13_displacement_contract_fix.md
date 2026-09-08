# P0-F9 v13: displacement-preserving learned motion transport

## Why v12 is discarded

v12 predicted residuals from a Strong visible-component centroid to an absolute
future annotation-box center.  That silently erased the offset between the
visible occupancy component and the true object center.  As a result ADE/FDE
could improve while rigid occupancy transport became worse.

v13 fixes the target to

`residual = (GT_future_center - GT_t0_center) - KTA_displacement`.

Deployment applies `KTA_displacement + residual` to the observed Strong source
component.  The component is never forced to align its centroid to a GT box
center.  The GT-center oracle likewise translates the source component by the GT
annotation displacement, matching the successful v11 oracle definition.

## Reuse instead of rebuilding

The expensive v12 six-frame Strong source tracks/features are valid and should
be reused.  `upgrade_p0_f9_motion_transport_cache_v2.py` reads only the t0
annotation center for each already-supervised source and rewrites target tensors.
It does not rerun Strong history decomposition or rebuild features.

The resulting cache version is `p0_f9_motion_transport_v2`; the v13 trainer and
evaluator reject v1 caches and v12 checkpoints.

A clean from-scratch builder is also provided as
`build_p0_f9_motion_transport_cache_v2.py` for future reproduction.

## Required validation invariant

Before interpreting a learned result, the v13 evaluator's `gt_center_rigid`
must recover the v11 translation-only oracle (approximately Overall 48.91,
Moving 45.27 on the frozen 128-window protocol).  If it does not, stop: the
transport contract is still inconsistent.

## Artifact policy

Keep:
- full/val MSP window caches;
- P0-F9 frozen 128-window evaluation cache;
- raw nuScenes/Occ3D data;
- v11 rigid-transport oracle report;
- v12 motion caches only until their v2 upgrades have been written and checked.

Discard after successful v2 cache upgrade:
- v12 learned-motion checkpoint directory/log;
- v12 learned-motion deployment JSON/log;
- v1 motion train/val `.pt`, `.summary.json`, and `.resolved.yaml` caches.

Do not reuse the v12 `best.pt`: its learned residual has the wrong geometric
meaning even though the architecture is identical.
