"""Isolated history-registration velocity study; NOT a verification default.

The already-handed-off verify_all/replay fingerprints stay unchanged. Common
anchor positions are used only as measurements of velocity, never substituted
for the original source identities, registration-validity masks or read points.
"""
from dataclasses import replace
import numpy as np
from .replay import multiframe_velocity


def registered_velocity(row,provider=None,times=None,filter_center=False):
    rec,raw,ego,history=row;track=raw.get('_shared_source_track')
    if track is None:
        from real_motion.shared_source_track import from_raw
        if provider is None or times is None:raise ValueError('history provider/times required')
        _,track=from_raw(raw,provider,times,current=history[0]['state']['current'])
    measured=track.centers.copy();valid=track.track_valid.copy();used=0
    for i in track.pair_velocity:
        accepted=np.asarray([r is not None for r in track.registrations[i]]) & valid[i]
        # Do not infer a velocity for an unmatched last pair or bridge rejected
        # t-1 registration from older frames. Fallback is the original raw LS.
        if not accepted[2] or accepted.sum()<3:continue
        anchor=np.r_[track.centers[i,3],1.]
        for f in np.flatnonzero(accepted):
            measured[i,f]=np.linalg.solve(track.registrations[i][f][0],anchor)[:3]
        valid[i]=accepted;used+=1
    temporary=replace(track,centers=measured,track_valid=valid)
    temporary_row=(rec,dict(raw,_shared_source_track=temporary),ego,history)
    changed,audit=multiframe_velocity(temporary_row,provider,times,filter_center)
    fitted=changed[1]['_shared_source_track']
    restored=replace(track,pair_velocity=fitted.pair_velocity)
    changed=(changed[0],dict(changed[1],_shared_source_track=restored),changed[2],changed[3])
    audit.update(velocity_model='registered_half',registered_velocity_sources=used,
        source_track_geometry_and_validity_restored=True)
    return changed,audit
