"""Rebuild derived track geometry in bounded RAM, never alter legacy caches."""
import numpy as np
from real_motion.ego_navigation import ego_history_features
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.unified_source_tracks.geometry import build_tracks, track_velocity, make_row

ARMS = ('baseline', 'unified_fit')


def ego_inputs(source, record, raw):
    tokens = tuple(record['history_tokens'])[-4:]
    times = np.asarray([source.nusc.get('sample', str(t))['timestamp']/1e6 for t in tokens])
    raw['_track_history_seconds'] = times
    return ego_history_features(raw['history_poses'], times), None, None


def track_from_cached(raw, provider):
    # Legacy compact artifacts keep t0 sources/points but omit previous and
    # velocity. Reconstruct ONLY those history facts, not learned or GT fields.
    causal = raw['_column_causal_preparation']; state = dict(causal['prepared_state'])
    previous = runtime.extract_instances_cropped_exact(raw['history_occ'][-2], raw['history_poses'][-2],
        grid=provider.pcfg.grid, cfg=provider.strong)
    state['previous'] = previous
    state['velocities'] = runtime.match_instances(previous, state['current'], .5,
        max_speed_mps=provider.strong.max_match_speed_mps)
    historical = {k: raw[k] for k in ('history_occ', 'history_observed', 'history_poses')}
    historical['_column_causal_preparation'] = dict(prepared_state=state)
    return build_tracks((None, historical, None, None), provider, raw['_track_history_seconds'])


def history_inputs(arm):
    if arm not in ARMS: raise ValueError('unknown track arm')
    class TrackHistoryInputs(ResidualHistoryInputs):
        def build(self, raw):
            if arm == 'baseline': return super().build(raw)
            track = track_from_cached(raw, self.provider)
            fixed = raw['_column_causal_preparation']
            changed = dict(raw, _column_causal_preparation=dict(fixed, registrations=track.registrations))
            history = super().build(changed)
            history[0]['track_fit'] = dict(velocity_world=track_velocity(track, 'fit'),
                valid_point_counts=np.bincount(track.valid.sum(1), minlength=5), association=track.audit)
            return history
    return TrackHistoryInputs


def publish_row(row, arm):
    """Keep history reader, encoder raw and KTA record consistent on cache HIT too."""
    if arm == 'baseline': return row
    record, raw, ego, history = row
    fitted = history[0]['track_fit']; velocity = np.asarray(fitted['velocity_world'])
    current = raw['_column_causal_preparation']['prepared_state']['current_pose']
    if velocity.shape != (len(record['source_class_id']), 3): raise RuntimeError('track/source order differs')
    kta = ((velocity @ np.asarray(current)[:3, :3])[:, None, :2]*
           (np.arange(1, 7)[None, :, None]*.5)).astype(np.float32)
    rec = dict(record, kta_displacement_xy_m=kta,
        anchors_xy_t0_m=(np.asarray(record['source_centroid_xy_t0_m'])[:, None]+kta).astype(np.float32))
    fixed = raw['_column_causal_preparation']
    changed = dict(raw, _column_causal_preparation=dict(fixed, registrations=history[0]['registrations']))
    # ALL original future labels/poses remain supervised targets/GT condition;
    # track construction above only receives historical fields.
    return rec, changed, ego, history


def prefetch_adapter(base, arm):
    def batches(*args, **kwargs):
        iterator = base(*args, **kwargs)
        try:
            for rows in iterator: yield [publish_row(r, arm) for r in rows]
        finally:
            close = getattr(iterator, 'close', None)
            if close: close()
    return batches


def live_variant(row, provider, reader, times, arm):
    """Evaluation uses the SAME fit rule, but rebuilds from actual OCC/STC input."""
    if arm == 'baseline': return row
    track = build_tracks(row, provider, times)
    return make_row(row, track, track_velocity(track, 'fit'), reader)
