"""Opt-in training clocks and bounded CPU call profiles; no method changes."""
import cProfile
import io
import math
import pstats
from threading import Lock
import time
import torch


class StageTimer:
    def __init__(self, device, enabled=False):
        self.device, self.enabled = device, enabled
        self.host, self.events = {}, []

    def call(self, name, fn, *args, gpu=False, **kwargs):
        if not self.enabled: return fn(*args, **kwargs)
        events = None
        if gpu and self.device.type == 'cuda':
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        tick = time.perf_counter()
        try: return fn(*args, **kwargs)
        finally:
            self.host[name] = self.host.get(name, 0.)+time.perf_counter()-tick
            if events:
                events[1].record(); self.events.append((name, events))

    def finish(self):
        if not self.enabled: return {}
        tick = time.perf_counter()
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        self.host['gpu_finish_wait'] = time.perf_counter()-tick
        gpu = {}
        for name, (a, b) in self.events: gpu[name] = gpu.get(name, 0.)+a.elapsed_time(b)/1000
        return {'host_stage_seconds': dict(self.host), 'cuda_stream_stage_seconds': gpu}


class CpuProfiles:
    """Private profile per CPU job; never share cProfile across threads."""
    # Python 3.14 cProfile uses a process-wide monitoring-tool slot. Serialize
    # ONLY this separate diagnostic pass; throughput trials never use profiles.
    profile_lock = Lock()
    def __init__(self): self.rows = {}; self.lock = Lock()

    def run(self, name, fn, *args, **kwargs):
        p = cProfile.Profile()
        try:
            with self.profile_lock: return p.runcall(fn, *args, **kwargs)
        finally:
            with self.lock: self.rows.setdefault(name, []).append(p)

    def text(self, limit=35):
        output = io.StringIO()
        for name, profiles in self.rows.items():
            output.write('\n===== '+name+' (CPU cumulative, overlapping worker sums) =====\n')
            stats = pstats.Stats(profiles[0], stream=output)
            for p in profiles[1:]: stats.add(p)
            stats.strip_dirs().sort_stats('cumulative').print_stats(limit)
        return output.getvalue()


def trial_summary(rows):
    if not rows: raise ValueError('empty measured trial')
    seconds = sum(float(r['wall_seconds']) for r in rows)
    windows = sum(int(r['windows']) for r in rows)
    if seconds <= 0 or not windows: raise ValueError('invalid benchmark counts')
    totals = {}
    for r in rows:
        for k, v in r.get('host_stage_seconds', {}).items(): totals[k] = totals.get(k, 0.)+v
        # These are serial WAIT/selection times, not worker CPU sums.
        for k in ('input_wait_seconds', 'online_candidate_wait_seconds', 'online_selection_seconds', 'online_feature_wait_seconds'):
            totals[k] = totals.get(k, 0.)+r.get(k, 0.)
    return dict(windows=windows, batches=len(rows), wall_seconds=seconds,
        seconds_per_window=seconds/windows, windows_per_second=windows/seconds,
        mean_windows_per_batch=windows/len(rows), sources=sum(r['sources'] for r in rows),
        peak_allocated_mib=max(r.get('peak_memory_mib') or 0. for r in rows),
        peak_reserved_mib=max(r.get('peak_reserved_mib') or 0. for r in rows),
        stage_seconds=totals,
        cuda_stream_seconds={k: sum(r.get('cuda_stream_stage_seconds', {}).get(k, 0.) for r in rows)
            for k in sorted({k for r in rows for k in r.get('cuda_stream_stage_seconds', {})})},
        cache_hits=sum(r.get('causal_geometry_cache_hits', 0) for r in rows),
        dominant_host_stage=max(totals, key=totals.get) if totals else None,
        epoch_train_hours_if_representative=20430*seconds/windows/3600,
        train15_hours_if_representative=20430*15*seconds/windows/3600)


def recommend_trials(trials, headroom=.10):
    if not 0 <= headroom < 1: raise ValueError('invalid memory reserve')
    safe = [t for t in trials if t.get('status') == 'ok'
        and math.isfinite(t['measurement']['windows_per_second'])
        and t['measurement']['windows_per_second'] > 0
        and t['capacity_peak_reserved_mib'] <= t['available_memory_mib']*(1-headroom)]
    if not safe: return {'recommended': None, 'largest_safe': None, 'route': 'no_safe_batch_do_not_launch_full'}
    fastest = max(t['measurement']['windows_per_second'] for t in safe)
    # Prefer smaller batches when throughput is within 3%: less optimization
    # recipe drift / CPU memory pressure, not a race to maximize allocation.
    best = min((t for t in safe if t['measurement']['windows_per_second'] >= fastest*.97), key=lambda t: t['window_batch'])
    largest = max(safe, key=lambda t: t['window_batch'])
    return {'recommended': {'window_batch': best['window_batch'], 'source_budget': best['source_budget']},
        'largest_safe': {'window_batch': largest['window_batch'], 'source_budget': largest['source_budget']},
        'recommended_trial': best.get('name'), 'sampling_workers': best.get('workers'),
        'route': 'diagnostic_recommendation_not_automatic_training_or_resume', 'memory_headroom_fraction': headroom}
