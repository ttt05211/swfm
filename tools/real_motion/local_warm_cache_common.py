"""Explicit causal disk-cache prefill; no GT, optimizer or learned-state cache."""
from dataclasses import asdict
from pathlib import Path
import time
import torch
from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.v21_source_induction import stable_json_fingerprint


def geometry_namespace(cfg, provider, info_fingerprints, cache_fingerprints, dataroot):
    # EXACT legacy full15 namespace: existing cache files remain reusable.
    identity = dict(runtime_config=cfg, strong=asdict(provider.strong),
        columns=asdict(provider.joint.columns.config), info=info_fingerprints,
        caches=cache_fingerprints, dataroot=str(Path(dataroot).resolve()))
    history = provider.joint.transport.config.history_frames
    if history != 6:
        from real_motion.local_history_contract import HISTORY4_CONTRACT
        identity.update(active_history_frames=history, observation_contract=HISTORY4_CONTRACT)
    return stable_json_fingerprint(identity)


def warm_causal_cache(provider, source, records, *, progress=None, stop_event=None):
    cache = getattr(provider, 'causal_geometry_cache', None)
    if cache is None: raise ValueError('explicit geometry cache required for prewarm')
    started = time.perf_counter(); before = cache.stats(); pending = []; count = 0
    joint = provider.joint; training = joint.training
    devices = [provider.device.index if provider.device.index is not None else torch.cuda.current_device()] if provider.device.type == 'cuda' else []
    def drain():
        cache.flush()
        for key, raw, value in pending:
            if not cache.is_persisted(key, raw):
                # Explicit prefill may wait; normal training never waits for a
                # full asynchronous write queue. Budget failures remain fatal.
                cache.store(key, raw, value)
            if not cache.is_persisted(key, raw):
                raise RuntimeError('warm cache incomplete: disk quota/free-space prevented persistence; '
                    'existing artifacts preserved, do not report this as a warm run')
        pending.clear()
    try:
        joint.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for record, raw in prefetch_raw_columns(provider, source, records, include_gt=False):
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('cache prefill stopped safely')
                if raw.get('future_gt_occ') is not None: raise RuntimeError('cache prefill must not load future GT')
                prep = provider.prepare_columns(source, record, include_gt=False, raw_window=raw)
                value = {**raw['_column_causal_preparation'],
                    'prepared_state': {k: v for k, v in prep.state.items() if k not in ('rec', 'window', 'gpu')}}
                pending.append(((str(record['scene_name']), str(record['t0_token'])), raw, value)); count += 1
                if count % 4 == 0: drain()
                if progress and (count == 1 or count % 16 == 0):
                    progress({'event': 'warm_causal_cache', 'windows': count,
                        'seconds': time.perf_counter()-started, 'cache': cache.stats()})
            drain()
    finally:
        cache.flush(); joint.train(training)
    after = cache.stats()
    return {'complete': True, 'windows': count, 'seconds': time.perf_counter()-started,
        'new_writes': after['writes']-before['writes'], 'initial_hits': after['hits']-before['hits'],
        'cache': after, 'future_GT_loaded': False, 'optimizer_steps': 0,
        'contract': 'original_device_Strong_fixed_causal_evidence_persisted_all_requested_keys'}
