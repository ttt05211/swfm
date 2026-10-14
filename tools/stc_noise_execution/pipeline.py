"""Bounded CPU lookahead; CUDA and all optimizer work stay on the consumer."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from contextlib import contextmanager, nullcontext
from queue import Queue, Full
from threading import Thread, Event
from pathlib import Path
import hashlib
import time
import numpy as np
from tools.stc_noise_adaptation import data, augmentation
from tools.scene_ego_head import integration
from real_motion.strong_majority_execution import strong_majority_execution

PROTOCOL = 'empirical_CPU_lookahead_reuse_Strong_history_v1'
EXPECTED_NOISE_SOURCE = 'ce6794bb325d8fecd0e2d484335ce0708b2db5da67c911fde0a878bf4e7d6e10'
_FIELDS = ContextVar('empirical_noise_field_cursor', default=None)


@contextmanager
def noise_execution():
    """Skip one DEAD field's filter/sort, retaining its exact RNG consumption.

    Original wrong-class field is never read (mutually exclusive FN/wrong use
    the FN field). Source hash pins this fact; no mathematical rewrite, new
    probability, reduction, frame sampling or changed subsequent random draw.
    """
    source = Path(augmentation.__file__).read_text(encoding='utf-8')
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_NOISE_SOURCE:
        raise RuntimeError('noise source changed; do not apply the dead-field optimization')
    uniform, corrupt = augmentation.uniform_field, data.corrupt
    def field(shape, sigma, rng):
        cursor = _FIELDS.get()
        if cursor is not None:
            cursor += 1; _FIELDS.set(cursor)
            if cursor == 5:
                rng.standard_normal(shape)  # IDENTICAL draw, discarded downstream in original
                return np.zeros(shape, np.float32)
        return uniform(shape, sigma, rng)
    def exact_corrupt(*args, **kwargs):
        token = _FIELDS.set(0)
        try: return corrupt(*args, **kwargs)
        finally: _FIELDS.reset(token)
    try:
        augmentation.uniform_field, data.corrupt = field, exact_corrupt
        yield
    finally:
        augmentation.uniform_field, data.corrupt = uniform, corrupt


def prefetch_adapter(base, arm, model, seed, *, stop_event=None):
    original = data.prefetch_adapter(base, arm, model, seed)
    def batches(provider, source, groups, descriptors, *, workers=4, depth=2):
        if arm != 'empirical' or source is not provider.ccr_history_cache_source:
            yield from original(provider, source, groups, descriptors, workers=workers, depth=depth)
            return
        if not 1 <= workers <= 8 or not 1 <= depth <= 4: raise ValueError('invalid bounded CPU budget')
        # Two queued groups plus one producer group; <= 3 B4 augmented groups
        # AHEAD of consumer, one global window pool. Never materialize an epoch.
        queue, closing, end = Queue(maxsize=2), Event(), object()
        def stopped(): return closing.is_set() or stop_event is not None and stop_event.is_set()
        def put(packet):
            while not closing.is_set():
                try: queue.put(packet, timeout=.1); return
                except Full: pass
        def produce():
            iterator = base(provider, source, groups, descriptors, workers=workers, depth=depth)
            pool = ThreadPoolExecutor(max_workers=min(workers, 4), thread_name_prefix='empirical-window')
            try:
                for rows in iterator:
                    if stopped(): break
                    dense = [(r, data.policy_raw(raw, 'dense_valid'), inp, hist) for r, raw, inp, hist in rows]
                    futures = [pool.submit(data.augment_row, provider, row, model, seed) for row in dense]
                    put((None, futures))
                put((None, end))
            except BaseException as error:
                put((error, end))
            finally:
                close = getattr(iterator, 'close', None)
                try:
                    if close: close()
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
        producer = Thread(target=produce, name='empirical-prefetch', daemon=False); producer.start()
        try:
            while True:
                tick = time.perf_counter(); error, packet = queue.get()
                if error is not None: raise error
                if packet is end: break
                rows = [f.result() for f in packet]
                wait = time.perf_counter()-tick
                # Report only consumer stall; producer CPU work runs overlapped.
                # Existing augmentation CPU cumulative counters remain separate.
                raw = dict(rows[0][1]); prior = raw.get('_joint_prepare_timing', {})
                raw['_execution_CPU_base_wait'] = prior.get('input_wait', 0.)
                raw['_joint_prepare_timing'] = dict(input_wait=0., descriptors=0., simulation_and_rebuild_wait=wait)
                rows[0] = rows[0][0], raw, rows[0][2], rows[0][3]
                yield rows
        finally:
            closing.set(); producer.join()  # waits bounded CPU work only, NO CUDA worker
    return batches


def finish_backgrounds(provider, row):
    if not row[1].get('_robust_background_pending'): return row
    rec, original, inputs, history = row
    raw = dict(original); causal = dict(raw['_column_causal_preparation']); state = dict(causal['prepared_state'])
    if 'previous' not in state or 'velocities' not in state:
        raise RuntimeError('fresh corrupted history lacks the already prepared instance match')
    if (len(state['current']) != len(rec['source_class_id'])
            or not np.array_equal(state['current_pose'], raw['history_poses'][-1])):
        raise RuntimeError('Strong prepared state/source history identity differs')
    # Same FP64 poses, GPU inverse warp, native majority and CLEAR as reference.
    # Only avoid re-extracting previous CC and re-running its SAME match.
    poses = np.asarray(raw['future_poses'], np.float64)
    majority = getattr(provider, 'strong_majority', None)
    with strong_majority_execution(majority) if majority is not None else nullcontext():
        anchors, components = integration.runtime._strong_all_horizons(raw['history_occ'][-1],
            state['current_pose'], poses, state['current'], state['velocities'], state['source_world_points'],
            frame_dt_s=float(provider.pcfg.frame_dt_s), grid=provider.pcfg.grid, cfg=provider.strong,
            runtime_device=provider.device, majority_backend='native' if majority is not None else 'sparse_cuda')
    state['column_backgrounds'] = [integration.compose_component_replacements_fast_exact(a, c, [],
        dynamic_class_ids=integration.DYNAMIC_CLASS_IDS, free_label=17, grid=provider.pcfg.grid,
        precomputed_clear_flat_indices=integration.runtime.baseline_clear_flat_indices(c, grid=provider.pcfg.grid))
        for a, c in zip(anchors, components)]
    causal['prepared_state'] = state; raw['_column_causal_preparation'] = causal
    raw['_robust_background_pending'] = False
    return rec, raw, inputs, history


class CheckedStrong:
    def __init__(self, record): self.checked = False; self.record = record
    def __call__(self, provider, row):
        fast = finish_backgrounds(provider, row)
        if not self.checked and row[1].get('_robust_background_pending'):
            reference = data.finish_backgrounds(provider, row)
            left = reference[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
            right = fast[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
            differences = [int(np.count_nonzero(a != b)) for a, b in zip(left, right)]
            if len(left) != 6 or len(right) != 6 or any(differences):
                self.record(dict(pass_gate=False, changed_voxels=differences))
                raise RuntimeError('prepared-history Strong reuse is not six-frame byte exact; old checkpoint preserved')
            self.checked = True
            self.record(dict(pass_gate=True, changed_voxels=differences,
                scope='first changed TRAIN window, six integer backgrounds', key=[row[0]['scene_name'], row[0]['t0_token']]))
        return fast
