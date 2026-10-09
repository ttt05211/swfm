"""Read-only surface-CCR expansion, integer accumulation and paired execution."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import numpy as np
import torch

from real_motion.ccr_frozen_b import frozen_b_probabilities
from real_motion.final_dataflow import prepare_history, forecast_six
from real_motion.source_evidence_audit import edit_quality
from real_motion.surface_canonical_repair import SurfaceAtlas, augment_evidence, augment_projection
from real_motion.surface_ccr_execution import SurfaceExecution
from real_motion.surface_projection_execution import map_surface_evidence
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import ccr_screen_common as ccr
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics, delta
from tools.real_motion.point_ccr_v18_fps_common import select_population
from tools.real_motion.validate_p0_f9_ccr_frozen_b_expanded import _bucket, _update_bucket, _finish_bucket

PROTOCOL = 'p0_f9_surface_ccr_frozen_expanded_validation_v1'
VARIANTS = ('baseline', 'frozen_B', 'surface_CCR', 'old_local_remove_off')
SUBSETS = ('full4369', 'dev512', 'outside_dev512_windows', 'outside_dev512_scenes')
COUNT_FIELDS = ('oi', 'ou', 'si', 'su', 'mi', 'mu')


class Accumulator:
    """Compute a dense confusion count once, reuse exact integers in all subsets."""
    def __init__(self):
        self.cursor = 0
        self.subsets = {name: self._new() for name in SUBSETS}
        self.scenes = {}

    @staticmethod
    def _new():
        return dict(windows=0, scenes=set(), metrics={k: Metrics() for k in VARIANTS},
                    quality={k: defaultdict(int) for k in VARIANTS if k != 'baseline'},
                    action=defaultdict(_bucket),
                    static_errors={k: np.zeros((3, 2, 3), np.int64) for k in VARIANTS})

    def update(self, record, predictions, gt, moving, evidence, target, valid, scores,
               *, dev512_keys, dev512_scenes):
        key = (str(record['scene_name']), str(record['t0_token']))
        scene = key[0]
        active = ['full4369', 'dev512' if key in dev512_keys else 'outside_dev512_windows']
        if scene not in dev512_scenes: active.append('outside_dev512_scenes')
        state = self.scenes.setdefault(scene, dict(windows=0, metrics={k: Metrics() for k in VARIANTS}))
        state['windows'] += 1
        for name in active:
            self.subsets[name]['windows'] += 1
            self.subsets[name]['scenes'].add(scene)
        for ri, h in enumerate((1, 3, 5)):
            ids = np.array([11, 13])
            gt_totals = np.bincount(np.asarray(gt[h]).reshape(-1), minlength=18)[ids]
            for name in VARIANTS:
                counts = Metrics.counts(predictions[name][h], gt[h], moving[h], 17)
                pred_totals = np.bincount(np.asarray(predictions[name][h]).reshape(-1), minlength=18)[ids]
                tp = counts[2][ids]
                static_errors = np.stack((tp, pred_totals - tp, gt_totals - tp), -1)
                state['metrics'][name].update(ri, counts=counts)
                quality = None if name == 'baseline' else edit_quality(predictions['baseline'][h], predictions[name][h], gt[h])
                for subset in active:
                    st = self.subsets[subset]
                    st['metrics'][name].update(ri, counts=counts)
                    st['static_errors'][name][ri] += static_errors
                    if quality is not None:
                        for k, v in quality.items(): st['quality'][name][k] += v
        for name, score in scores.items():
            pred = (score[..., 0] >= .5) & valid[..., 0]
            for role, mask in (('static', evidence.actor < 0), ('dynamic', evidence.actor >= 0)):
                for horizon, h in [('all6', None), ('1.0s', 1), ('2.0s', 3), ('3.0s', 5)]:
                    if h is None: m, y, z = mask[:, None] & valid[..., 0], target[..., 0], pred
                    else: m, y, z = mask & valid[:, h, 0], target[:, h, 0], pred[:, h]
                    for subset in active:
                        _update_bucket(self.subsets[subset]['action'][name + '/' + role + '/' + horizon], m, y, z)
        self.cursor += 1

    @staticmethod
    def _dump_metrics(metrics):
        return {name: {k: getattr(m, k).copy() for k in COUNT_FIELDS} for name, m in metrics.items()}

    @staticmethod
    def _load_metrics(rows):
        if set(rows) != set(VARIANTS): raise RuntimeError('metric variants changed')
        result = {}
        for name, data in rows.items():
            m = Metrics()
            for k in COUNT_FIELDS:
                a = np.asarray(data[k])
                if a.dtype != np.int64 or a.shape != getattr(m, k).shape or (a < 0).any():
                    raise RuntimeError('invalid saved integer metric counts')
                setattr(m, k, a.copy())
            result[name] = m
        return result

    def state_dict(self):
        return dict(cursor=self.cursor, subsets={name: dict(windows=s['windows'], scenes=sorted(s['scenes']),
            metrics=self._dump_metrics(s['metrics']), quality={k: dict(v) for k, v in s['quality'].items()},
            action=dict(s['action']), static_errors={k: v.copy() for k, v in s['static_errors'].items()})
            for name, s in self.subsets.items()},
            scenes={name: dict(windows=s['windows'], metrics=self._dump_metrics(s['metrics']))
                    for name, s in self.scenes.items()})

    def load_state_dict(self, saved):
        if set(saved['subsets']) != set(SUBSETS): raise RuntimeError('saved subsets changed')
        self.cursor = int(saved['cursor'])
        for name, row in saved['subsets'].items():
            self.subsets[name] = dict(windows=int(row['windows']), scenes=set(row['scenes']),
                metrics=self._load_metrics(row['metrics']), quality={k: defaultdict(int, v) for k, v in row['quality'].items()},
                action=defaultdict(_bucket, row['action']), static_errors={k: v.copy() for k, v in row['static_errors'].items()})
            for error in self.subsets[name]['static_errors'].values():
                if error.shape != (3, 2, 3) or error.dtype != np.int64 or (error < 0).any():
                    raise RuntimeError('invalid saved static TP/FP/FN counts')
        self.scenes = {k: dict(windows=int(v['windows']), metrics=self._load_metrics(v['metrics']))
                       for k, v in saved['scenes'].items()}
        if (self.cursor < 0 or self.subsets['full4369']['windows'] != self.cursor
                or sum(s['windows'] for s in self.scenes.values()) != self.cursor
                or self.subsets['dev512']['windows'] + self.subsets['outside_dev512_windows']['windows'] != self.cursor):
            raise RuntimeError('evaluation cursor/count mismatch')

    def report(self):
        result = {}
        for name, st in self.subsets.items():
            if not st['windows']:
                result[name] = dict(available=False, windows=0, scenes=0)
                continue
            metrics = {k: v.compute() for k, v in st['metrics'].items()}
            result[name] = dict(available=True, windows=st['windows'], scenes=len(st['scenes']), metrics=metrics,
                surface_vs_B=delta(metrics['surface_CCR'], metrics['frozen_B']),
                surface_vs_old=delta(metrics['surface_CCR'], metrics['old_local_remove_off']),
                surface_vs_transport=delta(metrics['surface_CCR'], metrics['baseline']),
                action_ADD={k: _finish_bucket(v) for k, v in st['action'].items()},
                road_sidewalk_TP_FP_FN={h: {str(cls): {k: dict(zip(('TP', 'FP', 'FN'), map(int, v[ri, ci])))
                    for k, v in st['static_errors'].items()} for ci, cls in enumerate((11, 13))}
                    for ri, h in enumerate(('1.0', '2.0', '3.0'))},
                quality={k: {**dict(v), 'addition_semantic_precision':
                    v['added_semantic_tp'] / v['added'] if v['added'] else None} for k, v in st['quality'].items()})
        scene_rows = {}
        for scene, row in self.scenes.items():
            m = {k: v.compute() for k, v in row['metrics'].items()}
            scene_rows[scene] = dict(windows=row['windows'], metrics=m,
                surface_vs_B=delta(m['surface_CCR'], m['frozen_B']),
                surface_vs_old=delta(m['surface_CCR'], m['old_local_remove_off']))
        return dict(subsets=result, per_scene=scene_rows)


def save_progress(path, contract, accumulator, speed, performance):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(dict(protocol=PROTOCOL, contract=contract, accumulator=accumulator.state_dict(),
                    speed=speed, performance=dict(performance)), temporary)
    os.replace(temporary, path)


def load_progress(path, contract, accumulator, *, compatible_implementations=()):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    previous = saved.get('contract')
    # Only the CLI's explicitly allow-listed byte-identical execution fix may
    # migrate the implementation fingerprint. Everything else stays strict.
    if (isinstance(previous, dict) and isinstance(contract, dict)
            and previous.get('implementation') in compatible_implementations):
        previous = {**previous, 'implementation': contract.get('implementation')}
    if (saved.get('protocol') != PROTOCOL
            or stable_json_fingerprint(previous) != stable_json_fingerprint(contract)):
        raise RuntimeError('resume requires identical checkpoints/population/execution contract')
    accumulator.load_state_dict(saved['accumulator'])
    return saved['speed'], defaultdict(float, saved['performance'])


@torch.no_grad()
def paired_speed(provider, source, records, teacher, head, reference, *, stop_event=None,
                 windows=20, repeats=3, stress_windows=2, surface_only=False, progress=None,
                 compare_strong_warp=False):
    if type(repeats) is not int or repeats < 1:
        raise ValueError('positive integer FPS repeats required')
    if compare_strong_warp and not surface_only:
        raise ValueError('Strong comparison requires ONE frozen Surface model')
    from real_motion.strong_warp_execution import strong_warp_execution
    selected, population = select_population(records, tuple((str(r['scene_name']), str(r['t0_token'])) for r in records),
                                             windows=windows, stress_windows=stress_windows)
    device = provider.device
    graph = SurfaceExecution(head, device, capture_full_chunks_only=surface_only)
    modes = (('surface_fused_eager', 'surface_fused_graph') if surface_only else
             ('frozen_B', 'surface_eager', 'surface_graph', 'surface_fused_eager', 'surface_fused_graph'))
    if compare_strong_warp:
        modes = ('surface_fused_graph', 'surface_fused_buffered_graph')
    trials, parity, prepares, descriptors, optimized_descriptors = [], 0, 0., 0., 0.
    def sync(): torch.cuda.synchronize(device) if device.type == 'cuda' else None
    try:
        with ThreadPoolExecutor(max_workers=min(4, provider.workers)) as pool:
            for wi, record in enumerate(selected, 1):
                if stop_event is not None and stop_event.is_set(): raise InterruptedError('paired FPS stopped')
                tick = time.perf_counter()
                history = prepare_history(provider, source, record, kernels=ccr.execution_kernels(provider), executor=pool)
                prepares += time.perf_counter() - tick
                plain = history.canonical_evidence
                tick = time.perf_counter()
                atlas = SurfaceAtlas(plain.world, plain.classes, plain.presence, plain.actor, history.current_pose, provider.pcfg.grid)
                enriched = augment_evidence(plain, atlas)
                descriptors += time.perf_counter() - tick
                tick = time.perf_counter()
                parallel_atlas=SurfaceAtlas(plain.world,plain.classes,plain.presence,plain.actor,history.current_pose,provider.pcfg.grid)
                parallel_atlas.query_workers=min(4,provider.workers)
                optimized_enriched=augment_evidence(plain,parallel_atlas)
                optimized_descriptors += time.perf_counter()-tick
                if not np.array_equal(enriched.features,optimized_enriched.features):
                    raise RuntimeError('parallel history surface descriptor bytes differ')
                matrices = [np.linalg.inv(p) for p in history.future_poses]
                phase_time = [0.]
                def probability(model, evidence, plan, output, where, mode):
                    if mode != 'frozen_B':
                        tick = time.perf_counter()
                        plan = augment_projection(evidence, plan, history.current_pose, matrices, provider.pcfg.grid)
                        phase_time[0] += time.perf_counter() - tick
                    return (graph(model, evidence, plan, output, where) if mode.endswith('graph')
                            else frozen_b_probabilities(model, evidence, plan, output, where))
                def run(mode, timed):
                    model = reference if mode == 'frozen_B' else head
                    history.canonical_evidence = plain if mode == 'frozen_B' else enriched
                    phase_time[0] = 0.
                    sync()
                    if timed and device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
                    before = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
                    tick = time.perf_counter()
                    with strong_warp_execution('buffered' if mode == 'surface_fused_buffered_graph' else 'reference'):
                        out = forecast_six(history, provider, teacher.transport, model,
                            lambda *args: probability(*args, mode), kernels=ccr.execution_kernels(provider), executor=pool,
                            projection_fn=map_surface_evidence if mode.startswith('surface_fused') else None)
                    sync(); seconds = time.perf_counter() - tick
                    out['stages_seconds']['live_surface_phase_INCLUDED_in_readout'] = phase_time[0]
                    memory = (dict(peak_allocated=torch.cuda.max_memory_allocated(device)/2**20,
                        incremental_peak=(torch.cuda.max_memory_allocated(device)-before)/2**20,
                        peak_reserved=torch.cuda.max_memory_reserved(device)/2**20) if device.type == 'cuda' else None)
                    return out, seconds, memory
                warm_modes = ('surface_eager', *modes) if surface_only else modes
                expected = {name: run(name, False)[0] for name in warm_modes}
                if any(len(value['dense']) != 6 for value in expected.values()):
                    raise RuntimeError('FPS requires all SIX dense frames')
                for mode in modes:
                    if mode == 'frozen_B': continue
                    if (not np.array_equal(expected[mode]['probability'],expected['surface_eager']['probability'])
                            or any(not np.array_equal(a,b) for a,b in zip(expected[mode]['dense'],expected['surface_eager']['dense']))):
                        raise RuntimeError('fused projection probability/dense bytes differ; use reference execution')
                dynamic = plain.actor >= 0
                if not surface_only and not np.array_equal(expected['frozen_B']['probability'][dynamic], expected['surface_eager']['probability'][dynamic]):
                    raise RuntimeError('frozen dynamic predictions changed')
                if surface_only and np.any(expected['surface_eager']['probability'][..., 1] != 0):
                    raise RuntimeError('formal ADD-only benchmark requires REMOVE disabled')
                parity += 1
                captures = (graph.counts['captures_verified'], graph.counts['capture_rejections'])
                for repeat in range(repeats):
                    # Rotate all arms, do not always favour the same warm-order.
                    offset = (wi + repeat - 1) % len(modes)
                    order = modes[offset:] + modes[:offset]
                    for mode in order:
                        out, seconds, memory = run(mode, True)
                        if surface_only and (graph.counts['captures_verified'], graph.counts['capture_rejections']) != captures:
                            raise RuntimeError('graph captured inside FPS timer; reject this measurement')
                        if (len(out['dense']) != 6 or not np.array_equal(out['probability'], expected[mode]['probability'])
                                or any(not np.array_equal(a, b) for a, b in zip(out['dense'], expected[mode]['dense']))):
                            raise RuntimeError('repeated six-frame output bytes changed')
                        row = dict(mode=mode, window=wi, repeat=repeat+1, seconds=seconds,
                                   stages_seconds=out['stages_seconds'], memory_mib=memory)
                        if compare_strong_warp:
                            row['strong_profile_ms'] = out.get('strong_profile_ms', {})
                        trials.append(row)
                        if progress is not None: progress(row)
                print(f'SURFACE_PAIRED_FPS {wi}/{windows} graph/eager/six-dense parity=PASS', flush=True)
        means = {name: float(np.mean([r['seconds'] for r in trials if r['mode'] == name])) for name in modes}
        # Prefer reference unless the verified graph actually ran and its
        # paired mean is at least 2% faster; don't select on tiny timing noise.
        eligible = ['surface_fused_eager' if surface_only else 'surface_eager']
        if surface_only:
            # Fixed verified graph backend, not a fastest-arm winner picked on noise.
            chosen = 'surface_fused_graph'
        elif (not graph.failures and graph.counts['graph_replays'] > 0
                and means['surface_graph'] < .98 * means['surface_eager']):
            eligible.append('surface_graph')
        if not surface_only: chosen = min(eligible, key=lambda k: means[k])
        return dict(population=population, trials=trials, six_frame_mean_seconds=means,
            six_frame_amortized_FPS={k: 6/v for k, v in means.items()},
            p90_six_ms={k: float(np.percentile([r['seconds']*1000 for r in trials if r['mode'] == k], 90)) for k in modes},
            stages_mean_ms={k: {s: float(np.mean([r['stages_seconds'][s]*1000 for r in trials if r['mode'] == k]))
                for s in trials[0]['stages_seconds'] if all(s in r['stages_seconds'] for r in trials if r['mode'] == k)} for k in modes},
            probability_and_six_dense_parity_windows=parity,
            dynamic_byte_parity_windows=None if surface_only else parity,
            selected_execution=chosen, selection_basis=('fixed fused_graph backend with eager fallbacks; no latency selection'
                if surface_only else 'latency only after byte parity, NOT accuracy/method selection'),
            strong_warp_comparison=bool(compare_strong_warp),
            graph_execution=graph.stats(), history_prepare_seconds_per_window=prepares/windows,
            surface_descriptor_prepare_seconds_per_window=descriptors/windows,
            optimized_surface_descriptor_prepare_seconds_per_window=optimized_descriptors/windows,
            actual_cuda=device.type == 'cuda', fps_windows=windows, repeats=repeats,
            memory_mib={k: {field: max(r['memory_mib'][field] for r in trials if r['mode'] == k)
                for field in ('peak_allocated', 'incremental_peak', 'peak_reserved')} for k in modes}
                if device.type == 'cuda' else None,
            boundary='CausalHistoryState -> fresh Strong/KTA + live motion + learned CCR + live projection conditioning + SIX dense outputs',
            excludes='history-only representation/I-O/GT/metrics/capture/parity checks; NOT raw-input E2E FPS')
    finally: graph.close()
