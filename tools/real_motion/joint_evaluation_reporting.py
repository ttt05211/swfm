"""Read existing results/progress without Torch, GPU, predictions or writes."""
from collections import defaultdict
import json
import math
from pathlib import Path


def number(value, *, signed=False):
    if value is None or not math.isfinite(float(value)): return 'N/A'
    return format(float(value), '+.6f' if signed else '.6f')


def difference(a, b):
    return None if a is None or b is None else a-b


def summary_text(result):
    row = result['reports']['all']; ref = row['reference_metrics']['frozen_E14']
    metrics = ('IoU', 'mIoU', 'MovingMacro', 'MovingMicro')
    def values(m): return ' '.join(k+'='+number(m[k]) for k in metrics)
    lines = ['===== INTERIM FULL LOCAL JOINT (DIAGNOSTIC ONLY) =====',
        f"population: {result['population']} / {row['windows']} windows / {row['scenes']} scenes",
        f"update: {result['attempted_updates']}  completed_epochs: {result['cursor_epoch']}  batch_cursor: {result['cursor_batch']}",
        f"history_frames: {result['history_frames']}; future_frames: 6",
        f"thresholds: {result['thresholds']}; source: {result['threshold_source']}",
        'frozen_E14: '+values(ref)+' (legacy SIX histories; not a matched four-history budget)']
    items = {'transport': row['baseline'], **{k: row['variants'][k]['metrics'] for k in ('generation', 'refine', 'joint')}}
    for name, m in items.items():
        lines.append(name+': '+values(m)+' '+
            ' '.join('d'+k+'_vs_E14='+number(difference(m[k], ref[k]), signed=True) for k in metrics)+
            ' dMiOU_vs_transport='+number(difference(m['mIoU'], row['baseline']['mIoU']), signed=True))
    for h in ('1.0', '2.0', '3.0'):
        m, r = items['joint']['per_horizon'][h], ref['per_horizon'][h]
        lines.append(h+'s joint: '+values(m)+' '+
            ' '.join('d'+k+'_vs_E14='+number(difference(m[k], r[k]), signed=True) for k in metrics))
    scene = {k: v for k, v in row['variants']['joint']['scene_delta'].items() if k != 'by_scene'}
    lines += ['joint addition/removal: '+str(row['variants']['joint']['quality']),
        'joint scenes vs CURRENT transport: '+str(scene),
        f"seconds: {result['seconds']:.2f}", f"checkpoint_snapshot: {result['snapshot']}",
        'No optimizer/RNG changes, TRAIN recalibration, dev-selected checkpoint, promotion or source checkpoint writes.']
    return '\n'.join(lines)+'\n'


def summarize_progress(path):
    """Host times are not active GPU utilization. Worker times overlap."""
    totals = defaultdict(float); samples = defaultdict(int); durations = []; events = 0
    if not Path(path).is_file(): return 'No progress.jsonl: metric report only.\n'
    with Path(path).open(encoding='utf-8') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip(): continue
            try: row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f'incomplete/corrupt progress JSON at line {lineno}; no timings fabricated')
            if row.get('event') != 'evaluation': continue
            events += 1
            def add(key, value):
                if isinstance(value, (int, float)) and math.isfinite(value):
                    totals[key] += value; samples[key] += 1
            for key in ('seconds', 'compute_seconds', 'input_wait_seconds', 'reference_seconds',
                        'candidate_inference_metrics_seconds', 'candidate_wait_seconds', 'column_probability_seconds',
                        'composition_metrics_seconds', 'moving_support_seconds'):
                add(key, row.get(key))
            if isinstance(row.get('seconds'), (int, float)): durations.append(row['seconds'])
            for name, value in row.get('prepare_seconds', {}).items(): add('prepare.'+name, value)
            for name, value in row.get('raw_prefetch_worker_seconds', {}).items():
                if name != 'geometry_workers': add('raw_worker_OVERLAPPING.'+name, value)
            horizon_totals = defaultdict(float)
            for profile in row.get('prediction_seconds_by_horizon', {}).values():
                for name in ('queries', 'inverse_map_seconds', 'patch_gather_seconds', 'sampling_wait_seconds',
                             'network_transfer_and_other_seconds','probability_readback_transfers',
                             'probability_readback_buffer_bytes'):
                    value = profile.get(name)
                    if isinstance(value, (int, float)) and math.isfinite(value): horizon_totals[name] += value
            for name, value in horizon_totals.items(): add('all_3_horizons.'+name, value)
    lines = ['===== EXISTING PROGRESS TIMINGS (NO EVALUATION RERUN) =====', f'logged_windows={events}']
    if not events: return '\n'.join(lines)+'\n'
    for key in sorted(totals): lines.append(f"{key}/window={totals[key]/samples[key]:.6f} samples={samples[key]}")
    if durations:
        ordered = sorted(durations)
        lines.append('p50_seconds/window='+number(ordered[(len(ordered)-1)//2]))
        lines.append('p90_seconds/window='+number(ordered[int((len(ordered)-1)*.9)]))
    if totals['seconds'] > 0:
        lines.append(f"input_wait_fraction={totals['input_wait_seconds']/totals['seconds']:.2%}")
    lines.append('Overlapping worker time and inclusive stages must NOT be summed as serial wall time; host timings are NOT GPU utilization.')
    return '\n'.join(lines)+'\n'


def read_report(path):
    path = Path(path)
    if path.is_dir(): path = path/'evaluation.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'complete': raise ValueError('completed evaluation.json required')
    return summary_text(result)+summarize_progress(path.parent/'progress.jsonl')
