"""Extract actual training throughput; never infer FPS from training/evaluation."""
import json
import math
from pathlib import Path
import statistics


def training(folder, skip=16, tail=256):
    rows = {}; duplicates = 0
    for line in (Path(folder)/'progress.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        item = json.loads(line)
        if item.get('event') != 'scene_stage1_train': continue
        update = int(item['update'])
        duplicates += update in rows
        # Resume may replay a logged but not checkpointed update. Keep the last
        # occurrence by cursor, NEVER choose the faster occurrence.
        rows[update] = item
    selected = [rows[k] for k in sorted(rows) if k > skip]
    if not selected: raise ValueError('no training updates after startup exclusion')
    def aggregate(values):
        n = sum(int(r['windows']) for r in values)
        if n <= 0 or any(int(r['windows']) <= 0 or not math.isfinite(r['seconds_per_window']) or
                         r['seconds_per_window'] <= 0 for r in values):
            raise ValueError('invalid training time/window count')
        seconds = sum(r['seconds_per_window']*int(r['windows']) for r in values)
        return dict(updates=[values[0]['update'], values[-1]['update']], count=len(values), windows=n,
                    seconds=seconds, seconds_per_window=seconds/n,
                    median_update_seconds_per_window=statistics.median(r['seconds_per_window'] for r in values))
    return dict(post_startup=aggregate(selected), tail=aggregate(selected[-tail:]),
                skipped_first_updates=skip, replayed_log_rows=duplicates,
                observed_updates=sorted(rows),
                scope='max rank pipeline elapsed / GLOBAL windows; includes input wait and gradient collectives; excludes checkpoint/eval/startup; not single-GPU training speed')


def pair(root):
    root = Path(root); records = {}
    for arm in ('baseline', 'reader'):
        result = json.loads((root/arm/'results.json').read_text(encoding='utf-8'))
        c = result['contract']
        if result['status'] != 'complete' or c['arm'] != arm: raise ValueError('two completed arms required')
        records[arm] = dict(contract=c, training=training(root/arm))
    a, b = (records[k] for k in ('baseline', 'reader'))
    for k in ('protocol', 'train_keys', 'eval_keys', 'world_size', 'epochs', 'total_updates', 'schedule',
              'objective', 'known', 'source_policy', 'seed', 'learning_rates', 'precision', 'execution',
              'data', 'overrides', 'source_implementation', 'implementation', 'STC_inventory'):
        if a['contract'][k] != b['contract'][k]: raise ValueError('mismatched training contract: '+k)
    if a['training']['observed_updates'] != b['training']['observed_updates']:
        raise ValueError('training log update coverage differs')
    if a['training']['observed_updates'] != list(range(1, int(a['contract']['total_updates'])+1)):
        raise ValueError('incomplete original training log')
    return records


def summary(result):
    lines = ['===== MATCHED BASELINE / READER RESOURCES =====', 'status='+result['status'],
             'Scientific updates=0; source checkpoints/logs/caches read only.',
             'Training times are actual original GLOBAL-B4 multi-GPU logs, not single-GPU estimates.',
             'arm       parameters encoder motion_ex_encoder refiner train_s/window tail_s/window']
    for arm, v in result['arms'].items():
        p, t = v['parameters'], v['training']
        lines.append(f'{arm:8s} {p["total"]:10d} {p["encoder"]:7d} {p["motion_ex_encoder"]:17d} {p["refiner"]:7d} '
                     f'{t["post_startup"]["seconds_per_window"]:.6f} {t["tail"]["seconds_per_window"]:.6f}')
    left, right = (result['arms'][a] for a in ('baseline', 'reader'))
    lines.append('reader_parameter_reduction_percent='+str(100*(1-right['parameters']['total']/left['parameters']['total'])))
    lines.append('reader_training_speedup='+str(left['training']['post_startup']['seconds_per_window']/right['training']['post_startup']['seconds_per_window']))
    lines += ['Training first16 updates excluded; tail=last256 updates (weighted by actual windows).',
              'FPS: ONE GPU / ONE process / one active model; SAME keys / H6 / original BF16 / dense-valid.',
              'arm modality six_frame_ms output_frame_fps window_fps byte_fail_calls max_changed_voxels']
    for arm, v in result['arms'].items():
        for kind, speed in v.get('fps', {}).items():
            lines.append(f'{arm:8s} {kind:8s} {speed["six_frame_ms"]:.6f} {speed["output_frame_fps"]:.6f} '
                         f'{speed["window_fps"]:.6f} {speed["byte_failed_calls"]} {speed["max_changed_voxels"]}')
            lines.append(f'  serialized_profile_ms (NOT summed FPS)={speed["profile_mean_ms"]}')
            lines.append(f'  strict_byte_parity={speed["strict_byte_parity"]}; reference_repeat_max_changed_voxels={speed["reference_repeat_max_changed_voxels"]}; p90_ms={speed["p90_six_frame_ms"]}; peak_allocated_MiB={speed["peak_allocated_mib"]}; peak_reserved_MiB={speed["peak_reserved_mib"]}')
    for kind in sorted(set(left.get('fps', {})) & set(right.get('fps', {}))):
        lines.append(f'{kind} reader_FPS_speedup='+str(right['fps'][kind]['output_frame_fps']/left['fps'][kind]['output_frame_fps']))
    lines += ['FPS includes live original two-frame KTA + fresh Strong/CLEAR + Encoder/STWM/lift + Transport + memory + Refiner + SIX CPU semantics.',
              'Excludes history disk I/O/CC/association/registration/read-coordinate preparation/initial H2D/warmup/profiling/labels/metrics/writes.',
              'No old half-KTA/age/static support/calibration/norm192 or optimized backend imported. Not camera end-to-end FPS.',
              'All timed samples retained even on strict byte FAIL. No automatic model selection/promotion.']
    return '\n'.join(lines)+'\n'
