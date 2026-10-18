"""Extract actual training throughput; never infer FPS from training/evaluation."""
import json
import math
from pathlib import Path
import statistics


def training(folder, event, skip=16, tail=256):
    rows = {}; duplicates = 0
    for line in (Path(folder)/'progress.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        item = json.loads(line)
        if item.get('event') != event: continue
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


def pair(control_dir, integrated_dir):
    from tools.source_scene_integrated.report import reference, MATCH_FIELDS
    from tools.source_scene_integrated.run import PROTOCOL
    from tools.joint_occ_experiments.run import sha256
    control = reference(control_dir)
    integrated = json.loads((Path(integrated_dir)/'results.json').read_text(encoding='utf-8'))
    c = integrated['contract']
    if (integrated['status'] != 'complete' or c['protocol'] != PROTOCOL or
            c['arm'] != 'integrated_encoder_AdaGN' or c['model']['variant'] != 'source_scene_integrated_AdaGN' or
            integrated['updates'] != c['total_updates'] or integrated['epoch'] != 1 or
            c['model']['kta_conditioning'] is not True or c['registration'] != 'none' or
            c['source_policy'] != 'shared_pinned' or c['known'] != 'TRAIN_OCC_STC_dense_valid' or
            integrated['evaluations']['1']['windows'] != 64 or
            integrated['evaluations']['1']['keys'] != c['eval_keys']):
        raise ValueError('completed integrated TRAIN20%x1 required')
    for name in (*MATCH_FIELDS, 'source_policy', 'registration', 'execution'):
        if c[name] != control['contract'][name]: raise ValueError('mismatched training contract: '+name)
    if c['reference_results_sha256'] != sha256(Path(control_dir)/'results.json'):
        raise ValueError('not the original KTA-query control of this integrated run')
    records = {}
    for arm, folder, result, event in (
            ('KTA_query', control_dir, control, 'scene_stage2_kta_query_train'),
            ('integrated', integrated_dir, integrated, 'scene_integrated_train')):
        t = training(folder, event)
        if t['observed_updates'] != list(range(1, c['total_updates']+1)):
            raise ValueError('incomplete original training log: '+arm)
        records[arm] = dict(folder=str(Path(folder).resolve()), contract=result['contract'], training=t,
            source_files={name: sha256(Path(folder)/name) for name in ('results.json', 'progress.jsonl')})
    return records


def select_integrated(root, explicit=None):
    from tools.source_scene_integrated.run import PROTOCOL
    def completed(folder):
        v = json.loads((folder/'results.json').read_text(encoding='utf-8'))
        return (v['status'] == 'complete' and v['contract']['protocol'] == PROTOCOL and
                v['updates'] == v['contract']['total_updates'] and v['epoch'] == 1)
    if explicit:
        path = Path(explicit).resolve()
        if not completed(path): raise ValueError('completed integrated pilot required')
        return path
    candidates = []
    for path in Path(root).glob('scene_integrated_1ep_*/results.json'):
        try:
            if completed(path.parent): candidates.append(path.parent.resolve())
        except (ValueError, KeyError): continue
    if len(candidates) != 1: raise ValueError('Set INTEGRATED_DIR explicitly; completed candidates='+str(candidates))
    return candidates[0]


def summary(result):
    lines = ['===== MATCHED KTA QUERY / INTEGRATED RESOURCES =====', 'status='+result['status'],
             'Scientific updates=0; source checkpoints/logs/caches read only.',
             'Training times are actual original GLOBAL-B4 multi-GPU logs, not single-GPU estimates.',
             'arm       parameters encoder motion_ex_encoder refiner train_s/window tail_s/window']
    for arm, v in result['arms'].items():
        p, t = v['parameters'], v['training']
        lines.append(f'{arm:8s} {p["total"]:10d} {p["encoder"]:7d} {p["motion_ex_encoder"]:17d} {p["refiner"]:7d} '
                     f'{t["post_startup"]["seconds_per_window"]:.6f} {t["tail"]["seconds_per_window"]:.6f}')
    left, right = (result['arms'][a] for a in ('KTA_query', 'integrated'))
    lines.append('integrated_parameter_reduction_percent='+str(100*(1-right['parameters']['total']/left['parameters']['total'])))
    lines.append('integrated_training_speedup='+str(left['training']['post_startup']['seconds_per_window']/right['training']['post_startup']['seconds_per_window']))
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
        lines.append(f'{kind} integrated_FPS_speedup='+str(right['fps'][kind]['output_frame_fps']/left['fps'][kind]['output_frame_fps']))
    lines += ['FPS includes live original two-frame KTA + fresh Strong/CLEAR + Encoder/STWM/lift + Transport + memory + Refiner + SIX CPU semantics.',
              'Excludes history disk I/O/CC/association/read-coordinate preparation/initial H2D/warmup/profiling/labels/metrics/writes.',
              'No old half-KTA/age/static support/calibration/norm192 or optimized backend imported. Not camera end-to-end FPS.',
              'All timed samples retained even on strict byte FAIL. No automatic model selection/promotion or full training launch.']
    return '\n'.join(lines)+'\n'
