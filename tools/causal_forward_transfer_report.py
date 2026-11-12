"""Summarize completed paper-epoch2 nuScenes long6 and Waymo evaluations."""
import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import mean


SOURCE_SHA256 = '7760c6f4f92277aa248161cd3e90f87ac2354329babceb6e5ed3b872adde33a0'
RUNS = Path(__file__).resolve().parents[1] / 'outputs/p0_f9_shared_history'
WINDOWS = {'long6': 2569, 'waymo2': 7998, 'waymo10': 39987}
WAYMO_ROUTES = (('gt/transport', 'Transport'), ('gt/semantic_refine', 'Refiner'))


def load_report(path, task):
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('status') != 'complete' or value.get('task') != task
            or value.get('windows') != WINDOWS[task] or value.get('source_update') != 10292
            or value.get('source_sha256') != SOURCE_SHA256
            or value.get('population', {}).get('diagnostic_prefix') is not False):
        raise ValueError(f'{path} is not a complete full-population {task} report for the fixed epoch2')
    if 'imported_windows' in value and value['imported_windows'] + value['new_windows'] != WINDOWS[task]:
        raise ValueError(f'{path} has inconsistent parallel window totals')
    return value['reports']


def scores(row, miou_key):
    pair = row[miou_key], row['IoU']
    if any(not isinstance(x, (int, float)) or not isfinite(x) or not 0 <= x <= 100 for x in pair):
        raise ValueError(f'invalid {miou_key}/IoU values {pair}')
    return pair


def average_row(label, points, stored, miou_key):
    pair = scores(stored, miou_key)
    if any(abs(mean(values) - expected) > 1e-6 for values, expected in zip(zip(*points), pair)):
        raise ValueError(f'stored average differs from the per-horizon scores for {label}')
    return label, pair


def long_rows(report):
    table = report['semantic_refine_rollout']
    points = [scores(table['per_horizon'][f'{h}.0'], 'mIoU') for h in range(1, 7)]
    rows = [(f'{h} s', score) for h, score in enumerate(points, start=1)]
    return rows + [average_row('1–3 s 平均', points[:3], table['average_1s_2s_3s'], 'mIoU'),
                   average_row('4–6 s 平均', points[3:], table['average_4s_5s_6s'], 'mIoU')]


def waymo_rows(report, task, miou_key):
    if task == 'waymo10':
        keys, horizons = ('eval_time_1', 'eval_time_3', 'eval_time_5'), (.2, .4, .6)
    else:
        keys, horizons = ('1', '2', '3'), (1, 2, 3)
    for route, name in WAYMO_ROUTES:
        table = report[route]
        values = [table['horizons'][key] for key in keys]
        if task == 'waymo10' and [v.get('nominal_seconds') for v in values] != list(horizons):
            raise ValueError('Waymo 10 Hz result has incorrect physical horizons')
        points = [scores(value, miou_key) for value in values]
        for horizon, score in zip(horizons, points):
            yield name, f'{horizon:g} s', score
        label, score = average_row('平均', points, table['average'], miou_key)
        yield name, label, score


def render(run, waymo10, miou_key):
    sections = []
    for task, title in (('long6', 'nuScenes 长时预测'), ('waymo2', 'Occ3D-Waymo 2 Hz'),
                        ('waymo10', 'Occ3D-Waymo 10 Hz')):
        path = waymo10 if task == 'waymo10' else run / task / 'evaluation.json'
        report = load_report(path, task)
        if task == 'long6':
            rows = [('Refiner rollout', horizon, score) for horizon, score in long_rows(report)]
        else:
            rows = list(waymo_rows(report, task, miou_key))
        sections.append('\n'.join([f'## {title}', '', f'来源 `{path}`', '',
                                   '| 输出 | 时距 | mIoU (%) | IoU (%) |', '|---|---|---:|---:|',
                                   *(f'| {route} | {horizon} | {m:.3f} | {i:.3f} |'
                                     for route, horizon, (m, i) in rows)]))
    scope = '标准 mIoU' if miou_key == 'standard_mIoU' else 'I²-World mIoU（排除 IoU 为零的类别）'
    return '\n\n'.join(['# CarryWorld epoch2 评估汇总', f'GT 未来 ego · Waymo 采用{scope}', *sections])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', nargs='?', type=Path, help='directory containing long6, waymo2 and waymo10 results')
    parser.add_argument('--waymo10-dir', type=Path, help='explicit completed Waymo 10 Hz directory')
    parser.add_argument('--waymo-miou', choices=('standard', 'i2world'), default='standard')
    args = parser.parse_args(argv)
    try:
        run = args.run_dir
        if run is None:
            runs = [path for path in RUNS.glob('forward_transfer_*') if path.is_dir()]
            if not runs:
                raise ValueError('no forward_transfer_* run found; pass its directory explicitly')
            run = max(runs, key=lambda path: path.name[-15:])
        if args.waymo10_dir is not None:
            waymo10 = args.waymo10_dir / 'evaluation.json'
        else:
            candidates = list(run.glob('waymo10_parallel*/evaluation.json'))
            serial = run / 'waymo10/evaluation.json'
            if serial.is_file():
                candidates.append(serial)
            if not candidates:
                raise ValueError(f'no completed Waymo 10 Hz evaluation.json under {run}')
            waymo10 = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        print(render(run, waymo10, args.waymo_miou + '_mIoU'))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'ERROR {error}\n')


if __name__ == '__main__':
    main()
