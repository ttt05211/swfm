"""Read existing prior-strength JSONs only: no Torch, GPU, forecasting or writes."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

PROTOCOL = 'frozen_final_prior_strength_DEV64_c1_c05_v1'
INPUTS = ('occ_gt', 'stc_gt')
LABELS = ('transport', 'original_c1', 'diagnostic_c05')
CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
           'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
           'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')


def score(c):
    """Original report: mean valid semantic IoUs, then mean 1/2/3s."""
    values = []
    for m in c:
        t = np.diag(m)[:17]; u = (m.sum(0)+m.sum(1)-np.diag(m))[:17]
        values.append(float((100*t[u > 0]/u[u > 0]).mean()) if (u > 0).any() else np.nan)
    return float(np.nanmean(values))


def binary_errors(c):
    return dict(FP=int(c[:, 17, :17].sum()), FN=int(c[:, :17, 17].sum()))


def damage_split(before, after, edits):
    """Recover exact free->occupied damage from confusion + paired edits.

    FP_after - FP_before = damaged_correct_free - corrected_false_occupancy.
    Newly missed occupancy = FN_after - FN_before + all changed wrong-free.
    The latter includes transitions from previously wrong occupied classes;
    it must NOT be called damage to previously correct occupied semantics.
    """
    a, b = binary_errors(before), binary_errors(after)
    damaged = edits['all']['damaged']
    free = b['FP']-a['FP']+edits['false_occupancy']['corrected']
    newly_missed = b['FN']-a['FN']+edits['wrong_free']['changed']
    if not 0 <= free <= damaged or newly_missed < 0:
        raise ValueError('paired edit/confusion identity inconsistent')
    return dict(damaged_correct_free=free, damaged_correct_occupied=damaged-free,
        free_share_of_damage_pct=100*free/damaged if damaged else 0.,
        newly_missed_occupancy=newly_missed, delta_FP=b['FP']-a['FP'], delta_FN=b['FN']-a['FN'])


def resolve_result(value=None):
    if value:
        p = Path(value).resolve()
        return p/'evaluation.json' if p.is_dir() else p
    env = os.environ.get('PRIOR_DIAG_DIR')
    if env: return Path(env).resolve()/'evaluation.json'
    candidates = sorted(Path('outputs/p0_f9_shared_history').glob('prior_strength_dev64_*/evaluation.json'),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for p in candidates:
        r = json.loads(p.read_text(encoding='utf-8'))
        if r.get('status') == 'complete' and r.get('contract', {}).get('protocol') == PROTOCOL: return p.resolve()
    raise FileNotFoundError('No complete result: pass --result /actual/directory/evaluation.json')


def load(path):
    r = json.loads(path.read_text(encoding='utf-8'))
    state = json.loads(path.with_name('state.json').read_text(encoding='utf-8'))
    contract = r['contract']; rows = state['rows']; keys = contract['keys']
    if (r['status'] != 'complete' or contract['protocol'] != PROTOCOL or r['windows'] != 64
            or r['scientific_updates'] != 0 or r['strengths'] != [1., .5]
            or r.get('source_epoch') != 2 or state['contract'] != contract
            or len(rows) != len(keys) or len(rows) != 64 or len({tuple(k) for k in keys}) != 64):
        raise ValueError('Expected complete original fixed64 prior diagnostic, matching state/contract')
    expected = {n+'/'+v for n in INPUTS for v in LABELS}
    counts = {n: np.zeros((3, 18, 18), np.int64) for n in expected}
    volume = None
    for row, key in zip(rows, keys):
        if row['key'] != key or set(row['counts']) != expected: raise ValueError('saved prefix/routes differ')
        for name, v in row['counts'].items():
            c, m = np.asarray(v['confusion']), np.asarray(v['moving'])
            if (c.shape != (3, 18, 18) or m.shape != c.shape or c.dtype.kind not in 'iu' or m.dtype.kind not in 'iu'
                    or (c < 0).any() or (m < 0).any() or (m > c).any()): raise ValueError('invalid saved integer counts')
            if volume is None: volume = int(c[0].sum())
            if volume <= 0 or np.any(c.sum((1, 2)) != volume): raise ValueError('saved grid totals differ')
            counts[name] += c
    for name, c in counts.items():
        if abs(score(c)-r['reports'][name]['average']['mIoU']) > 1e-8:
            raise ValueError('report does not reproduce raw counts: '+name)
    for name in INPUTS:
        for comparison, a, b in (('original_c1', 'transport', 'original_c1'),
                                 ('diagnostic_c05', 'transport', 'diagnostic_c05'),
                                 ('half_vs_original', 'original_c1', 'diagnostic_c05')):
            e = r['edits'][name][comparison]['totals']
            if (e['all']['count'] != 3*len(rows)*volume
                    or e['all']['corrected']+e['all']['damaged']+e['all']['wrong_to_other_wrong'] != e['all']['changed']):
                raise ValueError('invalid paired edit totals')
            damage_split(counts[name+'/'+a], counts[name+'/'+b], e)
            for g, item in e.items():
                for k in ('count', 'changed', 'corrected', 'damaged', 'wrong_to_other_wrong'):
                    if item[k] != sum(row['edits'][name][comparison][h][g][k] for row in rows for h in range(3)):
                        raise ValueError('report does not reproduce saved paired edits')
    return r, rows, counts


def report(path):
    r, rows, counts = load(path)
    lines = [f'Result: {path}', '===== READ-ONLY PRIOR FAILURE BREAKDOWN =====',
             f"windows=64; epoch=2; source_update={r['source_update']}; raw counts reproduced; NO GPU/forecast/train/writes"]
    for name in INPUTS:
        lines.append('\n===== '+name+' =====')
        for label in LABELS:
            v = r['reports'][name+'/'+label]['average']; err = binary_errors(counts[name+'/'+label])
            lines.append(f"{label:16s} mIoU={v['mIoU']:.6f} IoU={v['IoU']:.6f} FP={err['FP']} FN={err['FN']}")
        for comparison, a, b in (('original_c1', 'transport', 'original_c1'),
                                 ('diagnostic_c05', 'transport', 'diagnostic_c05'),
                                 ('half_vs_original', 'original_c1', 'diagnostic_c05')):
            e = r['edits'][name][comparison]['totals']; s = damage_split(counts[name+'/'+a], counts[name+'/'+b], e)
            lines.append(f"{comparison}: correct={e['all']['corrected']} damage={e['all']['damaged']} "
                         f"damage_correct_free={s['damaged_correct_free']} ({s['free_share_of_damage_pct']:.2f}%) "
                         f"damage_correct_occupied={s['damaged_correct_occupied']}")
            lines.append('  repaired_semantics: '+', '.join(f"{g}={e[g]['corrected']}/{e[g]['count']} "
                f"({100*e[g]['corrected']/e[g]['count'] if e[g]['count'] else 0:.2f}%)"
                for g in ('wrong_free', 'false_occupancy', 'wrong_class')))
            lines.append(f"  binary_delta_FP={s['delta_FP']} delta_FN={s['delta_FN']} newly_missed_occupancy={s['newly_missed_occupancy']}")
        lines.append('half-original by horizon:')
        for h in ('1.0', '2.0', '3.0'):
            a = r['reports'][name+'/original_c1']['per_horizon'][h]
            b = r['reports'][name+'/diagnostic_c05']['per_horizon'][h]
            lines.append(f"  {h}s dmIoU={b['mIoU']-a['mIoU']:+.6f} dIoU={b['IoU']-a['IoU']:+.6f}")
        changes = []
        for i, label in enumerate(CLASSES):
            a, b = (r['reports'][name+'/'+v]['per_horizon'] for v in ('original_c1', 'diagnostic_c05'))
            valid = [(b[h]['semantic_per_class'][str(i)]-a[h]['semantic_per_class'][str(i)]) for h in a
                     if a[h]['semantic_per_class'][str(i)] is not None and b[h]['semantic_per_class'][str(i)] is not None]
            if valid: changes.append((float(np.mean(valid)), label))
        changes.sort()
        lines.append('worst5 semantic class dIoU (mean common-valid horizons): '+', '.join(f'{label}:{v:+.3f}' for v, label in changes[:5]))
        deltas, scenes = [], {}
        for row in rows:
            a, b = (np.asarray(row['counts'][name+'/'+v]['confusion'], np.int64) for v in ('original_c1', 'diagnostic_c05'))
            deltas.append(score(b)-score(a))
            scene = row['key'][0]
            if scene not in scenes: scenes[scene] = [np.zeros_like(a), np.zeros_like(b), 0]
            scenes[scene][0] += a; scenes[scene][1] += b; scenes[scene][2] += 1
        scene_deltas = sorted((score(b)-score(a), scene, n) for scene, (a, b, n) in scenes.items())
        lines.append(f"single-window dmIoU positive={sum(v > 0 for v in deltas)} negative={sum(v < 0 for v in deltas)} "
                     f"median={np.median(deltas):+.4f}; scene-negative={sum(v < 0 for v, _, _ in scene_deltas)}/{len(scenes)}")
        lines.append('worst3 scenes: '+', '.join(f'{scene}(n={n}):{v:+.3f}' for v, scene, n in scene_deltas[:3]))
    lines += ['\nFP = GT-free predicted occupied; FN = GT-occupied predicted free.',
        'Damage to correct occupied includes deletion OR semantic relabeling; newly missed includes previously wrong occupied classes.',
        'Class/window/scene diagnostics do NOT decompose aggregate mIoU into causal contributions.',
        'Existing JSON has no spatial coordinates/logits: cannot infer distance, temporal-noise distribution or local gate quality.',
        'Fixed DEV mechanism audit, NOT an independent test or coefficient/augmentation selection. No source files changed.']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', help='Existing evaluation.json or its directory; otherwise PRIOR_DIAG_DIR/latest complete run')
    print(report(resolve_result(parser.parse_args().result)))
