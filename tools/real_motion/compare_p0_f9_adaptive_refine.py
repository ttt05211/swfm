#!/usr/bin/env python3
"""Read-only paired summaries, strict identical population/budget, no dev tuning."""
import argparse
import json
import math
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from real_motion.joint_causal_columns import PROTOCOL, CONTRACT
from real_motion.adaptive_column_context import PROTOCOL as NEW_PROTOCOL, TRAINING_CONTRACT
from real_motion.v21_source_induction import stable_json_fingerprint


def safe_summary_metrics(current, baseline):
    # JSON turns absent per-class IoUs into None. Only the frozen aggregate /
    # per-horizon gate enters safety; never call the in-memory delta on nulls.
    metrics = ('mIoU', 'IoU', 'MovingMacro', 'MovingMicro')
    rows = [(current, baseline)]
    if set(current['per_horizon']) != set(baseline['per_horizon']): raise RuntimeError('horizon population mismatch')
    rows += [(r, baseline['per_horizon'][h]) for h, r in current['per_horizon'].items()]
    return all(math.isfinite(float(a[k])) and math.isfinite(float(b[k]))
        and float(a[k])-float(b[k]) >= -1e-10 for a, b in rows for k in metrics)


def compare(local, adaptive):
    if (local['protocol'] != PROTOCOL or adaptive['protocol'] != NEW_PROTOCOL
            or local['training_contract'] != CONTRACT or adaptive['training_contract'] != TRAINING_CONTRACT):
        raise RuntimeError('paired method protocol mismatch')
    for key in ('seed', 'train_keys', 'calibration_keys', 'dev_keys', 'dev_manifest_fingerprint',
                'runtime_config_fingerprint', 'reference_checkpoint_sha256', 'info_fingerprints',
                'mode', 'target_updates', 'executed_windows', 'training_window_passes', 'train_windows', 'patch_resolution_m'):
        if stable_json_fingerprint(local[key]) != stable_json_fingerprint(adaptive[key]):
            raise RuntimeError(f'paired population/budget mismatch: {key}')
    if local['model_configs']['motion'] != adaptive['model_configs']['motion'] or local['model_configs']['columns'] != adaptive['model_configs']['columns']:
        raise RuntimeError('paired base architecture mismatch')
    if local['successful_updates'] != adaptive['successful_updates']:
        raise RuntimeError('different successful updates; not a matched training comparison')
    rows = {}; lines = ['===== LOCAL vs ADAPTIVE REFINE: MATCHED BUDGET =====',
        f"mode={local['mode']} train_windows={local['train_windows']} window_passes={local['training_window_passes']} updates={local['successful_updates']}",
        'TRAIN-only calibration is independent in each arm; no dev threshold selection.',
        'Generation readout/candidates/actions/loss unchanged; jointly learned transport can differ.']
    for population in ('dev64', 'all'):
        if population not in local['evaluation']: continue
        a, b = local['evaluation'][population], adaptive['evaluation'][population]
        if (a['windows'], a['scenes']) != (b['windows'], b['scenes']): raise RuntimeError('eval population mismatch')
        metrics = ('mIoU', 'IoU', 'MovingMicro')
        baseline = {k: b['baseline'][k]-a['baseline'][k] for k in metrics}
        joint = {k: b['variants']['joint']['metrics'][k]-a['variants']['joint']['metrics'][k] for k in metrics}
        refine = {k: b['variants']['refine']['delta_vs_v18_pp'][k]-a['variants']['refine']['delta_vs_v18_pp'][k] for k in metrics}
        quality = b['variants']['joint']['quality']; edits = quality.get('added', 0)+quality.get('removed', 0)
        row = dict(windows=a['windows'], adaptive_minus_local_joint_pp=joint,
            adaptive_minus_local_transport_pp=baseline, refine_gain_difference_pp=refine,
            adaptive_edits=edits, local_thresholds=local['thresholds'], adaptive_thresholds=adaptive['thresholds'],
            adaptive_gate=adaptive['gates'][population], local_gate=local['gates'][population])
        row['adaptive_safe_vs_local_joint'] = safe_summary_metrics(b['variants']['joint']['metrics'], a['variants']['joint']['metrics'])
        horizons = a['variants']['joint']['metrics'].get('per_horizon', {})
        row['joint_per_horizon_difference_pp'] = {h: {k:
            b['variants']['joint']['metrics']['per_horizon'][h][k]-m[k] for k in metrics} for h, m in horizons.items()}
        fixed = {}
        for name in ('diagnostic_joint', 'diagnostic_refine'):
            if name not in a['variants'] or name not in b['variants']: continue
            fixed[name] = {k: b['variants'][name]['metrics'][k]-a['variants'][name]['metrics'][k] for k in metrics}
        row['fixed_0p5_difference_pp'] = fixed
        rows[population] = row
        lines += [f"\n{population} windows={a['windows']}",
            f"adaptive-local JOINT: dMiOU={joint['mIoU']:+.6f} dIoU={joint['IoU']:+.6f} dMovingMicro={joint['MovingMicro']:+.6f}",
            f"transport difference: dMiOU={baseline['mIoU']:+.6f}; refine-only gain difference={refine['mIoU']:+.6f}",
            f"adaptive_edits={edits}; adaptive_gate={adaptive['gates'][population]}"]
        lines += [f"{h}: joint adaptive-local dMiOU={d['mIoU']:+.6f} dIoU={d['IoU']:+.6f} dMovingMicro={d['MovingMicro']:+.6f}"
            for h, d in row['joint_per_horizon_difference_pp'].items()]
        lines += [f"fixed0.5 {name}: adaptive-local dMiOU={d['mIoU']:+.6f} dIoU={d['IoU']:+.6f}" for name, d in fixed.items()]
    active = rows['all']['adaptive_edits'] > 0
    improved = rows['all']['adaptive_minus_local_joint_pp']['mIoU'] > 0
    passed = (adaptive['mode'] == 'screen' and adaptive['screen_pass'] and active and improved
        and all(r['adaptive_safe_vs_local_joint'] for r in rows.values()))
    route = 'adaptive_screen_improved_and_safe_not_auto_promoted' if passed else 'no_validated_improvement_no_automatic_retry'
    timing = {name: {'elapsed_seconds': s['elapsed_seconds'], 'stage_seconds': s['stage_seconds'],
        'column_parameters': s.get('column_parameters'), 'geometry_cache': s.get('geometry_cache')}
        for name, s in (('local', local), ('adaptive', adaptive))}
    lines += [f'\nroute: {route}', 'Timing note: local warms the fixed geometry cache; total runtime is not a fair neural speed comparison.',
        'This single window-pass screen does NOT establish convergence or full-data superiority.']
    return dict(populations=rows, timing=timing, route=route, pass_gate=passed), '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local', required=True); p.add_argument('--adaptive', required=True); p.add_argument('--out-dir', required=True)
    args = p.parse_args(); out = Path(args.out_dir)
    if out.exists(): p.error('NEW report directory required')
    local, adaptive = (json.loads(Path(x).read_text(encoding='utf-8')) for x in (args.local, args.adaptive))
    report, text = compare(local, adaptive)
    out.mkdir(parents=True); (out/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    combined = text+'\n===== LOCAL ORIGINAL SUMMARY =====\n'+Path(args.local).with_suffix('.txt').read_text(encoding='utf-8')
    combined += '\n===== ADAPTIVE ORIGINAL SUMMARY =====\n'+Path(args.adaptive).with_suffix('.txt').read_text(encoding='utf-8')
    (out/'combined_summary.txt').write_text(combined, encoding='utf-8'); print(text)


if __name__ == '__main__': main()
