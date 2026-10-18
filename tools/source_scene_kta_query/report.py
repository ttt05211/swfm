"""Reuse the completed one-epoch no-query Stage2, not any full-data result."""
import json
from pathlib import Path
from tools.source_scene_stage2.report import MATCH_FIELDS


def reference(path):
    from tools.source_scene_stage2.run import PROTOCOL
    v = json.loads((Path(path)/'results.json').read_text(encoding='utf-8')); c = v['contract']
    if (v['status'] != 'complete' or c['protocol'] != PROTOCOL or c['epochs'] != 1 or
            c['model']['variant'] != 'source_scene_stage2' or c['model']['kta_conditioning'] is not False or
            c['registration'] != 'none' or c['source_policy'] != 'shared_pinned' or
            c['known'] != 'TRAIN_OCC_STC_dense_valid' or v['updates'] != c['total_updates'] or
            v['evaluations']['1']['windows'] != 64 or v['evaluations']['1']['keys'] != c['eval_keys']):
        raise RuntimeError('completed matched RANDOM TRAIN20% x1 no-query Stage2 control required')
    return v


def select_reference(root, explicit=None):
    if explicit:
        reference(explicit); return Path(explicit).resolve()
    candidates = []
    for path in Path(root).glob('scene_stage2_1ep_*/results.json'):
        try: reference(path.parent)
        except (RuntimeError, KeyError, ValueError): continue
        candidates.append(path.parent.resolve())
    if len(candidates) != 1:
        raise RuntimeError('Set SCENE_NO_QUERY_DIR to the completed TRAIN20%x1 Stage2; candidates='+str(candidates))
    return candidates[0]


def compare(new, control):
    if new['status'] != 'complete': return 'KTA-query control incomplete; no final comparison.\n'
    for name in MATCH_FIELDS:
        if new['contract'][name] != control['contract'][name]: raise RuntimeError('unmatched no-query control: '+name)
    lines = ['===== MATCHED STAGE2 / KTA QUERY ONLY =====',
        'RANDOM TRAIN4086 x1; common fresh weights; SAME order/B4-source128/H6 CE+Lovasz/dense-valid/GT ego.',
        'No-query results reused, NOT retrained. No ICP/old trained weights/extra loss imported.',
        'Only new KTA-to-query projection (+640 parameters at d128); not equal parameter count.',
        'model      route                         mIoU       IoU MovingMicro']
    for name, item in (('no_query', control), ('KTA_query', new)):
        e = item['evaluations']['1']
        if e['windows'] != 64 or e['keys'] != new['contract']['eval_keys']: raise RuntimeError('same64 required')
        for route, m in e['variants'].items():
            lines.append(f'{name:10s} {route:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}')
    for route, old in control['evaluations']['1']['variants'].items():
        now = new['evaluations']['1']['variants'][route]
        lines.append(route+' query_minus_no_query='+json.dumps({k: now[k]-old[k] for k in ('mIoU', 'IoU', 'MovingMicro')}))
    e = new['evaluations']['1']
    for route, kta in e.get('kta_variants', {}).items():
        now = e['variants'][route]
        lines.append(route+' learned_minus_KTA='+json.dumps({k: now[k]-kta[k] for k in ('mIoU', 'IoU', 'MovingMicro')}))
    for kind, motion in e.get('source_motion', {}).items():
        a = motion['all']
        lines.append(f'{kind} XY_correction_m={a["XY_correction_m"]}; future_yaw_deg={a["future_yaw_abs_deg"]}')
        for group in ('moving', 'slow_or_stationary'):
            s = motion['groups'][group]
            lines.append(f'{kind}/{group}: XY_error learned/KTA={s["learned_error_m"]}/{s["KTA_error_m"]}; helped/harmed={s["XY_helped"]}/{s["XY_harmed"]}')
    lines += ['GT-box errors are conditional source-displacement diagnostics, not perfect point-motion truth.',
              'Fixed one epoch, no DEV-best selection/extension/promotion. NOT FPS or full-data proof.']
    return '\n'.join(lines)+'\n'
