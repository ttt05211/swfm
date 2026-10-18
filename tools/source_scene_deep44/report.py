"""Reuse completed integrated TWO+TWO; report joint encoder/decoder depth change."""
import json
from pathlib import Path
from tools.source_scene_stage2.report import MATCH_FIELDS


def reference(path):
    from tools.source_scene_integrated.run import PROTOCOL
    v = json.loads((Path(path)/'results.json').read_text(encoding='utf-8')); c = v['contract']
    if (v['status'] != 'complete' or c['protocol'] != PROTOCOL or c['epochs'] != 1 or
            c['model']['variant'] != 'source_scene_integrated_AdaGN' or c['model']['kta_conditioning'] is not True or
            c['model']['encoder']['blocks'] != 2 or c['model']['motion']['decoder_blocks'] != 2 or
            c['model']['reader']['layers'] != 2 or c['model']['reader']['temporal_layers'] != 2 or
            c['registration'] != 'none' or c['source_policy'] != 'shared_pinned' or
            c['known'] != 'TRAIN_OCC_STC_dense_valid' or v['updates'] != c['total_updates'] or
            v['evaluations']['1']['windows'] != 64 or v['evaluations']['1']['keys'] != c['eval_keys']):
        raise RuntimeError('completed matched RANDOM TRAIN20% x1 integrated TWO+TWO control required')
    return v


def select_reference(root, explicit=None):
    if explicit:
        reference(explicit); return Path(explicit).resolve()
    candidates = []
    for path in Path(root).glob('scene_integrated_1ep_*/results.json'):
        try: reference(path.parent)
        except (RuntimeError, KeyError, ValueError): continue
        candidates.append(path.parent.resolve())
    if len(candidates) != 1:
        raise RuntimeError('Set INTEGRATED_DIR to the completed TRAIN20%x1 integrated TWO+TWO; candidates='+str(candidates))
    return candidates[0]


def compare(new, control):
    if new['status'] != 'complete': return 'Integrated run incomplete; no final comparison.\n'
    for name in MATCH_FIELDS:
        if new['contract'][name] != control['contract'][name]: raise RuntimeError('unmatched integrated TWO+TWO control: '+name)
    lines = ['===== MATCHED INTEGRATED TWO+TWO / FOUR+FOUR =====',
        'RANDOM TRAIN4086 x1; common fresh weights; SAME order/B4-source128/H6 CE+Lovasz/dense-valid/GT ego.',
        'integrated TWO+TWO results reused, NOT retrained. No ICP/old trained weights/extra loss imported.',
        'ONLY encoder2->4 AND decoder2->4; SAME AdaGN/width/geometry. JOINT depth, NOT equal-parameter/compute or a one-module ablation.',
        'model      route                         mIoU       IoU MovingMicro']
    for name, item in (('two_two', control), ('four_four', new)):
        e = item['evaluations']['1']
        if e['windows'] != 64 or e['keys'] != new['contract']['eval_keys']: raise RuntimeError('same64 required')
        for route, m in e['variants'].items():
            lines.append(f'{name:10s} {route:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}')
    for route, old in control['evaluations']['1']['variants'].items():
        now = new['evaluations']['1']['variants'][route]
        lines.append(route+' four_four_minus_two_two='+json.dumps({k: now[k]-old[k] for k in ('mIoU', 'IoU', 'MovingMicro')}))
    lines.append('parameters='+json.dumps({name: item['contract']['parameters'] for name, item in (('two_two', control), ('four_four', new))}))
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
