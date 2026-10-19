"""Only the completed RANDOM TRAIN20%x1 Integrated TWO+TWO is a matched control."""
import json
from pathlib import Path
from tools.source_scene_stage2.report import MATCH_FIELDS


def reference(path):
    from tools.source_scene_integrated.run import PROTOCOL
    v = json.loads((Path(path)/'results.json').read_text(encoding='utf-8')); c = v['contract']; s = c['model']
    if (v.get('status') != 'complete' or c.get('protocol') != PROTOCOL or c.get('epochs') != 1 or
            s.get('variant') != 'source_scene_integrated_AdaGN' or s.get('kta_conditioning') is not True or
            s['encoder']['blocks'] != 2 or s['motion']['decoder_blocks'] != 2 or
            s['repair'].get('conditioning') != 'residual_branch_AdaGN_zero_projection' or
            c.get('registration') != 'none' or c.get('source_policy') != 'shared_pinned' or
            c.get('known') != 'TRAIN_OCC_STC_dense_valid' or v.get('updates') != c['total_updates'] or
            v['evaluations']['1']['windows'] != 64 or v['evaluations']['1']['keys'] != c['eval_keys']):
        raise RuntimeError('completed matched RANDOM TRAIN20%x1 Integrated TWO+TWO control required; NOT full epoch2/deep44/old Stage2')
    return v


def select_reference(root, explicit=None):
    if explicit:
        reference(explicit); return Path(explicit).resolve()
    candidates = []
    for p in Path(root).glob('scene_integrated_1ep_*/results.json'):
        try: reference(p.parent)
        except (RuntimeError, ValueError, KeyError): continue
        candidates.append(p.parent.resolve())
    if len(candidates) != 1:
        raise RuntimeError('Set INTEGRATED_PILOT_CONTROL_DIR to completed latest Integrated2+2 TRAIN20%x1; candidates='+str(candidates))
    return candidates[0]


def compare(new, control):
    if new['status'] != 'complete': return 'Three-change candidate incomplete; no final comparison.\n'
    for k in MATCH_FIELDS:
        if new['contract'][k] != control['contract'][k]: raise RuntimeError('unmatched Integrated control: '+k)
    lines = ['===== MATCHED INTEGRATED2+2 / EGO QUERY + HISTORY MEMORY + CROSS-FIRST =====',
        'RANDOM TRAIN4086 x1; SAME fresh common weights/order/globalB4-source128/H6/full6 CE+Lovasz/dense-valid.',
        'Completed Integrated2+2 control reused, NOT retrained; NO trained weights imported.',
        'Candidate future GT ego conditions Decoder AND renderer; control uses it for rendering only.',
        'THREE JOINT changes, NOT single-variable/equal-parameter or proof of each component.',
        'Same remaining64/full-grid; no best-route/epoch selection. No navigation/ego head/extra supervision.',
        'model        route                         mIoU       IoU MovingMicro']
    for name, v in (('Integrated22', control), ('ego_pure_cross', new)):
        e = v['evaluations']['1']
        if e['windows'] != 64 or e['keys'] != new['contract']['eval_keys']: raise RuntimeError('same ordered64 required')
        for route, m in e['variants'].items():
            lines.append(f'{name:14s} {route:28s} {m["mIoU"]:.6f} {m["IoU"]:.6f} {m["MovingMicro"]:.6f}')
    for route, m in new['evaluations']['1']['variants'].items():
        old = control['evaluations']['1']['variants'][route]
        lines.append(route+' new_minus_control='+json.dumps({k:m[k]-old[k] for k in ('mIoU','IoU','MovingMicro')}))
    lines.append('parameters='+json.dumps({k:v['contract']['parameters'] for k,v in (('control',control),('candidate',new))}))
    e = new['evaluations']['1']
    for route, kta in e.get('kta_variants', {}).items():
        now = e['variants'][route]
        lines.append(route+' learned_minus_KTA='+json.dumps({k:now[k]-kta[k] for k in ('mIoU','IoU','MovingMicro')}))
    for kind, motion in e.get('source_motion', {}).items():
        a = motion['all']
        lines.append(f'{kind} XY_correction_m={a["XY_correction_m"]}; future_yaw_deg={a["future_yaw_abs_deg"]}')
        for group in ('moving','slow_or_stationary'):
            s = motion['groups'][group]
            lines.append(f'{kind}/{group}: XY_error learned/KTA={s["learned_error_m"]}/{s["KTA_error_m"]}; helped/harmed={s["XY_helped"]}/{s["XY_harmed"]}')
    lines += ['GT-box diagnostics are conditional; GT-ego conditioned forecast, NOT navigation-free or predicted-ego proof.',
        'Fixed development experiment, NOT independent test or FPS; no automatic extension/promotion.']
    return '\n'.join(lines)+'\n'
