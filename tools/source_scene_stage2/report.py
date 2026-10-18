"""Reuse completed matched stage1 Reader metrics; NEVER retrain control."""
import json
from pathlib import Path

MATCH_FIELDS = ('train_keys', 'eval_keys', 'epochs', 'total_updates', 'epochs_updates', 'seed',
                'learning_rates', 'schedule', 'objective', 'known', 'precision', 'world_size',
                'data', 'overrides', 'STC_inventory')


def reference(path):
    from tools.source_scene_stage1.run import PROTOCOL
    p = Path(path)/'results.json'; v = json.loads(p.read_text(encoding='utf-8')); c = v['contract']
    if (v['status'] != 'complete' or c['protocol'] != PROTOCOL or c['arm'] != 'reader'
            or c['epochs'] != 1 or c['source_policy'] != 'shared_pinned'
            or c['known'] != 'TRAIN_OCC_STC_dense_valid' or v['updates'] != c['total_updates']):
        raise RuntimeError('completed matched stage1 RANDOM Reader one-epoch control required')
    return v


def compare(new, control):
    if new['status'] != 'complete': return 'Stage2 incomplete; no final comparison.\n'
    for key in MATCH_FIELDS:
        if new['contract'][key] != control['contract'][key]: raise RuntimeError('unmatched control: '+key)
    rows = []; lines = ['===== STAGE1 READER / STAGE2 DIRECT HISTORY =====',
        'SAME RANDOM TRAIN20% x1/global B4/H6 CE+Lovasz/remaining64/dense-valid/GT ego.',
        'Stage1 results reused, NOT retrained. No old trained weights imported into stage2.',
        'Stage2 JOINT change: remove KTA query + ICP/gate, track-valid + learned relative history yaw.',
        'Not a single-variable ablation; no automatic promotion.',
        'model        route                         mIoU       IoU MovingMicro']
    for name, result in (('stage1_reader', control), ('stage2_direct', new)):
        e = result['evaluations']['1']
        if e['windows'] != 64 or e['keys'] != new['contract']['eval_keys']: raise RuntimeError('eval population mismatch')
        for route, m in e['variants'].items():
            rows.append(dict(model=name, route=route, metrics=m))
            lines.append(f'{name:13s} {route:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}')
    for route, old in control['evaluations']['1']['variants'].items():
        now = new['evaluations']['1']['variants'][route]
        lines.append(route+' stage2_minus_reader='+json.dumps({k: now[k]-old[k] for k in ('mIoU', 'IoU', 'MovingMicro')}))
    return '\n'.join(lines)+'\n'
