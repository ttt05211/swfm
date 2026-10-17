"""Fixed BOTH-epoch paired report, never select or promote an arm."""
import argparse
import json
from pathlib import Path
from tools.joint_occ_experiments.run import write_json
from .run import PROTOCOL


def compare(folder):
    root = Path(folder); results = {}
    for arm in ('baseline', 'reader'):
        result = json.loads((root/arm/'results.json').read_text(encoding='utf-8'))
        if result['status'] != 'complete' or result['contract']['protocol'] != PROTOCOL or result['contract']['arm'] != arm:
            raise RuntimeError('both complete stage1 runs required')
        results[arm] = result
    left, right = [results[a]['contract'] for a in ('baseline', 'reader')]
    for field in ('train_keys', 'eval_keys', 'epochs', 'total_updates', 'epochs_updates', 'seed', 'learning_rates',
                  'schedule', 'objective', 'known', 'source_policy', 'source_implementation', 'data', 'overrides',
                  'precision', 'world_size', 'STC_inventory', 'implementation'):
        if left[field] != right[field]: raise RuntimeError('unmatched scientific contract: '+field)
    rows = []; lines = ['===== MATCHED RANDOM BASELINE / SCENE READER =====',
        'TRAIN20% x2; same latest shared_pinned / dense-valid / globalB4-source128 / H6 CE+Lovasz.',
        'Same fresh common modules, NOT old checkpoints. Different motion architecture/parameter count.',
        'SAME fixed remaining64; both epochs reported. NO best-head/epoch selection.',
        'epoch arm      route                         mIoU       IoU MovingMicro']
    for epoch in ('1', '2'):
        for arm in ('baseline', 'reader'):
            value = results[arm]['evaluations'][epoch]
            if value['windows'] != 64 or value['keys'] != left['eval_keys']: raise RuntimeError('evaluation keys differ')
            for route, m in value['variants'].items():
                rows.append(dict(epoch=int(epoch), arm=arm, route=route, metrics=m))
                lines.append(f'{epoch:>5s} {arm:8s} {route:28s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}')
        for route in results['baseline']['evaluations'][epoch]['variants']:
            a = results['baseline']['evaluations'][epoch]['variants'][route]
            b = results['reader']['evaluations'][epoch]['variants'][route]
            lines.append(f'epoch{epoch} {route} reader_minus_baseline='+json.dumps({k: b[k]-a[k] for k in ('mIoU', 'IoU', 'MovingMicro')}))
    lines.append('Development population, not an independent test. No automatic promotion/extension. NOT FPS.')
    output = dict(status='complete', rows=rows, selection='none', source_results={a: str(root/a/'results.json') for a in results})
    write_json(root/'comparison.json', output)
    (root/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True)
    print(compare(p.parse_args().root))
