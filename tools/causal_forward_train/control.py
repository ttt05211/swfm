"""Resolve ONE completed AdaLN pilot, never select weights by DEV score/date."""
import argparse
import hashlib
import json
from pathlib import Path

CONTROL_PROTOCOL = 'matched_FPN_UNet_AdaLN_Zero_RANDOM_TRAIN20x1_GT_ego_H6_v1'
NAMES = tuple(k+'_'+r for k in ('occ_gt','stc_gt') for r in ('transport','semantic_refine'))


def reference(folder):
    path = Path(folder).resolve()/'results.json'
    value = json.loads(path.read_text(encoding='utf-8')); c = value['contract']
    e = value.get('evaluations',{}).get('1',{})
    if (value.get('status') != 'complete' or value.get('epoch') != 1 or c.get('protocol') != CONTROL_PROTOCOL
            or value.get('updates') != c.get('total_updates') or c.get('epochs') != 1
            or c.get('model',{}).get('variant') != 'FPN_UNet_ego_AdaLN_Zero_cross_self_FFN_v1'
            or c.get('model',{}).get('source_policy') != 'shared_pinned'
            or c.get('source_policy') != 'shared_pinned' or c.get('registration') != 'none'
            or c.get('known') != 'TRAIN_OCC_STC_dense_valid'
            or c.get('objective') != dict(CE=1.,Lovasz=1.,ego=0.,motion=0.,existence=0.)
            or len(c.get('train_keys',[])) != 4086 or len(set(map(tuple,c['train_keys']))) != 4086
            or len(c.get('eval_keys',[])) != 64 or len(set(map(tuple,c['eval_keys']))) != 64
            or e.get('windows') != 64 or e.get('keys') != c['eval_keys'] or set(e.get('variants',{})) != set(NAMES)):
        raise RuntimeError('completed original-association AdaLN TRAIN4086x1 / same64 control required')
    return path,value


def resolve(root, explicit=None):
    if explicit:
        return reference(explicit)[0].parent
    matches = []
    for path in Path(root).glob('fpn_adaln_zero_20pct_1ep_*/results.json'):
        try: matches.append(reference(path.parent)[0].parent)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError):continue
    if len(matches) != 1:
        raise RuntimeError(f'found {len(matches)} completed AdaLN pilots; set FORWARD_CONTROL_DIR explicitly. '
            'No score/date-based selection: '+str(matches))
    return matches[0]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True); p.add_argument('--control-dir')
    a = p.parse_args(); print(resolve(a.root,a.control_dir))
