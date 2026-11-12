"""Build the three OCC ablation tables from finished window counts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from real_motion.waymo_i2world import SHAPE
from tools.final_semantic_evaluation.metrics import report
from .workflow import TABLE_ORDER

TABLES = (
    ('主组件', (('A 零物体运动', 'a_static'), ('B 仅 KTA', 'b_kta'), ('C KTA＋STWM', 'c_transport'),
              ('D KTA＋Refiner', 'd_kta_refiner'), ('E 完整模型', 'epoch2'))),
    ('Memory', (('M0 无 Memory', 'm0_no_memory'), ('M1 自车对齐', 'm1_ego_memory'),
                ('M2 完整 Memory', 'epoch2'))),
    ('多尺度与 Ego 条件', (('池化＋AdaLN＋GT', 's0_pool'), ('FPN＋Query＋GT', 'query'),
                        ('FPN＋AdaLN＋GT', 'epoch2'), ('FPN＋AdaLN＋Pred', 'pred_ego'))),
)


def read_row(path, arm, population_sha, windows):
    state = json.loads(path.read_text(encoding='utf-8'))
    contract = state['contract']
    if (contract.get('arm') != arm or contract.get('input') != 'occ'
            or contract.get('population_sha256') != population_sha or contract.get('windows') != windows):
        raise ValueError(f'{arm}: OCC evaluation state does not match this population')
    completed = state['completed_windows']
    if completed != windows:
        raise ValueError(f'{arm}: only {completed}/{windows} OCC windows saved')
    confusion = np.asarray(state['confusion'])
    if (confusion.shape != (3, 18, 18) or confusion.dtype.kind not in 'iu'
            or np.any(confusion < 0) or np.any(confusion.sum((1, 2)) != windows * int(np.prod(SHAPE)))):
        raise ValueError(f'{arm}: invalid full-grid confusion counts')
    scores = report(dict(confusion=confusion), (1., 2., 3.), moving=False)['average']
    return windows, scores['mIoU'], scores['IoU']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args(argv)
    data = (args.out_dir/'population.json').read_bytes()
    population = json.loads(data)
    digest = hashlib.sha256(data).hexdigest()
    keys = {job[2] for job in population['jobs']}
    covered = set(population['predicted'])
    if len(population['jobs']) != 4369 or len(keys) != 4369 or not covered or not covered <= keys:
        raise ValueError('Expected full VAL4369 and its planner-covered subset')

    rows = {arm: read_row(args.out_dir/arm/'occ_state.json', arm, digest,
                          len(covered) if arm == 'pred_ego' else len(keys)) for arm in TABLE_ORDER}
    rows['epoch2'] = (4369, 48.711339, 62.809965)
    rows['query'] = (4369, 48.513241, 62.593428)
    for title, configurations in TABLES:
        print(f'## {title}\n')
        print('| 配置 | 窗数 | mIoU | IoU |')
        print('|---|---:|---:|---:|')
        for name, arm in configurations:
            windows, miou, iou = rows[arm]
            print(f'| {name} | {windows} | {miou:.3f} | {iou:.3f} |')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
