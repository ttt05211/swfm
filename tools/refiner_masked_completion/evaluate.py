"""Paired clean and artificial-cut diagnostics, with fixed DEV64 keys."""
import argparse
import json
from pathlib import Path
import signal
import threading

import numpy as np
import torch

from real_motion.runtime_config import load_runtime_config, make_prepare_config
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.coverage_projection.eval_inputs import prepare_evaluation_row
from tools.coverage_projection.evaluate import evaluation_plan
from tools.coverage_projection.run import ROOT, sha, write_json
from tools.final_semantic_evaluation import metrics
from tools.final_semantic_evaluation.sources import window
from tools.fpn_unet_eval.common import DenseInputs
from tools.joint_occ_experiments.run import load_cache
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock

from .adapter import forward_window
from .masking import depth_for, regions
from .train import PROTOCOL, model_from_spec, verify_implementation

AT = (1, 3, 5)
MODELS = ('original', 'masked_training')
REGIONS = ('full_grid', 'natural_front_3s', 'artificial_cut_3s')
CLASSES = {'road': 11, 'sidewalk': 13, 'terrain': 14, 'manmade': 15, 'vegetation': 16}


def compare(counts, predictions, target, masks):
    """Original and augmented training use the same regions and target labels."""
    if np.any(target > 17) or np.any(target < 0):
        raise ValueError('dense-valid DEV targets must be full 18-class labels')
    for name, (clean, cut) in predictions.items():
        for slot, h in enumerate(AT):
            code = 18*target[slot].astype(np.int64) + clean[h]
            counts[name]['full_grid'][slot] += np.bincount(code.ravel(), minlength=324).reshape(18, 18)
        for region, pred in (('natural_front_3s', clean), ('artificial_cut_3s', cut)):
            mask = masks[region]
            code = 18*target[-1][mask].astype(np.int64) + pred[5][mask]
            counts[name][region][0] += np.bincount(code, minlength=324).reshape(18, 18)


def describe(counts):
    result = {'full_grid': metrics.report(dict(confusion=counts['full_grid']), (1., 2., 3.), moving=False)}
    for region in REGIONS[1:]:
        c = counts[region][0]
        report = metrics.report(dict(confusion=c[None]), (3.,), moving=False)['per_horizon']['3.0']
        report['voxels'] = int(c.sum())
        report['pred_occupied_on_gt_free'] = int(c[17, :17].sum())
        report['gt_free_voxels'] = int(c[17, :].sum())
        report['classes'] = {name: dict(iou=report['semantic_per_class'][str(i)],
                                        tp=int(c[i, i]), fp=int(c[:, i].sum()-c[i, i]),
                                        fn=int(c[i].sum()-c[i, i])) for name, i in CLASSES.items()}
        result[region] = report
    return result


def render(path, target, predictions, masks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.coverage_projection.visuals import COLORS

    fig, grid = plt.subplots(2, 3, figsize=(10, 6), squeeze=False)
    for i, region in enumerate(REGIONS[1:]):
        panels = (target[-1], predictions[MODELS[0]][i][5], predictions[MODELS[1]][i][5])
        for j, labels in enumerate(panels):
            ax = grid[i, j]
            occupied = (labels > 0) & (labels < 17)
            height = np.where(occupied, np.arange(labels.shape[-1]), -1).max(-1)
            surface = np.take_along_axis(labels, height.clip(0)[..., None], axis=-1)[..., 0]
            surface[height < 0] = 17
            ax.imshow(COLORS[surface].transpose(1, 0, 2), origin='lower', interpolation='nearest')
            area = masks[region].any(-1).T
            if area.any() and not area.all():
                ax.contour(area, levels=[.5], colors=['#009990'], linewidths=.65)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(('GT', 'Original training', 'Masked training')[j])
            if j == 0:
                ax.set_ylabel(('Natural front / clean inputs', 'Artificial cut')[i])
    fig.suptitle('Fixed DEV case at 3 s')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def run(a, stop):
    torch.set_num_threads(1)
    candidate = a.out_dir/'epoch_0001.pt'
    saved = torch.load(candidate, map_location='cpu', weights_only=False)
    c = saved['contract']
    if (saved['protocol'] != PROTOCOL or c['protocol'] != PROTOCOL
            or saved['updates'] != 1031 or c['total_updates'] != 1031):
        raise ValueError('completed 1031-update masked training epoch required')
    verify_implementation(c['implementations'])
    verify_implementation((c['model'].get('experiment') or {}).get('candidate_implementation', {}))
    control = Path(c['control'])
    if sha(control) != c['control_sha']:
        raise ValueError('control checkpoint changed')
    old = torch.load(control, map_location='cpu', weights_only=False)
    source = old['contract']
    if source['model'] != c['model'] or source['eval_keys'] != c['eval_keys'] or old['updates'] != 1031:
        raise ValueError('paired control differs from matched architecture or DEV population')
    if (sha(a.config) != c['data']['config'] or sha(a.dev_cache) != source['data']['dev_cache']
            or sha(a.dev_info) != source['data']['dev_info']):
        raise ValueError('VAL metadata/config differs from control')
    selected = evaluation_plan(load_cache(a.dev_cache)[1], c['eval_keys'])
    pcfg = make_prepare_config(load_runtime_config(a.config, c['overrides']))
    device = torch.device(a.device)
    models = {}
    for name, checkpoint in zip(MODELS, (old, saved)):
        model = model_from_spec(c['model'])
        model.load_state_dict(checkpoint['models']['gt'], strict=True)
        models[name] = model.eval().requires_grad_(False).to(device)
    del old, saved
    inputs = DenseInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=128)
    inventory = {kind: inputs.preflight([window(record) for record in selected], kind)
                 for kind in ('occ', 'stc')}
    receipt = dict(protocol=PROTOCOL, candidate_sha=sha(candidate), control_sha=c['control_sha'],
                   keys=c['eval_keys'], inventory=inventory, horizons=[1, 2, 3],
                   rules=dict(natural_front='x>=0 in future ego and outside all four historical XY grids',
                              artificial_cut='in original t0 XYZ grid and beyond synthetic cutoff'),
                   training_world_sizes=[c['control_world_size'], c['world_size']])
    state_path = a.out_dir/'evaluation_state.json'
    state = (json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists()
             else dict(receipt=receipt, completed={}, counts={}))
    if state['receipt'] != receipt:
        raise ValueError('evaluation resume sources differ')
    predictor = RecoveryPredictor(models['masked_training'], pcfg, device, workers=a.cpu_workers)
    visual = a.out_dir/'visuals'
    visual.mkdir(exist_ok=True)
    reports = {}
    try:
        for kind in ('occ', 'stc'):
            done = state['completed'].get(kind, 0)
            if not 0 <= done <= len(selected):
                raise ValueError('invalid completed windows')
            counts = {name: {region: np.zeros((3 if region == 'full_grid' else 1, 18, 18), np.int64)
                             for region in REGIONS} for name in MODELS}
            if done:
                counts = {name: {key: np.asarray(value, np.int64) for key, value in part.items()}
                          for name, part in state['counts'][kind].items()}
            for index in range(done, len(selected)):
                if stop.is_set():
                    raise InterruptedError('evaluation stopped at the last complete window')
                record = selected[index]
                row, poses, _ = prepare_evaluation_row(inputs, predictor, record, kind=kind)
                depth = depth_for(record, c['seed'], force=True)
                masks = regions(pcfg.grid, row[1]['history_poses'], poses[5], depth)
                predictions = {}
                for name, model in models.items():
                    pair = []
                    for cut in (None, depth):
                        fw = forward_window(model, predictor.provider, row, poses=poses, depth=cut)
                        pair.append(fw['logits'].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy())
                        del fw
                    predictions[name] = pair
                target = np.stack(inputs.targets(window(record), AT))
                compare(counts, predictions, target, masks)
                if index < 4:
                    render(visual/f'{kind}_{index+1:02d}_3s.png', target, predictions, masks)
                state['completed'][kind] = index+1
                state['counts'][kind] = {name: {k: v.tolist() for k, v in part.items()}
                                         for name, part in counts.items()}
                write_json(state_path, state)
                print(f'MASKED_REFINER_EVAL {kind} {index+1}/{len(selected)}', flush=True)
            reports[kind] = {name: describe(part) for name, part in counts.items()}
    finally:
        predictor.close()
    write_json(a.out_dir/'evaluation.json', dict(status='complete', receipt=receipt, reports=reports))
    lines = ['model input region mIoU IoU RoadIoU SidewalkIoU FalsePositiveOnGTFree']
    for kind, by_model in reports.items():
        for name, by_region in by_model.items():
            for region, score in by_region.items():
                view = score['average'] if region == 'full_grid' else score
                road = view.get('classes', {}).get('road', {}).get('iou')
                side = view.get('classes', {}).get('sidewalk', {}).get('iou')
                lines.append(f'{name} {kind} {region} {view["mIoU"]} {view["IoU"]} {road} {side} '
                             f'{view.get("pred_occupied_on_gt_free")}')
    (a.out_dir/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-dir', 'dataroot', 'dev-info', 'dev-cache', 'stc-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/real_motion_occfm.yaml')
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    lock = args.out_dir/'masked_eval_lock'
    lock.mkdir(exist_ok=True)
    with evaluation_lock(lock):
        run(args, stop)


if __name__ == '__main__':
    main()
