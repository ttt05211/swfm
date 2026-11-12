"""Score one frozen row on OCC using complete-window integer counts."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import threading

import numpy as np
import torch

from real_motion.shared_source_track import kta_and_anchors
from tools.causal_forward_all.common import validate_source
from tools.causal_forward_full.model import checkpoint_model
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.final_semantic_evaluation import metrics
from tools.joint_occ_experiments import run as base
from tools.joint_occ_experiments.soft_geometry import source_targets
from tools.paper_ablation_train import control, model
from tools.paper_ablation_train.geometry import forward_window
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.shared_history_experiments.refine.geometry import TransportPrior
from tools.shared_source_track.adapter import backgrounds
from .population import PreparedInputs, select_jobs

PROTOCOL = 'paper_ablation_OCC_VAL4369_eval_v2'
# The original worker used the same OCC predictions and integer counts.
LEGACY_OCC_CODE = {
    'tools/paper_ablation_eval/run.py': '738d683db63cfefa542e304a51e81698153566ab0d0ab990efa8b78532d32bab',
    'tools/paper_ablation_eval/population.py': 'c86015f5f7dc4f4b414e240ba48e0aa16d564c36d6a37acf65153db8614a2798',
}
GEOMETRY = ('a_static', 'b_kta')
HORIZONS = (1., 2., 3.)
INDICES = (1, 3, 5)


def load_body(arm, path, population, device='cuda:0'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if arm == 'pred_ego':
        contract = validate_source(saved)
        if contract['data']['config'] != population['files']['config']:
            raise RuntimeError('Pred checkpoint config differs from E evaluation')
        return checkpoint_model(saved, device), contract['train_keys']

    contract = saved['contract']
    budget = contract['budget']
    full = model.ARMS[arm]['fraction'] == 1.
    windows, updates, denominator = (20430, 5145, 10292) if full else (4086, 1031, 1031)
    if (saved.get('protocol') != control.PROTOCOL or contract.get('protocol') != control.PROTOCOL
            or saved.get('training_only') is not True or contract.get('arm') != arm
            or contract.get('recipe') != model.ARMS[arm] or contract['model'].get('arm') != arm
            or contract.get('world_size') != 2 or contract.get('known') != 'TRAIN_dense_valid'
            or (saved.get('epoch'), saved.get('batch'), saved.get('updates')) != (1, 0, updates)
            or (budget['train_windows'], budget['stop_updates'], budget['schedule_reference_updates'])
            != (windows, updates, denominator)
            or len(set(map(tuple, contract['train_keys']))) != windows
            or contract['data']['config'] != population['files']['config']
            or contract.get('overrides') != population['overrides']
            or set(saved['models']) != {'gt'}):
        raise RuntimeError(f'{arm}: completed matching one-epoch checkpoint required')
    body = model.from_spec(contract['model'])
    body.load_state_dict(saved['models']['gt'], strict=True)
    return body.to(device).eval().requires_grad_(False), contract['train_keys']


def geometry_prediction(provider, row, poses, stationary):
    record, raw, _, history = row
    state = history[0]['state']
    track = raw['_shared_source_track']
    if stationary:
        track = replace(track, pair_velocity={})
        _, _, anchors = kta_and_anchors(state['current'], track, state['current_pose'])
        record = dict(record, anchors_xy_t0_m=anchors)
    device = provider.device
    poses = torch.as_tensor(poses, dtype=torch.float64, device=device)
    state = dict(state, column_backgrounds=backgrounds(raw, state, poses, provider, track))
    actors = len(state['current'])
    residual = dict(residual_xy_m=poses.new_zeros((actors, 6, 2)))
    targets = source_targets(residual, record, state['current_pose'], state['source_z_t0'])
    renderer = TransportPrior(raw, state, provider.pcfg.grid, device)
    prior = renderer.render(poses, targets, poses.new_zeros((actors, 6)))
    return prior, renderer.shape


def predict(arm, predictor, record, raw, times, poses):
    row = predictor.prepare(record, raw, times=times)
    if arm in GEOMETRY:
        logits, shape = geometry_prediction(predictor.provider, row, poses, stationary=arm == 'a_static')
    else:
        forward = (predictor.infer(row, poses) if arm == 'pred_ego'
                   else forward_window(predictor.body, predictor.provider, row, poses=poses))
        logits, shape = forward['logits'], forward['shape']
    return logits.argmax(dim=1).reshape(6, *shape).byte().cpu().numpy()


def read_counts(path, contract, windows, voxels):
    if not path.is_file():
        return 0, np.zeros((3, 18, 18), dtype=np.int64)
    saved = json.loads(path.read_text(encoding='utf-8'))
    count = saved['completed_windows']
    confusion = np.asarray(saved['confusion'])
    accepted = saved['contract'] == contract
    if not accepted and contract.get('protocol') == PROTOCOL and contract.get('input') == 'occ':
        legacy = dict(contract, protocol='paper_ablation_VAL4369_eval_v1',
                      implementation={**contract['implementation'], **LEGACY_OCC_CODE})
        accepted = saved['contract'] == legacy
    if (not accepted or type(count) is not int or not 0 <= count <= windows
            or confusion.shape != (3, 18, 18) or confusion.dtype.kind not in 'iu'
            or np.any(confusion < 0) or np.any(confusion.sum((1, 2)) != count * voxels)):
        raise ValueError(f'Invalid or incompatible evaluation state: {path}')
    return count, confusion.astype(np.int64)


@torch.no_grad()
def evaluate(arm, predictor, inputs, jobs, path, contract, stop):
    voxels = int(np.prod(predictor.provider.pcfg.grid.shape_hwd))
    count, confusion = read_counts(path, contract, len(jobs), voxels)
    if count:
        print(f'ABLATION_EVAL {arm} occ {count}/{len(jobs)}', flush=True)

    def save():
        base.write_json(path, dict(contract=contract, completed_windows=count, confusion=confusion.tolist()))

    try:
        for job in jobs[count:]:
            if stop.is_set():
                raise InterruptedError('Stopping at a complete evaluation window')
            record, raw, times = inputs.history(job)
            poses = inputs.future_poses(job, predicted=arm == 'pred_ego')
            prediction = predict(arm, predictor, record, raw, times, poses)
            targets = inputs.targets(job, INDICES)
            if len(targets) != len(INDICES):
                raise ValueError('Expected targets at 1, 2 and 3 seconds')
            delta = dict(confusion=np.zeros_like(confusion))
            for index, (at, target) in enumerate(zip(INDICES, targets)):
                metrics.add(delta, index, prediction[at], target)
            confusion += delta['confusion']
            count += 1
            if count % 32 == 0 or count == len(jobs):
                save()
                print(f'ABLATION_EVAL {arm} occ {count}/{len(jobs)}', flush=True)
    finally:
        save()
    return metrics.report(dict(confusion=confusion), HORIZONS, moving=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=(*GEOMETRY, *model.ARMS, 'pred_ego'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--population', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--cpu-workers', type=int, default=2)
    args = parser.parse_args(argv)
    if (args.checkpoint is None) != (args.arm in GEOMETRY):
        parser.error('Supply a checkpoint for the learned rows only')
    if not 1 <= args.cpu_workers <= 4 or torch.cuda.device_count() != 1:
        raise RuntimeError('Each worker requires one visible GPU and 1–4 CPU geometry workers')
    torch.set_num_threads(1)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    out = args.out_dir.resolve()
    if args.checkpoint and args.checkpoint.resolve().parent in (out, *out.parents):
        raise ValueError('Evaluation output must be separate from the training run')
    out.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(out):
        population_sha = base.sha256(args.population)
        population = json.loads(args.population.read_text(encoding='utf-8'))
        if base.sha256(args.population) != population_sha:
            raise RuntimeError('Population metadata changed while loading')
        if base.sha256(args.config) != population['files']['config']:
            raise RuntimeError('Runtime config changed after population preparation')
        jobs = select_jobs(population, predicted=args.arm == 'pred_ego')
        checkpoint_sha = base.sha256(args.checkpoint) if args.checkpoint else None
        body, train_keys = load_body(args.arm, args.checkpoint, population) if args.checkpoint else (None, [])
        if {job.scene_name for job in jobs} & {key[0] for key in train_keys}:
            raise RuntimeError('TRAIN and evaluation scenes overlap')
        config = base.make_prepare_config(base.load_runtime_config(args.config, population['overrides']))
        root = Path(__file__).resolve().parents[2]
        dependencies = ('tools/paper_ablation_eval/run.py', 'tools/paper_ablation_eval/population.py',
                        'tools/paper_ablation_train/model.py', 'tools/paper_ablation_train/geometry.py',
                        'tools/causal_forward_train/data.py', 'tools/fpn_unet/adapter.py',
                        'tools/fpn_unet_eval/common.py', 'tools/final_semantic_evaluation/metrics.py')
        implementation = {name: base.sha256(root/name) for name in dependencies}
        contract = dict(protocol=PROTOCOL, arm=args.arm, population_sha256=population_sha,
                        checkpoint=str(args.checkpoint) if args.checkpoint else None,
                        checkpoint_sha256=checkpoint_sha, windows=len(jobs), requested_windows=4369,
                        cpu_workers=args.cpu_workers, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        torch=str(torch.__version__), implementation=implementation)
        inputs = PreparedInputs(population)
        predictor = RecoveryPredictor(body, config, 'cuda:0', workers=args.cpu_workers)
        result = dict(contract=contract, status='running', reports={})
        try:
            print(f'ABLATION_EVAL_START {args.arm} GPU={contract["gpu"]} windows={len(jobs)}/4369', flush=True)
            result['reports']['occ'] = evaluate(args.arm, predictor, inputs, jobs, out/'occ_state.json',
                                               dict(contract, input='occ'), stop)
            if args.checkpoint and base.sha256(args.checkpoint) != checkpoint_sha:
                raise RuntimeError('Evaluation checkpoint changed during the run')
            result['status'] = 'complete'
            return 0
        except InterruptedError:
            result['status'] = 'interrupted'
            return 130
        except Exception as error:
            result.update(status='failed', error=str(error))
            raise
        finally:
            predictor.close()
            base.write_json(out/'evaluation.json', result)


if __name__ == '__main__':
    raise SystemExit(main())
