"""Export only selected CarryWorld qualitative windows, without scoring them."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'outputs/p0_f9_shared_history'
SOURCE_SHA = '7760c6f4f92277aa248161cd3e90f87ac2354329babceb6e5ed3b872adde33a0'
SHORT = [1, 3, 5]
LONG = [1, 3, 5, 7, 9, 11]


def turn_stats(poses):
    poses = np.asarray(poses)
    headings = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    steps = np.diff(headings)
    rotation = np.abs(steps).sum()
    return dict(yaw_deg=float(np.degrees(headings[-1] - headings[0])),
                path_m=float(np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1).sum()),
                direction_consistency=float(abs(steps.sum()) / rotation) if rotation else 0.)


def distinct_scenes(ranked, count):
    selected = {}
    for job, info in ranked:
        selected.setdefault(job.scene_name, (job, info))
        if len(selected) == count:
            break
    return list(selected.values())


def turn_candidates(jobs, pose, count, *, visible_motion=None):
    by_turn = defaultdict(list)
    for job in jobs:
        poses = [pose(t) for t in (job.t0_token, *job.future_tokens)]
        stats = turn_stats(poses)
        yaw = stats['yaw_deg']
        if not (25 <= abs(yaw) <= 125 and stats['path_m'] >= 12
                and stats['direction_consistency'] >= .7):
            continue
        direction = 'left' if yaw > 0 else 'right'
        if visible_motion is not None:
            stats.update(visible_motion(job, poses))
        by_turn[direction].append((job, dict(stats, direction=direction)))
    used = set()
    for direction in ('left', 'right'):
        def rank(pair):
            job, stats = pair
            travel = stats.get('visible_gt_vehicle_displacement_m', 0.)
            priority = (travel < 10., abs(travel - 24.), stats['central_vehicle_count']) if visible_motion else ()
            return (*priority, abs(abs(stats['yaw_deg']) - 75), -stats['direction_consistency'],
                    job.scene_name, job.t0_token)
        ranked = sorted(by_turn[direction], key=rank)
        selected = distinct_scenes(((job, info) for job, info in ranked if job.scene_name not in used), count)
        if not selected or (visible_motion is None and len(selected) != count):
            raise RuntimeError(f'Only {len(selected)} distinct {direction}-turn scenes meet the shortlist criteria')
        for job, info in selected:
            used.add(job.scene_name)
            yield job, info


def annotated_vehicles(nusc, token):
    from real_motion.nuscenes_adapter import category_to_dynamic_class

    rows = (nusc.get('sample_annotation', t) for t in nusc.get('sample', token)['anns'])
    return {r['instance_token']: np.asarray(r['translation'][:2]) for r in rows
            if category_to_dynamic_class(r['category_name']) in (3, 4, 5, 9, 10)}


def visible_turn_motion(job, poses, vehicle_positions):
    tokens = [job.future_tokens[index] for index in LONG]
    detections = [vehicle_positions(token) for token in tokens]
    anchor = np.asarray(poses[2])  # t0, 0.5 s, 1 s, ...
    center, rotation = anchor[:2, 3], anchor[:2, :2]
    common = set.intersection(*(set(frame) for frame in detections))
    distances = []
    for key in common:
        world = np.array([frame[key] for frame in detections])
        local = (world - center) @ rotation
        central = (-18 <= local[:, 0]) & (local[:, 0] <= 34) & (np.abs(local[:, 1]) <= 20)
        net = float(np.linalg.norm(world[-1] - world[0]))
        if central.sum() >= 5 and 10 <= net <= 50:
            distances.append(net)
    ref = [(point - center) @ rotation for point in detections[0].values()]
    count = sum(-18 <= p[0] <= 34 and abs(p[1]) <= 20 for p in ref)
    return dict(visible_gt_vehicle_displacement_m=min(distances, key=lambda x: abs(x - 24), default=0.),
                central_vehicle_count=int(count))


def short_candidates(jobs, inputs, poses, count, *, sparse=False):
    from functools import lru_cache

    nusc = inputs.source.nusc

    @lru_cache(maxsize=64)
    def vehicles(token):
        return annotated_vehicles(nusc, token)

    def sparse_window(job):
        tokens = [job.t0_token, *(job.future_tokens[index] for index in SHORT)]
        centered = []
        counts = []
        for token in tokens:
            matrix = np.asarray(poses[token])
            points = {key: (point - matrix[:2, 3]) @ matrix[:2, :2]
                      for key, point in vehicles(token).items()}
            in_grid = {key: point for key, point in points.items() if np.abs(point).max() < 40}
            if not 1 <= len(in_grid) <= 4:
                return None
            counts.append(len(in_grid))
            centered.append({key for key, point in in_grid.items()
                             if -10 <= point[0] <= 32 and abs(point[1]) <= 18})
        common = set.intersection(*centered)
        motion = max((float(np.linalg.norm(vehicles(tokens[-1])[key] - vehicles(tokens[0])[key]))
                      for key in common), default=0.)
        if motion >= 5.:
            return dict(box_vehicle_counts_0_1_2_3s=counts, largest_visible_gt_displacement_m=motion)
        return None

    if sparse:
        ranked = [(job, info) for job in jobs if (info := sparse_window(job)) is not None]
        ranked.sort(key=lambda pair: (sum(pair[1]['box_vehicle_counts_0_1_2_3s']),
                                      abs(pair[1]['largest_visible_gt_displacement_m'] - 12),
                                      pair[0].scene_name, pair[0].t0_token))
        selected = distinct_scenes(ranked, count)
        if not selected:
            raise RuntimeError('No sparse short windows meet the 1–4 GT vehicles and visible motion criteria')
        return selected

    ranked = []
    for job in jobs:
        previous, current = job.history_tokens[-2:]
        old, now = vehicles(previous), vehicles(current)
        dt = (nusc.get('sample', current)['timestamp'] - nusc.get('sample', previous)['timestamp']) / 1e6
        ego = np.asarray(poses[current])[:2, 3]
        moving = [float(np.linalg.norm(now[k] - ego)) for k in now.keys() & old.keys()
                  if dt > 0 and np.linalg.norm(now[k] - old[k]) / dt >= 1.]
        nearby = [distance for distance in moving if distance <= 28]
        if nearby:
            ranked.append((job, dict(nearby_moving_vehicles=len(nearby), nearest_moving_vehicle_m=min(nearby))))
    ranked.sort(key=lambda x: (-x[1]['nearby_moving_vehicles'], x[1]['nearest_moving_vehicle_m'],
                               x[0].scene_name, x[0].t0_token))
    selected = distinct_scenes(ranked, count)
    if len(selected) != count:
        raise RuntimeError(f'Only {len(selected)} distinct short candidates with nearby motion')
    return selected


def final_semantics(predictor, record, raw, times, poses):
    row = predictor.prepare(record, raw, times=times)
    output = predictor.forward(row, poses=poses)
    return output['logits'].argmax(dim=1).reshape(6, *output['shape']).byte().cpu().numpy()


def long_rollout(predictor, inputs, job):
    from tools.real_motion.joint_long_rollout_common import inherited_observation_masks

    record, raw, times = inputs.history(job, 'occ')
    poses = inputs.poses(job)
    first = final_semantics(predictor, record, raw, times, poses[:6])
    # Same second block as final_semantic_evaluation.engine: its only semantic
    # histories are the last four forecasts, with original observation support.
    raw2 = dict(history_occ=first[-4:].copy(), history_poses=poses[2:6],
                history_observed=inherited_observation_masks(raw, poses[2:6], predictor.provider.pcfg.grid, 1))
    rec2 = dict(scene_name=job.scene_name, t0_token=job.future_tokens[5],
                history_tokens=job.future_tokens[2:6], future_tokens=job.future_tokens[6:12])
    times2 = [inputs.source.nusc.get('sample', t)['timestamp'] / 1e6 for t in rec2['history_tokens']]
    second = final_semantics(predictor, rec2, raw2, times2, poses[6:12])
    return np.concatenate((first, second))[LONG], poses


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def server_cases(args):
    from functools import lru_cache
    from tools.causal_forward_all.common import DenseInputs, Epoch2Predictor, load_frozen
    from tools.final_semantic_evaluation.sources import WindowTokens
    from tools.paper_ablation_eval.population import select_jobs
    from tools.paper_ablation_eval.run import load_body, predict
    from tools.causal_forward_recovery.adapter import RecoveryPredictor
    from tools.real_motion.geniedrive_eval_alignment import POPULATION_PROTOCOL, validate_start_manifest
    from real_motion.runtime_config import load_runtime_config, make_prepare_config
    from real_motion.waymo_i2world import file_sha256

    population = read_json(args.population)
    result = read_json(args.long_result)
    long_population = result['population']
    if (result['status'], result['task'], result['source_sha256'], result['windows'],
            long_population['alignment'], long_population['diagnostic_prefix']) != (
            'complete', 'long6', SOURCE_SHA, 2569, POPULATION_PROTOCOL, False):
        raise ValueError('Use the completed paper long6 result for scene identities')
    validate_start_manifest(long_population['selected_keys'])
    chains = long_population['selected_sequence_tokens_4_plus20']
    if len(chains) != 2569 or any(len(c) != 24 or len(set(c)) != 24 for c in chains):
        raise ValueError('Archived long6 token chains differ')
    long_jobs = [WindowTokens(scene, tuple(c[:4]), token, tuple(c[4:16]))
                 for (scene, token), c in zip(long_population['selected_keys'], chains)]
    if any(w.history_tokens[-1] != w.t0_token for w in long_jobs):
        raise ValueError('long6 chain does not end its history at the anchor')
    jobs = select_jobs(population, False)
    if len({(w.scene_name, w.t0_token) for w in jobs}) != 4369:
        raise ValueError('Use the original complete OCC ablation population')
    config_sha = file_sha256(args.config)
    if file_sha256(args.checkpoint) != SOURCE_SHA or config_sha != population['files']['config']:
        raise ValueError('Use the released E checkpoint and OCC ablation config')
    root = args.dataroot or Path(population['dataroot'])
    dev_info = args.dev_info or root / 'nuscenes_infos_val_temporal_v3_scene.pkl'
    if file_sha256(dev_info) != population['files']['dev_info']:
        raise ValueError('Use the original VAL metadata')
    inputs = DenseInputs(root, dev_info, cache_mib=32)
    pose = lru_cache(maxsize=None)(inputs.source.pose)
    short = short_candidates(jobs, inputs, population['poses'], args.short_count, sparse=args.sparse_short)
    motion = None
    if args.moving_turns:
        @lru_cache(maxsize=1024)
        def vehicle_positions(token):
            return annotated_vehicles(inputs.source.nusc, token)

        def motion(job, poses):
            return visible_turn_motion(job, poses, vehicle_positions)

    turns = list(turn_candidates(long_jobs, pose, args.turn_count, visible_motion=motion)) if args.turn_count else []
    for job, _ in short + turns:
        samples = [inputs.source.nusc.get('sample', t) for t in (*job.history_tokens, *job.future_tokens)]
        if any(a['next'] != b['token'] or a['scene_token'] != b['scene_token']
               for a, b in zip(samples, samples[1:])):
            raise ValueError(f'Non-contiguous window: {job.scene_name}/{job.t0_token}')
    print(f'QUALITATIVE_SHORTLIST {len(short)} short, {len(turns)} long', flush=True)

    contract, body = load_frozen(args.checkpoint, args.device)
    if contract['data']['config'] != config_sha or contract['data']['dev_info'] != population['files']['dev_info']:
        raise ValueError('E checkpoint and evaluation metadata differ')
    baseline, train_keys = load_body('c_transport', args.c_checkpoint, population, args.device)
    selected_scenes = {w.scene_name for w, _ in short + turns}
    if selected_scenes & ({k[0] for k in train_keys} | {k[0] for k in contract['train_keys']}):
        raise ValueError('Qualitative candidate overlaps training scenes')
    config = make_prepare_config(load_runtime_config(args.config, contract.get('overrides')))
    short_selection = ('1–4 GT vehicles at 0/1/2/3 s; at least one central same-instance GT vehicle moves ≥5 m'
                       if args.sparse_short else 'historical moving boxes within 28 m')
    turn_selection = ('signed future ego heading, prioritizing visible GT vehicle motion'
                      if args.moving_turns else 'signed future ego heading')
    metadata = dict(source_sha256=SOURCE_SHA, baseline_sha256=file_sha256(args.c_checkpoint),
                    short_population=4369, long_population=2569, preview=False,
                    selection=f'OCC short: {short_selection}; long: {turn_selection}')

    def generate():
        final = Epoch2Predictor(body, config, args.device, workers=args.workers)
        try:
            weak = RecoveryPredictor(baseline, config, args.device, workers=args.workers)
            try:
                for index, (job, stats) in enumerate(short, 1):
                    record, raw, times = inputs.history(job, 'occ')
                    poses = inputs.poses(job)
                    c = predict('c_transport', weak, record, raw, times, poses)[SHORT]
                    e = final_semantics(final, record, raw, times, poses)[SHORT]
                    gt = np.stack(inputs.targets(job, SHORT))
                    yield f'short{index:02}', case_info(job, stats, poses), dict(t0=raw['history_occ'][-1], gt=gt, c=c, e=e)
            finally:
                weak.close()
            for index, (job, stats) in enumerate(turns, 1):
                future, poses = long_rollout(final, inputs, job)
                yield f'turn{index:02}', case_info(job, stats, poses), dict(e=future)
        finally:
            final.close()

    return metadata, config.grid, generate()


def case_info(job, info, poses):
    return dict(scene=job.scene_name, t0_token=job.t0_token, history_tokens=list(job.history_tokens),
                future_tokens=list(job.future_tokens), selection=info, future_ego_poses=np.asarray(poses).tolist())


def local_cases(args):
    from tools.causal_forward_all.common import Epoch2Predictor
    from tools.causal_forward_local.portable import load_weights
    from tools.final_semantic_evaluation.sources import window
    from tools.stc_overnight.bundle import Bundle

    body, contract, info = load_weights(args.weights, args.device)
    if info['manifest']['source_checkpoint_sha256'] != SOURCE_SHA:
        raise ValueError('Preview must use the released E weights')
    config = info['prepare_config']
    metadata = dict(source_sha256=SOURCE_SHA, preview=True, short_population=len(contract['eval_keys']),
                    selection='local preview: near vehicle voxels and road coverage in t0 OCC')

    def generate():
        bundle = Bundle(args.data, cache_mib=16)
        try:
            ranked = []
            for key in contract['eval_keys']:
                row = bundle.by_key[tuple(key)]
                t0 = bundle.frame('gt', *key)['semantics']
                center = t0[55:145, 55:145]
                cars = int(np.isin(center, [3, 4, 10]).sum())
                road = int((center == 11).sum())
                ranked.append((-(min(cars, 700) + min(road, 2200) * .2), tuple(key), cars, road))
            predictor = Epoch2Predictor(body, config, args.device, workers=args.workers)
            try:
                for index, (_, key, cars, road) in enumerate(sorted(ranked)[:args.short_count], 1):
                    row = bundle.by_key[key]
                    record, raw, times = bundle.history(row, 'gt', 'dense_valid')
                    poses = row['future_poses']
                    e = final_semantics(predictor, record, raw, times, poses)[SHORT]
                    gt = bundle.targets(row, SHORT)
                    stats = dict(near_vehicle_voxels=cars, near_road_voxels=road)
                    yield f'short{index:02}', case_info(window(row), stats, poses), dict(t0=raw['history_occ'][-1], gt=gt, e=e)
            finally:
                predictor.close()
        finally:
            bundle.close()

    return metadata, config.grid, generate()


def export(path, metadata, grid, cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + '.partial')
    manifest = dict(metadata, format='carryworld_qualitative_v1', cases={},
                    grid={k: getattr(grid, k) for k in ('x_min', 'y_min', 'z_min', 'shape_hwd', 'voxel_size')})
    with zipfile.ZipFile(temp, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, info, arrays in cases:
            for name, value in arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, value, allow_pickle=False)
                archive.writestr(f'{key}/{name}.npy', buffer.getvalue())
            manifest['cases'][key] = dict(info, arrays=list(arrays))
            print('QUALITATIVE_EXPORTED', key, info['scene'], flush=True)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    temp.rename(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    modes = parser.add_subparsers(dest='mode', required=True)
    server = modes.add_parser('server')
    server.add_argument('--checkpoint', type=Path, default=RUNS/'adaln_forward_full_2ep_20261026_215351/epoch_0002.pt')
    server.add_argument('--c-checkpoint', type=Path, default=RUNS/'ablation_five_dual_20261107_203511/c_transport/epoch_0001.pt')
    server.add_argument('--population', type=Path, default=RUNS/'ablation_all_parallel_20261112_105214/population.json')
    server.add_argument('--long-result', type=Path, default=RUNS/'forward_transfer_1gpu3proc_20261112_183352/long6/evaluation.json')
    server.add_argument('--config', type=Path, default=ROOT/'configs/real_motion_occfm.yaml')
    server.add_argument('--dataroot', type=Path)
    server.add_argument('--dev-info', type=Path)
    server.add_argument('--short-count', type=int, choices=range(1, 33), default=16,
                        help='number of short windows (default: 16)')
    server.add_argument('--turn-count', type=int, choices=range(0, 17), default=8,
                        help='number of long windows per turn direction (default: 8; 0 skips long inference)')
    server.add_argument('--sparse-short', action='store_true',
                        help='short windows with 1–4 GT vehicles and one clearly moving GT car')
    server.add_argument('--moving-turns', action='store_true',
                        help='prioritize turns with a moving GT vehicle in the main display area')
    local = modes.add_parser('local-preview')
    local.add_argument('--weights', type=Path, required=True)
    local.add_argument('--data', type=Path, required=True)
    local.add_argument('--short-count', type=int, choices=range(1, 9), default=3)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f'Output already exists: {args.out}')
    import torch
    torch.set_num_threads(1)
    with torch.inference_mode():
        metadata, grid, cases = (server_cases(args) if args.mode == 'server' else local_cases(args))
        export(args.out, metadata, grid, cases)


if __name__ == '__main__':
    main()
