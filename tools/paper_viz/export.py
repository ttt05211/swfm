"""Paper qualitative figures: READ-ONLY export of CarryWorld intermediates.

Runs the frozen epoch2 forward model (weights-only portable pack) on the
portable replay bundle, CPU friendly.  Nothing in the source repo, weights or
bundle is modified; every output goes to a NEW --out directory.

Per window it saves one compressed npz with
  * history association (source links/centers over the four history frames),
  * t0 sources (class, world centroid, world voxel points),
  * kinematic prior anchors, predicted per-source future centers / yaw deltas,
  * transport prior / refined prediction / GT for all six futures (uint8),
  * memory support maps (coarse grid) for the learned source motion AND for an
    ego-only alignment (dynamic evidence left at its historical location),
and one row of selection scores in scores.csv.

Future labels are read only AFTER all six predictions of a window are done.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

AT = (1, 3, 5)  # 1s / 2s / 3s
DYNAMIC = (2, 3, 4, 5, 6, 7, 9, 10)  # bicycle bus car cons.veh motorcycle pedestrian trailer truck
FREE = 17


def support_maps(columns, lifted, output, history, raw, targets, yaws, poses, grid, ego_only=False):
    """Same weights as aligned_memory's denominator; returns (static, dynamic) per horizon."""
    from tools.shared_history_experiments.refine.geometry import centers, inverse_grid
    from tools.joint_occ_experiments.soft_geometry import grid_arrays, transform
    from tools.source_scene_ego_conditioned.geometry import direct_move
    data = history[0]['memory_reads']; device = lifted.device
    cast = lambda x: torch.as_tensor(np.asarray(x), device=device)
    query = centers(grid, device, coarse=True); coarse_shape = tuple(query.shape[:3]); volume = int(np.prod(coarse_shape))
    static_valid = cast(data['static_valid']).float()
    scores = columns.importance(lifted.permute(0, 2, 3, 4, 1).float()).squeeze(-1).clamp(-8, 8).exp()
    static_weight = (scores*static_valid)[:, None]
    cells, frames, actors = (cast(data[k]).long() for k in ('cell', 'frame', 'actor'))
    source_feature = lifted.flatten(2).permute(0, 2, 1)[frames, cells].float()
    importance = columns.importance(source_feature).float().squeeze(-1).clamp(-8, 8).exp()*cast(data['coverage']).float()
    points = cast(data['world']).double()
    historical_centers = cast(data['centers_world']).double()[actors, frames]
    relative_yaw = output['history_yaw_to_t0_rad'][actors, frames].double()
    history_poses = poses.new_tensor(np.asarray(raw['history_poses']))
    origin, step, shape = grid_arrays(grid)
    corigin, cstep = origin, step*np.array([2, 2, 1])
    static_out, dynamic_out = [], []
    for h in range(6):
        matrix = torch.linalg.inv(history_poses) @ poses[h][None]
        sampling = inverse_grid(query[None].expand(4, -1, -1, -1, -1), matrix, grid, coarse=True)
        warped = F.grid_sample(static_weight, sampling, align_corners=False, padding_mode='zeros').sum(0).flatten()
        dyn = torch.zeros(volume, dtype=torch.float32, device=device)
        if len(points):
            moved = points if ego_only else direct_move(points, historical_centers, targets[actors, h], relative_yaw, yaws[actors, h])
            mapped = transform(moved, torch.linalg.inv(poses[h]))
            q = (mapped-mapped.new_tensor(corigin))/mapped.new_tensor(cstep)-.5
            lower = q.floor().long(); fraction = (q-lower).float()
            offsets = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], device=device)
            at = lower[:, None]+offsets
            weight = torch.where(offsets.bool()[None], fraction[:, None], 1-fraction[:, None]).prod(-1)
            weight = weight*((at >= 0) & (at < at.new_tensor(coarse_shape))).all(-1)
            ids = ((at[..., 0]*coarse_shape[1]+at[..., 1])*coarse_shape[2]+at[..., 2]).clamp(0, volume-1)
            dyn = dyn.scatter_add(0, ids.flatten(), (weight*importance[:, None]).flatten())
        static_out.append(warped.reshape(coarse_shape)); dynamic_out.append(dyn.reshape(coarse_shape))
    return torch.stack(static_out).cpu().numpy().astype(np.float16), torch.stack(dynamic_out).cpu().numpy().astype(np.float16)


def ragged(arrays, width):
    arrays = [np.zeros((0, width)) if a is None else np.asarray(a).reshape(-1, width) for a in arrays]
    offsets = np.cumsum([0]+[len(a) for a in arrays]).astype(np.int64)
    return (np.concatenate(arrays) if arrays else np.zeros((0, width))), offsets


def iou_counts(pred, gt):
    occ_p, occ_g = pred != FREE, gt != FREE
    inter = int((occ_p & occ_g).sum()); union = int((occ_p | occ_g).sum())
    return inter/max(union, 1)


def miou(pred, gt, classes=range(1, 17)):
    vals = []
    for c in classes:
        p, g = pred == c, gt == c
        u = int((p | g).sum())
        if (g.sum()) > 0: vals.append(int((p & g).sum())/u)
    return float(np.mean(vals)) if vals else float('nan')


@torch.no_grad()
def run_window(p, body, b, r, grid):
    from real_motion.ego_navigation import ego_history_features  # noqa: F401 (import parity)
    rec, raw, times = b.history(r, 'gt', 'dense_valid')
    row = p.prepare(rec, raw, times=times)
    future = np.asarray(r['future_poses'], np.float64)[:6]  # model is FOUR -> SIX; frames bundle carries 12
    fw = p.forward(row, poses=future)
    dense = p.dense(fw)
    rec2, raw2, _, history = row
    poses = fw['poses']
    sup_learned = support_maps(body.columns, fw['lifted'], fw['output'], history, raw2, fw['targets'], fw['yaws'], poses, grid)
    sup_ego = support_maps(body.columns, fw['lifted'], fw['output'], history, raw2, fw['targets'], fw['yaws'], poses, grid, ego_only=True)
    identity = raw2['_shared_source_track']; state = fw['state']; current = state['current']
    # ---- all predictions done; only now read future labels ----
    labels = b.targets(r, tuple(range(6)))
    N = len(current)
    vox = [v for row_ in identity.voxels for v in row_]
    hist_vox, hist_off = ragged(vox, 3)
    pts, pts_off = ragged(state['source_world_points'], 3)
    out = dict(
        key=np.asarray([r['scene_name'], r['t0_token']]), role=np.asarray(r['role']),
        history_poses=np.asarray(raw['history_poses'], np.float64), future_poses=future,
        history_times=np.asarray(times, np.float64),
        link=np.asarray(identity.links, np.int64).reshape(N, 4), link_centers=np.asarray(identity.centers).reshape(N, 4, 3),
        hist_voxels=hist_vox.astype(np.int16), hist_voxel_offsets=hist_off,  # (N*4 ragged, row-major source, frame)
        src_class=np.asarray([c['class_id'] for c in current], np.int64).reshape(N),
        src_centroid_world=np.asarray([c['centroid_world'] for c in current], np.float64).reshape(N, 3),
        src_points_world=pts.astype(np.float32), src_point_offsets=pts_off,
        anchors_xy_t0=np.asarray(rec2['anchors_xy_t0_m'], np.float32).reshape(N, 6, 2),
        centroid_xy_t0=np.asarray(rec2['source_centroid_xy_t0_m'], np.float32).reshape(N, 2),
        target_world=fw['targets'].double().cpu().numpy().reshape(N, 6, 3),
        yaw_delta=fw['yaws'].double().cpu().numpy().reshape(N, 6),
        prior=dense['transport'], refined=dense['semantic_refine'], gt=labels.astype(np.uint8),
        hist_occ=np.asarray(raw['history_occ'], np.uint8),
        support_static=sup_learned[0], support_dynamic=sup_learned[1],
        support_dynamic_ego=sup_ego[1])
    return out


def scores(o):
    N = len(o['src_class'])
    disp = np.linalg.norm(o['target_world'][:, 5, :2]-o['src_centroid_world'][:, :2], axis=-1) if N else np.zeros(0)
    yaw3 = np.abs(np.degrees(o['yaw_delta'][:, 5])) if N else np.zeros(0)
    moving = disp > 2.0
    matched = o['link'][:, 2] >= 0 if N else np.zeros(0, bool)
    unmatched = (o['link'][:, :3] < 0).all(1) if N else np.zeros(0, bool)
    ego_yaw = lambda T: np.arctan2(T[1, 0], T[0, 0])
    ego_turn = float(np.degrees(abs(np.angle(np.exp(1j*(ego_yaw(o['future_poses'][5])-ego_yaw(o['history_poses'][3])))))))
    ego_dist = float(np.linalg.norm(o['future_poses'][5][:2, 3]-o['history_poses'][3][:2, 3]))
    fix = dmg = changed = 0; mi_t = []; mi_r = []; io_t = []; io_r = []
    for h in AT:
        p, r, g = o['prior'][h], o['refined'][h], o['gt'][h]
        ch = p != r; changed += int(ch.sum())
        fix += int((ch & (r == g) & (p != g)).sum()); dmg += int((ch & (p == g) & (r != g)).sum())
        mi_t.append(miou(p, g)); mi_r.append(miou(r, g)); io_t.append(iou_counts(p, g)); io_r.append(iou_counts(r, g))
    return dict(scene=str(o['key'][0]), token=str(o['key'][1]), role=str(o['role']), sources=N,
        moving=int(moving.sum()), moving_dynamic=int((moving & np.isin(o['src_class'], DYNAMIC)).sum()),
        max_moving_yaw_deg=float(yaw3[moving].max()) if moving.any() else 0.,
        max_disp_m=float(disp.max()) if N else 0., matched_last_pair=int(matched.sum()), unmatched_current=int(unmatched.sum()),
        ego_turn_deg_3s=ego_turn, ego_dist_m_3s=ego_dist,
        refiner_changed=changed, refiner_fixed=fix, refiner_damaged=dmg, refiner_net=fix-dmg,
        mIoU_transport=float(np.nanmean(mi_t)), mIoU_refined=float(np.nanmean(mi_r)),
        IoU_transport=float(np.mean(io_t)), IoU_refined=float(np.mean(io_r)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', required=True); ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cpu')
    ap.add_argument('--roles', default='reference64,reference_scene,stc_fit,stc_selection,long6,val')
    ap.add_argument('--keys', help='optional CSV with scene,token columns: only these windows (second pass)')
    ap.add_argument('--no-npz', action='store_true', help='first pass: scores.csv only, no per-window npz')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(argv); torch.set_num_threads(a.threads); torch.manual_seed(21)
    from tools.causal_forward_local.portable import load_weights
    from tools.causal_forward_all.common import Epoch2Predictor
    from tools.paper_viz.frames import open_bundle
    out = Path(a.out); (out/'windows').mkdir(parents=True, exist_ok=True)
    body, c, info = load_weights(a.weights, a.device); b = open_bundle(a.bundle, cache_mib=64)
    if b.pcfg.grid != info['prepare_config'].grid: raise ValueError('bundle grid differs from model')
    roles = set(a.roles.split(','))
    records = [r for r in b.records if r['role'] in roles]
    if a.keys:
        with open(a.keys) as f: wanted = {(x['scene'], x['token']) for x in csv.DictReader(f)}
        records = [r for r in records if (r['scene_name'], r['t0_token']) in wanted]
    if a.limit: records = records[:a.limit]
    p = Epoch2Predictor(body, info['prepare_config'], a.device, 2)
    path = out/'scores.csv'; rows = []
    if path.exists():
        with open(path) as f: rows = list(csv.DictReader(f))
    done = {(x['scene'], x['token']) for x in rows}
    start = time.monotonic()
    try:
        for i, r in enumerate(records):
            if (r['scene_name'], r['t0_token']) in done: continue
            o = run_window(p, body, b, r, info['prepare_config'].grid)
            name = f"{r['scene_name']}__{r['t0_token'][:8]}.npz"
            if not a.no_npz: np.savez_compressed(out/'windows'/name, **o)
            s = dict(scores(o), file=name); rows.append(s)
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(s)); w.writeheader(); w.writerows(rows)
            print(f'VIZ {i+1}/{len(records)} {name} sec={time.monotonic()-start:.0f} moving={s["moving"]} '
                  f'yaw={s["max_moving_yaw_deg"]:.1f} net_fix={s["refiner_net"]}', flush=True)
    finally:
        p.close(); b.close()
    (out/'README.json').write_text(json.dumps(dict(weights=str(a.weights), bundle=str(a.bundle), windows=len(rows),
        note='GT history, GT future ego; read-only export for qualitative figures'), indent=2))


if __name__ == '__main__':
    main()
