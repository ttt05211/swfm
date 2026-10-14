import numpy as np
from tools.final_semantic_evaluation import metrics

BINS = np.array([-np.inf, -16., -8., -4., 0., 1., 2., 4., 6., 8., 9., 10., 12., 16., np.inf])


def confusion(pred, target):
    c = metrics.empty(1); metrics.add(c, 0, pred, target)
    return c['confusion'][0]


def input_report(c):
    c = np.asarray(c, np.int64)
    r = metrics.report(dict(confusion=c[None], moving=np.zeros((1, 18, 18), np.int64)), (0,), moving=False)
    tp = int(c[:17, :17].sum()); predicted = int(c[:, :17].sum()); truth = int(c[:17].sum())
    return dict(**r['average'], occupied_precision=100*tp/predicted if predicted else None,
        occupied_recall=100*tp/truth if truth else None,
        semantic_precision=100*np.diag(c)[:17].sum()/predicted if predicted else None,
        wrong_occupancy=predicted-tp, missed_occupancy=truth-tp,
        per_class=r['per_horizon']['0']['semantic_per_class'])


def distribution(values):
    x = np.asarray(values, float)
    if not np.isfinite(x).all(): raise ValueError('nonfinite diagnostic value')
    return dict(count=int(x.size), sum=float(x.sum()), sum_sq=float(np.square(x).sum()),
                histogram=np.histogram(x, BINS)[0].tolist())


def edits_and_margins(snapshot, gt, moving=None):
    """Exact GT-vs-prior-winner competition, not epsilon/threshold selection."""
    before, after = snapshot['transport'], snapshot['semantic_refine']
    gt = np.stack(gt); p, z = snapshot['prior_logits'], snapshot['logits']
    if p.shape != z.shape or p.shape != (3, 18, gt[0].size): raise ValueError('diagnostic logit shape differs')
    if before.shape != gt.shape or after.shape != gt.shape: raise ValueError('diagnostic label shape differs')
    if not np.isfinite(p).all() or not np.isfinite(z).all(): raise ValueError('nonfinite logits')
    out = []
    for h in range(3):
        b, a, target = before[h].ravel(), after[h].ravel(), gt[h].ravel()
        ids = np.arange(len(b)); correct, final_correct = b == target, a == target
        barrier = p[h, b, ids]-p[h, target, ids]
        # ΔZ_GT - ΔZ_oldwinner must exceed this prior barrier; a third class can still win.
        advantage = (z[h, target, ids]-p[h, target, ids])-(z[h, b, ids]-p[h, b, ids])
        groups = dict(wrong_prior=~correct, correct_prior=correct,
            wrong_free=(b == 17) & (target != 17), false_occupancy=(b != 17) & (target == 17),
            wrong_occupied_class=(b != 17) & (target != 17) & ~correct)
        row = dict(changed=int((b != a).sum()), corrected=int((~correct & final_correct).sum()),
            damaged=int((correct & ~final_correct).sum()), wrong_to_other_wrong=int((~correct & ~final_correct & (b != a)).sum()),
            added=int(((b == 17) & (a != 17)).sum()), removed=int(((b != 17) & (a == 17)).sum()),
            relabeled=int(((b != 17) & (a != 17) & (b != a)).sum()), groups={})
        from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
        target_groups = dict(ground=np.isin(target, (11, 12, 13, 14)),
            building_vegetation=np.isin(target, (15, 16)), dynamic_semantics=np.isin(target, DYNAMIC_CLASS_IDS))
        if moving is not None: target_groups['moving_support'] = np.asarray(moving[h], bool).ravel()
        row['edits_by_group'] = {name: dict(corrected=int((mask & ~correct & final_correct).sum()),
            damaged=int((mask & correct & ~final_correct).sum()),
            removed=int((mask & (b != 17) & (a == 17)).sum())) for name, mask in target_groups.items()}
        for name, mask in groups.items():
            row['groups'][name] = dict(count=int(mask.sum()), final_correct=int((mask & final_correct).sum()),
                overcame_old_winner=int((mask & (advantage > barrier)).sum()),
                barrier=distribution(barrier[mask]), residual_advantage=distribution(advantage[mask]),
                final_GT_minus_oldwinner=distribution((advantage-barrier)[mask]))
        out.append(row)
    return out


def merge_edits(rows):
    result = []
    for h in range(3):
        values = [v[h] for v in rows]
        row = {k: sum(v[k] for v in values) for k in values[0] if k not in ('groups', 'edits_by_group')}
        row['net_semantic_corrected'] = row['corrected']-row['damaged']; row['groups'] = {}
        row['edits_by_group'] = {name: {k: sum(v['edits_by_group'][name][k] for v in values)
            for k in values[0]['edits_by_group'][name]} for name in values[0]['edits_by_group']}
        for name in values[0]['groups']:
            g = [v['groups'][name] for v in values]
            item = {k: sum(v[k] for v in g) for k in ('count', 'final_correct', 'overcame_old_winner')}
            item['final_correct_fraction'] = item['final_correct']/item['count'] if item['count'] else None
            for k in ('barrier', 'residual_advantage', 'final_GT_minus_oldwinner'):
                n = sum(v[k]['count'] for v in g); total = sum(v[k]['sum'] for v in g)
                item[k] = dict(count=n, sum=total, mean=total/n if n else None,
                    histogram=np.sum([v[k]['histogram'] for v in g], axis=0).tolist())
            row['groups'][name] = item
        result.append(row)
    return result


def pose_error(value, target):
    delta = np.asarray(value)-np.asarray(target)
    return dict(xy=np.linalg.norm(delta[:, :2], axis=1).tolist(),
                yaw_deg=(np.abs(np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2])))*180/np.pi).tolist())


def trajectory_report(rows):
    xy = np.array([r['xy'] for r in rows]); yaw = np.array([r['yaw_deg'] for r in rows])
    return dict(ADE_m=float(xy.mean()), FDE3s_m=float(xy[:, -1].mean()),
        xy_mean_m=xy.mean(0).tolist(), xy_p90_m=np.percentile(xy, 90, axis=0).tolist(),
        yaw_mean_deg=yaw.mean(0).tolist(), yaw_p90_deg=np.percentile(yaw, 90, axis=0).tolist())


def rigid_pose_errors(current, predicted_se2, gt_poses):
    from real_motion.ego_navigation import poses_from_se2
    pred, truth = poses_from_se2(current, predicted_se2), np.asarray(gt_poses)
    cosine = (pred[:, :3, 2]*truth[:, :3, 2]).sum(1).clip(-1, 1)
    relative = np.swapaxes(pred[:, :3, :3], 1, 2) @ truth[:, :3, :3]
    full_angle = np.arccos(((np.trace(relative, axis1=1, axis2=2)-1)*.5).clip(-1, 1))*180/np.pi
    return dict(world_xy_m=np.linalg.norm(pred[:, :2, 3]-truth[:, :2, 3], axis=1).tolist(),
        z_m=np.abs(pred[:, 2, 3]-truth[:, 2, 3]).tolist(), tilt_deg=(np.arccos(cosine)*180/np.pi).tolist(),
        full_rotation_deg=full_angle.tolist())
