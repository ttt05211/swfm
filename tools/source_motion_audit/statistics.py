"""Integer full-grid counts and POST-prediction GT-box motion diagnostics."""
from collections import Counter
import numpy as np
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, NUSCENES_LABELS, SPEED_THRESHOLD_MPS
from real_motion.local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS
from tools.shared_history_experiments.full_gt.engine import Metrics
from tools.source_scene_stage1.association import attribute, boxes_for_sample

HORIZONS = (1, 3, 5)


def confusion(p, g):
    p, g = np.asarray(p), np.asarray(g)
    if p.shape != g.shape or np.any((p < 0)|(p > 17)|(g < 0)|(g > 17)): raise ValueError('0..17 full-grid labels required')
    return np.bincount(g.astype(np.int64).ravel()*18+p.ravel(), minlength=324).reshape(18, 18)


def empty_counts(): return dict(full=np.zeros((3, 18, 18), np.int64), moving=np.zeros((3, 18, 18), np.int64))


def finish(counts):
    full, moving = counts['full'], counts['moving']; m = Metrics()
    for i, (f, a) in enumerate(zip(full, moving)):
        diagonal, rows, cols = np.diag(f), f.sum(1), f.sum(0)
        m.oi[i] = f[:17, :17].sum(); m.ou[i] = f.sum()-f[17, 17]
        m.si[i] = diagonal[:17]; m.su[i] = (rows+cols-diagonal)[:17]
        d = np.diag(a); ids = np.asarray(DYNAMIC_CLASS_IDS)
        m.mi[i] = d[ids]; m.mu[i] = (a.sum(0)+a.sum(1)-d)[ids]
    report = m.compute()
    def regional(cm):
        items = {}
        for i, c in enumerate(cm):
            d = np.diag(c); u = c.sum(0)+c.sum(1)-d
            per = [float(100*d[j]/u[j]) if u[j] else None for j in range(17)]
            valid = [v for v in per if v is not None]
            union = c.sum()-c[17, 17]
            items[str(i+1)] = dict(mIoU=float(np.mean(valid)) if valid else None,
                IoU=float(100*c[:17, :17].sum()/union) if union else None,
                per_class={NUSCENES_LABELS[j]: v for j, v in enumerate(per)},
                false_occupied=int(c[17, :17].sum()), missed_occupied=int(c[:17, 17].sum()))
        return items
    report['regions'] = dict(moving_support=regional(moving), outside_moving_support=regional(full-moving))
    return report


def window_counts(predictions, targets, masks):
    result = {}
    for route, pred in predictions.items():
        counts = empty_counts()
        for i, h in enumerate(HORIZONS):
            p, g, mask = pred[h], targets[i], np.asarray(masks[i], bool)
            if p.shape != mask.shape: raise ValueError('moving support shape differs')
            counts['full'][i] = confusion(p, g); counts['moving'][i] = confusion(p[mask], g[mask])
        result[route] = counts
    return result


def edits(predictions, target, masks):
    result = {}
    for readout in ('transport', 'semantic_refine'):
        sums = Counter()
        for i, h in enumerate(HORIZONS):
            a, b, g = predictions['learned/'+readout][h], predictions['KTA_future/'+readout][h], target[i]
            for label, mask in (('full', np.ones(g.shape, bool)), ('moving', masks[i]), ('outside_moving', ~masks[i])):
                sums[label+'/corrected'] += int(((a == g)&(b != g)&mask).sum())
                sums[label+'/damaged'] += int(((a != g)&(b == g)&mask).sum())
                sums[label+'/damaged_correct_free'] += int(((a != g)&(b == g)&(g == 17)&mask).sum())
                sums[label+'/wrong_to_other_wrong'] += int(((a != g)&(b != g)&(a != b)&mask).sum())
        result[readout] = dict(sums)
    return result


def box_motion_rows(row, diagnostic, nusc, grid):
    rec, raw, _, history = row
    current_boxes = boxes_for_sample(nusc, rec['t0_token'])
    attribution = attribute(history[0]['state']['current'], current_boxes, raw['history_poses'][-1], grid)
    boxes0 = {b['instance_token']: b for b in current_boxes}
    future = [{b['instance_token']: b for b in boxes_for_sample(nusc, str(t))} for t in rec['future_tokens']]
    t0 = nusc.get('sample', rec['t0_token'])['timestamp']/1e6
    times = [nusc.get('sample', str(t))['timestamp']/1e6-t0 for t in rec['future_tokens']]
    if len(times) != 6 or not np.isfinite(times).all() or min(times) <= 0: raise ValueError('six positive forecast timestamps')
    sources = []
    for actor, assignment in enumerate(attribution):
        cls = int(rec['source_class_id'][actor]); token = assignment['token']; old = boxes0.get(token)
        for h in range(6):
            residual = diagnostic['residual_xy_m'][actor, h]
            item = dict(actor=actor, class_id=cls, horizon_s=.5*(h+1), attribution=assignment['status'],
                attribution_purity=assignment['purity'], fragments=assignment['fragments_for_instance'],
                XY_correction_m=float(np.linalg.norm(residual)),
                future_yaw_abs_deg=float(abs(diagnostic['effective_future_yaw_rad'][actor, h])*180/np.pi),
                source_size_voxels=len(history[0]['state']['current'][actor]['voxel_indices']),
                group='unresolved')
            later = future[h].get(token)
            if old is not None and later is not None:
                gt = np.asarray(later['center'])[:2]-np.asarray(old['center'])[:2]
                speed = float(np.linalg.norm(gt)/times[h]); yaw0 = np.arctan2(old['rotation'][1, 0], old['rotation'][0, 0])
                yawh = np.arctan2(later['rotation'][1, 0], later['rotation'][0, 0]); gt_yaw = np.arctan2(np.sin(yawh-yaw0), np.cos(yawh-yaw0))
                item.update(group='moving' if speed >= SPEED_THRESHOLD_MPS else 'slow_or_stationary', GT_speed_mps=speed,
                    learned_error_m=float(np.linalg.norm(diagnostic['learned_world_displacement'][actor, h, :2]-gt)),
                    KTA_error_m=float(np.linalg.norm(diagnostic['KTA_world_displacement'][actor, h, :2]-gt)))
                if cls in YAW_ENABLED_CLASS_IDS:
                    pred = diagnostic['effective_future_yaw_rad'][actor, h]
                    item.update(learned_yaw_error_deg=float(abs(np.arctan2(np.sin(pred-gt_yaw), np.cos(pred-gt_yaw)))*180/np.pi),
                                KTA_yaw_error_deg=float(abs(gt_yaw)*180/np.pi))
            sources.append(item)
    return sources


def distribution(values):
    a = np.asarray(values, float)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.median(a)), p90=float(np.quantile(a, .9))) if len(a) else dict(n=0)


def source_summary(rows):
    fields = ('XY_correction_m', 'future_yaw_abs_deg', 'learned_error_m', 'KTA_error_m', 'learned_yaw_error_deg', 'KTA_yaw_error_deg')
    def one(items):
        out = dict(samples=len(items), attribution=dict(Counter(r['attribution'] for r in items)),
                   **{k: distribution([r[k] for r in items if k in r]) for k in fields})
        valid = [r for r in items if 'learned_error_m' in r]
        out['XY_helped'] = sum(r['learned_error_m'] < r['KTA_error_m'] for r in valid)
        out['XY_harmed'] = sum(r['learned_error_m'] > r['KTA_error_m'] for r in valid)
        out['XY_near_zero_fraction_1cm'] = sum(r['XY_correction_m'] < .01 for r in items)/len(items) if items else None
        return out
    return dict(all=one(rows), groups={g: one([r for r in rows if r['group'] == g]) for g in ('moving', 'slow_or_stationary', 'unresolved')},
                classes={NUSCENES_LABELS[c]: one([r for r in rows if r['class_id'] == c]) for c in sorted({r['class_id'] for r in rows})},
                horizon={str(h): one([r for r in rows if r['horizon_s'] == h]) for h in (.5, 1., 1.5, 2., 2.5, 3.)})
