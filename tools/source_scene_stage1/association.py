"""GT boxes label the completed causal matches; NEVER change matcher inputs."""
from collections import Counter
import numpy as np
from real_motion.geometry import pose_matrix
from real_motion.canonical_causal_repair import grid_arrays
from real_motion.source_evidence_audit import transform_points, register_history_shape
from real_motion.shared_source_track import match_pair

PURITY = .5
SECONDARY = .2


def attribute(components, boxes, pose, grid):
    """Independent per-frame box containment, not nearest-centroid pseudo GT.

Majority attribution is approximate: merged/low-purity/tied support is marked
unresolved. Multiple fragments MAY share an identity; never force a one-to-one
box assignment that could hide component fragmentation.
"""
    origin, step, _ = grid_arrays(grid); result = []
    for component in components:
        points = transform_points(origin+(np.asarray(component['voxel_indices'])+.5)*step, pose)
        counts = []
        for box in boxes:
            if int(box['class_id']) != int(component['class_id']): continue
            local = (points-np.asarray(box['center'])) @ np.asarray(box['rotation'])
            count = int((np.abs(local) <= np.asarray(box['size_lwh'])/2+1e-9).all(1).sum())
            if count: counts.append((count, str(box['instance_token'])))
        counts.sort(key=lambda item: (-item[0], item[1]))
        first = counts[0][0] if counts else 0; second = counts[1][0] if len(counts) > 1 else 0
        purity = first/max(1, len(points)); secondary = second/max(1, len(points))
        status = ('no_box_support' if not counts else 'merged_or_overlap' if secondary >= SECONDARY or first == second else
                  'low_purity' if purity < PURITY else 'assigned')
        result.append(dict(token=counts[0][1] if status == 'assigned' else None, status=status,
            purity=purity, secondary=secondary, voxel_count=len(points), candidates=counts))
    fragments = Counter(row['token'] for row in result if row['token'] is not None)
    for row in result: row['fragments_for_instance'] = fragments.get(row['token'], 0)
    return result


def inspect_pair(previous, current, previous_boxes, current_boxes, previous_pose, current_pose,
                 grid, *, dt=.5, actual_dt=.5, speed_gate=25., check_registration=True):
    if not np.isfinite(actual_dt) or actual_dt <= 0: raise ValueError('positive actual timestamp difference required')
    # This executes BEFORE GT attribution and receives no boxes or GT identity.
    links, velocities = match_pair(previous, current, dt, max_speed_mps=speed_gate)
    pa = attribute(previous, previous_boxes, previous_pose, grid)
    ca = attribute(current, current_boxes, current_pose, grid)
    before = {str(b['instance_token']): b for b in previous_boxes}
    now = {str(b['instance_token']): b for b in current_boxes}
    prev_ids = {a['token'] for a in pa if a['token'] is not None}
    cur_ids = {a['token'] for a in ca if a['token'] is not None}
    eligible = prev_ids & cur_ids; recovered = set(); edges = []
    origin, step, _ = grid_arrays(grid)
    for i, j in enumerate(links):
        c = ca[i]; p = pa[int(j)] if j >= 0 else None
        resolved = bool(p and c['token'] is not None and p['token'] is not None)
        correct = bool(resolved and c['token'] == p['token'])
        center_ego = (np.asarray(current[i]['centroid_world'])-np.asarray(current_pose)[:3, 3]) @ np.asarray(current_pose)[:3, :3]
        record = dict(current=i, previous=int(j), class_id=int(current[i]['class_id']),
            range_m=float(np.linalg.norm(center_ego[:2])),
            current_attribution=c, previous_attribution=p, resolved=resolved, correct=correct,
            status='unmatched' if j < 0 else 'correct' if correct else 'wrong_identity' if resolved else 'unresolved')
        token = c['token']
        if token in before and token in now:
            gt_velocity = (np.asarray(now[token]['center'])-np.asarray(before[token]['center']))/actual_dt
            record['GT_speed_mps'] = float(np.linalg.norm(gt_velocity[:2]))
        if correct:
            recovered.add(token)
            record['KTA_velocity_error_mps'] = float(np.linalg.norm(velocities[i][:2]-gt_velocity[:2]))
        if j >= 0 and check_registration:
            old = transform_points(origin+(np.asarray(previous[int(j)]['voxel_indices'])+.5)*step, previous_pose)
            ref = transform_points(origin+(np.asarray(current[i]['voxel_indices'])+.5)*step, current_pose)
            record['registration_accepted'] = bool(register_history_shape(old, ref,
                allow_yaw=int(current[i]['class_id']) != 7).accepted)
        edges.append(record)
    return dict(edges=edges, previous_attributions=pa, current_attributions=ca,
        eligible_instances=sorted(eligible), recovered_instances=sorted(recovered),
        pair_dt_nominal_s=float(dt), actual_dt_s=float(actual_dt),
        annotated_instances_both_frames=len(set(before) & set(now)))


def aggregate(rows):
    counts = Counter(); errors = []; buckets = {}
    def add(bucket, edge):
        bucket['sources'] += 1
        if edge['previous'] >= 0: bucket['accepted_pairs'] += 1
        if edge['resolved']: bucket['resolved_pairs'] += 1
        if edge['correct']: bucket['correct_pairs'] += 1
        if edge['status'] == 'wrong_identity': bucket['wrong_pairs'] += 1
        if edge['previous'] >= 0 and not edge['resolved']: bucket['unresolved_pairs'] += 1
    for row in rows:
        counts['eligible_instances'] += len(row['eligible_instances'])
        counts['recovered_instances'] += len(row['recovered_instances'])
        for edge in row['edges']:
            add(counts, edge)
            if edge['current_attribution']['token'] is not None: counts['attributed_current_sources'] += 1
            counts['attribution/'+edge['current_attribution']['status']] += 1
            if edge['current_attribution']['fragments_for_instance'] > 1: counts['fragmented_current_sources'] += 1
            for name in ('class/'+str(edge['class_id']), 'range/'+str(min(int(edge['range_m']//10), 5))):
                add(buckets.setdefault(name, Counter()), edge)
            if 'GT_speed_mps' in edge:
                add(buckets.setdefault('GT_speed/'+str(min(int(edge['GT_speed_mps']//5), 5)), Counter()), edge)
            if 'KTA_velocity_error_mps' in edge:
                errors.append(edge['KTA_velocity_error_mps'])
            if 'registration_accepted' in edge:
                group = 'correct' if edge['correct'] else 'wrong' if edge['resolved'] else 'unresolved'
                counts['registration/'+group+'/accepted' if edge['registration_accepted'] else 'registration/'+group+'/rejected'] += 1
    ratio = lambda x, y: counts[x]/counts[y] if counts[y] else None
    def finish(values):
        return {**dict(values), 'precision_resolved': values['correct_pairs']/values['resolved_pairs'] if values['resolved_pairs'] else None}
    return dict(counts=dict(counts), precision_resolved=ratio('correct_pairs', 'resolved_pairs'),
        attributable_pair_coverage=ratio('resolved_pairs', 'accepted_pairs'),
        current_attribution_coverage=ratio('attributed_current_sources', 'sources'),
        identifiable_instance_recall=ratio('recovered_instances', 'eligible_instances'),
        velocity_error_mps_correct_pairs=(dict(mean=float(np.mean(errors)), median=float(np.median(errors)),
                                              p90=float(np.quantile(errors, .9))) if errors else None),
        strata={k: finish(v) for k, v in sorted(buckets.items())},
        caveat='Conditional box-attributed identity metrics, not perfect voxel-instance GT. Ambiguous/merged support remains unresolved; repeated windows are not independent.')


def boxes_for_sample(nusc, token):
    from real_motion.nuscenes_adapter import category_to_dynamic_class
    rows = []
    for annotation in nusc.get('sample', token)['anns']:
        ann = nusc.get('sample_annotation', annotation)
        cls = category_to_dynamic_class(ann['category_name'])
        if cls is None: continue
        width, length, height = ann['size']
        rows.append(dict(instance_token=str(ann['instance_token']), class_id=cls,
            center=ann['translation'], size_lwh=[length, width, height],
            rotation=pose_matrix([0, 0, 0], ann['rotation'])[:3, :3]))
    return rows
