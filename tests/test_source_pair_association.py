"""Box-based independent attribution, wrong identity, ambiguity and recall."""
import copy
import numpy as np
import pytest
from real_motion.geometry import OccupancyGrid
from tools.source_scene_stage1.association import attribute, inspect_pair, aggregate

GRID = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(32, 16, 4))


def component(x, cls=4):
    ijk = np.array([[x, 2, 0], [x, 2, 1]], np.int64)
    return dict(class_id=cls, voxel_indices=ijk, centroid_world=(ijk+.5).mean(0))


def box(x, token, cls=4):
    return dict(class_id=cls, instance_token=token, center=[x+.5, 2.5, 1.], size_lwh=[1.1, 1.1, 2.1], rotation=np.eye(3))


def test_correct_pair_velocity_error_and_distinct_recall():
    row = inspect_pair([component(2)], [component(3)], [box(2, 'A')], [box(3, 'A')],
                       np.eye(4), np.eye(4), GRID, actual_dt=.5, check_registration=False)
    report = aggregate([row])
    assert report['precision_resolved'] == report['identifiable_instance_recall'] == 1.
    assert report['velocity_error_mps_correct_pairs']['mean'] == 0.
    assert report['counts']['correct_pairs'] == 1


def test_crossing_wrong_identity_not_declared_correct_by_nearest_gt():
    before = [component(2), component(10)]; current = [component(9), component(3)]
    pb = [box(2, 'A'), box(10, 'B')]; cb = [box(9, 'A'), box(3, 'B')]
    row = inspect_pair(before, current, pb, cb, np.eye(4), np.eye(4), GRID, check_registration=False)
    assert [e['previous'] for e in row['edges']] == [1, 0]
    report = aggregate([row])
    assert report['precision_resolved'] == report['identifiable_instance_recall'] == 0.
    assert report['counts']['wrong_pairs'] == 2
    assert report['velocity_error_mps_correct_pairs'] is None
    assert all(v['precision_resolved'] == 0. for k, v in report['strata'].items() if k.startswith('GT_speed/'))


def test_split_merge_unassigned_coverage_not_false_high_accuracy():
    a, b = component(2), component(3)
    merged = dict(a, voxel_indices=np.concatenate((a['voxel_indices'], b['voxel_indices'])))
    rows = attribute([merged], [box(2, 'A'), box(3, 'B')], np.eye(4), GRID)
    assert rows[0]['token'] is None and rows[0]['status'] == 'merged_or_overlap'
    fragments = attribute([a, a], [box(2, 'A')], np.eye(4), GRID)
    assert [r['fragments_for_instance'] for r in fragments] == [2, 2]
    row = inspect_pair([a], [a], [], [], np.eye(4), np.eye(4), GRID, check_registration=False)
    report = aggregate([row])
    assert report['precision_resolved'] is None and report['attributable_pair_coverage'] == 0.
    assert report['counts']['unresolved_pairs'] == 1


def test_gate_unmatched_ego_transform_and_GT_boxes_cannot_change_edges():
    a, b = component(2), component(20)
    row = inspect_pair([a], [b], [box(2, 'A')], [box(20, 'A')], np.eye(4), np.eye(4), GRID, check_registration=False)
    assert row['edges'][0]['status'] == 'unmatched'
    assert aggregate([row])['identifiable_instance_recall'] == 0.
    old = copy.deepcopy(a); cur = copy.deepcopy(a)
    pose = np.eye(4); pose[0, 3] = 1.
    cur['centroid_world'] += [1., 0, 0]
    row = inspect_pair([old], [cur], [box(2, 'A')], [box(3, 'A')], np.eye(4), pose, GRID, check_registration=False)
    assert row['edges'][0]['correct'] and row['edges'][0]['KTA_velocity_error_mps'] == 0.
    other = inspect_pair([old], [cur], [], [], np.eye(4), pose, GRID, check_registration=False)
    assert other['edges'][0]['previous'] == row['edges'][0]['previous']
    with pytest.raises(ValueError): inspect_pair([], [], [], [], np.eye(4), np.eye(4), GRID, actual_dt=0)


def test_registration_acceptance_is_diagnostic_only(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('tools.source_scene_stage1.association.register_history_shape', lambda *a, **k: SimpleNamespace(accepted=False))
    row = inspect_pair([component(2)], [component(3)], [box(2, 'A')], [box(3, 'A')], np.eye(4), np.eye(4), GRID)
    assert row['edges'][0]['correct'] and not row['edges'][0]['registration_accepted']
    assert aggregate([row])['counts']['registration/correct/rejected'] == 1
