from types import SimpleNamespace
import pytest
from real_motion.stc_camera_protocol import STCFourSettingSource
from real_motion.v21_source_induction import select_scene_balanced_round_robin
from tools.ego_experiments.eval_surface_ego_three_population import select_dev512


def source(keys):
    value = object.__new__(STCFourSettingSource)
    value.by_key = {k:SimpleNamespace(scene=k[0],t0=k[1]) for k in keys}
    return value


def keys(n):return [(f'scene-{i//32:04d}',str(i)) for i in range(n)]


def test_full_parent512_same_exact_order():
    parent = keys(512)
    windows,audit = select_dev512(source(parent),dict(parent_keys=parent,selected_keys=parent[:64]))
    assert [(w.scene,w.t0) for w in windows] == parent
    assert audit['population_label'] == 'dev512' and audit['requested_windows'] == 512


def test_missing_planner_rows_disclosed_without_padding_or_resampling():
    parent = keys(512);available = parent[3:]+keys(600)[512:]
    windows,audit = select_dev512(source(available),dict(parent_keys=parent,selected_keys=parent[:64]))
    assert len(windows) == audit['actual_planner_covered_windows'] == 509
    assert audit['missing_parent_keys'] == [list(k) for k in parent[:3]]
    assert [(w.scene,w.t0) for w in windows] == parent[3:]
    assert audit['no_top_up_or_padding'] and audit['population_label'] == 'dev512_planner_covered_intersection'


def test_larger_parent_freezes512_before_filtering():
    parent = keys(4369);requested = select_scene_balanced_round_robin(parent,512)
    windows,audit = select_dev512(source(parent),dict(parent_keys=parent,selected_keys=parent[:64]))
    assert [(w.scene,w.t0) for w in windows] == list(requested)
    assert len(windows) == 512 and audit['parent_windows'] == 4369


def test_existing_selected512_not_replaced_by_parent_order():
    parent = keys(600);selected = select_scene_balanced_round_robin(parent,512)
    windows,audit = select_dev512(source(parent),dict(parent_keys=parent,selected_keys=selected))
    assert [(w.scene,w.t0) for w in windows] == list(selected)
    assert audit['selection_rule'] == 'existing_manifest_selected512'


def test_insufficient_manifest_fails_instead_of_relabeling64():
    with pytest.raises(RuntimeError,match='only 64 parent'):
        select_dev512(source(keys(64)),dict(parent_keys=keys(64),selected_keys=keys(64)))
