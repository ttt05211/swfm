"""Conditional GT identities, changed-only denominators and no invented matches."""
import copy
from types import SimpleNamespace
import numpy as np
from tools.causal_forward_train import audit
from tools.source_scene_stage2.track import build as old_build
from tools.causal_forward_track.track import build as new_build
from test_source_pair_association import component,box,GRID


def test_crossing_gt_audit_changed_subset_and_frozen_links():
    frames = [[component(0),component(10)],[component(2),component(8)],
        [component(4),component(6)],[component(4),component(6)]]
    poses = np.repeat(np.eye(4)[None],4,0);times = np.arange(4)*.5
    strong = SimpleNamespace(max_match_speed_mps=25.)
    old,new = old_build(frames,poses,times,strong),new_build(frames,poses,times,strong)
    old_links,new_links = old.links.copy(),new.links.copy()
    boxes = [[box(0,'A'),box(10,'B')],[box(2,'A'),box(8,'B')],
        [box(4,'A'),box(6,'B')],[box(6,'A'),box(4,'B')]]
    a,b = (audit.inspect(i,frames,poses,times,boxes,GRID) for i in (old,new))
    assert all(e['status']=='wrong_identity' for e in a['edges'])
    assert all(e['correct'] and e['KTA_velocity_error_mps']==0. for e in b['edges'])
    entry = dict(baseline=a,forward=b,changed=[0,1])
    report = audit.reports([dict(occ=entry,stc=copy.deepcopy(entry))])
    for kind in ('occ','stc'):
        assert report[kind]['all']['baseline']['precision_resolved']==0.
        assert report[kind]['changed_only']['forward']['identifiable_instance_recall']==1.
        assert report[kind]['changed_both_resolved']['wrong_to_correct']==2
        assert report[kind]['changed_both_resolved'].get('correct_to_wrong',0)==0
    np.testing.assert_array_equal(old.links,old_links);np.testing.assert_array_equal(new.links,new_links)
    unresolved = audit.inspect(new,frames,poses,times,[[],[],[],[]],GRID)
    assert all(e['status']=='unresolved' for e in unresolved['edges'])
    subset = audit.subset(b,[0]);assert len(subset['eligible_instances'])==len(subset['recovered_instances'])==1


def test_missing_tminus1_remains_unmatched_even_with_old_history_and_no_changes():
    frames = [[component(0)],[component(1)],[],[component(3)]]
    poses = np.repeat(np.eye(4)[None],4,0);times = np.arange(4)*.5
    identity = new_build(frames,poses,times,SimpleNamespace(max_match_speed_mps=25.))
    row = audit.inspect(identity,frames,poses,times,[[box(0,'A')],[box(1,'A')],[],[box(3,'A')]],GRID)
    assert row['edges'][0]['status']=='unmatched' and row['recovered_instances']==[]
    empty = audit.reports([dict(occ=dict(baseline=row,forward=row,changed=[]),
        stc=dict(baseline=row,forward=row,changed=[]))])
    assert empty['occ']['changed_only']['forward']['precision_resolved'] is None
