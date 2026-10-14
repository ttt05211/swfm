import copy
import hashlib
import json
import sys
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from tools.stc_noise_profile import statistics as s, data, run


def test_real_components_are_class_split_and_keep_tiny_errors():
    gt = np.full((8, 8, 4), 17, np.uint8); pred = gt.copy(); known = np.ones_like(gt, bool)
    pred[1:3, 1:4, 1] = 11  # six-voxel thin false surface
    pred[3, 1, 1] = 15     # touching, different semantic class -> separate component
    gt[6, 6, 2] = 4        # missed object
    gt[5, 5, 2] = 15; pred[5, 5, 2] = 16
    v = s.frame_statistics(pred, gt, known, origin=(0., 0., 0.), step=(1., 1., 1.))
    assert v['families']['false_occupancy']['voxels'] == 7
    assert v['families']['false_occupancy']['components'] == 2
    assert v['families']['false_occupancy']['component_sizes'] == {'6': 1, '1': 1}
    assert '11,0,2,3,1,6' in v['families']['false_occupancy']['component_joint']
    assert v['families']['missed_occupancy']['voxels'] == 1
    assert v['families']['wrong_class']['voxels'] == 1
    assert np.asarray(v['distance_confusion']).sum() == gt.size
    assert np.asarray(v['historical_unknown_confusion']).sum() == 0
    s.validate(v, gt.size)
    bad = copy.deepcopy(v); bad['families']['false_occupancy']['component_sizes']['6'] += 1
    with pytest.raises(ValueError, match='component'): s.validate(bad, gt.size)
    assert s.weighted_quantiles({'1': 9, '6': 1}) == [1, 1]
    model = s.noise_model(v)
    assert not model['simulator_implemented']
    support = np.asarray(model['condition_counts'])
    p = np.asarray(model['conditional_pred_given_GT_and_distance'])
    np.testing.assert_allclose(p.sum(-1)[support > 0], 1)
    assert 'torch' not in sys.modules


def test_static_temporal_compensates_ego_and_excludes_real_dynamic_changes():
    history = []
    for i in range(4):
        gt = np.full((8, 8, 4), 17, np.uint8); pred = gt.copy()
        pred[6-i, 2, 1] = 15
        gt[1+i, 5, 1] = 4; pred[1+i, 5, 1] = 17
        pose = np.eye(4); pose[0, 3] = i
        history.append(dict(gt=gt, pred=pred, pose=pose, objects={}))
    v = s.temporal_statistics(history, origin=(0., 0., 0.), step=(1., 1., 1.))
    assert v['static_patterns']['false_occupancy'][15] == 1
    assert sum(v['static_patterns']['missed_occupancy'][1:]) == 0
    assert all(sum(hist) == v['static_stable_overlap_voxels'] for hist in v['static_patterns'].values())


def test_object_local_residual_removes_motion_before_jitter_and_counts_dropout():
    history = []
    for i in range(4):
        gt = np.full((10, 8, 4), 17, np.uint8); pred = gt.copy()
        gt[2+i, 3, 1] = 4; pred[3+i, 3, 1] = 4
        annotations = [dict(instance_token='car', class_id=4, translation=[2.5+i, 3.5, 1.5],
                            rotation_matrix=np.eye(3), size_lwh=(4., 4., 2.))]
        objects, counts = s.object_statistics(gt, pred, np.eye(4), annotations, origin=(0., 0., 0.), step=(1., 1., 1.))
        np.testing.assert_allclose(objects['car']['offset'], [1., 0., 0.])
        assert counts['jointly_supported_boxes'] == 1
        history.append(dict(gt=gt, pred=pred, pose=np.eye(4), objects=objects))
    v = s.temporal_statistics(history, origin=(0., 0., 0.), step=(1., 1., 1.))
    assert v['dynamic']['delta_offset_xy_hist'][0] == 1  # object moves; residual does not
    assert v['dynamic']['missing_STC_patterns'][0] == 1
    history[-1]['objects']['car'].update(pred_voxels=0, offset=None)
    v = s.temporal_statistics(history, origin=(0., 0., 0.), step=(1., 1., 1.))
    assert v['dynamic']['missing_STC_patterns'][8] == 1 and v['dynamic']['residual_pair_tracks'] == 0


def prior_fixture():
    return dict(status='complete', windows=64, source_epoch=2, source_update=10292, scientific_updates=0,
        contract=dict(protocol='frozen_final_prior_strength_DEV64_c1_c05_v1',
                      keys=[['fit', str(i)] for i in range(64)], STC_receipt_sha256='receipt', body_sha256='body'))


def test_fit_split_never_claims_remaining_is_pristine_or_accepts_duplicate_keys():
    p = prior_fixture(); v = data.split(p, ['fit', 'remaining'])
    assert v['fit_scenes'] == ['fit'] and v['remaining_scenes'] == ['remaining']
    assert not v['remaining_scene_labels_read'] and not v['remaining_is_pristine_test']
    p['contract']['keys'][1] = p['contract']['keys'][0]
    with pytest.raises(ValueError, match='DEV64'): data.split(p, ['fit', 'remaining'])


def test_real_source_reads_only_fit_labels_and_no_future_quads(tmp_path):
    # Minimal real nuScenes raw-table fixture; source never imports SDK/Torch.
    root, compact = tmp_path/'nusc', tmp_path/'stc'; base = root/'v1.0-trainval'; base.mkdir(parents=True)
    tables = dict(scene=[dict(token='S', name='fit', first_sample_token='f0'),
                         dict(token='H', name='remaining', first_sample_token='h0')],
        sample=[], sensor=[dict(token='sensor', channel='LIDAR_TOP')],
        calibrated_sensor=[dict(token='cal', sensor_token='sensor')], sample_data=[], ego_pose=[],
        category=[], instance=[], sample_annotation=[])
    extraction = dict(files={})
    gt = np.full(s.SHAPE, 17, np.uint8); known = np.ones(s.SHAPE, bool)
    digest = hashlib.sha256(gt.tobytes()).hexdigest()
    for scene, stoken, prefix in (('fit', 'S', 'f'), ('remaining', 'H', 'h')):
        for i in range(4):
            token = prefix+str(i)
            tables['sample'].append(dict(token=token, scene_token=stoken, timestamp=i*500000,
                prev=prefix+str(i-1) if i else '', next=prefix+str(i+1) if i < 3 else ''))
            tables['sample_data'].append(dict(sample_token=token, is_key_frame=True, calibrated_sensor_token='cal', ego_pose_token=token))
            tables['ego_pose'].append(dict(token=token, rotation=[1., 0., 0., 0.], translation=[0., 0., 0.]))
            extraction['files'][scene+'/'+token] = digest
            if scene == 'fit':
                for directory in (root/'gts'/scene/token, compact/scene/token): directory.mkdir(parents=True)
                np.savez_compressed(root/'gts'/scene/token/'labels.npz', semantics=gt, mask_lidar=known)
                # Deliberately wrong mask: camera path reads semantics ONLY.
                np.savez_compressed(compact/scene/token/'labels.npz', semantics=gt, mask_camera=np.zeros_like(known))
    for name, value in tables.items(): (base/(name+'.json')).write_text(json.dumps(value))
    source = data.Source(root, compact, data.split(prior_fixture(), ['fit', 'remaining']), extraction)
    assert set(source.frames) == {'fit'} and all('remaining' not in v[0] for v in source.inventory)
    row = source.read(source.frames['fit'][0]); np.testing.assert_array_equal(row['gt'], gt)
    assert row['known'].all()
    with pytest.raises(RuntimeError, match='MUST NOT'): source.paths(SimpleNamespace(scene='remaining', token='h0'))
    path = compact/'fit/f0/labels.npz'
    pred = gt.copy(); pred[0, 0, 0] = 4
    np.savez_compressed(path, semantics=pred)
    with pytest.raises(RuntimeError, match='changed'): source.check_inventory()


def test_actual_cli_scene_boundary_resume_and_sources_read_only(tmp_path, monkeypatch):
    prior_path = tmp_path/'prior/evaluation.json'; prior_path.parent.mkdir()
    prior_path.write_text(json.dumps(prior_fixture())); before = prior_path.read_bytes()
    monkeypatch.setattr(run, 'receipt', lambda _: (dict(files={}), 'receipt', ['fit', 'remaining']))
    shape = (8, 8, 4); monkeypatch.setattr(s, 'SHAPE', shape)
    gt = np.full(shape, 17, np.uint8); pred = gt.copy(); pred[3, 3, 1] = 15
    frame_stats = s.frame_statistics(pred, gt, np.ones(shape, bool), origin=(0., 0., 0.), step=(1., 1., 1.))
    objects, object_counts = s.object_statistics(gt, pred, np.eye(4), [], origin=(0., 0., 0.), step=(1., 1., 1.))
    history = [dict(gt=gt, pred=pred, pose=np.eye(4), objects=objects)]*4
    temporal = s.temporal_statistics(history, origin=(0., 0., 0.), step=(1., 1., 1.))
    scene_stats = {}
    for _ in range(4): s.accumulate(scene_stats, frame_stats); s.accumulate(scene_stats, dict(objects=object_counts))
    s.accumulate(scene_stats, dict(temporal=temporal))
    frames = [SimpleNamespace(token=str(i)) for i in range(4)]
    class Source:
        metadata = []; inventory = []
        def __init__(self, *a): self.frames = dict(fit=frames)
        def check_inventory(self): pass
    monkeypatch.setattr(run, 'Source', Source)
    calls = []
    def profile(*args):
        calls.append(1)
        if args[-1].is_set(): raise InterruptedError('unfinished scene discarded')
        return dict(scene='fit', statistics=scene_stats, frame_ledger=[dict(token=str(i)) for i in range(4)], seconds=1.)
    monkeypatch.setattr(run, 'profile_scene', profile)
    out = tmp_path/'profile'
    argv = ['--prior-result', str(prior_path), '--dataroot', str(tmp_path/'nusc'), '--stc-root', str(tmp_path/'stc'),
            '--out-dir', str(out), '--workers', '1']
    stopped = threading.Event(); stopped.set()
    assert run.main(argv=argv, event=stopped) == 130
    assert json.loads((out/'state.json').read_text())['scenes'] == []
    assert run.main(argv=argv+['--resume']) == 0
    assert run.main(argv=argv+['--resume']) == 0 and len(calls) == 2
    assert prior_path.read_bytes() == before
    result = json.loads((out/'evaluation.json').read_text())
    model = json.loads((out/'noise_model.json').read_text())
    assert result['statistics']['frames'] == 4 and result['scientific_updates'] == 0
    assert 'seed_DEV64_keys' not in model['fitting_split'] and not model['simulator_implemented']
    assert 'CPU only' in (out/'summary.txt').read_text()
    with pytest.raises(FileExistsError): run.main(argv=argv)
    state_path = out/'state.json'; state = json.loads(state_path.read_text())
    state['scenes'][0]['statistics']['confusion'][0][0] += 1
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='totals'): run.main(argv=argv+['--resume'])
    assert 'torch' not in sys.modules
