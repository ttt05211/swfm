import copy
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import numpy as np
import pytest
import torch
from tools.pinned_source_track import geometry as g, run
from tools.unified_source_training import train as old, workflow as old_workflow, evaluation as old_eval
from tools.joint_occ_experiments import run as legacy
from real_motion.strong_w2det import match_instances, StrongW2DetConfig
from real_motion.geometry import OccupancyGrid
from tools.scene_ego_head.frozen_gt.engine import deterministic_entrance
from tools.scene_ego_head.diagnostics.engine import external_forward
from test_unified_source_tracks import component, real_fixture
from test_unified_source_training import training_row, metadata_args


def test_pair_math_exact_original_ties_classes_height_gate_and_missing():
    rng = np.random.default_rng(21)
    previous = [dict(class_id=int(c), centroid_world=x) for c, x in zip(rng.integers(4, 7, 32), rng.normal(size=(32, 3))*10)]
    current = [dict(class_id=int(c), centroid_world=x) for c, x in zip(rng.integers(4, 7, 40), rng.normal(size=(40, 3))*10)]
    for before, now in ((previous, current), ([], current), ([component(2), component(2)], [component(3)])):
        links, velocity = g.match_pair(before, now)
        expected = match_instances(before, now, .5, max_speed_mps=25.)
        assert set(velocity) == set(expected)
        for i in velocity:
            np.testing.assert_array_equal(velocity[i], expected[i])
            assert int(before[links[i]]['class_id']) == int(now[i]['class_id'])


def test_latest_pair_is_pinned_without_four_frame_reassociation_or_older_velocity_fallback():
    frames = [[component(1)], [component(2)], [component(3), component(3)], [component(4)]]
    track, velocity = g.build_track(frames, np.tile(np.eye(4), (4, 1, 1)), np.arange(4)*.5,
        OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(24, 8, 4)), StrongW2DetConfig())
    assert track.links[0, 2] == 0 and track.audit['pinned_pairs'] == 1
    np.testing.assert_array_equal(velocity[0], [2., 0., 0.])
    frames[2] = []
    track, velocity = g.build_track(frames, np.tile(np.eye(4), (4, 1, 1)), np.arange(4)*.5,
        OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(24, 8, 4)), StrongW2DetConfig())
    assert track.links[0, 2] == -1 and track.valid[0, 1] and 0 not in velocity
    assert not g.kta_from_velocity(frames[-1], velocity, np.eye(4)).any()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_live_pair_one_match_four_extractions_old_kta_and_six_strong_backgrounds_exact(device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1); original, rec, raw = real_fixture(device)
    pinned = g.PinnedPredictor(original.body, original.provider.pcfg, device, 1)
    count = dict(pair=0, extract=0); pair, extract = g.match_pair, g.runtime.extract_instances_cropped_exact
    def paired(*a, **kw): count['pair'] += 1; return pair(*a, **kw)
    def extracted(*a, **kw): count['extract'] += 1; return extract(*a, **kw)
    try:
        with torch.no_grad(), deterministic_entrance():
            base = original.prepare(rec, raw, times=np.arange(4)*.5)
            monkeypatch.setattr(g, 'match_pair', paired)
            monkeypatch.setattr(g.runtime, 'extract_instances_cropped_exact', extracted)
            row = pinned.prepare(rec, raw, times=np.arange(4)*.5)
            assert count == dict(pair=1, extract=4)
            for k in ('kta_displacement_xy_m', 'anchors_xy_t0_m', 'source_centroid_xy_t0_m'):
                np.testing.assert_array_equal(row[0][k], base[0][k])
            old_match = g.runtime.match_instances
            monkeypatch.setattr(g.runtime, 'match_instances', lambda *a, **kw: pytest.fail('matched a SECOND time'))
            actual = g.forward_pair(original.body, pinned.provider, row, np.tile(np.eye(4), (6, 1, 1)))
            assert count == dict(pair=1, extract=4)
            monkeypatch.setattr(g.runtime, 'match_instances', old_match)
            reference = external_forward(original.body, pinned.provider, row, np.tile(np.eye(4), (6, 1, 1)))
            for a, b in zip(actual['state']['column_backgrounds'], reference['state']['column_backgrounds']):
                np.testing.assert_array_equal(a, b)
            for k in pinned.dense(actual): np.testing.assert_array_equal(pinned.dense(actual)[k], pinned.dense(reference)[k])
    finally: pinned.close(); original.close()


def test_training_record_bytes_unchanged_on_cache_hits_and_future_fields_not_consumed():
    runner, row = training_row()
    try:
        history = g.PinnedHistory(runner.provider, ram_mib=1).get(row[0], row[1])
        poison = dict(row[1], future_poses=object(), future_gt_occ=object())
        second = g.PinnedHistory(runner.provider, ram_mib=0).build(poison)
        np.testing.assert_array_equal(history[0]['pinned_pair']['expected_kta'], second[0]['pinned_pair']['expected_kta'])
        def base(*a, **kw): yield [(row[0], row[1], row[2], history)]
        output = next(g.prefetch_adapter(base, 'unified_pair')())
        assert output[0][0] is row[0]
        assert output[0][1]['_column_causal_preparation']['registrations'] is history[0]['registrations']
    finally: runner.close()


def test_real_baseline_then_only_new_pair_partial_resume_contract_and_binding_restore(tmp_path, monkeypatch):
    torch.set_num_threads(1); runner, row = training_row(); p = runner.provider
    paths, args = metadata_args(tmp_path)
    torch.save({'model_config': asdict(runner.body.transport.config)}, paths['base-checkpoint'])
    train = [{**row[0], 'scene_name': 'TRAIN', 't0_token': f'train{i}', 'features': torch.zeros(1, 46)} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}', 'features': torch.zeros(1, 46)} for i in range(4369)]
    keys = [('DEV', r['t0_token']) for r in dev[:512]]
    monkeypatch.setattr(old, 'stc_inventory', lambda _: [])
    monkeypatch.setattr(legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(legacy, 'load_cache', lambda path: ({}, train if path == str(paths['train-cache']) else dev))
    monkeypatch.setattr(legacy, 'load_manifest', lambda _: ({'parent_keys': keys}, keys[:64], None))
    monkeypatch.setattr(legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(legacy, 'make_prepare_config', lambda _: p.pcfg)
    def provider(*args): p.joint = args[-2]; return p
    monkeypatch.setattr(legacy, 'PilotProvider', provider)
    monkeypatch.setattr(legacy, 'NuScenesWindowSource', lambda root, *, info_pkl: SimpleNamespace(nusc=None))
    class Disposable:
        kernels = pool = None
        def __init__(self, *a): pass
        def close(self): pass
    monkeypatch.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
    monkeypatch.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
    def caches(provider, *a):
        provider.ccr_history_cache = provider.ccr_val_history_cache = SimpleNamespace(stats=lambda: {}, close=lambda: None)
        return {'train': {'namespace': 'train'}, 'val': {'namespace': 'val'}}
    monkeypatch.setattr(legacy, 'attach_caches', caches)
    monkeypatch.setattr(legacy, 'epoch_groups', lambda records, *a: [[records[0]], [records[1]]])
    def prefetch(provider, source, groups, reader, **kw):
        for group in groups: yield [(r, row[1], row[2], reader.get(r, row[1])) for r in group]
    monkeypatch.setattr(legacy, 'prefetched_batches', prefetch)
    body = old.GTBody
    monkeypatch.setattr(old, 'GTBody', lambda c: body(c, feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32))
    def evaluator(root, arm, **kw):
        return lambda *a, **kw: dict(windows=64, track_arm=arm,
            variants={'gt': {n: dict(mIoU=20., IoU=30., MovingMicro=10.) for n in old_eval.ROUTES}}, trajectory={})
    monkeypatch.setattr(old, 'evaluator', evaluator)
    source = tmp_path/'reference'; source.mkdir()
    baseline_args = args+['--out-dir', str(source/'baseline'), '--track-arm', 'baseline']
    try:
        assert old.main(argv=baseline_args) == 0
        ref = old_workflow.completed(source/'baseline'); original_bytes = (source/'baseline/last.pt').read_bytes()
        parser_before, predictor_before, protocol_before = old.parser, old_eval.Predictor, old.PROTOCOL
        child = tmp_path/'candidate/unified_pair'
        pair_args = args+['--out-dir', str(child), '--track-arm', 'unified_pair']
        with run.training_scope(ref): assert old.main(argv=pair_args+['--max-updates', '1']) == 0
        assert old.parser is parser_before and old_eval.Predictor is predictor_before and old.PROTOCOL == protocol_before
        with run.training_scope(ref): assert old.main(argv=pair_args+['--resume', str(child/'last.pt')]) == 0
        final = torch.load(child/'last.pt', weights_only=False)
        assert final['updates'] == 2 and final['epoch'] == 1 and final['batch'] == 0
        assert final['random_initializer'] == ref['random_initializer']
        run.publish(child.parent, ref)
        assert json.loads((child.parent/'comparison.json').read_text())['baseline_retrained'] is False
        assert (source/'baseline/last.pt').read_bytes() == original_bytes
        bad = copy.deepcopy(ref); bad['random_initializer'] = 'wrong'
        with pytest.raises(RuntimeError, match='initializer'):
            with run.training_scope(bad): old.main(argv=args+['--out-dir', str(tmp_path/'bad'), '--track-arm', 'unified_pair'])
        assert old.parser is parser_before and old_eval.Predictor is predictor_before
        # Actual new outer entry: locate/read existing baseline, replay its exact
        # arguments, train ONLY a new arm, bind reference and skip on full resume.
        fields = vars(old_workflow.parser().parse_args(args+['--out-dir', str(source)]))
        for k in ('out_dir', 'resume_workflow', 'max_updates'): fields.pop(k)
        code = Path(run.__file__).resolve().parents[2]
        manifest = dict(protocol='matched_track_KTA_one_epoch_workflow_v1', args=fields,
            data={k: legacy.sha256(fields[k]) for k in ('config', 'train_cache', 'dev_cache', 'population_manifest',
                'base_checkpoint', 'train_info', 'dev_info')},
            implementation={str(f.relative_to(code)): legacy.sha256(f) for f in old.dependency_files()})
        (source/'workflow.json').write_text(json.dumps(manifest))
        assert run.reference(source)[1]['random_initializer'] == ref['random_initializer']
        controller = tmp_path/'single_new_run'
        assert run.main(argv=['--reference-dir', str(source), '--out-dir', str(controller)]) == 0
        candidate = controller/'unified_pair/last.pt'; candidate_bytes = candidate.read_bytes()
        assert run.main(argv=['--out-dir', str(controller), '--resume']) == 0
        assert candidate.read_bytes() == candidate_bytes and (source/'baseline/last.pt').read_bytes() == original_bytes
        assert not (controller/'baseline').exists() and not (controller/'unified_fit').exists()
    finally: runner.close()
