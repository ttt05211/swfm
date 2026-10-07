import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from concurrent.futures import ThreadPoolExecutor

from real_motion.canonical_causal_repair import CanonicalRepairHead
from tools.real_motion.point_ccr_v18_fps_common import (
    ARMS, POINT_PROTOCOL, select_population, load_point_head,
    assert_prior_exact, aggregate, forecast, rebuild_prior, resolve_arm_models,
)


def test_resolver_uses_saved_reference_not_provider_live_four_history_model():
    clean = SimpleNamespace(config=SimpleNamespace(history_frames=6))
    live = SimpleNamespace(config=SimpleNamespace(history_frames=4))
    teacher = SimpleNamespace(transport=live)
    provider = SimpleNamespace(reference=clean, model=live, joint=teacher)
    models = resolve_arm_models(provider, teacher)
    assert models['clean_e14_6h_original'] is clean
    assert models['clean_e14_6h_native'] is clean
    assert all(models[k] is live for k in ARMS if not k.startswith('clean'))
    assert provider.model is live  # Never break the CCR renderer/preparation.
    provider.reference = live
    with pytest.raises(RuntimeError, match='distinct'): resolve_arm_models(provider, teacher)
    provider.reference = clean; clean.config.history_frames = 4
    with pytest.raises(RuntimeError, match='identity'): resolve_arm_models(provider, teacher)
    clean.config.history_frames = 6; live.config.history_frames = 6
    with pytest.raises(RuntimeError, match='identity'): resolve_arm_models(provider, teacher)
    live.config.history_frames = 4; provider.model = clean
    with pytest.raises(RuntimeError, match='live'): resolve_arm_models(provider, teacher)


def records():
    return [dict(scene_name=scene, t0_token=str(i), features=np.zeros((n, 2)))
            for scene, i, n in [('b', 0, 1), ('b', 1, 99), ('a', 0, 2), ('a', 1, 3), ('c', 0, 4)]]


def test_population_is_fixed_unique_scene_balanced_with_causal_stress():
    pool = records(); keys = [[r['scene_name'], r['t0_token']] for r in pool]
    chosen, meta = select_population(pool, keys, windows=4, stress_windows=1)
    assert [r['scene_name'] for r in chosen] == ['a', 'b', 'c', 'b']
    assert meta[-1]['sources'] == 99 and meta[-1]['stratum'] == 'high_source_stress'
    assert select_population(pool[::-1], keys, windows=4, stress_windows=1)[1] == meta
    with pytest.raises(RuntimeError, match='duplicate'):
        select_population(pool, keys+[keys[0]], windows=4, stress_windows=1)
    with pytest.raises(RuntimeError, match='missing'):
        select_population(pool[:-1], keys, windows=4, stress_windows=1)
    with pytest.raises(ValueError): select_population(pool, keys, windows=4, stress_windows=4)


def saved_head():
    head = CanonicalRepairHead(source_dim=8)
    contract = dict(protocol=POINT_PROTOCOL, teacher_sha256='teacher', config_fingerprint='cfg',
        model=dict(mode='point_CCR', source_dim=8, width=64),
        thresholds=dict(CCR_ADD=.5, CCR_REMOVE=.95), epochs=3,
        epoch_batches=[2, 2, 2], epoch_batch_sizes=[[4, 2]]*3, schedule_steps=6)
    return dict(protocol=POINT_PROTOCOL, contract=contract, head=head.state_dict(),
        transport_frozen=True, deployable=False, epoch=3, batch=0, updates=6, executed=18,
        reports=dict(train_prior=dict(positive_weights=head.positive_weight.tolist())))


def load_saved(saved):
    return load_point_head(saved, teacher_sha256='teacher', config_fingerprint='cfg', source_dim=8, device='cpu')


def test_read_only_diagnostic_loader_accepts_completed_epoch_boundary_only():
    saved=saved_head()
    saved['epoch']=2; saved['batch']=0; saved['updates']=4; saved['executed']=12
    # Strict FPS/deployment loader still rejects an unfinished 3-pass schedule.
    with pytest.raises(RuntimeError,match='THREE-pass'):
        load_saved(saved)
    head=load_point_head(saved,teacher_sha256='teacher',config_fingerprint='cfg',
                         source_dim=8,device='cpu',allow_completed_epoch_boundary=True)
    assert not head.training and not any(p.requires_grad for p in head.parameters())
    saved['batch']=1; saved['updates']=5; saved['executed']=16
    with pytest.raises(RuntimeError,match='completed epoch boundary'):
        load_point_head(saved,teacher_sha256='teacher',config_fingerprint='cfg',
                        source_dim=8,device='cpu',allow_completed_epoch_boundary=True)


def test_completed_point_head_loader_does_not_restore_optimizer_or_rng():
    saved = saved_head(); saved['optimizer'] = {'must_never_restore': True}
    rng = torch.get_rng_state().clone()
    # Construction may consume the process's diagnostic RNG; saved training
    # RNG/optimizer are never restored. Weight equality, not RNG equality.
    head = load_saved(saved)
    assert not head.training and not any(p.requires_grad for p in head.parameters())
    assert all(torch.equal(v, saved['head'][k]) for k, v in head.state_dict().items())
    assert saved['optimizer'] == {'must_never_restore': True}
    assert rng.ndim == 1


@pytest.mark.parametrize('change', ['teacher', 'threshold', 'budget', 'missing_prior', 'correction', 'nan'])
def test_head_contract_fails_closed(change):
    saved = saved_head()
    if change == 'teacher': saved['contract']['teacher_sha256'] = 'different'
    elif change == 'threshold': saved['contract']['thresholds']['CCR_ADD'] = .75
    elif change == 'budget': saved['epoch'] = 2
    elif change == 'missing_prior': saved['reports'] = {}
    elif change == 'correction': saved['head']['positive_weight'][0, 0] = 2
    else: saved['head']['positive_weight'][0, 0] = float('nan')
    with pytest.raises(RuntimeError): load_saved(saved)


def test_fps_is_six_divided_by_mean_latency_not_mean_of_fps():
    rows = [dict(arm=ARMS[0], boundary='fresh_prior', seconds=t,
            six_complete_dense=True, stages_seconds={'forward': t/2}, key=str(i), stratum='representative')
            for i, t in enumerate((1., 3.))]
    value = aggregate(rows)['fresh_prior'][ARMS[0]]
    assert value['FPS'] == 3 and value['mean_six_ms'] == 2000
    assert value['FPS'] != np.mean([6., 2.])
    with pytest.raises(RuntimeError): aggregate([{**rows[0], 'six_complete_dense': False}])
    with pytest.raises(RuntimeError): aggregate([{**rows[0], 'seconds': float('nan')}])


def test_prior_exactness_includes_classes_counts_and_all_six_horizons():
    row = SimpleNamespace(class_id=4, source_voxel_count=1, voxel_indices=np.array([[1, 2, 0]]))
    state = dict(anchors=[np.zeros((3, 3, 1), np.uint8) for _ in range(6)],
                 baseline_clear_flat_by_hi=[np.array([5]) for _ in range(6)],
                 baseline_by_hi=[[copy.deepcopy(row)] for _ in range(6)])
    assert_prior_exact(state, copy.deepcopy(state))
    bad = copy.deepcopy(state); bad['baseline_by_hi'][5][0].class_id = 7
    with pytest.raises(RuntimeError, match='class'): assert_prior_exact(bad, state)
    bad = copy.deepcopy(state); bad['baseline_by_hi'].pop()
    with pytest.raises(RuntimeError, match='six|horizons'): assert_prior_exact(state, bad)


def test_no_future_gt_allowed_even_before_timer():
    with pytest.raises(RuntimeError, match='future occupancy'):
        forecast({'raw': {'future_gt_occ': [np.zeros(1)]}}, None, None, None, native=False, boundary='fresh_prior')


@pytest.mark.skipif(not os.getenv('SWFM_CCR_REPLAY'), reason='explicit real replay path required')
def test_real_cuda_replay_all_six_outputs_and_probabilities_exact(tmp_path, monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    from real_motion.local_replay_bundle import ReplayBundle, file_digest
    from real_motion.native_column_cpu import prepare_native,get_prepared_native
    from real_motion.runtime_config import make_prepare_config
    from tools.real_motion.pilot_p0_f9_canonical_causal_repair import load_exported_config
    from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
    from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
    from tools.real_motion.joint_column_full_common import build_fixed_geometry
    from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256
    from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
    torch.set_num_threads(1); device = torch.device('cuda')
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND', 'numpy')
    bundle = ReplayBundle(os.environ['SWFM_CCR_REPLAY']);pool=ThreadPoolExecutor(max_workers=4)
    try:
        paths = {}
        for member in ('runtime.yaml', 'checkpoints/epoch_0019.pt', 'checkpoints/clean_e14.pt'):
            paths[Path(member).name] = tmp_path/Path(member).name
            bundle.copy_member(member, paths[Path(member).name])
        cfg = load_exported_config(paths['runtime.yaml'], bundle.manifest['config_fingerprint'])
        _, teacher = load_joint(paths['epoch_0019.pt'], device, reference_sha=CLEAN_SHA256,
                                config_sha=bundle.manifest['config_fingerprint'], allow_diagnostic=True)
        teacher.eval().requires_grad_(False)
        provider = PilotProvider(paths['clean_e14.pt'], CLEAN_SHA256, make_prepare_config(cfg), device, 2, teacher, None)
        models = resolve_arm_models(provider, teacher)
        assert models['clean_e14_6h_original'] is provider.reference
        assert provider.reference.config.history_frames == 6
        assert provider.model is teacher.transport and provider.model.config.history_frames == 4
        provider.reference.eval().requires_grad_(False)
        head = CanonicalRepairHead(teacher.columns.source_dim).to(device).eval().requires_grad_(False)
        point_path = Path(os.environ['SWFM_CCR_POINT_HEAD'])
        saved = torch.load(point_path, map_location='cpu', weights_only=True)
        assert saved['teacher_sha256'] == bundle.manifest['teacher_sha256']
        head.load_state_dict(saved['head'], strict=True)
        digest = file_digest(point_path)
        prepare_native(tmp_path/'native_build')
        seen = set()
        for index, meta in enumerate(bundle.manifest['windows']):
            wanted = (meta['split'], meta['stratum'] == 'representative')
            if wanted not in (('train', True), ('dev', False)) or wanted in seen: continue
            seen.add(wanted)
            record, raw, _ = bundle.window(index, labels=False); raw['future_gt_occ'] = None
            raw['_column_causal_preparation'] = build_fixed_geometry(raw, record, provider.pcfg,
                provider.strong, 2, teacher.columns.config)
            case = dict(record=record, raw=raw, gpu=runtime._gpu_inputs(record, device))
            native, _ = rebuild_prior(case, provider, native=True, backgrounds=True)
            dense, _ = rebuild_prior(case, provider, native=False, backgrounds=True)
            assert_prior_exact(native, dense)
            assert_prior_exact(dense, raw['_column_causal_preparation']['prepared_state'])
            output = runtime._model_forward(teacher.transport, case['gpu'], device, return_latents=True)
            provider.prepare_columns(None, record, include_gt=False, raw_window=raw, outputs=output)
            before = {k: v.clone() for k, v in teacher.transport.state_dict().items()}
            expected = {}
            for boundary in ('fresh_prior', 'cached_prior'):
                for arm in ARMS:
                    family = arm.rsplit('_', 1)[0]
                    result = forecast(case, provider, models[arm], head if arm.startswith('point') else None,
                                      native=arm.endswith('native'), boundary=boundary,
                                      kernels=get_prepared_native() if arm in ('point_ccr_4h_fused','point_ccr_4h_parallel') else None,
                                      executor=pool if arm=='point_ccr_4h_parallel' else None)
                    if family in expected: assert result['signature'] == expected[family]
                    else: expected[family] = result['signature']
                    assert result['six_complete_dense'] and result['seconds'] > 0
                    assert ('fresh_strong_prior' in result['stages_seconds']) == (boundary == 'fresh_prior')
            assert expected['clean_e14_6h']['motion'] != expected['epoch19_v18_4h']['motion']
            assert all(torch.equal(before[k], v) for k, v in teacher.transport.state_dict().items())
        assert len(seen) == 2
        assert file_digest(point_path) == digest
    finally: bundle.close();pool.shutdown(wait=True)
