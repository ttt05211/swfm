"""Scientific/gradient safety for online one-stage source-linked columns."""
import copy
import json
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from test_causal_columns import scene_fixture, fake_provider, moving_fixture
from real_motion.motion_transport import FEATURE_DIM
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.joint_causal_columns import JointCausalColumns, PROTOCOL, CONTRACT, LINK_PROTOCOL
from real_motion.causal_column_completion import action_targets
from real_motion.causal_column_model import column_loss
from tools.real_motion import joint_column_common as common
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint


def fixture():
    torch.set_num_threads(1); torch.manual_seed(47)
    prep, grid, cfg = scene_fixture()
    mc = LocalSTWMV17Config(d_model=8, semantic_dim=4, heads=2, blocks=1, decoder_blocks=1)
    joint = JointCausalColumns(mc, cfg); control = copy.deepcopy(joint.transport)
    tube = torch.full((1, 6, 20, 20), 17, dtype=torch.uint8); tube[:, :, 10, 10] = 4
    mask = (tube == 4).to(torch.uint8)
    record = dict(features=torch.randn(1, FEATURE_DIM), local_semantic_tube=tube,
        kta_displacement_xy_m=torch.tensor([[[2., 0.]]*6]), frame_motion_features=torch.zeros(1, 6, 5),
        target_source_mask_tube=mask, target_source_residual_xy_m=torch.tensor([[[.5, .3]]*6]),
        target_source_displacement_xy_m=torch.tensor([[[2.5, .3]]*6]), target_yaw_rad=torch.full((1, 6), .05),
        yaw_enabled=torch.ones(1, dtype=torch.bool), yaw_label_valid=torch.ones(1, 6, dtype=torch.bool),
        se2_target_valid=torch.ones(1, 6, dtype=torch.bool), existence=torch.ones(1, 6),
        supervised_source=torch.ones(1, dtype=torch.bool), source_class_id=torch.tensor([4]),
        source_centroid_xy_t0_m=torch.tensor([[4.5, 6.5]]), anchors_xy_t0_m=torch.tensor([[[6.5, 6.5]]*6]))
    return prep, grid, joint, control, record


def optimizers(joint, control):
    opt = torch.optim.AdamW([{'params': joint.transport.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4, 'weight_decay': 1e-4},
        {'params': joint.columns.parameters(), 'lr': 3e-4, 'initial_lr': 3e-4, 'weight_decay': .01}])
    co = torch.optim.AdamW(control.parameters(), lr=5e-4, weight_decay=1e-4); co.param_groups[0]['initial_lr'] = 5e-4
    return opt, co


def provider_for(prep, grid, joint):
    def prepare(source, record, *, include_gt, raw_window=None, outputs=None):
        result = copy.deepcopy(prep)
        result.outputs = outputs if outputs is not None else joint.motion(record, torch.device('cpu'))
        return result
    return SimpleNamespace(device=torch.device('cpu'), pcfg=SimpleNamespace(grid=grid), prepare_columns=prepare)


def test_random_initialization_paired_identical_and_future_labels_not_inputs():
    prep, grid, joint, control, rec = fixture(); joint.eval()
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in joint.transport.state_dict().items())
    out = joint.motion(rec, torch.device('cpu'))
    assert torch.count_nonzero(out['residual_xy_m']) == 0 and torch.count_nonzero(out['yaw_delta_rad']) == 0
    altered = copy.deepcopy(rec)
    for key in ('target_yaw_rad', 'target_source_residual_xy_m', 'target_source_displacement_xy_m', 'existence'):
        altered[key] += 100
    altered['future_gt_occ'] = np.zeros((6, 16, 12, 2), np.uint8)
    assert all(torch.equal(v, joint.motion(altered, torch.device('cpu'))[k]) for k, v in out.items())


def test_column_only_gradient_reaches_transport_decoder_not_hard_xy_yaw_heads():
    prep, grid, joint, control, rec = fixture()
    # Conservative KEEP/empty zero-initialized output weights delay encoder
    # gradient until step2. Exercise the established nonzero-weight path.
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    prep.outputs = joint.motion(rec, torch.device('cpu'))
    batch = common.online_columns(prep, joint.columns, grid, np.random.default_rng(8), torch.device('cpu'))
    g, r = joint.columns(**{k: batch[k] for k in (*common.FEATURE_KEYS, 'source_features')})
    loss, _ = column_loss(joint.columns, g, r, batch['kind'], batch['legal'], batch['target'], batch['weight'])
    loss.backward()
    assert joint.columns.source_projection.weight.grad.abs().sum() > 0
    linked = [(k, p.grad) for k, p in joint.transport.named_parameters() if p.grad is not None]
    assert any('decoder' in k and g.abs().sum() > 0 for k, g in linked)
    assert any('spatial' in k and g.abs().sum() > 0 for k, g in linked)
    assert joint.transport.residual_head.weight.grad is None
    assert joint.transport.yaw_head.weight.grad is None
    assert len(batch['kind']) <= 256


def test_actor_source_identity_zero_static_and_fail_closed():
    prep, grid, joint, control, rec = fixture()
    prep.outputs = joint.motion(rec, torch.device('cpu'))
    plan = common.candidate_plan(prep, 2, grid, joint.columns.config)
    features = joint.columns.source_features_for(prep, 2, plan, torch.device('cpu'))
    assert torch.count_nonzero(features[plan.actor < 0]) == 0
    assert torch.equal(features[plan.actor >= 0], prep.outputs['future_transport_queries'][plan.actor[plan.actor >= 0], 2])
    bad = plan.subset([np.flatnonzero(plan.actor >= 0)[0]]); bad.actor[:] = 1
    with pytest.raises(RuntimeError, match='actor/source'): joint.columns.source_features_for(prep, 2, bad, torch.device('cpu'))
    prep.outputs = None
    with pytest.raises(RuntimeError, match='live'): joint.columns.source_features_for(prep, 2, plan, torch.device('cpu'))


def test_online_labels_change_with_gt_not_features():
    prep, grid, joint, control, rec = fixture(); prep.outputs = joint.motion(rec, torch.device('cpu'))
    a = common.online_columns(prep, joint.columns, grid, np.random.default_rng(8), torch.device('cpu'))
    other = copy.copy(prep); other.raw = {**prep.raw, 'future_gt_occ': [np.full_like(x, 17) for x in prep.raw['future_gt_occ']]}
    # Request every query: no positive-conditioned subsampling identity confound.
    with patch.object(common, 'sample_queries', side_effect=lambda plan, labels, budget, rng:
            (np.arange(min(len(plan), budget)), np.ones(min(len(plan), budget)))):
        a = common.online_columns(prep, joint.columns, grid, np.random.default_rng(8), torch.device('cpu'))
        b = common.online_columns(other, joint.columns, grid, np.random.default_rng(8), torch.device('cpu'))
    assert any(not torch.equal(a[k], b[k]) for k in ('target',))
    assert all(torch.equal(a[k], b[k]) for k in (*common.FEATURE_KEYS, 'source_features'))


def test_motion_objective_matches_original_and_unsupervised_safe():
    from tools.real_motion.train_p0_f9_v18_se2_pair import se2_objective_loss
    prep, grid, joint, control, rec = fixture(); out = joint.motion(rec, torch.device('cpu'))
    batch = {**rec, 'target_valid': rec['se2_target_valid']}
    a, _ = common.motion_loss(out, rec, torch.device('cpu'))
    b, _ = se2_objective_loss(out, batch, yaw_weight=19., shape_weight=.25, patch_resolution_m=.8)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-6)
    ga = torch.autograd.grad(a, tuple(joint.parameters()), retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(b, tuple(joint.parameters()), allow_unused=True)
    assert all((x is None and y is None) or torch.allclose(x, y, atol=1e-7) for x, y in zip(ga, gb))
    rec['supervised_source'][:] = False
    zero, _ = common.motion_loss(joint.motion(rec, torch.device('cpu')), rec, torch.device('cpu'))
    assert zero.item() == 0


def test_two_update_joint_step_and_resume_exact():
    prep, grid, joint, control, rec = fixture(); provider = provider_for(prep, grid, joint)
    opt, co = optimizers(joint, control); rng = np.random.default_rng(29)
    initial = copy.deepcopy(joint.state_dict()); initial_control = copy.deepcopy(control.state_dict())
    first = common.train_window(joint, control, opt, co, provider, None, rec, None, rng, 1, 3, probe=True)
    assert first['source_query_gradient_norm'] == 0  # conservative zero head initialization
    snapshot = copy.deepcopy((joint.state_dict(), control.state_dict(), opt.state_dict(), co.state_dict(), rng.bit_generator.state))
    state = torch.get_rng_state()
    second = common.train_window(joint, control, opt, co, provider, None, rec, None, rng, 2, 3, probe=True)
    expected = copy.deepcopy((joint.state_dict(), control.state_dict()))
    assert second['source_query_gradient_norm'] > 0 and np.isfinite(second['loss'])
    assert not torch.equal(initial['columns.refinement.weight'], expected[0]['columns.refinement.weight'])
    assert not torch.equal(initial_control['residual_head.weight'], expected[1]['residual_head.weight'])
    joint.load_state_dict(snapshot[0]); control.load_state_dict(snapshot[1]); opt.load_state_dict(snapshot[2]); co.load_state_dict(snapshot[3])
    rng.bit_generator.state = snapshot[4]; torch.set_rng_state(state)
    repeated = common.train_window(joint, control, opt, co, provider, None, rec, None, rng, 2, 3, probe=True)
    assert repeated == second
    assert all(torch.equal(v, joint.state_dict()[k]) for k, v in expected[0].items())
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in expected[1].items())


def test_empty_legal_window_not_false_success_or_parameter_decay():
    prep, grid, joint, control, rec = fixture(); opt, co = optimizers(joint, control)
    for key, value in rec.items():
        if isinstance(value, torch.Tensor): rec[key] = value[:0]
    before = copy.deepcopy(joint.state_dict()); control_before = copy.deepcopy(control.state_dict())
    provider = provider_for(prep, grid, joint)
    with patch.object(common, 'online_columns', return_value=None):
        stats = common.train_window(joint, control, opt, co, provider, None, rec, None, np.random.default_rng(8), 1, 3, probe=True)
    assert not stats['optimizer_updated'] and stats['sources'] == 0 and stats['sampled_columns'] == 0
    assert all(torch.equal(v, joint.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in control_before.items())
    assert not opt.state and not co.state


def test_strict_persisted_joint_protocol_and_no_failed_deployment(tmp_path):
    prep, grid, joint, control, rec = fixture(); path = tmp_path/'joint.pt'
    ck = dict(protocol=PROTOCOL, training_contract=CONTRACT, source_link=LINK_PROTOCOL,
        reference_checkpoint_sha256='a'*64, runtime_config_fingerprint='b'*64, checkpoint_role='calibrated_candidate',
        mode='screen', screen_pass=True, successful_updates=2, model_configs=joint.configs(), state_dict=joint.state_dict(),
        thresholds=[.5, .75, None], TRAIN_weights={'generation_pos_weight': 1., 'refine_class_weights': [1., 1., 1.]})
    torch.save(ck, path)
    _, restored = load_joint(path, torch.device('cpu'), reference_sha='a'*64, config_sha='b'*64)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in joint.state_dict().items())
    for field, value, match in [('screen_pass', False, 'deployed'), ('mode', 'smoke', 'deployed'),
            ('protocol', 'legacy_frozen_columns', 'contract'), ('thresholds', [None]*3, 'disabled')]:
        torch.save({**ck, field: value}, path)
        with pytest.raises(RuntimeError, match=match): load_joint(path, torch.device('cpu'), reference_sha='a'*64, config_sha='b'*64)


def test_current_motion_rerender_updates_geometry_and_supervision_without_detaching_latents():
    from real_motion.rigid_transport import RasterizedRigidComponent
    from real_motion.strong_w2det import StrongW2DetConfig
    from tools.real_motion import causal_column_common as cc
    prep, grid, joint, control, rec = fixture()
    current = prep.state['current'][0]; idx = current['voxel_indices']
    prior = RasterizedRigidComponent(4, idx, len(idx)); anchor = prep.raw['history_occ'][-1]
    state = {**prep.state, 'rec': rec, 'future_poses': prep.raw['future_poses'],
        'anchors': [anchor.copy() for _ in range(6)], 'baseline_by_hi': [[prior]]*6,
        'baseline_clear_flat_by_hi': [np.ravel_multi_index(idx.T, grid.shape_hwd)]*6,
        'source_world_points': [np.array([[4.5, 6.5, .5]])], 'source_rel_xy': [np.zeros((1, 2))],
        'source_z_t0': np.array([.5]), 'velocities': {0: np.array([4., 0, 0])}, 'gpu': None}
    provider = cc.FrozenColumns.__new__(cc.FrozenColumns)
    provider.pcfg = SimpleNamespace(grid=grid, free_label=17, frame_dt_s=.5)
    provider.device = torch.device('cpu'); provider.workers = 1; provider.strong = StrongW2DetConfig()
    provider.model = joint.transport; provider.columns_checked = True
    output = joint.motion(rec, torch.device('cpu'))
    changed = {**output, 'residual_xy_m': output['residual_xy_m']+torch.tensor([0., 1.])}
    with patch.object(cc, 'window_from_record', return_value=prep.window), patch.object(cc.runtime, '_prepare_record', return_value=state):
        a = provider.prepare_columns(None, rec, include_gt=True, raw_window=prep.raw, outputs=output)
        b = provider.prepare_columns(None, rec, include_gt=True, raw_window=prep.raw, outputs=changed)
        assert a.baseline[0][6, 6, 0] == 4 and b.baseline[0][6, 7, 0] == 4
        assert a.owners[0][6, 6, 0] == 0 and b.owners[0][6, 6, 0] == -1
        assert a.outputs['future_transport_queries'] is output['future_transport_queries']
        # This toy is only one voxel (below Strong's detection threshold).
        # Supply the fixture's verified causal historical registrations; the
        # actual current prediction-dependent renderer/planner are unchanged.
        a.registrations = b.registrations = prep.registrations
        pa, pb = (cc.candidate_plan(p, 1, grid, joint.columns.config) for p in (a, b))
        ta, tb = (action_targets(p, prep.raw['future_gt_occ'][1]) for p in (pa, pb))
        assert not np.array_equal(pa.flat, pb.flat) or not np.array_equal(ta, tb)
        with pytest.raises(RuntimeError, match='future occupancy'):
            provider.prepare_columns(None, rec, include_gt=False, raw_window=prep.raw, outputs=output)


@pytest.mark.parametrize('adaptive', (False, True))
def test_complete_cli_smoke_resume_real_losses_sampler_calibration_evaluation(tmp_path, adaptive):
    from tools.real_motion import train_p0_f9_joint_causal_columns as trainer
    from tools.real_motion import causal_column_common as cc
    from tools.real_motion import joint_column_full_common as full_common
    from real_motion.strong_w2det import StrongW2DetConfig
    prep, grid, sample_joint, _, template = fixture()
    train = [{**copy.deepcopy(template), 'scene_name': f'train{s}', 't0_token': f'{s}:{i}'} for s in range(10) for i in range(4)]
    dev = [{**copy.deepcopy(template), 'scene_name': 'dev', 't0_token': f'd{i}'} for i in range(64)]
    files = {k: tmp_path/k for k in ('train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint', 'train-info', 'dev-info')}
    for f in files.values(): f.write_bytes(b'original')
    torch.save({'model_config': asdict(sample_joint.transport.config), 'yaw_weight': 19.}, files['base-checkpoint'])
    originals = {k: f.read_bytes() for k, f in files.items()}
    manifest = dict(parent_keys=[('dev', f'd{i}') for i in range(512)], selected_key_fingerprint=trainer.DEV64_FP, manifest_fingerprint='c'*64)
    devkeys = [(r['scene_name'], r['t0_token']) for r in dev]
    pcfg = SimpleNamespace(grid=grid)
    def make_provider(checkpoint, expected_sha, cfg, device, workers, joint, control):
        result = provider_for(prep, grid, joint); result.workers = workers; result.reference_enabled = False
        result.strong = StrongW2DetConfig()
        result.control = control; result.joint = joint; result.model = joint.transport
        def prepare(source, record, *, include_gt, raw_window=None, outputs=None):
            row = copy.deepcopy(prep); row.window.scene_name = record['scene_name']; row.window.t0_token = record['t0_token']
            row.outputs = outputs if outputs is not None else result.joint.motion(record, device)
            return row
        result.prepare_columns = prepare
        result.reference_predictions = lambda p, r: {'frozen_E14': p.baseline, 'paired_scratch_V18_only': p.baseline} if result.reference_enabled else {}
        return result
    # Keep model widths tiny, all other actual orchestration/loss/report paths run.
    def build(mc, config, **extra): return JointCausalColumns(mc, sample_joint.columns.config, **extra)
    def run(out, resume=None, adaptive_arm=adaptive):
        argv = ['train', '--config', str(Path(__file__).resolve().parents[1]/'configs/real_motion_occfm.yaml'),
            '--dataroot', str(tmp_path), '--out-dir', str(out), '--mode', 'smoke', '--device', 'cpu', '--cpu-workers', '1']
        for k, f in files.items(): argv += ['--'+k, str(f)]
        if resume: argv += ['--resume', str(resume)]
        if adaptive_arm: argv += ['--adaptive-refine', '--causal-geometry-cache', str(tmp_path/'fixed-cache')]
        with patch('sys.argv', argv), patch.object(trainer, 'make_prepare_config', return_value=pcfg), \
            patch.object(trainer, 'load_manifest', return_value=(manifest, devkeys, None)), \
            patch.object(trainer, 'load_cache', side_effect=[({}, train), ({}, dev)]), \
            patch.object(trainer, 'JointCausalColumns', side_effect=build), \
            patch.object(trainer, 'JointColumnProvider', side_effect=make_provider), \
            patch.object(full_common, 'FullJointColumnProvider', side_effect=make_provider), \
            patch.object(trainer, 'validate_clean_e14_checkpoint', return_value='a'*64), \
            patch.object(trainer, 'NuScenesWindowSource', return_value=SimpleNamespace(nusc=None)), \
            patch.object(cc, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
            trainer.main()
    out = tmp_path/'run'; run(out)
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert summary['successful_updates'] == 2 and summary['train_windows'] == 4
    assert summary['training_window_passes'] == .5 and summary['sampled_columns'] <= 512
    assert summary['gradient_link_observed'] and not summary['screen_pass']
    assert summary['evaluation']['all']['windows'] == 2
    assert set(summary['evaluation']['all']['reference_metrics']) == {'frozen_E14', 'paired_scratch_V18_only'}
    assert summary['route'] == 'smoke_only_not_effectiveness_evidence'
    if adaptive:
        from tools.real_motion.compare_p0_f9_adaptive_refine import compare
        local_out = tmp_path/'local'; run(local_out, adaptive_arm=False)
        local = json.loads((local_out/'summary.json').read_text(encoding='utf-8'))
        comparison, text = compare(local, summary)
        assert not comparison['pass_gate'] and 'fixed0.5 diagnostic_refine' in text
        with pytest.raises(RuntimeError, match='budget mismatch: seed'): compare({**local, 'seed': 1}, summary)
    assert {k: f.read_bytes() for k, f in files.items()} == originals
    assert sorted(p.name for p in out.glob('*.pt')) == ['candidate.pt', 'last.pt']
    resumed = tmp_path/'resume'; run(resumed, out/'last.pt')
    other = json.loads((resumed/'summary.json').read_text(encoding='utf-8'))
    assert summary['evaluation'] == other['evaluation'] and summary['thresholds'] == other['thresholds']
    with pytest.raises(RuntimeError, match='resume population'): run(tmp_path/'bad', out/'candidate.pt')
