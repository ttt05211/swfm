"""Causal/refine-only contracts and real sampler/loss integration, not gain claims."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from test_joint_causal_columns import fixture, provider_for, optimizers
from real_motion.adaptive_column_context import (AdaptiveContextConfig, AdaptiveRefineContext,
    static_atlas, PROTOCOL, TRAINING_CONTRACT, LINK_PROTOCOL)
from real_motion.joint_causal_columns import JointCausalColumns
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.real_motion import joint_column_common as common
from tools.real_motion.causal_column_common import predict_probabilities
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint


def setup():
    prep, grid, base, _, record = fixture()
    torch.manual_seed(47)
    joint = JointCausalColumns(config_from_mapping_v17(base.configs()['motion']), base.columns.config, AdaptiveContextConfig())
    prep.outputs = joint.motion(record, torch.device('cpu'))
    batch = common.online_columns(prep, joint.columns, grid, np.random.default_rng(8), torch.device('cpu'))
    return prep, grid, base, joint, record, batch


def inputs(model, batch):
    return {k: batch[k] for k in (*common.FEATURE_KEYS, *model.extra_input_keys)}


def test_base_weights_rng_and_initial_prediction_are_exact():
    prep, grid, base, joint, record, batch = setup()
    for k, v in base.state_dict().items(): assert torch.equal(v, joint.state_dict()[k]), k
    base.eval(); joint.eval()
    a = base.columns(**inputs(base.columns, batch)); b = joint.columns(**inputs(joint.columns, batch))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    mc = config_from_mapping_v17(base.configs()['motion'])
    torch.manual_seed(123); JointCausalColumns(mc, base.columns.config); state = torch.get_rng_state()
    torch.manual_seed(123); JointCausalColumns(mc, base.columns.config, AdaptiveContextConfig())
    assert torch.equal(state, torch.get_rng_state())


def test_initial_context_exact_even_with_trained_heads_and_generation_never_uses_adapter():
    prep, grid, base, joint, record, batch = setup()
    for model in (base, joint):
        torch.manual_seed(91); torch.nn.init.normal_(model.columns.refinement.weight)
        torch.manual_seed(92); torch.nn.init.normal_(model.columns.generation.weight)
    a = base.columns(**inputs(base.columns, batch)); b = joint.columns(**inputs(joint.columns, batch))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    torch.nn.init.normal_(joint.columns.adaptive_context.static_out.weight)
    torch.nn.init.normal_(joint.columns.adaptive_context.dynamic_out.weight)
    b = joint.columns(**inputs(joint.columns, batch))
    assert torch.equal(a[0], b[0]) and not torch.equal(a[1], b[1])
    generate = batch['kind'] == 0
    assert torch.equal(a[1][generate], b[1][generate])


def test_atlas_unknown_is_not_free_and_partial_padding_coverage():
    m = np.full((5, 9, 2), 18, np.uint8); footprint = np.zeros((5, 9), bool)
    assert np.count_nonzero(static_atlas(m, footprint)) == 0
    m[4, 8, 1] = 13; footprint[4, 8] = True
    a = static_atlas(m, footprint)
    assert a.shape == (8, 2, 3) and a[5, -1, -1] == 1/16
    assert a[1, -1, -1] == 1/32 and a[3, -1, -1] == .75/16
    # Content outside historical footprint is excluded even if a value exists.
    m[0, 0, 0] = 11
    assert static_atlas(m, footprint)[0, 0, 0] == 0


def test_xy_flip_metres_and_same_actor_all_six_queries_no_gt_read():
    prep, grid, _, joint, record, batch = setup()
    plan = common.candidate_plan(prep, 3, grid, joint.columns.config)
    a = joint.columns.extra_inputs_for(prep, 3, plan, grid, torch.device('cpu'))
    assert np.allclose(a['lookup_xy'], ((plan.xy+.5)/np.array([16, 12])*2-1)[:, ::-1])
    assert np.allclose(a['lookup_scale'][0], [2/12, 2/16])
    assert torch.count_nonzero(a['source_sequence'][plan.actor < 0]) == 0
    assert torch.equal(a['source_sequence'][plan.actor >= 0], prep.outputs['future_transport_queries'][plan.actor[plan.actor >= 0]])
    other = copy.copy(prep); other.raw = {**prep.raw, 'future_gt_occ': [np.zeros_like(x) for x in prep.raw['future_gt_occ']]}
    b = joint.columns.extra_inputs_for(other, 3, plan, grid, torch.device('cpu'))
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    bad = plan.subset([np.flatnonzero(plan.actor >= 0)[0]]); bad.actor[:] = 100
    with pytest.raises(RuntimeError, match='actor/source'): joint.columns.extra_inputs_for(prep, 3, bad, grid, torch.device('cpu'))


def test_spatial_offsets_read_correct_axis_and_invalid_field_is_zero():
    adapter = AdaptiveRefineContext(8, 8, 2, AdaptiveContextConfig())
    with torch.no_grad():
        adapter.offsets.weight.zero_(); adapter.offsets.bias.zero_()
        adapter.static_out.weight.copy_(torch.eye(8))
        adapter.static_encoder = torch.nn.Identity()
    atlas = torch.zeros(1, 8, 3, 5); atlas[:, 5] = 1
    atlas[0, 0, 1, 3] = 7
    q = torch.zeros(1, 8); xy = torch.tensor([[2*(3+.5)/5-1, 2*(1+.5)/3-1]])
    got = adapter.spatial_context(q, atlas, torch.zeros(1, dtype=torch.long), xy, torch.ones(1, 2), torch.ones(1, 2))
    assert got[0, 0].item() == pytest.approx(7.)
    outside = adapter.spatial_context(q, atlas, torch.zeros(1, dtype=torch.long), xy+10, torch.ones(1, 2), torch.ones(1, 2))
    assert torch.count_nonzero(outside) == 0
    atlas.zero_(); adapter.static_encoder = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU())
    empty = adapter.spatial_context(q, atlas, torch.zeros(1, dtype=torch.long), xy, torch.ones(1, 2), torch.ones(1, 2))
    assert torch.count_nonzero(empty) == 0 and torch.isfinite(empty).all()


def test_gradient_to_all_same_source_horizons_and_spatial_offsets():
    prep, grid, _, joint, record, batch = setup()
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.1)
    torch.nn.init.normal_(joint.columns.adaptive_context.static_out.weight, std=.1)
    torch.nn.init.normal_(joint.columns.adaptive_context.dynamic_out.weight, std=.1)
    g, r = joint.columns(**inputs(joint.columns, batch))
    grad = torch.autograd.grad(r.square().mean(), prep.outputs['future_transport_queries'], retain_graph=True)[0]
    assert (grad.abs().sum(-1) > 0).all()
    r.square().mean().backward()
    assert joint.columns.adaptive_context.offsets.weight.grad.abs().sum() > 0
    assert joint.transport.residual_head.weight.grad is None and joint.transport.yaw_head.weight.grad is None


def test_two_real_training_updates_and_eval_chunk_invariance():
    prep, grid, _, joint, record, batch = setup()
    prep.outputs = None
    control = copy.deepcopy(joint.transport); provider = provider_for(prep, grid, joint)
    opt, co = optimizers(joint, control); rng = np.random.default_rng(5)
    for update in (1, 2):
        stats = common.train_window(joint, control, opt, co, provider, None, record, None, rng, update, 2, probe=True)
        assert np.isfinite(stats['loss']) and stats['optimizer_updated']
    assert stats['source_query_gradient_norm'] > 0
    prep.outputs = joint.motion(record, torch.device('cpu'))
    plan = common.candidate_plan(prep, 1, grid, joint.columns.config)
    a = predict_probabilities(joint.columns, prep, 1, plan, grid, torch.device('cpu'), batch_size=3)
    b = predict_probabilities(joint.columns, prep, 1, plan, grid, torch.device('cpu'), batch_size=256)
    assert np.allclose(a, b, atol=1e-6)


def test_context_checkpoint_roundtrip_and_protocol_cannot_be_swapped(tmp_path):
    _, _, _, joint, _, _ = setup(); path = tmp_path/'ctx.pt'
    ck = dict(protocol=PROTOCOL, training_contract=TRAINING_CONTRACT, source_link=LINK_PROTOCOL,
        reference_checkpoint_sha256='a'*64, runtime_config_fingerprint='b'*64, checkpoint_role='resume_last',
        mode='screen', screen_pass=False, successful_updates=2, model_configs=joint.configs(), state_dict=joint.state_dict(),
        TRAIN_weights={'generation_pos_weight': 1., 'refine_class_weights': [1., 1., 1.]})
    torch.save(ck, path)
    _, restored = load_joint(path, torch.device('cpu'), reference_sha='a'*64, config_sha='b'*64, allow_diagnostic=True)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in joint.state_dict().items())
    from real_motion.joint_causal_columns import PROTOCOL as OLD, CONTRACT as OC, LINK_PROTOCOL as OL
    torch.save({**ck, 'protocol': OLD, 'training_contract': OC, 'source_link': OL}, path)
    with pytest.raises(RuntimeError, match='model/protocol'): load_joint(path, torch.device('cpu'),
        reference_sha='a'*64, config_sha='b'*64, allow_diagnostic=True)
