"""Toy expressivity and effect safety, NOT real nuScenes quality tests."""
import inspect
import numpy as np
import pytest
import torch

from real_motion.sparse_evidence_repair import (CanonicalObservation, EvidenceMemory,
    SparseRepairHead, build_memory, render_add_only, FREE, STATIC)
from tools.real_motion.benchmark_sparse_repair_synthetic import (make_scene, column_inputs,
    column_probabilities, expressivity, quality_screen, source_context_from_observations)
from real_motion.causal_column_completion import ColumnConfig
from real_motion.joint_causal_columns import LinkedColumns


def observation(rows=(), visible=None):
    rows = np.asarray(rows, np.int64).reshape(-1, 5)
    return CanonicalObservation(rows, rows.copy() if visible is None else np.asarray(visible, np.int64).reshape(-1,5))


def memory_fixture():
    rows = [[0,4,1,1,1], [0,4,1,1,2], [1,5,1,1,1], [STATIC,11,3,2,0]]
    return build_memory([observation(rows), observation(rows), observation(), observation([rows[1]])])


def renderer_arguments(memory):
    return dict(memory=memory, probabilities=np.ones((len(memory),6)),
        baseline=np.full((6,6,6,5), FREE, np.uint8), source_centers=np.zeros((2,3)),
        future_centers=np.zeros((6,2,3)), yaw=np.zeros((6,2)),
        world_to_future=np.repeat(np.eye(4)[None],6,0), origin=np.zeros(3), step=np.ones(3))


def test_only_four_past_frames_accept_gt_free_full_z_class_actor_evidence():
    m = memory_fixture()
    assert len(m) == 4
    assert len(np.unique(m.keys[:,[0,2,3,4]], axis=0)) == 4  # sources never collapsed
    assert m.features().shape == (4,12)
    assert np.any(m.keys[:,4] == 2)  # vertical bins retained
    assert 'target' not in inspect.signature(build_memory).parameters
    assert 'gt' not in inspect.signature(render_add_only).parameters
    with pytest.raises(ValueError, match='four-history'):
        build_memory([observation()]*6)


def test_free_visible_is_not_evidence_and_unknown_is_not_free():
    rows = [[0,4,1,1,1], [0,4,2,1,1]]
    m = build_memory([observation([rows[0]], rows), observation(), observation(), observation((), rows)])
    assert len(m) == 1
    assert m.presence.tolist() == [[True,False,False,False]]
    assert m.visibility.tolist() == [[True,False,False,True]]
    with pytest.raises(ValueError, match='observed'):
        build_memory([observation(rows, ())]*4)


def test_static_semantic_disagreement_fails_closed_not_last_class_wins():
    rows = [[STATIC,11,1,1,1], [STATIC,13,1,1,1], [STATIC,15,1,1,2]]
    m = build_memory([observation(rows), observation(), observation(), observation()])
    assert m.ambiguous_voxels == 1
    assert m.keys.tolist() == [rows[-1]]
    with pytest.raises(ValueError, match='class changed'):
        build_memory([observation([[0,4,1,1,1], [0,5,1,1,2]])]*4)


@pytest.mark.parametrize('mode', SparseRepairHead.MODES)
def test_conservative_init_no_edit_and_no_unknown_invention(mode):
    m = memory_fixture()
    model = SparseRepairHead(mode)
    scores = model(**m.tensors(neighborhood=model.neighborhood), source_context=torch.randn(2,128), future_queries=torch.randn(2,6,128))
    assert torch.equal(scores, torch.full_like(scores,-4.))
    args = renderer_arguments(m)
    args['probabilities'] = scores.sigmoid().detach().numpy()
    out, stat = render_add_only(**args)
    assert np.array_equal(out, args['baseline']) and stat['added'] == 0


def test_cached_future_matches_repeated_math_and_live_gradients():
    torch.manual_seed(51)
    m = memory_fixture()
    cached, repeated = SparseRepairHead('cached_future'), SparseRepairHead('repeated_future')
    torch.nn.init.normal_(cached.score.weight, std=.1)
    repeated.load_state_dict(cached.state_dict())
    ctx, fut = torch.randn(2,128,requires_grad=True), torch.randn(2,6,128,requires_grad=True)
    a = cached(**m.tensors(), source_context=ctx, future_queries=fut)
    b = repeated(**m.tensors(), source_context=ctx, future_queries=fut)
    assert torch.allclose(a,b,rtol=1e-5,atol=1e-6)
    a.sum().backward()
    assert ctx.grad.abs().sum() > 0 and fut.grad.abs().sum() > 0
    assert cached.point[0].weight.grad.abs().sum() > 0


def test_free_only_original_occupancy_protected_source_collision_order_and_current_not_repaired():
    m = memory_fixture(); args = renderer_arguments(m)
    args['baseline'][:,3,2,0] = 15
    out, stat = render_add_only(**args)
    assert (out[:,1,1,1] == 5).all()  # source1 ADD wins source0 ADD
    assert (out[:,1,1,2] == FREE).all()  # t0 point already delegated to V18
    assert (out[:,3,2,0] == 15).all()  # static must not destroy unrelated occupied
    assert stat['protected_original_voxels'] == 6
    assert np.all(args['baseline'][:,1,1,1] == FREE)  # input not modified


def test_rotation_ego_transform_full_z_oob_existence_and_empty_sources():
    m = build_memory([observation([[0,4,1,0,2]])]*3+[observation()])
    args = renderer_arguments(m)
    args['yaw'][:] = np.pi/2
    args['future_centers'][:,:,0] = 3
    args['future_centers'][:,:,2] = 100  # source Z motion must NOT change world Z
    args['world_to_future'][:,0,3] = 1
    out, _ = render_add_only(**args)
    assert np.all(out[:,3,1,2] == 4)
    args['exists'] = np.zeros((6,2),bool)
    assert np.array_equal(render_add_only(**args)[0],args['baseline'])
    args.pop('exists'); args['future_centers'][:,:,0] = -100
    assert render_add_only(**args)[1]['out_of_bounds_points'] == 6
    args['yaw'][:] = np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        render_add_only(**args)
    empty = build_memory([observation()]*4)
    static = SparseRepairHead('cached_future')(**empty.tensors(),
        source_context=torch.zeros(0,128), future_queries=torch.zeros(0,6,128))
    assert static.shape == (0,6)


def test_ego_raster_class_collision_fails_closed():
    m = build_memory([observation([[STATIC,11,0,0,0],[STATIC,13,1,0,0]])]*3+[observation()])
    args = renderer_arguments(m)
    # A proper rotation maps these nearby centers into one discrete grid cell.
    theta = np.pi/4
    args['world_to_future'][:,:2,:2] = [[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
    args['world_to_future'][:,0,3] = .1
    args['world_to_future'][:,1,3] = -.6
    out,stat = render_add_only(**args)
    assert np.array_equal(out,args['baseline'])
    assert stat['ambiguous_static_raster_voxels'] == 6


def test_changing_shape_oracle_exposes_once_limit_and_synthetic_seeds_are_deterministic():
    a,b = make_scene(500,changing=True), make_scene(500,changing=True)
    assert np.array_equal(a.memory.keys,b.memory.keys) and np.array_equal(a.target,b.target)
    q = expressivity(a)
    assert q['fixed_shape_best_binary_accuracy'] < 1
    assert q['fixed_shape_oracle']['iou'] < q['future_point_oracle']['iou'] == 1
    rigid = expressivity(make_scene(500,changing=False))
    assert rigid['fixed_shape_best_binary_accuracy'] == 1
    assert np.array_equal(a.context,source_context_from_observations(a.observations,len(a.context)))
    before = a.memory.features(neighborhood=True).copy()
    a.target[:] = ~a.target  # supervision edits cannot change causal features
    assert np.array_equal(before,a.memory.features(neighborhood=True))


def test_old_input_full_z_and_both_memory_readers_are_finite():
    scene = make_scene(17,sources=1,static_columns=4)
    model = LinkedColumns(ColumnConfig(width=16,semantic_dim=4),128,history_frames=4).eval()
    x = column_inputs(scene,np.arange(2),[0,5])
    assert x['history'].shape == (2,4,7,7,16)
    with torch.inference_mode():
        for tokens in (196,36):
            out = column_probabilities(model,scene,tokens=tokens,device=torch.device('cpu'))
            assert out.shape == (len(scene.memory),6) and np.isfinite(out).all()


def test_short_bundle_no_threshold_tuning_and_disjoint_scene_seeds():
    torch.set_num_threads(1)
    q, _ = quality_screen(torch.device('cpu'),steps=2,batch=4)
    assert q['train_seeds'][1] < q['test_seeds'][0]
    assert q['thresholds'] == 'fixed 0.5, no tuning'
    assert set(q['results']) == {'once','actor_gate','cached_future','local_consensus','column_196','column_36'}


def test_vectorized_signed_row_lookup_and_neighbor_features_match_reference():
    rng = np.random.default_rng(150)
    for _ in range(10):
        rows = np.column_stack((rng.choice([-2,0,1],30), np.full(30,4), rng.integers(-3,4,(30,3))))
        observations = [observation(rows[rng.random(30)>.5], rows) for _ in range(4)]
        m = build_memory(observations)
        lookup = {tuple(k):i for i,k in enumerate(m.keys)}
        presence = np.zeros((len(m),4),bool)
        for t,f in enumerate(observations):
            for k in f.occupied:
                presence[lookup[tuple(k)],t] = True
        assert np.array_equal(m.presence,presence)
        counts = np.zeros((len(m),4),np.float32)
        for i,k in enumerate(m.keys):
            for axis in (2,3,4):
                for sign in (-1,1):
                    q = k.copy(); q[axis] += sign
                    if tuple(q) in lookup:
                        counts[i] += m.presence[lookup[tuple(q)]]/6
        assert np.allclose(m.features(neighborhood=True)[:,-4:],counts)
