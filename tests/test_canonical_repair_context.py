import numpy as np
import pytest
import torch
from test_canonical_causal_repair import scene
from real_motion.canonical_causal_repair import (build_canonical_evidence,
    map_canonical_evidence, materialize_canonical_features, repair_targets)
from real_motion.canonical_repair_context import (canonical_neighbors, FixedCanonicalCache,
    fixed_history_digest, build_fixed_canonical, SpatialCanonicalRepairHead, attach_neighbors,
    sample_causal_points,map_sampled_canonical,full_static_conflicts)


def test_cached_history_has_exact_inputs_and_live_motion_is_not_cached():
    grid, prep = scene(); cache = FixedCanonicalCache(1)
    evidence, graph = cache.get(prep, grid)
    eager = build_canonical_evidence(prep, grid)
    for field in ('features', 'labels', 'actor', 'classes', 'world', 'presence'):
        np.testing.assert_array_equal(getattr(evidence, field), getattr(eager, field))
    first = map_canonical_evidence(evidence, prep, grid)
    prep.targets[:,:,0] += 1.
    prep.raw['future_gt_occ'] = 'POISON'
    again, g = cache.get(prep, grid)
    assert again is evidence and g is graph
    second = map_canonical_evidence(again, prep, grid)
    assert not np.array_equal(first.flat[evidence.actor >= 0], second.flat[evidence.actor >= 0])
    assert cache.stats()['hits']==1 and not cache.stats()['future_supervision_cached']


@pytest.mark.parametrize('change', ['occupancy', 'observed', 'pose', 'source_class', 'registration'])
def test_actual_causal_content_change_invalidates_cache(change):
    grid, prep = scene(); cache = FixedCanonicalCache(1); cache.get(prep, grid)
    digest = fixed_history_digest(prep, grid)
    if change=='occupancy': prep.raw['history_occ'][0,1,1,1]=13
    elif change=='observed': prep.raw['history_observed'][0,1,1,1]=False
    elif change=='pose': prep.raw['history_poses'][0,0,3]=.1
    elif change=='source_class': prep.state['current'][0]['class_id']=5
    else: prep.registrations[0][0][0][1,3]=.1
    assert fixed_history_digest(prep, grid) != digest
    cache.get(prep, grid)
    assert cache.stats()['misses']==2 and not cache.stats()['hits']


def test_zero_cache_budget_does_not_truncate_inputs_or_write_disk():
    grid, prep = scene(); cache = FixedCanonicalCache(0)
    a, ga = cache.get(prep, grid); b, gb = cache.get(prep, grid)
    np.testing.assert_array_equal(a.world, b.world); np.testing.assert_array_equal(ga, gb)
    assert cache.stats()['mib']==0 and cache.stats()['entries']==0
    assert cache.stats()['disk_writes']==0 and cache.stats()['misses']==2


@pytest.mark.parametrize('limit', [1, 4000000])
def test_sparse_graph_no_cross_entity_or_face_wrapping(limit):
    grid, prep = scene()
    lazy = build_canonical_evidence(prep, grid, max_lattice_cells=limit, materialize_features=False)
    graph = canonical_neighbors(lazy)
    known = graph>=0
    actors = np.broadcast_to(lazy.actor[:,None],graph.shape)
    classes = np.broadcast_to(lazy.classes[:,None],graph.shape)
    assert np.array_equal(lazy.actor[graph[known]], actors[known])
    assert np.array_equal(lazy.classes[graph[known]], classes[known])
    # Candidate center distances retain all three axes and never row-wrap.
    for d in range(12):
        ids=np.flatnonzero(known[:,d]); diff=lazy.world[graph[ids,d]]-lazy.world[ids]
        assert np.allclose(np.linalg.norm(diff,axis=1),1 if d<6 else 2)
    other = build_canonical_evidence(prep, grid, max_lattice_cells=1 if limit!=1 else 4000000,
                                     materialize_features=False)
    np.testing.assert_array_equal(graph, canonical_neighbors(other))


def test_sampled_neighbour_closure_matches_full_encoding_and_keeps_gradients():
    torch.manual_seed(19); grid, prep = scene()
    evidence, graph = build_fixed_canonical(prep, grid); attach_neighbors(evidence, graph)
    head=SpatialCanonicalRepairHead(8,16)
    output={'history_source_context':torch.randn(1,8,requires_grad=True),
            'future_transport_queries':torch.randn(1,6,8,requires_grad=True)}
    projected=head.project_sources(output)
    all_features=head.encode_queries(evidence,graph,np.arange(len(evidence)),projected,'cpu')
    ids=np.array([0,2,5,9])
    small=head.encode_queries(evidence,graph,ids,projected,'cpu')
    torch.testing.assert_close(small,all_features[ids],atol=1e-6,rtol=1e-5)
    plan=map_canonical_evidence(evidence,prep,grid)
    logits=head.decode(small,torch.as_tensor(evidence.actor[ids]),torch.as_tensor(plan.context[ids]),
        torch.as_tensor(plan.base[ids]),torch.as_tensor(plan.fallback[ids]),
        torch.as_tensor(plan.legal[ids]),projected)
    logits.sum().backward()
    assert output['history_source_context'].grad.abs().sum()>0
    assert output['future_transport_queries'].grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.spatial.parameters())
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


def test_empty_fixed_cache_and_spatial_encoder():
    grid, prep=scene();prep.raw['history_occ'][:]=17;prep.state['current']=[];prep.registrations=[]
    e,g=FixedCanonicalCache().get(prep,grid);assert len(e)==0 and g.shape==(0,12)
    h=SpatialCanonicalRepairHead(8,16)
    o={'history_source_context':torch.empty(0,8),'future_transport_queries':torch.empty(0,6,8)}
    out=h.encode_queries(e,g,np.empty(0,np.int64),h.project_sources(o),'cpu')
    assert out.shape==(0,16)


def test_identity_spatial_residual_preserves_the_existing_point_encoder():
    from real_motion.canonical_causal_repair import CanonicalRepairHead
    grid,prep=scene();e,g=build_fixed_canonical(prep,grid)
    old=CanonicalRepairHead(8,16);new=SpatialCanonicalRepairHead(8,16,normalized=False,zero_residual=True)
    missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
    assert not unexpected and all(k.startswith('spatial.') for k in missing)
    out={'history_source_context':torch.randn(1,8),'future_transport_queries':torch.randn(1,6,8)}
    projected=new.project_sources(out);ids=np.arange(len(e))
    a=new.encode_queries(e,g,ids,projected,'cpu')
    b=old.encode(torch.as_tensor(e.features),torch.as_tensor(e.labels),torch.as_tensor(e.actor),torch.as_tensor(e.classes),out)
    assert torch.equal(a,b)


def test_causal_sampling_keeps_every_stratum_and_population_weights():
    grid,prep=scene();e,_=build_fixed_canonical(prep,grid)
    ids,w=sample_causal_points(e,np.random.default_rng(5),per_role=7)
    assert len(ids)==len(np.unique(ids))
    assert w.sum()==pytest.approx(len(e))
    a,b=sample_causal_points(e,np.random.default_rng(19),per_role=100000)
    np.testing.assert_array_equal(a,np.arange(len(e)));assert np.all(b==1)
    # Changing future supervision cannot change this sample population/order.
    prep.raw['future_gt_occ']='POISON'
    c,d=sample_causal_points(e,np.random.default_rng(5),per_role=7)
    np.testing.assert_array_equal(ids,c);np.testing.assert_array_equal(w,d)


def test_presampling_projection_and_gt_labels_equal_full_population_rows():
    grid,prep=scene();prep.yaws[:,0]=np.linspace(-.2,.2,6)
    e,_=build_fixed_canonical(prep,grid);full=map_canonical_evidence(e,prep,grid)
    ids,_=sample_causal_points(e,np.random.default_rng(11),per_role=8)
    small,plan=map_sampled_canonical(e,ids,prep,grid,full_static_conflicts(e,prep,grid))
    for key in ('flat','base','fallback','legal','context'):
        np.testing.assert_array_equal(getattr(plan,key),getattr(full,key)[ids])
    a,av=repair_targets(e,full,prep.baseline);b,bv=repair_targets(small,plan,prep.baseline)
    np.testing.assert_array_equal(a[ids],b);np.testing.assert_array_equal(av[ids],bv)


def test_full_static_conflicts_are_protected_even_when_other_class_not_sampled():
    grid,prep=scene();prep.raw['history_occ'][:,5,7,0]=13
    cache=FixedCanonicalCache();e,_=cache.get(prep,grid)
    conflict=cache.static_conflicts(e,prep,grid)
    full=map_canonical_evidence(e,prep,grid)
    flat=(5*8+6)*4
    ids=np.flatnonzero((e.classes==11)&(full.flat[:,0]==flat))
    assert len(ids)==1 and flat in conflict[0]
    _,plan=map_sampled_canonical(e,ids,prep,grid,conflict)
    assert not plan.legal[...,0].any()
    first=conflict;prep.state['world_to_future'][:,0,3]=1.
    second=cache.static_conflicts(e,prep,grid)
    assert first is not second


def test_disk_only_cache_exact_and_future_gt_not_serialized(tmp_path):
    grid,prep=scene();cache=FixedCanonicalCache(0,disk_root=tmp_path/'fixed',max_disk_mib=1)
    e,g=cache.get(prep,grid);prep.raw['future_gt_occ']='GT_POISON_NEVER_CACHE'
    again,graph=cache.get(prep,grid)
    for field in ('features','labels','actor','classes','world','presence'):
        np.testing.assert_array_equal(getattr(e,field),getattr(again,field))
    np.testing.assert_array_equal(g,graph)
    stats=cache.stats();assert stats['mib']==0 and stats['disk']['hits']==1
    assert stats['disk']['writes']==1 and stats['disk']['disk_mib']<=1
    # Artifacts are existing repository's trusted-local compressed geometry
    # format; no head/GT/model state enters their explicit value fields.
    assert set(vars(again)) <= {'features','labels','actor','classes','world','presence','audit','layouts','fixed_history_sha256','causal_strata'}
    assert not again.audit['future_GT_used'] and not again.audit['learned_features_cached']
    cache.close()


def test_cached_strata_preserve_selection_rng_and_importance_exactly():
    grid,prep=scene();e,_=build_fixed_canonical(prep,grid)
    population=e.causal_strata
    rng=np.random.default_rng(71);a,w=sample_causal_points(e,rng,per_role=8);after=rng.bit_generator.state
    del e.causal_strata
    fresh=np.random.default_rng(71);b,v=sample_causal_points(e,fresh,per_role=8)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(w,v)
    assert after==fresh.bit_generator.state
    assert sum(len(rows) for rows,_ in population)==len(e)


def test_corrupt_disk_cache_fails_closed_without_overwriting(tmp_path):
    grid,prep=scene();cache=FixedCanonicalCache(0,disk_root=tmp_path/'fixed',max_disk_mib=1)
    cache.get(prep,grid);path=next((tmp_path/'fixed').glob('*/*.cgc'))
    damaged=path.read_bytes()[:-1]+b'X';path.write_bytes(damaged)
    with pytest.raises(RuntimeError,match='corrupt'):
        cache.get(prep,grid)
    assert path.read_bytes()==damaged
    cache.close()
