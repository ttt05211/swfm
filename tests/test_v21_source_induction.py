import numpy as np
from real_motion.geometry import OccupancyGrid
from real_motion.v21_source_induction import (
    AnchorLattice,CanonicalShape,FrontierAnchor,V21Target,assign_causal_coverage,
    attribute_instance_shapes,build_frontier_anchors,build_prototype_bank,
    compose_v21_add_only,oracle_best_prototype,rasterize_canonical_shape,
    select_scene_balanced_round_robin,shape_iou,
)

def birth(token,xy,onset=1):
    centers=[None]*6; yaws=[None]*6; ex=[False]*6
    centers[onset]=(float(xy[0]),float(xy[1]),0.0); yaws[onset]=0.0; ex[onset]=True
    if onset!=1: centers[1]=(float(xy[0]),float(xy[1]),0.0); yaws[1]=0.0; ex[1]=True
    return V21Target(token,4,"BIRTH",tuple(ex),tuple(centers),tuple(yaws),onset,True,False)

def test_dev64_scene_balanced_round_robin():
    keys=(("a","a0"),("a","a1"),("b","b0"),("c","c0"),("b","b1"),("a","a2"))
    assert select_scene_balanced_round_robin(keys,6)==(
        ("a","a0"),("b","b0"),("c","c0"),("a","a1"),("b","b1"),("a","a2"))

def test_frontier_assignment_is_one_to_one():
    targets=[birth("a",(0.2,0.0)),birth("b",(0.3,0.0))]
    anchors=[FrontierAnchor(5050,(0.0,0.0,0.0),1,0b111111,0)]
    matches,rep=assign_causal_coverage(targets,[],anchors,t0_pose=np.eye(4),coverage_radius_m=0.8)
    assert len(matches)==1 and matches[0].target_token=="a"
    assert rep["duplicate_target_assignment"]==0 and rep["duplicate_anchor_assignment"]==0

def test_onset_cannot_precede_frontier():
    t=birth("a",(0,0),onset=0)
    a=FrontierAnchor(1,(0,0,0),1,0b000100,2)
    m,_=assign_causal_coverage([t],[],[a],t0_pose=np.eye(4),coverage_radius_m=3.2)
    assert m==[]

def test_query_entry_onset_can_match_later_frontier():
    centers=[(10.0,0.0,0.0),None,None,(0.0,0.0,0.0),None,None]
    yaws=[0.0,None,None,0.0,None,None]
    t=V21Target("a",4,"BIRTH",(True,False,False,True,False,False),tuple(centers),tuple(yaws),0,True,False)
    a=FrontierAnchor(1,(0.0,0.0,0.0),1,0b001000,3)
    old,_=assign_causal_coverage([t],[],[a],t0_pose=np.eye(4),coverage_radius_m=0.8)
    new,_=assign_causal_coverage(
        [t],[],[a],t0_pose=np.eye(4),coverage_radius_m=0.8,
        onset_index_by_token={"a":3})
    assert old==[]
    assert len(new)==1 and new[0].target_token=="a" and new[0].distance_m==0.0

def test_add_only_priority_and_collision():
    base=np.full((3,3,1),17,np.uint8); base[0,0,0]=1
    hist=np.asarray([[1,1,0],[0,0,0]]); front=np.asarray([[1,1,0],[2,2,0]])
    out,r=compose_v21_add_only(base,[("frontier",10,6,front),("historical",-1,4,hist)])
    assert out[0,0,0]==1 and out[1,1,0]==4 and out[2,2,0]==6
    assert r.blocked_by_v18_voxels==1 and r.v21_collision_voxels==1
    assert r.historical_frontier_collision_voxels==1

def test_add_only_rejects_oob_without_indexing_or_overwrite():
    base=np.full((2,2,1),17,np.uint8); base[0,0,0]=3
    idx=np.asarray([[-1,0,0],[2,0,0],[0,0,0],[1,1,0],[1,1,0]])
    out,r=compose_v21_add_only(base,[("frontier",1,4,idx)])
    assert out[0,0,0]==3 and out[1,1,0]==4
    assert r.out_of_bounds_voxels==2 and r.blocked_by_v18_voxels==1

def test_kmedoids_deterministic_and_oracle_best():
    a=CanonicalShape(4,np.asarray([[0,0,0],[1,0,0]],np.int32),("s0","i0"))
    b=CanonicalShape(4,np.asarray([[0,0,0],[1,0,0],[2,0,0]],np.int32),("s1","i1"))
    c=CanonicalShape(4,np.asarray([[0,0,0],[0,1,0]],np.int32),("s2","i2"))
    x=build_prototype_bank({4:[a,b,c]},requested_k=2,population_manifest=[a.observation_key,b.observation_key,c.observation_key])
    y=build_prototype_bank({4:[c,a,b]},requested_k=2,population_manifest=[c.observation_key,a.observation_key,b.observation_key])
    assert x.fingerprint==y.fingerprint
    best=oracle_best_prototype(a,x)
    assert shape_iou(a,best)==max(shape_iou(a,z) for z in x.medoids_by_class[4])

def test_frontier_keeps_type_and_time_bits():
    grid=OccupancyGrid(x_min=-0.8,y_min=-0.8,z_min=-0.2,voxel_size=(0.4,0.4,0.4),shape_hwd=(4,4,1))
    obs=np.zeros((6,4,4,1),bool); obs[:,1:3,1:3,0]=True
    poses=np.repeat(np.eye(4)[None],6,axis=0)
    lattice=AnchorLattice((-1.1,-0.7),0.4,(7,5))
    anchors,rep=build_frontier_anchors(obs,poses,poses,grid=grid,lattice=lattice)
    assert anchors and rep["frontier_anchor_count"]==len(anchors)
    assert all(a.eligible_horizon_mask and a.frontier_type_mask for a in anchors)
    assert [a.canonical_anchor_id for a in anchors]==sorted(a.canonical_anchor_id for a in anchors)
    assert lattice.anchor_shape_xy==(2,2)

def test_exact_shape_round_trip_preserves_subvoxel_phase():
    grid=OccupancyGrid(x_min=-0.8,y_min=-0.8,z_min=-0.2,voxel_size=(0.4,0.4,0.4),shape_hwd=(4,4,1))
    sem=np.full(grid.shape_hwd,17,np.uint8); sem[2,2,0]=4
    anns={"a":{"instance_token":"a","class_id":4,"center_world":np.asarray([0.13,0.17,0.0]),
               "yaw_world":0.31,"size_lwh":np.asarray([1.0,1.0,1.0])}}
    attr=attribute_instance_shapes(sem,np.eye(4),anns,grid=grid,tokens=("a",),observation_keys={"a":("s","a")})["a"]
    assert attr.shape is not None and not attr.ambiguous
    idx,oob=rasterize_canonical_shape(attr.shape,anns["a"]["center_world"],anns["a"]["yaw_world"],np.eye(4),grid=grid)
    assert oob==0 and np.array_equal(idx,np.asarray([[2,2,0]]))

def test_shape_attribution_fails_closed_without_target_box_overlap():
    grid=OccupancyGrid(x_min=-2,y_min=-2,z_min=-0.2,voxel_size=(0.4,0.4,0.4),shape_hwd=(10,10,1))
    sem=np.full(grid.shape_hwd,17,np.uint8); sem[5,5,0]=4
    anns={"a":{"instance_token":"a","class_id":4,"center_world":np.asarray([1.0,0.2,0.0]),
               "yaw_world":0.0,"size_lwh":np.asarray([0.1,0.1,0.1])}}
    attr=attribute_instance_shapes(sem,np.eye(4),anns,grid=grid,tokens=("a",))["a"]
    assert attr.ambiguous and attr.shape is None and not attr.unresolved

def test_precomputed_strong_components_are_reused_for_shape_attribution():
    grid=OccupancyGrid(x_min=-0.8,y_min=-0.8,z_min=-0.2,voxel_size=(0.4,0.4,0.4),shape_hwd=(4,4,1))
    sem=np.full(grid.shape_hwd,17,np.uint8)
    anns={"a":{"instance_token":"a","class_id":4,"center_world":np.asarray([0.2,0.2,0.0]),
               "yaw_world":0.0,"size_lwh":np.asarray([1.0,1.0,1.0])}}
    components=[{"class_id":4,"centroid_world":np.asarray([0.2,0.2,0.0]),
                 "voxel_indices":np.asarray([[2,2,0]]),"voxel_count":1}]
    attr=attribute_instance_shapes(
        sem,np.eye(4),anns,grid=grid,tokens=("a",),components=components,
        component_matches=("a",),observation_keys={"a":("s","a")})["a"]
    assert attr.shape is not None and not attr.ambiguous
    assert np.array_equal(attr.voxel_indices,np.asarray([[2,2,0]]))

def test_shape_attribution_fails_closed_on_two_box_overlap():
    grid=OccupancyGrid(x_min=-0.8,y_min=-0.8,z_min=-0.2,voxel_size=(0.4,0.4,0.4),shape_hwd=(4,4,1))
    sem=np.full(grid.shape_hwd,17,np.uint8); sem[2,2,0]=4
    anns={
        "a":{"instance_token":"a","class_id":4,"center_world":np.asarray([0.2,0.2,0.0]),
             "yaw_world":0.0,"size_lwh":np.asarray([1.0,1.0,1.0])},
        "b":{"instance_token":"b","class_id":4,"center_world":np.asarray([0.3,0.2,0.0]),
             "yaw_world":0.0,"size_lwh":np.asarray([1.0,1.0,1.0])},
    }
    attrs=attribute_instance_shapes(sem,np.eye(4),anns,grid=grid,tokens=("a","b"))
    assert attrs["a"].ambiguous and attrs["b"].ambiguous
    assert attrs["a"].shape is None and attrs["b"].shape is None

def test_large_kmedoids_path_is_deterministic():
    rows=[]
    for i in range(520):
        cells=np.asarray([[0,0,0],[1+(i%7),i%5,0]],np.int32)
        rows.append(CanonicalShape(4,cells,(f"s{i:04d}",f"i{i:04d}")))
    pop=[x.observation_key for x in rows]
    a=build_prototype_bank({4:rows},requested_k=4,population_manifest=pop)
    b=build_prototype_bank({4:list(reversed(rows))},requested_k=4,population_manifest=list(reversed(pop)))
    assert a.fingerprint==b.fingerprint
    assert a.algorithm=="exact_pam_le_512_else_deterministic_clara_v1"
