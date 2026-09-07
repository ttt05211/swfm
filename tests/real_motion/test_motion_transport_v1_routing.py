import numpy as np
from real_motion.motion_transport_v1.contracts import SourceRecord,SourceDecomposition,MSPCandidateRecord
from real_motion.motion_transport_v1.source_adapter import map_msp_to_sources
from real_motion.motion_transport_v1.routing import route_sources,training_budget_selection,scene_mirror_flag
def src(i,cls,vox):
    vox=np.asarray(vox,int);p=vox.astype(float);return SourceRecord(i,cls,vox,p,p.mean(0),np.zeros(3),True,np.array([[0,0],[1,1]],float),len(vox))
def test_explicit_overlap_mapping_not_index_mapping():
    s0=src(0,4,[[1,1,1],[1,2,1],[1,3,1]]);s1=src(1,4,[[5,5,1],[5,6,1]]);d=SourceDecomposition([s0,s1],np.zeros((6,10,10,3),np.uint8),np.zeros((0,3),int),np.zeros(0,np.uint8),np.zeros((0,3)),np.arange(1,7)*.5);c0=MSPCandidateRecord(0,7,4,0,np.array([[5,5,1],[5,6,1]]),np.zeros(19,np.float32));c1=MSPCandidateRecord(1,3,4,1,np.array([[1,1,1],[1,2,1]]),np.zeros(19,np.float32));map_msp_to_sources(d,[c0,c1],shape_xyz=(10,10,3));assert d.sources[0].msp_candidate_indices==(1,) and d.sources[1].msp_candidate_indices==(0,);assert np.isclose(d.sources[0].mapping_coverage,2/3)
def test_route_and_random_budget_deterministic_independent_of_msp():
    ss=[src(i,4,[[i+1,1,1]]) for i in range(20)]
    for s in ss:s.msp_score=.5
    assert route_sources(ss,2,strategy='msp')==(0,1);a,_=training_budget_selection(ss,.7,seed=99,sample_id='s:1',later_all_probability=0,sparse_budget=16)
    for i,s in enumerate(ss):s.msp_score=1000-i
    b,_=training_budget_selection(ss,.7,seed=99,sample_id='s:1',later_all_probability=0,sparse_budget=16);assert a==b and len(a)==16;assert scene_mirror_flag(seed=3407,sample_id='x')==scene_mirror_flag(seed=3407,sample_id='x')
