from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np

@dataclass(frozen=True)
class CausalInputs:
    """Inference-only contract. Future ego pose is protocol input; future object/occupancy GT is not."""
    sample_id: str
    scene_name: str
    history_semantics: np.ndarray
    history_valid: np.ndarray
    history_ego_to_world: tuple[np.ndarray, ...]
    future_ego_to_world: tuple[np.ndarray, ...]
    history_timestamps_s: np.ndarray
    future_timestamps_s: np.ndarray
    def __post_init__(self):
        h=np.asarray(self.history_semantics);v=np.asarray(self.history_valid)
        if h.ndim!=4: raise ValueError('history_semantics must be [T,X,Y,Z]')
        if v.shape!=h.shape: raise ValueError('history_valid shape mismatch')
        if not np.issubdtype(h.dtype,np.integer): raise TypeError('history semantics must be integer labels')
        if len(self.history_ego_to_world)!=h.shape[0]: raise ValueError('history pose count mismatch')
        if len(self.future_ego_to_world)!=len(self.future_timestamps_s): raise ValueError('future pose/time count mismatch')
        if np.asarray(self.history_timestamps_s).shape!=(h.shape[0],): raise ValueError('history timestamps mismatch')
        for T in (*self.history_ego_to_world,*self.future_ego_to_world):
            if np.asarray(T).shape!=(4,4): raise ValueError('ego pose must be 4x4')
    @property
    def history_frames(self): return int(self.history_semantics.shape[0])
    @property
    def future_frames(self): return int(len(self.future_ego_to_world))

@dataclass(frozen=True)
class MotionTarget:
    source_id:int
    valid:np.ndarray
    point_indices:np.ndarray
    gt_xy_world:np.ndarray
    instance_token:str|None=None
    match_best_coverage:float=0.0
    match_second_coverage:float=0.0
    gt_speed_mps:np.ndarray|None=None

@dataclass(frozen=True)
class TrainingTargets:
    future_semantics:np.ndarray
    future_valid:np.ndarray
    motion_targets:Mapping[int,MotionTarget]=field(default_factory=dict)
    def __post_init__(self):
        f=np.asarray(self.future_semantics);v=np.asarray(self.future_valid)
        if f.ndim!=4 or v.shape!=f.shape: raise ValueError('future target shape mismatch')

@dataclass
class SourceRecord:
    source_id:int;class_id:int;voxel_indices_t0:np.ndarray;points_world:np.ndarray;centroid_world:np.ndarray;velocity_world:np.ndarray;kta_matched:bool;bbox_F:np.ndarray;voxel_count:int
    msp_candidate_indices:tuple[int,...]=();overlap_weights:np.ndarray=field(default_factory=lambda:np.zeros((0,),np.float64));mapping_coverage:float=0.0;observed_moving_fraction:float=0.0;dormant_fraction:float=0.0;crop_eligible:bool=True;fallback_reason:str='';msp_score:float=0.0
    def __post_init__(self):
        self.voxel_indices_t0=np.asarray(self.voxel_indices_t0,np.int64);self.points_world=np.asarray(self.points_world,np.float64);self.centroid_world=np.asarray(self.centroid_world,np.float64);self.velocity_world=np.asarray(self.velocity_world,np.float64);self.bbox_F=np.asarray(self.bbox_F,np.float64);self.overlap_weights=np.asarray(self.overlap_weights,np.float64)
        if self.voxel_indices_t0.ndim!=2 or self.voxel_indices_t0.shape[1]!=3: raise ValueError('voxel_indices_t0 must be [N,3]')
        if self.points_world.shape!=self.voxel_indices_t0.shape: raise ValueError('points/source shape mismatch')
        if self.centroid_world.shape!=(3,) or self.velocity_world.shape!=(3,): raise ValueError('centroid/velocity must be 3D')
        if abs(float(self.velocity_world[2]))>1e-12: raise ValueError('source z velocity must be zero')
        if int(self.voxel_count)!=len(self.voxel_indices_t0): raise ValueError('voxel_count mismatch')

@dataclass(frozen=True)
class MSPCandidateRecord:
    candidate_id:int;original_enum_index:int;class_id:int;state:int;voxel_indices_t0:np.ndarray;feature:np.ndarray
@dataclass
class SourceDecomposition:
    sources:list[SourceRecord];background_future:np.ndarray;rest_voxel_indices_t0:np.ndarray;rest_labels:np.ndarray;rest_points_world:np.ndarray;horizons_s:np.ndarray
@dataclass(frozen=True)
class CropBatch:
    source_ids:np.ndarray;semantics:Any;valid:Any;source_mask:Any;relative_times:Any;metadata:Any;mirror_flags:Any
    @property
    def num_sources(self): return int(len(self.source_ids))
@dataclass
class SoftHorizon:
    flat_indices:Any;probabilities:Any;kta_labels:Any
@dataclass
class SoftScene:
    horizons:list[SoftHorizon];hard_kta:np.ndarray;shape_xyz:tuple[int,int,int]
@dataclass
class PredictionResult:
    hard_occupancy:np.ndarray;deltas:Any;selected_source_ids:tuple[int,...];source_scores:Mapping[int,float];decomposition:SourceDecomposition;soft_scene:SoftScene|None=None;diagnostics:dict[str,Any]=field(default_factory=dict)
