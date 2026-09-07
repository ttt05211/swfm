from __future__ import annotations
import torch
from .config import make_grid,make_strong_cfg,make_msp_motion_cfg,make_msp_kta_cfg,get
from .source_adapter import decompose_strong_sources,extract_original_msp_candidates,map_msp_to_sources
from .routing import load_frozen_msp,infer_source_scores,route_sources
from .crop_history import build_history_crops
from .stpn import STPNMotionNetwork
from .compositor import compose_hard,render_soft_ordered,hard_kta_identity
from .contracts import PredictionResult
class MotionTransportV1:
    def __init__(self,cfg,*,msp_checkpoint,device='cpu',source_network=None):
        self.cfg=cfg;self.device=torch.device(device);self.grid=make_grid(cfg);self.strong_cfg=make_strong_cfg(cfg);self.motion_cfg=make_msp_motion_cfg(cfg);self.msp_kta_cfg=make_msp_kta_cfg(cfg);self.msp_ckpt,self.msp=load_frozen_msp(msp_checkpoint,self.device);self.source_network=source_network or STPNMotionNetwork(class_tokens=int(get(cfg,'crop.embedding_tokens',19)),embedding_dim=int(get(cfg,'crop.embedding_dim',4)),metadata_dim=int(get(cfg,'network.metadata_dim',19)),future_frames=int(get(cfg,'input.future_frames',6)),activation_checkpointing=bool(get(cfg,'network.activation_checkpointing',False))).to(self.device)
    def prepare_scene(self,causal):
        d=decompose_strong_sources(causal,grid=self.grid,cfg=self.strong_cfg,frame_dt_s=float(get(self.cfg,'input.dt_seconds',.5)),crop_radius_limit_m=float(get(self.cfg,'crop.max_source_xy_radius_about_centroid_m',11.2)));c=extract_original_msp_candidates(causal,grid=self.grid,motion_cfg=self.motion_cfg,kta_cfg=self.msp_kta_cfg);map_msp_to_sources(d,c,shape_xyz=tuple(self.grid.shape_hwd));scores=infer_source_scores(c,d.sources,self.msp,self.device);return d,c,scores
    def forward_selected(self,causal,decomp,selected,*,mirror=False,soft=True,hard=True):
        crops=build_history_crops(causal,decomp.sources,selected,grid=self.grid,crop_shape_xyz=tuple(get(self.cfg,'crop.shape_xyz',[64,64,16])),xy_resolution_m=float(get(self.cfg,'crop.xy_resolution_m',.4)),mirror=mirror,device=self.device);delta,zero=self.source_network(crops,source_microbatch=int(get(self.cfg,'training.source_microbatch',16)));hout=compose_hard(causal,decomp,delta,selected,grid=self.grid) if hard else None;sout=render_soft_ordered(causal,decomp,delta,selected,grid=self.grid,class_count=int(get(self.cfg,'input.class_count',18)),halo_voxels=int(get(self.cfg,'renderer.query_bbox_halo_voxels',2)),query_chunk=int(get(self.cfg,'renderer.query_chunk',65536))) if soft else None;return delta,zero,hout,sout,crops
    @torch.no_grad()
    def predict(self,causal,budget=16,*,strategy='msp',seed=3407):
        self.source_network.eval()
        if budget in (0,'0'):
            d=decompose_strong_sources(causal,grid=self.grid,cfg=self.strong_cfg,frame_dt_s=float(get(self.cfg,'input.dt_seconds',.5)),crop_radius_limit_m=float(get(self.cfg,'crop.max_source_xy_radius_about_centroid_m',11.2)));hard=hard_kta_identity(causal,d,grid=self.grid);z=torch.zeros((0,len(d.horizons_s),3),device=self.device);return PredictionResult(hard,z,(),{},d,None,{'stpn_sources_executed':0,'msp_candidates':0,'pure_kta':True})
        d,c,scores=self.prepare_scene(causal);sel=route_sources(d.sources,budget,strategy=strategy,scores=scores,seed=seed);delta,_,hard,_,_=self.forward_selected(causal,d,sel,mirror=False,soft=False,hard=True);return PredictionResult(hard,delta,sel,scores,d,None,{'stpn_sources_executed':len(sel),'msp_candidates':len(c),'pure_kta':False})
