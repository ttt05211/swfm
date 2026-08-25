"""Frozen Moving-mIoU v2: interval_displacement_v2."""
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

@dataclass(frozen=True)
class Box3D:
    token:str; class_id:int; center_xyz:tuple; size_lwh:tuple; yaw:float
@dataclass(frozen=True)
class GridSpec:
    x_min:float; y_min:float; z_min:float; voxel_size:tuple; shape_hwd:tuple

def interval_speed_xy(center0,centerh,dt):
    if dt<=0: raise ValueError("dt must be positive")
    return float(np.linalg.norm(np.asarray(centerh)[:2]-np.asarray(center0)[:2])/dt)
def is_moving(box0,boxh,dt,threshold=0.5):
    if box0.token!=boxh.token: raise ValueError("instance token mismatch")
    return interval_speed_xy(box0.center_xyz,boxh.center_xyz,dt)>=threshold

def rasterize_oriented_box(box,grid,margin=0.5):
    H,W,D=grid.shape_hwd; vx,vy,vz=grid.voxel_size
    xs=grid.x_min+(np.arange(W)+0.5)*vx; ys=grid.y_min+(np.arange(H)+0.5)*vy; zs=grid.z_min+(np.arange(D)+0.5)*vz
    cx,cy,cz=box.center_xyz; l,w,h=box.size_lwh; radius=0.5*math.hypot(l+2*margin,w+2*margin)
    xi=np.where((xs>=cx-radius)&(xs<=cx+radius))[0]; yi=np.where((ys>=cy-radius)&(ys<=cy+radius))[0]; zi=np.where((zs>=cz-h/2-margin)&(zs<=cz+h/2+margin))[0]
    out=np.zeros((H,W,D),dtype=bool)
    if not len(xi) or not len(yi) or not len(zi): return out
    X,Y=np.meshgrid(xs[xi],ys[yi]); dx=X-cx; dy=Y-cy; c=math.cos(box.yaw); s=math.sin(box.yaw)
    lx=c*dx+s*dy; ly=-s*dx+c*dy; inside=(np.abs(lx)<=l/2+margin)&(np.abs(ly)<=w/2+margin); yy,xx=np.where(inside)
    for z in zi: out[yi[yy],xi[xx],z]=True
    return out

def moving_support(box0_future_grid,boxh_future_grid,dt,grid,speed_threshold=0.5,margin=0.5):
    if not is_moving(box0_future_grid,boxh_future_grid,dt,speed_threshold): return np.zeros(grid.shape_hwd,dtype=bool)
    return rasterize_oriented_box(box0_future_grid,grid,margin)|rasterize_oriented_box(boxh_future_grid,grid,margin)

class MovingMIoUV2Accumulator:
    def __init__(self,dynamic_classes:Sequence[int]):
        self.classes=tuple(int(c) for c in dynamic_classes); self.inter={c:0 for c in self.classes}; self.union={c:0 for c in self.classes}
    def update(self,pred,gt,support):
        pred=np.asarray(pred); gt=np.asarray(gt); support=np.asarray(support,dtype=bool)
        if pred.shape!=gt.shape or pred.shape!=support.shape: raise ValueError("shape mismatch")
        for c in self.classes:
            p=(pred==c)&support; g=(gt==c)&support; self.inter[c]+=int((p&g).sum()); self.union[c]+=int((p|g).sum())
    def compute(self):
        ious={c:(self.inter[c]/self.union[c] if self.union[c] else np.nan) for c in self.classes}; vals=[v for v in ious.values() if not np.isnan(v)]
        return {"mIoU":100*float(np.mean(vals)) if vals else float("nan"),"per_class":{c:100*v for c,v in ious.items()}}
