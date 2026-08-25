"""Frozen Moving-mIoU v2 (interval_displacement_v2) core implementation."""
from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import numpy as np

@dataclass(frozen=True)
class Box3D:
    token: str
    class_id: int
    center_xyz: tuple
    size_lwh: tuple
    yaw: float

@dataclass(frozen=True)
class GridSpec:
    x_min: float
    y_min: float
    z_min: float
    voxel_size: tuple
    shape_hwd: tuple

def interval_speed_xy(center0,centerh,dt):
    if dt<=0: raise ValueError("dt must be positive")
    return float(np.linalg.norm(np.asarray(centerh)[:2]-np.asarray(center0)[:2])/dt)

def is_moving(box0:Box3D,boxh:Box3D,dt,threshold=0.5):
    if box0.token!=boxh.token: raise ValueError("instance token mismatch")
    return interval_speed_xy(box0.center_xyz,boxh.center_xyz,dt)>=threshold

def rasterize_oriented_box(box:Box3D,grid:GridSpec,margin=0.5):
    """Rasterize oriented 3D box to bool [H,W,D]."""
    H,W,D=grid.shape_hwd
    vx,vy,vz=grid.voxel_size
    # cell centers; H corresponds y, W corresponds x
    xs=grid.x_min+(np.arange(W)+0.5)*vx
    ys=grid.y_min+(np.arange(H)+0.5)*vy
    zs=grid.z_min+(np.arange(D)+0.5)*vz
    cx,cy,cz=box.center_xyz; l,w,h=box.size_lwh
    # Bounding crop first to avoid full 3D mesh
    radius=0.5*math.hypot(l+2*margin,w+2*margin)
    xi=np.where((xs>=cx-radius)&(xs<=cx+radius))[0]
    yi=np.where((ys>=cy-radius)&(ys<=cy+radius))[0]
    zi=np.where((zs>=cz-h/2-margin)&(zs<=cz+h/2+margin))[0]
    out=np.zeros((H,W,D),dtype=bool)
    if not len(xi) or not len(yi) or not len(zi): return out
    X,Y=np.meshgrid(xs[xi],ys[yi])
    dx=X-cx; dy=Y-cy
    c=math.cos(box.yaw); s=math.sin(box.yaw)
    lx=c*dx+s*dy
    ly=-s*dx+c*dy
    inside=(np.abs(lx)<=l/2+margin)&(np.abs(ly)<=w/2+margin)
    yy,xx=np.where(inside)
    for z in zi:
        out[yi[yy],xi[xx],z]=True
    return out

def moving_support(box0_future_grid:Box3D,boxh_future_grid:Box3D,
                   dt,grid:GridSpec,speed_threshold=0.5,margin=0.5):
    if not is_moving(box0_future_grid,boxh_future_grid,dt,speed_threshold):
        return np.zeros(grid.shape_hwd,dtype=bool)
    return (rasterize_oriented_box(box0_future_grid,grid,margin) |
            rasterize_oriented_box(boxh_future_grid,grid,margin))

class MovingMIoUV2Accumulator:
    def __init__(self,dynamic_classes:Sequence[int]):
        self.classes=tuple(int(c) for c in dynamic_classes)
        self.inter={c:0 for c in self.classes}
        self.union={c:0 for c in self.classes}
    def update(self,pred,gt,support):
        pred=np.asarray(pred); gt=np.asarray(gt); support=np.asarray(support,dtype=bool)
        if pred.shape!=gt.shape or pred.shape!=support.shape: raise ValueError("shape mismatch")
        for c in self.classes:
            p=(pred==c)&support; g=(gt==c)&support
            self.inter[c]+=int((p&g).sum()); self.union[c]+=int((p|g).sum())
    def compute(self):
        ious={c:(self.inter[c]/self.union[c] if self.union[c] else np.nan) for c in self.classes}
        vals=[v for v in ious.values() if not np.isnan(v)]
        return {"mIoU":100*float(np.mean(vals)) if vals else float("nan"),"per_class":{c:100*v for c,v in ious.items()}}


class MovingMIoUV2MultiHorizon:
    """Enforce the frozen contract: mIoU per horizon, then mean across horizons.

    Do not merge all 1s/2s/3s voxels into a single accumulator: that would
    weight horizons by support size and change the metric definition.
    """
    def __init__(self, dynamic_classes: Sequence[int],
                 horizons_s=(1.0, 2.0, 3.0)):
        self.horizons=tuple(float(h) for h in horizons_s)
        self.acc={h:MovingMIoUV2Accumulator(dynamic_classes) for h in self.horizons}

    def update(self, horizon_s, pred, gt, support):
        h=float(horizon_s)
        if h not in self.acc:
            raise KeyError(f"horizon {h} not in frozen report horizons {self.horizons}")
        self.acc[h].update(pred,gt,support)

    def compute(self):
        per_horizon={h:self.acc[h].compute() for h in self.horizons}
        vals=[per_horizon[h]["mIoU"] for h in self.horizons]
        if any(np.isnan(v) for v in vals):
            return {"mIoU":float("nan"),"per_horizon":per_horizon}
        return {"mIoU":float(np.mean(vals)),"per_horizon":per_horizon}
