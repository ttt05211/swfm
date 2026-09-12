from __future__ import annotations
import math,numpy as np
from real_motion.geometry import OccupancyGrid

def transform_points(T,points):
    p=np.asarray(points,np.float64);T=np.asarray(T,np.float64);return p@T[:3,:3].T+T[:3,3]
def metric_to_center_index(points_xyz,grid):
    return (np.asarray(points_xyz,np.float64)-np.asarray([grid.x_min,grid.y_min,grid.z_min]))/np.asarray(grid.voxel_size)-.5
def index_to_metric_center(indices_xyz,grid):
    return np.asarray([grid.x_min,grid.y_min,grid.z_min])+(np.asarray(indices_xyz,np.float64)+.5)*np.asarray(grid.voxel_size)
def metric_to_floor_index(points_xyz,grid):
    return np.floor((np.asarray(points_xyz,np.float64)-np.asarray([grid.x_min,grid.y_min,grid.z_min]))/np.asarray(grid.voxel_size)).astype(np.int64)
def in_grid(indices_xyz,grid):
    q=np.asarray(indices_xyz,np.int64);s=np.asarray(grid.shape_hwd);return ((q>=0)&(q<s[None])).all(-1)
def yaw_from_rotation(R): return float(math.atan2(float(R[1,0]),float(R[0,0])))
def f_to_world_matrix(t0_ego_to_world):
    T=np.asarray(t0_ego_to_world,np.float64);yaw=yaw_from_rotation(T[:3,:3]);c,s=math.cos(yaw),math.sin(yaw);F=np.eye(4);F[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];F[:3,3]=T[:3,3];return F
def world_to_F(points_world,t0_ego_to_world): return transform_points(np.linalg.inv(f_to_world_matrix(t0_ego_to_world)),points_world)
def F_to_world(points_F,t0_ego_to_world): return transform_points(f_to_world_matrix(t0_ego_to_world),points_F)
def source_bbox_F(points_world,t0_ego_to_world):
    p=world_to_F(points_world,t0_ego_to_world);return np.stack([p[:,:2].min(0),p[:,:2].max(0)])
