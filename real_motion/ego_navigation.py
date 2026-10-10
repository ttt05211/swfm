"""Causal ego state, exact pose convention, explicit GT-derived navigation labels."""
import math
import numpy as np

COMMAND_PROTOCOL = 'i2world_endpoint_lidar_x_ge2_right_leminus2_left_gast_destination_rows_v1'


def validate_window_identity(nusc, record):
    """Validate chronological causal identity before opening historical grids."""
    hist=tuple(record['history_tokens']);future=tuple(record['future_tokens'])
    if len(hist)!=4 or len(future)!=6 or hist[-1]!=record['t0_token']:
        raise ValueError('exact four histories ending at t0 and six futures required')
    tokens=hist+future
    if len(set(tokens))!=10:raise ValueError('repeated token in ego window')
    samples=[nusc.get('sample',t) for t in tokens]
    if (len({r['scene_token'] for r in samples})!=1
            or nusc.get('scene',samples[0]['scene_token'])['name']!=record['scene_name']
            or any(a['next']!=tb or b['prev']!=ta for ta,tb,a,b in zip(tokens,tokens[1:],samples,samples[1:]))
            or any(b['timestamp']<=a['timestamp'] for a,b in zip(samples,samples[1:]))):
        raise ValueError('ego window identity/order/timestamps do not match nuScenes keyframe chain')


def checked_poses(poses, count):
    p=np.asarray(poses,np.float64)
    if (p.shape!=(count,4,4) or not np.isfinite(p).all()
            or not np.allclose(p[:,3], [0,0,0,1],atol=1e-8,rtol=0)
            or not np.allclose(p[:,:3,:3].transpose(0,2,1)@p[:,:3,:3],np.eye(3),atol=1e-6,rtol=0)
            or not np.allclose(np.linalg.det(p[:,:3,:3]),1.,atol=1e-6,rtol=0)):
        raise ValueError('finite proper rigid poses required')
    return p


def relative_se2(current, future):
    p=checked_poses([current],1)[0]; f=checked_poses(future,6)
    # Matches the existing cached planner: preserve world z and relative t0 tilt.
    # Solve the 2x2 horizontal block rather than silently dropping t0 tilt.
    if abs(np.linalg.det(p[:2,:2]))<.5: raise ValueError('near-vertical ego frame')
    xy=np.linalg.solve(p[:2,:2],(f[:,:2,3]-p[None,:2,3]).T).T
    r=p[:3,:3].T[None]@f[:,:3,:3]
    return np.c_[xy,np.arctan2(r[:,1,0],r[:,0,0])].astype(np.float32)


def poses_from_se2(current, se2):
    p=checked_poses([current],1)[0]; a=np.asarray(se2,np.float64)
    if a.shape!=(6,3) or not np.isfinite(a).all(): raise ValueError('six finite absolute XY/yaw predictions required')
    out=np.repeat(p[None],6,axis=0)
    for h,(x,y,yaw) in enumerate(a):
        cs,sn=np.cos(yaw),np.sin(yaw)
        out[h,:3,:3]=p[:3,:3]@np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])
        out[h,:2,3]=p[:2,3]+p[:2,:2]@np.array([x,y])
    return checked_poses(out,6)


def ego_history_features(poses, timestamps_s=None):
    p=checked_poses(poses,4); times=(np.arange(4)*.5 if timestamps_s is None else np.asarray(timestamps_s,np.float64))
    if times.shape!=(4,) or not np.isfinite(times).all() or not (np.diff(times)>0).all():
        raise ValueError('four increasing historical timestamps required')
    inv=np.linalg.inv(p[-1]); relative=inv[None]@p
    xyz=relative[:,:3,3]; yaw=np.unwrap(np.arctan2(relative[:,1,0],relative[:,0,0]))
    dt=np.diff(times); velocity=np.diff(xyz,axis=0)/dt[:,None]; rate=np.diff(yaw)/dt
    velocity=np.concatenate((velocity[:1],velocity));rate=np.r_[rate[0],rate]
    # No future fallback for the oldest velocity; only known historical secants.
    return np.c_[xyz/40,np.sin(yaw),np.cos(yaw),velocity/10,rate/math.pi,(times-times[-1])/1.5].astype(np.float32)


def replace_future_geometry(raw, poses):
    """No mixing: downstream preparation re-renders EVERY future-dependent field."""
    if raw.get('future_gt_occ') is not None: raise ValueError('prediction cannot receive future occupancy')
    out={k:raw[k] for k in ('history_occ','history_observed','history_poses')}
    out['future_poses']=[p.copy() for p in checked_poses(poses,6)]
    out['future_gt_occ']=None
    # ONLY pure historical frame geometry can be carried across pose substitutions.
    if '_waymo_frame_geometry' in raw: out['_waymo_frame_geometry']=raw['_waymo_frame_geometry']
    return out


def _quat_matrix(q):
    q=np.asarray(q,np.float64)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12: raise ValueError('invalid quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def lidar_pose(nusc, token):
    sample=nusc.get('sample',token); sd=nusc.get('sample_data',sample['data']['LIDAR_TOP'])
    ep=nusc.get('ego_pose',sd['ego_pose_token']);cs=nusc.get('calibrated_sensor',sd['calibrated_sensor_token'])
    ego=np.eye(4);ego[:3,:3]=_quat_matrix(ep['rotation']);ego[:3,3]=ep['translation']
    sensor=np.eye(4);sensor[:3,:3]=_quat_matrix(cs['rotation']);sensor[:3,3]=cs['translation']
    return ego@sensor


def navigation_commands(nusc, future_tokens):
    """I² converter's endpoint rule; GAST's six DESTINATION command rows.

    Explicit benchmark navigation condition: each future row looks another six
    keyframes ahead, repeating its final scene pose at scene end. These discrete
    GT-derived commands are NOT claimed history-only. No continuous GT motion is
    exposed to the predictor through this function.
    """
    if len(future_tokens)!=6: raise ValueError('six destination tokens required')
    result=[]
    for token in future_tokens:
        start=lidar_pose(nusc,token); last=token
        for _ in range(6):
            nxt=nusc.get('sample',last)['next']
            if not nxt: break
            last=nxt
        endpoint=(np.linalg.inv(start)@lidar_pose(nusc,last))[:3,3]
        result.append(0 if endpoint[0]>=2 else 1 if endpoint[0]<=-2 else 2)
    return np.asarray(result,np.int64)
