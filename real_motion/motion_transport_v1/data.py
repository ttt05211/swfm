from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from real_motion.nuscenes_adapter import WindowTokens
from .contracts import CausalInputs

def _ts(nusc,t):return float(nusc.get('sample',t)['timestamp'])/1e6
def causal_from_window(source,window):
    sem=[];valid=[]
    for tok in window.history_tokens:
        s,v=source.load_occ3d(window.scene_name,tok,require_lidar_mask=True);sem.append(np.asarray(s));valid.append(np.asarray(v,bool))
    hp=tuple(np.asarray(source.pose(t),float) for t in window.history_tokens);fp=tuple(np.asarray(source.pose(t),float) for t in window.future_tokens);ht=np.asarray([_ts(source.nusc,t) for t in window.history_tokens]);ft=np.asarray([_ts(source.nusc,t) for t in window.future_tokens]);return CausalInputs(f'{window.scene_name}:{window.t0_token}',window.scene_name,np.stack(sem),np.stack(valid),hp,fp,ht,ft)
def manifest_entry(w):return {'sample_id':f'{w.scene_name}:{w.t0_token}','scene_name':w.scene_name,'history_tokens':list(w.history_tokens),'t0_token':w.t0_token,'future_tokens':list(w.future_tokens)}
def window_from_entry(e):return WindowTokens(str(e['scene_name']),tuple(e['history_tokens']),str(e['t0_token']),tuple(e['future_tokens']))
def write_scene_disjoint_manifest(source,output,*,seed=3407,dev_scene_fraction=.1,max_windows=None,dt_s=.5,tolerance_s=.05,fixed_train_scenes=None,fixed_dev_scenes=None,split_provenance=None):
    windows=list(source.iter_windows(history=6,future=6,stride=1,max_windows=max_windows));scenes=sorted({w.scene_name for w in windows})
    if fixed_train_scenes is not None or fixed_dev_scenes is not None:
        train=set(map(str,fixed_train_scenes or []));dev=set(map(str,fixed_dev_scenes or []))
        if train&dev or not train or not dev:raise RuntimeError('invalid fixed MSP scene split')
        protocol='msp_checkpoint_scene_split_v1'
    else:
        order=sorted(scenes,key=lambda s:hashlib.sha256(f'{seed}:{s}:split'.encode()).digest());ndev=max(1,int(round(len(order)*dev_scene_fraction))) if len(order)>1 else 0;dev=set(order[:ndev]);train=set(order[ndev:]);protocol='scene_hash_split_v1'
    rows=[];excluded=[]
    for w in windows:
        if w.scene_name not in train|dev:excluded.append({'sample_id':f'{w.scene_name}:{w.t0_token}','reason':'scene_not_in_fixed_split'});continue
        ts=np.asarray([_ts(source.nusc,t) for t in list(w.history_tokens)+list(w.future_tokens)])
        if np.max(np.abs(np.diff(ts)-dt_s))>tolerance_s:excluded.append({'sample_id':f'{w.scene_name}:{w.t0_token}','reason':'non_0p5s_cadence'});continue
        e=manifest_entry(w);e['split']='dev' if w.scene_name in dev else 'train';rows.append(e)
    payload={'version':'motion_transport_v1_manifest_v1','seed':seed,'dt_seconds':dt_s,'tolerance_seconds':tolerance_s,'split_protocol':protocol,'split_provenance':split_provenance or {},'train_scenes':sorted({r['scene_name'] for r in rows if r['split']=='train'}),'dev_scenes':sorted({r['scene_name'] for r in rows if r['split']=='dev'}),'num_train_windows':sum(r['split']=='train' for r in rows),'num_dev_windows':sum(r['split']=='dev' for r in rows),'excluded':excluded,'windows':rows}
    if not payload['num_train_windows'] or not payload['num_dev_windows']:raise RuntimeError('empty train/dev split')
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(payload,indent=2));return payload
def load_manifest(path):return json.loads(Path(path).read_text())
class ManifestDataset(Dataset):
    def __init__(self,source,manifest,split):
        p=load_manifest(manifest) if not isinstance(manifest,dict) else manifest;self.source=source;self.rows=[r for r in p['windows'] if r['split']==split]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        w=window_from_entry(self.rows[i]);return w,causal_from_window(self.source,w)
