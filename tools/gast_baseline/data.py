"""CPU-only selected metadata streaming; no SDK, model or CUDA import required."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from .common import ROOT,GRID,sha,checked_archive,select_assets,read_json,write_json,file_inventory


class Metadata:
    """Selected nuScenes table rows, matching SDK's LIDAR_TOP reverse lookup."""
    def __init__(self,root,tokens):
        import ijson
        folder=Path(root)/'v1.0-trainval';tokens=set(tokens)
        def small(name):return read_json(folder/(name+'.json'))
        sensors={r['token']:r for r in small('sensor')}
        lidar={r['token'] for r in small('calibrated_sensor') if sensors[r['sensor_token']]['channel']=='LIDAR_TOP'}
        samples={r['token']:dict(r,data={}) for r in small('sample') if r['token'] in tokens}
        if set(samples)!=tokens:raise ValueError('selected sample token missing from nuScenes tables')
        selected={}
        with (folder/'sample_data.json').open('rb') as f:
            for row in ijson.items(f,'item',use_float=True):
                if row['sample_token'] in tokens and row['is_key_frame'] and row['calibrated_sensor_token'] in lidar:
                    sample=samples[row['sample_token']]
                    if sample['data']:raise ValueError('duplicate selected LIDAR_TOP keyframe')
                    sample['data']['LIDAR_TOP']=row['token'];selected[row['token']]=row
        if any('LIDAR_TOP' not in r['data'] for r in samples.values()):raise ValueError('selected LIDAR_TOP metadata missing')
        wanted={r['ego_pose_token'] for r in selected.values()};egos={}
        with (folder/'ego_pose.json').open('rb') as f:
            for row in ijson.items(f,'item',use_float=True):
                if row['token'] in wanted:egos[row['token']]=row
        if set(egos)!=wanted:raise ValueError('selected ego pose missing')
        self.tables=dict(sample=samples,sample_data=selected,ego_pose=egos,scene={r['token']:r for r in small('scene')})
    def get(self,table,token):return self.tables[table][token]


def info_index(path):
    with Path(path).open('rb') as f:payload=pickle.load(f)
    rows=payload['infos']
    rows=[r for values in rows.values() for r in values] if isinstance(rows,dict) else rows
    if not isinstance(rows,list):raise ValueError('infos must be frame list or scene->frame lists')
    result={}
    for row in rows:
        token=row['token']
        if token in result:raise ValueError('duplicate info token: '+token)
        result[token]=row
    return result


def required_fields(manifest):
    states=set();commands=set();all_tokens=set()
    for c in manifest['cases'].values():
        hist,future=c['history_tokens'],c['future_tokens'];all_tokens.update(hist+future)
        states.update(hist);commands.update(future[:6])
        if len(future)==12:states.update(future[2:6]);commands.update(future[6:12])
    return states,commands,all_tokens


def pose_matrix(translation,rotation):
    q=np.asarray(rotation,np.float64)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-5:raise ValueError('unit wxyz quaternion required')
    w,x,y,z=q/np.linalg.norm(q)
    pose=np.eye(4)
    pose[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    pose[:3,3]=translation
    return pose


def probe_info(path,manifest):
    idx=info_index(path);states,commands,all_tokens=required_fields(manifest);missing=[]
    for t in sorted(all_tokens):
        if t not in idx:missing.append([t,'info row']);continue
        if t in states:
            a=np.asarray(idx[t].get('gt_ego_lcf_feat',[])).reshape(-1)
            if len(a)<5 or not np.isfinite(a).all():missing.append([t,'gt_ego_lcf_feat[0,1,4]'])
        if t in commands:
            a=np.asarray(idx[t].get('gt_ego_fut_cmd',[])).reshape(-1)
            if a.shape!=(3,) or not np.isin(a,[0,1]).all() or a.sum()!=1:missing.append([t,'one-hot gt_ego_fut_cmd'])
    return idx,missing


def paths(a):
    work=a.work.resolve();assets=select_assets(a.assets);m=checked_archive(assets)
    preferred=work/'downloads/official_val.pkl'
    info=a.info.resolve() if a.info else (preferred if preferred.is_file() else a.dataroot/'nuscenes_infos_val_temporal_v3_scene.pkl')
    return work,assets,m,info


def prepare(a,*,source=None):
    work,assets,m,info=paths(a);idx,missing=probe_info(info,m)
    if missing:raise ValueError('Missing genuine ego fields; run step.sh infos. Examples='+str(missing[:5]))
    if source is None:
        source=Metadata(a.dataroot,{t for c in m['cases'].values() for t in c['history_tokens']+c['future_tokens']})
    all_paths=[];result={};metadata=[]
    states,commands,_=required_fields(m)
    for key,c in m['cases'].items():
        tokens=c['history_tokens']+c['future_tokens'];samples=[source.get('sample',t) for t in tokens]
        if (len({s['scene_token'] for s in samples})!=1
                or source.get('scene',samples[0]['scene_token'])['name']!=c['scene']
                or any(x['next']!=y['token'] or y['prev']!=x['token'] for x,y in zip(samples,samples[1:]))):
            raise ValueError('non-contiguous/cross-scene case: '+key)
        times=np.array([s['timestamp'] for s in samples],np.float64)/1e6
        if not np.all((np.diff(times)>.35)&(np.diff(times)<.65)):raise ValueError('expected actual 2Hz keyframes: '+key)
        frames=[]
        for sample,token in zip(samples,tokens):
            sd=source.get('sample_data',sample['data']['LIDAR_TOP']);ego=source.get('ego_pose',sd['ego_pose_token'])
            translation=np.asarray(ego['translation'],np.float64);rotation=np.asarray(ego['rotation'],np.float64)
            pose=pose_matrix(translation,rotation)
            record=idx[token]
            for name,truth in (('ego2global_translation',translation),('ego2global_rotation',rotation)):
                if name in record:
                    value=np.asarray(record[name],np.float64)
                    if name.endswith('rotation') and np.allclose(value,-truth,atol=1e-7,rtol=0):value=-value
                    if not np.allclose(value,truth,atol=1e-5,rtol=0):raise ValueError('info is not LIDAR_TOP ego2global: '+token)
            path=(a.dataroot/'gts'/c['scene']/token/'labels.npz').resolve()
            if not path.is_file() or not path.stat().st_size:raise FileNotFoundError(path)
            all_paths.append(path)
            feat=np.asarray(record['gt_ego_lcf_feat'],np.float32).reshape(-1)[[0,1,4]].tolist() if token in states else None
            cmd=np.asarray(record['gt_ego_fut_cmd'],np.float32).reshape(3).tolist() if token in commands else None
            frames.append(dict(token=token,semantics=str(path),pose=pose.tolist(),translation=translation.tolist(),
                rotation=rotation.tolist(),lcf=feat,command=cmd,timestamp_s=float(sample['timestamp']/1e6)))
        if not np.allclose([f['pose'] for f in frames[4:]],c['future_ego_poses'],atol=1e-7,rtol=0):
            raise ValueError('future poses differ from CarryWorld archive: '+key)
        result[key]=dict(original=c,frames=frames,report_indices=[1,3,5] if key.startswith('short') else [1,3,5,7,9,11])
        print('GAST_PREPARED',key,c['scene'],flush=True)
    for name in ('sample.json','sample_data.json','ego_pose.json','scene.json','sensor.json','calibrated_sensor.json'):
        p=a.dataroot/'v1.0-trainval'/name
        if p.is_file():metadata.append([str(p.resolve()),sha(p)])
    payload=dict(format='GAST_CPU_prepared_v1',assets=str(assets),assets_sha256=sha(assets),info=str(info),info_sha256=sha(info),
        dataroot=str(a.dataroot.resolve()),grid=GRID,cases=result,inventory=file_inventory(all_paths),metadata=metadata,
        preparation='CPU only; genuine ego fields; future occupancy paths only; no new dense cache')
    target=work/'prepared.json'
    if target.exists() and read_json(target)!=payload:raise ValueError('prepared data changed; use a NEW GAST_WORK directory')
    write_json(target,payload)
    print('GAST_DATA_READY',len(result),'cases;',target,flush=True)
    return payload


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('probe','prepare'))
    p.add_argument('--work',type=Path,required=True);p.add_argument('--assets',type=Path)
    p.add_argument('--dataroot',type=Path,default=Path('/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes'))
    p.add_argument('--info',type=Path);return p


def main():
    a=parser().parse_args()
    if a.action=='prepare':prepare(a)
    else:
        _,assets,m,info=paths(a)
        if not info.is_file():
            print('GAST_INFO_MISSING',info);raise SystemExit(2)
        _,missing=probe_info(info,m)
        print(json.dumps(dict(assets=str(assets),cases=len(m['cases']),info=str(info),missing=len(missing),examples=missing[:5])))
        if missing:raise SystemExit(2)


if __name__=='__main__':main()
