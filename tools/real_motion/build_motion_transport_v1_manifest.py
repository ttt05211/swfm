#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.motion_transport_v1.config import load_config,get,make_grid,make_strong_cfg,make_msp_motion_cfg,make_msp_kta_cfg
from real_motion.motion_transport_v1.data import write_scene_disjoint_manifest,window_from_entry,causal_from_window
from real_motion.motion_transport_v1.source_adapter import decompose_strong_sources,extract_original_msp_candidates,map_msp_to_sources
from real_motion.motion_transport_v1.routing import sha256_file,msp_checkpoint_provenance
def _split(path):
    if not path or not Path(path).exists():return None,None,None
    ck=torch.load(path,map_location='cpu',weights_only=False);p=msp_checkpoint_provenance(path,ck);tr=set(p['train_scene_names']);dv=set(p['val_scene_names']);return (tr,dv,p) if tr and dv and not tr&dv else (None,None,p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=None);ap.add_argument('--override',action='append',default=[]);ap.add_argument('--output',default=None);ap.add_argument('--dev-scene-fraction',type=float,default=None);ap.add_argument('--max-windows',type=int,default=None);ap.add_argument('--no-source-scan',action='store_true');ap.add_argument('--no-split-from-msp',action='store_true');a=ap.parse_args();cfg=load_config(a.config,a.override);dr,info,msp=[get(cfg,x) for x in ('paths.dataroot','paths.info_pkl','paths.msp_checkpoint')];out=Path(a.output or get(cfg,'paths.manifest','data/motion_transport_v1/manifest.json'))
    if not dr or not info:raise RuntimeError('set dataroot/info_pkl')
    src=NuScenesWindowSource(dr,info_pkl=info,verbose=False);tr=dv=prov=None
    if not a.no_split_from_msp:tr,dv,prov=_split(msp)
    p=write_scene_disjoint_manifest(src,out,seed=int(get(cfg,'training.initial_seed',3407)),dev_scene_fraction=float(a.dev_scene_fraction if a.dev_scene_fraction is not None else get(cfg,'data.dev_scene_fraction',.1)),max_windows=a.max_windows,dt_s=float(get(cfg,'input.dt_seconds',.5)),tolerance_s=float(get(cfg,'input.timestamp_tolerance_seconds',.05)),fixed_train_scenes=tr,fixed_dev_scenes=dv,split_provenance=prov or {});cfg.setdefault('paths',{})['manifest']=str(out.resolve());out.with_name('resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    if not a.no_source_scan:
        grid=make_grid(cfg);scfg=make_strong_cfg(cfg);mcfg=make_msp_motion_cfg(cfg);kcfg=make_msp_kta_cfg(cfg);rows=[];classes=Counter();fallback=Counter();cover=[]
        for e in p['windows']:
            w=window_from_entry(e);c=causal_from_window(src,w);d=decompose_strong_sources(c,grid=grid,cfg=scfg,frame_dt_s=float(get(cfg,'input.dt_seconds',.5)),crop_radius_limit_m=float(get(cfg,'crop.max_source_xy_radius_about_centroid_m',11.2)));cand=extract_original_msp_candidates(c,grid=grid,motion_cfg=mcfg,kta_cfg=kcfg);map_msp_to_sources(d,cand,shape_xyz=tuple(grid.shape_hwd))
            for s in d.sources:classes[s.class_id]+=1;fallback[s.fallback_reason or 'eligible']+=1;cover.append(s.mapping_coverage)
            rows.append({'sample_id':c.sample_id,'split':e['split'],'num_sources':len(d.sources),'num_crop_eligible':sum(s.crop_eligible for s in d.sources),'num_msp_candidates':len(cand),'num_rest_voxels':len(d.rest_voxel_indices_t0),'mean_mapping_coverage':float(np.mean([s.mapping_coverage for s in d.sources])) if d.sources else 0.})
        sm={'version':'motion_transport_v1_source_manifest_v1','spec_version':'MT-V1-SPEC-2','window_manifest_sha256':sha256_file(out),'num_sources':sum(r['num_sources'] for r in rows),'per_class_sources':dict(classes),'crop_fallback':dict(fallback),'mean_msp_mapping_coverage':float(np.mean(cover)) if cover else 0.,'msp':prov,'windows':rows};out.with_name(out.stem+'_sources.json').write_text(json.dumps(sm,indent=2))
    print(json.dumps({k:p[k] for k in ('split_protocol','num_train_windows','num_dev_windows','train_scenes','dev_scenes')},indent=2))
if __name__=='__main__':main()
