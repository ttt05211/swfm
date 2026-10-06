#!/usr/bin/env python3
"""Same-eight-window actual backward: old, point/spatial, RAM/disk inputs.

    No training result/checkpoint selection. Geometry inputs only may persist;
    disk-only mode disables descriptor RAM LRU, so a full dataset need not fit
    in memory. All live predictions, projections and GT edit targets are fresh.
"""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import time
import numpy as np
import torch
from real_motion.canonical_repair_context import FixedCanonicalCache,SpatialCanonicalRepairHead
from real_motion.canonical_causal_repair import CanonicalRepairHead,map_canonical_evidence
from real_motion.local_replay_bundle import ReplayBundle,file_digest
from real_motion.runtime_config import make_prepare_config
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import load_exported_config
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.joint_column_full_common import build_fixed_geometry
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,finite_json
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import joint_probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','source-run','reference-run','out-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.out_dir);source=Path(a.source_run);reference=Path(a.reference_run)
    if out.exists():p.error('fresh output required; no original checkpoint/optimizer writes')
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA required')
    out.mkdir(parents=True);torch.set_num_threads(1);device=torch.device('cuda');bundle=ReplayBundle(a.bundle)
    report={'status':'running','actual_cuda':True,'saved_scientific_updates':False}
    caches=[]
    def save():(out/'timing.json').write_text(json.dumps(finite_json(report),ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        for member in ('runtime.yaml','checkpoints/epoch_0019.pt','checkpoints/clean_e14.pt'):
            if file_digest(reference/Path(member).name)!=bundle.manifest['members'][member]['sha256']:
                raise RuntimeError('reference snapshot mismatch')
        cfg=load_exported_config(reference/'runtime.yaml',bundle.manifest['config_fingerprint'])
        _,teacher=load_joint(reference/'epoch_0019.pt',device,reference_sha=CLEAN_SHA256,
            config_sha=bundle.manifest['config_fingerprint'],allow_diagnostic=True)
        teacher.eval().requires_grad_(False)
        provider=PilotProvider(reference/'clean_e14.pt',CLEAN_SHA256,make_prepare_config(cfg),device,2,teacher,None)
        cases=[];preparations=[];tick=time.perf_counter()
        for i,meta in enumerate(bundle.manifest['windows']):
            if meta['split']!='train' or meta['stratum']!='representative':continue
            record,raw,labels=bundle.window(i,labels=True);raw['future_gt_occ']=None
            causal={**raw,'_column_causal_preparation':build_fixed_geometry(raw,record,provider.pcfg,provider.strong,2,teacher.columns.config)}
            with torch.no_grad():
                outputs=teacher.motion(record,device)
                prep=provider.prepare_columns(None,record,include_gt=False,raw_window=causal,outputs=outputs)
            cases.append(dict(record=record,causal=causal,gt=labels['future_gt_occ'].numpy()))
            preparations.append(prep)
            if len(cases)==8:break
        report['fixed_registration_preparation_seconds']=time.perf_counter()-tick
        report['keys']=[[c['record']['scene_name'],c['record']['t0_token']] for c in cases]
        saved=torch.load(source/'candidate.pt',map_location='cpu',weights_only=True)
        if saved['teacher_sha256']!=bundle.manifest['teacher_sha256']:raise RuntimeError('candidate teacher mismatch')
        spatial=SpatialCanonicalRepairHead(teacher.columns.source_dim,normalized=saved['spatial_normalized']).to(device)
        spatial.load_state_dict(saved['head']);spatial.eval()
        point=CanonicalRepairHead(teacher.columns.source_dim).to(device)
        point.load_state_dict(torch.load(reference/'candidate.pt',map_location='cpu',weights_only=True)['head']);point.eval()
        comparisons={}
        for name,head,ram,disk in [('point_MC_RAM',point,128,None),('spatial_MC_RAM',spatial,128,None),
                                  ('spatial_MC_DISK',spatial,0,out/'descriptor_cache')]:
            cache=FixedCanonicalCache(ram,neighbors=hasattr(head,'encode_queries'),disk_root=disk,max_disk_mib=128)
            caches.append(cache);tick=time.perf_counter();values=[]
            for prep in preparations:
                e,_=cache.get(prep,provider.pcfg.grid);cache.static_conflicts(e,prep,provider.pcfg.grid)
                values.append(e)
            prefill=time.perf_counter()-tick
            # Disk mode must round-trip every array exactly, without RAM hits.
            if disk is not None:
                for prep,expected in zip(preparations,values):
                    observed,_=cache.get(prep,provider.pcfg.grid)
                    for field in ('features','labels','actor','classes','world','presence'):
                        if not np.array_equal(getattr(observed,field),getattr(expected,field)):
                            raise RuntimeError('disk descriptor roundtrip mismatch: '+field)
            result=joint_probe(provider,teacher,head,cases,device,fixed_cache=cache,causal_sampling=True)
            comparisons[name]=dict(prefill_seconds=prefill,**result)
            report['comparisons']=comparisons;save()
        report['status']='complete'
        report['quality_source']=str(source.resolve())
        report['note']='Only eight TRAIN windows and two measured batches per mode; warm fixed-input preparation/IO accounted separately; no convergence or L40S claim. New MC estimator differs from old GT-stratified column CE.'
        for member in ('runtime.yaml','checkpoints/epoch_0019.pt','checkpoints/clean_e14.pt'):
            if file_digest(reference/Path(member).name)!=bundle.manifest['members'][member]['sha256']:
                raise RuntimeError('original snapshot changed')
        save();text=['===== ACTUAL CCR JOINT TRAINING / RAM AND DISK =====']
        for name,row in comparisons.items():
            text.append(f'{name}: old={row["old_joint"]["seconds_per_window"]:.6f} new={row["CCR"]["seconds_per_window"]:.6f} s/window speedup={row["measured_speedup"]:.3f} prefill={row["prefill_seconds"]:.3f}s')
            text.append('stages='+json.dumps(finite_json(row['CCR']['stage_seconds_per_window'])))
            text.append('cache='+json.dumps(finite_json(row['fixed_history_cache'])))
        text.append(report['note']);(out/'summary.txt').write_text('\n'.join(text)+'\n',encoding='utf-8');print('\n'.join(text),flush=True)
    except BaseException as error:report.update(status='failed',error=str(error));save();raise
    finally:
        for cache in caches:cache.close()
        bundle.close()


if __name__=='__main__':main()
