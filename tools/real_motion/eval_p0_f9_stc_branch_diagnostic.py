#!/usr/bin/env python3
"""Fixed dev64 frozen Strong/Transport/Joint attribution for all four settings."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contextlib import ExitStack
import json
import os
import signal
import threading
import numpy as np
import torch

from real_motion.waymo_i2world import fingerprint, file_sha256
from tools.real_motion import eval_p0_f9_joint_surface_stc as base
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.eval_p0_f9_joint_surface_stc_shared import EXTRA_FILES
from tools.real_motion.stc_branch_diagnostic import PROTOCOL, ROUTES, GROUPS, PlannerOriginAudit, evaluate, restore, summary
from tools.real_motion.waymo_zero_shot_common import write_json


def main(stop_event=None, argv=None):
    parser=base.parser(); parser.description=__doc__
    parser.add_argument('--planner-json',help='Optional exact original BEV-Planner JSON; never required to forecast')
    a=parser.parse_args(argv)
    if a.population!='dev64' or a.audit_only or a.execution!='native_parallel':
        parser.error('fixed dev64 branch attribution only; no full/input-only mode')
    if not 1<=a.cpu_workers<=8 or not 0<=a.frame_cache_mib<=4096 or a.checkpoint_every<1:
        parser.error('invalid bounded execution settings')
    out=Path(a.out_dir).resolve(); previous=None
    roots=[Path(p).resolve() for p in (a.dataroot,a.stc_root,a.plan_cache)]
    if any(out.is_relative_to(p) for p in roots) or any((p/'training.json').is_file() for p in (out,*out.parents)):
        parser.error('distinct output outside input/cache/training required')
    if a.resume: previous=json.loads((out/'contract.json').read_text(encoding='utf-8'))
    elif out.exists(): parser.error('new output required; cannot overwrite old results')
    if not a.population_manifest: parser.error('frozen dev64 parent manifest required')
    source=base.STCFourSettingSource.from_files(a.dataroot,a.stc_root,a.plan_cache,cache_mib=a.frame_cache_mib)
    manifest,_,_=base.load_manifest(a.population_manifest)
    windows,population=source.select('dev64',manifest['parent_keys']); inventory=source.preflight(windows)
    origin=PlannerOriginAudit(source,a.planner_json)
    if not torch.cuda.is_available(): parser.error('actual CUDA required for dataset evaluation')
    pcfg=base.make_prepare_config(base.load_runtime_config(a.config))
    if (tuple(pcfg.grid.shape_hwd)!=base.SHAPE or pcfg.future_frames!=6 or pcfg.frame_dt_s!=.5
        or pcfg.free_label!=17 or not np.allclose(pcfg.grid.voxel_size,(.4,)*3,rtol=0,atol=1e-12)
        or not np.allclose((pcfg.grid.x_min,pcfg.grid.y_min,pcfg.grid.z_min),(-40,-40,-1),rtol=0,atol=1e-12)):
        parser.error('unchanged trained Occ3D geometry required')
    if a.checkpoint: checkpoint=Path(a.checkpoint).resolve()
    elif previous: checkpoint=Path(previous['checkpoint']).resolve()
    else:
        if not a.runs_root or not a.run_dir: parser.error('checkpoint or fixed mean discovery required')
        bundle=base.find_frozen_bundle(a.runs_root,a.run_dir,a.source_bundle_dir)
        checkpoint=Path(bundle['candidates'][base.AVERAGE_NAME]['path']).resolve()
    if out.is_relative_to(checkpoint.parent): parser.error('output cannot be inside source mean directory')
    digest=file_sha256(checkpoint); root=Path(__file__).resolve().parents[2]
    files=(*base.IMPLEMENTATION_FILES,*EXTRA_FILES,'real_motion/stc_causal_geometry.py',
        'tools/real_motion/eval_p0_f9_stc_causal_geometry.py','tools/real_motion/stc_branch_diagnostic.py',
        'tools/real_motion/eval_p0_f9_stc_branch_diagnostic.py','tools/real_motion/run_p0_f9_stc_branch_diagnostic.sh')
    contract=dict(protocol=PROTOCOL,routes=ROUTES,groups=GROUPS,windows=len(windows),population=population,
        inventory=inventory,source=source.metadata,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        config_sha256=file_sha256(a.config),population_manifest_sha256=file_sha256(a.population_manifest),
        planner_origin=origin.metadata,implementation={f:file_sha256(root/f) for f in dict.fromkeys(files)},
        cpu_workers=a.cpu_workers,frame_cache_mib=a.frame_cache_mib,graphs=not a.no_graphs,
        parallel_majority=a.parallel_majority,
        runtime_environment={k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')},
        torch_version=str(torch.__version__))
    if previous and fingerprint(previous)!=fingerprint(contract): raise RuntimeError('branch diagnostic resume contract changed')
    if not a.resume: out.mkdir(parents=True)
    with ExitStack() as stack:
        stack.enter_context(base.evaluation_lock(out))
        saved=json.loads((out/'state.json').read_text(encoding='utf-8')) if a.resume else None
        if saved is not None: restore(saved,contract,source.shape)
        else: write_json(out/'contract.json',contract)
        meta,joint=base.load_evaluation_model(checkpoint,device='cuda',z_bins=16)
        if (meta.get('source_epochs')!=list(base.AVERAGE_EPOCHS) or not meta.get('averaging')
            or joint.transport.config.history_frames!=4 or file_sha256(checkpoint)!=digest):
            raise RuntimeError('unchanged frozen four-history mean required')
        torch.set_num_threads(1)
        predictor=Predictor(joint,pcfg,'cuda',workers=a.cpu_workers,graphs=not a.no_graphs,
                            parallel_majority=a.parallel_majority,geometry_mib=512)
        stack.callback(predictor.close)
        with (out/'progress.jsonl').open('a',encoding='utf-8') as handle:
            def progress(row):
                handle.write(json.dumps(row,allow_nan=False)+'\n'); handle.flush()
                if row['window']%8==0 or row['window']==1:
                    print(f'STC_BRANCH_DIAGNOSTIC {row["window"]}/{row["windows"]} seconds={row["seconds"]:.3f}',flush=True)
            result=evaluate(source,windows,predictor,contract,origin_audit=origin,saved=saved,
                save=lambda s:write_json(out/'state.json',s),progress=progress,
                stop_event=stop_event,checkpoint_every=a.checkpoint_every)
        if file_sha256(checkpoint)!=digest: raise RuntimeError('source mean changed during diagnostic')
        if origin.metadata.get('sha256') and file_sha256(origin.metadata['path'])!=origin.metadata['sha256']:
            raise RuntimeError('original planner JSON changed during diagnostic')
        result['contract']=contract; write_json(out/'evaluation.json',result)
        report=summary(result); (out/'summary.txt').write_text(report,encoding='utf-8')
        print(report,flush=True); print('RESULT: '+str(out/'summary.txt'),flush=True)
    return 0


if __name__=='__main__':
    stopped=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stopped.set())
    sys.exit(main(stopped))
