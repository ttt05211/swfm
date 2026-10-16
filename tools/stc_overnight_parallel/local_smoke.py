"""Real portable CUDA replay; not a server speed claim or full target proof."""
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
from tools.final_semantic_evaluation.predictor import Predictor
from tools.shared_source_track.adapter import SharedTrackPredictor
from tools.stc_overnight.bundle import Bundle
from tools.stc_overnight import verify_all as old
from . import run
from .common import read,write,routes,serialized


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True);ap.add_argument('--bundle',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args(argv);torch.set_num_threads(1);torch.manual_seed(21)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);reference=Path(a.reference).resolve()
    contract=read(reference/'contract.json')
    if not contract['portable_subset'] or contract['population']!='dev64':raise ValueError('portable64 reference required')
    # This is a NEW, empty fixture: no old predictions/counts are imported.
    # The local historical reference predates the final serial entry checksum.
    # Bind this new fixture to the actual current execution, without changing
    # the reference or weakening inspect_source for real server migration.
    contract['implementation']={name:old.base.sha256(Path(old.__file__).parent/name)
        for name in contract['implementation']}
    serial=out/'serial_empty_fixture';serial.mkdir()
    write(serial/'contract.json',contract)
    fit=(reference/'unlabeled_posterior_fit.json').read_bytes();(serial/'unlabeled_posterior_fit.json').write_bytes(fit)
    write(serial/'state.json',dict(contract=contract,keys=[],
        counts=serialized({n:old.metrics.empty(3) for n in routes(contract)}),
        calibration_sha256=old.base.sha256(serial/'unlabeled_posterior_fit.json')))
    b=Bundle(a.bundle,cache_mib=64);body,saved,policy=b.load_model(contract['model'],'cuda')
    p=(SharedTrackPredictor if policy=='shared_pinned' else Predictor)(body,b.pcfg,'cuda',workers=2)
    r=next(r for r in b.records if r['role']=='reference64');rec,raw,times=b.history(r)
    row=p.prepare(rec,raw,times=times);modified,_=old.candidate(row,p.provider,times)
    bias=read(reference/'unlabeled_posterior_fit.json')['bias']
    torch.cuda.reset_peak_memory_stats()
    h1=run.forecast(p,modified,r['future_poses'],bias,True,1)
    h6=run.forecast(p,modified,r['future_poses'],bias,True,6)
    changed={n:int(np.count_nonzero(h1[n]!=h6[n])) for n in h1}
    probe=dict(key=[r['scene_name'],r['t0_token']],changed_voxels_six_frames=changed,
        strict_semantic_byte_parity=not any(changed.values()),peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        scope='one real window H1/H6 diagnostic; numerical FAIL is recorded, not waived')
    write(out/'one_window_h6_probe.json',probe);print('REAL_H6_PROBE',probe,flush=True)
    p.close();b.close();del p,body,b,row,modified,h1,h6;gc.collect();torch.cuda.empty_cache()
    # RTX3050 has only4GiB and desktop use. Test TWO real CUDA processes with H1
    # to verify orchestration/merge; server H6 concurrency is NOT claimed tested.
    result=run.main(['--source-dir',str(serial),'--out-dir',str(out/'two_process_H1'),
        '--bundle',a.bundle,'--horizon-group','1'])
    if result!=0:raise RuntimeError('real two-process replay failed')
    replay=read(out/'two_process_H1/results.json');ref=read(reference/'results.json')
    delta={n:{k:replay['metrics'][n]['average'][k]-ref['metrics'][n]['average'][k]
        for k in ('mIoU','IoU')} for n in replay['metrics']}
    write(out/'smoke_report.json',dict(status='complete',h6_one_window=probe,paired_metric_differences=delta,
        processes=2,physical_GPUs=1,parallel_group_size=1,windows=replay['windows'],
        scope='real CUDA64 wiring and integer-merge check, NOT server H6 speed/FPS or ALL verification'))
    print('REAL_TWO_PROCESS64_DIFFERENCES',delta,flush=True)


if __name__=='__main__':main()
