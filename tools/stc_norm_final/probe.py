"""Matched single-GPU 2/4 process throughput, alternating order; no fitting."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import threading
import numpy as np
from .common import read,write,ROUTES


def command(a,folder,workers):
    args=[sys.executable,'-u','-m','tools.stc_norm_final.run','--out',str(folder),'--bundle',a.bundle,
        '--dev-cache',a.dev_cache,'--dev-info',a.dev_info,'--dataroot',a.dataroot,'--stc-root',a.stc_root,
        '--processes',str(workers),'--horizon-group',str(a.horizon_group),'--cpu-workers',str(a.cpu_workers),
        '--device',a.device,'--probe','--limit',str(a.windows)]
    if a.portable_subset:args.append('--portable-subset')
    return args


def compare(left,right):
    if left['contract']['keys']!=right['contract']['keys']:raise ValueError('probe populations differ')
    return {n:{k:right['metrics'][n]['average'][k]-left['metrics'][n]['average'][k]
               for k in ('mIoU','IoU')} for n in ROUTES}


def recommendation(runs):
    rates={w:[r['steady_windows_per_second'] for r in runs if r['processes']==w and r['exit_code']==0] for w in (2,4)}
    if len(rates[2])!=2:return None,rates
    # Prefer4 only when faster in BOTH alternating passes, not one noisy mean.
    use4=len(rates[4])==2 and all(four>two for four,two in zip(rates[4],rates[2]))
    return (4 if use4 else 2),rates


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--bundle',required=True)
    for n in ('dev-cache','dev-info','dataroot','stc-root'):ap.add_argument('--'+n,default='')
    ap.add_argument('--horizon-group',type=int,default=6);ap.add_argument('--cpu-workers',type=int,default=2)
    ap.add_argument('--device',default='cuda');ap.add_argument('--windows',type=int,default=32)
    ap.add_argument('--portable-subset',action='store_true');a=ap.parse_args(argv)
    if a.windows<4:raise ValueError('at least4 windows required')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);runs=[];results={};event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    for index,workers in enumerate((2,4,4,2)):
        folder=out/f'pass_{index+1}_workers{workers}';log=out/f'pass_{index+1}_workers{workers}.log';tic=time.monotonic()
        with log.open('w',encoding='utf-8') as handle:
            child=subprocess.Popen(command(a,folder,workers),stdout=handle,stderr=subprocess.STDOUT,env=os.environ.copy())
            try:
                while child.poll() is None:
                    if event.is_set():
                        child.terminate();child.wait();write(out/'speed.json',dict(status='stopped',runs=runs));return 130
                    time.sleep(.5)
                code=child.returncode
            finally:
                if child.poll() is None:child.terminate();child.wait()
        elapsed=time.monotonic()-tic
        row=dict(processes=workers,exit_code=code,whole_launch_seconds=elapsed,log=str(log))
        if code==0:
            r=read(folder/'results.json')
            if r['status']!='complete' or r['windows']!=min(a.windows,r['contract']['parent_windows']):raise ValueError('incomplete probe')
            row.update(steady_windows_per_second=r['speed']['steady_windows_per_second'],windows=r['windows'])
            results[index]=r
        runs.append(row);write(out/'speed.json',dict(status='running',runs=runs))
        print('NORM192_SPEED',row,flush=True)
    selected,rates=recommendation(runs)
    comparison={}
    for index,value in results.items():
        if results:comparison[str(index+1)]=compare(results[min(results)],value)
    value=dict(status='complete',runs=runs,recommended_processes=selected,
        mean_steady_windows_per_second={str(w):float(np.mean(rates[w])) if rates[w] else None for w in rates},
        score_deltas_diagnostic_only=comparison,
        scope='ONE GPU, same paired-method readouts/windows,2 warmups per worker, alternating2/4/4/2; NOT FPS',
        selection='4 only if faster in BOTH alternating passes, otherwise2;2 must complete both; THROUGHPUT only; no quality selection or byte-parity claim')
    write(out/'speed.json',value)
    lines=['===== NORM192 / ONE GPU / 2 vs4 PROCESS SPEED =====',f'recommended_processes={selected}']
    for w in (2,4):lines.append(f'{w} workers: {rates[w]} windows/second; mean={value["mean_steady_windows_per_second"][str(w)]}')
    lines+=['Warmup excluded from steady timing; whole launch timing retained. BOTH method readouts included; NOT FPS.',
        'OOM/failed runs stay failed; no tolerance changes, automatic quality selection or fitting.']
    (out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)
    return 0 if selected else 1


if __name__=='__main__':sys.exit(main() or 0)
