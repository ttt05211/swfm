#!/usr/bin/env python3
"""Train two small ego heads on an EXISTING read-only feature bank; no WM training."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import signal
import threading

import torch

from real_motion.ego_trajectory_head import EgoHeadConfig,HistoryEgoTrajectoryHead
from tools.real_motion.surface_ego_ablation_common import (
    PROTOCOL,load_source_bank,recover_original_initialization,paired_train,verify_sources,
)
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.waymo_zero_shot_common import write_json

ROOT=Path(__file__).resolve().parents[2]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',required=True,help='completed original TRAIN1024 ego-head screen')
    p.add_argument('--out-dir',required=True);p.add_argument('--max-updates',type=int,default=2000)
    p.add_argument('--checkpoint-every',type=int,default=32);p.add_argument('--monitor-every',type=int,default=200)
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda')
    return p


def main(stop_event=None,argv=None):
    a=parser().parse_args(argv);out=Path(a.out_dir).resolve();source=Path(a.source_dir).resolve()
    if out==source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('new experiment directory must be separate from original screen')
    if out.exists() and not a.resume:raise FileExistsError('NEW paired output required, or explicit --resume SAME directory')
    if a.resume and not (out/'contract.json').is_file():raise ValueError('missing paired experiment contract')
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    rows,saved,audit=load_source_bank(source,ROOT)
    old_schedule=saved['contract']['schedule'];count=len(rows)
    if count!=1024 or saved['updates']!=320 or old_schedule['batch_size']!=64:
        raise ValueError('this experiment requires the approved TRAIN1024 / batch64 / old320 source')
    initial=recover_original_initialization(saved,audit['frozen_checkpoint']).to(a.device)
    old=HistoryEgoTrajectoryHead(EgoHeadConfig(**saved['config'])).to(a.device).eval().requires_grad_(False)
    old.load_state_dict(saved['state_dict'],strict=True)
    contract=dict(protocol=PROTOCOL,source=audit,head_config=saved['config'],
        schedule=dict(max_updates=a.max_updates,batch_size=64,lr=old_schedule['lr'],min_lr=old_schedule['min_lr'],seed=old_schedule['seed']),
        objectives=dict(A_original='original XY Smooth-L1 + original yaw_weight*(1-cos yaw)',
                        B_geometry='one Smooth-L1 over five SE2 transformed fixed reference points',radius_m=10.),
        initializer='original seed + original frozen model construction; exact saved CPU RNG witness verified',
        TRAIN_reports='entire SAME in-sample bank; NOT held-out validation',
        selection='fixed final update only; no dev-selected head/threshold/radius/budget',
        device=str(torch.device(a.device)),torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
    if a.resume:
        if json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:raise RuntimeError('paired training contract changed')
    else:out.mkdir(parents=True);write_json(out/'contract.json',contract)
    with evaluation_lock(out):
        result=paired_train(initial,rows,old,contract,out,max_updates=a.max_updates,batch_size=64,
            lr=old_schedule['lr'],min_lr=old_schedule['min_lr'],seed=old_schedule['seed'],radius_m=10.,
            checkpoint_every=a.checkpoint_every,monitor_every=a.monitor_every,resume=a.resume,stop_event=stop_event)
        verify_sources(audit)
        write_json(out/'training_summary.json',result)
        text=training_text(result,count);(out/'training_summary.txt').write_text(text,encoding='utf-8');print(text,flush=True)
    return 0 if result['status']=='complete' else 130


def training_text(result,count):
    lines=['===== PAIRED EGO HEAD TRAINING =====',
        f'status={result["status"]}; windows={count}; updates_each={result["updates"]}; complete_epochs={result["completed_epochs"]}',
        'WM/CCR frozen; legacy bank read only; A/B same original initializer/order/Adam/cosine.',
        'A: original objective. B: single fixed-R10m geometry objective. No dev tuning.',
        f'Full TRAIN report update={result["last_TRAIN_monitor_update"]} (IN-SAMPLE, not validation)',
        'candidate        ADE_m   FDE_3s_m   XY@1/2/3s_m   yaw@1/2/3s_deg']
    for name,r in result['TRAIN_reports'].items():
        xy='/'.join(f'{r["xy_mean_m"][i]:.3f}' for i in (1,3,5))
        yaw='/'.join(f'{r["yaw_mean_deg"][i]:.3f}' for i in (1,3,5))
        lines.append(f'{name:14} {r["ADE_m"]:.6f} {r["FDE_3s_m"]:.6f} {xy} {yaw}')
    lines+=['command_counts_by_horizon(right/left/straight)='+json.dumps(result['TRAIN_reports']['prior']['command_counts_by_horizon']),
        'No automatic retry, full-data training or promotion.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
