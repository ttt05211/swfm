#!/usr/bin/env python3
"""Bounded synthetic head overhead check; NOT L40S FPS or quality evidence."""
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import copy
import json
import time

import numpy as np
import torch

from real_motion.canonical_causal_repair import FEATURE_DIM, CanonicalRepairHead, CanonicalEvidence, RepairPlan
from real_motion.ccr_frozen_b import frozen_b_probabilities
from real_motion.surface_canonical_repair import SurfaceCanonicalRepairHead, SURFACE_DIM, PHASE_DIM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--points',type=int,default=48000);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    if not 1<=a.points<=100000 or not 1<=a.repeats<=20:p.error('bounded microbenchmark required')
    device=torch.device(a.device);torch.set_num_threads(1);torch.manual_seed(34)
    rng=np.random.default_rng(34);n=a.points
    dynamic=np.arange(n)<n//8  # actual canonical population groups actors/roles
    actors=np.where(dynamic,np.arange(n)%32,-2).astype(np.int32)
    classes=np.where(dynamic,4,11).astype(np.uint8)
    features=rng.standard_normal((n,FEATURE_DIM+SURFACE_DIM)).astype(np.float32)
    labels=np.broadcast_to(classes[:,None],(n,4)).copy()
    context=rng.standard_normal((n,6,8+PHASE_DIM)).astype(np.float32)
    original=CanonicalRepairHead(128).to(device).eval().requires_grad_(False)
    surface=SurfaceCanonicalRepairHead(128).to(device);surface.initialize_from(original)
    surface.eval().requires_grad_(False)
    unrouted=copy.deepcopy(surface)
    unrouted.inference_batch=lambda live,actors: live
    heads={'frozen_B':original,'surface_unrouted':unrouted,'surface_CCR':surface}
    def inputs(augmented):
        evidence=CanonicalEvidence(features if augmented else features[:,:FEATURE_DIM],labels,actors,classes,
            np.zeros((n,3),np.float64),np.ones((n,4),bool),{})
        plan=RepairPlan(np.zeros((n,6),np.int64),np.full((n,6),17,np.uint8),np.full((n,6),17,np.uint8),
            np.ones((n,6,2),bool),context if augmented else context[...,:8])
        return evidence,plan
    output={'history_source_context':torch.randn(32,128,device=device),
            'future_transport_queries':torch.randn(32,6,128,device=device)}
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    times={name:[] for name in heads};expected={}
    for name,head in heads.items():expected[name]=frozen_b_probabilities(head,*inputs(name!='frozen_B'),output,device)
    if any(not np.array_equal(expected['frozen_B'],rows) for rows in expected.values()):
        raise RuntimeError('zero-geometry initialization does not reproduce B')
    for repeat in range(a.repeats):
        names=list(heads)
        if repeat%2:names.reverse()
        for name in names:
            sync();tick=time.perf_counter()
            actual=frozen_b_probabilities(heads[name],*inputs(name!='frozen_B'),output,device)
            sync();times[name].append(time.perf_counter()-tick)
            if not np.array_equal(actual,expected[name]):raise RuntimeError('repeated head probability bytes changed')
    result=dict(scope='synthetic_HEAD_ONLY_six_readouts_upload_D2H_NOT_forecast_FPS_NOT_quality',
        excludes='history atlas/descriptor prep, live projection phase prep, motion, prior, compositor',
        device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        points=n,dynamic_fraction=float(dynamic.mean()),zero_init_probability_bytes_exact=True,
        milliseconds={name:1000*float(np.mean(rows)) for name,rows in times.items()},
        repeats=a.repeats)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
