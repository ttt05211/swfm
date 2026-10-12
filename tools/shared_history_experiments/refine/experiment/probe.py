"""Real full-six training probe; temporary optimizer updates are NEVER saved."""
import json
import math
from pathlib import Path
import signal
import sys
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as legacy
from .. import engine, run


def aggregate(rows, links, device, warmup, measured):
    windows = sum(r['windows'] for r in rows)
    seconds = sum(r['seconds'] for r in rows)
    reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
    total = torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else 0
    free = torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else 0
    finite = bool(rows) and all(math.isfinite(r[k]) for r in rows
        for k in ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry', 'grad_norm', 'seconds'))
    gradient_pass = all(math.isfinite(links.get(k, 0.)) and links.get(k, 0.) > 0 for k in
        ('occupancy_to_memory', 'occupancy_to_shared_history', 'occupancy_to_xy'))
    headroom = total > 0 and reserved <= .9*total and free >= .1*total
    return dict(status='complete' if len(rows) == measured else 'incomplete',
        scientific_updates_published=0, temporary_updates=warmup+len(rows), warmup_updates=warmup,
        measured_updates=len(rows), windows=windows, seconds_per_window=seconds/max(1, windows),
        p90_update_seconds_per_window=float(np.percentile([r['seconds']/r['windows'] for r in rows], 90)) if rows else None,
        host_stages_seconds_per_window={k: sum(r['stages_seconds'].get(k, 0.) for r in rows)/max(1, windows)
            for k in set().union(*(r['stages_seconds'] for r in rows))},
        input_preparation_seconds_per_window=sum(r.get('input_preparation_seconds', 0.) for r in rows)/max(1, windows),
        peak_reserved_mib=reserved/2**20, peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20
            if device.type == 'cuda' else 0., free_vram_mib=free/2**20,
        gradient_links=links, finite_losses_and_gradients=finite, gradient_gate=gradient_pass,
        memory_headroom_ge_10percent=headroom, actual_cuda=device.type == 'cuda',
        pass_gate=len(rows) == measured and finite and gradient_pass and headroom,
        timing='warm real forward/full6 CE-Lovasz/backward/Adam; input preparation reported separately; NOT FPS',
        training_two_epochs_hours_if_representative=seconds/max(1, windows)*4086*2/3600)


def main(event=None, argv=None, *, warmup=2, measured=8):
    event = event or threading.Event()
    argv = list(sys.argv[1:] if argv is None else argv)
    a = legacy.parser().parse_args(argv)
    if a.resume or a.evaluate_only or a.max_updates:
        raise ValueError('fresh discarded probe only; no resume/evaluation/max-updates')
    out = Path(a.out_dir)
    original_train, original_save = engine.train_arm, legacy.save_resume_checkpoint
    rows, links, device = [], {}, torch.device(a.device)
    completed_updates = 0
    def train(model, optimizer, provider, batch, mode, **kwargs):
        nonlocal completed_updates
        # Gradient audit on second warmup ONLY: zero-initial last layer has now updated.
        kwargs['probe'] = kwargs['update'] == warmup
        result = original_train(model, optimizer, provider, batch, mode, **kwargs)
        completed_updates = kwargs['update']
        result['input_preparation_seconds'] = sum(batch[0][1].get('_joint_prepare_timing', {}).values())
        if kwargs['update'] == warmup:
            links.update(result['gradient_links'])
            if provider.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(provider.device)
        if kwargs['update'] > warmup: rows.append(result)
        return result
    report = None
    try:
        engine.train_arm = train
        legacy.save_resume_checkpoint = lambda *args, **kwargs: None
        result = run.main(event, argv+['--max-updates', str(warmup+measured)])
        report = aggregate(rows, links, device, warmup, measured)
        report['temporary_updates'] = completed_updates
        report['runner_returncode'] = result or 0
        report['pass_gate'] = report['pass_gate'] and not result and not event.is_set()
    except Exception as error:
        report = dict(status='failed', pass_gate=False, scientific_updates_published=0, error=repr(error))
        raise
    finally:
        engine.train_arm, legacy.save_resume_checkpoint = original_train, original_save
        if report is not None:
            out.mkdir(parents=True, exist_ok=True)
            legacy.write_json(out/'probe.json', report)
            print('REFINER_PROBE '+json.dumps(report), flush=True)
    return 0 if report['pass_gate'] or event.is_set() else 1


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
