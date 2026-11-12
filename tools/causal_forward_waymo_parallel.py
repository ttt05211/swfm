"""Continue the fixed epoch2 Waymo 10Hz evaluation with same-GPU workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.waymo_i2world import SHAPE, WaymoMetrics, file_sha256, fingerprint
from real_motion.waymo_i2world_10hz import WaymoI2World10HzSource, format_10hz_reports
from tools.causal_forward_transfer import ROOT, SOURCE_SHA256, TransferPredictor, summary
from tools.causal_forward_all.common import load_frozen
from tools.final_semantic_evaluation.engine import evaluate
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.geniedrive_eval_alignment import validate_grid
from tools.real_motion.static_evidence_selector_common import write_json


ROUTES = ('gt/transport', 'gt/semantic_refine')
DEFAULT_CHECKPOINT = ROOT / 'outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351/epoch_0002.pt'
DEFAULT_CONFIG = ROOT / 'configs/real_motion_occfm.yaml'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_counts(state, contract, keys, *, complete=False, volume=int(np.prod(SHAPE))):
    done = len(state['keys'])
    if (state['contract'] != contract or done > len(keys) or state['keys'] != keys[:done]
            or (complete and done != len(keys)) or state['ego_errors'] or set(state['counts']) != set(ROUTES)):
        raise ValueError('invalid or incomplete Waymo window state')
    counts = {}
    for name in ROUTES:
        fields = state['counts'][name]
        counts[name] = value = np.asarray(fields['confusion'])
        moving = np.asarray(fields['moving'])
        if (value.shape != (3, 18, 18) or value.dtype.kind not in 'iu' or np.any(value < 0)
                or np.any(value.sum((1, 2)) != done * volume)
                or moving.shape != value.shape or np.any(moving != 0)):
            raise ValueError(f'invalid Waymo integer counts for {name}')
    return counts


def partition(start, end):
    count = min(4, end - start)
    return [(start + i * (end - start) // count, start + (i + 1) * (end - start) // count)
            for i in range(count)]


def merge_counts(prefix_state, serial, shard_states, ranges, job_sha, *, volume=int(np.prod(SHAPE))):
    keys = serial['keys']
    total = read_counts(prefix_state, serial, keys, volume=volume)
    if list(map(tuple, ranges)) != partition(len(prefix_state['keys']), len(keys)) or len(shard_states) != len(ranges):
        raise ValueError('shards must partition the unevaluated suffix exactly once')
    for rank, (state, (start, end)) in enumerate(zip(shard_states, ranges)):
        counts = read_counts(state, dict(job_sha256=job_sha, rank=rank), keys[start:end], complete=True, volume=volume)
        for name in ROUTES:
            total[name] += counts[name]
    return total


def make_inputs(serial, waymo_root):
    pop = serial['population']
    metadata = pop['metadata']
    paths = [Path(f['path']) for f in metadata['source_files']]
    source = WaymoI2World10HzSource.from_files(waymo_root, info_file=paths[0], pose_file=paths[1],
                                             raw_free_label=metadata['raw_free_label'], cache_mib=128)
    keys = [[source.frames[w.anchor].scene, source.frames[w.anchor].token] for w in source.windows]
    if (source.metadata != metadata or source.manifest_fingerprint != pop['manifest_fingerprint']
            or keys != serial['keys'] or len(keys) != len(set(map(tuple, keys))) or len(keys) != 39987):
        raise ValueError('Waymo metadata or evaluation population differs from the serial run')
    return source


def check_original(serial, config, checkpoint):
    if (serial['protocol'] != 'forward_adaln_epoch2_transfer_v1' or serial['task'] != 'waymo10'
            or serial['population']['diagnostic_prefix'] or serial['source_sha256'] != SOURCE_SHA256
            or serial['torch_version'] != str(torch.__version__)
            or file_sha256(config) != serial['config_sha256'] or file_sha256(checkpoint) != SOURCE_SHA256
            or any(file_sha256(ROOT / name) != sha for name, sha in serial['implementation'].items())):
        raise ValueError('serial model, inputs, or evaluator changed')


def worker(out, rank, stop):
    job = read(out / 'contract.json')
    serial = job['serial']
    start, end, inventory = job['shards'][rank]
    source = make_inputs(serial, job['waymo_root'])
    jobs, keys = source.windows[start:end], serial['keys'][start:end]
    if source.preflight(jobs) != inventory:
        raise ValueError('Waymo shard files changed since the parallel split')
    checkpoint, config = Path(job['checkpoint']), Path(job['config'])
    check_original(serial, config, checkpoint)
    root = out / f'shard_{rank}'
    root.mkdir(exist_ok=True)
    with evaluation_lock(root):
        path = root / 'state.json'
        saved = read(path) if path.exists() else None
        contract = dict(job_sha256=fingerprint(job), rank=rank)
        if saved:
            read_counts(saved, contract, keys)
            if saved['keys'] == keys:
                return 0
        header, body = load_frozen(checkpoint, 'cuda:0')
        pcfg = make_prepare_config(load_runtime_config(config, header.get('overrides')))
        validate_grid(pcfg)
        predictor = TransferPredictor(body, pcfg, 'cuda:0', workers=job['cpu_workers'])
        try:
            evaluate(predictor, source, jobs, 'waymo10', keys, contract, saved=saved, stop_event=stop,
                     save=lambda state: write_json(path, state), checkpoint_every=64)
        except InterruptedError:
            return 130
        finally:
            predictor.close()
    return 0


def prepare_job(args, out, original):
    serial, state = read(original / 'contract.json'), read(original / 'state.json')
    checkpoint, config = args.checkpoint.resolve(), args.config.resolve()
    check_original(serial, config, checkpoint)
    read_counts(state, serial, serial['keys'])
    waymo_root = args.waymo_root.resolve()
    source = make_inputs(serial, waymo_root)
    if source.preflight(source.windows) != serial['population']['inventory']:
        raise ValueError('Waymo files changed since the serial run')
    ranges = partition(len(state['keys']), len(serial['keys']))
    shards = [[start, end, source.preflight(source.windows[start:end])] for start, end in ranges]
    job = dict(serial=serial, checkpoint=str(checkpoint), config=str(config), waymo_root=str(waymo_root),
               cpu_workers=args.cpu_workers,
               prefix_sha256=file_sha256(original / 'state.json'), shards=shards,
               coordinator_sha256=file_sha256(Path(__file__)))
    if args.resume:
        if (read(out / 'contract.json') != job
                or read(out / 'serial_prefix.json') != state):
            raise ValueError('parallel resume inputs, original prefix, or shard configuration changed')
    else:
        write_json(out / 'serial_prefix.json', state)
        write_json(out / 'contract.json', job)
    return job, state, ranges


def run_shards(out, job, prefix_windows, processes, stop):
    active = {}
    files = [out / f'shard_{rank}' / 'state.json' for rank in range(len(job['shards']))]
    keys, job_sha = job['serial']['keys'], fingerprint(job)
    pending = []
    for rank, (path, (start, end, _)) in enumerate(zip(files, job['shards'])):
        saved = read(path) if path.exists() else None
        if saved is not None:
            read_counts(saved, dict(job_sha256=job_sha, rank=rank), keys[start:end])
        if saved is None or len(saved['keys']) != end - start:
            pending.append(rank)
    last = float('-inf')
    try:
        while active or pending:
            if stop.is_set():
                return None
            for rank, child in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f'shard {rank} exited {code}; see {out / f"shard_{rank}.log"}')
                del active[rank]
            while pending and len(active) < processes:
                rank = pending.pop(0)
                with (out / f'shard_{rank}.log').open('ab') as log:
                    active[rank] = subprocess.Popen([sys.executable, '-u', '-m', 'tools.causal_forward_waymo_parallel',
                                                     '--out-dir', str(out), '--worker', str(rank)],
                                                    stdout=log, stderr=subprocess.STDOUT)
                print(f'WAYMO10_WORKER {rank} pid={active[rank].pid}', flush=True)
            now = time.monotonic()
            if now - last >= 30:
                done = prefix_windows + sum(len(read(path)['keys']) for path in files if path.exists())
                print(f'WAYMO10_PARALLEL {done}/{len(keys)} saved; imported={prefix_windows}', flush=True)
                last = now
            if active:
                stop.wait(1)
    finally:
        for child in active.values():
            if child.poll() is None:
                child.terminate()
        for child in active.values():
            child.wait()
    return [read(path) for path in files]


def main(argv=None, stop=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--waymo-root', type=Path, default=ROOT / 'data/waymo')
    parser.add_argument('--processes', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--cpu-workers', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(4), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is None and args.continue_from is None:
        parser.error('--continue-from is required')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        parser.error('set CUDA_VISIBLE_DEVICES to one GPU shared by all workers')
    stop = stop or threading.Event()
    out = args.out_dir.resolve()
    if args.worker is not None:
        if torch.cuda.device_count() != 1:
            raise RuntimeError('each worker requires exactly one visible GPU')
        torch.set_num_threads(1)
        return worker(out, args.worker, stop)
    original = args.continue_from.resolve()
    if original == out or original in out.parents:
        parser.error('use a separate output directory')
    with evaluation_lock(original):
        out.mkdir(parents=True, exist_ok=args.resume)
        with evaluation_lock(out):
            job, state, ranges = prepare_job(args, out, original)
            shards = run_shards(out, job, len(state['keys']), args.processes, stop)
            if shards is None:
                return 130
            serial = job['serial']
            counts = merge_counts(state, serial, shards, ranges, fingerprint(job))
            windows = len(serial['keys'])
            result = dict(status='complete', task='waymo10', windows=windows, source_sha256=SOURCE_SHA256,
                          population=serial['population'], imported_windows=len(state['keys']),
                          new_windows=windows-len(state['keys']), source_update=10292, scientific_updates=0,
                          reports=format_10hz_reports({n: WaymoMetrics(v, windows).report() for n, v in counts.items()}))
            write_json(out / 'evaluation.json', result)
            write_json(out / 'counts.json', {n: v.tolist() for n, v in counts.items()})
            text = (summary(result) + f'imported_windows={result["imported_windows"]}; '
                    f'new_windows={result["new_windows"]}\n')
            (out / 'summary.txt').write_text(text, encoding='utf-8')
            print(text, flush=True)
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: event.set())
    raise SystemExit(main(stop=event))
