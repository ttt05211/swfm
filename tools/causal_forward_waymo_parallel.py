"""Continue the fixed epoch2 Waymo 10Hz evaluation with same-GPU workers."""
import argparse
from contextlib import ExitStack
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
VOLUME = int(np.prod(SHAPE))
MAX_SHARDS = 8
PARALLEL_PROTOCOL = 'forward_waymo10_parallel_v2'
DEFAULT_CHECKPOINT = ROOT / 'outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351/epoch_0002.pt'
DEFAULT_CONFIG = ROOT / 'configs/real_motion_occfm.yaml'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_counts(state, contract, keys, *, complete=False):
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
                or np.any(value.sum((1, 2)) != done * VOLUME)
                or moving.shape != value.shape or np.any(moving != 0)):
            raise ValueError(f'invalid Waymo integer counts for {name}')
    return counts


def gaps(intervals, windows):
    missing, cursor = [], 0
    for start, end in sorted(intervals):
        if not cursor <= start < end <= windows:
            raise ValueError('invalid or overlapping completed Waymo intervals')
        if cursor < start:
            missing.append((cursor, start))
        cursor = end
    if cursor < windows:
        missing.append((cursor, windows))
    return missing


def split_ranges(missing, processes):
    ranges = list(missing)
    while ranges and len(ranges) < processes:
        largest = max(range(len(ranges)), key=lambda i: ranges[i][1] - ranges[i][0])
        start, end = ranges[largest]
        if end - start <= 1:
            break
        middle = (start + end) // 2
        ranges[largest:largest+1] = [(start, middle), (middle, end)]
    return ranges


def read_seed(out, job):
    serial = job['serial']
    if job.get('protocol') != PARALLEL_PROTOCOL:
        state = read(out / 'serial_prefix.json')
        counts = read_counts(state, serial, serial['keys'])
        done = len(state['keys'])
        return ([(0, done)] if done else []), counts
    path = out / 'imported.json'
    if file_sha256(path) != job['imported_sha256']:
        raise ValueError('imported snapshot changed')
    state = read(path)
    intervals = [tuple(pair) for pair in state['intervals']]
    gaps(intervals, len(serial['keys']))
    done = sum(end - start for start, end in intervals)
    counts = {name: np.asarray(state['counts'][name]) for name in ROUTES}
    for value in counts.values():
        if (value.shape != (3, 18, 18) or value.dtype.kind not in 'iu' or np.any(value < 0)
                or np.any(value.sum((1, 2)) != done * VOLUME)):
            raise ValueError('invalid imported Waymo counts')
    return intervals, counts


def collect_shards(job, seed, shard_states, *, complete=False):
    intervals, counts = seed
    intervals = list(intervals)
    counts = {name: value.astype(np.int64, copy=True) for name, value in counts.items()}
    keys = job['serial']['keys']
    ranges = [(start, end) for start, end, _ in job['shards']]
    if gaps(intervals + ranges, len(keys)) or len(shard_states) != len(ranges):
        raise ValueError('Waymo shards and imported windows must cover the population exactly once')
    job_sha = fingerprint(job)
    for rank, (state, (start, end)) in enumerate(zip(shard_states, ranges)):
        if state is None and not complete:
            continue
        if state is None:
            raise ValueError(f'missing state for shard {rank}')
        shard_counts = read_counts(state, dict(job_sha256=job_sha, rank=rank), keys[start:end], complete=complete)
        done = len(state['keys'])
        if done:
            intervals.append((start, start + done))
        for name in ROUTES:
            counts[name] += shard_counts[name]
    return intervals, counts


def import_source(original):
    contract = read(original / 'contract.json')
    if 'serial' not in contract:
        state = read(original / 'state.json')
        counts = read_counts(state, contract, contract['keys'])
        done = len(state['keys'])
        return contract, (([(0, done)] if done else []), counts)
    with ExitStack() as stack:
        states = []
        for rank, _ in enumerate(contract['shards']):
            shard = original / f'shard_{rank}'
            if shard.is_dir():
                stack.enter_context(evaluation_lock(shard))
            path = shard / 'state.json'
            states.append(read(path) if path.exists() else None)
        seed = collect_shards(contract, read_seed(original, contract), states)
    return contract['serial'], seed


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
    if args.resume:
        job = read(out / 'contract.json')
        if job.get('source_dir', str(original)) != str(original):
            raise ValueError('resume with the same continuation source')
        check_original(job['serial'], Path(job['config']), Path(job['checkpoint']))
        return job, read_seed(out, job)
    serial, seed = import_source(original)
    checkpoint, config = args.checkpoint.resolve(), args.config.resolve()
    check_original(serial, config, checkpoint)
    waymo_root = args.waymo_root.resolve()
    source = make_inputs(serial, waymo_root)
    if source.preflight(source.windows) != serial['population']['inventory']:
        raise ValueError('Waymo files changed since the serial run')
    intervals, counts = seed
    ranges = split_ranges(gaps(intervals, len(serial['keys'])), args.processes)
    shards = [[start, end, source.preflight(source.windows[start:end])] for start, end in ranges]
    snapshot = out / 'imported.json'
    write_json(snapshot, dict(intervals=sorted(intervals), counts={name: v.tolist() for name, v in counts.items()}))
    job = dict(protocol=PARALLEL_PROTOCOL, serial=serial, source_dir=str(original), checkpoint=str(checkpoint),
               config=str(config), waymo_root=str(waymo_root), cpu_workers=args.cpu_workers,
               imported_sha256=file_sha256(snapshot), shards=shards,
               coordinator_sha256=file_sha256(Path(__file__)))
    write_json(out / 'contract.json', job)
    return job, seed


def run_shards(out, job, imported_windows, processes, stop):
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
                done = imported_windows + sum(len(read(path)['keys']) for path in files if path.exists())
                print(f'WAYMO10_PARALLEL {done}/{len(keys)} saved; imported={imported_windows}', flush=True)
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
    parser.add_argument('--processes', type=int, choices=range(1, MAX_SHARDS+1), default=4)
    parser.add_argument('--cpu-workers', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(MAX_SHARDS), help=argparse.SUPPRESS)
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
            job, seed = prepare_job(args, out, original)
            imported = sum(end - start for start, end in seed[0])
            shards = run_shards(out, job, imported, args.processes, stop)
            if shards is None:
                return 130
            serial = job['serial']
            _, counts = collect_shards(job, seed, shards, complete=True)
            windows = len(serial['keys'])
            result = dict(status='complete', task='waymo10', windows=windows, source_sha256=SOURCE_SHA256,
                          population=serial['population'], imported_windows=imported,
                          new_windows=windows-imported, source_update=10292, scientific_updates=0,
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
