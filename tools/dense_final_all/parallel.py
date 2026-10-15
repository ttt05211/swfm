"""TWO independent inference processes on ONE visible GPU, raw-count merge."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.dense_valid_epoch3.run import select_source, validate_source
from tools.final_semantic_evaluation import metrics
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .run import PROTOCOL, select_jobs, summary

PARALLEL_PROTOCOL = 'frozen_dense_epoch2_all4369_TWOprocesses_ONE_GPU_integer_merge_v1'


def merge_states(states, keys, source_sha, voxel_count, modality):
    if len(states) != 2: raise RuntimeError('exactly TWO same-GPU shards required')
    totals = {name: metrics.empty(3) for name in ('gt/transport', 'gt/semantic_refine')}
    seen = []
    for rank, state in enumerate(states):
        contract = state['contract']; c = contract['base']; expected = keys[rank::2]
        if (c['protocol'] != PROTOCOL or c['source_sha256'] != source_sha or c['keys'] != expected
                or c['shard'] != [rank, 2] or c['windows'] != len(expected) or c['parent_windows'] != 4369
                or c['known'] != dict(occ='dense_valid', stc='dense_valid')
                or contract['modality'] != modality or state['keys'] != expected or set(state['counts']) != set(totals)
                or state['ego_errors']):
            raise RuntimeError('shard identity/known/population/source differs')
        seen += list(map(tuple, state['keys']))
        for name, arrays in state['counts'].items():
            if set(arrays) != {'confusion', 'moving'}: raise RuntimeError('shard count fields differ')
            parsed = {}
            for field, array in arrays.items():
                value = np.asarray(array)
                if value.shape != (3, 18, 18) or value.dtype.kind not in 'iu' or np.any(value < 0):
                    raise RuntimeError('invalid shard integer counts')
                parsed[field] = value.astype(np.int64)
            if (np.any(parsed['confusion'].sum((1, 2)) != len(expected)*voxel_count)
                    or np.any(parsed['moving'] > parsed['confusion'])):
                raise RuntimeError('shard count totals incomplete')
            for field in parsed: totals[name][field] += parsed[field]
    if len(seen) != len(keys) or len(set(seen)) != len(keys) or set(seen) != set(map(tuple, keys)):
        raise RuntimeError('dropped or duplicate shard windows')
    prefix = 'occ_gt' if modality == 'occ' else 'stc_gt'
    reports = {prefix+'/'+name.split('/', 1)[1]: metrics.report(counts, (1., 2., 3.)) for name, counts in totals.items()}
    return reports, {n: {k: v.tolist() for k, v in arrays.items()} for n, arrays in totals.items()}


def main(event=None, argv=None):
    event = event or threading.Event(); p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    for name in ('config', 'dataroot', 'dev-info', 'dev-cache', 'stc-root'): p.add_argument('--'+name, required=True)
    p.add_argument('--cpu-workers', type=int, default=2)
    a = p.parse_args(argv); root = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    if torch.cuda.device_count() != 1: raise RuntimeError('ONE visible physical GPU; both workers use that SAME GPU')
    if not 1 <= a.cpu_workers <= 4: raise ValueError('1--4 CPU workers per inference process')
    if a.resume and not a.source_dir:
        a.source_dir = str(Path(json.loads((out/'contract.json').read_text(encoding='utf-8'))['source']).parent)
    source = select_source(a.source_dir, root/'outputs/p0_f9_shared_history')
    if out == source.parent or source.parent in out.parents: raise ValueError('independent output outside source run required')
    sha = base.sha256(source); saved = torch.load(source, map_location='cpu', weights_only=False)
    c = validate_source(saved)
    if saved['updates'] != 10292: raise RuntimeError('fixed original epoch2/update10292 required')
    del saved
    _, records = base.load_cache(a.dev_cache); _, keys = select_jobs(records)
    for name in ('config', 'dev_info', 'dev_cache'):
        if base.sha256(getattr(a, name)) != c['data'][name]: raise RuntimeError('source data changed: '+name)
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, c.get('overrides')))
    params = vars(a).copy(); params.pop('resume'); params.update(source_dir=str(source.parent), out_dir=str(out))
    contract = dict(protocol=PARALLEL_PROTOCOL, source=str(source), source_sha256=sha, source_contract=c,
        args=params, keys=keys, windows=4369, scenes=150, known=dict(occ='dense_valid', stc='dense_valid'),
        physical_GPUs=1, inference_processes=2, visible_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
        partition='ordered_keys_rank_stride2_NO_padding', merge='SUM_integer_confusion_before_IoU',
        implementation={str(f.relative_to(root)): base.sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))})
    lock = out.parent/'.dense_parallel_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock), ExitStack() as stack:
        if a.resume:
            if json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract:
                raise RuntimeError('same unchanged single-GPU parallel evaluation required')
        else: out.mkdir(parents=True, exist_ok=False)
        base.write_json(out/'contract.json', contract)
        result = dict(status='running', contract=contract, reports={}, scientific_updates=0)
        def publish():
            base.write_json(out/'evaluation.json', result)
            (out/'summary.txt').write_text('ONE GPU / TWO inference processes; not two GPUs.\n'+summary(result), encoding='utf-8')
        publish(); children = []
        try:
            for rank in range(2):
                child_dir = out/f'shard_{rank:02d}'; child_contract = child_dir/'contract.json'
                flags = ['--source-dir', str(source.parent), '--out-dir', str(child_dir), '--device', 'cuda',
                    '--cpu-workers', str(a.cpu_workers), '--shard-index', str(rank), '--shard-count', '2']
                for name in ('config', 'dataroot', 'dev_info', 'dev_cache', 'stc_root'):
                    flags += ['--'+name.replace('_', '-'), str(getattr(a, name))]
                if child_contract.is_file(): flags += ['--resume']
                elif child_dir.exists(): raise RuntimeError('incomplete child setup; retain failed output and use NEW DENSE_ALL_DIR')
                handle = stack.enter_context((out/f'shard_{rank:02d}.log').open('a', encoding='utf-8'))
                env = os.environ.copy()  # IDENTICAL CUDA_VISIBLE_DEVICES for BOTH processes.
                child = subprocess.Popen([sys.executable, '-u', '-m', 'tools.dense_final_all.run', *flags],
                    cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                print(f'SAME_GPU worker={rank} pid={child.pid} windows={len(keys[rank::2])} log={out/f"shard_{rank:02d}.log"}', flush=True)
            signaled = False
            while any(child.poll() is None for child in children):
                failed = any(child.poll() not in (None, 0) for child in children)
                if (event.is_set() or failed) and not signaled:
                    for child in children:
                        if child.poll() is None: child.terminate()
                    signaled = True
                time.sleep(.5)
            codes = [child.returncode for child in children]
            if event.is_set(): result.update(status='stopped_workers_saved', returncodes=codes); return 130
            if codes != [0, 0]: raise RuntimeError('worker failed; see shard_00.log/shard_01.log; returncodes='+str(codes))
            for kind in ('occ', 'stc'):
                states = []
                for rank in range(2):
                    report = json.loads((out/f'shard_{rank:02d}/evaluation.json').read_text(encoding='utf-8'))
                    if report['status'] != 'complete' or report['contract']['source_contract'] != c:
                        raise RuntimeError('worker report incomplete or different original source contract')
                    states.append(json.loads((out/f'shard_{rank:02d}/{kind}_state.json').read_text(encoding='utf-8')))
                merged, counts = merge_states(states, keys, sha, math.prod(pcfg.grid.shape_hwd), kind)
                result['reports'].update(merged)
                base.write_json(out/f'{kind}_merged_counts.json', dict(keys=keys, counts=counts, source_sha256=sha))
            result['status'] = 'complete'; return 0
        except Exception as error:
            result.update(status='failed_original_unchanged', error=repr(error)); raise
        finally:
            for child in children:
                if child.poll() is None: child.terminate()
            for child in children: child.wait()
            if base.sha256(source) != sha: raise RuntimeError('source checkpoint changed')
            publish(); print(summary(result), flush=True)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
