"""Start all eight evaluation processes, four on each of two GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

GPU_TASKS = (
    ('a_static', 'c_transport', 'm1_ego_memory', 's0_pool'),
    ('b_kta', 'd_kta_refiner', 'm0_no_memory', 'pred_ego'),
)
TABLE_ORDER = ('a_static', 'b_kta', 'c_transport', 'd_kta_refiner',
               'm0_no_memory', 'm1_ego_memory', 's0_pool', 'pred_ego')


def launch_all(args, gpus, out, checkpoints, stop):
    commands = []
    for gpu, arms in zip(gpus, GPU_TASKS):
        for arm in arms:
            command = [sys.executable, '-u', '-m', 'tools.paper_ablation_eval.run',
                       '--arm', arm, '--population', str(out/'population.json'),
                       '--config', str(args.config), '--out-dir', str(out/arm),
                       '--cpu-workers', str(args.cpu_workers)]
            if arm in checkpoints:
                command.extend(('--checkpoint', str(checkpoints[arm])))
            commands.append((arm, gpu, command))

    children = {}
    try:
        # Launch the entire wave before waiting on any child.
        for arm, gpu, command in commands:
            if stop.is_set():
                break
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            with (out/f'{arm}.log').open('a', encoding='utf-8') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            children[arm] = child
            print(f'ABLATION_START {arm} GPU={gpu} PID={child.pid}', flush=True)
        if len(children) == len(TABLE_ORDER):
            print('ABLATION_WAVE_STARTED 8 processes / 4 per GPU', flush=True)

        pending = dict(children)
        codes = {}
        while pending and not stop.is_set():
            for arm, child in list(pending.items()):
                code = child.poll()
                if code is not None:
                    codes[arm] = code
                    del pending[arm]
                    print(f'ABLATION_EXIT {arm} code={code}', flush=True)
            if pending:
                stop.wait(1)
        return codes
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for child in children.values():
            child.wait()


def summarize(out, codes):
    rows = ['arm                 windows  OCC mIoU / IoU    STC mIoU / IoU']
    for arm in TABLE_ORDER:
        if codes.get(arm) != 0:
            rows.append(f'{arm:19s} pending or failed; see {arm}.log')
            continue
        value = json.loads((out/arm/'evaluation.json').read_text(encoding='utf-8'))
        if value['status'] != 'complete' or value['contract']['arm'] != arm:
            raise RuntimeError(f'{arm}: worker exited without a complete result')
        occ = value['reports']['occ']['average']
        stc = value['reports']['stc']['average']
        rows.append(f'{arm:19s} {value["contract"]["windows"]:7d}  '
                    f'{occ["mIoU"]:7.3f} / {occ["IoU"]:6.3f}  {stc["mIoU"]:7.3f} / {stc["IoU"]:6.3f}')
    summary = '\n'.join(rows)+'\n'
    (out/'summary.txt').write_text(summary, encoding='utf-8')
    print(summary, end='', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-dir', 'train-dir', 'epoch2', 'config', 'dataroot', 'dev-info',
                 'dev-cache', 'stc-root', 'plan-cache'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--cpu-workers', type=int, default=2)
    args = parser.parse_args(argv)
    gpus = [gpu.strip() for gpu in args.gpus.split(',')]
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(gpus):
        parser.error('Specify exactly two distinct GPU IDs')
    if not 1 <= args.cpu_workers <= 4:
        parser.error('Use 1–4 CPU geometry workers per evaluation process')
    out = args.out_dir.resolve()
    train_dir = args.train_dir.resolve()
    epoch2 = args.epoch2.resolve()
    if any(parent in (out, *out.parents) for parent in (train_dir, epoch2.parent)):
        parser.error('Evaluation output must be separate from the training runs')
    checkpoints = {arm: train_dir/arm/'epoch_0001.pt' for arm in TABLE_ORDER[2:-1]}
    checkpoints['pred_ego'] = epoch2
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
    import torch
    from real_motion.native_column_cpu import prepare_native
    from tools.joint_occ_experiments.run import write_json
    from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
    from .population import prepare

    if torch.cuda.device_count() != 2:
        raise RuntimeError('The coordinator requires exactly the two selected GPUs to be visible')
    torch.set_num_threads(1)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    out.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(out):
        print(f'ABLATION_PREPARE shared VAL4369 metadata, output={out}', flush=True)
        population = prepare(args, epoch2)
        path = out/'population.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != population:
                raise RuntimeError('Evaluation population, poses or inputs changed; use a separate output directory')
        else:
            write_json(path, population)
        print(f'ABLATION_POPULATION GT=4369 Pred={len(population["predicted"])}/4369', flush=True)
        if stop.is_set():
            return 130
        prepare_native()  # Publish the shared native library before the workers start.
        codes = launch_all(args, gpus, out, checkpoints, stop)
        summarize(out, codes)
        status = 'complete' if all(codes.get(arm) == 0 for arm in TABLE_ORDER) else 'failed'
        if stop.is_set():
            status = 'interrupted'
        write_json(out/'suite_state.json', dict(status=status, exit_codes=codes))
        print(f'ABLATION_SUITE {status}', flush=True)
        return {'complete': 0, 'interrupted': 130, 'failed': 1}[status]


if __name__ == '__main__':
    raise SystemExit(main())
