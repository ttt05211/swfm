"""Evaluate the paper's fixed forward AdaLN epoch2 on long nuScenes and Waymo."""
import argparse
import json
from pathlib import Path
import signal
import threading

import numpy as np
import torch

from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.waymo_i2world import WaymoI2WorldSource, file_sha256, fingerprint
from real_motion.waymo_i2world_10hz import WaymoI2World10HzSource
from tools.causal_forward_all.common import DenseInputs, Epoch2Predictor, load_frozen
from tools.final_semantic_evaluation.engine import evaluate
from tools.final_semantic_evaluation.sources import nusc_jobs
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.geniedrive_eval_alignment import validate_grid
from tools.real_motion.static_evidence_selector_common import write_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = '7760c6f4f92277aa248161cd3e90f87ac2354329babceb6e5ed3b872adde33a0'
NUSCENES_ROOT = Path('/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes')
WAYMO_SOURCES = {'waymo2': (WaymoI2WorldSource, 7998), 'waymo10': (WaymoI2World10HzSource, 39987)}


class TransferPredictor(Epoch2Predictor):
    def prepare(self, record, raw, *, times=None):
        # Waymo reuses the trained slot clock, also for repeated boundary frames.
        if times is None:
            times = np.arange(-3, 1, dtype=np.float64) * self.provider.pcfg.frame_dt_s
        return super().prepare(record, raw, times=times)


class LongInputs(DenseInputs):
    def moving(self, job, indices, horizons, grid, workers):
        return [None] * len(indices)


def select_population(args, source):
    if args.task == 'long6':
        info = args.dev_info or args.dataroot / 'nuscenes_infos_val_temporal_v3_scene.pkl'
        for name, path in (('dev_info', info), ('dev_cache', args.dev_cache)):
            if file_sha256(path) != source['data'][name]:
                raise ValueError(f'{name} differs from the model evaluation contract')
        inputs = LongInputs(args.dataroot, info)
        _, records = load_cache(args.dev_cache)
        jobs, population = nusc_jobs(inputs, records, source['eval_keys'], 'all', 'long6',
                                     genie_info=args.geniedrive_info)
        if {w.scene_name for w in jobs} & {k[0] for k in source['train_keys']}:
            raise ValueError('nuScenes TRAIN and long evaluation scenes overlap')
        jobs = jobs[:args.max_windows or None]
        keys = [[w.scene_name, w.t0_token] for w in jobs]
        inventory = fingerprint(inputs.preflight(jobs))
    else:
        source_cls, expected_windows = WAYMO_SOURCES[args.task]
        inputs = source_cls.from_files(args.waymo_root, raw_free_label=args.raw_free_label, cache_mib=256)
        if (inputs.metadata['scenes'], len(inputs.windows)) != (202, expected_windows):
            raise ValueError('Waymo validation population differs from the archived protocol')
        jobs = inputs.windows[:args.max_windows or None]
        keys = [[inputs.frames[w.anchor].scene, inputs.frames[w.anchor].token] for w in jobs]
        inventory = inputs.preflight(jobs)
        population = dict(metadata=inputs.metadata, manifest_fingerprint=inputs.manifest_fingerprint,
                          model_clock_s=.5, full_windows=expected_windows)
    return inputs, jobs, keys, dict(population, inventory=inventory, actual_windows=len(jobs),
                                    diagnostic_prefix=bool(args.max_windows))


def summary(result):
    lines = ['===== FORWARD AdaLN EPOCH2 / FROZEN TRANSFER =====',
             f"status={result['status']}; task={result['task']}; windows={result['windows']}",
             f"source_update=10292; SHA256={result['source_sha256']}",
             f"diagnostic_prefix={result['population']['diagnostic_prefix']}; GT future ego; full-grid scores",
             'route                     horizon       mIoU        IoU']
    for route, report in result['reports'].items():
        rows = list(report.get('per_horizon', report.get('horizons')).items())
        rows += [(k, v) for k, v in report.items() if k.startswith('average')]
        for horizon, scores in rows:
            miou = scores.get('mIoU', scores.get('standard_mIoU'))
            values = ['NA' if value is None else f'{value:.6f}' for value in (miou, scores['IoU'])]
            lines.append(f'{route:25} {horizon:>15} {values[0]:>10} {values[1]:>10}')
    return '\n'.join(lines) + '\n'


def main(argv=None, stop=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('long6', *WAYMO_SOURCES))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/p0_f9_shared_history'
                        / 'adaln_forward_full_2ep_20261026_215351/epoch_0002.pt')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/real_motion_occfm.yaml')
    parser.add_argument('--dataroot', type=Path, default=NUSCENES_ROOT)
    parser.add_argument('--dev-info', type=Path)
    parser.add_argument('--dev-cache', type=Path, default=ROOT / 'data/p0_f9_v18_se2_val_all_4369.pt')
    parser.add_argument('--geniedrive-info', type=Path, default=ROOT / 'data/geniedrive/world-nuscenes_infos_val.pkl')
    parser.add_argument('--waymo-root', type=Path, default=ROOT / 'data/waymo')
    parser.add_argument('--raw-free-label', type=int, choices=(15, 23), default=23)
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--max-windows', type=int, default=0, help='run a diagnostic prefix if nonzero')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_workers <= 8 or args.max_windows < 0:
        parser.error('invalid CPU worker count or window limit')
    if torch.cuda.device_count() != 1:
        parser.error('set CUDA_VISIBLE_DEVICES to one GPU for this evaluation')
    torch.set_num_threads(1)
    checkpoint, out = args.checkpoint.resolve(), args.out_dir.resolve()
    if out == checkpoint.parent or checkpoint.parent in out.parents:
        parser.error('output directory must be separate from the checkpoint run')
    if file_sha256(checkpoint) != SOURCE_SHA256:
        raise ValueError('checkpoint differs from the fixed paper epoch2 weights')
    source, body = load_frozen(checkpoint, 'cuda:0')
    config_sha = file_sha256(args.config)
    if config_sha != source['data']['config']:
        raise ValueError('runtime config differs from the fixed model')
    pcfg = make_prepare_config(load_runtime_config(args.config, source.get('overrides')))
    validate_grid(pcfg)
    inputs, jobs, keys, population = select_population(args, source)
    dependency_paths = ('tools/causal_forward_transfer.py', 'tools/causal_forward_train/data.py',
                        'tools/final_semantic_evaluation/engine.py', 'tools/final_semantic_evaluation/metrics.py',
                        'tools/final_semantic_evaluation/sources.py', 'tools/real_motion/joint_long_rollout_common.py',
                        'real_motion/waymo_i2world.py', 'real_motion/waymo_i2world_10hz.py')
    contract = dict(protocol='forward_adaln_epoch2_transfer_v1', task=args.task, source_sha256=SOURCE_SHA256,
                    config_sha256=config_sha, keys=keys, population=population, cpu_workers=args.cpu_workers,
                    implementation={name: file_sha256(ROOT / name) for name in dependency_paths},
                    torch_version=str(torch.__version__))
    out.mkdir(parents=True, exist_ok=args.resume)
    with evaluation_lock(out):
        contract_path, state_path = out / 'contract.json', out / 'state.json'
        if args.resume and json.loads(contract_path.read_text(encoding='utf-8')) != contract:
            raise ValueError('resume inputs, population, or evaluator changed')
        saved = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        write_json(contract_path, contract)
        predictor = TransferPredictor(body, pcfg, 'cuda:0', workers=args.cpu_workers)
        try:
            print(f'FORWARD_TRANSFER {args.task} {len(saved["keys"]) if saved else 0}/{len(jobs)}', flush=True)
            result = evaluate(predictor, inputs, jobs, args.task, keys, contract, saved=saved, stop_event=stop,
                              save=lambda state: write_json(state_path, state), checkpoint_every=128)
        except InterruptedError:
            return 130
        finally:
            predictor.close()
        if args.task == 'long6':
            result['rollout']['history2'] = 'last four predictions with projected initial dense-valid support'
        result.update(population=population, source_sha256=SOURCE_SHA256, source_update=10292,
                      moving_evaluated=False)
        write_json(out / 'evaluation.json', result)
        (out / 'summary.txt').write_text(summary(result), encoding='utf-8')
        print(summary(result), flush=True)
        return 0


if __name__ == '__main__':
    stop_event = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop_event.set())
    raise SystemExit(main(stop=stop_event))
