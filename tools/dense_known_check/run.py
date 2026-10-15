"""Frozen update6000: SAME remaining64, OCC lidar-known vs dense-valid."""
import argparse
import json
from pathlib import Path
import signal
import threading
import torch
from tools.joint_occ_experiments import run as base
from tools.dense_valid_ddp.run import PROTOCOL as SOURCE_PROTOCOL
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.final_semantic_evaluation.predictor import Predictor
from tools.shared_source_track.adapter import SharedTrackPredictor
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.final_semantic_evaluation.engine import evaluate
from tools.history_validity.data import policy_raw
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


class KnownInputs:
    def __init__(self, inputs, policy): self.inputs, self.policy = inputs, policy
    def history(self, job, kind='occ'):
        record, raw, times = self.inputs.history(job, 'occ')
        return record, policy_raw(raw, self.policy), times
    def poses(self, job): return self.inputs.poses(job)
    def targets(self, job, indices): return self.inputs.targets(job, indices)
    def moving(self, *args): return self.inputs.moving(*args)


def load_body(path, step, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    c = saved['contract']
    if (saved.get('protocol') != SOURCE_PROTOCOL or c['protocol'] != SOURCE_PROTOCOL
            or saved['updates'] != step or c['training_known'] != 'dense_valid'
            or len(c['eval_keys']) != 64 or len(set(map(tuple, c['eval_keys']))) != 64):
        raise RuntimeError('matched dense-valid DDP checkpoint/update/remaining64 required')
    spec = c['model']; shared, repair = spec['shared_history'], spec['repair']
    model = GTBody(base.config_from_mapping_v17(spec['motion']), feature_width=shared['width'],
        z_bins=shared['z_bins'], memory_width=repair['memory_width'], width=repair['width'], chunk_size=repair['chunk_size'])
    model.columns.epsilon = repair['epsilon']
    model.load_state_dict(saved['models']['gt'], strict=True)
    return c, model.to(device).eval().requires_grad_(False)


def summary(result):
    lines = ['===== FROZEN DENSE TRAIN / MATCHED OCC KNOWN CHECK =====',
        'status='+result['status'], f'update={result["step"]}; same remaining64; scientific_updates=0',
        'Same frozen model/source/KTA/GT ego/full-grid score. Only historical known policy changes.',
        'Encoder inputs AND registration-conditioned feature memory rebuilt for EACH policy.',
        'Fresh history/Strong inference, no training caches/writer roots, no STC or Pred run.',
        'policy       route                    mIoU       IoU MovingMicro']
    for policy, values in result['reports'].items():
        for route, m in values.items():
            v = m['average']
            lines += [f'{policy:12s} {route:22s} {v["mIoU"]:9.6f} {v["IoU"]:9.6f} {v["MovingMicro"]:11.6f}']
    if len(result['reports']) == 2:
        delta = {route: {key: result['reports']['dense_valid'][route]['average'][key]-
            result['reports']['lidar_known'][route]['average'][key] for key in ('mIoU', 'IoU', 'MovingMicro')}
            for route in result['reports']['dense_valid']}
        lines += ['dense_minus_lidar_known='+json.dumps(delta)]
    lines += ['Original checkpoint/training/evaluations unchanged. Diagnostic, NOT FPS or automatic promotion.']
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event(); p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--step', type=int, default=6000)
    p.add_argument('--out-dir', required=True); p.add_argument('--config', required=True)
    p.add_argument('--dataroot', required=True); p.add_argument('--dev-info', required=True); p.add_argument('--dev-cache', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-workers', type=int, default=4)
    a = p.parse_args(argv); root = Path(__file__).resolve().parents[2]
    if a.source_dir: parent = Path(a.source_dir).resolve()
    else:
        candidates = [r for r in (root/'outputs/p0_f9_shared_history').glob('dense_valid_shared_full2_ddp_*')
                      if (r/f'update_{a.step:06d}.pt').is_file()]
        if not candidates: raise RuntimeError(f'no update_{a.step:06d}.pt; set DENSE_DDP_DIR')
        parent = max(candidates, key=lambda r: (r/f'update_{a.step:06d}.pt').stat().st_mtime_ns).resolve()
    source = parent/f'update_{a.step:06d}.pt'; out = Path(a.out_dir).resolve()
    if out == parent or parent in out.parents: raise ValueError('independent output OUTSIDE training run required')
    if not 1 <= a.cpu_workers <= 8: raise ValueError('bounded workers required')
    out.mkdir(parents=True, exist_ok=False)
    snapshot_checkpoint(source, out/'frozen.pt')
    torch.set_num_threads(1)
    c, body = load_body(out/'frozen.pt', a.step, a.device)
    for name in ('config', 'dev_info', 'dev_cache'):
        if base.sha256(getattr(a, name)) != c['data'][name]: raise RuntimeError('source data differs: '+name)
    _, records = base.load_cache(a.dev_cache)
    chosen = base.align_records(records, c['eval_keys']); jobs = [window(r) for r in chosen]
    inputs = NuscInputs(a.dataroot, a.dev_info, cache_mib=256); inputs.preflight(jobs, 'occ')
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, c.get('overrides')))
    factory = SharedTrackPredictor if c['source_policy'] == 'shared_pinned' else Predictor
    predictor = factory(body, pcfg, a.device, workers=a.cpu_workers)
    before = tensor_state_fingerprint(body.state_dict())
    result = dict(status='running', step=a.step, source=str(source), reports={}, scientific_updates=0)
    print(f'KNOWN_CHECK frozen={source}; windows=64; policies=lidar_known,dense_valid; output={out}', flush=True)
    try:
        with torch.no_grad():
            for policy in ('lidar_known', 'dense_valid'):
                value = evaluate(predictor, KnownInputs(inputs, policy), jobs, 'gt', c['eval_keys'],
                    dict(population={'population': 'same_remaining64'}, source_sha256=base.sha256(out/'frozen.pt'), policy=policy),
                    save=lambda state, policy=policy: base.write_json(out/(policy+'_state.json'), state), stop_event=event)
                result['reports'][policy] = value['reports']
                base.write_json(out/'results.json', result)
        result['status'] = 'complete'
    except InterruptedError: result['status'] = 'interrupted_no_training_changed'
    finally:
        predictor.close()
        if before != tensor_state_fingerprint(body.state_dict()): raise RuntimeError('frozen model changed')
        base.write_json(out/'results.json', result)
        (out/'summary.txt').write_text(summary(result), encoding='utf-8')
        print(summary(result), flush=True)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    main(event)
