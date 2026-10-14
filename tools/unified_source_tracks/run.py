"""Independent frozen GTepoch2 track/KTA DEV64 screen; no training or old cache writes."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.waymo_i2world import file_sha256, fingerprint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.static_evidence_selector_common import write_json
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.final_semantic_evaluation.predictor import load_body, Predictor, audit_gt_entrance
from tools.final_semantic_evaluation.sources import NuscInputs, nusc_jobs
from tools.real_motion.geniedrive_eval_alignment import validate_grid
from .engine import evaluate
from .geometry import ARMS

PROTOCOL = 'frozen_GTepoch2_unified_source_tracks_multiframe_KTA_dev64_v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for k in ('gt-checkpoint', 'out-dir', 'dataroot', 'dev-info', 'dev-cache'): p.add_argument('--'+k, required=True)
    p.add_argument('--stc-root'); p.add_argument('--inputs', choices=('occ', 'stc', 'both'), default='both')
    p.add_argument('--resume', action='store_true'); p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-workers', type=int, default=4); p.add_argument('--frame-cache-mib', type=int, default=256)
    p.add_argument('--checkpoint-every', type=int, default=8)
    return p


def summary(result):
    lines = ['===== FROZEN UNIFIED SOURCE TRACK / MULTI-FRAME KTA =====',
        f'status={result["status"]}; windows={result["windows"]}; no training or cache rebuild',
        'GTepoch2 frozen; four real histories -> six futures; GT future ego conditioned.',
        'baseline / unified_two / unified_fit; STWM and feature transport use the SAME registrations.',
        'Existing association ambiguity/gap safeguards and ICP retained; NO new shape match cost.',
        'Actual historical seconds for velocity; nominal 0.5s association/model future clock unchanged.',
        'KTA slope only, observed t0 centroid retained. Matched but ICP-rejected history is NOT silently re-enabled.',
        'Strong/CLEAR backgrounds FIXED across arms (controlled screen, not a fully unified Strong variant).',
        'kind arm route mIoU IoU MovingMicro | dMiOU dIoU dMoving_vs_baseline']
    for name, report in result['reports'].items():
        a = report['average']; vals = [a[k] for k in ('mIoU', 'IoU', 'MovingMicro')]
        kind, _, route = name.split('/'); baseline = result['reports'][f'{kind}/baseline/{route}']['average']
        deltas = [a[k]-baseline[k] if a[k] is not None and baseline[k] is not None else None
                  for k in ('mIoU', 'IoU', 'MovingMicro')]
        lines.append(name+' '+' '.join('NA' if v is None else f'{v:.6f}' for v in vals)+
            ' | '+' '.join('NA' if v is None else f'{v:+.6f}' for v in deltas))
    lines.append('===== KTA DISPLACEMENT ERROR / SAME EVALUATION LABELS =====')
    for kind, arms in result['kta_errors'].items():
        for arm, groups in arms.items():
            lines.append(kind+'/'+arm+' @1/2/3s '+json.dumps({g: groups[g] for g in ('all', 'three_plus')}, ensure_ascii=False))
    lines.append('track_audits='+json.dumps(result['track_audits']))
    lines += ['seconds='+str(result['seconds']), 'Quality wall time NOT FPS; no gate-based selection, full training, threshold tuning or promotion.']
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event(); a = parser().parse_args(argv)
    if not 1 <= a.cpu_workers <= 8 or not 0 <= a.frame_cache_mib <= 2048 or a.checkpoint_every < 1:
        raise ValueError('invalid bounded execution budgets')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    kinds = ('occ', 'stc') if a.inputs == 'both' else (a.inputs,)
    if 'stc' in kinds and not a.stc_root: raise ValueError('STC root required, no missing-window fallback')
    torch.set_num_threads(1); torch.manual_seed(21)
    source, out = Path(a.gt_checkpoint).resolve(), Path(a.out_dir).resolve()
    if out == source.parent or source.parent in out.parents: raise ValueError('output must be outside source checkpoint directory')
    root = Path(__file__).resolve().parents[2]
    lock = out.parent/'.unified_track_locks'/hashlib.sha256(str(out).encode()).hexdigest()
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(source, out/'frozen/GTepoch2.pt')
        snapshot = out/'frozen/GTepoch2.pt'; digest = file_sha256(snapshot)
        if file_sha256(source) != digest: raise RuntimeError('source checkpoint SHA changed')
        saved, body = load_body(snapshot, a.device); before = tensor_state_fingerprint(body.state_dict())
        for field in ('dev_info', 'dev_cache'):
            if file_sha256(Path(getattr(a, field))) != saved['contract']['data'][field]:
                raise RuntimeError('source validation metadata changed: '+field)
        pcfg = make_prepare_config(load_runtime_config(a.config, a.override)); validate_grid(pcfg)
        if pcfg.free_label != 17 or float(pcfg.frame_dt_s) != .5: raise ValueError('original free17/2Hz protocol required')
        inputs = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=a.frame_cache_mib)
        _, records = load_cache(a.dev_cache)
        jobs, population = nusc_jobs(inputs, records, saved['contract']['eval_keys'], 'dev64', 'gt')
        if len(jobs) != 64: raise RuntimeError('exact original DEV64 required')
        if set(w.scene_name for w in jobs) & set(k[0] for k in saved['contract']['train_keys']):
            raise RuntimeError('TRAIN/VAL scenes overlap')
        # Freeze population BEFORE preflight; missing STC fails, never trims/replaces.
        inventory = {kind: fingerprint(inputs.preflight(jobs, kind)) for kind in kinds}
        keys = [[w.scene_name, w.t0_token] for w in jobs]
        population.update(actual_windows=64, scenes=len({w.scene_name for w in jobs}), inventory=inventory)
        files = list(Path(__file__).parent.glob('*.py'))
        files += [root/p for p in ('tools/final_semantic_evaluation/predictor.py', 'tools/final_semantic_evaluation/sources.py',
            'tools/final_semantic_evaluation/metrics.py', 'real_motion/strong_w2det.py', 'real_motion/source_evidence_audit.py',
            'real_motion/motion_transport.py', 'real_motion/motion_transport_v2.py', 'real_motion/ego_navigation.py',
            'tools/real_motion/causal_column_common.py', 'tools/real_motion/benchmark_p0_f9_v18_runtime.py',
            'tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/refine/history.py',
            'tools/shared_history_experiments/refine/model.py', 'tools/shared_history_experiments/refine/geometry.py',
            'tools/shared_history_experiments/full_gt/model.py', 'tools/scene_ego_head/integration.py',
            'tools/scene_ego_head/diagnostics/engine.py', 'tools/scene_ego_head/frozen_gt/engine.py')]
        args = vars(a).copy(); args.pop('resume')
        contract = dict(protocol=PROTOCOL, args=args, keys=keys, population=population, GT_sha256=digest,
            arms=list(ARMS), implementation={str(p.relative_to(root)): file_sha256(p) for p in files},
            cache_policy='original source/cache READ ONLY; new tracks/registrations/read coordinates RAM only',
            strong_background='fixed baseline control', future_GT='ego conditioned; labels/annotations after all forecasts',
            execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, deterministic=True, precision='original BF16 on CUDA'))
        path = out/'contract.json'
        if a.resume and json.loads(path.read_text(encoding='utf-8')) != contract: raise RuntimeError('resume contract changed')
        write_json(path, contract); write_json(out/'population.json', population)
        predictor = Predictor(body, pcfg, a.device, a.cpu_workers)
        try:
            gate_path = out/'baseline_entrance.json'
            if gate_path.exists(): gate = json.loads(gate_path.read_text(encoding='utf-8'))
            else:
                by = {(str(r['scene_name']), str(r['t0_token'])): r for r in records}
                gate = audit_gt_entrance(predictor, inputs, by[tuple(keys[0])]); write_json(gate_path, gate)
            if not gate['pass_gate']: raise RuntimeError('original GT baseline entrance parity failed')
            state_path = out/'state.json'
            state = json.loads(state_path.read_text(encoding='utf-8')) if a.resume and state_path.exists() else None
            print(f'UNIFIED_TRACK_SCREEN kinds={kinds} windows=64; frozen epoch2; no training/cache writes', flush=True)
            try:
                result = evaluate(predictor, inputs, jobs, keys, contract, kinds=kinds, saved=state,
                    save=lambda v: write_json(state_path, v), stop_event=event, checkpoint_every=a.checkpoint_every)
            except InterruptedError:
                write_json(out/'status.json', dict(status='stopped_at_complete_window')); return 130
            finally:
                if tensor_state_fingerprint(body.state_dict()) != before or file_sha256(source) != digest:
                    raise RuntimeError('frozen source/body changed')
            result.update(protocol=PROTOCOL, population=population, checkpoint_sha256=digest,
                source_epoch=saved['epoch'], source_update=saved['updates'])
            write_json(out/'evaluation.json', result); (out/'summary.txt').write_text(summary(result), encoding='utf-8')
            print(summary(result), flush=True); return 0
        finally: predictor.close()


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
