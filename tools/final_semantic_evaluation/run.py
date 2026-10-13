"""New SemanticRefiner ONLY: GT/STC/6s/Waymo frozen final evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.waymo_i2world import WaymoI2WorldSource, fingerprint, file_sha256
from real_motion.waymo_i2world_10hz import WaymoI2World10HzSource
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.scene_ego_head.frozen_gt.comparison import find_head10
from tools.real_motion.static_evidence_selector_common import write_json
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .predictor import load_body, load_head10, Predictor, audit_gt_entrance
from .sources import NuscInputs, nusc_jobs
from .engine import evaluate


def parser():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    p.add_argument('--task', choices=('gt', 'stc', 'long6', 'waymo2', 'waymo10'), required=True)
    p.add_argument('--gt-checkpoint', required=True); p.add_argument('--head10-checkpoint')
    p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    p.add_argument('--dataroot'); p.add_argument('--dev-info'); p.add_argument('--dev-cache')
    p.add_argument('--stc-root'); p.add_argument('--geniedrive-info')
    p.add_argument('--long-alignment', choices=('geniedrive', 'legacy'), default='geniedrive')
    p.add_argument('--population', choices=('all', 'dev512', 'dev64'), default='all')
    p.add_argument('--waymo-root'); p.add_argument('--raw-free-label', type=int, choices=(15, 23), default=23)
    p.add_argument('--expected-waymo-scenes', type=int, default=202)
    p.add_argument('--max-windows', type=int, default=0); p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--frame-cache-mib', type=int, default=256); p.add_argument('--checkpoint-every', type=int, default=8)
    p.add_argument('--device', default='cuda')
    return p


def summary(result):
    lines = ['===== FIXED FINAL SEMANTIC REFINER =====', 'status='+result['status'],
        f"task={result['task']}; windows={result.get('windows', 0)}; GT epoch={result.get('source_epoch')}/update={result.get('source_update')}; no training",
        'No Surface/old mean/action thresholds/legacy static carry. Full grid, no metric visibility masks.',
        'Pred=original head10 + declared GT-derived navigation; GT future ego conditioned for other routes.',
        'route horizon IoU mIoU MovingMicro']
    for route, report in result.get('reports', {}).items():
        for h, row in report.get('per_horizon', report.get('horizons', {})).items():
            if 'i2world_mIoU' in row:
                lines.append(f"{route} {h} IoU={row['IoU']} standard_mIoU={row['standard_mIoU']} I2_mIoU={row['i2world_mIoU']}")
            else:
                vals = [row.get(k) for k in ('IoU', 'mIoU', 'MovingMicro')]
                lines.append(route+' '+str(h)+' '+' '.join('NA' if v is None else f'{v:.6f}' for v in vals))
        lines.append(route+' averages='+json.dumps({k: v for k, v in report.items() if k.startswith('average')}))
    for k in ('trajectory', 'rollout'):
        if k in result: lines.append(k+'='+json.dumps(result[k]))
    population = result.get('population', {})
    lines.append('population='+json.dumps({k: population[k] for k in
        ('population', 'actual_windows', 'scenes', 'parent_windows', 'requested', 'planner_trimmed',
         'alignment', 'diagnostic_prefix_only', 'paper_table_population_verified') if k in population}))
    if 'public_code_compatibility' in result:
        lines.append('GenieDrive public-code compatibility (NOT independently verified paper population)='+
                     json.dumps(result['public_code_compatibility']))
    lines.append('Source checkpoints/caches read only; quality evaluation time is NOT FPS. No automatic selection.')
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event(); a = parser().parse_args(argv)
    if not 1 <= a.cpu_workers <= 8 or not 0 <= a.frame_cache_mib <= 2048 or a.max_windows < 0 or a.checkpoint_every < 1:
        raise ValueError('invalid bounded execution budgets')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(1); torch.manual_seed(21)
    source_ck, out = Path(a.gt_checkpoint).resolve(), Path(a.out_dir).resolve()
    if out == source_ck.parent or source_ck.parent in out.parents: raise ValueError('output outside checkpoint source')
    if a.task == 'stc':
        if not a.stc_root: raise ValueError('--stc-root required')
        head_source = Path(a.head10_checkpoint or find_head10(source_ck.parents[2], source_ck)).resolve()
        if out == head_source.parent or head_source.parent in out.parents: raise ValueError('output outside head source')
    else: head_source = None
    lock = out.parent/'.final_semantic_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(source_ck, out/'frozen/GTepoch2.pt')
            if head_source: snapshot_checkpoint(head_source, out/'frozen/head10.pt')
        body_path = out/'frozen/GTepoch2.pt'; body_sha = file_sha256(body_path)
        if file_sha256(source_ck) != body_sha: raise RuntimeError('source GT SHA changed')
        saved, body = load_body(body_path, a.device); before = tensor_state_fingerprint(body.state_dict())
        head = load_head10(out/'frozen/head10.pt', body, body_sha, a.device) if head_source else None
        if head_source and file_sha256(head_source) != file_sha256(out/'frozen/head10.pt'): raise RuntimeError('source head SHA changed')
        pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
        if tuple(pcfg.grid.shape_hwd) != (200, 200, 16) or pcfg.free_label != 17:
            raise ValueError('full Occ3D grid/free label required')
        from tools.real_motion.geniedrive_eval_alignment import validate_grid
        validate_grid(pcfg)
        if float(pcfg.frame_dt_s) != .5: raise ValueError('unchanged trained nominal 2Hz model clock required')
        predictor = Predictor(body, pcfg, a.device, a.cpu_workers, head)
        try:
            waymo = a.task.startswith('waymo')
            if waymo:
                if not a.waymo_root or a.population != 'all': raise ValueError('Waymo root and all native population required')
                cls = WaymoI2World10HzSource if a.task == 'waymo10' else WaymoI2WorldSource
                inputs = cls.from_files(a.waymo_root, raw_free_label=a.raw_free_label, cache_mib=a.frame_cache_mib)
                if inputs.metadata['scenes'] != a.expected_waymo_scenes: raise RuntimeError('Waymo scene inventory differs')
                jobs = inputs.windows
                audit = dict(metadata=inputs.metadata, manifest_fingerprint=inputs.manifest_fingerprint,
                    model_clock='unchanged 0.5s slot clock; 10Hz index protocol NOT physical 1/2/3s',
                    parent_windows=len(jobs))
            else:
                if not all((a.dataroot, a.dev_info, a.dev_cache)): raise ValueError('nuScenes root/info/VAL cache required')
                for key in ('dev_info', 'dev_cache'):
                    if file_sha256(Path(getattr(a, key))) != saved['contract']['data'][key]: raise RuntimeError('GT source VAL metadata changed')
                inputs = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=a.frame_cache_mib)
                _, records = load_cache(a.dev_cache)
                jobs, audit = nusc_jobs(inputs, records, saved['contract']['eval_keys'], a.population, a.task,
                    genie_info=a.geniedrive_info, long_alignment=a.long_alignment)
                if set(w.scene_name for w in jobs) & set(k[0] for k in saved['contract']['train_keys']):
                    raise RuntimeError('TRAIN/VAL scenes overlap')
            parent_size = len(jobs)
            if a.max_windows: jobs = jobs[:a.max_windows]
            if not jobs: raise RuntimeError('empty population')
            if waymo:
                inventory = inputs.preflight(jobs)
                keys = [[inputs.frames[w.anchor].scene, inputs.frames[w.anchor].token] for w in jobs]
            else:
                inventory = fingerprint(inputs.preflight(jobs, 'stc' if a.task == 'stc' else 'occ'))
                keys = [[w.scene_name, w.t0_token] for w in jobs]
            audit.update(actual_windows=len(jobs), parent_windows=parent_size, diagnostic_prefix_only=bool(a.max_windows),
                selected_key_fingerprint=fingerprint(keys), inventory=inventory,
                scenes=len({k[0] for k in keys}))
            impl = list(Path(__file__).parent.glob('*.py'))
            root = Path(__file__).resolve().parents[2]
            impl += [root/p for p in ('tools/scene_ego_head/integration.py', 'tools/scene_ego_head/model.py',
                'tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/refine/model.py',
                'tools/shared_history_experiments/refine/history.py', 'tools/shared_history_experiments/refine/geometry.py',
                'tools/real_motion/benchmark_p0_f9_v18_runtime.py', 'tools/real_motion/causal_column_common.py',
                'real_motion/ego_navigation.py', 'real_motion/waymo_i2world.py', 'real_motion/waymo_i2world_10hz.py')]
            args = vars(a).copy(); args.pop('resume')
            contract = dict(protocol='final_GTepoch2_semantic_refiner_eval_v1', args=args, keys=keys,
                GT_sha256=body_sha, head10_sha256=file_sha256(out/'frozen/head10.pt') if head else None,
                population=audit, implementation={str(p.relative_to(root)): file_sha256(p) for p in impl},
                execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda, precision='BF16_on_CUDA_FP32_head'))
            cp = out/'contract.json'
            if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract: raise RuntimeError('resume contract/inputs/code changed')
            write_json(cp, contract); write_json(out/'population.json', audit)
            if not waymo:
                gate_path = out/'GT_entrance_audit.json'
                if gate_path.exists(): gate = json.loads(gate_path.read_text(encoding='utf-8'))
                else:
                    gate = audit_gt_entrance(predictor, inputs, records[0]); write_json(gate_path, gate)
                if not gate['pass_gate']: raise RuntimeError('new GT history adapter parity failed; no evaluation')
            state_path = out/'state.json'; state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
            print(f'FIXED_FINAL task={a.task} windows={len(jobs)}; only new SemanticRefiner', flush=True)
            try:
                result = evaluate(predictor, inputs, jobs, a.task, keys, contract, saved=state,
                    save=lambda s: write_json(state_path, s), stop_event=event, checkpoint_every=a.checkpoint_every)
            except InterruptedError:
                write_json(out/'status.json', dict(status='stopped_at_complete_window', task=a.task)); return 130
            result.update(population=audit, source_GT_sha256=body_sha, source_head10_sha256=contract['head10_sha256'],
                          source_epoch=saved['epoch'], source_update=saved['updates'])
            if tensor_state_fingerprint(body.state_dict()) != before: raise RuntimeError('frozen GT body changed')
            write_json(out/'evaluation.json', result); (out/'summary.txt').write_text(summary(result), encoding='utf-8')
            print(summary(result), flush=True); return 0
        finally: predictor.close()


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
