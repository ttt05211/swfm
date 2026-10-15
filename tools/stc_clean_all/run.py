"""Frozen cumulative clean1000; ALL VAL4369; OCC+GT and STC+GT only."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.final_semantic_evaluation.predictor import Predictor
from tools.final_semantic_evaluation.engine import evaluate
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.stc_clean_continue.run import PROTOCOL as SOURCE_PROTOCOL

PROTOCOL = 'frozen_clean1000_ALL_VAL4369_occ_stc_GTego_v1'


def select_source(directory, search_root):
    if directory: return Path(directory).resolve()
    candidates = []
    for root in Path(search_root).glob('stc_clean_continue1000_*'):
        path = root/'results.json'
        if path.is_file():
            report = json.loads(path.read_text(encoding='utf-8'))
            if (report.get('status') == 'complete' and report.get('cumulative_adaptation_updates') == 1000
                    and report.get('contract', {}).get('protocol') == SOURCE_PROTOCOL): candidates.append(root.resolve())
    if len(candidates) != 1:
        raise RuntimeError(f'need ONE completed clean1000, found {len(candidates)}; set CLEAN_ALL_SOURCE explicitly')
    return candidates[0]


def load_frozen(path, report, device):
    saved = torch.load(path, map_location='cpu', weights_only=False); c = saved['contract']
    if (saved['protocol'] != SOURCE_PROTOCOL or c != report['contract'] or report['status'] != 'complete'
            or saved['updates'] != 500 or saved['epoch'] != 1 or saved['batch'] != 0 or report['updates'] != 500
            or report['cumulative_adaptation_updates'] != 1000 or c['cumulative_budget'] != 1000
            or c['source_completed_updates'] != 500 or c['augmentation'] != 'clean'
            or not c['fully_unfrozen'] or c['training_known'] != 'dense_valid'
            or c['objective'] != dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.)
            or set(saved['models']) != {'gt'}):
        raise RuntimeError('only completed clean cumulative1000 with matched report allowed; no500/empirical/internal ego')
    spec = c['model']; shared, repair = spec['shared_history'], spec['repair']
    body = GTBody(config_from_mapping_v17(spec['motion']), feature_width=shared['width'], z_bins=shared['z_bins'],
        memory_width=repair['memory_width'], width=repair['width'], chunk_size=repair['chunk_size'])
    body.columns.epsilon = repair['epsilon']; body.load_state_dict(saved['models']['gt'], strict=True)
    return saved, body.to(device).eval().requires_grad_(False)


def select_jobs(records):
    keys = legacy.record_keys(records)
    if len(keys) != 4369 or len(set(keys)) != 4369 or len({k[0] for k in keys}) != 150:
        raise RuntimeError('complete unique VAL4369 / 150scenes required; no planner trimming or DEV subset')
    return [window(r) for r in records], [list(k) for k in keys]


class ModalInputs:
    """Keep the unchanged GT-ego engine; switch only historical input modality."""
    def __init__(self, inputs, kind):
        if kind not in ('occ', 'stc'): raise ValueError('OCC/STC history only')
        self.inputs, self.kind = inputs, kind
    def history(self, job, _kind='occ'): return self.inputs.history(job, self.kind)
    def poses(self, job): return self.inputs.poses(job)
    def targets(self, job, indices): return self.inputs.targets(job, indices)
    def moving(self, *args): return self.inputs.moving(*args)


def summary(result):
    lines = ['===== FROZEN CLEAN1000 / ALL VAL / OCC+GT AND STC+GT =====', 'status='+result['status'],
        f'windows={result["windows"]}; scenes={result["scenes"]}; cumulative_adaptation_updates=1000; scientific_updates=0',
        'GTepoch2 -> clean dense-valid full500 -> next500; this is the NEW body, not original GTepoch2.',
        'Four histories -> six futures; GT future ego conditioned; NO internal/external planner inference.',
        'OCC history original lidar-known; STC released full semantics/dense-valid, no GT visibility inputs.',
        'Standard FULL GRID; NO metric masks, alignment, coefficient tuning, training or automatic promotion.',
        'route                      horizon       IoU      mIoU MovingMicro']
    for name, metrics in result.get('reports', {}).items():
        for horizon, m in list(metrics['per_horizon'].items())+[('Avg', metrics['average'])]:
            vals = ['NA' if m.get(k) is None else f'{m[k]:10.6f}' for k in ('IoU', 'mIoU', 'MovingMicro')]
            lines += [f'{name:26s} {horizon:>7s} '+' '.join(vals)]
    lines += ['Source checkpoint/Adam/RNG/caches READ ONLY. ALL4369, no planner coverage trimming.',
              'VAL repeatedly inspected during development; NOT an independent test. Quality evaluation, NOT FPS.']
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event(); p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--frame-cache-mib', type=int, default=256); p.add_argument('--checkpoint-every', type=int, default=8)
    a = p.parse_args(argv)
    if not 1 <= a.cpu_workers <= 8 or not 0 <= a.frame_cache_mib <= 2048 or a.checkpoint_every < 1:
        raise ValueError('invalid bounded evaluation budgets')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(1); torch.manual_seed(21)
    repo = Path(__file__).resolve().parents[2]
    parent = select_source(a.source_dir, repo/'outputs/p0_f9_shared_history'); out = Path(a.out_dir).resolve()
    if out == parent or parent in out.parents: raise ValueError('independent evaluation output OUTSIDE source run required')
    source = parent/'train/last.pt'; source_sha = legacy.sha256(source)
    report_path = parent/'results.json'; report = json.loads(report_path.read_text(encoding='utf-8'))
    lock = out.parent/'.clean_all_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(source, out/'frozen/clean1000.pt')
        frozen = out/'frozen/clean1000.pt'
        if legacy.sha256(frozen) != source_sha: raise RuntimeError('source checkpoint changed')
        saved, body = load_frozen(frozen, report, a.device); before = tensor_state_fingerprint(body.state_dict())
        prior500 = Path(saved['contract']['source500']['source_checkpoint']).parent.parent
        ref_path = prior500/'reference.json'; provenance = json.loads(ref_path.read_text(encoding='utf-8'))
        workflows = [Path(x) for x in provenance['reference'] if Path(x).name == 'workflow.json']
        if len(workflows) != 1: raise RuntimeError('original workflow provenance missing')
        if legacy.sha256(workflows[0]) != provenance['reference'][str(workflows[0])]: raise RuntimeError('source workflow changed')
        fields = json.loads(workflows[0].read_text(encoding='utf-8'))['args']
        for name in ('dev_cache', 'dev_info'):
            if legacy.sha256(fields[name]) != saved['contract']['data'][name]: raise RuntimeError('VAL metadata changed')
        _, records = legacy.load_cache(fields['dev_cache']); jobs, keys = select_jobs(records)
        if {k[0] for k in keys} & {k[0] for k in saved['contract']['train_keys']}: raise RuntimeError('TRAIN/VAL scenes overlap')
        inputs = NuscInputs(fields['dataroot'], fields['dev_info'], stc_root=fields['stc_root'], cache_mib=a.frame_cache_mib)
        inventory = {kind: legacy.stable_json_fingerprint(inputs.preflight(jobs, kind)) for kind in ('occ', 'stc')}
        files = list(Path(__file__).parent.glob('*.py'))
        for sub in ('tools/final_semantic_evaluation', 'tools/shared_history_experiments/refine'):
            files += list((repo/sub).glob('*.py'))
        files += [repo/x for x in ('tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
            'tools/shared_history_experiments/full_gt/model.py', 'tools/scene_ego_head/diagnostics/engine.py',
            'tools/scene_ego_head/integration.py')]
        args = vars(a).copy(); args.pop('resume')
        args.update(source_dir=str(parent), out_dir=str(out))
        contract = dict(protocol=PROTOCOL, args=args, source_sha256=source_sha, source_report_sha256=legacy.sha256(report_path),
            workflow_sha256=legacy.sha256(workflows[0]), runtime_config_sha256=legacy.sha256(fields['config']),
            keys=keys, inventory=inventory,
            implementation={str(f.relative_to(repo)): legacy.sha256(f) for f in files},
            execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
            known=dict(occ='original_lidar_known', stc='dense_valid'), metric_mask='none', planner_trimmed=False)
        cp = out/'contract.json'
        if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract: raise RuntimeError('same unchanged eval contract required')
        legacy.write_json(cp, contract)
        pcfg = legacy.make_prepare_config(legacy.load_runtime_config(fields['config'], fields.get('override')))
        predictor = Predictor(body, pcfg, a.device, workers=a.cpu_workers)
        result = dict(status='running', windows=len(jobs), scenes=len({k[0] for k in keys}), reports={},
            contract=contract, scientific_updates=0, cumulative_adaptation_updates=1000)
        try:
            for kind, prefix in (('occ', 'occ_gt'), ('stc', 'stc_gt')):
                state_path = out/(kind+'_state.json')
                state = json.loads(state_path.read_text(encoding='utf-8')) if a.resume and state_path.is_file() else None
                print(f'CLEAN1000_ALL modality={prefix} windows={len(jobs)}; frozen/no training/no ego head.', flush=True)
                value = evaluate(predictor, ModalInputs(inputs, kind), jobs, 'gt', keys,
                    dict(base=contract, history_kind=kind), saved=state,
                    save=lambda s, path=state_path: legacy.write_json(path, s), stop_event=event,
                    checkpoint_every=a.checkpoint_every)
                result['reports'].update({prefix+'/'+name.split('/', 1)[1]: m for name, m in value['reports'].items()})
                legacy.write_json(out/'evaluation.json', result)
            result['status'] = 'complete'
            return 0
        except InterruptedError:
            result['status'] = 'stopped_at_complete_window'; return 130
        except Exception as error:
            result.update(status='failed_checkpoint_unchanged', error=repr(error)); raise
        finally:
            predictor.close()
            if tensor_state_fingerprint(body.state_dict()) != before or legacy.sha256(source) != source_sha:
                raise RuntimeError('read-only body/source checkpoint changed')
            legacy.write_json(out/'evaluation.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8'); print(summary(result), flush=True)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
