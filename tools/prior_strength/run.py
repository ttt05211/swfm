"""Frozen GT epoch2: one fixed DEV64 final-prior reweighting diagnostic."""
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
from tools.final_semantic_evaluation.predictor import Predictor, load_body, audit_gt_entrance
from tools.final_semantic_evaluation.sources import NuscInputs, nusc_jobs
from tools.final_semantic_diagnostics.input_audit import ArtifactAudit
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.geniedrive_eval_alignment import validate_grid
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.static_evidence_selector_common import write_json
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .engine import evaluate


def parser():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for k in ('gt-checkpoint', 'dataroot', 'dev-info', 'dev-cache', 'stc-root', 'out-dir'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--stc-archive'); p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-workers', type=int, default=3); p.add_argument('--resume', action='store_true')
    return p


def summary(result):
    fmt = lambda x: 'NA' if x is None else f'{x:.6f}'
    lines = ['===== FROZEN FINAL PRIOR STRENGTH / DEV64 =====',
        f"status={result['status']}; windows={result['windows']}; scientific_updates=0",
        'GT epoch2 body; GT future ego; no head/navigation/training/GT pose alignment/masks.',
        'Same forward per modality; ONLY final log-prior coefficient 1 -> 0.5; no epsilon/input/residual changes.',
        'route                         mIoU       IoU MovingMicro']
    for name, report in result['reports'].items():
        v = report['average']
        lines.append(f'{name:28s} '+ ' '.join(fmt(v[k]) for k in ('mIoU', 'IoU', 'MovingMicro')))
    lines.append('half_minus_original='+json.dumps(result['half_minus_original']))
    lines.append('===== SEMANTIC EDITS / SUM 1,2,3s =====')
    for name, comparisons in result['edits'].items():
        for comparison, values in comparisons.items():
            v = values['totals']['all']
            lines.append(f"{name}/{comparison}: corrected={v['corrected']} damaged={v['damaged']} "
                         f"wrong_to_other_wrong={v['wrong_to_other_wrong']} net={v['net_corrected']}")
    lines += ['Details include wrong-free / false-occupancy / wrong-class and per-horizon scores.',
              'DEV diagnosis, NOT an independent test or new formal score; no coefficient sweep/automatic promotion.',
              'Frozen source weights/Adam/caches unchanged. Quality diagnostic wall time is NOT FPS.',
              'seconds='+fmt(result['seconds'])]
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event(); a = parser().parse_args(argv)
    if not 1 <= a.cpu_workers <= 8: raise ValueError('bounded CPU workers required')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(1); torch.manual_seed(21)
    ck, out = Path(a.gt_checkpoint).resolve(), Path(a.out_dir).resolve()
    if ck.parent == out or ck.parent in out.parents: raise ValueError('new output outside source checkpoint directory required')
    lock = out.parent/'.prior_strength_locks'/hashlib.sha256(str(out).encode()).hexdigest()
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(ck, out/'frozen/GTepoch2.pt')
        bp = out/'frozen/GTepoch2.pt'; digest = file_sha256(bp)
        if file_sha256(ck) != digest: raise RuntimeError('frozen source SHA changed')
        saved, body = load_body(bp, a.device)
        before = tensor_state_fingerprint(body.state_dict())
        pcfg = make_prepare_config(load_runtime_config(a.config, a.override)); validate_grid(pcfg)
        if pcfg.frame_dt_s != .5 or pcfg.free_label != 17: raise ValueError('original 2Hz/free17 protocol required')
        for name in ('dev_cache', 'dev_info'):
            if file_sha256(Path(getattr(a, name))) != saved['contract']['data'][name]: raise RuntimeError('GT source metadata changed')
        source = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
        _, records = load_cache(a.dev_cache)
        jobs, population = nusc_jobs(source, records, saved['contract']['eval_keys'], 'dev64', 'stc')
        keys = [[w.scene_name, w.t0_token] for w in jobs]
        if len(saved['contract']['eval_keys']) != 512 or len(jobs) != 64 or len({tuple(k) for k in keys}) != 64:
            raise RuntimeError('exact original DEV512 parent / unique DEV64 required; no missing-STC replacement')
        if {w.scene_name for w in jobs} & {k[0] for k in saved['contract']['train_keys']}:
            raise RuntimeError('TRAIN/DEV scene overlap')
        inventories = [source.preflight(jobs, kind) for kind in ('occ', 'stc')]
        artifact, runner = ArtifactAudit(a.stc_root, a.stc_archive), None
        try:
            root = Path(__file__).resolve().parents[2]
            files = set(Path(__file__).parent.glob('*.py'))
            for directory in ('tools/final_semantic_evaluation', 'tools/final_semantic_diagnostics',
                              'tools/shared_history_experiments/refine', 'tools/shared_history_experiments/full_gt'):
                files.update((root/directory).glob('*.py'))
            files.update(root/p for p in ('tools/shared_history_experiments/model.py',
                'tools/scene_ego_head/diagnostics/engine.py', 'real_motion/nuscenes_adapter.py',
                'real_motion/stc_camera_protocol.py'))
            args = vars(a).copy(); args.pop('resume')
            contract = dict(protocol='frozen_final_prior_strength_DEV64_c1_c05_v1', args=args, keys=keys,
                body_sha256=digest, strengths=[1., .5], single_forward_per_modality=True,
                input_inventory=fingerprint(inventories), STC_receipt_sha256=artifact.sha,
                implementation={str(p.relative_to(root)): file_sha256(p) for p in sorted(files)},
                torch=str(torch.__version__), cuda=torch.version.cuda)
            cp = out/'contract.json'
            if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract:
                raise RuntimeError('resume identity/data/code changed')
            write_json(cp, contract); write_json(out/'population.json', population)
            runner = Predictor(body, pcfg, a.device, a.cpu_workers)
            gp = out/'GT_entrance_audit.json'
            if gp.exists(): gate = json.loads(gp.read_text(encoding='utf-8'))
            else:
                gate = audit_gt_entrance(runner, source, records[0]); write_json(gp, gate)
            if not gate['pass_gate']: raise RuntimeError('GT entrance parity FAIL; no quality diagnostic')
            sp = out/'state.json'
            state = json.loads(sp.read_text(encoding='utf-8')) if sp.exists() else None
            try:
                result = evaluate(runner, source, jobs, keys, contract, artifact=artifact, saved=state,
                    save=lambda s: write_json(sp, s), stop_event=event)
            except InterruptedError:
                write_json(out/'status.json', dict(status='stopped_at_complete_window')); return 130
            if before != tensor_state_fingerprint(body.state_dict()): raise RuntimeError('frozen body changed')
            result.update(source_epoch=saved['epoch'], source_update=saved['updates'], contract=contract)
            write_json(out/'evaluation.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
            print(summary(result), flush=True); return 0
        finally:
            if runner: runner.close()
            artifact.close()


if __name__ == '__main__':
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    sys.exit(main(stop) or 0)
