"""Fixed DEV64: input / validity / residual prior / scene ego cause diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import threading
import sys
import torch
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.waymo_i2world import file_sha256, fingerprint
from tools.final_semantic_evaluation.predictor import Predictor, load_body, load_head10, audit_gt_entrance
from tools.final_semantic_evaluation.sources import NuscInputs, nusc_jobs
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.geniedrive_eval_alignment import validate_grid
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.static_evidence_selector_common import write_json
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.scene_ego_head.frozen_gt.comparison import find_head10
from .input_audit import ArtifactAudit
from .engine import evaluate, ROUTES


def parser():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for k in ('gt-checkpoint', 'dataroot', 'dev-info', 'dev-cache', 'stc-root', 'out-dir'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--head10-checkpoint'); p.add_argument('--stc-archive')
    p.add_argument('--cpu-workers', type=int, default=3); p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    return p


def summary(r):
    fmt = lambda v: 'NA' if v is None else f'{v:.4f}'
    lines = ['===== FROZEN GT/STC CAUSE DIAGNOSTIC =====', f"status={r['status']}; fixed windows={r['windows']}; scientific_updates=0",
        'GTepoch2+original head10; NO training/mask insertion/epsilon or axis selection/alignment.',
        'DIAGNOSTIC routes contain oracle historical inputs, NOT deployable Camera scores.',
        'route mIoU IoU MovingMicro (average1/2/3s)']
    for name, report in r['reports'].items():
        v = report['average']; lines.append(name+' '+' '.join(fmt(v[k]) for k in ('mIoU', 'IoU', 'MovingMicro')))
    lines.append('===== INPUT t0 FULL GRID / AXES / NEIGHBOR SANITY ONLY =====')
    for name, v in r['input_quality'].items(): lines.append(name+' mIoU='+fmt(v['mIoU'])+' IoU='+fmt(v['IoU']))
    v = r['input_quality']['identity']
    lines.append('input occupied precision/recall='+fmt(v['occupied_precision'])+'/'+fmt(v['occupied_recall']))
    lines.append('===== PAIRED ISOLATED DIAGNOSTIC DELTAS =====')
    lines.append(json.dumps(r['isolated_differences'], ensure_ascii=False))
    lines.append('===== EGO ADE/FDE3s/yaw3s =====')
    for name, v in r['trajectory'].items():
        lines.append(name+' '+fmt(v['ADE_m'])+' '+fmt(v['FDE3s_m'])+' '+fmt(v['yaw_mean_deg'][-1]))
    lines.append('Pred rigid geometry z/tilt/full-rotation@3s:')
    for name, v in r['rigid_pose_errors'].items():
        lines.append(name+' '+' '.join(fmt(v[k]['mean'][-1]) for k in ('z_m', 'tilt_deg', 'full_rotation_deg')))
    lines.append('===== WRONG PRIOR: CORRECTED / TOTAL, MEAN BARRIER / RESIDUAL ADVANTAGE =====')
    for name in ('occ_gt', 'stc_gt', 'stc_internal'):
        for h, row in enumerate(r['edits_and_prior_competition'][name], 1):
            v = row['groups']['wrong_prior']
            lines.append(f"{name} {h}s corrected={v['final_correct']}/{v['count']} damaged={row['damaged']} "
                f"barrier={fmt(v['barrier']['mean'])} residual_advantage={fmt(v['residual_advantage']['mean'])}")
    lines.append('geometry='+json.dumps(r['geometry'], ensure_ascii=False))
    lines.append('evidence='+json.dumps(r['evidence_summary'], ensure_ascii=False))
    if 'STC_artifact_audit' in r: lines.append('artifact='+json.dumps(r['STC_artifact_audit']))
    lines += ['Input axis/neighbor scores do not prove wrong axes/tokens alone; no transformed input used for predictions.',
              'Detailed integer counts, class errors, distributions and per-window evidence: state.json / evaluation.json.',
              'No source weights/caches changed. Quality diagnostic wall time is NOT FPS.']
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event(); a = parser().parse_args(argv)
    if not 1 <= a.cpu_workers <= 8: raise ValueError('bounded CPU workers required')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(1); torch.manual_seed(21)
    ck, out = Path(a.gt_checkpoint).resolve(), Path(a.out_dir).resolve()
    hp = Path(a.head10_checkpoint or find_head10(ck.parents[2], ck)).resolve()
    if any(out == p.parent or p.parent in out.parents for p in (ck, hp)): raise ValueError('new output outside checkpoint source required')
    lock = out.parent/'.semantic_diag_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(ck, out/'frozen/GTepoch2.pt'); snapshot_checkpoint(hp, out/'frozen/head10.pt')
        bp = out/'frozen/GTepoch2.pt'; hh = out/'frozen/head10.pt'; digest = file_sha256(bp)
        if file_sha256(ck) != digest or file_sha256(hp) != file_sha256(hh): raise RuntimeError('frozen source SHA changed')
        saved, body = load_body(bp, a.device); head = load_head10(hh, body, digest, a.device)
        before = (tensor_state_fingerprint(body.state_dict()), tensor_state_fingerprint(head.state_dict()))
        pcfg = make_prepare_config(load_runtime_config(a.config, a.override)); validate_grid(pcfg)
        if pcfg.frame_dt_s != .5 or pcfg.free_label != 17: raise ValueError('original 2Hz/free17 grid required')
        for name in ('dev_cache', 'dev_info'):
            if file_sha256(Path(getattr(a, name))) != saved['contract']['data'][name]: raise RuntimeError('GT source metadata changed')
        source = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
        _, records = load_cache(a.dev_cache)
        jobs, audit = nusc_jobs(source, records, saved['contract']['eval_keys'], 'dev64', 'stc')
        if len(jobs) != 64: raise RuntimeError('exact frozen DEV64 required; no replacement of missing STC')
        if {w.scene_name for w in jobs} & {k[0] for k in saved['contract']['train_keys']}: raise RuntimeError('TRAIN/DEV overlap')
        inventories = [source.preflight(jobs, kind) for kind in ('occ', 'stc')]
        keys = [[w.scene_name, w.t0_token] for w in jobs]
        artifact = ArtifactAudit(a.stc_root, a.stc_archive)
        runner = None
        try:
            root = Path(__file__).resolve().parents[2]
            files = [*Path(__file__).parent.glob('*.py'), *(root/'tools/final_semantic_evaluation').glob('*.py')]
            files += [root/p for p in ('tools/scene_ego_head/frozen_gt/engine.py', 'tools/scene_ego_head/model.py',
                'tools/scene_ego_head/integration.py', 'tools/shared_history_experiments/model.py',
                'tools/shared_history_experiments/refine/model.py', 'tools/shared_history_experiments/refine/geometry.py',
                'tools/shared_history_experiments/refine/history.py', 'real_motion/ego_navigation.py',
                'real_motion/stc_camera_protocol.py', 'real_motion/nuscenes_adapter.py')]
            args = vars(a).copy(); args.pop('resume')
            contract = dict(protocol='frozen_semantic_STC_cause_DEV64_v1', args=args, keys=keys,
                body_sha256=digest, head_sha256=file_sha256(hh), routes=list(ROUTES),
                input_inventory=fingerprint(inventories), STC_receipt_sha256=artifact.sha,
                implementation={str(p.relative_to(root)): file_sha256(p) for p in files},
                torch=str(torch.__version__), cuda=torch.version.cuda)
            cp = out/'contract.json'
            if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract: raise RuntimeError('diagnostic resume identity/code changed')
            write_json(cp, contract); write_json(out/'population.json', audit)
            runner = Predictor(body, pcfg, a.device, a.cpu_workers, head)
            gate_path = out/'GT_entrance_audit.json'
            if gate_path.exists(): gate = json.loads(gate_path.read_text(encoding='utf-8'))
            else:
                gate = audit_gt_entrance(runner, source, records[0]); write_json(gate_path, gate)
            if not gate['pass_gate']: raise RuntimeError('GT entrance parity FAIL; no diagnostics')
            state_path = out/'state.json'; state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
            try:
                result = evaluate(runner, source, jobs, keys, contract, artifact=artifact, saved=state,
                    save=lambda s: write_json(state_path, s), stop_event=event)
            except InterruptedError:
                write_json(out/'status.json', dict(status='stopped_at_complete_window')); return 130
            after = (tensor_state_fingerprint(body.state_dict()), tensor_state_fingerprint(head.state_dict()))
            if before != after: raise RuntimeError('frozen body/head changed')
            result.update(source_epoch=saved['epoch'], source_update=saved['updates'], contract=contract)
            write_json(out/'evaluation.json', result); (out/'summary.txt').write_text(summary(result), encoding='utf-8')
            print(summary(result), flush=True); return 0
        finally:
            if runner: runner.close()
            artifact.close()


if __name__ == '__main__':
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    sys.exit(main(stop) or 0)
