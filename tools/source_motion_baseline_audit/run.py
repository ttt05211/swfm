"""Freeze original dense/shared-pinned epoch2; same64 future-motion audit."""
import argparse
import json
from pathlib import Path
import signal
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.dense_final_all.run import load_frozen
from tools.shared_source_track.adapter import SharedTrackPredictor
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.source_motion_audit.statistics import window_counts, finish, edits, box_motion_rows, source_summary
from tools.source_motion_audit.run import PROTOCOL as STAGE2_PROTOCOL
from tools.source_motion_audit_brief import brief
from .engine import predictions, MODES
from .report import comparison

PROTOCOL = 'frozen_original_dense_epoch2_same64_future_motion_v1'


def select_checkpoint(explicit, directory, root):
    if explicit: return Path(explicit).resolve()
    if directory: return (Path(directory)/'epoch_0002.pt').resolve()
    paths = list(Path(root).glob('dense_valid_shared_full2_ddp_*/epoch_0002.pt'))
    if len(paths) != 1:
        raise RuntimeError('请设置BASELINE_MOTION_CKPT或DENSE_DDP_DIR；不能静默选多个旧模型: '+str(paths))
    return paths[0].resolve()


def validate_reference(reference, keys):
    if (reference.get('status') != 'complete' or reference.get('windows') != 64 or
            reference['contract']['protocol'] != STAGE2_PROTOCOL or reference['contract']['keys'] != keys
            or len(keys) != 64 or len(set(map(tuple, keys))) != 64 or
            reference['contract']['modes'] != list(MODES)):
        raise RuntimeError('completed Stage2 audit / exact ordered same64 / fixed four modes required')
    expected = {m+'/'+r for m in MODES for r in ('transport', 'semantic_refine')}
    if any(set(reference['metrics'].get(k, {})) != expected for k in ('occ_gt', 'stc_gt')):
        raise RuntimeError('reference routes incomplete')


def restore(saved, contract):
    if saved.get('protocol') != PROTOCOL or saved['contract'] != contract: raise RuntimeError('baseline audit contract differs')
    if saved['keys'] != contract['keys'][:len(saved['keys'])] or len(saved['keys']) != len(saved['rows']):
        raise RuntimeError('invalid completed-window prefix')
    for routes in saved['counts'].values():
        for fields in routes.values():
            for name in ('full', 'moving'):
                a = np.asarray(fields[name])
                if a.shape != (3, 18, 18) or a.dtype.kind not in 'iu' or np.any(a < 0): raise RuntimeError('invalid integer confusion')
                fields[name] = a.copy()
            if np.any(fields['moving'] > fields['full']): raise RuntimeError('moving exceeds full')
    return saved


@torch.no_grad()
def main(argv=None, event=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'stage2-report', 'config', 'dev-cache', 'dev-info', 'dataroot', 'stc-root'): p.add_argument('--'+name, required=True)
    p.add_argument('--checkpoint'); p.add_argument('--source-dir'); p.add_argument('--resume', action='store_true')
    p.add_argument('--cpu-workers', type=int, default=4)
    a = p.parse_args(argv); event = event or threading.Event()
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(): raise RuntimeError('ONE BF16 GPU required')
    if not 1 <= a.cpu_workers <= 4: raise ValueError('bounded CPU workers')
    torch.set_num_threads(1); device = torch.device('cuda:0'); root = Path(__file__).resolve().parents[2]
    source = select_checkpoint(a.checkpoint, a.source_dir, root/'outputs/p0_f9_shared_history')
    source_sha = base.sha256(source); c, model = load_frozen(source, device)
    reference_path = Path(a.stage2_report); reference_path = reference_path/'results.json' if reference_path.is_dir() else reference_path
    reference_sha = base.sha256(reference_path); reference = json.loads(reference_path.read_text(encoding='utf-8'))
    validate_reference(reference, c['eval_keys'])
    for file, sha in c['implementation'].items():
        if base.sha256(root/file) != sha: raise RuntimeError('original baseline implementation changed: '+file)
    for name in ('config', 'dev_cache', 'dev_info'):
        if base.sha256(getattr(a, name)) != c['data'][name]: raise RuntimeError('original config/data changed: '+name)
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, c.get('overrides')))
    _, records = base.load_cache(a.dev_cache); jobs = base.align_records(records, c['eval_keys'])
    if len(jobs) != 64: raise RuntimeError('same64 required')
    inputs = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=128); files = []
    for rec in jobs:
        w = window(rec)
        for kind in ('occ', 'stc'):
            for token in w.history_tokens:
                path = inputs.path(kind, w.scene_name, token); files.append((str(path), *inputs.stat(path)))
        for h in (1, 3, 5):
            path = inputs.path('occ', w.scene_name, w.future_tokens[h]); files.append((str(path), *inputs.stat(path)))
    contract = dict(protocol=PROTOCOL, checkpoint=str(source), checkpoint_sha256=source_sha, source_update=10292,
        keys=c['eval_keys'], modes=MODES, files=sorted(set(files)), fixed_Q_and_ICP=True, history_yaw='not_applicable_original_ICP',
        known='TRAIN_OCC_STC_dense_valid', stage2_reference=str(reference_path.resolve()), stage2_sha256=reference_sha,
        cpu_workers=a.cpu_workers, runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        metadata={n: base.sha256(Path(a.dataroot)/'v1.0-trainval'/n) for n in
            ('sample.json', 'sample_annotation.json', 'instance.json', 'category.json', 'sample_data.json', 'ego_pose.json')},
        implementation={str(f.relative_to(root)): base.sha256(f) for f in [*Path(__file__).parent.glob('*.py'),
            root/'tools/source_motion_audit/engine.py', root/'tools/source_motion_audit/statistics.py',
            root/'tools/source_motion_audit_brief.py', root/'tools/dense_final_all/run.py',
            root/'tools/dense_valid_epoch3/run.py', root/'tools/dense_checkpoint_average/weights.py']})
    contract = json.loads(json.dumps(contract))
    for name in ('files', 'metadata', 'runtime', 'cpu_workers'):
        if contract[name] != reference['contract'][name]: raise RuntimeError('Stage2 comparison input/runtime differs: '+name)
    out = Path(a.out).resolve()
    if out == source.parent or source.parent in out.parents: raise ValueError('use NEW output outside original training run')
    lock = out.parent/'.baseline_motion_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if a.resume: state = restore(json.loads((out/'state.json').read_text(encoding='utf-8')), contract)
        else:
            out.mkdir(parents=True, exist_ok=False)
            state = dict(protocol=PROTOCOL, contract=contract, counts={'occ_gt': {}, 'stc_gt': {}}, keys=[], rows=[])
        runner = SharedTrackPredictor(model, pcfg, device, workers=a.cpu_workers)
        def publish(status):
            result = dict(status=status, windows=len(state['keys']), contract=contract, scientific_updates=0,
                stage2_reference=reference,
                metrics={kind: {r: finish(v) for r, v in routes.items()} for kind, routes in state['counts'].items()},
                source_motion={kind: source_summary([r for w in state['rows'] for r in w[kind]['sources']]) for kind in ('occ_gt', 'stc_gt')},
                history_yaw_abs_deg={kind: {'n': 0} for kind in ('occ_gt', 'stc_gt')},
                reference_repeat={kind: {r: dict(failed_windows=sum(w[kind]['repeat'][r] > 0 for w in state['rows']),
                    max_changed_voxels=max((w[kind]['repeat'][r] for w in state['rows']), default=0))
                    for r in ('transport', 'semantic_refine')} for kind in ('occ_gt', 'stc_gt')},
                edit_counts={kind: {r: {k: sum(w[kind]['edits'][r].get(k, 0) for w in state['rows'])
                    for k in {k for w in state['rows'] for k in w[kind]['edits'][r]}}
                    for r in ('transport', 'semantic_refine')} for kind in ('occ_gt', 'stc_gt')})
            serial = dict(state, counts={kind: {r: {k: v.tolist() for k, v in f.items()} for r, f in routes.items()}
                for kind, routes in state['counts'].items()})
            base.write_json(out/'state.json', serial); base.write_json(out/'results.json', result)
            text = 'OLD dense/shared-pinned epoch2; original Q/ICP retained; history learned yaw=NA.\n'+brief(result, out/'results.json')
            text = text.replace('历史yaw与Q不变', '原ICP与Q不变')
            (out/'summary.txt').write_text(text+'\n'+comparison(result, reference), encoding='utf-8')
            (out/'comparison.txt').write_text(comparison(result, reference), encoding='utf-8')
            return result
        try:
            for rec in jobs[len(state['keys']):]:
                if event.is_set(): break
                w = window(rec); forecasts = {}; diagnostics = {}; rows = {}
                for kind, modality in (('occ_gt', 'occ'), ('stc_gt', 'stc')):
                    record, historical, times = inputs.history(w, modality)
                    sem = np.asarray(historical['history_occ'])
                    historical = dict(historical, history_observed=(sem >= 0)&(sem < 18))
                    rows[kind] = runner.prepare(record, historical, times=times)
                    forecasts[kind], diagnostics[kind] = predictions(model, runner.provider, rows[kind], inputs.poses(w))
                targets = inputs.targets(w, (1, 3, 5))
                masks = np.asarray(inputs.moving(w, (1, 3, 5), (1., 2., 3.), pcfg.grid, a.cpu_workers), bool)
                wr = dict(key=[str(rec['scene_name']), str(rec['t0_token'])]); local = {}
                for kind in forecasts:
                    local[kind] = window_counts(forecasts[kind], targets, masks)
                    wr[kind] = dict(sources=box_motion_rows(rows[kind], diagnostics[kind], inputs.source.nusc, pcfg.grid),
                        edits=edits(forecasts[kind], targets, masks), repeat=diagnostics[kind]['reference_repeat_changed_voxels'])
                for kind, routes in local.items():
                    for r, fields in routes.items():
                        if r not in state['counts'][kind]: state['counts'][kind][r] = fields
                        else:
                            for k, v in fields.items(): state['counts'][kind][r][k] += v
                state['keys'].append(wr['key']); state['rows'].append(wr)
                if len(state['keys']) % 8 == 0:
                    publish('running'); print(f'BASELINE_MOTION {len(state["keys"])}/64', flush=True)
            if base.sha256(source) != source_sha or base.sha256(reference_path) != reference_sha:
                raise RuntimeError('read-only source/reference changed')
            result = publish('complete' if len(state['keys']) == 64 else 'stopped_at_complete_window')
            print((out/'summary.txt').read_text(encoding='utf-8'), flush=True)
            return result
        finally: runner.close()


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    main(event=event)
