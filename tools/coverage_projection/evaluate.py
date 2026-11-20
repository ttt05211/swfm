"""Same64 original/candidate metrics and bounded native-grid visual exports."""
import argparse
import json
from pathlib import Path
import signal
import threading
import time
import numpy as np
import torch
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.causal_forward_train.data import forward_window as original_forward
from tools.final_semantic_evaluation import metrics
from tools.final_semantic_evaluation.sources import window
from tools.fpn_unet_eval.common import DenseInputs
from .run import PROTOCOL, ROOT, reference, sha, write_json, prepare_row
from .model import from_spec
from .adapter import forward_window


def evaluation_plan(records, keys):
    by = {(str(r['scene_name']), str(r['t0_token'])): r for r in records}
    keys = list(map(tuple, keys))
    if len(records) != 4369 or len(by) != 4369 or len(keys) != 64 or len(set(keys)) != 64:
        raise ValueError('original VAL4369 and SAME ordered unique64 required')
    if any(k not in by for k in keys): raise ValueError('missing dev key; no replacement allowed')
    return [by[k] for k in keys]


def counts_from_json(value):
    result = {k: {f: np.asarray(a, dtype=np.int64) for f, a in c.items()} for k, c in value.items()}
    for c in result.values():
        if set(c) != {'confusion', 'moving'} or any(a.shape != (3, 18, 18) or np.any(a < 0) for a in c.values()):
            raise ValueError('invalid saved integer counts')
    return result


def run(a, stop):
    device = torch.device(a.device); torch.set_num_threads(1)
    checkpoint = a.out_dir/'epoch_0001.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False); c = saved['contract']
    if saved.get('protocol') != PROTOCOL or saved.get('updates') != 1031:
        raise ValueError('completed fixed candidate epoch1 only')
    for name, digest in c['implementations'].items():
        path = (ROOT/name).resolve()
        if ROOT not in path.parents or sha(path) != digest: raise ValueError('candidate implementation changed: '+name)
    control = reference(Path(c['control_dir'])); cc = control['contract']
    control_path = Path(c['control_dir'])/'epoch_0001.pt'
    if sha(control_path) != c['control_sha256']: raise ValueError('original control checkpoint changed')
    if sha(Path(c['control_dir'])/'results.json') != c['control_results_sha256']:
        raise ValueError('original control report changed')
    if cc['eval_keys'] != c['eval_keys']: raise ValueError('control/candidate populations differ')
    from tools.joint_occ_experiments import run as base
    _, records = base.load_cache(a.dev_cache); selected = evaluation_plan(records, c['eval_keys'])
    if sha(a.dev_info) != cc['data']['dev_info'] or sha(a.dev_cache) != cc['data']['dev_cache']:
        raise ValueError('VAL metadata differs from control')
    pcfg = make_prepare_config(load_runtime_config(a.config, cc.get('overrides', [])))
    if sha(a.config) != c['train_data']['config']: raise ValueError('geometry config differs')
    inputs = DenseInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=128)
    inventory = {kind: inputs.preflight([window(r) for r in selected], kind) for kind in ('occ', 'stc')}
    receipt = dict(candidate=sha(checkpoint), control=c['control_sha256'], keys=c['eval_keys'], inventory=inventory,
        full_grid=True, horizons=[1., 2., 3.], indices=[1, 3, 5], known='dense_valid',
        FREE=False, fixed_age=False, calibration=False, label_timing='all six forecasts before target labels')
    path = a.out_dir/'evaluation_state.json'
    state = json.loads(path.read_text()) if path.exists() else dict(receipt=receipt, progress={}, counts={})
    if state['receipt'] != receipt: raise ValueError('eval resume data/checkpoint differs')
    visual = a.out_dir/'visuals'; visual.mkdir(exist_ok=True)
    # Fixed first8 for paired quantitative/visual reproduction; no outcome ranking.
    visual_cases = [dict(index=i+1, scene=r['scene_name'], t0=r['t0_token']) for i, r in enumerate(selected[:8])]
    def save(): write_json(path, state)
    for label, source, candidate in (('control', control_path, False), ('candidate', checkpoint, True)):
        blob = torch.load(source, map_location='cpu', weights_only=False)
        model = from_spec(blob['contract']['model'], candidate=candidate)
        model.load_state_dict(blob['models']['gt'], strict=True); del blob
        model.to(device).eval().requires_grad_(False)
        predictor = RecoveryPredictor(model, pcfg, device, workers=a.cpu_workers)
        try:
            for kind in ('occ', 'stc'):
                key = label+'/'+kind; done = int(state['progress'].get(key, 0))
                if not 0 <= done <= 64: raise ValueError('invalid evaluation cursor')
                counts = counts_from_json(state['counts'][key]) if key in state['counts'] else {
                    route: metrics.empty(3) for route in ('transport', 'semantic_refine')}
                for index in range(done, 64):
                    if stop.is_set(): raise InterruptedError('saved complete evaluation windows')
                    row, poses, _ = prepare_row(inputs, predictor, selected[index], kind=kind)
                    with torch.no_grad():
                        fw = forward_window(model, predictor.provider, row, poses=poses) if candidate else original_forward(
                            model, predictor.provider, row, poses=poses, fresh=True)
                        predictions = {r: fw[t].argmax(1).reshape(6, *fw['shape']).cpu().numpy().astype(np.uint8)
                            for r, t in (('transport', 'prior'), ('semantic_refine', 'logits'))}
                        coverage = fw['prior_coverage'].reshape(6, *fw['shape']).cpu().numpy() if candidate else None
                    del fw
                    # No future occupancy/masks enter any part of the predictor.
                    job = window(selected[index]); targets = np.stack(inputs.targets(job, (1, 3, 5)))
                    for route, values in predictions.items():
                        for h, slot in enumerate((1, 3, 5)): metrics.add(counts[route], h, values[slot], targets[h])
                    if index < 8:
                        export = dict(prediction=predictions['semantic_refine'][[1, 3, 5]],
                            transport=predictions['transport'][[1, 3, 5]], gt=targets,
                            t0=np.asarray(row[1]['history_occ'][-1], dtype=np.uint8), future_poses=np.asarray(poses)[[1, 3, 5]])
                        if candidate: export['prior_coverage'] = (coverage[[1, 3, 5]]*255).round().astype(np.uint8)
                        np.savez_compressed(visual/f'{kind}_{index+1:02d}_{label}.npz', **export)
                    state['progress'][key] = index+1
                    state['counts'][key] = {r: {f: a.tolist() for f, a in v.items()} for r, v in counts.items()}
                    save(); print(f'COVERAGE_EVAL {key} {index+1}/64', flush=True)
        finally:
            predictor.close(); del predictor, model
            if device.type == 'cuda': torch.cuda.empty_cache()
    reports = {k: {r: metrics.report(v, (1., 2., 3.), moving=False) for r, v in counts_from_json(value).items()}
        for k, value in state['counts'].items()}
    lines = ['===== COVERAGE + VALID PRIOR + PROJECTION / MATCHED TRAIN20%x1 =====',
        'status=complete; windows=64; ONE GPU; original latest forward AdaLN control; full-grid.',
        'model input route mIoU IoU']
    for key, value in reports.items():
        for route, report in value.items():
            m = report['average']; lines.append(f'{key:16s} {route:18s} {m["mIoU"]:.6f} {m["IoU"]:.6f}')
    delta = {}
    for kind in ('occ', 'stc'):
        delta[kind] = {r: {m: reports['candidate/'+kind][r]['average'][m]-reports['control/'+kind][r]['average'][m]
            for m in ('mIoU', 'IoU')} for r in ('transport', 'semantic_refine')}
        lines.append(kind+' candidate_minus_control='+json.dumps(delta[kind]))
    lines += ['THREE JOINT changes; one fixed epoch/weight, no DEV-best selection or automatic expansion.',
        'Training-only volume projection inspired by GaussRender; not its Gaussian renderer.',
        'Repeated development64; NOT full-VAL/independent test/FPS. Historical control metrics remain unchanged.']
    result = dict(status='complete', receipt=receipt, reports=reports, delta=delta,
        control_recorded=control.get('evaluations'), visual_cases=visual_cases)
    write_json(a.out_dir/'evaluation.json', result)
    write_json(visual/'manifest.json', dict(cases=visual_cases, horizons=[1, 2, 3],
        palette='OccWorld/COME', selection='fixed first8 ordered DEV64; no outcome selection',
        grid=dict(origin=[pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min], step=list(pcfg.grid.voxel_size),
            shape=list(pcfg.grid.shape_hwd))))
    (a.out_dir/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-dir', 'dataroot', 'dev-info', 'dev-cache', 'stc-root'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--config', type=Path, default=ROOT/'configs/real_motion_occfm.yaml')
    p.add_argument('--cpu-workers', type=int, default=4); p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
    lock = a.out_dir/'eval_lock'; lock.mkdir(exist_ok=True)
    with evaluation_lock(lock): run(a, stop)


if __name__ == '__main__': main()
