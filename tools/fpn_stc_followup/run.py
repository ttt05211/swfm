"""Real64 latest-FPN follow-up, no training of the source body or full eval."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.fpn_stc_local.run import load_model
from tools.fpn_stc_local import calibration as cal
from tools.fpn_unet_all.run import AllPredictor
from tools.stc_overnight.bundle import Bundle
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from . import core

PROTOCOL = 'local_FULL_FPN_STC_age_weight_temperature_conditional_semantic_calibration_v1'


def finish(out, contract, arrays):
    groups = {}
    for group, selected in (('fit32_in_sample', slice(0, 32)), ('held32_repeated_dev', slice(32, 64)),
            ('all64_descriptive_includes_fit', slice(None))):
        groups[group] = {name: cal.report(arrays[selected, i].sum(0)) for i, name in enumerate(contract['names'])}
    ref = groups['fit32_in_sample']['free_minus1']['average']
    def objective(name):
        row = groups['fit32_in_sample'][name]['average']; ds = [row[k]-ref[k] for k in ('mIoU', 'IoU')]
        return min(ds), sum(ds), -contract['names'].index(name)
    selected = max(contract['names'][1:], key=objective)
    result = dict(status='complete', contract=contract, reports=groups, selected_on_fit32_only=selected,
        source_body_optimizer_updates=0, Moving_available=False,
        caveat='Repeated VAL64 development, labeled FIT32 for semantic biases/any selection; not ALL4369 or independent test.')
    base.write_json(out/'results.json', result)
    lines = ['===== LOCAL LATEST FPN / '+contract['study'].upper()+' =====', 'status=complete; windows=64',
        'Original frozen FULL TRAIN20430x2, STC age0.5, GT future ego, all six predictions.',
        'No source weight/cache/default writes. No Moving masks in local bundle; NOT FPS.',
        'FIRST32 only for any fitting/selection; LAST32 repeated development, NOT independent test.',
        'selected_on_fit32_only='+selected]
    for group, reports in groups.items():
        old = reports['free_minus1']['average']; lines.append(group+' route mIoU IoU | delta_vs_fixed_free_minus1')
        for n, r in reports.items():
            m = r['average']
            lines.append(f'{n:24s} {m["mIoU"]:.6f} {m["IoU"]:.6f} | {m["mIoU"]-old["mIoU"]:+.6f} {m["IoU"]-old["IoU"]:+.6f}')
    lines.append(result['caveat']); text = '\n'.join(lines)+'\n'
    (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)


def main(argv=None, event=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights-zip', required=True); ap.add_argument('--bundle', default='../stc_overnight_bundle.zip')
    ap.add_argument('--out-dir', required=True); ap.add_argument('--study', choices=('importance', 'semantic', 'prior'), required=True)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(argv); event = event or threading.Event(); repo = Path(__file__).resolve().parents[2]
    torch.set_num_threads(1); torch.manual_seed(21); out = Path(a.out_dir).resolve()
    if not a.resume: out.mkdir(parents=True, exist_ok=False)
    body, source, _ = load_model(a.weights_zip, a.device); body.columns.horizon_chunk = 1
    bundle = Bundle(a.bundle, cache_mib=128); predictor = None
    before = tensor_state_fingerprint(body.state_dict()); started = time.perf_counter()
    try:
        keys = source['source_contract']['eval_keys']; rows = [bundle.by_key[tuple(k)] for k in keys]
        if len(rows) != 64 or len({r['scene_name'] for r in rows}) != 64: raise ValueError('exact original64 distinct scenes')
        pcfg = base.make_prepare_config(base.load_runtime_config(repo/'configs/real_motion_occfm.yaml', source['source_contract'].get('overrides')))
        if asdict(pcfg.grid) != asdict(bundle.pcfg.grid): raise ValueError('portable grid differs')
        names = ['original', 'free_minus1']+dict(importance=['importance_T2', 'importance_T2_free_minus1',
            'importance_uniform', 'importance_uniform_free_minus1'], semantic=['semantic_ce', 'semantic_soft_iou'],
            prior=['temporal_prior', 'temporal_prior_free_minus1'])[a.study]
        contract = dict(protocol=PROTOCOL, study=a.study, source_sha256=source['source_sha256'],
            weights_zip_sha256=base.sha256(a.weights_zip), bundle_sha256=base.sha256(a.bundle), keys=keys, names=names,
            free_bias=-1., half_life_s=.5, horizon_chunk=1,
            fixed_settings=dict(importance_temperatures=[2., 'infinite'], semantic_bias_scalars=17,
                semantic_steps=200, semantic_lr=.05, semantic_ridge=.01, sample_cap_per_class_horizon_window=128),
            implementation={f.name: base.sha256(f) for f in Path(__file__).parent.glob('*.py')},
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=a.device))
        cp = out/'contract.json'
        if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract: raise ValueError('resume SAME frozen study only')
        base.write_json(cp, contract)
        predictor = AllPredictor(body, pcfg, a.device, workers=2); predictor.kind = 'stc'
        def forward(row):
            rec, raw, times = bundle.history(row, 'stc', 'dense_valid')
            prepared = predictor.prepare(rec, raw, times=times)
            with torch.no_grad(): fw = predictor.forward(prepared, poses=np.asarray(row['future_poses']))
            return prepared, fw
        biases = {}
        if a.study == 'semantic':
            fit_path = out/'fit.json'
            if a.resume and fit_path.exists():
                fit = json.loads(fit_path.read_text(encoding='utf-8'))
                if fit['contract'] != contract: raise ValueError('fit identity differs')
                biases = fit['biases']
            else:
                samples = []
                for i, row in enumerate(rows[:32]):
                    if event.is_set(): raise InterruptedError('fit sample saved')
                    path = out/f'fit_samples_{i:02d}.npz'
                    if a.resume and path.exists():
                        with np.load(path, allow_pickle=False) as z: sample = {k: z[k].copy() for k in z.files}
                    else:
                        prepared, fw = forward(row)
                        # This is explicitly labeled FIT32 calibration. Held32
                        # future labels are never accessed here.
                        target = bundle.targets(row)
                        sample = core.sample_fit(fw['logits'], target, seed=21+i)
                        np.savez_compressed(path, **sample); del prepared, fw, target
                    samples.append(sample)
                    if (i+1) % 8 == 0: print(f'SEMANTIC_FIT_SAMPLES {i+1}/32', flush=True)
                data = {k: np.concatenate([s[k] for s in samples]) for k in samples[0]}; trace = {}
                for objective in ('ce', 'soft_iou'):
                    bias, values = core.fit_bias(data, objective); biases['semantic_'+objective] = bias; trace[objective] = values
                base.write_json(fit_path, dict(contract=contract, fit_keys=keys[:32], biases=biases, trace=trace,
                    labeled_target_calibration=True, optimizer_updates_each=200, source_body_updated=False))
                del samples, data
                print('SEMANTIC_FIT finalized BEFORE held32 target access; 17 offsets/arm, source frozen', flush=True)
        counts = []
        for i, row in enumerate(rows):
            if event.is_set(): raise InterruptedError('complete evaluation window saved')
            path = out/f'counts_{i:02d}.npz'
            if a.resume and path.exists():
                with np.load(path, allow_pickle=False) as z:
                    if str(z['key']) != json.dumps(keys[i]): raise ValueError('saved key differs')
                    value = z['counts'].copy()
            else:
                prepared, fw = forward(row); predictions = core.readouts(fw['logits'], fw['shape'], biases)
                if a.study == 'importance':
                    del fw['logits'], fw['memory']
                    for tag, temperature in (('T2', 2.), ('uniform', float('inf'))):
                        z = core.memory_logits(body, predictor.provider, prepared, fw, temperature)
                        extra = core.readouts(z, fw['shape']); del z
                        predictions['importance_'+tag] = extra['original']
                        predictions['importance_'+tag+'_free_minus1'] = extra['free_minus1']
                elif a.study == 'prior':
                    del fw['logits']
                    z = core.temporal_prior_logits(body, predictor.provider, prepared, fw)
                    extra = core.readouts(z, fw['shape']); del z
                    predictions['temporal_prior'] = extra['original']
                    predictions['temporal_prior_free_minus1'] = extra['free_minus1']
                del prepared, fw
                # Complete all route predictions BEFORE GT labels for scoring.
                target = bundle.targets(row); value = cal.counts(predictions, target, names=names)
                if a.study == 'semantic':
                    masks = [predictions[n] != 17 for n in ('free_minus1', 'semantic_ce', 'semantic_soft_iou')]
                    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]): raise RuntimeError('binary support changed')
                np.savez_compressed(path, key=json.dumps(keys[i]), counts=value)
                del predictions, target
            if value.shape != (len(names), 3, 18, 18) or value.dtype.kind not in 'iu' or np.any(value < 0): raise ValueError('invalid counts')
            if np.any(value.sum((-1, -2)) != np.prod(pcfg.grid.shape_hwd)): raise ValueError('incomplete full-grid counts')
            counts.append(value)
            if (i+1) % 8 == 0:
                print(f'FOLLOWUP_{a.study} {i+1}/64 seconds={time.perf_counter()-started:.1f}', flush=True)
                base.write_json(out/'progress.json', dict(status='running', windows=i+1, total=64))
        finish(out, contract, np.stack(counts)); base.write_json(out/'progress.json', dict(status='complete', windows=64, total=64))
        return 0
    except InterruptedError:
        base.write_json(out/'progress.json', dict(status='stopped_complete_item', total=64)); return 130
    finally:
        if predictor is not None: predictor.close()
        bundle.close()
        if tensor_state_fingerprint(body.state_dict()) != before: raise RuntimeError('source model changed')


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event=event) or 0)
