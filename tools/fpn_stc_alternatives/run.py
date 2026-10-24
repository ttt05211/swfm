"""Fixed first16 screen; explicit same-output extension to original64 only."""
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
from tools.fpn_stc_followup.core import readouts
from tools.fpn_unet_all.run import AllPredictor
from tools.stc_overnight.bundle import Bundle
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from . import core

PROTOCOL = 'local_FULL_FPN_STC_embedding_tiled_readout162_v1'


def finish(out, contract, counts):
    groups = [('screen16_inspected_or_fit', slice(0, 16))] if len(counts) == 16 else [
        ('fit32_in_sample', slice(0, 32)), ('held32_repeated_dev', slice(32, 64)),
        ('all64_descriptive_includes_fit', slice(None))]
    reports = {g: {n: cal.report(counts[at, i].sum(0)) for i, n in enumerate(contract['names'])} for g, at in groups}
    result = dict(status='complete', windows=len(counts), contract=contract, reports=reports,
        source_body_optimizer_updates=0, Moving_available=False,
        caveat='Inspected original VAL scenes. FIT32 labels used for embedding/head only; NOT zero-adaptation or independent test. NOT ALL4369.')
    base.write_json(out/'results.json', result)
    lines = ['===== LOCAL FPN STC ALTERNATIVES / '+contract['study'].upper()+' =====',
        f'status=complete; windows={len(counts)}; FULL TRAINx2 frozen; original real64 key order.',
        'STC age0.5 / GT future ego / FOUR histories / SIX futures / full-grid 1/2/3s.',
        'Same-window original + fixed FREE=-1 references. No Moving package. Quality time NOT FPS.']
    for g, rows in reports.items():
        old = rows['free_minus1']['average']; lines.append(g+' route mIoU IoU | delta_vs_free_minus1')
        for n, r in rows.items():
            m = r['average']; lines.append(f'{n:22s} {m["mIoU"]:.6f} {m["IoU"]:.6f} | {m["mIoU"]-old["mIoU"]:+.6f} {m["IoU"]-old["IoU"]:+.6f}')
    lines.append(result['caveat']); text = '\n'.join(lines)+'\n'
    (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)


def main(argv=None, event=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights-zip', required=True); ap.add_argument('--bundle', default='../stc_overnight_bundle.zip')
    ap.add_argument('--out-dir', required=True); ap.add_argument('--study', choices=('embedding', 'tiled', 'head', 'conditional'), required=True)
    ap.add_argument('--fit-samples-from', help='completed head FIT32 sample directory; conditional study only')
    ap.add_argument('--windows', type=int, choices=(16, 64), default=16); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--device', default='cuda'); a = ap.parse_args(argv); event = event or threading.Event()
    torch.set_num_threads(1); torch.manual_seed(21); repo = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    if not a.resume: out.mkdir(parents=True, exist_ok=False)
    body, source, _ = load_model(a.weights_zip, a.device); body.columns.horizon_chunk = 1
    before = tensor_state_fingerprint(body.state_dict()); bundle = Bundle(a.bundle, cache_mib=128); predictor = None
    started = time.perf_counter()
    try:
        keys = source['source_contract']['eval_keys']; rows = [bundle.by_key[tuple(k)] for k in keys]
        if len(rows) != 64 or len({r['scene_name'] for r in rows}) != 64: raise ValueError('original64 distinct scenes required')
        pcfg = base.make_prepare_config(base.load_runtime_config(repo/'configs/real_motion_occfm.yaml', source['source_contract'].get('overrides')))
        if asdict(pcfg.grid) != asdict(bundle.pcfg.grid): raise ValueError('portable grid differs')
        names = ['original', 'free_minus1']+dict(embedding=['embedding_half', 'embedding_full'], tiled=['tiled128'],
            head=['readout162'], conditional=['conditional153'])[a.study]
        contract = dict(protocol=PROTOCOL, source_sha256=source['source_sha256'], keys=keys, study=a.study, names=names,
            bundle_sha256=base.sha256(a.bundle), weights_zip_sha256=base.sha256(a.weights_zip),
            settings=dict(free_bias=-1., age_half_life=.5, horizon_chunk=1, embedding_strengths=[.5, 1.],
                tile_side=128, gaussian_sigma_scale=.125, head_parameters=162, head_steps=200, head_lr=.02, head_ridge=.01),
            implementation={f.name: base.sha256(f) for f in Path(__file__).parent.glob('*.py')},
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=a.device))
        sample_dir = Path(a.fit_samples_from).resolve() if a.fit_samples_from else None
        if sample_dir is not None:
            old = json.loads((sample_dir/'fit.json').read_text(encoding='utf-8'))
            if a.study != 'conditional' or old['contract']['study'] != 'head' or old['fit_keys'] != keys[:32]:
                raise ValueError('reuse only original head FIT32 samples')
            for n in ('source_sha256', 'keys', 'bundle_sha256', 'weights_zip_sha256'):
                if old['contract'][n] != contract[n]: raise ValueError('sample identity differs: '+n)
            contract['fit_samples'] = {str(sample_dir/f'head_samples_{i:02d}.npz'): base.sha256(sample_dir/f'head_samples_{i:02d}.npz') for i in range(32)}
        if a.study == 'conditional': contract['settings']['head_parameters'] = 153
        cp = out/'contract.json'
        if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract: raise ValueError('resume SAME frozen study only')
        base.write_json(cp, contract)
        predictor = AllPredictor(body, pcfg, a.device, workers=2); predictor.kind = 'stc'
        def prepare(row):
            rec, raw, times = bundle.history(row, 'stc', 'dense_valid')
            return predictor.prepare(rec, raw, times=times)
        def forward(prepared, row):
            with torch.no_grad(): return predictor.forward(prepared, poses=np.asarray(row['future_poses']))
        fit = {}; fit_path = out/'fit.json'
        if a.study in ('embedding', 'head', 'conditional'):
            if a.resume and fit_path.exists():
                fit = json.loads(fit_path.read_text(encoding='utf-8'))
                if fit['contract'] != contract: raise ValueError('fit identity differs')
            else:
                confusion = np.zeros((18, 18), np.int64); samples = []
                for i, row in enumerate(rows[:32]):
                    if event.is_set(): raise InterruptedError('fit interrupted; original unchanged')
                    if a.study == 'embedding':
                        # Paired HISTORICAL semantics only, no future labels or masks.
                        for token in row['history_tokens']:
                            stc = bundle.frame('stc', row['scene_name'], token)['semantics']
                            gt = bundle.frame('gt', row['scene_name'], token)['semantics']
                            confusion += np.bincount(18*stc.ravel().astype(np.int64)+gt.ravel(), minlength=324).reshape(18, 18)
                    else:
                        path = (sample_dir or out)/f'head_samples_{i:02d}.npz'
                        if (a.resume or sample_dir is not None) and path.exists():
                            with np.load(path, allow_pickle=False) as z: sample = {k: z[k].copy() for k in z.files}
                        else:
                            prepared = prepare(row)
                            with core.capture_readout(body.columns) as activations: fw = forward(prepared, row)
                            target = bundle.targets(row)  # explicitly supervised FIT32 only
                            sample = core.sample_head(fw['logits'], activations, target, seed=21+i)
                            np.savez_compressed(path, **sample); del fw, prepared, activations, target
                        samples.append(sample)
                    if (i+1) % 8 == 0: print(f'ALTERNATIVE_FIT_{a.study} {i+1}/32', flush=True)
                fit = dict(contract=contract, fit_keys=keys[:32], calibration='labeled target domain; NOT zero-adaptation')
                if a.study == 'embedding': fit['confusion_STC_row_GT_column'] = confusion.tolist()
                else:
                    fit['patch'] = core.fit_head({k: np.concatenate([r[k] for r in samples]) for k in samples[0]}, conditional=a.study == 'conditional'); del samples
                base.write_json(fit_path, fit)
                print('FIT finalized BEFORE any held32 future target access', flush=True)
        tables = {n: core.embedding_table(fit['confusion_STC_row_GT_column'], body.transport.history_encoder.semantic.weight, strength)
            for n, strength in (('embedding_half', .5), ('embedding_full', 1.))} if a.study == 'embedding' else {}
        counts = []
        for i, row in enumerate(rows[:a.windows]):
            if event.is_set(): raise InterruptedError('complete evaluation window saved')
            path = out/f'counts_{i:02d}.npz'
            if a.resume and path.exists():
                with np.load(path, allow_pickle=False) as z:
                    if str(z['key']) != json.dumps(keys[i]): raise ValueError('saved key differs')
                    value = z['counts'].copy()
            else:
                prepared = prepare(row); fw = forward(prepared, row); predictions = readouts(fw['logits'], fw['shape'])
                if a.study == 'tiled':
                    del fw['logits']
                    z = core.tiled_logits(body.columns, fw['memory'], fw['prior'], fw['shape'])
                    predictions['tiled128'] = readouts(z, fw['shape'])['free_minus1']; del z, fw
                else:
                    del fw
                    if a.study == 'embedding':
                        for n, table in tables.items():
                            with core.embedding_override(body.transport.history_encoder, table): changed = forward(prepared, row)
                            predictions[n] = readouts(changed['logits'], changed['shape'])['free_minus1']; del changed
                    else:
                        with core.readout_patch(body.columns, fit['patch']): changed = forward(prepared, row)
                        if a.study == 'conditional':
                            semantic = changed['logits'][[1, 3, 5], :17].argmax(1).reshape(3, *changed['shape']).byte().cpu().numpy()
                            predictions['conditional153'] = np.where(predictions['free_minus1'] != 17, semantic, 17).astype(np.uint8)
                        else: predictions['readout162'] = readouts(changed['logits'], changed['shape'])['free_minus1']
                        del changed
                del prepared
                target = bundle.targets(row)  # all route forecasts precede scoring target reads
                value = cal.counts(predictions, target, names=names)
                np.savez_compressed(path, key=json.dumps(keys[i]), counts=value); del predictions, target
            if value.shape != (len(names), 3, 18, 18) or value.dtype.kind not in 'iu' or np.any(value < 0): raise ValueError('invalid integer counts')
            if np.any(value.sum((-1, -2)) != np.prod(pcfg.grid.shape_hwd)): raise ValueError('incomplete full-grid counts')
            counts.append(value)
            if (i+1) % 8 == 0:
                print(f'ALTERNATIVE_{a.study} {i+1}/{a.windows} seconds={time.perf_counter()-started:.1f}', flush=True)
                base.write_json(out/'progress.json', dict(status='running', windows=i+1, total=a.windows))
        finish(out, contract, np.stack(counts)); base.write_json(out/'progress.json', dict(status='complete', windows=a.windows, total=a.windows))
        return 0
    except InterruptedError:
        base.write_json(out/'progress.json', dict(status='stopped_complete_item', total=a.windows)); return 130
    finally:
        if predictor is not None: predictor.close()
        bundle.close()
        if tensor_state_fingerprint(body.state_dict()) != before: raise RuntimeError('source model changed')


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event=event) or 0)
