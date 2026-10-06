#!/usr/bin/env python3
"""One CCR revision: causal warm-cache training + shared spatial context.

Same real TRAIN16/DEV12+4 and 64 passes as v1; no distillation/AE, DEV threshold
search, automatic server training or promotion. Cold prep is reported apart
from warm throughput. Old quality is reused from a validated immutable run;
paired six-frame speed and actual full-joint backward are measured anew.
"""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections import defaultdict
import json
import time

import numpy as np
import torch

from real_motion.canonical_repair_context import (SpatialCanonicalRepairHead,
    FixedCanonicalCache, attach_neighbors, PROTOCOL)
from real_motion.canonical_causal_repair import (map_canonical_evidence, map_canonical_reference,
                                               build_canonical_evidence, repair_targets)
from real_motion.local_replay_bundle import ReplayBundle, file_digest
from real_motion.column_execution import execution_session
from real_motion.runtime_config import make_prepare_config
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import load_exported_config
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.joint_column_full_common import build_fixed_geometry
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, finite_json
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import (fit, score, timed_full,
                                                                 joint_probe)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--reference-run', required=True)
    p.add_argument('--passes', type=int, default=64)
    p.add_argument('--warm-start-ccr', action='store_true', help='initialize the base from local v1; zero residual spatial layer, then 64 extra passes')
    a = p.parse_args()
    if a.passes != 64: p.error('fixed-budget comparison requires 64 passes; do not tune on DEV')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('actual verified CUDA/BF16 required')
    out, reference = Path(a.out_dir), Path(a.reference_run)
    if out.exists(): p.error('fresh output required; original runs/checkpoints are read-only')
    old = json.loads((reference/'pilot.json').read_text(encoding='utf-8'))
    if old['status'] != 'complete' or old['fit']['passes'] != 64:
        raise RuntimeError('completed v1 fixed-budget reference required')
    out.mkdir(parents=True); torch.set_num_threads(1); torch.manual_seed(20261006)
    device = torch.device('cuda'); cache = FixedCanonicalCache(384, neighbors=True)
    report = dict(status='running', protocol=PROTOCOL, GPU=torch.cuda.get_device_name(),
        reference_run=str(reference.resolve()), thresholds=[.5, .95], no_KD=True, no_AE=True,
        local_only=True, no_old_checkpoint_edits=True, no_auto_server_training=True,
        old_joint=old['old_joint'], previous_CCR=old['learned'], previous_CCR_fit=old['fit'],
        initialization='local_v1_plus_zero_spatial_residual' if a.warm_start_ccr else 'RANDOM',
        effective_local_updates=2048 if a.warm_start_ccr else 1024)
    def save(): (out/'pilot.json').write_text(json.dumps(finite_json(report), ensure_ascii=False, indent=2), encoding='utf-8')
    def progress(row):
        with (out/'progress.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(finite_json(row), ensure_ascii=False)+'\n')
    save(); bundle = ReplayBundle(a.bundle)
    try:
        report['manifest_fingerprint'] = bundle.manifest['manifest_fingerprint']
        for member in ('runtime.yaml', 'checkpoints/epoch_0019.pt', 'checkpoints/clean_e14.pt'):
            if file_digest(reference/Path(member).name) != bundle.manifest['members'][member]['sha256']:
                raise RuntimeError('reference snapshot/replay mismatch')
            bundle.copy_member(member, out/Path(member).name)
        cfg = load_exported_config(out/'runtime.yaml', bundle.manifest['config_fingerprint'])
        _, teacher = load_joint(out/'epoch_0019.pt', device, reference_sha=CLEAN_SHA256,
            config_sha=bundle.manifest['config_fingerprint'], allow_diagnostic=True)
        teacher.eval().requires_grad_(False)
        provider = PilotProvider(out/'clean_e14.pt', CLEAN_SHA256, make_prepare_config(cfg), device, 2, teacher, None)
        teacher.columns.column_inference_optimized = True
        teacher.columns.column_async_readback = True
        teacher.columns.column_probability_optimized = False
        teacher.columns.column_sampling_workers = 2
        teacher.columns.column_inference_verify_remaining = 0
        cases = []; cold = defaultdict(float); count = defaultdict(int)
        for index, meta in enumerate(bundle.manifest['windows']):
            record, raw, labels = bundle.window(index, labels=True); raw['future_gt_occ'] = None
            tick = time.perf_counter()
            causal = {**raw, '_column_causal_preparation': build_fixed_geometry(
                raw, record, provider.pcfg, provider.strong, 2, teacher.columns.config)}
            cold['registration_Strong_static_memory'] += time.perf_counter()-tick
            with torch.no_grad():
                output = teacher.motion(record, device)
                prep = provider.prepare_columns(None, record, include_gt=False, raw_window=causal, outputs=output)
            tick = time.perf_counter(); evidence, graph = cache.get(prep, provider.pcfg.grid)
            cold['fixed_canonical_build_and_neighbors'] += time.perf_counter()-tick
            attach_neighbors(evidence, graph)
            plan = map_canonical_evidence(evidence, prep, provider.pcfg.grid)
            exact = map_canonical_reference(evidence, prep, provider.pcfg.grid)
            for field in ('flat', 'base', 'fallback', 'legal', 'context'):
                if not np.array_equal(getattr(plan, field), getattr(exact, field)):
                    raise RuntimeError('live/reference geometry mismatch at '+field)
            del exact
            # Every window is compared with v1's full causal feature population.
            v1 = build_canonical_evidence(prep, provider.pcfg.grid)
            for field in ('features', 'labels', 'actor', 'classes', 'world', 'presence'):
                if not np.array_equal(getattr(evidence, field), getattr(v1, field)):
                    raise RuntimeError('fixed/v1 input mismatch at '+field)
            del v1
            gt = labels['future_gt_occ'].numpy(); target, valid = repair_targets(evidence, plan, gt)
            cases.append(dict(meta=meta, record=record, causal=causal, prep=prep, evidence=evidence, plan=plan,
                output=output, gt=gt, moving=labels['moving_support'].numpy(), target=target, valid=valid,
                grid=provider.pcfg.grid,static_conflicts=cache.static_conflicts(evidence,prep,provider.pcfg.grid)))
            count['support_points'] += len(evidence)
            print(f'CCR_V2_PREP {index+1}/32 points={len(evidence)} cache_mib={cache.stats()["mib"]:.1f}', flush=True)
        train = [c for c in cases if c['meta']['split']=='train']
        dev = [c for c in cases if c['meta']['split']=='dev' and c['meta']['stratum']=='representative']
        stress = [c for c in cases if c['meta']['split']=='dev' and c['meta']['stratum']!='representative']
        if ([c['meta']['key'] for c in train] != old['TRAIN_keys'] or [c['meta']['key'] for c in dev] != old['DEV_keys']):
            raise RuntimeError('reference population/order mismatch')
        report.update(TRAIN_keys=old['TRAIN_keys'], DEV_keys=old['DEV_keys'], cold_preparation=dict(cold),
            preparation_cache=cache.stats(), exact_input_windows=len(cases), exact_live_mapping_windows=len(cases))
        head = SpatialCanonicalRepairHead(teacher.columns.source_dim,normalized=not a.warm_start_ccr,
                                         zero_residual=a.warm_start_ccr).to(device)
        if a.warm_start_ccr:
            saved = torch.load(reference/'candidate.pt',map_location='cpu',weights_only=True)
            if saved['teacher_sha256']!=bundle.manifest['teacher_sha256'] or saved['TRAIN_keys']!=old['TRAIN_keys']:
                raise RuntimeError('local warm-start head population/teacher mismatch')
            missing,unexpected=head.load_state_dict(saved['head'],strict=False)
            if unexpected or any(not k.startswith('spatial.') for k in missing):
                raise RuntimeError('incompatible local warm-start head')
            report['warm_start_initial_score']=score(dev,head,device)
        report['fit'] = fit(head, train, device, a.passes, progress, .25,causal_sampling=a.warm_start_ccr); save()
        for variant in ('learned', 'static', 'dynamic'):
            report[variant] = score(dev, head, device, variant=variant); save()
        report['TRAIN_in_sample'] = score(train, head, device)
        report['stress'] = score(stress, head, device); save()
        torch.save(dict(protocol=PROTOCOL, head=head.state_dict(), passes=a.passes, deployable=False,
            TRAIN_keys=report['TRAIN_keys'], teacher_sha256=bundle.manifest['teacher_sha256'],
            thresholds=[.5, .95],spatial_normalized=not a.warm_start_ccr,initialization=report['initialization']), out/'candidate.pt')
        # Inference NEVER uses the warm canonical cache: charge construction of
        # all fixed evidence + neighbour graph in every timed six-frame call.
        import tools.real_motion.pilot_p0_f9_canonical_causal_repair as base_pilot
        original_build = base_pilot.build_canonical_evidence
        def build_for_spatial(prep, grid, **kwargs):
            from real_motion.canonical_repair_context import build_fixed_canonical
            ev, graph = build_fixed_canonical(prep, grid, neighbors=True)
            return attach_neighbors(ev, graph)
        speeds = []
        try:
            base_pilot.build_canonical_evidence = build_for_spatial
            speed_cases = [train[0], dev[0], next(c for c in train if c['meta']['stratum']!='representative'), stress[0]]
            with execution_session(teacher.columns, graphs=True, reuse=False):
                for case in speed_cases:
                    for is_old in (True, False): timed_full(case, teacher, provider, head, old=is_old)
                    for repeat in range(2):
                        for is_old in ((True, False) if not repeat else (False, True)):
                            row = timed_full(case, teacher, provider, head, old=is_old)
                            row.update(key=case['meta']['key'], stratum=case['meta']['stratum'], repeat=repeat+1)
                            speeds.append(row); print(f'CCR_V2_FPS {row["mode"]} {row["stratum"]} six_seconds={row["seconds"]:.4f}', flush=True)
        finally: base_pilot.build_canonical_evidence = original_build
        report['speed_samples'] = speeds
        report['speed'] = {}
        for role in ('representative', 'high_source_stress'):
            mean = {m:np.mean([s['seconds'] for s in speeds if s['mode']==m and s['stratum']==role]) for m in ('old_joint', 'CCR')}
            report['speed'][role] = dict(six_seconds=mean, FPS={m:6/v for m,v in mean.items()}, speedup=mean['old_joint']/mean['CCR'])
        report['speed_boundary'] = 'resident source tensors + registered FOUR histories -> fresh prior + live motion + fresh full canonical support/neighbours + all SIX dense frames; excludes I/O, initial registration, GT/metrics/warmup; NOT L40S or raw E2E'
        # Warm fixed canonical data is a real immutable input cache, not cached
        # motion/labels. Prefill time/bytes are separately reported above.
        reps = [c for c in train if c['meta']['stratum']=='representative']
        probe_cache = FixedCanonicalCache(192, neighbors=True)
        tick = time.perf_counter()
        for case in reps[:8]:
            ev,_=probe_cache.get(case['prep'],provider.pcfg.grid)
            if a.warm_start_ccr:probe_cache.static_conflicts(ev,case['prep'],provider.pcfg.grid)
        report['probe_prefill_seconds'] = time.perf_counter()-tick
        report['joint_training_probe'] = joint_probe(provider, teacher, head, reps, device, fixed_cache=probe_cache,causal_sampling=a.warm_start_ccr)
        q, oq = report['learned']['joint'], old['old_joint']['joint']
        report['delta_vs_old'] = dict(mIoU=q['mIoU']-oq['mIoU'], MovingMicro=q['MovingMicro']-oq['MovingMicro'])
        report['gate'] = dict(mIoU_within_0_20pp=q['mIoU']>=oq['mIoU']-.2,
            Moving_within_0_20pp=q['MovingMicro']>=oq['MovingMicro']-.2,
            all_horizons_Moving_within_0_20pp=all(q['per_horizon'][h]['MovingMicro']>=oq['per_horizon'][h]['MovingMicro']-.2 for h in ('1.0','2.0','3.0')),
            representative_inference_speedup_ge_3=report['speed']['representative']['speedup']>=3,
            warm_joint_faster=report['joint_training_probe']['measured_speedup']>1)
        report['gate_note'] = '.20pp is a local diagnostic reference for slightly lower quality, NOT user-approved deployment tolerance; no automatic promotion'
        for member in ('runtime.yaml', 'checkpoints/epoch_0019.pt', 'checkpoints/clean_e14.pt'):
            if file_digest(out/Path(member).name) != bundle.manifest['members'][member]['sha256']:
                raise RuntimeError('original snapshot changed')
        report['status'] = 'complete'; report['route'] = 'local_candidate_only' if all(report['gate'].values()) else 'local_gate_failed_no_automatic_server_training'
        report['warning'] = 'Old epoch19 full-data head vs TRAIN16 new head: NOT equal-budget architecture comparison. Local 3050 short probes cannot predict L40S FPS or convergence. Mini-fit time excludes online geometry/labels.'
        save()
        summary = ['===== CCR SPATIAL + FIXED HISTORY WARM CACHE =====', 'TRAIN16 x64 / DEV12+4; no KD/AE; fixed thresholds=.5/.95',
            f'old mIoU={oq["mIoU"]:.6f} MovingMicro={oq["MovingMicro"]:.6f}',
            f'new mIoU={q["mIoU"]:.6f} MovingMicro={q["MovingMicro"]:.6f}',
            'delta='+json.dumps(finite_json(report['delta_vs_old'])), 'speed='+json.dumps(finite_json(report['speed'])),
            'actual_joint_training='+json.dumps(finite_json(report['joint_training_probe'])),
            'cold_preparation='+json.dumps(finite_json(report['cold_preparation'])),
            'gate='+json.dumps(finite_json(report['gate'])), report['route'], report['warning']]
        (out/'summary.txt').write_text('\n'.join(summary)+'\n', encoding='utf-8'); print('\n'.join(summary), flush=True)
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error)); save(); raise
    finally: bundle.close()


if __name__ == '__main__': main()
