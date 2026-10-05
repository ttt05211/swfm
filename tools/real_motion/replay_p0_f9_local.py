#!/usr/bin/env python3
"""Validate/replay exported real windows without nuScenes or full cache files.

No quality selection, deployment, threshold search or original weight edits.
RTX3050 timings are local development measurements, NEVER L40S predictions.
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

from real_motion.local_replay_bundle import ReplayBundle, file_digest
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from real_motion.sparse_evidence_repair import SparseRepairHead
from real_motion.source_repair_evidence import build_evidence, map_evidence, compose_repair, oracle_probabilities
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.real_motion.joint_column_full_common import build_fixed_geometry
from tools.real_motion.source_repair_pilot_common import probabilities, sample_pairs
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256
from tools.real_motion.export_p0_f9_local_replay import check_student


def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def replay_one(provider, teacher, head, record, raw, labels, *, train_step=False, chunk=512):
    """All fixed geometry rebuilt from RAW four histories, then all SIX frames.

    Timing labels are explicit: this is sparse REFINE only, NOT old GEN/full
    joint FPS. The training probe has fresh isolated AdamW, GT only/no KD and
    frozen motion; it is NOT full-joint optimizer throughput or a training run.
    """
    device = provider.device; stages = {}; begun = time.perf_counter()
    def call(name, fn):
        synchronize(device); tick = time.perf_counter(); value = fn(); synchronize(device)
        stages[name] = time.perf_counter()-tick
        return value
    with torch.inference_mode():
        geometry = call('cold_fixed_geometry_CPU', lambda: build_fixed_geometry(raw, record,
            provider.pcfg, provider.strong, min(3, provider.workers), teacher.columns.config))
        prepared_raw = {**raw, '_column_causal_preparation': geometry}
        output = call('live_V18_forward', lambda: teacher.motion(record, device))
        prep = call('live_renderer_and_prepare', lambda: provider.prepare_columns(None, record,
            include_gt=False, raw_window=prepared_raw, outputs=output))
        evidence = call('raw_history_evidence_union', lambda: build_evidence(prep, provider.pcfg.grid))
        mapped = call('all_six_live_motion_mapping', lambda: map_evidence(evidence, prep, provider.pcfg.grid))
        call('full_neighbor_features_CPU', lambda: evidence.memory.neighbor_features)
        p = call('sparse_head_all_six', lambda: probabilities(head, evidence, output, device, chunk=chunk))
        predictions = call('all_six_dense_ADD_composition', lambda: [compose_repair(prep.baseline[h],
            evidence, mapped, p, h) for h in range(6)])
    inference_seconds = time.perf_counter()-begun
    # Hashes/GT labels intentionally outside the inference timing boundary.
    changed = sum(int(np.sum(a != b)) for a, b in zip(predictions, prep.baseline))
    protected = all(np.array_equal(a[b != 17], b[b != 17]) for a, b in zip(predictions, prep.baseline))
    result = dict(stages=stages, cold_sparse_refine_seconds=inference_seconds,
        warm_geometry_sparse_refine_seconds_excluding_fixed_rebuild=inference_seconds-stages['cold_fixed_geometry_CPU'],
        changed=changed, original_occupied_unchanged=protected, evidence=evidence.audit,
        source_count=len(record['features']), all_six_frames=True)
    if not protected: raise RuntimeError('replay changed an originally occupied transport voxel')
    if train_step:
        # Independent in-memory probe; original migration optimizer/RNG never restored.
        # Clone source contexts out of inference mode for autograd's saved tensors.
        context = output['history_source_context'].clone(); queries = output['future_transport_queries'].clone()
        head.train(); optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
        rng = np.random.default_rng(20261005); pairs, weights = sample_pairs(mapped, evidence.memory.keys[:, 0], rng)
        if len(pairs):
            ids, inv = np.unique(pairs[:, 0], return_inverse=True)
            targets = oracle_probabilities(evidence, mapped, labels['future_gt_occ'].numpy())[pairs[:, 0], pairs[:, 1]]
            def step():
                optimizer.zero_grad(set_to_none=True)
                m = evidence.memory
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                    logits = head(torch.as_tensor(m.neighbor_features[ids], device=device),
                        torch.as_tensor(m.keys[ids, 0], device=device), torch.as_tensor(m.keys[ids, 1], device=device), context, queries)
                    chosen = logits[torch.as_tensor(inv, device=device), torch.as_tensor(pairs[:, 1], device=device)].float()
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(chosen,
                        torch.as_tensor(targets, device=device), reduction='none')*torch.as_tensor(weights, device=device)).sum()
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm): raise RuntimeError('nonfinite local probe')
                optimizer.step()
                return float(loss.detach().cpu())
            result['isolated_GT_only_head_probe_loss'] = call('isolated_head_backward_optimizer_NO_KD', step)
            result['isolated_head_probe_pairs'] = len(pairs)
        head.eval()
    if device.type == 'cuda':
        result['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/2**20
        result['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device)/2**20
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True); p.add_argument('--validate-only', action='store_true')
    p.add_argument('--device', default='cuda'); p.add_argument('--windows', type=int, default=4)
    p.add_argument('--cpu-workers', type=int, default=2); p.add_argument('--head-chunk', type=int, default=512)
    p.add_argument('--train-probe', action='store_true'); p.add_argument('--out-dir')
    p.add_argument('--trust-repository-checkpoints', action='store_true',
                   help='required for model replay: checkpoints are trusted server pickle artifacts')
    a = p.parse_args()
    if min(a.windows, a.cpu_workers, a.head_chunk) < 1: p.error('positive replay budgets required')
    if not a.validate_only and (not a.out_dir or not a.trust_repository_checkpoints):
        p.error('model replay requires NEW --out-dir and --trust-repository-checkpoints')
    torch.set_num_threads(1); bundle = ReplayBundle(a.bundle)
    try:
        for i in range(len(bundle.manifest['windows'])): bundle.window(i)
        print(f'REPLAY VALIDATED windows={len(bundle.manifest["windows"])} four_history/six_future/full_grid/label_isolation/hashes PASS', flush=True)
        if a.validate_only: return 0
        device = torch.device(a.device)
        if device.type=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError('actual CUDA/BF16 required; no silent CPU fallback')
        out = Path(a.out_dir).resolve()
        if out.exists(): raise RuntimeError('NEW local diagnostic output required')
        out.mkdir(parents=True); report = dict(status='running', actual_cuda=device.type=='cuda', windows=[])
        try:
            paths = {}
            for name in ('runtime.yaml', 'checkpoints/epoch_0019.pt', 'checkpoints/clean_e14.pt', 'checkpoints/migration_last.pt'):
                if name in bundle.manifest['members']:
                    paths[name] = out/Path(name).name; bundle.copy_member(name, paths[name])
            cfg = load_runtime_config(paths['runtime.yaml']); config_sha = stable_json_fingerprint(cfg)
            if config_sha != bundle.manifest['config_fingerprint']: raise RuntimeError('config content mismatch')
            ck, teacher = load_joint(paths['checkpoints/epoch_0019.pt'], device, reference_sha=CLEAN_SHA256,
                                    config_sha=config_sha, allow_diagnostic=True)
            if (file_digest(paths['checkpoints/epoch_0019.pt']) != bundle.manifest['teacher_sha256']
                    or file_digest(paths['checkpoints/clean_e14.pt']) != CLEAN_SHA256
                    or teacher.transport.config.history_frames != 4): raise RuntimeError('checkpoint lineage mismatch')
            teacher.eval().requires_grad_(False)
            head = SparseRepairHead('local_consensus', source_dim=teacher.columns.source_dim).to(device)
            if 'checkpoints/migration_last.pt' in paths:
                saved = torch.load(paths['checkpoints/migration_last.pt'], map_location='cpu', weights_only=False)
                check_student(saved, bundle.manifest['teacher_sha256'], config_sha, head.source_dim)
                head.load_state_dict(saved['head'], strict=True); del saved
            head.eval(); initial_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            provider = PilotProvider(paths['checkpoints/clean_e14.pt'], CLEAN_SHA256, make_prepare_config(cfg),
                                     device, a.cpu_workers, teacher, None)
            # Interleave TRAIN/dev, representative/stress rather than taking only first TRAIN rows.
            groups = defaultdict(list)
            for i, row in enumerate(bundle.manifest['windows']): groups[(row['stratum'], row['split'])].append(i)
            order = [g[rank] for rank in range(max(map(len, groups.values()))) for _, g in sorted(groups.items()) if rank < len(g)]
            for wi, index in enumerate(order[:a.windows], 1):
                record, raw, labels = bundle.window(index, labels=True)
                raw['future_gt_occ'] = None  # labels never enter causal preparation or head inputs
                head.load_state_dict(initial_head)
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                row = replay_one(provider, teacher, head, record, raw, labels,
                                 train_step=a.train_probe, chunk=a.head_chunk)
                row.update(key=bundle.manifest['windows'][index]['key'], split=bundle.manifest['windows'][index]['split'],
                           stratum=bundle.manifest['windows'][index]['stratum'])
                report['windows'].append(row)
                print(f'REPLAY {wi}/{min(a.windows,len(order))} '+json.dumps(row, ensure_ascii=False), flush=True)
                del record, raw, labels, row
            report.update(status='complete', device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                scope='local development only, NOT L40S speed/FPS or real quality validation',
                boundaries='fixed CPU geometry + live V18 + sparse evidence + full neighbors + head + SIX dense refine outputs; excludes I/O/GT/old GEN/metrics',
                train_probe='optional fresh isolated head-only AdamW / frozen motion / GT only / no KD; NOT full joint training throughput')
        except BaseException as error:
            report.update(status='failed', error=type(error).__name__+': '+str(error)); raise
        finally:
            (out/'replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    finally: bundle.close()
    return 0


if __name__ == '__main__': sys.exit(main())
