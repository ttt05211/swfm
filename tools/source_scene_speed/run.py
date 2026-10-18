"""Read-only parameter/training-log report + matched single-GPU H6 FPS."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.source_scene_stage1.model import checkpoint_model
from tools.source_scene_stage1.history import StageHistory
from tools.shared_source_track.adapter import SharedTrackPredictor
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .engine import aggregate, fresh, parameters, prepare, sync
from .report import pair, summary


def load_pair(root):
    records = pair(root); saved = {}
    for arm, value in records.items():
        path = Path(root)/arm/'last.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state['contract'] != value['contract'] or state['updates'] != value['contract']['total_updates']
                or state['epoch'] != value['contract']['epochs'] or state['batch'] != 0):
            raise RuntimeError('completed checkpoint/results differ: '+arm)
        model = checkpoint_model(state).requires_grad_(False)
        value['checkpoint_sha256'] = base.sha256(path)
        value['parameters'] = parameters(model)
        if value['parameters']['total'] != value['contract']['parameters']: raise RuntimeError('parameter count differs')
        saved[arm] = model
    return records, saved


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'out', 'config', 'dev-cache', 'dev-info', 'dataroot'): ap.add_argument('--'+name, required=True)
    ap.add_argument('--stc-root'); ap.add_argument('--modality', choices=('gt', 'stc', 'both'), default='both')
    ap.add_argument('--windows', type=int, default=64); ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--cpu-workers', type=int, default=4)
    a = ap.parse_args(argv)
    if not 1 <= a.windows <= 64 or not 1 <= a.repeats <= 4 or not 1 <= a.cpu_workers <= 4:
        raise ValueError('bounded 1..64 windows / 1..4 repeats / 1..4 CPU workers')
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(): raise RuntimeError('ONE visible BF16 GPU required')
    if a.modality != 'gt' and not a.stc_root: raise ValueError('STC historical data root required')
    torch.set_num_threads(1); device = torch.device('cuda:0'); out = Path(a.out).resolve()
    # New folder only; partial timings are retained but never represented as a
    # complete sample. No changes to original arm directories or fingerprints.
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.parent/('.'+out.name+'.lock'); lock.mkdir(exist_ok=True)
    with evaluation_lock(lock):
        if out.exists(): raise RuntimeError('NEW speed output required')
        arms, models = load_pair(a.root); c = arms['baseline']['contract']
        repo = Path(__file__).resolve().parents[2]
        for name, expected in c['implementation'].items():
            if base.sha256(repo/name) != expected: raise RuntimeError('source implementation changed: '+name)
        for name in ('config', 'dev_cache', 'dev_info'):
            if base.sha256(getattr(a, name)) != c['data'][name]: raise RuntimeError('source data/config changed: '+name)
        pcfg = base.make_prepare_config(base.load_runtime_config(a.config, c['overrides']))
        _, all_dev = base.load_cache(a.dev_cache)
        keys = c['eval_keys'][:a.windows]; jobs = base.align_records(all_dev, keys)
        modalities = ('gt', 'stc') if a.modality == 'both' else (a.modality,)
        inputs = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=128)
        out.mkdir(); samples = {arm: {kind: [] for kind in modalities} for arm in arms}
        result = dict(status='running', arms=arms, scientific_updates=0, keys=keys,
            GPU=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
            cpu_workers=a.cpu_workers, repeats=a.repeats, modality=a.modality,
            implementation={str(p.name): base.sha256(p) for p in Path(__file__).parent.glob('*.py')})
        for arm in arms: arms[arm]['fps'] = {}
        def store():
            base.write_json(out/'results.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
        store()
        # Alternating complete passes; only one body on GPU at a time. H2D of
        # weights / preparation / warmup is excluded, exactly as declared.
        try:
            for pass_index in range(a.repeats):
                order = ('baseline', 'reader') if pass_index % 2 == 0 else ('reader', 'baseline')
                for arm in order:
                    model = models[arm].to(device).eval()
                    runner = SharedTrackPredictor(model, pcfg, device, workers=a.cpu_workers)
                    runner.history = StageHistory(runner.provider, arm=arm, ram_mib=0)
                    try:
                        for index, job in enumerate(jobs):
                            w = window(job)
                            for kind in modalities if index % 2 == 0 else modalities[::-1]:
                                record, historical, times = inputs.history(w, 'occ' if kind == 'gt' else 'stc')
                                sem = np.asarray(historical['history_occ'])
                                historical = dict(historical, history_observed=(sem >= 0)&(sem < 18))
                                poses = np.asarray(inputs.poses(w))
                                row = runner.prepare(record, historical, times=times)
                                ready = prepare(row, poses, device)
                                # Reference full live entrance, unmodified model;
                                # keep its own repeat drift as well as cross-path.
                                fw = runner.forward(row, poses=poses)
                                reference = fw['logits'].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy(); del fw
                                fw = runner.forward(row, poses=poses)
                                repeat = fw['logits'].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy(); del fw
                                for _ in range(3 if index == 0 else 1): fresh(model, runner.provider, ready)
                                sync(device); torch.cuda.reset_peak_memory_stats(device)
                                tick = time.perf_counter(); pred = fresh(model, runner.provider, ready); sync(device)
                                elapsed = time.perf_counter()-tick
                                item = dict(pass_index=pass_index, index=index, key=keys[index], seconds=elapsed,
                                    changed_voxels=int(np.count_nonzero(pred != reference)),
                                    reference_repeat_changed_voxels=int(np.count_nonzero(reference != repeat)),
                                    peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                                    peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20, profile={})
                                if pass_index == 0 and index < 4:
                                    fresh(model, runner.provider, ready, profile=item['profile'])
                                samples[arm][kind].append(item)
                                with (out/'samples.jsonl').open('a', encoding='utf-8') as f:
                                    f.write(json.dumps(dict(arm=arm, modality=kind, **item))+'\n')
                                del ready, row, reference, repeat, pred
                            if (index+1) % 8 == 0: print(f'SCENE_SPEED pass={pass_index+1}/{a.repeats} arm={arm} windows={index+1}/{len(jobs)}', flush=True)
                    finally:
                        runner.close(); del runner; model.cpu(); torch.cuda.empty_cache()
            for arm in arms:
                for kind in modalities: arms[arm]['fps'][kind] = aggregate(samples[arm][kind], len(jobs)*a.repeats)
                if base.sha256(Path(a.root)/arm/'last.pt') != arms[arm]['checkpoint_sha256']:
                    raise RuntimeError('source checkpoint changed during measurement')
            result['status'] = 'complete'; store(); print(summary(result), flush=True)
        except Exception as error:
            result.update(status='failed', error=repr(error)); store(); raise


if __name__ == '__main__': main()
