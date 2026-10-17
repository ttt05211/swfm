"""Discarded real-history H6 plumbing; optional two steps on ORIGINAL TRAIN."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.stc_overnight.bundle import Bundle
from tools.shared_source_track.adapter import SharedTrackPredictor
from .history import StageHistory
from .model import make_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True); p.add_argument('--out', required=True)
    p.add_argument('--windows', type=int, default=2); p.add_argument('--device', default='cuda')
    p.add_argument('--train-updates', type=int, choices=(0, 1, 2), default=0)
    a = p.parse_args(); out = Path(a.out)
    if out.exists(): raise RuntimeError('new replay report path required')
    if not 1 <= a.windows <= 8: raise ValueError('bounded plumbing check only')
    if a.train_updates and a.windows != 1: raise ValueError('one TRAIN window only for discarded step check')
    torch.set_num_threads(1); bundle = Bundle(a.bundle, cache_mib=32)
    # Architecture ONLY: saved trained tensors never enter either fresh model.
    archived, saved, _ = bundle.load_model('dense_epoch2', 'cpu')
    config = archived.transport.config; del archived, saved
    role = 'clean_train' if a.train_updates else 'reference64'
    records = [r for r in bundle.records if r['role'] == role][:a.windows]
    if len(records) != a.windows: raise RuntimeError('required original TRAIN/check windows absent')
    reports = []
    for arm in ('baseline', 'reader'):
        torch.manual_seed(20261011)
        model = make_model(arm, config).to(a.device).eval()
        runner = SharedTrackPredictor(model, bundle.pcfg, a.device, workers=1)
        runner.history = StageHistory(runner.provider, arm=arm, ram_mib=0)
        try:
            for record in records:
                rec, raw, times = bundle.history(record, kind='gt', policy='dense_valid')
                tick = time.perf_counter()
                row = runner.prepare(rec, raw, times=times)
                with torch.no_grad(): fw = runner.forward(row, poses=np.asarray(record['future_poses']))
                torch.cuda.synchronize() if torch.device(a.device).type == 'cuda' else None
                assert tuple(fw['logits'].shape) == tuple(fw['prior'].shape)
                assert len(fw['logits']) == 6 and bool(torch.isfinite(fw['logits']).all())
                assert not bool(fw['output']['residual_xy_m'].any())
                assert torch.equal(fw['logits'].argmax(1), fw['prior'].argmax(1))
                item = dict(arm=arm, scene=record['scene_name'], t0=record['t0_token'],
                    sources=len(row[0]['source_class_id']), finite_six_frame=True, initial_KTA=True,
                    initial_transport_argmax=True, whole_plumbing_seconds=time.perf_counter()-tick,
                    memory_shape=list(fw['memory'].shape), parameters=sum(v.numel() for v in model.parameters()))
                if a.train_updates:
                    from tools.shared_history_experiments.full_gt.engine import train_arm
                    historical = dict(row[1], future_poses=np.asarray(record['future_poses']),
                                      future_gt_occ=bundle.targets(record, indices=range(6)))
                    historical['_column_causal_preparation']['prepared_state']['column_backgrounds'] = fw['state']['column_backgrounds']
                    train_row = row[0], historical, row[2], row[3]
                    del fw
                    optimizer = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=5e-4, initial_lr=5e-4, weight_decay=1e-4),
                        dict(params=model.columns.parameters(), lr=3e-4, initial_lr=3e-4, weight_decay=.01)])
                    item['discarded_TRAIN_steps'] = []
                    for update in range(1, a.train_updates+1):
                        stats = train_arm(model, optimizer, runner.provider, [train_row], 'gt', update=update, total_updates=2)
                        item['discarded_TRAIN_steps'].append({k: stats[k] for k in ('CE', 'Lovasz', 'grad_norm', 'seconds', 'peak_memory_mib')})
                    del optimizer, train_row
                else: del fw
                reports.append(item)
                del row
        except torch.OutOfMemoryError as error:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(dict(status='failed_resources', arm=arm,
                discarded_optimizer_updates_requested=a.train_updates, scientific_updates_published=0,
                error=str(error), note='Resource failure retained; no smaller horizons/loss fallback or checkpoint written.'), indent=2), encoding='utf-8')
            raise
        finally: runner.close()
        del runner, model; gc.collect()
        if torch.device(a.device).type == 'cuda': torch.cuda.empty_cache()
    bundle.close(); out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='complete', rows=reports, scientific_updates_published=0,
                  discarded_optimizer_updates_each=a.train_updates, quality_scores=False,
                  FPS=False, note='Small real-history plumbing only; trained archive used for architecture, not weights. Optional optimizer steps use ORIGINAL TRAIN only and are discarded, no checkpoints.')
    out.write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
