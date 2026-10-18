"""Real portable histories, RANDOM full-grid H6 plumbing only; no optimization."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tools.stc_overnight.bundle import Bundle
from .model import make_model
from .adapter import DirectPredictor


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True); p.add_argument('--out', required=True)
    p.add_argument('--windows', type=int, default=2); p.add_argument('--device', default='cuda')
    a = p.parse_args(); out = Path(a.out)
    if out.exists() or not 1 <= a.windows <= 4: raise ValueError('NEW report / bounded 1..4 windows')
    torch.set_num_threads(1); bundle = Bundle(a.bundle, cache_mib=32)
    archived, saved, _ = bundle.load_model('dense_epoch2', 'cpu')
    config = archived.transport.config; del archived, saved
    # Trained archive is consulted for architecture ONLY, not its weights.
    torch.manual_seed(20261011); model = make_model(config).to(a.device).eval()
    runner = DirectPredictor(model, bundle.pcfg, a.device, workers=1); reports = []
    def forbidden(*args, **kwargs): raise AssertionError('ICP unexpectedly called')
    try:
        records = [r for r in bundle.records if r['role'] == 'reference64'][:a.windows]
        if len(records) != a.windows: raise ValueError('requested bounded real histories absent')
        with patch('real_motion.shared_source_track.register_history_shape', forbidden), patch('real_motion.source_evidence_audit.register_history_shape', forbidden):
            for record in records:
                for modality in ('gt', 'stc'):
                    rec, raw, times = bundle.history(record, kind=modality, policy='dense_valid')
                    row = runner.prepare(rec, raw, times=times)
                    fw = runner.forward(row, poses=np.asarray(record['future_poses']))
                    assert bool(torch.isfinite(fw['logits']).all()) and len(fw['logits']) == 6
                    assert tuple(fw['logits'].shape) == tuple(fw['prior'].shape)
                    assert not bool(fw['output']['residual_xy_m'].any())
                    assert not bool(fw['output']['history_yaw_to_t0_rad'].any())
                    assert torch.equal(fw['logits'].argmax(1), fw['prior'].argmax(1))
                    reports.append(dict(modality=modality, key=[rec['scene_name'], rec['t0_token']],
                        sources=len(row[0]['source_class_id']), history_tokens=int(row[1]['_shared_source_track'].track_valid.sum()),
                        full6_finite=True, initial_KTA=True, initial_history_yaw_zero=True,
                        initial_transport_argmax=True, memory_shape=list(fw['memory'].shape)))
                    del fw, row
        result = dict(status='complete', parameters=sum(v.numel() for v in model.parameters()),
            rows=reports, scientific_updates=0, quality_scores=False, FPS=False,
            note='Actual full-grid GT/STC H6 no-ICP plumbing only; no target occupancy read or optimizer.')
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
    finally: runner.close(); bundle.close()


if __name__ == '__main__': main()
