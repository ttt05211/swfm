"""ONE real TRAIN window execution check; never a quality/development score."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.stc_overnight.bundle import Bundle, prepare_config
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.joint_occ_experiments.soft_geometry import grid_arrays
from .model import make_model
from .adapter import forward_window
from .projection import VolumeProjection
from .run import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--spec-archive', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path); p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); device = torch.device(a.device); torch.set_num_threads(1)
    with zipfile.ZipFile(a.spec_archive) as z:
        spec = torch.load(io.BytesIO(z.read('epoch_0002.pt')), map_location='cpu', weights_only=False)['contract']['model']
        pcfg = prepare_config(json.loads(z.read('manifest.json'))['prepare_config'])
    torch.manual_seed(20261011)
    model = make_model(config_from_mapping_v17(spec['motion']), feature_width=spec['shared_history']['width'],
        z_bins=spec['shared_history']['z_bins'], memory_width=spec['repair']['memory_width'],
        reader_config=ReaderConfig(**spec['reader'])).to(device).eval().requires_grad_(False)
    bundle = Bundle(a.bundle, cache_mib=32); predictor = RecoveryPredictor(model, pcfg, device, workers=2)
    try:
        case = next(r for r in bundle.records if r['split'] == 'train')
        rec, raw, times = bundle.history(case, 'gt', 'dense_valid')
        row = predictor.prepare(rec, raw, times=times)
        with torch.no_grad(): fw = forward_window(model, predictor.provider, row, poses=case['future_poses'])
        if not bool(torch.isfinite(fw['logits']).all()): raise RuntimeError('nonfinite native-grid output')
        shape = fw['shape']; native_shape = list(fw['logits'].shape)
        unknown = float((fw['prior_coverage'] < .01).float().mean())
        memory_support = float(fw['memory_coverage'].float().mean())
        # All SIX predictions already exist; GT only enters this supervision check.
        labels = bundle.targets(case, (5,))[0]
        leaf = fw['logits'][5].detach().clone().requires_grad_(); del fw
        projection = VolumeProjection(*grid_arrays(pcfg.grid))
        loss, sem, depth = projection.loss(leaf, labels)
        loss.backward()
        receipt = dict(scope='RANDOM untrained ONE real TRAIN window; execution sanity ONLY, NOT quality/FPS',
            trained_model_weights_imported=False, scene=case['scene_name'], t0=case['t0_token'],
            grid=list(shape), logits_shape=native_shape, parameters=sum(p.numel() for p in model.parameters()),
            unknown_prior_fraction=unknown, memory_supported_fraction=memory_support,
            projection_loss=float(loss.detach()), semantic=float(sem.detach()), depth=float(depth.detach()),
            finite_projection_gradient=bool(torch.isfinite(leaf.grad).all()),
            nonzero_projection_gradient=int(leaf.grad.count_nonzero()),
            peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None)
        a.out.parent.mkdir(parents=True, exist_ok=True); write_json(a.out, receipt); print(json.dumps(receipt, indent=2))
    finally: predictor.close(); bundle.close()


if __name__ == '__main__': main()
