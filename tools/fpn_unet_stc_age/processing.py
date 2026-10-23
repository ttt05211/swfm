"""Frozen old-to-new memory weighting; no occupancy edits or motion changes."""
import numpy as np
import torch
from tools.source_scene_ego_conditioned.geometry import aligned_memory

HALF_LIFE_SECONDS = .5


def weight_history(history, times):
    times = np.asarray(times, np.float64)
    if times.shape != (4,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('FOUR increasing actual historical timestamps required')
    weights = np.exp2((times-times[-1])/HALF_LIFE_SECONDS).astype(np.float32)
    causal = dict(history[0]); reads = dict(causal['memory_reads'])
    frames = np.asarray(reads['frame'])
    if frames.dtype.kind not in 'iu' or np.any((frames < 0)|(frames >= 4)):
        raise ValueError('dynamic memory frame IDs must be 0..3')
    # Weight BOTH feature numerators and denominators via the original fusion's
    # static support/dynamic coverage. Never attenuate lifted features alone.
    reads['static_valid'] = np.asarray(reads['static_valid'])*weights[:, None, None, None]
    reads['coverage'] = np.asarray(reads['coverage'])*weights[frames]
    causal['memory_reads'] = reads
    return (causal, *history[1:]), weights


@torch.no_grad()
def candidate(model, provider, row, baseline):
    _, raw, _, history = row
    changed, weights = weight_history(history, raw['_shared_source_track'].times)
    memory = aligned_memory(model.columns, baseline['lifted'], baseline['output'], changed, raw,
        baseline['targets'], baseline['yaws'], baseline['poses'], provider.pcfg.grid)
    logits = model.columns(memory, baseline['prior'], range(6), baseline['shape'])
    return logits, weights
