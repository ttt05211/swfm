"""Reuse completed integer quality counts, NEVER old inference timing."""
import json
from pathlib import Path
import numpy as np
from .. import run as legacy
from ..diagnostics.run import restore_metrics
from .engine import ROUTES

QUALITY_PROTOCOLS = {'joint_occ_frozen_soft_hard_dev512_paired_inference_v1',
                     'joint_occ_frozen_soft_hard_dev512_paired_inference_v2',
                     'joint_occ_frozen_soft_hard_dev512_paired_inference_v3'}


def completed_quality(directory, contract=None):
    directory = Path(directory).resolve()
    read = lambda name: json.loads((directory/name).read_text(encoding='utf-8'))
    old, state, result = read('contract.json'), read('state.json'), read('evaluation.json')
    if (old.get('protocol') not in QUALITY_PROTOCOLS or result.get('status') != 'complete'
            or state.get('windows') != 512 or result.get('windows') != 512
            or result.get('scientific_updates') != 0 or result.get('contract') != old
            or len(old.get('dev_keys', [])) != 512
            or len({tuple(k) for k in old['dev_keys']}) != 512
            or old.get('thresholds') != [.5, .95]
            or old.get('quality_routes') != list(ROUTES)
            or legacy.sha256(directory/'source_snapshot.pt') != old['source_sha256']):
        raise RuntimeError('completed compatible frozen DEV512 quality required')
    metrics = restore_metrics(state.get('metric_counts'))
    if set(metrics) != set(ROUTES) or set(result.get('metrics', {})) != set(ROUTES):
        raise RuntimeError('all SIX frozen quality readouts required')
    for name, metric in metrics.items():
        computed = metric.compute()
        for key in ('mIoU', 'IoU', 'MovingMicro'):
            if not np.isclose(computed[key], result['metrics'][name][key], rtol=0, atol=1e-10, equal_nan=True):
                raise RuntimeError('quality integer counts do not reproduce result: '+name+'/'+key)
    if contract is not None:
        for key in ('source_sha256', 'source_epochs', 'source_updates', 'model_digest', 'dev_keys',
                    'thresholds', 'quality_routes', 'source_implementation', 'data', 'cache_namespace'):
            if old.get(key) != contract.get(key): raise RuntimeError('reused quality contract differs: '+key)
        for key in ('config_sha256', 'overrides'):
            if old['runtime'].get(key) != contract['runtime'].get(key):
                raise RuntimeError('reused quality runtime differs: '+key)
    receipt = dict(directory=str(directory), protocol=old['protocol'], windows=512,
        files={name: legacy.sha256(directory/name) for name in
               ('contract.json', 'state.json', 'evaluation.json', 'source_snapshot.pt')},
        old_timing_discarded=True)
    # Deliberately drop old speed_windows. Start boundary-v2 timing from zero.
    imported = dict(windows=512, metric_counts=state['metric_counts'], speed_windows=[],
        quality_seconds=state['quality_seconds'], input_wait_seconds=state['input_wait_seconds'])
    return old, imported, receipt


def discover_completed(root):
    """Newest completed soft DEV512 receipt, NOT a best-score model selector."""
    candidates = sorted(Path(root).glob('soft_dev512_*/evaluation.json'),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for candidate in candidates:
        try:
            completed_quality(candidate.parent)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            continue
        return str(candidate.parent.resolve())
    raise RuntimeError('no completed soft_dev512 directory; set SOFT_RETIME_FROM to the completed evaluation directory')
