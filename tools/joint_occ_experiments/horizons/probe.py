"""Read-only A/B: original sequential networks, one vs two render horizons."""
from pathlib import Path
from .. import run as legacy
from ..performance import probe as base
from ..batching import probe as audit
from .engine import train_arm

BACKENDS = ('current_fused', 'sequential_h2')


def execution_fingerprint(root, group=2):
    if group != 2: raise ValueError('fixed two-horizon execution')
    files = list(Path(__file__).parent.glob('*.py'))
    files += [Path(audit.__file__), Path(audit.__file__).with_name('geometry.py'),
              Path(audit.__file__).with_name('run.py'), Path(base.__file__)]
    sources = {str(p.relative_to(root).as_posix()): legacy.sha256(p) for p in sorted(files)}
    return legacy.stable_json_fingerprint(dict(base=base.legacy_fingerprint(root), sources=sources,
        protocol='sequential_WM_CCR_two_horizons_GT_v1'))


def measure(model, optimizer, provider, saved, batches, *, passes=2, progress=print):
    # Reuse established read-only reset/audit/timing/gates, not packed forward.
    def step(*args, horizon_group, projection_pool, **kwargs):
        if horizon_group != 2: raise ValueError('fixed two horizons')
        return train_arm(*args, **kwargs)
    previous = audit.BACKENDS, audit.train_arm, audit.execution_fingerprint
    try:
        audit.BACKENDS, audit.train_arm, audit.execution_fingerprint = BACKENDS, step, execution_fingerprint
        result = audit.measure(model, optimizer, provider, saved, batches, passes=passes, progress=progress)
        result.update(projection_workers=0, network_execution='original single-window forward AND shared backward',
                      horizon_group=2, scope='GT full forward/backward/Adam+targets; shared input I/O excluded')
        return result
    finally:
        audit.BACKENDS, audit.train_arm, audit.execution_fingerprint = previous


def main(argv=None):
    previous = base.measure, base.BACKENDS
    try:
        base.measure, base.BACKENDS = measure, BACKENDS
        return base.main(argv)
    finally:
        base.measure, base.BACKENDS = previous


if __name__ == '__main__': main()
