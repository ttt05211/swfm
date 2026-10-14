"""Execution-only overlay; old scientific contracts and original Adam/RNG restored."""
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.stc_noise_adaptation import run as training
from tools.joint_occ_experiments import run as legacy
from .pipeline import prefetch_adapter, noise_execution, CheckedStrong, PROTOCOL


def main(event=None, argv=None):
    event = event or threading.Event(); a = training.parser().parse_args(argv)
    out = Path(a.out_dir).resolve()
    ledger = dict(protocol=PROTOCOL, source_update=None, scientific_contract_modified=False,
        Adam_RNG_schedule_reset=False, loss_model_noise_recipe_changed=False,
        CUDA_in_worker=False, bounded_augmented_lookahead_batches=3,
        implementation={f.name: legacy.sha256(f) for f in Path(__file__).parent.glob('*.py')})
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        if saved['contract']['protocol'] != training.PROTOCOL: raise RuntimeError('wrong adaptation resume')
        ledger.update(source_update=saved['updates'], source_checkpoint_sha256=legacy.sha256(a.resume))
    def write(): legacy.write_json(out/'execution_speed_resume.json', ledger)
    def record(audit): ledger['Strong_reuse_audit'] = audit; write()
    before = training.prefetch_adapter, training.finish_backgrounds
    try:
        training.prefetch_adapter = lambda *args, **kw: prefetch_adapter(*args, **kw, stop_event=event)
        training.finish_backgrounds = CheckedStrong(record)
        with noise_execution():
            print('EXECUTION OVERLAY: next-batch CPU lookahead + reuse prepared Strong match + exact RNG dead-field skip.', flush=True)
            status = training.main(event, argv)
        ledger['status'] = 'returned_at_safe_boundary'; write()
        return status
    except BaseException as error:
        ledger.update(status='failed_no_automatic_retry', error=repr(error))
        if out.exists(): write()
        raise
    finally:
        training.prefetch_adapter, training.finish_backgrounds = before


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
