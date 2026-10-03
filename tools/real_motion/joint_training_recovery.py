"""I/O and diagnostic-only recovery helpers; never alter the training recipe."""
from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch


def save_resume_checkpoint(path, value):
    """Flush a new complete checkpoint, retain one old last, then atomic publish.

    No in-place writes, no background serialization of live optimizer tensors.
    A failure before publication leaves last.pt readable. Only files belonging
    to this run's checkpoint writer are replaced; no historical runs touched.
    """
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    previous_tmp = None
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(value, handle)
            handle.flush(); os.fsync(handle.fileno())
        if path.exists():
            backup = path.with_name(path.stem+'.previous'+path.suffix)
            bfd, previous_tmp = tempfile.mkstemp(prefix=backup.name+'.', suffix='.tmp', dir=path.parent)
            os.close(bfd); os.unlink(previous_tmp)
            try: os.link(path, previous_tmp)
            except OSError: shutil.copyfile(path, previous_tmp)
            os.replace(previous_tmp, backup); previous_tmp = None
        os.replace(temporary, path)
    finally:
        for draft in (temporary, previous_tmp):
            if draft is not None and Path(draft).exists(): Path(draft).unlink()


def snapshot_checkpoint(source, destination):
    """Copy ONE opened inode, accepting atomic last.pt replacement by a writer.

    Path rehashing after an evaluation wrongly rejects a valid immutable
    snapshot when training publishes another last.pt. Hash the copied stream
    instead; reject an in-place writer, never modify the source checkpoint.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists(): raise RuntimeError('NEW diagnostic snapshot required')
    digest = hashlib.sha256(); created = False
    try:
        with source.open('rb') as original:
            before = os.fstat(original.fileno())
            with destination.open('xb') as frozen:
                created = True
                while True:
                    block = original.read(1024*1024)
                    if not block: break
                    digest.update(block); frozen.write(block)
                frozen.flush(); os.fsync(frozen.fileno())
            after = os.fstat(original.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('checkpoint changed IN PLACE while snapshotting; stop its writer first')
        return digest.hexdigest()
    except BaseException:
        if created and destination.exists(): destination.unlink()
        raise


@contextmanager
def preserve_training_rng(sampling_rng=None):
    """Evaluation/calibration never advances training RNG, even on interruption."""
    state = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             random.getstate(), np.random.get_state(),
             copy.deepcopy(sampling_rng.bit_generator.state) if sampling_rng is not None else None)
    try: yield
    finally:
        torch.set_rng_state(state[0])
        if state[1]: torch.cuda.set_rng_state_all(state[1])
        random.setstate(state[2]); np.random.set_state(state[3])
        if sampling_rng is not None: sampling_rng.bit_generator.state = state[4]


def process_start_token(pid):
    """Linux PID reuse protection; None off Linux or when the process is gone."""
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text()
        return stat.rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError): return None


def stop_requested(event): return event is not None and event.is_set()


def validate_prior_resume(checkpoint, prior_windows):
    complete = checkpoint.get('prior_completed', True)  # older full last.pt was written AFTER prior
    cursor = checkpoint.get('prior_cursor', prior_windows if complete else 0)
    counts = checkpoint.get('prior_counts', {'generation': [0., 0.], 'refine': [0., 0., 0.]})
    counts = {k: np.asarray(counts[k], np.float64).copy() for k in ('generation', 'refine')}
    if (type(complete) is not bool or type(cursor) is not int or not 0 <= cursor <= prior_windows
            or complete and cursor != prior_windows
            or not complete and checkpoint['attempted_updates'] != 0
            or counts['generation'].shape != (2,) or counts['refine'].shape != (3,)
            or any(not np.isfinite(v).all() or np.any(v < 0) for v in counts.values())):
        raise RuntimeError('invalid prior resume state')
    return complete, cursor, counts
