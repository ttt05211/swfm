"""Strict completed-update recovery for the GT-only shared-field screen."""
import copy
import torch
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.height_field_screen_common import PROTOCOL


def validate_cursor(contract, epoch, batch, updates, executed):
    counts = contract['epoch_batches']; sizes = contract['epoch_batch_sizes']
    values = (epoch, batch, updates, executed)
    if any(type(v) is not int for v in values) or not 0 <= epoch <= len(counts):
        raise RuntimeError('invalid shared-field recovery cursor')
    limit = counts[epoch] if epoch < len(counts) else 0
    if not 0 <= batch <= limit:
        raise RuntimeError('invalid epoch batch cursor')
    expected_updates = sum(counts[:epoch])+batch
    expected_windows = sum(sum(s) for s in sizes[:epoch])+sum(sizes[epoch][:batch] if epoch < len(sizes) else [])
    if updates != expected_updates or executed != expected_windows or updates > contract['schedule_steps']:
        raise RuntimeError('checkpoint counters disagree with fixed population/order')


def payload(head, optimizer, rng, contract, *, epoch, batch, updates, executed, reports, protocol=PROTOCOL):
    validate_cursor(contract, epoch, batch, updates, executed)
    return dict(protocol=protocol, contract=contract, head=head.state_dict(), optimizer=optimizer.state_dict(),
                numpy_rng=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                epoch=epoch, batch=batch, updates=updates, executed=executed, reports=copy.deepcopy(reports),
                transport_frozen=True, deployable=False)


def restore(saved, head, optimizer, rng, contract, *, protocol=PROTOCOL):
    if (saved.get('protocol') != protocol or saved.get('transport_frozen') is not True
            or saved.get('deployable') is not False
            or stable_json_fingerprint(saved.get('contract')) != stable_json_fingerprint(contract)):
        raise RuntimeError('only identical shared-field screen resumes; old Local optimizer/budget changes rejected')
    validate_cursor(contract, *[saved[k] for k in ('epoch', 'batch', 'updates', 'executed')])
    if not isinstance(saved.get('reports'), dict) or 'train_prior' not in saved['reports']:
        raise RuntimeError('missing completed TRAIN-only calibration')
    head.load_state_dict(saved['head'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'].cpu())
    if saved['cuda_rng']:
        if not torch.cuda.is_available():
            raise RuntimeError('cannot restore CUDA screen RNG on CPU')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return tuple(saved[k] for k in ('epoch', 'batch', 'updates', 'executed')), copy.deepcopy(saved['reports'])


def restore_execution_upgrade(saved, head, optimizer, rng, contract, *, protocol=PROTOCOL):
    """Resume identical scientific training after execution-only code/settings changes.

    Optimizer/RNG/cursor/population/schedule are preserved. Only explicitly
    whitelisted performance metadata may differ.
    """
    if saved.get('protocol') != protocol or saved.get('transport_frozen') is not True or saved.get('deployable') is not False:
        raise RuntimeError('invalid execution-upgrade checkpoint role/protocol')
    old=copy.deepcopy(saved.get('contract'))
    new=copy.deepcopy(contract)
    if not isinstance(old,dict) or not isinstance(new,dict):
        raise RuntimeError('missing execution-upgrade contracts')
    # These fields describe how identical math is executed, not what is trained.
    execution_keys={
        'implementation','ccr_implementation','torch_version',
        'cpu_execution','cpu_workers','batched_head','batched_motion',
        'fast_train','lazy_sampled_features','prefetch_workers',
        'motion_superbatch_updates','motion_streams'
    }
    for key in execution_keys:
        old.pop(key,None);new.pop(key,None)
    if stable_json_fingerprint(old)!=stable_json_fingerprint(new):
        raise RuntimeError('execution-upgrade resume changed scientific training contract')
    validate_cursor(contract,*[saved[k] for k in ('epoch','batch','updates','executed')])
    if not isinstance(saved.get('reports'),dict) or 'train_prior' not in saved['reports']:
        raise RuntimeError('missing completed TRAIN-only calibration')
    head.load_state_dict(saved['head'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state=saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'].cpu())
    if saved['cuda_rng']:
        if not torch.cuda.is_available():
            raise RuntimeError('cannot restore CUDA screen RNG on CPU')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    reports=copy.deepcopy(saved['reports'])
    reports.setdefault('execution_upgrades',[]).append(dict(
        from_contract_fingerprint=stable_json_fingerprint(saved.get('contract')),
        to_contract_fingerprint=stable_json_fingerprint(contract),
        preserved='head+optimizer+numpy_rng+torch_rng+cuda_rng+cursor+population+schedule'))
    return tuple(saved[k] for k in ('epoch','batch','updates','executed')),reports
