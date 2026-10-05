"""Explicit migration-only recovery; never interpret an old full checkpoint."""
import torch
from real_motion.shared_column_evidence import PROTOCOL
from real_motion.v21_source_induction import stable_json_fingerprint


def migration_payload(student,optimizer,generator,contract,*,role,cursor,successful,executed):
    return dict(protocol=PROTOCOL,contract=contract,role=role,
        student=student.state_dict(),optimizer=optimizer.state_dict(),sampling_rng=generator.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        cursor=cursor,successful_updates=successful,executed_windows=executed,
        transport_frozen=True,deployable=False)


def validate_migration(saved,contract):
    if (saved.get('protocol')!=PROTOCOL or not saved.get('transport_frozen') or saved.get('deployable')
            or stable_json_fingerprint(saved.get('contract'))!=stable_json_fingerprint(contract)):
        raise RuntimeError('ONLY identical new migration contract can resume; old full optimizer rejected')
    cursor=saved['cursor'];successful=saved['successful_updates'];executed=saved['executed_windows']
    if (not all(isinstance(v,int) and not isinstance(v,bool) for v in (cursor,successful,executed))
            or not 0<=successful<=cursor<=contract['schedule_steps'] or executed<0):
        raise RuntimeError('invalid migration recovery cursor')
    return cursor,successful,executed


def restore_migration(saved,student,optimizer,generator,contract):
    cursor,successful,executed=validate_migration(saved,contract)
    student.load_state_dict(saved['student'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    generator.set_state(saved['sampling_rng'].cpu());torch.set_rng_state(saved['torch_rng'].cpu())
    if saved['cuda_rng']:
        if not torch.cuda.is_available():raise RuntimeError('CUDA migration cannot resume on CPU')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return cursor,successful,executed
