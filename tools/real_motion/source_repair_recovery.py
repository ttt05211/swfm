"""Strict sparse-head migration recovery, never resumes the old optimizer."""
import torch
from real_motion.source_repair_evidence import PROTOCOL
from real_motion.v21_source_induction import stable_json_fingerprint


def payload(head, optimizer, rng, contract, *, cursor, successful, executed):
    return dict(protocol=PROTOCOL,contract=contract,head=head.state_dict(),optimizer=optimizer.state_dict(),
        numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        cursor=cursor,successful=successful,executed=executed,transport_frozen=True,deployable=False)


def restore(saved, head, optimizer, rng, contract):
    if (saved.get('protocol')!=PROTOCOL or saved.get('transport_frozen') is not True or saved.get('deployable') is not False
            or stable_json_fingerprint(saved.get('contract'))!=stable_json_fingerprint(contract)):
        raise RuntimeError('only identical sparse-head migration contract may resume; old optimizer rejected')
    values=[saved[k] for k in ('cursor','successful','executed')]
    if (any(not isinstance(v,int) or isinstance(v,bool) for v in values)
            or not 0<=values[1]<=values[0]<=contract['schedule_steps'] or values[2]<0):
        raise RuntimeError('invalid sparse recovery cursor')
    head.load_state_dict(saved['head'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu())
    if saved['cuda_rng']:
        if not torch.cuda.is_available():raise RuntimeError('CUDA migration cannot resume on CPU')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return values
