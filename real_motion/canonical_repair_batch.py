"""Window-batched canonical repair, SAME per-window objective and source link.

History/points are not shared across unrelated windows. Only row-independent
point MLP execution is batched. Actor offsets index the matching live motion
latents. Geometry remains hard/non-differentiable, source features do not.
Floating-point GEMM batching can change rounding; this is not a byte-exact
optimizer-trajectory promise. Inference/chunking/thresholds remain unchanged.
"""
import numpy as np
import torch
from .canonical_causal_repair import repair_loss


def batched_repair_losses(head,evidences,plans,output,source_sizes,targets,weights,device):
    count=len(evidences)
    if not (count==len(plans)==len(source_sizes)==len(targets)==len(weights)) or not count:
        raise ValueError('nonempty equal window lists required')
    if sum(source_sizes)!=len(output['history_source_context']):
        raise ValueError('live source population mismatch')
    actors=[];offset=0
    for e,n in zip(evidences,source_sizes):
        a=e.actor.astype(np.int64,copy=True);dynamic=a>=0
        if np.any(a[dynamic]>=n):raise ValueError('local actor exceeds its source population')
        a[dynamic]+=offset;offset+=n;actors.append(a)
    lengths=[len(e) for e in evidences]
    def upload(values):
        return torch.as_tensor(np.ascontiguousarray(np.concatenate(values)),device=device)
    actor=upload(actors)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        live=head.project_sources(output)
        encoded=head.encode(upload([e.features for e in evidences]),upload([e.labels for e in evidences]),
                            actor,upload([e.classes for e in evidences]),live)
        logits=head.decode(encoded,actor,upload([p.context for p in plans]),upload([p.base for p in plans]),
                           upload([p.fallback for p in plans]),upload([p.legal for p in plans]),live)
        y=upload(targets);w=upload(weights);cursor=0;losses=[]
        for n in lengths:
            sl=slice(cursor,cursor+n);cursor+=n
            losses.append(repair_loss(head,logits[sl],actor[sl],y[sl],w[sl]))
    return losses
