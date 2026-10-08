"""Frozen Point-CCR B decision rule.

The learned checkpoint/support are unchanged. ADD uses raw weighted logits
through sigmoid at 0.5; REMOVE is disabled. This is algebraically equivalent
to thresholding the existing corrected ADD probability at 1/(1+w_role), but
avoids doing correction-and-inversion in the timed deployment path.
"""
from __future__ import annotations

import numpy as np
import torch

REMOVE_DISABLED_SCORE = 0.0


def tensor(value, device):
    return torch.as_tensor(np.ascontiguousarray(value), device=device)


@torch.no_grad()
def frozen_b_probabilities(head, evidence, plan, output, device, *, chunk=8192):
    """Point-CCR forward for the frozen B decision rule.

    Returns [N,6,2] scores for existing composition:
    channel 0 = sigmoid(raw weighted ADD logit)
    channel 1 = 0, so REMOVE never passes the existing 0.95 gate.
    """
    result=[]
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        live=head.project_sources(output)
        shared=(head.encode_queries(
                    evidence,evidence.neighbor_graph,np.arange(len(evidence)),live,device)
                if hasattr(head,"encode_queries") and len(evidence) else None)
        for start in range(0,len(evidence),chunk):
            sl=slice(start,start+chunk)
            batch=(head.inference_batch(live,evidence.actor[sl])
                   if hasattr(head,'inference_batch') else live)
            actor=tensor(evidence.actor[sl],device)
            encoded=(shared[sl] if shared is not None else
                     head.encode(
                         tensor(evidence.features[sl],device),
                         tensor(evidence.labels[sl],device),
                         actor,tensor(evidence.classes[sl],device),batch))
            logits=head.decode(
                encoded,actor,tensor(plan.context[sl],device),
                tensor(plan.base[sl],device),tensor(plan.fallback[sl],device),
                tensor(plan.legal[sl],device),batch)
            score=torch.zeros_like(logits,dtype=torch.float32)
            score[...,0]=torch.sigmoid(logits[...,0].float())
            result.append(score.cpu().numpy())
    return np.concatenate(result) if result else np.empty((0,6,2),np.float32)


def effective_corrected_add_thresholds(head):
    """Exact corrected-score thresholds corresponding to raw sigmoid ADD@0.5."""
    w=np.asarray(head.positive_weight.detach().cpu(),np.float64)
    if w.shape!=(2,2) or not np.isfinite(w).all() or np.any(w<1):
        raise RuntimeError("invalid Point-CCR positive_weight")
    return {
        "static_ADD_weight":float(w[0,0]),
        "dynamic_ADD_weight":float(w[1,0]),
        "static_corrected_threshold":float(1.0/(1.0+w[0,0])),
        "dynamic_corrected_threshold":float(1.0/(1.0+w[1,0])),
    }
