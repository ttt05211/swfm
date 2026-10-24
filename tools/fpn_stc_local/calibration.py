"""Two shared scalars, not per-class/region/time-specific rules."""
import numpy as np
import torch
from tools.final_semantic_evaluation import metrics

BASELINE = (1., 0.)
CANDIDATES = (BASELINE,)+tuple((g,b) for g in (.75,1.,1.25,1.5) for b in (-.25,0.,.25) if (g,b)!=BASELINE)


def name(candidate):
    gain,bias=candidate
    return f'gain{gain:g}_free{bias:+g}'


@torch.no_grad()
def predict(logits,prior_logits,shape,*,candidates=CANDIDATES):
    """All predictions precede target reads. Identity uses original logits.

    Z = L_prior + gain*(Z_original-L_prior); add bias to FREE logit only.
    All occupied categories share the same adjustment; no semantic class list.
    """
    hs=(1,3,5); original=logits[list(hs)]; prior=prior_logits[list(hs)]
    residual=original-prior; results={}
    for gain in dict.fromkeys(c[0] for c in candidates):
        z=original if gain==1. else original+(gain-1)*residual
        semantic,classes=z[:,:17].max(1)
        for bias in [c[1] for c in candidates if c[0]==gain]:
            # FREE is class17, so a tie goes to an occupied class, exactly as
            # argmax over the original class order (0..17).
            pred=torch.where(semantic>=z[:,17]+bias,classes,17)
            results[name((gain,bias))]=pred.reshape(3,*shape).byte().cpu().numpy()
    return results


def counts(predictions,targets,*,names=None):
    rows=[]
    for key in names or [name(c) for c in CANDIDATES]:
        c=metrics.empty(3)
        for h in range(3):metrics.add(c,h,predictions[key][h],targets[h])
        rows.append(c['confusion'])
    return np.stack(rows)


def report(confusion):
    return metrics.report(dict(confusion=confusion,moving=np.zeros_like(confusion)),(1.,2.,3.),moving=False)


def summarize(window_counts,*,names=None,parameters=None):
    names=names or [name(c) for c in CANDIDATES]
    parameters=parameters or [dict(zip(('residual_gain','free_bias'),c)) for c in CANDIDATES]
    values=np.asarray(window_counts)
    if values.shape!=(64,len(names),3,18,18) or values.dtype.kind not in 'iu' or np.any(values<0):
        raise ValueError('exact64 window integer confusion arrays required')
    fit=values[:32].sum(0); held=values[32:].sum(0)
    fit_reports=[report(c) for c in fit]; base=fit_reports[0]['average']
    # Selection reads FIRST32 only; prioritize simultaneous metric gains.
    # The baseline is included, so a negative minimum gain cannot win.
    def objective(index):
        row=fit_reports[index]['average']; d=[row[k]-base[k] for k in ('mIoU','IoU')]
        return min(d),sum(d),-index
    selected=max(range(len(names)),key=objective)
    held_reports={names[i]:report(held[i]) for i in dict.fromkeys((0,selected))}
    all_reports={names[i]:report(fit[i]+held[i]) for i in dict.fromkeys((0,selected))}
    old=held_reports[names[0]]['average']; new=held_reports[names[selected]]['average']
    return dict(selected_index=selected,selected=parameters[selected],
        selection='maximize min(delta_mIoU,delta_IoU) on FIRST32 ONLY; tie=sum then fixed order',
        fit32={n:r for n,r in zip(names,fit_reports)},held32=held_reports,
        held32_delta={k:new[k]-old[k] for k in ('mIoU','IoU')},
        all64_descriptive_includes_selection=all_reports,
        held32_both_improve=selected!=0 and new['mIoU']>old['mIoU'] and new['IoU']>old['IoU'])
