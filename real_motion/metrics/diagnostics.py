"""Post-training WM-vs-KTA diagnostics. Never used as an inference oracle."""
import torch


def harm_repair_counts(kta, wm, gt, support):
    m=support.bool()
    kc=(kta==gt)&m; wc=(wm==gt)&m
    return {
        "repair":int((~kc&wc&m).sum()),
        "harm":int((kc&~wc&m).sum()),
        "preserve":int((kc&wc&m).sum()),
        "unresolved":int((~kc&~wc&m).sum()),
        "support":int(m.sum()),
    }


def harm_repair_regions(kta, wm, gt, region_masks):
    """Instance/tube-level macro diagnosis.

    A region is repair/harm if exact-label accuracy changes strictly in the
    corresponding direction. This prevents large buses from dominating a pure
    voxel micro statistic.
    """
    rows=[]
    for i,m in enumerate(region_masks):
        m=m.bool()
        n=int(m.sum())
        if n==0: continue
        ka=float(((kta==gt)&m).sum())/n
        wa=float(((wm==gt)&m).sum())/n
        rows.append({"region":i,"kta_accuracy":ka,"wm_accuracy":wa,
                     "repair":wa>ka,"harm":wa<ka,"delta":wa-ka})
    if not rows:
        return {"regions":[],"repair_rate":0.0,"harm_rate":0.0,"mean_delta":0.0}
    return {
        "regions":rows,
        "repair_rate":sum(r["repair"] for r in rows)/len(rows),
        "harm_rate":sum(r["harm"] for r in rows)/len(rows),
        "mean_delta":sum(r["delta"] for r in rows)/len(rows),
    }


def oracle_selector(kta,wm,gt,support):
    """Analysis-only selector: choose WM where it is correct, else retain KTA."""
    out=kta.clone(); choose=(wm==gt)&support.bool(); out[choose]=wm[choose]; return out
