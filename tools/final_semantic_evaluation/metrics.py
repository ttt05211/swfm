import numpy as np
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS


def empty(horizons):
    return dict(confusion=np.zeros((horizons, 18, 18), np.int64),
                moving=np.zeros((horizons, 18, 18), np.int64))


def add(counts, h, pred, gt, moving=None):
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.shape != gt.shape or any(a.dtype.kind not in 'ui' or np.any(a > 17) or np.any(a < 0) for a in (pred, gt)):
        raise ValueError('full-grid metric requires integer labels 0..17 without silent masks/remapping')
    code = 18*gt.ravel().astype(np.int64)+pred.ravel()
    counts['confusion'][h] += np.bincount(code, minlength=324).reshape(18, 18)
    if moving is not None:
        mask = np.asarray(moving, bool)
        if mask.shape != gt.shape: raise ValueError('moving support grid differs')
        counts['moving'][h] += np.bincount(code[mask.ravel()], minlength=324).reshape(18, 18)


def ratio(a, b):
    return np.divide(100*np.asarray(a, float), b, out=np.full(np.shape(a), np.nan), where=np.asarray(b) > 0)


def number(v): return float(v) if np.isfinite(v) else None
def mean(v):
    v = np.asarray(v, float); valid = v[np.isfinite(v)]
    return float(valid.mean()) if len(valid) else None


def report(counts, labels, *, moving=True):
    rows = {}; values = []
    for h, name in enumerate(labels):
        c = counts['confusion'][h]; tp = np.diag(c); u = c.sum(0)+c.sum(1)-tp
        sem = ratio(tp[:17], u[:17]); occ_i = c[:17, :17].sum(); occ_u = c.sum()-c[17, 17]
        row = dict(IoU=number(ratio(np.asarray(occ_i), np.asarray(occ_u))), mIoU=mean(sem),
            semantic_per_class={str(i): number(v) for i, v in enumerate(sem)})
        if moving:
            mc = counts['moving'][h]; ids = np.asarray(DYNAMIC_CLASS_IDS)
            mi = np.diag(mc)[ids]; mu = (mc.sum(0)+mc.sum(1)-np.diag(mc))[ids]
            row.update(MovingMicro=number(ratio(np.asarray(mi.sum()), np.asarray(mu.sum()))), MovingMacro=mean(ratio(mi, mu)))
        rows[str(name)] = row; values.append(row)
    keys = ('IoU', 'mIoU', 'MovingMicro', 'MovingMacro') if moving else ('IoU', 'mIoU')
    result = dict(per_horizon=rows, average={k: mean([r[k] if r[k] is not None else np.nan for r in values]) for k in keys})
    if len(labels) == 6:
        for name, part in (('average_1s_2s_3s', values[:3]), ('average_4s_5s_6s', values[3:])):
            result[name] = {k: mean([r[k] if r[k] is not None else np.nan for r in part]) for k in keys}
    return result
