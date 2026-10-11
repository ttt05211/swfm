"""Opt-in exact full-grid loss execution; legacy training files are untouched."""
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import torch
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class PreparedTarget:
    volume: int
    positions: object  # None means ALL voxels valid; otherwise ascending indices.
    labels: torch.Tensor
    classes: torch.Tensor
    class_ids: tuple
    class_counts: tuple


def prepare_target(target, device):
    """Supervision ONLY. No target-derived values enter prediction/candidates."""
    if isinstance(target, torch.Tensor):
        if target.device.type != 'cpu': raise ValueError('prepare CPU labels before uploading')
        target = target.detach().numpy()
    flat = np.asarray(target).reshape(-1)
    valid = (flat >= 0) & (flat <= 17)
    all_valid = bool(valid.all())
    labels = np.asarray(flat if all_valid else flat[valid], dtype=np.int64)
    classes = np.flatnonzero(np.bincount(labels, minlength=18)[:17]).astype(np.int64)
    positions = None if all_valid else torch.as_tensor(np.flatnonzero(valid), device=device)
    counts = np.bincount(labels, minlength=18)
    return PreparedTarget(len(flat), positions, torch.as_tensor(labels, device=device),
                          torch.as_tensor(classes, device=device), tuple(map(int, classes)),
                          tuple(int(counts[c]) for c in classes))


def sparse_lovasz(p, prepared, epsilon, normalizer):
    """Exact zero-tail compression, NOT voxel sampling/top-k/approximate sort.

    With nonnegative probabilities <=1, epsilon>0, a raw-zero foreground is
    the maximum-error prefix; raw-zero BACKGROUND is the minimum-error tail.
    After all foreground has appeared, intersection=0 and every tail weight
    is zero, INCLUDING its probability derivative. Prefix gradients remain.
    Variable errors retain their original stable sort and full GT counts.
    """
    classes, positions, labels = prepared.classes, prepared.positions, prepared.labels
    selected = p.index_select(0, classes)
    if positions is not None: selected = selected.index_select(1, positions)
    # General callers may provide signed/non-probability tensors; then the
    # ordering proof is invalid. Keep the exact dense evaluator as fallback.
    if bool(((selected < 0) | (selected >= 1+17*epsilon)).any()): return None
    variable = torch.nonzero(selected != 0)
    if len(variable) > selected.numel()*.4: return None  # dense probabilities do not benefit.
    missing = torch.nonzero((selected == 0) & (labels[None] == classes[:, None]))
    vc = torch.bincount(variable[:, 0], minlength=len(classes)).cpu().tolist()
    mc = torch.bincount(missing[:, 0], minlength=len(classes)).cpu().tolist()
    terms, saved, cursor_v, cursor_m = [], [], 0, 0
    for i, (cls, total) in enumerate(zip(prepared.class_ids, prepared.class_counts)):
        columns = variable[cursor_v:cursor_v+vc[i], 1]
        prefix = missing[cursor_m:cursor_m+mc[i], 1]
        cursor_v += vc[i]; cursor_m += mc[i]
        t = selected.new_tensor(float(total))
        ranks = torch.arange(1, mc[i]+1, device=p.device, dtype=torch.float32)
        head = 1-(t-ranks)/t
        head_weights = torch.cat((head[:1], head[1:]-head[:-1]))
        head_error = 1-selected.new_tensor(epsilon)/normalizer
        term = (head_weights*head_error).sum()
        previous = head[-1:] if mc[i] else selected.new_zeros(1)
        if len(columns):
            foreground = (labels[columns] == cls).float()
            prediction = (selected[i, columns]+epsilon)/normalizer
            delta = prediction-foreground
            errors, order = delta.abs().sort(descending=True, stable=True)
            foreground = foreground[order]
            intersection = t-(mc[i]+foreground.cumsum(0))
            union = t+(1-foreground).cumsum(0)
            jaccard = 1-intersection/union.clamp_min(1)
            weights = torch.cat((jaccard[:1]-previous, jaccard[1:]-jaccard[:-1]))
            term = term+(errors*weights).sum()
            variable_ids = columns[order]
            coefficient = weights*delta.sign()[order]/len(classes)
        else:
            variable_ids, coefficient = columns, selected.new_empty(0)
        ids = torch.cat((prefix, variable_ids))
        if positions is not None: ids = positions[ids]
        saved.extend((ids, torch.cat((-head_weights/len(classes), coefficient))))
        terms.append(term)
    return torch.stack(terms).mean(), saved


class _CombinedLoss(torch.autograd.Function):
    """One full-grid gradient buffer for CE AND Lovasz; same stable sort.

    Only required class/valid rows are normalized, rather than a separate
    full 18-class probability volume. All valid voxels still participate.
    """
    @staticmethod
    def forward(ctx, probability, prepared, epsilon, sparse):
        if probability.ndim != 2 or probability.shape != (18, prepared.volume):
            raise ValueError('expected 18-class full-grid probabilities')
        p = probability.float()
        labels, positions, classes = prepared.labels, prepared.positions, prepared.classes
        normalizer = 1+18*epsilon
        ce_probability = ((p.gather(0, labels[None])[0] if positions is None else p[labels, positions])
                          +epsilon)/normalizer
        ce = -ce_probability.log().mean() if len(labels) else p.new_zeros(())
        compressed = (sparse_lovasz(p, prepared, epsilon, normalizer)
                      if sparse and epsilon > 0 and len(classes) and len(labels) else None)
        ctx.sparse = compressed is not None
        if compressed is not None:
            lovasz, saved = compressed
            order = labels.new_empty(0); coefficients = p.new_empty(0)
        elif len(classes) and len(labels):
            selected = p.index_select(0, classes)
            if positions is not None: selected = selected.index_select(1, positions)
            selected = (selected+epsilon)/normalizer
            foreground = (labels[None] == classes[:, None]).float()
            delta = selected-foreground
            errors, order = delta.abs().sort(dim=1, descending=True, stable=True)
            sorted_fg = foreground.gather(1, order)
            total = sorted_fg.sum(1, keepdim=True)
            weights = 1-(total-sorted_fg.cumsum(1))/(total+(1-sorted_fg).cumsum(1)).clamp_min(1)
            if weights.shape[1] > 1:
                weights = torch.cat((weights[:, :1], weights[:, 1:]-weights[:, :-1]), 1)
            lovasz = (errors*weights).sum(1).mean()
            coefficients = weights*delta.sign().gather(1, order)/len(classes)
        else:
            lovasz = p.new_zeros(())
            order = labels.new_empty((0, len(labels)))
            coefficients = p.new_empty((0, len(labels)))
        if compressed is None: saved = []
        ctx.prepared = prepared
        ctx.dtype = probability.dtype
        ctx.normalizer = normalizer
        ctx.save_for_backward(ce_probability, order, coefficients, *saved)
        ctx.set_materialize_grads(False)
        return ce, lovasz

    @staticmethod
    @once_differentiable
    def backward(ctx, ce_upstream, lovasz_upstream):
        prepared = ctx.prepared
        ce_probability, order, coefficients = ctx.saved_tensors[:3]
        labels, positions, classes = prepared.labels, prepared.positions, prepared.classes
        gradient = ce_probability.new_zeros((18, prepared.volume))
        if lovasz_upstream is not None and ctx.sparse:
            sparse_saved = ctx.saved_tensors[3:]
            for cls, ids, coefficient in zip(prepared.class_ids, sparse_saved[::2], sparse_saved[1::2]):
                gradient[cls, ids] = coefficient*lovasz_upstream
        elif lovasz_upstream is not None and len(coefficients):
            local = torch.zeros_like(coefficients).scatter_(
                1, order, coefficients*lovasz_upstream)
            if positions is None:
                gradient.index_copy_(0, classes, local)
            else:
                gradient[classes[:, None], positions[None]] = local
        if ce_upstream is not None and len(labels):
            local = (-ce_upstream/len(labels))/ce_probability
            if positions is None:
                gradient.scatter_add_(0, labels[None], local[None])
            else:
                gradient[labels, positions] = gradient[labels, positions]+local
        # Match the reference graph: FIRST accumulate CE + Lovasz derivatives
        # with respect to normalized probabilities, THEN undo normalization.
        # Dividing the two terms separately changes FP32 rounding and can
        # become visible after several BF16 network/Adam updates.
        gradient.div_(ctx.normalizer)
        return gradient.to(ctx.dtype), None, None, None


def occupancy_loss_prepared(probability, prepared, *, lovasz_weight=1., epsilon=1e-4, sparse=False):
    ce, lovasz = _CombinedLoss.apply(probability, prepared, epsilon, sparse)
    return ce+lovasz_weight*lovasz, ce, lovasz


@contextmanager
def prepared_loss_execution(rows, device, *, sparse=False, audit=None):
    """Single-thread training/probe opt-in; no changes to the legacy contract.

    Preparation is inside the caller's elapsed time. The old engine still
    builds the SAME predictions; only its loss evaluator is temporarily bound.
    No prepared labels or indices are retained after this batch.
    """
    from .. import engine
    previous = engine.occupancy_loss
    prepared = [prepare_target(raw['future_gt_occ'][h], device)
                for _, raw, *_ in rows for h in range(6)]
    cursor = 0
    def loss(probability, target, *, lovasz_weight=1., epsilon=1e-4, class_ids=None, reference=False):
        nonlocal cursor
        if reference: raise ValueError('prepared loss requires the fused training backend')
        if cursor >= len(prepared): raise RuntimeError('unexpected horizon loss order')
        result = occupancy_loss_prepared(probability, prepared[cursor], lovasz_weight=lovasz_weight,
                                         epsilon=epsilon, sparse=sparse)
        if audit is not None and cursor == 1:
            # One REAL horizon from the first batch, OUTSIDE throughput timing.
            # Detached copies isolate the loss implementation from the network
            # graph and BF16 parameter-gradient accumulation. No model mutation.
            p = probability.detach().float().requires_grad_()
            old = previous(p, target, lovasz_weight=lovasz_weight, epsilon=epsilon,
                           class_ids=class_ids, reference=False)
            new = occupancy_loss_prepared(p, prepared[cursor], lovasz_weight=lovasz_weight,
                                          epsilon=epsilon, sparse=sparse)
            a = torch.autograd.grad(old[0], p)[0]
            b = torch.autograd.grad(new[0], p)[0]
            close = torch.isclose(a, b, rtol=3e-5, atol=2e-7)
            values_close = all(bool(torch.isclose(x, y, rtol=2e-6, atol=3e-7))
                               for x, y in zip(old, new))
            audit.update(loss_values_match=values_close,
                         gradient_numerical_parity=bool(close.all()),
                         pass_gate=values_close and bool(close.all()),
                         max_gradient_abs_difference=float((a-b).abs().max()),
                         mismatched_gradient_elements=int((~close).sum()),
                         compared_gradient_elements=a.numel(),
                         byte_equal_gradients=bool(torch.equal(a, b)),
                         budgets=dict(loss_rtol=2e-6, loss_atol=3e-7,
                                      gradient_rtol=3e-5, gradient_atol=2e-7))
        cursor += 1
        return result
    engine.occupancy_loss = loss
    try:
        yield
        if cursor != len(prepared): raise RuntimeError('not all six horizon losses were evaluated')
    finally:
        engine.occupancy_loss = previous
