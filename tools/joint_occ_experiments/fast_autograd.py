"""First-order exact training kernels; no changes to hard inference or loss."""
import torch
from torch.autograd.function import once_differentiable


class _SparseComposition(torch.autograd.Function):
    """One dense gradient buffer, rather than a CopySlices chain per source.

    Every operation has UNIQUE voxel ids internally. Different operations may
    overlap; their original order, including lower-layer REMOVE, is preserved.
    Only touched probabilities are saved. Gradients also reach splat weights.
    """
    @staticmethod
    def forward(ctx, background, operations, *alphas):
        result = background.clone()
        previous = []
        for (kind, ids, cls, destination), alpha in zip(operations, alphas):
            local = result[:, ids]
            if kind == 'overlay':
                previous.append(local)
                value = local*(1-alpha[None])
                value[cls] = value[cls]+alpha
            elif kind == 'remove':
                previous.append(local[cls].clone())
                amount = local[cls].clone()*alpha
                value = local
                value[cls] = value[cls]-amount
                value = value.scatter_add(0, destination[None], amount[None])
            else:
                raise ValueError('unknown sparse composition operation')
            result[:, ids] = value
        ctx.operations = operations
        ctx.save_for_backward(*alphas, *previous)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        operations = ctx.operations
        count = len(operations)
        alphas, previous = ctx.saved_tensors[:count], ctx.saved_tensors[count:]
        gradient = upstream.clone()
        gradients = [None]*count
        for i in range(count-1, -1, -1):
            kind, ids, cls, destination = operations[i]
            local = gradient[:, ids]
            if kind == 'overlay':
                gradients[i] = -(local*previous[i]).sum(0)+local[cls]
                local = local*(1-alphas[i][None])
            else:
                difference = local.gather(0, destination[None])[0]-local[cls]
                gradients[i] = previous[i]*difference
                local[cls] = local[cls]+alphas[i]*difference
            gradient[:, ids] = local
        return (gradient, None, *gradients)


def compose_sparse(background, operations, alphas):
    if len(operations) != len(alphas):
        raise ValueError('sparse operation/alpha count mismatch')
    if not operations:
        return background
    return _SparseComposition.apply(background, tuple(operations), *alphas)


class _Lovasz(torch.autograd.Function):
    """Same full stable sort and Lovasz objective; compact analytic backward."""
    @staticmethod
    def forward(ctx, probability, labels, positions, classes):
        selected = probability[classes[:, None], positions[None]]
        foreground = (labels[None] == classes[:, None]).float()
        delta = selected-foreground
        errors, order = delta.abs().sort(dim=1, descending=True, stable=True)
        sorted_fg = foreground.gather(1, order)
        total = sorted_fg.sum(1, keepdim=True)
        intersection = total-sorted_fg.cumsum(1)
        union = total+(1-sorted_fg).cumsum(1)
        weights = 1-intersection/union.clamp_min(1)
        if weights.shape[1] > 1:
            weights = torch.cat((weights[:, :1], weights[:, 1:]-weights[:, :-1]), 1)
        # sign(0)=0 is important at exact matches/ties, even though CE usually
        # supplies smoothed probabilities strictly inside (0,1).
        coefficients = weights*delta.sign().gather(1, order)/len(classes)
        ctx.shape = probability.shape
        ctx.save_for_backward(positions, classes, order, coefficients)
        return (errors*weights).sum(1).mean()

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        positions, classes, order, coefficients = ctx.saved_tensors
        local = torch.zeros_like(coefficients).scatter_(1, order, coefficients*upstream)
        gradient = local.new_zeros(ctx.shape)
        gradient[classes[:, None], positions[None]] = local
        return gradient, None, None, None


def lovasz_sorted(probability, labels, positions, classes):
    return _Lovasz.apply(probability, labels, positions, classes)


class RenderGradientBridge:
    """Accumulate SIX renderer VJPs, then backprop shared WM/CCR exactly once.

    Temporary leaves break execution, not the chain rule: every accumulated
    derivative is explicitly returned to its live original tensor. Shared
    ancestors and dependent boundary outputs are handled by ONE multi-root
    backward, together with direct auxiliary supervision.
    """
    names = ('poses', 'targets', 'yaws', 'mapped', 'actions')

    def __init__(self, forward):
        self.original = {key: forward[key] for key in self.names}
        self.leaves = {key: value.detach().requires_grad_(value.requires_grad)
                       for key, value in self.original.items()}

    def probe(self, loss, tensors):
        live = [(self.original[key], leaf) for key, leaf in self.leaves.items() if leaf.requires_grad]
        if not loss.requires_grad or not live:
            return [None]*len(tensors)
        gradients = torch.autograd.grad(loss, [leaf for _, leaf in live], retain_graph=True, allow_unused=True)
        roots = [(original, grad) for (original, _), grad in zip(live, gradients) if grad is not None]
        if not roots:
            return [None]*len(tensors)
        return torch.autograd.grad([original for original, _ in roots], tensors,
            grad_outputs=[grad for _, grad in roots], retain_graph=True, allow_unused=True)

    def backward_shared(self, auxiliary):
        roots, gradients = [auxiliary], [None]
        for key, leaf in self.leaves.items():
            if leaf.grad is not None:
                roots.append(self.original[key]); gradients.append(leaf.grad)
        torch.autograd.backward(roots, grad_tensors=gradients)
