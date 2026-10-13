"""Locate same-pose logits drift; full BF16 repeats + identical-input/FP32 controls."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from tools.shared_history_experiments.full_gt import engine as gt
from tools.scene_ego_head.diagnostics.engine import external_forward


def difference(a, b, *, rtol=2e-4, atol=2e-5):
    if a.shape != b.shape: return dict(pass_gate=False, shape_a=list(a.shape), shape_b=list(b.shape))
    if a.device != b.device: b = b.to(a.device)
    x, y = (a.double(), b.double()) if a.dtype == torch.float64 else (a.float(), b.float())
    delta = (x-y).abs(); bad = delta > atol+rtol*y.abs()
    finite = bool(torch.isfinite(x).all() & torch.isfinite(y).all())
    return dict(pass_gate=finite and not bool(bad.any()), max_abs=float(delta.max()) if delta.numel() else 0.,
        mismatched_elements=int(bad.sum()), compared_elements=a.numel(), dtype_a=str(a.dtype), dtype_b=str(b.dtype),
        stride_a=list(a.stride()), stride_b=list(b.stride()), rtol=rtol, atol=atol)


def semantics(a, b):
    pa, pb = a.argmax(1), b.argmax(1); changed = pa != pb
    details = []
    for h, cell in changed.nonzero()[:8].tolist():
        va = a[h, :, cell].float().topk(2).values
        vb = b[h, :, cell].float().topk(2).values
        details.append(dict(horizon_seconds=(h+1)*.5, flat_cell=cell, class_a=int(pa[h, cell]),
            class_b=int(pb[h, cell]), margin_a=float(va[0]-va[1]), margin_b=float(vb[0]-vb[1])))
    return dict(changed_voxels=int(changed.sum()), voxels=pa.numel(),
        per_horizon_changed=changed.sum(1).cpu().tolist(), first_changes=details)


def stages(a, b):
    values = {k: difference(a[k], b[k]) for k in ('field', 'poses', 'targets', 'yaws', 'prior', 'memory', 'logits')}
    for k in ('residual_xy_m', 'yaw_delta_rad', 'future_transport_queries'):
        values['motion/'+k] = difference(a['output'][k], b['output'][k])
    order = ['field', 'motion/residual_xy_m', 'motion/yaw_delta_rad', 'motion/future_transport_queries',
             'poses', 'targets', 'yaws', 'prior', 'memory', 'logits']
    return dict(fields=values, earliest_numerical_FAIL=next((k for k in order if not values[k]['pass_gate']), None),
                semantic=semantics(a['logits'], b['logits']))


@contextmanager
def fp32_reference():
    # Only this diagnostic Refiner call executes within the scoped override.
    # No weights/dtypes/configs are modified; always restore even on exceptions.
    original = torch.autocast
    matmul_tf32, conv_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    def disabled(*args, **kwargs):
        kwargs['enabled'] = False
        return original(*args, **kwargs)
    torch.autocast = disabled
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try: yield
    finally:
        torch.autocast = original
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = conv_tf32
        torch.set_float32_matmul_precision(matmul_precision)


def readout(columns, memory, prior, shape, *, fp32=False, trace=False):
    values = {}; handles = []
    selected = dict(input=columns.input, time=columns.time,
        **{f'blocks.{i}': b for i, b in enumerate(columns.blocks)},
        **{f'readout.{i}': columns.readout[i] for i in (0, 2, 4)})
    def hook(name):
        def capture(module, inputs, output):
            # Trace trunk and only FIRST point chunk: bounded CPU memory.
            if name not in values: values[name] = output.detach().cpu().clone()
        return capture
    try:
        if trace:
            handles = [module.register_forward_hook(hook(name)) for name, module in selected.items()]
        if fp32:
            with fp32_reference(): result = columns(memory, prior, range(6), shape)
        else: result = columns(memory, prior, range(6), shape)
        return result, values
    finally:
        for handle in handles: handle.remove()


def trace_difference(a, b):
    if a.keys() != b.keys(): raise RuntimeError('Refiner trace calls differ')
    result = {k: difference(a[k], b[k]) for k in a}
    return dict(fields=result, first_exact_difference=next((k for k in result if result[k]['max_abs'] != 0), None),
                earliest_numerical_FAIL=next((k for k in result if not result[k]['pass_gate']), None),
                point_trace_scope='first point chunk only; ALL output logits compared separately')


@torch.no_grad()
def audit(body, provider, row):
    body.eval(); a = gt.forward_window(body, provider, row)
    b = external_forward(body, provider, row, row[1]['future_poses'])
    background = a['state']['column_backgrounds']
    changed = [int((torch.as_tensor(x) != torch.as_tensor(y)).sum())
               for x, y in zip(background, b['state']['column_backgrounds'])]
    report = dict(key=[row[0]['scene_name'], row[0]['t0_token']], windows=1,
        background_changed_voxels_by_horizon=changed, cross_entrance=stages(a, b),
        strict_FAIL_retained=True, no_training=True, no_GT_alignment_score=True, not_FPS=True,
        original_precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            convolution_allow_tf32=torch.backends.cudnn.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision()), FP32_reference_TF32_disabled=True)
    again = gt.forward_window(body, provider, row)
    report['GT_original_repeat'] = stages(a, again); del again
    again = external_forward(body, provider, row, row[1]['future_poses'])
    report['external_original_repeat'] = stages(b, again); del again
    # Exact SAME raw inputs, rather than two geometry results that are only close.
    common1, trace1 = readout(body.columns, a['memory'], a['prior'], a['shape'], trace=True)
    common2, trace2 = readout(body.columns, a['memory'], a['prior'], a['shape'], trace=True)
    report['identical_input_BF16_repeat'] = dict(logits=difference(common1, common2),
        semantic=semantics(common1, common2), trace=trace_difference(trace1, trace2))
    del common2, trace2
    own, trace2 = readout(body.columns, b['memory'], b['prior'], b['shape'], trace=True)
    report['cross_input_BF16_refiner'] = dict(logits=difference(common1, own),
        semantic=semantics(common1, own), trace=trace_difference(trace1, trace2))
    del common1, own, trace1, trace2
    coarse = lambda v: F.avg_pool3d(v.reshape(6, 18, *a['shape']), (2, 2, 1))
    report['BF16_input_quantization'] = {
        'memory_changed_elements': int((a['memory'].bfloat16() != b['memory'].bfloat16()).sum()),
        'coarse_prior_changed_elements': int((coarse(a['prior']).bfloat16() != coarse(b['prior']).bfloat16()).sum()),
        'point_prior_changed_elements': int((a['prior'].bfloat16() != b['prior'].bfloat16()).sum())}
    fpa, _ = readout(body.columns, a['memory'], a['prior'], a['shape'], fp32=True)
    fpb, _ = readout(body.columns, b['memory'], b['prior'], b['shape'], fp32=True)
    report['FP32_cross_input_reference'] = dict(logits=difference(fpa, fpb), semantic=semantics(fpa, fpb))
    report['original_vs_FP32'] = dict(logits=difference(a['logits'], fpa), semantic=semantics(a['logits'], fpa))
    report['automatic_precision_switch'] = False
    return report
