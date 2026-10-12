"""Frozen train-mode forward replay: F -> Q -> P_T -> M -> logits."""
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.refine import engine
from .execution import BACKENDS,execution

STAGES=('F','Q','Transport_prior','memory','logits')


def compare(left,right):
    result={};first_exact=first_numeric=None
    for name in STAGES:
        a,b=left[name],right[name]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise RuntimeError('stage shape/dtype differs: '+name)
        if not bool(torch.isfinite(b).all()):raise RuntimeError('nonfinite frozen stage: '+name)
        exact=torch.equal(a,b);af,bf=a.float(),b.float()
        close=torch.isclose(af,bf,rtol=2e-4,atol=2e-5)
        numeric=bool(close.all())
        result[name]=dict(exact=exact,numerical_parity=numeric,max_abs=float((af-bf).abs().max()) if a.numel() else 0.,
            mismatched_elements=int((~close).sum()),elements=a.numel(),dtype=str(a.dtype))
        if not exact and first_exact is None:first_exact=name
        if not numeric and first_numeric is None:first_numeric=name
    result.update(first_exact_difference=first_exact,first_numerical_failure=first_numeric,
        changed_semantic_voxels=int((left['logits'].argmax(1)!=right['logits'].argmax(1)).sum()),
        budgets=dict(rtol=2e-4,atol=2e-5))
    return result


@torch.no_grad()
def audit(model,provider,saved,row,repeats=3,stop_event=None,backends=BACKENDS):
    if repeats<2:raise ValueError('at least two frozen reference replays')
    original_mode=model.training;flags=[p.requires_grad for p in model.parameters()]
    def replay(backend):
        if stop_event is not None and stop_event.is_set():raise InterruptedError('frozen stage audit interrupted')
        model.load_state_dict(saved['models']['gt']);legacy.restore_rng(saved['rng'])
        model.train().requires_grad_(True)  # same train-mode dispatch, despite no_grad audit
        with execution(backend):fw=engine.forward_window(model,provider,row)
        tensors=dict(F=fw['field'],Q=fw['output']['future_transport_queries'],
            Transport_prior=fw['prior'],memory=fw['memory'],logits=fw['logits'])
        return {k:v.detach().cpu().clone() for k,v in tensors.items()}
    try:
        first=replay('reference')
        references=[compare(first,replay('reference')) for _ in range(repeats-1)]
        candidates={};failed={}
        for backend in backends[1:]:
            try:candidates[backend]=compare(first,replay(backend))
            except Exception as error:
                # Compiler failures only: NEVER hide shape/NaN/numeric defects.
                if backend=='staged_compose' and type(error).__module__.startswith(('torch._dynamo','torch._inductor')):
                    failed[backend]=repr(error)
                else:raise
        return dict(protocol='frozen_stage_replay_v1',scientific_updates=0,reference_replays=references,
            candidates=candidates,failed=failed,stage_order=list(STAGES),
            scope='first TRAIN window, frozen weights+same RNG, original train-mode forward; NOT a whole-population exactness proof')
    finally:
        model.load_state_dict(saved['models']['gt']);legacy.restore_rng(saved['rng']);model.train(original_mode)
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)
