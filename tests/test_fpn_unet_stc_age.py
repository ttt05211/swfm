"""A tiny processing change: fixed timestamps, no in-place edits, frozen forward."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.fpn_unet_stc_age import processing, run
from tools.fpn_unet_eval.common import FPNPredictor


def test_age_weights_actual_seconds_copy_only_memory_reliability():
    arrays = dict(static_valid=np.ones((4,2,2,2),np.float32), frame=np.array([0,1,2,3]),
        coverage=np.ones(4,np.float32),world=np.ones((4,3)),cell=np.arange(4))
    history=(dict(memory_reads=arrays,state=object(),shared_reads=object()),None)
    original=copy.deepcopy(arrays)
    changed,weights=processing.weight_history(history,[100.,100.5,101.,101.5])
    np.testing.assert_array_equal(weights,[.125,.25,.5,1.])
    np.testing.assert_array_equal(changed[0]['memory_reads']['coverage'],weights)
    np.testing.assert_array_equal(changed[0]['memory_reads']['static_valid'][:,0,0,0],weights)
    assert changed[0]['state'] is history[0]['state'] and changed[0]['shared_reads'] is history[0]['shared_reads']
    assert changed[0]['memory_reads']['world'] is arrays['world']
    for name,value in arrays.items():np.testing.assert_array_equal(value,original[name])
    _,irregular=processing.weight_history(history,[0.,.25,.75,1.5])
    np.testing.assert_allclose(irregular,np.exp2((np.array([0.,.25,.75,1.5])-1.5)/.5))
    with pytest.raises(ValueError):processing.weight_history(history,[0.,0.,.5,1.])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_frozen_pair_preserves_prior_motion_source_and_parameters(device):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):pytest.skip('BF16 CUDA')
    from test_fpn_unet import fixture
    body,old,row=fixture(device);pcfg=old.provider.pcfg;old.close()
    body.eval().requires_grad_(False)
    with torch.no_grad():body.columns.head.weight.normal_(std=.02)
    predictor=FPNPredictor(body,pcfg,device,workers=1)
    try:
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        prepared=predictor.prepare(row[0],raw,times=np.arange(4)*.5)
        with torch.no_grad():fw=predictor.forward(prepared,poses=row[1]['future_poses'])
        prior=fw['prior'].clone();targets=fw['targets'].clone()
        state={k:v.clone() for k,v in body.state_dict().items()}
        original={k:np.array(v,copy=True) for k,v in prepared[3][0]['memory_reads'].items()}
        # Future target data are explicitly poisoned: processing can use only
        # historical reliability and the already-conditioned forecast tensors.
        poison=prepared[0],dict(prepared[1],future_gt_occ=object()),prepared[2],prepared[3]
        logits,weights=processing.candidate(body,predictor.provider,poison,fw)
        assert logits.shape==fw['logits'].shape and torch.isfinite(logits).all()
        torch.testing.assert_close(fw['prior'],prior,rtol=0,atol=0)
        torch.testing.assert_close(fw['targets'],targets,rtol=0,atol=0)
        np.testing.assert_array_equal(weights,[.125,.25,.5,1.])
        for name,v in original.items():np.testing.assert_array_equal(prepared[3][0]['memory_reads'][name],v)
        for name,v in state.items():torch.testing.assert_close(body.state_dict()[name],v,rtol=0,atol=0)
        assert not any(p.grad is not None for p in body.parameters())
    finally:predictor.close()


def test_saved_integer_counts_prefix_and_summary_include_negative_delta():
    from tools.final_semantic_evaluation import metrics
    keys=[['VAL','one']];c=dict(protocol=run.PROTOCOL)
    counts={n:metrics.empty(3) for n in run.NAMES}
    for name in run.NAMES:counts[name]['confusion'][:,4,4]=8
    counts['age/semantic_refine']['confusion'][:,4,4]=4
    counts['age/semantic_refine']['confusion'][:,4,17]=4
    state=dict(contract=c,keys=keys,counts={n:{k:v.tolist() for k,v in f.items()} for n,f in counts.items()})
    restored,done=run.restore(state,c,keys,8)
    assert done==keys
    report={n:metrics.report(v,(1.,2.,3.)) for n,v in restored.items()}
    text=run.summary(dict(status='complete',keys=keys,reports=report))
    assert 'age_minus_baseline' in text and '-50.0' in text
    with pytest.raises(RuntimeError):run.restore(state,c,[['OTHER','key']],8)
    bad=copy.deepcopy(state);bad['counts'][run.NAMES[0]]['confusion'][0][4][4]-=1
    with pytest.raises(RuntimeError):run.restore(bad,c,keys,8)
