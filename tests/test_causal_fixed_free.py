"""Fixed inherited calibration, exact ties, unchanged network/loss and no sweep."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from tools.causal_forward_train import calibration,evaluation,run


def test_fixed_FREE_minus1_matches_full_argmax_without_mutation_or_extra_coefficients():
    logits = torch.full((6,18,8),-100.,dtype=torch.float32)
    logits[:,0] = 0.;logits[:,1] = 0.
    logits[:,17] = torch.tensor([1.,1.01,.5,-.1,5.,0.,-.5,2.])
    before = logits.clone();values = calibration.predict(logits,(2,2,2))
    expected = logits.clone();expected[:,17] -= 1
    np.testing.assert_array_equal(values['semantic_refine'],logits.argmax(1).reshape(6,2,2,2).byte().numpy())
    np.testing.assert_array_equal(values['semantic_refine_free_minus1'],expected.argmax(1).reshape(6,2,2,2).byte().numpy())
    torch.testing.assert_close(logits,before,rtol=0,atol=0)
    assert values['semantic_refine_free_minus1'].reshape(6,-1)[0,0]==0  # occupied wins exact tie.
    old,new = (values[k].reshape(6,-1) for k in ('semantic_refine','semantic_refine_free_minus1'))
    np.testing.assert_array_equal(new[old!=17],old[old!=17])
    assert calibration.CALIBRATION['free_logit_bias']==-1. and not calibration.CALIBRATION['training_loss_changed']
    with pytest.raises(ValueError):calibration.predict(logits.bfloat16(),(2,2,2))


def test_eval_restore_rejects_old_readouts_and_calibration_changes():
    state = dict(protocol=evaluation.PROTOCOL,calibration=copy.deepcopy(calibration.CALIBRATION),
        windows=0,keys=[],metrics={n:{k:v.tolist() for k,v in vars(evaluation.gt.Metrics()).items()} for n in evaluation.NAMES})
    evaluation.restore(state)
    state['calibration']['free_logit_bias'] = -.5
    with pytest.raises(RuntimeError):evaluation.restore(state)


def test_workflow_is_single_GPU_and_training_forward_has_no_calibration():
    source = Path('tools/causal_forward_train/workflow.sh').read_text()
    assert '${SCENE_TRAIN_GPUS:-0}' in source and 'torch.distributed.run' not in source
    assert '--free-minus1' in source and 'Training AND evaluation must each use ONE GPU' in source
    # The optimizer loop still calls data.forward_window / unchanged raw H6 loss.
    assert run.runner.forward_window.__module__ == 'tools.causal_forward_train.data'
    assert run.runner.evaluate is evaluation.evaluate
