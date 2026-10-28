"""Select from declared history provenance, never from labels or scores."""
import torch
from tools.causal_forward_all import common as baseline
from tools.causal_forward_stc_candidate import common as candidate
from tools.causal_forward_stc_candidate.recipe import specification as stc_specification

TAG = '_declared_history_modality'
DenseInputs = baseline.DenseInputs
select_source = candidate.select_source


def specification():
    return dict(name='current_epoch2_OCC_original_STC_state_range_v1',
        source_checkpoint_sha256=stc_specification()['source_checkpoint_sha256'],
        network_parameters=2816122,network_weights_changed=False,
        same_weights_both_modalities=True,same_inference_policy_both_modalities=False,
        policy_selection='declared input provenance occ/stc BEFORE prediction; no labels or score selection',
        input_policies=dict(occ=dict(source_state='original',free_bias=0.,adaptation=False),
            stc=stc_specification()),
        training=False,calibration_refitted=False,old_results_unchanged=True,
        data_use='STC uses previously fitted two-coefficient supervised VAL calibration; OCC uses none')


def validate_source(saved):
    return dict(baseline.validate_source(saved),evaluation_method=specification())


def load_frozen(path,device):
    c,body=candidate.load_frozen(path,device)
    return dict(c,evaluation_method=specification()),body


class ModalInputs(baseline.ModalInputs):
    def __init__(self,source,kind):
        if kind not in ('occ','stc'):raise ValueError('explicit occ/stc input provenance required')
        super().__init__(source,kind)

    def history(self,job,_='occ'):
        record,raw,times=super().history(job)
        if TAG in raw:raise ValueError('history provenance is owned by the input adapter')
        return record,dict(raw,**{TAG:self.kind}),times


class InputPolicyPredictor(candidate.CandidatePredictor):
    def prepare(self,record,raw,*,times=None):
        kind=raw.get(TAG)
        if kind not in ('occ','stc'):raise ValueError('explicit occ/stc input provenance required')
        # Association still receives exactly the original three history arrays.
        history={k:v for k,v in raw.items() if k!=TAG}
        rec,value,ego,reads=super().prepare(record,history,times=times)
        return rec,dict(value,**{TAG:kind}),ego,reads

    @torch.no_grad()
    def forward(self,row,*,poses):
        rec,raw,ego,reads=row;kind=raw.get(TAG)
        if kind not in ('occ','stc'):raise ValueError('explicit occ/stc input provenance required')
        clean=rec,{k:v for k,v in raw.items() if k!=TAG},ego,reads
        if kind=='occ':return baseline.Epoch2Predictor.forward(self,clean,poses=poses)
        return candidate.CandidatePredictor.forward(self,clean,poses=poses)
