"""Pin the exact latest body; metadata declares the labelled calibration."""
import torch
from tools.causal_forward_all import common as original
from tools.joint_occ_experiments import run as base
from . import recipe
from .method import state_row,radius,calibrated_logits

DenseInputs,ModalInputs=original.DenseInputs,original.ModalInputs


def select_source(explicit,root):
    path=original.select_source(explicit,root)
    if base.sha256(path)!=recipe.SOURCE_SHA256:raise RuntimeError('candidate requires the exact downloaded current epoch2 source SHA, not an old or averaged body')
    return path


def validate_source(saved):
    c=original.validate_source(saved)
    return dict(c,evaluation_method=recipe.specification())


def load_frozen(path,device):
    if base.sha256(path)!=recipe.SOURCE_SHA256:raise RuntimeError('candidate original epoch2 SHA differs')
    c,body=original.load_frozen(path,device)
    return dict(c,evaluation_method=recipe.specification()),body


class CandidatePredictor(original.Epoch2Predictor):
    @torch.no_grad()
    def forward(self,row,*,poses):
        current=row[1]['history_poses'][-1]
        changed=state_row(row,recipe.POSITION_BLEND,recipe.VELOCITY_BLEND)
        fw=super().forward(changed,poses=poses)
        rho=radius(self.provider.pcfg.grid,current,fw['poses'])
        return dict(fw,logits=calibrated_logits(fw['logits'],rho,recipe.FREE_INTERCEPT,recipe.FREE_SLOPE))
