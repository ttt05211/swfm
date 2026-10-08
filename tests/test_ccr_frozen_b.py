import numpy as np
import torch

from real_motion.ccr_frozen_b import (
    frozen_b_probabilities,
    effective_corrected_add_thresholds,
)


class Evidence:
    def __init__(self):
        self.features=np.zeros((2,3),np.float32)
        self.labels=np.zeros((2,4),np.uint8)
        self.actor=np.array([-2,0],np.int32)
        self.classes=np.array([11,4],np.uint8)
        self.neighbor_graph=None
    def __len__(self):
        return len(self.actor)


class Plan:
    def __init__(self):
        self.context=np.zeros((2,6,8),np.float32)
        self.base=np.zeros((2,6),np.uint8)
        self.fallback=np.zeros((2,6),np.uint8)
        self.legal=np.ones((2,6,2),bool)


class Head:
    def __init__(self):
        self.positive_weight=torch.tensor([[3.,2.],[4.,5.]])
    def project_sources(self,output):
        return output
    def encode(self,features,labels,actor,classes,output):
        return torch.zeros((len(actor),4),device=actor.device)
    def decode(self,encoded,actor,context,base,fallback,legal,output):
        n=len(actor)
        z=torch.zeros((n,6,2),device=actor.device)
        z[0,:,0]=-1.0
        z[1,:,0]=1.0
        z[...,1]=9.0
        return z


def test_frozen_b_raw_add_and_remove_disabled():
    score=frozen_b_probabilities(Head(),Evidence(),Plan(),{},torch.device("cpu"))
    assert score.shape==(2,6,2)
    assert np.allclose(score[0,:,0],1/(1+np.exp(1.0)),rtol=0,atol=1e-7)
    assert np.allclose(score[1,:,0],1/(1+np.exp(-1.0)),rtol=0,atol=1e-7)
    assert np.array_equal(score[...,1],np.zeros((2,6),np.float32))


def test_frozen_b_effective_corrected_thresholds():
    rule=effective_corrected_add_thresholds(Head())
    assert abs(rule["static_corrected_threshold"]-.25)<1e-12
    assert abs(rule["dynamic_corrected_threshold"]-.20)<1e-12
