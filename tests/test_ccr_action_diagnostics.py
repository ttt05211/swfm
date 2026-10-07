import numpy as np

from tools.real_motion.diagnose_p0_f9_ccr_quality_gap import (
    _finish_ranking, _ranking_bucket, _update_ranking,
)


def test_streaming_ranking_perfect_separation():
    bucket=_ranking_bucket()
    scores=np.array([.95,.80,.20,.05],np.float32)
    target=np.array([1,1,0,0],bool)
    _update_ranking(bucket,scores,target,np.ones(4,bool))
    row=_finish_ranking(bucket)
    assert row["count"]==4 and row["positives"]==2
    assert abs(row["prevalence"]-.5)<1e-12
    assert row["average_precision"] > .999
    assert row["auroc"] > .999
    assert row["ap_lift_over_prevalence"] > 1.99
    assert row["mean_score_positive"] > row["mean_score_negative"]
    assert row["best_f1"]["f1"] > .999


def test_streaming_ranking_tied_scores_reduce_to_prevalence():
    bucket=_ranking_bucket()
    scores=np.full(10,.3,np.float32)
    target=np.array([1,0,0,1,0,0,0,1,0,0],bool)
    _update_ranking(bucket,scores,target,np.ones(10,bool))
    row=_finish_ranking(bucket)
    assert abs(row["prevalence"]-.3)<1e-12
    assert abs(row["average_precision"]-.3)<1e-12
    assert abs(row["auroc"]-.5)<1e-12
    assert abs(row["mean_score_positive"]-.3)<1e-6
    assert abs(row["mean_score_negative"]-.3)<1e-6


def test_streaming_ranking_mask_excludes_invalid_rows():
    bucket=_ranking_bucket()
    scores=np.array([.9,.8,.1,.2],np.float32)
    target=np.array([1,0,1,0],bool)
    mask=np.array([1,0,0,1],bool)
    _update_ranking(bucket,scores,target,mask)
    row=_finish_ranking(bucket)
    assert row["count"]==2 and row["positives"]==1 and row["negatives"]==1
    assert row["average_precision"] > .999
    assert row["auroc"] > .999
