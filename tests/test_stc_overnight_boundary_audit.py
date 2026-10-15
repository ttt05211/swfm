import json
import numpy as np
import pytest
from tools.stc_overnight.boundary_audit import audit


def test_boundary_audit_is_paired_and_never_extrapolates_to_full(tmp_path):
    base,candidate=tmp_path/'base',tmp_path/'candidate';base.mkdir();candidate.mkdir()
    keys=[[f'scene{i}',str(i)] for i in range(64)]
    contract=dict(keys=keys,model_sha256='frozen',modality='stc',known='dense_valid',
        view='identity',bundle_manifest_sha256='bundle',group_size=1)
    for path in (base,candidate):
        (path/'results.json').write_text(json.dumps(dict(status='complete',contract=contract)))
    for i,key in enumerate(keys):
        a=np.zeros((3,18,18),np.int64)
        for c in range(18):a[:,c,c]=10
        a[:,1,17]=1;b=a.copy();b[:,1,17]=0;b[:,1,1]+=1
        for path,counts in [(base,a),(candidate,b)]:
            (path/f'window_{i:03d}.json').write_text(json.dumps(dict(key=key,confusion={'semantic_refine':counts.tolist()})))
    report=audit(base,candidate)
    assert report['windows']==64 and report['full_goal_verified'] is False
    assert report['delta']['IoU']>0 and report['delta']['mIoU']>0
    assert report['positive_individual_window_counts']==dict(mIoU=64,IoU=64)
    assert set(report['by_horizon'])=={'1.0','2.0','3.0'}
    altered=dict(contract,known='lidar_known')
    (candidate/'results.json').write_text(json.dumps(dict(status='complete',contract=altered)))
    with pytest.raises(ValueError,match='not paired: known'):audit(base,candidate)
