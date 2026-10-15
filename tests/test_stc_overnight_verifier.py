"""Resume accounting for the isolated evaluator; NOT real quality evidence."""
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_overnight import verify_all as v


def test_complete_window_resume_and_calibration_integrity(tmp_path,monkeypatch):
    event=threading.Event();calls=[]
    class TinyBundle:
        def __init__(self,*args):
            self.records=[dict(role='reference64',scene_name=f'scene{i}',t0_token=str(i),
                future_poses=np.tile(np.eye(4),(6,1,1))) for i in range(64)]
            self.manifest={'reference64_keys':[[r['scene_name'],r['t0_token']] for r in self.records]}
            self.pcfg=SimpleNamespace(grid=SimpleNamespace(shape_hwd=(2,2,1)))
            self.zip=SimpleNamespace(read=lambda name:b'tiny-fixed-manifest')
        def load_model(self,*args):return torch.nn.Linear(1,1),dict(source_sha256='source'), 'baseline'
        def history(self,r):return r,{},np.arange(4)*.5
        def targets(self,r):return np.zeros((3,2,2,1),np.uint8)
        def close(self):pass
    class TinyPredictor:
        def __init__(self,body,pcfg,*args,**kwargs):self.provider=SimpleNamespace(pcfg=pcfg)
        def prepare(self,r,raw,times):return r
        def close(self):pass
    def forecast(p,row,poses,bias=None,include_support=False):
        calls.append(row['t0_token'])
        if len(calls)==16:event.set()
        names=['transport','semantic_refine']
        if bias is not None:names.append('calibrated_semantic_refine')
        if include_support:names+=['supported_'+n for n in list(names)]
        return {n:np.zeros((6,2,2,1),np.uint8) for n in names}
    monkeypatch.setattr(v,'Bundle',TinyBundle);monkeypatch.setattr(v,'Predictor',TinyPredictor)
    monkeypatch.setattr(v,'forecast',forecast);monkeypatch.setattr(v,'candidate',lambda row,*args:(row,{}))
    from tools.stc_overnight import calibration
    monkeypatch.setattr(calibration,'fit',lambda *args,**kwargs:dict(source_sha256='source',bias=[0.]*18))
    argv=['--bundle','unused','--out',str(tmp_path/'eval'),'--device','cpu',
        '--population','dev64','--portable-subset','--posterior-align','--history-support']
    v.main(argv,event)
    out=tmp_path/'eval';state=json.loads((out/'state.json').read_text())
    assert len(state['keys'])==8 and state['calibration_sha256']
    assert len(state['counts'])==8
    fit=out/'unlabeled_posterior_fit.json';original=fit.read_bytes()
    fit.write_text(json.dumps(dict(source_sha256='source',bias=[.1]*18)))
    with pytest.raises(ValueError,match='calibration missing/changed'):v.main(argv+['--resume'],threading.Event())
    fit.write_bytes(original);calls.clear()
    # Disable the first-run artificial interruption, while retaining call audit.
    def uninterrupted(p,row,poses,bias=None,include_support=False):
        result=forecast(p,row,poses,bias,include_support);event.clear();return result
    monkeypatch.setattr(v,'forecast',uninterrupted)
    v.main(argv+['--resume'],threading.Event())
    result=json.loads((out/'results.json').read_text());state=json.loads((out/'state.json').read_text())
    assert result['status']=='complete' and result['windows']==64
    assert result['full_target_verified'] is False  # DEV is never FULL proof.
    assert '0' not in calls and '7' not in calls and len(calls)==2*56
    for c in state['counts'].values():assert (np.asarray(c['confusion']).sum((1,2))==64*4).all()
