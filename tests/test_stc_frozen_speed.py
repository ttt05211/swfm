import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_frozen_speed import freeze as frozen,engine,run
from tools.stc_norm_final.common import BODY_SHA,ROOT as ASSETS,read,write
from tools.shared_history_experiments.refine.model import SemanticRefiner


def test_freeze_merges_only_norm192_and_writes_independent_no_optimizer_checkpoint(tmp_path):
    body=torch.nn.Module();body.columns=SemanticRefiner();body.requires_grad_(False)
    original={n:p.detach().clone() for n,p in body.named_parameters()}
    saved=dict(source_sha256=BODY_SHA,model_configs={},source_contract={})
    parameters,before,adapted,payload=frozen.freeze(body,saved,tmp_path/'frozen')
    assert sum(v.numel() for v in parameters.values())==192
    assert all(torch.equal(v,original[n]) for n,v in body.named_parameters() if n not in parameters)
    assert any(not torch.equal(parameters[n],before[n]) for n in parameters)
    snapshot=torch.load(tmp_path/'frozen/final_model.pt',map_location='cpu',weights_only=False)
    assert snapshot['optimizer_included'] is False and snapshot['training_allowed'] is False
    assert snapshot['state_sha256']==frozen.tensor_state_fingerprint(body.state_dict())
    for n,v in body.state_dict().items():assert torch.equal(v,snapshot['state_dict'][n])
    with pytest.raises(FileExistsError):frozen.freeze(body,saved,tmp_path/'frozen')


def test_complete_server_release_binding_and_changed_asset_refused(tmp_path):
    folder=tmp_path/'all';folder.mkdir();(folder/'frozen').mkdir()
    keys=[[f'scene-{i%150:04d}',f'token{i}'] for i in range(4369)]
    for n in ('norm192.json','posterior_fit.json'):(folder/'frozen'/n).write_bytes((ASSETS/n).read_bytes())
    job=dict(population='ALL4369/150',source_sha256=BODY_SHA,source_policy='baseline',keys=keys,
        metric_mask='none',GT_future_ego_conditioned=True,known='STC dense-valid; NO GT visibility',horizon_group=6,
        assets={n:frozen.base.sha256(folder/'frozen'/n) for n in ('norm192.json','posterior_fit.json')})
    r=dict(status='complete',windows=4369,contract=job,metrics={'norm192/semantic_refine':dict(average=dict(mIoU=19.190065,IoU=29.016647))})
    write(folder/'contract.json',job);write(folder/'results.json',r)
    assert frozen.validate_server(folder)['verified_average']['IoU']==29.016647
    p=folder/'frozen/norm192.json';p.write_text('{}')
    with pytest.raises(ValueError,match='assets'):frozen.validate_server(folder)


def test_fps_forward_is_fresh_complete_six_and_matches_existing_quality_route(monkeypatch):
    from test_shared_source_track import fixture
    p,rec,raw=fixture();torch.set_num_threads(1)
    try:
        row=p.prepare(rec,raw,times=np.arange(4)*.5);poses=np.tile(np.eye(4),(6,1,1));bias=[.2]*17+[0.]
        row[3][0]['state']['column_backgrounds']=object()  # cached background must NEVER be used
        ready=engine.prepared(p,row,poses,np.arange(4)*.5)
        assert 'column_backgrounds' not in ready[0][3][0]['state']
        expected=engine.single(p,row,poses,np.arange(4)*.5,bias,6)
        original=engine.backgrounds;calls=[]
        def counted(*args):calls.append(1);return original(*args)
        monkeypatch.setattr(engine,'backgrounds',counted)
        def forbidden_gc():raise AssertionError('forced GC in frozen inference')
        monkeypatch.setattr(engine.gc,'collect',forbidden_gc)
        a=engine.fresh(p,ready,bias,6);b=engine.fresh(p,ready,bias,6)
        assert len(calls)==2 and a.shape==(6,*raw['history_occ'].shape[1:])
        np.testing.assert_array_equal(a,expected);np.testing.assert_array_equal(b,expected)
    finally:p.close()


def test_evaluation_ab_measures_gc_only_for_legacy_and_restores_frozen_parameters(monkeypatch):
    from test_shared_source_track import fixture
    p,rec,raw=fixture();torch.set_num_threads(1);calls=[]
    try:
        params={n:v for n,v in p.body.columns.named_parameters() if '.block.0.' in n}
        original={n:v.detach().clone() for n,v in params.items()};adapted={n:v+.01 for n,v in original.items()}
        monkeypatch.setattr(engine.gc,'collect',lambda:calls.append(1) or 0)
        poses=np.tile(np.eye(4),(6,1,1));read_input=lambda:(rec,raw,np.arange(4)*.5,poses)
        result,reference=engine.evaluation_ab(p,read_input,(params,original,adapted),[0.]*18,6,0)
        assert calls==[1];assert result['frozen_single_auto_gc']['forced_gc_seconds']==0
        assert reference.shape[0]==6 and all(torch.equal(params[n],v) for n,v in adapted.items())
    finally:p.close()


def test_timing_keeps_byte_failures_and_uses_total_time_not_mean_of_reciprocals():
    ab={n:dict(seconds=1.,forced_gc_seconds=.1 if n.startswith('legacy') else 0.)
        for n in ('legacy_paired_forced_gc','frozen_single_auto_gc')};ab['changed_voxels']=1
    rows=[dict(fps_seconds=v,fps_changed_voxels=[0,1],evaluation_ab=ab,profile={}) for v in ([.1,.2],[.3,.4])]
    r=engine.aggregate(rows,2,True)
    assert r['timing_complete'] and not r['strict_byte_parity'] and r['measured_calls']==4
    assert r['window_fps']==4 and r['output_frame_fps']==24 and r['byte_failed_windows']==2
    assert engine.aggregate([],2,False)['window_fps'] is None


def test_actual_small_workflow_does_not_read_future_labels_or_modify_source(tmp_path,monkeypatch):
    from test_shared_source_track import fixture
    old,rec,raw=fixture();old.close();body=old.body
    body.columns=SemanticRefiner(source_dim=body.transport.config.d_model,feature_width=16,z_bins=4,memory_width=4,width=32,chunk_size=32)
    body.requires_grad_(False);source=tmp_path/'bundle.zip';source.write_bytes(b'fixture source');before=source.read_bytes()
    records=[dict(rec,scene_name='scene',t0_token=f'token{i}',role='reference64',future_poses=np.tile(np.eye(4),(6,1,1))) for i in range(2)]
    for r in records:r['history_tokens']=[*rec['history_tokens'][:-1],r['t0_token']]
    class B:
        def __init__(self,*args,**kwargs):self.pcfg=old.provider.pcfg;self.records=records
        def load_model(self,*args):return body,dict(source_sha256=BODY_SHA,model_configs=body.configs(),source_contract={}), 'baseline'
        def history(self,r):return r,copy.deepcopy(raw),np.arange(4)*.5
        def targets(self,*args):raise AssertionError('future GT occupancy accessed by inference benchmark')
        def close(self):pass
    monkeypatch.setattr(run,'Bundle',B)
    out=tmp_path/'speed'
    assert run.main(['--out',str(out),'--bundle',str(source),'--portable-subset','--device','cpu','--windows','2','--repeats','1'])==0
    r=read(out/'results.json');assert r['status']=='complete' and r['windows']==2
    assert r['contract']['processes']==1 and r['speed']['window_fps'] is None
    assert (out/'frozen/final_model.pt').is_file() and source.read_bytes()==before
