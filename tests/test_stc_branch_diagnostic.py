from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real_motion.stc_camera_protocol import SETTINGS
from real_motion.waymo_i2world import fingerprint, file_sha256
from tools.real_motion.stc_branch_diagnostic import (ROUTES, PlannerOriginAudit,
    confusion, quality, edit_counts, output_digest, evaluate, restore, summary)
from test_stc_camera_protocol import fixture


class ModelFixture:
    def __init__(self, *args, fail=None, mismatch=False, **kwargs):
        self.calls=0; self.fail=fail; self.mismatch=mismatch

    def close(self): pass

    def full(self, rec, raw, verify=False):
        self.calls+=1
        if self.calls==self.fail: raise RuntimeError('fixture interrupted')
        strong=np.full(raw['history_occ'].shape[1:],17,np.uint8)
        transport=raw['history_occ'][-1].copy()
        joint=transport.copy(); joint[0,0,1]=4
        outputs={'motion':torch.tensor([self.calls if self.mismatch else 0.])}
        prep=SimpleNamespace(state=dict(anchors=[strong]*6,current=[]),baseline=[transport]*6,
                             outputs=outputs,raw=raw)
        return prep,[joint]*6,None,{}


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32,torch.float64,
                                 torch.int64,torch.bool])
@pytest.mark.parametrize('shape',[(2,3),(),(0,3)])
def test_output_digest_raw_bytes_scalar_empty_and_dtypes(dtype,shape):
    value=torch.zeros(shape,dtype=dtype,requires_grad=dtype.is_floating_point)
    before=value.detach().clone()
    digest=output_digest(dict(motion=value))
    assert digest==output_digest(dict(motion=value.detach().clone()))
    assert torch.equal(value,before) and value.grad is None
    assert digest!=output_digest(dict(other=value))
    other_dtype=torch.float32 if dtype!=torch.float32 else torch.float64
    assert digest!=output_digest(dict(motion=value.detach().to(other_dtype)))
    if value.numel():
        changed=value.detach().clone(); changed.reshape(-1)[0]=1
        assert digest!=output_digest(dict(motion=changed))


def test_bfloat16_digest_noncontiguous_and_bit_exact():
    value=torch.arange(12,dtype=torch.bfloat16).reshape(3,4).T
    assert not value.is_contiguous()
    assert output_digest(dict(motion=value))==output_digest(dict(motion=value.contiguous()))
    assert output_digest(dict(motion=value))!=output_digest(dict(motion=value.reshape(-1)))
    # Equal numerical values, distinct bit patterns (signed zero / NaN payload).
    for bits in ((0,-32768),(32704,32705)):
        a=torch.tensor([bits[0]],dtype=torch.int16).view(torch.bfloat16)
        b=torch.tensor([bits[1]],dtype=torch.int16).view(torch.bfloat16)
        assert output_digest(dict(motion=a))!=output_digest(dict(motion=b))


def test_four_setting_diagnostic_accepts_bfloat16_motion_outputs(tmp_path):
    source=fixture(tmp_path); windows=source.windows[:1]; source.preflight(windows)
    predictor=ModelFixture(); full=predictor.full
    def bfloat16(*args,**kwargs):
        prep,pred,prob,detail=full(*args,**kwargs)
        prep.outputs={k:v.to(torch.bfloat16) for k,v in prep.outputs.items()}
        return prep,pred,prob,detail
    predictor.full=bfloat16
    result=evaluate(source,windows,predictor,dict(windows=1))
    assert result['status']=='complete' and predictor.calls==4
    assert result['audits'][0]['paired_motion_identical']


def test_confusion_orientation_group_semantic_difference_and_edit_accounting():
    gt=np.array([11,13,11,17,4,17],np.uint8).reshape(2,3,1)
    pred=np.array([13,13,17,11,17,17],np.uint8).reshape(2,3,1)
    c=confusion(pred,gt,gt.shape)
    assert c[11,13]==1 and c[17,11]==1
    q=quality(c,(11,13))
    assert (q['tp'],q['fp'],q['fn'],q['semantic_tp'],q['within_group_wrong_class'])==(2,1,1,1,1)
    assert q['IoU']==50 and q['precision']==pytest.approx(2/3)
    old=np.array([17,17,11,4],np.uint8)
    new=np.array([11,4,13,17],np.uint8)
    edits=edit_counts(old,new,np.array([11,17,11,17],np.uint8))
    assert edits==dict(changed=4,added=2,removed=1,relabeled=1,corrected=2,damaged=2,
                       added_occ_tp=1,added_semantic_tp=1)
    with pytest.raises(ValueError): confusion(pred.astype(float),gt,gt.shape)


def test_one_forecast_per_setting_truth_boundary_and_baseline_semantics(tmp_path):
    source=fixture(tmp_path); windows=source.windows[:2]; source.preflight(windows)
    predictor=ModelFixture(); reads=[]; target=source.metric_targets
    def read(w):
        assert predictor.calls==4*(len(reads)+1)
        reads.append(w.key); return target(w)
    source.metric_targets=read
    result=evaluate(source,windows,predictor,dict(windows=2))
    assert predictor.calls==8 and len(reads)==2
    assert result['reports']['occ_gt/strong']['average']['standard_mIoU']==0
    assert result['reports']['occ_gt/transport']['average']['standard_mIoU']==100
    assert result['reports']['occ_gt/joint']['average']['standard_mIoU']<100
    assert len(result['counts'])==12 and all(a['paired_motion_identical'] for a in result['audits'])
    assert result['audits'][0]['history_observed_fraction']==dict(occ=[0.]*4,stc=[1.]*4)
    assert result['t0_quality']['stc']['occupied']['precision']==1
    assert 'no metric mask' in summary(result)


def test_stop_resume_atomic_counts_and_source_readonly(tmp_path):
    source=fixture(tmp_path); windows=source.windows[:2]; source.preflight(windows)
    contract=dict(windows=2); state={}; event=Event()
    save=lambda v:state.update(deepcopy(v))
    first=evaluate(source,windows,ModelFixture(),contract,save=save,stop_event=event,progress=lambda _:event.set())
    assert first['status']=='stopped' and state['completed_windows']==1
    saved_before=deepcopy(state)
    resumed=evaluate(source,windows,ModelFixture(),contract,saved=state)
    assert state==saved_before
    original=evaluate(source,windows,ModelFixture(),contract)
    for k in ('counts','edits','t0_counts','reports'): assert resumed[k]==original[k]
    assert original['status']=='complete'
    with pytest.raises(RuntimeError,match='contract'): restore(state,dict(windows=3),source.shape)
    altered=deepcopy(state); altered['counts']['occ_gt/joint'][0][0][0]+=1
    with pytest.raises(RuntimeError): restore(altered,contract,source.shape)
    value=deepcopy(state); value.pop('fingerprint'); value['t0_counts']['stc'][0][0]=1.5
    value['fingerprint']=fingerprint(value)
    with pytest.raises(ValueError,match='integer'): restore(value,contract,source.shape)


@pytest.mark.parametrize('error',['forecast','motion','future_truth','remove'])
def test_failures_do_not_publish_partial_window_or_read_future_early(tmp_path,error):
    source=fixture(tmp_path); windows=source.windows[:1]; source.preflight(windows)
    predictor=ModelFixture(fail=3 if error=='forecast' else None,mismatch=error=='motion')
    state={}; reads=[]; targets=source.metric_targets
    def read(w): reads.append(w.key); return targets(w)
    source.metric_targets=read
    if error=='future_truth':
        inputs=source.prediction_inputs
        def poisoned(w,s):
            rec,raw=inputs(w,s); raw['future_gt_occ']=np.zeros(source.shape,np.uint8); return rec,raw
        source.prediction_inputs=poisoned
    if error=='remove':
        full=predictor.full
        def removes(*args,**kwargs):
            prep,pred,prob,detail=full(*args,**kwargs)
            pred=[p.copy() for p in pred]
            for p in pred: p[0,0,0]=17
            return prep,pred,prob,detail
        predictor.full=removes
    with pytest.raises(RuntimeError):
        evaluate(source,windows,predictor,dict(windows=1),save=lambda v:state.update(deepcopy(v)))
    assert state['completed_windows']==0 and not state['audits']
    assert sum(np.asarray(v).sum() for v in state['counts'].values())==0
    assert len(reads)==(1 if error=='remove' else 0)


def planner_json(source,path):
    trajs={}
    for i,f in enumerate(sorted(source.catalog.values(),key=lambda f:f.timestamp)):
        rows=[[f.pose[0,3],f.pose[1,3],0.]]
        rows += [[f.pose[0,3]+(j+1)*.04,(j+1)*.1,0.] for j in range(6)]
        trajs[f.scene+'-'+str(i)]=rows
    path.write_text(json.dumps(dict(trajs=trajs)),encoding='utf-8')
    return trajs


def test_original_json_mapping_rebuild_missing_and_wrong_origin(tmp_path):
    source=fixture(tmp_path); source.preflight(source.windows[:1]); w=source.windows[0]
    unavailable=PlannerOriginAudit(source)
    assert unavailable.window(w)['status']=='unverified_missing_original_json'
    assert not unavailable.metadata['cached_pose_origin_verified']
    path=tmp_path/'planner.json'; rows=planner_json(source,path)
    original=PlannerOriginAudit(source,path,expected_sha=file_sha256(path))
    row=original.window(w)
    assert row['status']=='verified' and row['t0_ordinal']==3
    assert row['cached_pose_max_abs_error']<1e-14
    before=np.asarray(source.plans[w.key]).copy()
    rows[row['json_key']][0][0]+=1
    path.write_text(json.dumps(dict(trajs=rows)),encoding='utf-8')
    bad=PlannerOriginAudit(source,path,expected_sha=file_sha256(path)).window(w)
    assert bad['status']=='mismatch' and bad['current_xy_error_m']==pytest.approx(1)
    assert np.array_equal(before,np.asarray(source.plans[w.key]))
    with pytest.raises(RuntimeError,match='SHA256'): PlannerOriginAudit(source,path)
    with pytest.raises(FileNotFoundError): PlannerOriginAudit(source,tmp_path/'missing.json')


def test_original_json_yaw_is_independent_not_cumulative(tmp_path):
    source=fixture(tmp_path); source.preflight(source.windows[:1]); w=source.windows[0]
    path=tmp_path/'yaw-planner.json'; trajs=planner_json(source,path)
    key=w.scene+'-3'; yaws=np.arange(1,7)*.07
    for row,yaw in zip(trajs[key][1:],yaws): row[2]=float(yaw)
    planned=np.asarray(source.plans[w.key]).copy()
    for p,yaw in zip(planned,yaws):
        c,s=np.cos(yaw),np.sin(yaw)
        p[:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    source.plans[w.key]=planned
    path.write_text(json.dumps(dict(trajs=trajs)),encoding='utf-8')
    audit=PlannerOriginAudit(source,path,expected_sha=file_sha256(path))
    assert audit.window(w)['status']=='verified'
    cumulative=planned.copy()
    for p,yaw in zip(cumulative,np.cumsum(yaws)):
        c,s=np.cos(yaw),np.sin(yaw); p[:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    source.plans[w.key]=cumulative
    assert audit.window(w)['status']=='mismatch'


def test_resume_prefix_identity_order_checked_even_with_recomputed_fingerprint(tmp_path):
    source=fixture(tmp_path); windows=source.windows[:2]; source.preflight(windows)
    contract=dict(windows=2,population=dict(selected_keys=[[w.scene,w.t0] for w in windows]))
    state={}
    evaluate(source,windows,ModelFixture(),contract,save=lambda v:state.update(deepcopy(v)))
    assert restore(state,contract,source.shape)['completed_windows']==2
    state.pop('fingerprint'); state['audits'].reverse(); state['fingerprint']=fingerprint(state)
    with pytest.raises(ValueError,match='identity/order'): restore(state,contract,source.shape)


def test_real_model_stage_readout_does_not_change_predictions_or_weights(tmp_path):
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.joint_surface_ccr import JointSurfaceCCR
    from real_motion.waymo_native_execution import prepare_waymo_native
    from tools.real_motion.stc_shared_execution import Predictor,bytes_equal
    prepare_waymo_native(tmp_path/'native')
    source=fixture(tmp_path,shape=(32,32,4)); windows=source.windows[:1]; source.preflight(windows)
    cfg=PrepareConfig(grid=OccupancyGrid(-6.4,-6.4,-1.,(.4,)*3,source.shape))
    joint=JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,blocks=1,
        decoder_blocks=1),width=16,z_bins=4).eval().requires_grad_(False)
    before={k:v.clone() for k,v in joint.state_dict().items()}
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    predictor=Predictor(joint,cfg,'cpu',workers=1,graphs=False)
    try:
        rec,raw=source.prediction_inputs(windows[0],'stc_pred')
        original=predictor.full(rec,dict(raw),verify=True)
        result=evaluate(source,windows,predictor,dict(windows=1))
        repeated=predictor.full(rec,dict(raw),verify=True)
        for a,b in zip(original[1],repeated[1]): bytes_equal(a,b,'diagnostic untouched six predictions')
        bytes_equal(original[2],repeated[2],'diagnostic untouched probabilities')
        assert all(torch.equal(v,before[k]) for k,v in joint.state_dict().items())
        assert result['status']=='complete' and len(result['reports'])==12
    finally: predictor.close(); torch.set_num_threads(threads)


def test_cli_resume_contract_and_source_readonly(tmp_path,monkeypatch):
    from real_motion.geometry import OccupancyGrid
    from real_motion.prepared import PrepareConfig
    from tools.real_motion import eval_p0_f9_stc_branch_diagnostic as cli
    source=fixture(tmp_path); windows=source.windows[:2]
    monkeypatch.setattr(source,'select',lambda *a:(windows,dict(population='dev64')))
    monkeypatch.setattr(cli.base.STCFourSettingSource,'from_files',lambda *a,**kw:source)
    monkeypatch.setattr(cli.base,'load_manifest',lambda *a:(dict(parent_keys=[1]),None,None))
    monkeypatch.setattr(cli.base,'load_runtime_config',lambda *a:None)
    cfg=PrepareConfig(grid=OccupancyGrid(-40,-40,-1.,(.4,)*3,source.shape))
    monkeypatch.setattr(cli.base,'make_prepare_config',lambda *a:cfg)
    monkeypatch.setattr(cli.base,'SHAPE',source.shape)
    monkeypatch.setattr(cli.torch.cuda,'is_available',lambda:True)
    model=SimpleNamespace(transport=SimpleNamespace(config=SimpleNamespace(history_frames=4)))
    monkeypatch.setattr(cli.base,'load_evaluation_model',lambda *a,**kw:
        (dict(source_epochs=list(cli.base.AVERAGE_EPOCHS),averaging=True),model))
    monkeypatch.setattr(cli,'Predictor',ModelFixture)
    checkpoint=tmp_path/'weights/mean.pt'; checkpoint.parent.mkdir(); checkpoint.write_text('fixture')
    config=tmp_path/'config.yaml'; config.write_text('fixture')
    manifest=tmp_path/'manifest.json'; manifest.write_text('fixture')
    digest=file_sha256(checkpoint); out=tmp_path/'diagnostic'
    common=['--dataroot',str(source.root),'--stc-root',str(source.stc_root),'--plan-cache',str(source.plan_cache),
        '--population','dev64','--population-manifest',str(manifest),'--config',str(config),'--checkpoint',str(checkpoint)]
    args=common+['--out-dir',str(out)]
    assert cli.main(argv=args)==0
    first=json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
    assert cli.main(argv=args+['--resume'])==0
    assert json.loads((out/'evaluation.json').read_text(encoding='utf-8'))['counts']==first['counts']
    assert file_sha256(checkpoint)==digest
    with pytest.raises(RuntimeError,match='contract'): cli.main(argv=args+['--resume','--cpu-workers','2'])
    with pytest.raises(SystemExit): cli.main(argv=args)
    with pytest.raises(SystemExit): cli.main(argv=common+['--out-dir',str(source.stc_root/'unsafe')])
    with pytest.raises(SystemExit): cli.main(argv=common+['--out-dir',str(tmp_path/'full'),'--population','all'])


def test_launcher_restores_environment_without_shell_eval():
    text=(Path(__file__).resolve().parents[1]/'tools/real_motion/run_p0_f9_stc_branch_diagnostic.sh').read_text(encoding='utf-8')
    assert 'STC_DIAG_RESUME' in text and 'STC_PLANNER_JSON' in text
    assert "recorded = json.load(handle)['runtime_environment']" in text
    assert 'os.execve' in text and 'eval ' not in text
