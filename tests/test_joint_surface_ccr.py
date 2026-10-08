import copy
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import random
import numpy as np
import pytest
import torch
from real_motion.joint_surface_ccr import JointSurfaceCCR, PROTOCOL
from real_motion.canonical_causal_repair import build_canonical_evidence
from real_motion.canonical_repair_context import FixedCanonicalCache
from real_motion.surface_canonical_repair import SurfaceAtlas, augment_evidence, augment_projection
from tools.real_motion.joint_surface_ccr_common import train_batch
from tools.real_motion.joint_surface_ccr_recovery import payload, restore
from tools.real_motion.train_p0_f9_joint_surface_ccr import parser
from test_joint_causal_columns import fixture as motion_fixture
from test_canonical_causal_repair import scene
from test_height_field_screen import assert_nested_equal


def fixture(device):
    _,_,old,_,record=motion_fixture(); grid,prep=scene()
    config=replace(old.transport.v17_config,history_frames=4)
    joint=JointSurfaceCCR(config,z_bins=4).to(device)
    prep.raw['future_gt_occ']=prep.baseline.copy()
    prep.raw['future_gt_occ'][:,3,2,1]=4
    prep.raw['future_gt_occ'][:,5:7,5:7,0]=11
    provider=SimpleNamespace(device=torch.device(device),pcfg=SimpleNamespace(grid=grid),workers=1,
        ccr_cache=FixedCanonicalCache(1,neighbors=False),ccr_samples_per_role=16,columns_checked=True)
    def prepare(*args,**kwargs):
        result=copy.deepcopy(prep)
        # Exercise the CURRENT predicted motion -> targets path, not stale GT
        # or a cached learned pose. Hard geometry intentionally detached.
        out=kwargs['outputs']
        assert not out['residual_xy_m'].requires_grad
        result.targets+=out['residual_xy_m'].float().cpu().numpy().transpose(1,0,2).mean(-1)[...,None]*.1
        return result
    provider.prepare_columns=prepare
    def augment(e,p,ready):
        full=build_canonical_evidence(ready,grid)
        atlas=SurfaceAtlas(full.world,full.classes,full.presence,full.actor,np.eye(4),grid)
        e=augment_evidence(e,atlas)
        return e,augment_projection(e,p,np.eye(4),ready.state['world_to_future'],grid)
    provider.ccr_augment_sample=augment
    opt=torch.optim.AdamW([
        dict(params=joint.transport.parameters(),lr=5e-4,initial_lr=5e-4),
        dict(params=joint.columns.parameters(),lr=3e-4,initial_lr=3e-4)])
    return joint,provider,[(record,prep.raw)],opt


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_clean_joint_updates_motion_static_dynamic_and_gradient_link(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    joint,p,rows,opt=fixture(device); before=copy.deepcopy(joint.state_dict())
    stats=train_batch(joint,opt,p,rows*2,np.random.default_rng(8),1,20,probe=True)
    assert stats['source_query_gradient_norm']>0 and stats['transport_frozen'] is False
    assert stats['windows']==2 and stats['sources']==2
    for key in ('transport.residual_head.weight','columns.encoder.0.weight',
                'columns.static_readout.3.weight','columns.readout.3.weight'):
        assert not torch.equal(before[key],joint.state_dict()[key]),key
    assert all(p.grad is None for p in joint.parameters())
    p.ccr_cache.close()


def contract():
    return dict(epoch_batches=[2,2],epoch_batch_sizes=[[2,2],[2,2]],schedule_steps=4,
                epochs=2,window_batch=4,source_budget=128,initialization='RANDOM')


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_resume_next_real_joint_update_exact_all_rng_optimizer_lr(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    joint,p,rows,opt=fixture(device); rng=np.random.default_rng(91)
    train_batch(joint,opt,p,rows*2,rng,1,4,probe=True)
    c=contract(); saved=copy.deepcopy(payload(joint,opt,rng,c,epoch=0,batch=1,updates=1,executed=2,
        reports={'train_prior':{'positive_weights':joint.columns.positive_weight.detach().cpu().tolist()}}))
    other=copy.deepcopy(joint); other_opt=torch.optim.AdamW([
        dict(params=other.transport.parameters(),lr=99.),dict(params=other.columns.parameters(),lr=99.)])
    other_rng=np.random.default_rng(3)
    reference_random=(random.random(),np.random.rand(),torch.rand(3,device=device))
    train_batch(joint,opt,p,rows*2,rng,2,4,probe=True)
    cursor,_=restore(saved,other,other_opt,other_rng,c)
    got_random=(random.random(),np.random.rand(),torch.rand(3,device=device))
    assert reference_random[:2]==got_random[:2] and torch.equal(reference_random[2],got_random[2])
    train_batch(other,other_opt,p,rows*2,other_rng,2,4,probe=True)
    assert cursor==(0,1,1,2)
    assert_nested_equal(joint.state_dict(),other.state_dict()); assert_nested_equal(opt.state_dict(),other_opt.state_dict())
    assert rng.bit_generator.state==other_rng.bit_generator.state
    for change in ({'epochs':20},{'window_batch':8},{'source_budget':256}):
        with pytest.raises(RuntimeError,match='identical'):
            restore(saved,other,other_opt,other_rng,{**c,**change})
    with pytest.raises(RuntimeError,match='identical'):
        restore({**saved,'transport_frozen':True},other,other_opt,other_rng,c)
    with pytest.raises(RuntimeError,match='counters'):
        restore({**saved,'updates':2},other,other_opt,other_rng,c)
    p.ccr_cache.close()


def test_parallel_sample_workers_keep_rng_and_updates_exact():
    joint,p,rows,opt=fixture('cpu'); other=copy.deepcopy(joint)
    other_opt=torch.optim.AdamW([
        dict(params=other.transport.parameters(),lr=5e-4,initial_lr=5e-4),
        dict(params=other.columns.parameters(),lr=3e-4,initial_lr=3e-4)])
    rng=np.random.default_rng(93); shadow=np.random.default_rng(93)
    with ThreadPoolExecutor(max_workers=2) as pool:
        train_batch(joint,opt,p,rows*2,rng,1,20,pool=pool,probe=True)
    train_batch(other,other_opt,p,rows*2,shadow,1,20,probe=True)
    assert_nested_equal(joint.state_dict(),other.state_dict()); assert_nested_equal(opt.state_dict(),other_opt.state_dict())
    assert rng.bit_generator.state==shadow.bit_generator.state
    p.ccr_cache.close()


def test_cli_recipe_not_frozen_teacher_and_whole_cosine_default():
    args=[]
    for key in ('config','train-cache','dev-cache','population-manifest','base-checkpoint','dataroot','train-info','dev-info','out-dir'):
        args+=['--'+key,'dummy']
    a=parser().parse_args(args)
    assert a.epochs==20 and a.ccr_motion_superbatch_updates==1
    assert a.motion_lr==5e-4 and a.repair_lr==3e-4
    assert a.window_batch_size==4 and a.source_budget==128 and not a.warm_start_head


def mock_cli(monkeypatch,tmp_path):
    from dataclasses import asdict
    from tools.real_motion import train_p0_f9_joint_surface_ccr as cli
    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    joint,provider,rows,_=fixture('cpu');record,raw=rows[0]
    files={}
    for k in ('config','train-cache','dev-cache','population-manifest','base-checkpoint','train-info','dev-info'):
        files[k]=tmp_path/k;files[k].write_bytes(b'fixture')
    torch.save({'model_config':asdict(joint.transport.v17_config)},files['base-checkpoint'])
    monkeypatch.setattr(cli,'TRAIN_WINDOWS',8);monkeypatch.setattr(cli,'VAL_WINDOWS',4)
    monkeypatch.setattr(cli,'DEV64_WINDOWS',2);monkeypatch.setattr(cli,'DEV512_WINDOWS',4)
    train=[{**record,'scene_name':'train','t0_token':str(i)} for i in range(8)]
    dev=[{**record,'scene_name':'dev','t0_token':str(i)} for i in range(4)]
    parent=[('dev',str(i)) for i in range(4)]
    monkeypatch.setattr(cli,'load_cache',lambda path:({},train if Path(path)==files['train-cache'] else dev))
    monkeypatch.setattr(cli,'load_manifest',lambda _:({'parent_keys':parent,'manifest_fingerprint':'fixed',
        'selected_key_fingerprint':cli.DEV64_FP},parent[:2],None))
    monkeypatch.setattr(cli,'require_cuda',lambda _:torch.device('cpu'))
    monkeypatch.setattr(cli,'validate_clean_e14_checkpoint',lambda *args:None)
    monkeypatch.setattr(cli,'load_runtime_config',lambda *args:{})
    monkeypatch.setattr(cli,'make_prepare_config',lambda _:provider.pcfg)
    monkeypatch.setattr(cli,'NuScenesWindowSource',lambda *args,**kw:SimpleNamespace())
    monkeypatch.setattr(cli,'CachedColumnSource',lambda source,_:source)
    def make_provider(*args):
        provider.joint=args[-2];provider.model=provider.joint.transport
        provider.load_raw_columns=lambda *args,**kwargs:copy.deepcopy(raw)
        return provider
    monkeypatch.setattr(cli,'PilotProvider',make_provider)
    def setup(p,args):
        p.ccr_cache=FixedCanonicalCache(1,neighbors=False);p.ccr_execution=CanonicalCpuExecution('numpy')
        p.ccr_samples_per_role=8;p.ccr_sample_workers=2;p.train_io_workers=2
        p.ccr_history_cache=SimpleNamespace(namespace='train',stats=lambda:{},close=lambda:None)
        p.ccr_val_history_cache=SimpleNamespace(namespace='val',stats=lambda:{},close=lambda:None)
    monkeypatch.setattr(cli.surface,'setup',setup)
    monkeypatch.setattr(cli,'fit_train_prior',lambda p,s,r,j,**kwargs:
                        dict(positive_weights=j.columns.positive_weight.cpu().tolist(),TRAIN_only=True))
    eval_calls=[]
    def evaluation(*args,**kwargs):
        eval_calls.append(len(args[2]))
        m=dict(mIoU=40.,IoU=52.,MovingMacro=26.,MovingMicro=31.)
        return dict(windows=len(args[2]),variants={'joint':dict(metrics=m)})
    monkeypatch.setattr(cli.ccr,'evaluate',evaluation)
    argv=[v for k,path in files.items() for v in ('--'+k,str(path))]
    argv+=['--dataroot',str(tmp_path),'--ccr-history-cache',str(tmp_path),
           '--ccr-val-history-cache',str(tmp_path),'--epochs','2','--prior-windows','1',
           '--checkpoint-every','1','--ccr-cpu-execution','numpy']
    return cli,argv,eval_calls


def test_cli_resume_epoch_monitor_final_phase_and_readonly_eval(monkeypatch,tmp_path):
    from pathlib import Path
    cli,argv,calls=mock_cli(monkeypatch,tmp_path)
    full,stop,resumed=[tmp_path/n for n in ('full','stop','resumed')]
    assert cli.main(argv=argv+['--out-dir',str(full)])==0
    assert cli.main(argv=argv+['--out-dir',str(stop),'--stop-after-epoch','1'])==0
    before_calls=len(calls)
    assert cli.main(argv=argv+['--out-dir',str(resumed),'--resume',str(stop/'last.pt')])==0
    assert calls[before_calls:]==[2,4]  # no repeat of completed epoch1 monitor
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
    assert (b['epoch'],b['batch'],b['updates'],b['executed'])==(2,0,4,16)
    for key in ('state_dict','optimizer','sampling_rng','torch_rng'):
        assert_nested_equal(a[key],b[key])
    assert [r['epoch'] for r in b['reports']['epochs']]==[1,2]
    assert (resumed/'epoch_0002.pt').is_file() and (resumed/'candidate.pt').is_file()
    from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
    digest=sha256(resumed/'last.pt')
    assert cli.main(argv=argv+['--out-dir',str(tmp_path/'eval'),'--resume',str(resumed/'last.pt'),
                              '--evaluate-only','--eval-population','dev512'])==0
    assert sha256(resumed/'last.pt')==digest
    assert not (tmp_path/'eval'/'last.pt').exists()
    with pytest.raises(RuntimeError,match='identical'):
        cli.main(argv=argv+['--out-dir',str(tmp_path/'bad'),'--resume',str(resumed/'last.pt'),'--epochs','3'])


def test_cli_failed_update_never_publishes_partial_weights(monkeypatch,tmp_path):
    cli,argv,_=mock_cli(monkeypatch,tmp_path);actual=cli.train_batch;calls=[]
    def fail(joint,*args,**kwargs):
        calls.append(1)
        if len(calls)==2:
            with torch.no_grad():joint.columns.surface.weight.fill_(99.)
            raise RuntimeError('incomplete optimizer update')
        return actual(joint,*args,**kwargs)
    monkeypatch.setattr(cli,'train_batch',fail);out=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='incomplete'):
        cli.main(argv=argv+['--out-dir',str(out)])
    saved=torch.load(out/'last.pt',weights_only=False)
    assert saved['updates']==1 and saved['batch']==1
    assert not (saved['state_dict']['columns.surface.weight']==99).all()
    assert (out/'last.previous.pt').is_file()


def test_real_train_prior_full_counts_and_projection_preflight(monkeypatch):
    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    from real_motion.surface_projection_execution import SurfaceMapExecution
    from real_motion.canonical_causal_repair import map_canonical_evidence,repair_targets
    from tools.real_motion.joint_surface_ccr_common import fit_train_prior,prepare_live
    from real_motion import column_runtime_pipeline as pipeline
    joint,p,rows,opt=fixture('cpu');record,raw=rows[0]
    p.ccr_execution=SurfaceMapExecution(CanonicalCpuExecution('numpy'))
    def enriched(e,ready):
        atlas=SurfaceAtlas(e.world,e.classes,e.presence,e.actor,ready.state['current_pose'],p.pcfg.grid)
        return augment_evidence(e,atlas)
    p.ccr_augment_evidence=enriched
    monkeypatch.setattr(pipeline,'prefetch_raw_columns',lambda *args:iter([(record,copy.deepcopy(raw))]*3))
    result=fit_train_prior(p,None,[record]*3,joint)
    joint.eval()
    with torch.no_grad():prep=prepare_live(p,record,copy.deepcopy(raw),joint.motion(record,p.device))
    evidence=build_canonical_evidence(prep,p.pcfg.grid)
    plan=map_canonical_evidence(evidence,prep,p.pcfg.grid)
    y,valid=repair_targets(evidence,plan,raw['future_gt_occ'])
    expected=np.zeros((2,2,2),np.int64)
    for role in range(2):
        for action in range(2):
            mask=valid[...,action]&((evidence.actor>=0)==bool(role))[:,None]
            pos=int(y[...,action][mask].sum());expected[role,action]=[int(mask.sum())-pos,pos]
    np.testing.assert_array_equal(result['counts'],expected*3)
    assert result['windows']==3 and result['probability_correction']=='none'
    expected_weight=np.sqrt(expected[...,0]*3/np.maximum(expected[...,1]*3,1)).clip(1,32).astype(np.float32)
    np.testing.assert_array_equal(joint.columns.positive_weight.numpy(),expected_weight)
    p.ccr_cache.close();p.ccr_execution.close()


def test_empty_causal_support_still_trains_motion_without_nan():
    joint,p,rows,opt=fixture('cpu');old_prepare=p.prepare_columns
    def prepare(*args,**kwargs):
        ready=old_prepare(*args,**kwargs)
        ready.raw['history_occ'].fill(17);ready.registrations=[[None]*4]
        ready.baseline.fill(17);ready.owners.fill(-1);ready.fallbacks.fill(17)
        return ready
    p.prepare_columns=prepare
    stat=train_batch(joint,opt,p,rows,np.random.default_rng(82),1,20,probe=True)
    assert stat['sampled_points']==0 and stat['repair_loss']==0
    assert np.isfinite(stat['loss']) and stat['motion_grad_norm']>0
    p.ccr_cache.close()


def test_cli_interrupt_monitor_resume_then_interrupt_final_resume(monkeypatch,tmp_path):
    cli,argv,calls=mock_cli(monkeypatch,tmp_path);actual=cli.ccr.evaluate
    def interrupt(*args,**kwargs):raise InterruptedError('monitor interrupted')
    monkeypatch.setattr(cli.ccr,'evaluate',interrupt);first=tmp_path/'monitor_stop'
    assert cli.main(argv=argv+['--out-dir',str(first)])==0
    saved=torch.load(first/'last.pt',weights_only=False)
    assert saved['updates']==2 and saved['epoch']==0 and saved['batch']==2
    assert not saved['reports'].get('epochs')
    def stop_final(*args,**kwargs):
        if len(args[2])==4:raise InterruptedError('final interrupted')
        return actual(*args,**kwargs)
    monkeypatch.setattr(cli.ccr,'evaluate',stop_final);second=tmp_path/'final_stop'
    assert cli.main(argv=argv+['--out-dir',str(second),'--resume',str(first/'last.pt')])==0
    saved=torch.load(second/'last.pt',weights_only=False)
    assert saved['updates']==4 and saved['epoch']==2 and saved['batch']==0
    assert len(saved['reports']['epochs'])==2 and 'final_dev512' not in saved['reports']
    before=len(calls);monkeypatch.setattr(cli.ccr,'evaluate',actual);last=tmp_path/'final_resume'
    assert cli.main(argv=argv+['--out-dir',str(last),'--resume',str(second/'last.pt')])==0
    assert calls[before:]==[4] and (last/'candidate.pt').is_file()
