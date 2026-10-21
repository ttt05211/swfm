"""Full two-epoch plan, unchanged architecture, actual small CUDA stop/resume."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.fpn_unet_full import run, model


def test_full_plan_two_epochs_no_drops_and_fixed_same64(tmp_path):
    train = [dict(scene_name=f'T{i//50}', t0_token=f't{i}', features=np.empty((i%5,0))) for i in range(20430)]
    dev = [dict(scene_name=f'V{i%150}', t0_token=f'v{i}', features=[]) for i in range(4369)]
    keys = [[r['scene_name'],r['t0_token']] for r in dev[:64]]
    reference = tmp_path/'reference.json'; reference.write_text(json.dumps(dict(contract=dict(eval_keys=keys))))
    selected, evaluation, groups = run.plan(train, dev, reference, 20261011)
    assert len(selected) == len(set(selected)) == 20430 and len(groups) == 2
    for epoch in groups:
        actual = [(r['scene_name'],r['t0_token']) for group in epoch for r in group]
        assert len(actual) == len(set(actual)) == 20430 and set(actual) == set(selected)
    assert list(run.base.record_keys(evaluation)) == list(map(tuple,keys))
    assert [r['t0_token'] for g in groups[0] for r in g] != [r['t0_token'] for g in groups[1] for r in g]


def test_resume_cadence_and_no_mid_epoch_evaluation():
    groups = [[1,2],[3,4,5]]; c = dict(world_size=2,schedule='whole_2_epoch_cosine_floor0.01')
    state = dict(protocol=run.PROTOCOL,contract=c,epoch=1,batch=1,updates=3,rank_rng=[{},{}])
    assert run.validate_resume(state,c,groups) == (1,1,3)
    assert run.validate_resume(dict(state,epoch=2,batch=0,updates=5),c,groups) == (2,0,5)
    for bad in (dict(state,updates=4),dict(state,protocol='pilot'),dict(state,epoch=2)):
        with pytest.raises(RuntimeError):run.validate_resume(bad,c,groups)
    with pytest.raises(RuntimeError):run.validate_resume(state,dict(c,schedule='restarted_cosine'),groups)
    assert not run.evaluation_due(0,0) and not run.evaluation_due(2000,10)
    assert run.evaluation_due(5145,0) and run.evaluation_due(10292,0)
    assert run.checkpoint_names(2000,0,20) == ['step_0002000.pt']
    assert run.checkpoint_names(5145,1,0) == ['epoch_0001.pt']
    assert run.checkpoint_names(4000,1,0) == ['step_0004000.pt','epoch_0001.pt']


def test_loader_same_architecture_and_pilot_isolation():
    from test_fpn_unet import fixture
    from tools.fpn_unet.model import checkpoint_model as pilot_loader
    body,runner,_=fixture()
    try:
        saved=dict(protocol=run.PROTOCOL,contract=dict(protocol=run.PROTOCOL,model=body.configs()),models={'gt':body.state_dict()})
        restored=model.checkpoint_model(saved)
        assert restored.configs()==body.configs() and not any(p.requires_grad for p in restored.parameters())
        for key,value in body.state_dict().items():torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
        with pytest.raises(RuntimeError):pilot_loader(saved)
        with pytest.raises(RuntimeError):model.checkpoint_model(dict(saved,protocol='joint_FPN_UNet_RANDOM_TRAIN20x1_GT_ego_H6_v1'))
    finally:runner.close()


def test_actual_cuda_two_epoch_pause_resume_Adam_RNG_and_only_two_evals(tmp_path,monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():pytest.skip('BF16 CUDA')
    from dataclasses import asdict
    from test_fpn_unet import fixture
    body,runner,row=fixture()
    try:
        paths={k:tmp_path/k for k in ('config','train_cache','dev_cache','base_checkpoint','train_info','dev_info','eval_reference','population_manifest')}
        for path in paths.values():path.write_text('fixture')
        torch.save(dict(model_config=asdict(body.transport.config)),paths['base_checkpoint'])
        stc=tmp_path/'stc'
        for token,sem in zip(row[0]['history_tokens'],row[1]['history_occ']):
            directory=stc/row[0]['scene_name']/str(token);directory.mkdir(parents=True);np.savez(directory/'labels.npz',semantics=sem)
        evaluation=[dict(row[0],t0_token='dev'+str(i)) for i in range(64)]
        class Provider:
            def __init__(self,*args):
                self.pcfg,self.strong=runner.provider.pcfg,runner.provider.strong
                self.device,self.workers,self.joint=args[3],1,args[5]
            def attach_tracks(self,*a,**k):return {}
            def close_tracks(self):pass
        def caches(provider,*a):
            provider.ccr_history_cache=SimpleNamespace(close=lambda:None)
            provider.ccr_val_history_cache=SimpleNamespace(close=lambda:None)
            return {}
        source=SimpleNamespace(nusc=None)
        monkeypatch.setattr(run,'DirectProvider',Provider)
        monkeypatch.setattr(run,'make_model',lambda cfg:model.make_model(cfg,feature_width=16,z_bins=4,memory_width=4))
        monkeypatch.setattr(run.base,'attach_caches',caches)
        monkeypatch.setattr(run.base,'load_cache',lambda *a:({},[]))
        monkeypatch.setattr(run.base,'load_runtime_config',lambda *a:{})
        monkeypatch.setattr(run.base,'make_prepare_config',lambda *a:runner.provider.pcfg)
        monkeypatch.setattr(run.base,'NuScenesWindowSource',lambda *a,**k:source)
        monkeypatch.setattr(run.base,'CachedColumnSource',lambda *a,**k:source)
        monkeypatch.setattr(run,'plan',lambda *a:([['TRAIN','one']],evaluation,[[[row[0]],[row[0]]],[[row[0]]]]))
        monkeypatch.setattr(run,'batches',lambda provider,source,groups,history,**k:([row] for group in groups))
        calls=[]
        def evaluate(*a,**k):
            calls.append(1)
            return dict(windows=64,keys=[list(k) for k in run.base.record_keys(evaluation)],variants={
                n:dict(mIoU=1.,IoU=2.,MovingMicro=3.) for n in ('occ_gt_transport','occ_gt_semantic_refine','stc_gt_transport','stc_gt_semantic_refine')})
        monkeypatch.setattr(run,'evaluate',evaluate)
        out=tmp_path/'new'
        argv=['--out-dir',str(out),'--stc-root',str(stc),'--dataroot',str(tmp_path),'--ccr-history-cache',str(tmp_path),
            '--ccr-val-history-cache',str(tmp_path),'--device','cuda','--cpu-workers','1','--prefetch-workers','1']
        for key,path in paths.items():argv+=['--'+key.replace('_','-'),str(path)]
        run.main(argv+['--max-updates','2'])
        partial=torch.load(out/'last.pt',weights_only=False)
        assert partial['updates']==2 and partial['epoch']==1 and len(calls)==1
        assert partial['contract']['schedule']=='whole_2_epoch_cosine_floor0.01'
        assert partial['contract']['initialization']=='RANDOM_no_pilot_weights'
        # On resume, verify the optimizer moments and per-rank RNG BEFORE the
        # next real update; max-updates does not change the planned cosine.
        actual_step=run.train_step;observed=[]
        def step(body,optimizer,provider,rows,ctx,**kw):
            state=optimizer.state_dict()
            assert kw['update']==3 and kw['total_updates']==3
            torch.testing.assert_close(torch.get_rng_state(),partial['rank_rng'][0]['torch'],rtol=0,atol=0)
            torch.testing.assert_close(torch.cuda.get_rng_state(provider.device).cpu(),partial['rank_rng'][0]['cuda'],rtol=0,atol=0)
            for key,value in partial['optimizers']['gt']['state'].items():
                for field,tensor in value.items():
                    if isinstance(tensor,torch.Tensor):torch.testing.assert_close(state['state'][key][field].cpu(),tensor.cpu(),rtol=0,atol=0)
            observed.append(1)
            return actual_step(body,optimizer,provider,rows,ctx,**kw)
        monkeypatch.setattr(run,'train_step',step)
        run.main(argv+['--resume',str(out/'last.pt')])
        final=torch.load(out/'last.pt',weights_only=False)
        assert final['updates']==3 and final['epoch']==2 and len(calls)==2 and observed==[1]
        assert (out/'epoch_0001.pt').is_file() and (out/'epoch_0002.pt').is_file()
        model.checkpoint_model(final)
        run.main(argv+['--resume',str(out/'last.pt')]);assert len(calls)==2
        again=torch.load(out/'last.pt',weights_only=False)
        for key,value in final['models']['gt'].items():torch.testing.assert_close(again['models']['gt'][key],value,rtol=0,atol=0)
        with pytest.raises(RuntimeError,match='NEW output'):run.main(argv)
    finally:runner.close()
