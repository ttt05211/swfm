"""Optional CUDA + real replay integration, not a performance claim."""
import os
from contextlib import closing
import numpy as np
import pytest
import torch


@pytest.mark.integration
def test_actual_old_and_ccr_joint_backward_interfaces_and_geometry(tmp_path):
    path=os.environ.get('SWFM_CCR_REPLAY')
    if not path:pytest.skip('optional user-provided real replay')
    if not torch.cuda.is_available():pytest.skip('actual CUDA required')
    from real_motion.local_replay_bundle import ReplayBundle,file_digest
    from real_motion.runtime_config import make_prepare_config
    from real_motion.canonical_causal_repair import (CanonicalRepairHead,build_canonical_evidence,
        map_canonical_evidence,map_canonical_reference,materialize_canonical_features)
    from tools.real_motion.pilot_p0_f9_canonical_causal_repair import load_exported_config
    from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
    from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
    from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256
    from tools.real_motion.joint_column_full_common import build_fixed_geometry
    from tools.real_motion.pilot_p0_f9_canonical_causal_repair import joint_probe
    torch.set_num_threads(1);device=torch.device('cuda')
    with closing(ReplayBundle(path)) as bundle:
        for member in ('runtime.yaml','checkpoints/epoch_0019.pt','checkpoints/clean_e14.pt'):
            bundle.copy_member(member,tmp_path/member.split('/')[-1])
        cfg=load_exported_config(tmp_path/'runtime.yaml',bundle.manifest['config_fingerprint'])
        _,teacher=load_joint(tmp_path/'epoch_0019.pt',device,reference_sha=CLEAN_SHA256,
                            config_sha=bundle.manifest['config_fingerprint'],allow_diagnostic=True)
        teacher.eval().requires_grad_(False)
        provider=PilotProvider(tmp_path/'clean_e14.pt',CLEAN_SHA256,make_prepare_config(cfg),device,2,teacher,None)
        idx=next(i for i,m in enumerate(bundle.manifest['windows']) if m['split']=='train' and m['stratum']=='representative')
        rec,raw,labels=bundle.window(idx,labels=True);raw['future_gt_occ']=None
        raw['_column_causal_preparation']=build_fixed_geometry(raw,rec,provider.pcfg,provider.strong,2,teacher.columns.config)
        with torch.no_grad():
            output=teacher.motion(rec,device)
            prep=provider.prepare_columns(None,rec,include_gt=False,raw_window=raw,outputs=output)
        e=build_canonical_evidence(prep,provider.pcfg.grid)
        deferred=build_canonical_evidence(prep,provider.pcfg.grid,materialize_features=False)
        ids=np.random.default_rng(9).choice(len(e),500,replace=False)
        small=materialize_canonical_features(deferred,prep,provider.pcfg.grid,ids)
        np.testing.assert_array_equal(small.features,e.features[ids]);np.testing.assert_array_equal(small.labels,e.labels[ids])
        a=map_canonical_evidence(e,prep,provider.pcfg.grid);b=map_canonical_reference(e,prep,provider.pcfg.grid)
        for key in ('flat','base','fallback','legal','context'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        head=CanonicalRepairHead(teacher.columns.source_dim).to(device)
        before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
        head_before={k:v.detach().clone() for k,v in head.state_dict().items()}
        # Repetition is only an API/gradient integration fixture, NEVER a sample
        # population for throughput/generalization reporting. Pilot uses 8 keys.
        case=dict(record=rec,causal=raw,gt=labels['future_gt_occ'].numpy())
        report=joint_probe(provider,teacher,head,[case]*8,device)
        for mode in ('old_joint','CCR'):
            assert report[mode]['actual_backward'] and not report[mode]['transport_frozen']
            assert report[mode]['windows']==8
        assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in before.items())
        assert all(torch.equal(v,head.state_dict()[k]) for k,v in head_before.items())
        assert file_digest(tmp_path/'epoch_0019.pt')==bundle.manifest['teacher_sha256']


@pytest.mark.integration
def test_real_cuda_causal_mc_spatial_backward_and_disk_input_exactness(tmp_path):
    path=os.environ.get('SWFM_CCR_REPLAY')
    if not path:pytest.skip('optional user-provided real replay')
    if not torch.cuda.is_available():pytest.skip('actual CUDA required')
    from real_motion.local_replay_bundle import ReplayBundle
    from real_motion.runtime_config import make_prepare_config
    from real_motion.canonical_repair_context import (FixedCanonicalCache,SpatialCanonicalRepairHead,
        attach_neighbors,sample_causal_points,map_sampled_canonical)
    from real_motion.canonical_causal_repair import map_canonical_evidence,repair_targets
    from tools.real_motion.pilot_p0_f9_canonical_causal_repair import load_exported_config
    from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
    from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
    from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256
    from tools.real_motion.joint_column_full_common import build_fixed_geometry
    from tools.real_motion.pilot_p0_f9_canonical_causal_repair import loss_for_causal
    torch.set_num_threads(1);device=torch.device('cuda')
    with closing(ReplayBundle(path)) as bundle:
        for member in ('runtime.yaml','checkpoints/epoch_0019.pt','checkpoints/clean_e14.pt'):
            bundle.copy_member(member,tmp_path/member.split('/')[-1])
        cfg=load_exported_config(tmp_path/'runtime.yaml',bundle.manifest['config_fingerprint'])
        _,teacher=load_joint(tmp_path/'epoch_0019.pt',device,reference_sha=CLEAN_SHA256,
                            config_sha=bundle.manifest['config_fingerprint'],allow_diagnostic=True)
        teacher.eval().requires_grad_(False)
        provider=PilotProvider(tmp_path/'clean_e14.pt',CLEAN_SHA256,make_prepare_config(cfg),device,2,teacher,None)
        index=next(i for i,m in enumerate(bundle.manifest['windows']) if m['split']=='train' and m['stratum']=='representative')
        rec,raw,labels=bundle.window(index,labels=True);raw['future_gt_occ']=None
        raw['_column_causal_preparation']=build_fixed_geometry(raw,rec,provider.pcfg,provider.strong,2,teacher.columns.config)
        # Actual live transport gradients, not a frozen/tensor-only speed proxy.
        teacher.transport.requires_grad_(True)
        output=teacher.motion(rec,device)
        prep=provider.prepare_columns(None,rec,include_gt=False,raw_window=raw,outputs=output)
        cache=FixedCanonicalCache(0,disk_root=tmp_path/'history',max_disk_mib=32)
        e,g=cache.get(prep,provider.pcfg.grid);attach_neighbors(e,g)
        conflicts=cache.static_conflicts(e,prep,provider.pcfg.grid)
        full=map_canonical_evidence(e,prep,provider.pcfg.grid)
        ids,_=sample_causal_points(e,np.random.default_rng(19),per_role=512)
        small,plan=map_sampled_canonical(e,ids,prep,provider.pcfg.grid,conflicts)
        for field in ('flat','base','fallback','legal','context'):
            np.testing.assert_array_equal(getattr(plan,field),getattr(full,field)[ids])
        gt=labels['future_gt_occ'].numpy();a,av=repair_targets(e,full,gt);b,bv=repair_targets(small,plan,gt)
        np.testing.assert_array_equal(a[ids],b);np.testing.assert_array_equal(av[ids],bv)
        second,graph=cache.get(prep,provider.pcfg.grid)
        np.testing.assert_array_equal(second.features,e.features);np.testing.assert_array_equal(graph,g)
        head=SpatialCanonicalRepairHead(teacher.columns.source_dim,normalized=False,zero_residual=True).to(device)
        loss,n=loss_for_causal(head,e,output,prep,provider.pcfg.grid,gt,np.random.default_rng(5),device,conflicts)
        loss.backward()
        assert n>0 and torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in teacher.transport.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.spatial.parameters())
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        cache.close()
        # Run the actual SERVER point-screen entry on this real window too:
        # complete epoch19 freeze, fresh MC labels, real encoder/backward/AdamW.
        from real_motion.canonical_causal_repair import CanonicalRepairHead
        from tools.real_motion.ccr_screen_common import train_step
        teacher.zero_grad(set_to_none=True); teacher.eval().requires_grad_(False)
        before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
        point=CanonicalRepairHead(teacher.columns.source_dim).to(device)
        prior=point.encoder[0].weight.detach().clone()
        optimizer=torch.optim.AdamW(point.parameters(),lr=.001)
        provider.ccr_cache=FixedCanonicalCache(32,neighbors=False)
        provider.ccr_samples_per_role=512
        full_raw={**raw,'future_gt_occ':gt}
        for _ in range(2):
            stats=train_step(provider,[(rec,full_raw)],teacher,point,optimizer,np.random.default_rng(21))
            assert stats['transport_frozen'] and stats['sampled_points']>0 and np.isfinite(stats['loss'])
        assert not torch.equal(prior,point.encoder[0].weight)
        assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in before.items())
        assert all(p.grad is None for p in teacher.parameters())
        provider.ccr_cache.close()
