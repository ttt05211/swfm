"""AdaLN algebra, fair fresh initialization, real geometry/backward and reused recovery."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.fpn_unet import model as additive, adapter, run as original
from tools.fpn_adaln_zero import model, run
from tools.shared_history_experiments.full_gt.engine import semantic_loss


def fixture(device='cpu'):
    from test_fpn_unet import fixture as old
    control, runner, row = old(device)
    cfg = replace(control.transport.config, decoder_blocks=2)
    body = model.make_model(cfg,feature_width=16,z_bins=4,memory_width=4).to(device)
    return body,runner,row


def test_block_identity_and_exact_cross_first_formula():
    torch.manual_seed(21); block = model.AdaLNZeroFutureBlock(16,4)
    q = torch.randn(3,6,16); history = torch.randn(3,4,16)
    valid = torch.tensor([[0,0,1,1],[1,1,1,1],[0,1,0,1]],dtype=torch.bool)
    condition = torch.randn(1,6,16)
    torch.testing.assert_close(block(q,history,valid,condition),q,rtol=0,atol=0)
    assert all(not getattr(block,n).elementwise_affine for n in ('cross_q_norm','self_norm','ffn_norm'))
    with torch.no_grad():
        block.adaLN_modulation[-1].weight.normal_(std=.03)
        block.adaLN_modulation[-1].bias.normal_(std=.02)
    sc,ac,gc,ss,ass,gs,sf,af,gf = block.adaLN_modulation(condition).chunk(9,-1)
    h = block.cross_ctx_norm(history)
    z = model.modulate(block.cross_q_norm(q),sc,ac)
    q1 = q+gc*block.cross_attn(z,h,h,key_padding_mask=~valid,need_weights=False)[0]
    z = model.modulate(block.self_norm(q1),ss,ass)
    q2 = q1+gs*block.self_attn(z,z,z,need_weights=False)[0]
    expected = q2+gf*block.ffn(model.modulate(block.ffn_norm(q2),sf,af))
    torch.testing.assert_close(block(q,history,valid,condition),expected,rtol=0,atol=0)
    corrupted = history.clone(); corrupted[~valid] = 50
    torch.testing.assert_close(block(q,corrupted,valid,condition),expected,rtol=0,atol=0)
    with pytest.raises(ValueError): block(q,history,valid,condition.expand(3,-1,-1))


def test_common_fresh_weights_RNG_and_parameter_count():
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    cfg = LocalSTWMV17Config(history_frames=4,decoder_blocks=2)
    torch.manual_seed(20261011); control = additive.make_model(cfg); rng = torch.get_rng_state().clone()
    torch.manual_seed(20261011); body = model.make_model(cfg)
    assert torch.equal(torch.get_rng_state(),rng)
    before = control.state_dict()
    for name,value in body.state_dict().items():
        if name in before: torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert sum(p.numel() for p in control.parameters()) == 2520442
    assert sum(p.numel() for p in body.parameters()) == 2816122
    assert not hasattr(body.transport,'ego_future_proj') and not hasattr(body.columns,'source_query')
    assert body.transport.ego_condition_encoder[-1].weight.abs().sum() > 0
    for block in body.transport.decoder:
        assert not torch.count_nonzero(block.adaLN_modulation[-1].weight)
        assert not torch.count_nonzero(block.adaLN_modulation[-1].bias)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_prediction_pure_memory_ego_condition_and_gradient(device):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    body,runner,row = fixture(device)
    try:
        poison = row[0],dict(row[1],future_gt_occ=object()),row[2],row[3]
        fw = adapter.forward_window(body,runner.provider,poison,fresh=True)
        assert fw['logits'].shape == (6,18,np.prod(fw['shape']))
        torch.testing.assert_close(fw['logits'].argmax(1),fw['prior'].argmax(1),rtol=0,atol=0)
        # At zero gates, output queries are independent of historical context/ego,
        # but the exact historical encoder/yaw and renderer still run normally.
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device == 'cuda'):
            query_expected = body.transport.initial_queries(torch.as_tensor(row[0]['kta_displacement_xy_m'],device=device),
                fw['ego_features'],fw['field'].dtype)
        torch.testing.assert_close(fw['output']['future_transport_queries'],query_expected,rtol=0,atol=0)
        semantic_loss(fw['logits'],row[1]['future_gt_occ'])[0].mean().backward()
        assert body.transport.residual_head.weight.grad.abs().sum() > 0
        # Open zero gates/heads for a structural gradient check, not a quality run.
        body.zero_grad(set_to_none=True)
        with torch.no_grad():
            body.columns.head.weight.normal_(std=.02)
            body.transport.residual_head.weight.normal_(std=.02)
            for block in body.transport.decoder:
                block.adaLN_modulation[-1].weight.normal_(std=.01)
                block.adaLN_modulation[-1].bias.normal_(std=.01)
        fw = adapter.forward_window(body,runner.provider,row,fresh=True)
        semantic_loss(fw['logits'],row[1]['future_gt_occ'])[0].mean().backward()
        for p in (body.transport.ego_condition_encoder[-1].weight,
                  body.transport.decoder[0].adaLN_modulation[-1].weight,
                  body.transport.history_encoder.down[1][0].weight,body.columns.lift.weight):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        # Perturb ONLY the embedding output; keep correct rendering poses fixed.
        before_history = fw['output']['history_source_context'].detach().clone()
        before_yaw = fw['output']['history_yaw_to_t0_rad'].detach().clone()
        hook = body.transport.ego_condition_encoder.register_forward_hook(lambda m,a,o:o+2)
        try: changed = adapter.forward_window(body,runner.provider,row,fresh=True)
        finally: hook.remove()
        torch.testing.assert_close(changed['poses'],fw['poses'],rtol=0,atol=0)
        torch.testing.assert_close(changed['output']['history_source_context'],before_history,rtol=0,atol=0)
        torch.testing.assert_close(changed['output']['history_yaw_to_t0_rad'],before_yaw,rtol=0,atol=0)
        assert not torch.equal(changed['output']['future_transport_queries'],fw['output']['future_transport_queries'])
        saved = dict(protocol=run.PROTOCOL,contract=dict(model=body.configs()),models={'gt':body.state_dict()})
        restored = model.checkpoint_model(saved,device)
        for k,v in body.state_dict().items(): torch.testing.assert_close(restored.state_dict()[k],v,rtol=0,atol=0)
        with pytest.raises(RuntimeError): model.checkpoint_model(dict(saved,protocol=original.PROTOCOL))
    finally: runner.close()


def test_zero_gates_open_condition_training_without_manual_activation():
    torch.manual_seed(21)
    encoder = torch.nn.Sequential(torch.nn.Linear(4,16),torch.nn.SiLU(),torch.nn.Linear(16,16))
    block = model.AdaLNZeroFutureBlock(16,4)
    optimizer = torch.optim.AdamW([*block.parameters(),*encoder.parameters()],lr=.01)
    q = torch.randn(2,6,16); context = torch.randn(2,4,16); valid = torch.ones(2,4,dtype=torch.bool)
    pose = torch.randn(1,6,4); target = torch.randn_like(q)
    before = encoder[-1].weight.detach().clone()
    for _ in range(4):
        optimizer.zero_grad(); loss = (block(q,context,valid,encoder(pose))-target).square().mean()
        loss.backward(); optimizer.step()
    assert not torch.equal(before,encoder[-1].weight)
    assert block.adaLN_modulation[-1].weight.abs().sum() > 0


def test_actual_full_network_zero_initialization_opens_ego_learning():
    body,runner,row = fixture()
    try:
        optimizer = torch.optim.AdamW(body.parameters(),lr=.003,weight_decay=0)
        before = body.transport.ego_condition_encoder[-1].weight.detach().clone()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            fw = adapter.forward_window(body,runner.provider,row,fresh=True)
            loss = semantic_loss(fw['logits'],row[1]['future_gt_occ'])[0].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(body.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
        assert not torch.equal(before,body.transport.ego_condition_encoder[-1].weight)
        assert torch.isfinite(loss)
    finally: runner.close()


def fake_control():
    return dict(status='complete',epoch=1,updates=1031,contract=dict(protocol=original.PROTOCOL,
        model=dict(variant=additive.VARIANT),epochs=1,total_updates=1031,
        known='TRAIN_OCC_STC_dense_valid',source_policy='shared_pinned',registration='none',
        objective=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.),
        train_keys=[['TRAIN',str(i)] for i in range(4086)],eval_keys=[['VAL',str(i)] for i in range(64)]),
        evaluations={'1':dict(windows=64,keys=[['VAL',str(i)] for i in range(64)],
            variants={n:dict(mIoU=1.,IoU=2.,MovingMicro=3.) for n in run.NAMES})})


def test_control_rejection_and_original_runner_untouched(tmp_path):
    assert '--control-dir' in run.parser().format_help()
    control = fake_control(); path = tmp_path/'results.json'; path.write_text(json.dumps(control))
    assert run.reference(tmp_path)[1]['updates'] == 1031
    control['contract']['model']['variant'] = 'old_model'; path.write_text(json.dumps(control))
    with pytest.raises(RuntimeError): run.reference(tmp_path)
    assert original.PROTOCOL != run.runner.PROTOCOL
    assert original.make_model is additive.make_model
    assert original.launch is not run.runner.launch
    assert run.runner.forward_window is original.forward_window and run.runner.evaluate is original.evaluate
    a = SimpleNamespace(seed=1,motion_lr=.001,repair_lr=.001,override=[])
    with pytest.raises(RuntimeError,match='seed'):
        run.check_settings(a,SimpleNamespace(world_size=2),dict(contract={}))


def test_reused_cuda_training_resume_and_final_evaluation(tmp_path,monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA')
    import test_fpn_unet as baseline_test
    # Exercise the SAME actual two-update trainer/recovery fixture with the NEW
    # model and protocol. Scientific control validation is independently tested.
    monkeypatch.setattr(baseline_test,'run',run.runner)
    monkeypatch.setattr(baseline_test,'model',model)
    monkeypatch.setattr(run.runner,'parser',run._parser)
    monkeypatch.setattr(run.runner,'launch',run._launch)
    baseline_test.test_actual_cuda_runner_resume_and_final_eval_once(tmp_path,monkeypatch)
