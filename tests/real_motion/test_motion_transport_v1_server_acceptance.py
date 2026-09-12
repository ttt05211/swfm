import pytest,torch,yaml
from real_motion.motion_transport_v1.engine import DistContext
from real_motion.motion_transport_v1.server_acceptance import expected_gpu_token,formal_server_environment

def _cfg(gpus=2):return {'training':{'gpus':gpus,'gpu_type':'L40S_48GB'}}

def test_expected_gpu_token_is_model_not_memory_suffix():assert expected_gpu_token(_cfg())=='L40S'

def test_formal_server_gate_does_not_enforce_configured_world_size_by_default():
    ctx=DistContext(rank=0,world_size=1,local_rank=0,device=torch.device('cpu'))
    with pytest.raises(RuntimeError,match='requires CUDA'):formal_server_environment(ctx,_cfg(gpus=2))

def test_formal_server_gate_can_explicitly_require_configured_world_size():
    ctx=DistContext(rank=0,world_size=1,local_rank=0,device=torch.device('cpu'))
    with pytest.raises(RuntimeError,match='WORLD_SIZE=2'):formal_server_environment(ctx,_cfg(gpus=2),require_world_size=True)

def test_formal_server_gate_rejects_cpu_for_single_gpu_launch():
    ctx=DistContext(rank=0,world_size=1,local_rank=0,device=torch.device('cpu'))
    with pytest.raises(RuntimeError,match='requires CUDA'):formal_server_environment(ctx,_cfg(gpus=1))

def test_formal_server_environment_is_yaml_safe(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda device:'NVIDIA L40S')
    ctx=DistContext(rank=0,world_size=1,local_rank=0,device=torch.device('cuda:0'))
    env=formal_server_environment(ctx,_cfg(gpus=1))
    assert type(env['torch']) is str
    assert env['torch']==str(torch.__version__)
    assert env['cuda_build'] is None or type(env['cuda_build']) is str
    dumped=yaml.safe_dump({'runtime':{'profile_environment':env}},sort_keys=False)
    assert 'profile_environment:' in dumped
