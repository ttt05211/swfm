import pytest,torch
from real_motion.motion_transport_v1.engine import DistContext
from real_motion.motion_transport_v1.server_acceptance import expected_gpu_token,formal_server_environment

def _cfg():return {'training':{'gpus':2,'gpu_type':'L40S_48GB'}}

def test_expected_gpu_token_is_model_not_memory_suffix():assert expected_gpu_token(_cfg())=='L40S'

def test_formal_server_gate_rejects_wrong_world_size_before_profile_lock():
    ctx=DistContext(rank=0,world_size=1,local_rank=0,device=torch.device('cpu'))
    with pytest.raises(RuntimeError,match='WORLD_SIZE=2'):formal_server_environment(ctx,_cfg())

def test_formal_server_gate_rejects_cpu_when_world_size_check_disabled():
    ctx=DistContext(rank=0,world_size=1,local_rank=0,device=torch.device('cpu'))
    with pytest.raises(RuntimeError,match='requires CUDA'):formal_server_environment(ctx,_cfg(),require_world_size=False)
