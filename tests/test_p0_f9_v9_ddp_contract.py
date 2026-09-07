from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DDP = ROOT / "tools" / "real_motion" / "train_p0_f9_v9_full_m_ddp.py"
CACHE = ROOT / "tools" / "real_motion" / "build_p0_f9_full_native_cache_ddp_ready.py"


def test_ddp_entrypoint_contract_is_explicit():
    text = DDP.read_text(encoding="utf-8")
    assert "DistributedDataParallel as DDP" in text
    assert 'backend="nccl"' in text
    assert '"RANK", "WORLD_SIZE", "LOCAL_RANK"' in text
    assert "--batch-size is GLOBAL batch size" in text
    assert "a.batch_size % world_size" in text
    assert "RankBatchSampler" in text
    assert "global_batch_preserving_rank_partition_global_weighted_loss_v1" in text


def test_ddp_weighted_loss_matches_global_normalized_gradient_contract():
    text = DDP.read_text(encoding="utf-8")
    assert "global_den = _all_reduce_sum(local_den)" in text
    assert "loss_for_backward = local_num * float(world_size) / global_den" in text
    assert "global_num = _all_reduce_sum(local_num)" in text
    assert "normalized_motion_weighted_mse" not in text


def test_ddp_is_deterministic_by_global_step_and_rank():
    text = DDP.read_text(encoding="utf-8")
    assert "global_step:{int(global_step)}:rank:{int(rank)}" in text
    assert 'stream=f"v9_ddp_{stream}"' in text
    assert "_step_noise_and_t" in text
    assert "--uncond-prob 0" in text or 'float(a.uncond_prob) != 0.0' in text


def test_ddp_checkpoints_remain_unwrapped_and_rank0_only():
    text = DDP.read_text(encoding="utf-8")
    assert '"unwrapped_checkpoint_state_dict": True' in text
    assert '"rank0_checkpoint_only": True' in text
    assert "base._payload(" in text
    assert "if _rank0(rank):" in text


def test_ddp_ready_cache_adds_zero_route_metadata_only_after_base_build():
    text = CACHE.read_text(encoding="utf-8")
    assert "base.main()" in text
    assert 'sample["window_valid"].bool().any()' in text
    assert 'meta["zero_route_sample_ids"]' in text
    assert 'meta["ddp_route_safety_annotation"]' in text
