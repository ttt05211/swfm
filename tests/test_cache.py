import torch
from real_motion.cache import save_sharded_cache,ShardedCacheDataset

def sample(i):
    return {'sample_id':str(i),'moving_history_latent':torch.zeros(2,16,4,4),
            'future_dynamic_target_latent':torch.zeros(3,16,4,4),'static_future_latent':torch.zeros(3,16,4,4),
            'kta_future_latent':torch.zeros(3,16,4,4),'generation_support':torch.zeros(3,4,4,dtype=torch.bool)}

def test_sharded_cache(tmp_path):
    save_sharded_cache(tmp_path,(sample(i) for i in range(5)),shard_size=2)
    ds=ShardedCacheDataset(tmp_path); assert len(ds)==5; assert ds[4]['sample_id']=='4'


def test_shard_shuffle_sampler_keeps_each_shard_contiguous(tmp_path):
    from real_motion.dataset import RealMotionCacheDataset, ShardShuffleSampler
    from real_motion.cache import save_sharded_cache
    import torch
    def sample(i):
        z=torch.zeros(1,1,2,2)
        return {"sample_id":str(i),"moving_history_latent":z,"future_dynamic_target_latent":z,
                "static_future_latent":z,"kta_future_latent":z,
                "generation_support":torch.zeros(1,2,2,dtype=torch.bool)}
    root=tmp_path/'cache'; save_sharded_cache(root,(sample(i) for i in range(6)),shard_size=2)
    ds=RealMotionCacheDataset(root); order=list(ShardShuffleSampler(ds,seed=7))
    shard_seq=[ds.backend.entries[i]['shard'] for i in order]
    # Each shard appears in a single contiguous run even though shard order and
    # local sample order are shuffled.
    runs=[]
    for s in shard_seq:
        if not runs or runs[-1]!=s: runs.append(s)
    assert len(runs)==3 and len(set(runs))==3


def test_legacy_target_key_is_upgraded_for_dataset(tmp_path):
    from real_motion.cache import load_cache
    import torch
    z=torch.zeros(1,1,2,2)
    old={"moving_history_latent":z,"future_moving_latent":z,"static_future_latent":z,
         "kta_future_latent":z,"generation_support":torch.zeros(1,2,2,dtype=torch.bool)}
    path=tmp_path/'legacy.pt'
    torch.save({"version":"real_motion_v2","metadata":{},"samples":[old]},path)
    payload=load_cache(path)
    assert "future_dynamic_target_latent" in payload["samples"][0]
