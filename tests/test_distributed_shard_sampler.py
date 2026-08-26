from real_motion.dataset import RealMotionCacheDataset, DistributedShardSampler


class _Backend:
    def __init__(self, sizes=(5, 4, 3)):
        self.entries=[]
        for sid,n in enumerate(sizes):
            for _ in range(n):
                self.entries.append({'shard':f'shard_{sid:03d}.pt'})
    def __len__(self):
        return len(self.entries)


def _fake_dataset(sizes=(5,4,3)):
    ds=object.__new__(RealMotionCacheDataset)
    ds.sharded=True; ds.backend=_Backend(sizes); ds.payload=None; ds.samples=None; ds.metadata={}
    return ds


def test_distributed_shard_sampler_equal_rank_lengths_and_full_cover():
    ds=_fake_dataset((5,4,3))
    samplers=[DistributedShardSampler(ds,3,r,shuffle=True,seed=7) for r in range(3)]
    for s in samplers:s.set_epoch(2)
    rows=[list(s) for s in samplers]
    assert [len(x) for x in rows]==[4,4,4]
    assert sorted(i for row in rows for i in row)==list(range(12))


def test_distributed_shard_sampler_padding_keeps_equal_steps():
    ds=_fake_dataset((4,3,3))
    rows=[list(DistributedShardSampler(ds,3,r,shuffle=False)) for r in range(3)]
    assert [len(x) for x in rows]==[4,4,4]
    assert all(0<=i<len(ds) for row in rows for i in row)


def test_distributed_shard_sampler_epoch_is_deterministic_and_changes_order():
    ds=_fake_dataset((8,8,8))
    a=DistributedShardSampler(ds,2,0,shuffle=True,seed=99);a.set_epoch(3)
    b=DistributedShardSampler(ds,2,0,shuffle=True,seed=99);b.set_epoch(3)
    c=DistributedShardSampler(ds,2,0,shuffle=True,seed=99);c.set_epoch(4)
    assert list(a)==list(b)
    assert list(DistributedShardSampler(ds,2,0,shuffle=False))!=list(c)
