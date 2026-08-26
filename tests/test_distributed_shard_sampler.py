from real_motion.dataset import RealMotionCacheDataset, DistributedShardSampler


class _Backend:
    def __init__(self, sizes=(5,4,3)):
        self.entries=[];self.shard_ranges={};start=0
        for sid,n in enumerate(sizes):
            name=f'shard_{sid:03d}.pt';idxs=[]
            for _ in range(n):
                idxs.append(len(self.entries));self.entries.append({'shard':name})
            self.shard_ranges[name]=set(idxs)
    def __len__(self):return len(self.entries)


def _fake_dataset(sizes=(5,4,3)):
    ds=object.__new__(RealMotionCacheDataset);ds.sharded=True;ds.backend=_Backend(sizes);ds.payload=None;ds.samples=None;ds.metadata={};return ds


def test_distributed_shard_sampler_whole_shard_ownership_and_equal_steps():
    ds=_fake_dataset((5,4,3));samplers=[DistributedShardSampler(ds,3,r,shuffle=True,seed=7) for r in range(3)]
    for s in samplers:s.set_epoch(2)
    rows=[list(s) for s in samplers]
    assert len(set(map(len,rows)))==1
    assert set().union(*(set(r) for r in rows))==set(range(12))
    # Every original sample has a single owner rank; padding may repeat only locally.
    for idx in range(12):
        assert sum(idx in set(row) for row in rows)==1
    # Whole physical shards are not split across ranks.
    for members in ds.backend.shard_ranges.values():
        owners={r for r,row in enumerate(rows) if members & set(row)}
        assert len(owners)==1


def test_distributed_shard_sampler_padding_keeps_equal_steps():
    ds=_fake_dataset((4,3,3));rows=[list(DistributedShardSampler(ds,3,r,shuffle=False)) for r in range(3)]
    assert len(set(map(len,rows)))==1
    assert set().union(*(set(r) for r in rows))==set(range(10))
    assert all(0<=i<len(ds) for row in rows for i in row)


def test_distributed_shard_sampler_epoch_is_deterministic():
    ds=_fake_dataset((8,8,8,8))
    a=DistributedShardSampler(ds,2,0,shuffle=True,seed=99);a.set_epoch(3)
    b=DistributedShardSampler(ds,2,0,shuffle=True,seed=99);b.set_epoch(3)
    c=DistributedShardSampler(ds,2,0,shuffle=True,seed=99);c.set_epoch(4)
    assert list(a)==list(b)
    assert list(a)!=list(c)
