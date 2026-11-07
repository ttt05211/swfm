"""Keep GLOBAL groups/order/budget; balance whole windows across TWO GPUs."""
from itertools import combinations

PARTITION = 'balanced_window_count_then_source_count_alternating_two_ranks_v1'


def source_count(row):
    if isinstance(row,(tuple,list)): row=row[0]  # prepared test/batch row.
    return len(row.get('features',row.get('source_class_id',())))


def partition(group,world_size,group_index=0):
    if world_size==1: return [list(group)]
    if world_size!=2 or not 1<=len(group)<=4: raise ValueError('one GLOBAL B1..B4 group and TWO ranks required')
    # Keep 2+2 for B4 and 2+1 for B3. Among those splits, balance source work.
    # Prefer the original round-robin when tied; keep original order on rank.
    first=(len(group)+1)//2;old=tuple(range(0,len(group),2));counts=list(map(source_count,group))
    choices=[]
    for left in combinations(range(len(group)),first):
        right=tuple(i for i in range(len(group)) if i not in left)
        loads=(sum(counts[i] for i in left),sum(counts[i] for i in right))
        choices.append(((abs(loads[0]-loads[1]),left!=old,left),left,right))
    _,left,right=min(choices)
    shards=[[group[i] for i in ids] for ids in (left,right)]
    return shards[::-1] if group_index%2 else shards
