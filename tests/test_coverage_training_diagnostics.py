import json
import torch
from tools.coverage_projection.training_diagnostics import progress_summary,parameter_summary,optimizer_steps


def test_progress_groups_and_loss_balance_keep_last_resumed_update(tmp_path):
    path=tmp_path/'progress.jsonl'
    rows=[dict(update=i,CE=.1,Lovasz=.5,projection_semantic=.3,projection_depth=.1,loss=1.,grad_norm=2.) for i in range(1,41)]
    path.write_text('\n'.join(map(json.dumps,rows+[rows[-1]]))+'\nBAD')
    result=progress_summary(path)
    assert result['unique_updates']==40 and result['duplicate_log_updates']==1
    assert result['invalid_lines']==[42] and result['loss_sum_max_abs_error']<1e-6
    assert abs(result['last128']['projection_over_main_value']-2/3)<1e-6


def test_head_coverage_weights_and_optimizer_steps_are_read_only():
    from types import SimpleNamespace
    transport=torch.nn.Linear(2,2)
    columns=torch.nn.Module();columns.head=torch.nn.Conv3d(8,18,1);columns.fine=torch.nn.Sequential(torch.nn.Conv3d(21,8,3))
    body=SimpleNamespace(transport=transport,columns=columns)
    optimizer=torch.optim.AdamW([dict(params=transport.parameters()),dict(params=columns.parameters())])
    for p in columns.parameters():p.grad=torch.ones_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    state={'columns.'+n:v for n,v in columns.state_dict().items()}
    before={n:v.clone() for n,v in state.items()}
    result=parameter_summary(state);steps=optimizer_steps(body,optimizer.state_dict())
    assert result['new_coverage_channel_weight_L2']['19']>0
    assert steps['columns_steps']['columns.head.weight']==1
    for n,v in state.items():torch.testing.assert_close(v,before[n],rtol=0,atol=0)


def test_archive_diagnostic_confusion_and_recall_count_attribution():
    import numpy as np
    from tools.coverage_projection.archive_diagnostics import confusion,report
    gt=np.array([11,11,17,17,4],np.uint8)
    pred=np.array([11,17,11,17,4],np.uint8)
    c=confusion(pred,gt,np.ones(5,bool));result=report(c)
    assert result['FP']==1 and result['FN']==1 and result['occupied_GT']==3
    assert abs(result['occupancy_recall']-200/3)<1e-8
    assert result['road_precision']==50 and result['road_recall']==50
