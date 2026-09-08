import json,os
import numpy as np,torch
from tests.real_motion.test_motion_transport_v1_renderer import fixture,_targets_for_delta,_calibrated_lambda,_grad_row,_one_sided_ce
from real_motion.motion_transport_v1.compositor import render_soft_ordered,compose_hard
from real_motion.motion_transport_v1.losses import occupancy_ce_full,motion_loss_sum,calibration_probe_like

def test_torch26_reachability_diagnostics():
    g,c,d=fixture();cases=[(.35,0,.08),(-.35,0,-.08),(0,.35,.08),(0,-.35,-.08),(.35,.35,0),(-.35,.35,0),(.35,-.35,0),(-.35,-.35,0)];rows=[];failed=[]
    for dx,dy,yaw in cases:
        td=torch.zeros((1,6,3));td[:,:,0]=dx;td[:,:,1]=dy;td[:,:,2]=yaw;gt,targets=_targets_for_delta(g,c,d,td);lam,_,_=_calibrated_lambda(g,c,d,targets);zero=torch.zeros((1,6,3));probe=calibration_probe_like(zero,1e-3);delta=torch.zeros((1,6,3),requires_grad=True);opt=torch.optim.Adam([delta],lr=.01);before=int(np.count_nonzero(compose_hard(c,d,delta,[0],grid=g)!=gt));snap={}
        for step in range(100):
            opt.zero_grad();scene=render_soft_ordered(c,d,delta,[0],grid=g);occ,_=occupancy_ce_full(scene,targets);mn,n,_=motion_loss_sum(delta,[0],d,targets,c.history_ego_to_world[-1]);(occ+lam*mn/n).backward();opt.step()
            if step in (0,9,24,49,99):snap[str(step+1)]={'delta_h0':delta.detach()[0,0].tolist(),'grad':_grad_row(g,c,d,targets,delta.detach())}
        after=int(np.count_nonzero(compose_hard(c,d,delta,[0],grid=g)!=gt));state=opt.state[delta];row={'target':[dx,dy,yaw],'before':before,'after':after,'lambda':lam,'zero':_grad_row(g,c,d,targets,zero),'probe':_grad_row(g,c,d,targets,probe),'one_sided_dx':_one_sided_ce(g,c,d,targets,0),'one_sided_dy':_one_sided_ce(g,c,d,targets,1),'snapshots':snap,'adam_exp_avg_h0':state['exp_avg'][0,0].tolist(),'adam_exp_avg_sq_h0':state['exp_avg_sq'][0,0].tolist()};rows.append(row)
        if not (before>0 and after<before):failed.append(row)
    path=os.environ.get('MT_V1_DIAG_PATH')
    if path:
        with open(path,'w') as f:json.dump({'torch':torch.__version__,'rows':rows,'failed':failed},f,indent=2)
    assert not failed,failed
