import torch
from real_motion.motion_transport_v1.ema import WarmupEMA
def test_ema_update_count_and_resume():
    m=torch.nn.Linear(2,1);e=WarmupEMA(m,100);e.start();assert e.num_updates==0
    with torch.no_grad():m.weight.add_(1)
    b=e.update();assert e.num_updates==1 and 0<b<1;state=e.state_dict();e2=WarmupEMA(m);e2.load_state_dict(state);assert e2.num_updates==1 and e2.started
