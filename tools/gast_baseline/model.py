"""Load the unmodified pinned official model, and build its GT-ego inputs."""
import importlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml
from .common import COMMIT,ROOT,read_json,sha,write_json


def check_code(folder):
    folder=Path(folder).resolve();head=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT:raise ValueError('pinned official commit required: '+COMMIT)
    import hashlib
    pin=read_json(Path(__file__).with_name('upstream.json'))
    raw={}
    for name,digest in pin.items():
        path=folder/name;content=path.read_text(encoding='utf-8').replace('\r\n','\n').rstrip('\n')+'\n'
        if hashlib.sha256(content.encode()).hexdigest()!=digest:raise ValueError('official code changed: '+name)
        raw[name]=sha(path)
    return raw


def load_model(folder,weight,work,device):
    folder=Path(folder).resolve();implementation=check_code(folder)
    if 'models' in sys.modules:
        paths=getattr(sys.modules['models'],'__path__',[])
        if not any(Path(p).resolve()==folder/'models' for p in paths):raise RuntimeError('unrelated models package already imported')
    sys.path.insert(0,str(folder))
    cls=importlib.import_module('models.gast').GAST
    if Path(sys.modules['models.gast'].__file__).resolve()!=folder/'models/gast.py':raise RuntimeError('wrong GAST model import')
    config=yaml.safe_load((folder/'config/gast_nuscenes.yaml').read_text())
    if config['model_params']['num_his_frames']!=4 or config['model_params']['num_fut_frames']!=6:raise ValueError('fixed 4->6 architecture')
    config['model_params']['use_gt']=True;config['common_params']['gpu']='0'
    config['infer_params'].update(model_path=str(Path(weight).resolve()),save_dir=str(Path(work).resolve()))
    model=cls(config)
    state=torch.load(weight,map_location='cpu',weights_only=False)
    if not isinstance(state,dict) or any(not isinstance(v,torch.Tensor) for v in state.values()):raise ValueError('official raw state_dict required')
    stripped=[k[7:] if k.startswith('module.') else k for k in state]
    if len(stripped)!=len(set(stripped)):raise ValueError('duplicate keys after DDP prefix removal')
    state=dict(zip(stripped,state.values()))
    model.load_state_dict(state,strict=True)
    model.to(device).eval().requires_grad_(False)
    Path(work).mkdir(parents=True,exist_ok=True)
    generated=Path(work)/'gast_infer.yaml';text=yaml.safe_dump(config,sort_keys=False)
    if generated.exists() and generated.read_text()!=text:raise ValueError('generated config differs; NEW GAST_WORK required')
    generated.write_text(text)
    return model,dict(commit=COMMIT,weight_sha256=sha(weight),config_sha256=sha(generated),implementation=implementation,
        parameters=sum(p.numel() for p in model.parameters()),precision='official FP32; no AMP or added calibration',
        execution=dict(torch=str(torch.__version__),cuda=torch.version.cuda))


def relative_rotation(q):
    current,previous=q[1:],q[:-1];conjugate=previous.clone();conjugate[:,1:]*=-1
    w1,x1,y1,z1=conjugate.unbind(-1);w2,x2,y2,z2=current.unbind(-1)
    return torch.stack((w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2),-1)


def input_dict(history,frames,device):
    if len(frames)!=10 or tuple(history.shape[:1])!=(4,):raise ValueError('4 histories + metadata for 6 futures required')
    make=lambda values:torch.tensor(np.asarray(values),dtype=torch.float32,device=device)
    poses=make([f['pose'] for f in frames]);translation=make([f['translation'] for f in frames]);rotation=make([f['rotation'] for f in frames])
    delta=(translation[1:]-translation[:-1])[:,:2];relative=relative_rotation(rotation)
    hist_delta=torch.cat((delta[:3],delta.new_zeros((1,2))),0)
    hist_rotation=torch.cat((relative[:3],relative.new_zeros((1,4))),0)
    lcf=make([f['lcf'] for f in frames[:4]]);commands=make([f['command'] for f in frames[4:]])
    if lcf.shape!=(4,3) or commands.shape!=(6,3) or not bool((commands.sum(-1)==1).all()):raise ValueError('genuine ego fields required')
    result=dict(his_occ=torch.as_tensor(np.asarray(history).astype(np.int64),device=device)[None],
        his_ego_pose=poses[:4][None],fut_ego_pose=poses[4:][None],
        his_e2g_translation=translation[:4][None],his_e2g_rotation=rotation[:4][None],
        his_ego_lcf_feat=lcf[None],fut_ego_cmd=commands[None],
        his_e2g_delta_translation=hist_delta[None],his_e2g_relative_rotation=hist_rotation[None],
        fut_e2g_delta_translation=delta[3:][None],fut_e2g_relative_rotation=relative[3:][None])
    # NO fut_occ, future labels or visibility mask enters the official model.
    return result


def forecast(model,history,frames,device):
    data=input_dict(history,frames,device)
    with torch.no_grad():
        logits,_,_=model(data)
        expected=(1,10,18,*history.shape[1:])
        if tuple(logits.shape)!=expected or not bool(torch.isfinite(logits).all()):raise ValueError('invalid complete official GAST output')
        result=logits[:,4:10].argmax(2)[0].byte().cpu().numpy()
    return result


def rollout(model,history,frames,device):
    if len(frames) not in (10,16):raise ValueError('one 3s block or two 3s blocks only')
    first=forecast(model,history,frames[:10],device)
    if len(frames)==10:return first
    # Global frames 6..9 are predicted at 1.5/2/2.5/3s. No semantic loader here.
    second=forecast(model,first[-4:].copy(),frames[6:16],device)
    return np.concatenate((first,second))


def cuda_check(device):
    import mmcv
    import torch_scatter
    from mmcv.ops import MultiScaleDeformableAttention
    if torch.device(device).type!='cuda' or not torch.cuda.is_available():raise RuntimeError('GPU phase only: allocate ONE GPU, then run check; CPU setup steps do not need this')
    x=torch.tensor([1.,3.],device=device);idx=torch.tensor([0,0],device=device)
    if not torch.allclose(torch_scatter.scatter_mean(x,idx),torch.tensor([2.],device=device)):raise RuntimeError('CUDA scatter_mean failed')
    net=MultiScaleDeformableAttention(embed_dims=128,num_heads=4,num_levels=1,num_points=8,batch_first=True).to(device).eval()
    q=torch.zeros(1,4,128,device=device)
    with torch.no_grad():
        y=net(query=q,value=q,reference_points=torch.full((1,4,1,2),.5,device=device),
            spatial_shapes=torch.tensor([[2,2]],device=device),level_start_index=torch.tensor([0],device=device))
    torch.cuda.synchronize(device)
    if y.shape!=q.shape or not bool(torch.isfinite(y).all()):raise RuntimeError('CUDA deformable attention failed')
    print('GAST_CUDA_OK',torch.cuda.get_device_name(device),torch.__version__,torch.version.cuda,mmcv.__version__,flush=True)
