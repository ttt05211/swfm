"""Full64 fresh confirmation: source / half-LS / history-only decoder, no sum of gains."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.causal_forward_all.common import Epoch2Predictor
from tools.stc_overnight.bundle import Bundle
from .portable import load_weights
from .geometry_screen import velocity_row
from .cache import AT,write,split_name
from .calibrate import confusion,scores

GRID=(-2.,-1.5,-1.,-.5,0.,.5,1.)


def youden(counts):
    out=[]
    for c in counts:
        tp=c[:17,:17].sum();fn=c[:17,17].sum();fp=c[17,:17].sum();tn=c[17,17]
        out.append(tp/max(tp+fn,1)+tn/max(tn+fp,1)-1)
    return float(np.mean(out))


def finalize_fits(out,contract,branches):
    truth={n:{str(b):np.zeros((3,18,18),np.int64) for b in GRID} for n in branches}
    pseudo={n:{str(b):np.zeros((3,18,18),np.int64) for b in GRID} for n in branches}
    for i in range(32):
        q=json.loads((out/'counts'/f'stc_{i:03d}.json').read_text())
        for n in branches:
            for b in GRID:
                truth[n][str(b)]+=np.asarray(q['truth'][n][str(b)],np.int64)
                pseudo[n][str(b)]+=np.asarray(q['pseudo'][n][str(b)],np.int64)
    chosen={};details={}
    for n in branches:
        reports={b:scores(c)['average'] for b,c in truth[n].items()}
        j={b:youden(c) for b,c in pseudo[n].items()}
        target=max(GRID,key=lambda b:reports[str(b)]['mIoU']+.5*reports[str(b)]['IoU'])
        history=max(GRID,key=lambda b:j[str(b)])
        chosen[n]=dict(raw=0.,fixed_free=-1.5,labelled_free=target,history_free=history)
        details[n]=dict(truth_fit32=reports,pseudo_Youden=j)
    fits=dict(contract=contract,chosen=chosen,details=details,
        labelled_free='uses first32 GT future occupancy labels ONLY',
        history_free='uses first32 causal static history-consensus pseudo-labels ONLY; no target future GT in this estimator',
        fixed_free='-1.5 selected in earlier first32 raw-only calibration; same value at all horizons and modalities')
    write(out/'fits.json',fits);print('CONFIRM_FITS',chosen,flush=True);return fits


@torch.no_grad()
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--weights',default='../_received/forward_adaln_epoch2_local.zip')
    ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--adapters',nargs='*',default=[]);ap.add_argument('--device',default='cuda');a=ap.parse_args(argv)
    torch.set_num_threads(1);torch.manual_seed(21);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'counts').mkdir(exist_ok=True)
    body,c,info=load_weights(a.weights,a.device);sha=info['manifest']['source_checkpoint_sha256']
    original={n:v.detach().cpu().clone() for n,v in body.columns.state_dict().items()};states={'source':original};adapter_info={}
    for path in a.adapters:
        pth=Path(path);v=torch.load(pth,map_location='cpu',weights_only=False);tag='history'+str(v['contract']['updates'])
        if v['arm']!='decoder_fine7106_history' or v['contract']['source_sha256']!=sha or v['contract']['target_fit_keys']!=c['eval_keys'][:32]:raise ValueError('only current history-only first32 adapter')
        if v['contract']['labelled_VAL_supervision'][v['arm']] is not False:raise ValueError('no target-supervised adapter in confirmation')
        if tag in states:raise ValueError('duplicate adapter budget')
        for n,value in v['columns'].items():
            if n not in v['trainable'] and not torch.equal(value,original[n]):raise ValueError('unexpected non-adapted weight change')
        states[tag]=v['columns'];adapter_info[tag]=dict(path=str(pth.resolve()),sha256=hashlib.sha256(pth.read_bytes()).hexdigest(),contract=v['contract'])
    branches=[n+'/'+motion for motion in ('pair','halfLS') for n in states]
    contract=dict(protocol='latest_forward_full64_joint_confirmation_v1',source_sha256=sha,reference64_keys=c['eval_keys'],adapters=adapter_info,
        beta_LS=.5,free_grid=list(GRID),GT_future_ego=True,known='dense_valid',full_grid=True,Moving_available=False,
        source_history_decoder_fitting='original TRAIN32 GT future + first32 target INPUT-only pseudo-labels',
        readout_fit='first32 ONLY, frozen before other32; GT future supervised and causal pseudo estimators reported separately',
        development='all64 previously inspected; this is not an independent test')
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:raise ValueError('new confirmation output required')
    write(out/'contract.json',contract);b=Bundle(a.bundle,cache_mib=16);p=Epoch2Predictor(body,info['prepare_config'],a.device,2);tic=time.monotonic()
    fits=json.loads((out/'fits.json').read_text()) if (out/'fits.json').exists() else None
    try:
        for domain in ('stc','gt'):
            for i in range(64):
                name=f'{domain}_{i:03d}';path=out/'counts'/(name+'.json')
                if path.exists():
                    if domain=='stc' and i==31 and fits is None:fits=finalize_fits(out,contract,branches)
                    continue
                if i>=32 and fits is None:raise RuntimeError('freeze first32 estimates before held32')
                r=b.by_key[tuple(c['eval_keys'][i])];rec,raw,times=b.history(r,domain,'dense_valid');row=p.prepare(rec,raw,times=times);pred={}
                for motion in ('pair','halfLS'):
                    body.columns.load_state_dict(original,strict=True)
                    changed=row if motion=='pair' else velocity_row(row,.5)
                    fw=p.forward(changed,poses=np.asarray(r['future_poses']))
                    for tag,state in states.items():
                        body.columns.load_state_dict(state,strict=True)
                        logits=fw['logits'] if tag=='source' else body.columns(fw['memory'],fw['prior'],range(6),fw['shape'])
                        maximum,winner=logits[:,:17].max(1)
                        # Compute all SIX readouts for each globally constant offset.
                        pred[tag+'/'+motion]={str(bias):torch.where(maximum>=logits[:,17]+bias,winner,17)[list(AT)].byte().cpu().numpy() for bias in GRID}
                        if tag!='source':del logits
                    del fw
                # Input-only pseudo-labels are separate from future GT. Never pass
                # either kind of targets to Encoder/Decoder/rendering/Refiner.
                pseudo=None
                if domain=='stc' and i<32:
                    pseudo=np.stack([torch.load(Path(a.cache)/'training_frames'/f'stc_{i:03d}_h{h}.pt',map_location='cpu',weights_only=False,mmap=True)['pseudo'][0].numpy().flatten() for h in AT])
                gt=b.targets(r).reshape(3,-1);truth={};pc={}
                for n,offsets in pred.items():
                    truth[n]={bias:[confusion(values[h],gt[h]).tolist() for h in range(3)] for bias,values in offsets.items()}
                    if pseudo is not None:
                        pc[n]={bias:[confusion(values[h][pseudo[h]<18],pseudo[h][pseudo[h]<18]).tolist() for h in range(3)] for bias,values in offsets.items()}
                write(path,dict(key=[r['scene_name'],r['t0_token']],truth=truth,pseudo=pc))
                if domain=='stc' and i==31:fits=finalize_fits(out,contract,branches)
                del pred,pseudo,gt,row,raw,truth,pc
                if (i+1)%8==0:print('JOINT_CONFIRM',name,'seconds',round(time.monotonic()-tic,1),flush=True)
        totals={}
        for domain in ('stc','gt'):
            for i in range(64):
                item=json.loads((out/'counts'/f'{domain}_{i:03d}.json').read_text())
                for split in (split_name(i),'held32' if i>=32 else 'fit_copy','all64'):
                    if split=='fit_copy':continue
                    key=domain+'/'+split;t=totals.setdefault(key,dict(windows=0,counts={}));t['windows']+=1
                    for n,choices in fits['chosen'].items():
                        for mode,bias in choices.items():
                            route=n+'/'+mode;t['counts'].setdefault(route,np.zeros((3,18,18),np.int64));t['counts'][route]+=np.asarray(item['truth'][n][str(bias)],np.int64)
        reports={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
        write(out/'results.json',dict(status='complete',contract=contract,fits=fits,reports=reports))
        for key in ('stc/held32','stc/all64','gt/all64'):
            print('JOINT_FINAL',key,{n:v['average'] for n,v in reports[key]['reports'].items()},flush=True)
    finally:body.columns.load_state_dict(original,strict=True);p.close();b.close()


if __name__=='__main__':main()
