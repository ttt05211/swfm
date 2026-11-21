"""CPU downloads only: GT weights and, when needed, ONLY the VAL info file."""
import argparse
from pathlib import Path
import pickle
from .common import sha,write_json,read_json

GT_ID='1cV4c4zyoiitZcJEdvFhr6aiCWwT3mI4-'
INFO_FOLDER='1nXMyP9FueCuJ-1kFLvFbzvdnaUkwL1L6'


def choose_val(files):
    selected=[f for f in files if Path(f.path).suffix=='.pkl' and 'val' in Path(f.path).stem.lower()
        and 'train' not in Path(f.path).stem.lower()]
    if len(selected)!=1:raise ValueError('Need one unambiguous official VAL pickle, no TRAIN download. Files='+str([f.path for f in files]))
    return selected[0]


def inspect_weights(path):
    import torch
    value=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(value,dict) or not value or any(not isinstance(v,torch.Tensor) for v in value.values()):
        raise ValueError('official raw state_dict expected; no partial checkpoint fallback')
    keys=[k[7:] if k.startswith('module.') else k for k in value]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate keys after DDP-prefix removal')
    state=dict(zip(keys,value.values()))
    if tuple(state['temp_embed'].shape)!=(1,10,128) or state['plan_translation_output.2.weight'].shape[0]!=108:
        raise ValueError('released 4-history / 6-future GAST architecture required')
    if any(not bool(torch.isfinite(v).all()) for v in state.values() if v.dtype.is_floating_point):
        raise ValueError('nonfinite released weights')
    return dict(tensors=len(state),tensor_elements=sum(v.numel() for v in state.values()),sha256=sha(path),bytes=Path(path).stat().st_size)


def run(action,work):
    import gdown
    folder=Path(work)/'downloads';folder.mkdir(parents=True,exist_ok=True)
    target=folder/('gast_gt.pth' if action=='weights' else 'official_val.pkl')
    receipt=folder/(action+'_receipt.json')
    if target.is_file() and receipt.is_file():
        saved=read_json(receipt)
        if sha(target)!=saved['sha256']:raise ValueError('previous download changed: '+str(target))
        print('GAST_DOWNLOAD_REUSED',target,flush=True);return
    if target.exists():raise ValueError('unreceipted file exists; keep it and use a NEW GAST_WORK directory: '+str(target))
    if action=='weights':file_id=GT_ID;name='GAST GT ego (official README link)'
    else:
        files=gdown.download_folder(id=INFO_FOLDER,output=str(folder/'listing_only'),quiet=False,skip_download=True,remaining_ok=True)
        if not files:raise RuntimeError('Drive listing unavailable; download ONLY official VAL info to '+str(target)+' then run the explicit adopt-info command')
        selected=choose_val(files);file_id=selected.id;name=selected.path
    partial=target.with_name(target.name+'.part')
    result=gdown.download(id=file_id,output=str(partial),quiet=False,resume=True)
    if not result:raise RuntimeError('Google Drive download failed; rerun the SAME step to resume. No dataset or TRAIN info was requested.')
    details=inspect_weights(partial) if action=='weights' else inspect_info(partial)
    partial.replace(target)
    write_json(receipt,dict(file_id=file_id,source=name,**details))
    print('GAST_DOWNLOAD_READY',target,details,flush=True)


def inspect_info(path):
    # Metadata pickle from the explicitly selected official source, not labels.
    with Path(path).open('rb') as f:value=pickle.load(f)
    if not isinstance(value,dict) or not isinstance(value.get('infos'),(dict,list)):raise ValueError('official info pickle expected')
    return dict(sha256=sha(path),bytes=Path(path).stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('weights','infos','adopt-info','adopt-weights'))
    p.add_argument('--work',type=Path,required=True);p.add_argument('--file',type=Path);a=p.parse_args()
    if a.action.startswith('adopt-'):
        if not a.file or not a.file.is_file():p.error('--file must point to the manually downloaded official file')
        import shutil
        what=a.action.split('-',1)[1];folder=a.work/'downloads';folder.mkdir(parents=True,exist_ok=True)
        target=folder/('official_val.pkl' if what=='info' else 'gast_gt.pth')
        check=inspect_info if what=='info' else inspect_weights;details=check(a.file)
        if target.exists() and sha(target)!=details['sha256']:raise ValueError('existing adopted file differs; NEW GAST_WORK required')
        if a.file.resolve()!=target.resolve() and not target.exists():shutil.copyfile(a.file,target)
        write_json(folder/('infos_receipt.json' if what=='info' else 'weights_receipt.json'),
            dict(source='user supplied official download; identity not independently authenticated',**details))
        print('GAST_ADOPTED',target)
    else:run(a.action,a.work)


if __name__=='__main__':main()
